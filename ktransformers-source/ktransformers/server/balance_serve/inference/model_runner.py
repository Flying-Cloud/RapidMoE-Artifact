"""CUDA Graph model runner for the RapidMoE balance-serve endpoint."""

import torch
import json
from ktransformers.server.balance_serve.inference.forward_batch import ForwardBatchInput, ForwardBatchOutput

from ktransformers.server.config.config import Config
from ktransformers.models.custom_modeling_deepseek_v3 import KDeepseekV3ForCausalLM
from ktransformers.models.custom_modeling_qwen2_moe import KQwen2MoeForCausalLM
from ktransformers.models.custom_modeling_qwen3_moe import KQwen3MoeForCausalLM
from ktransformers.server.balance_serve.inference.query_manager import QueryManager
from ktransformers.server.balance_serve.settings import sched_ext
from ktransformers.operators.experts import KExpertsHybrid
from ktransformers.server.balance_serve.inference.rapidmoe_deployment import configure_rapidmoe_layers

def pad_num_tokens(num_tokens):
    return (num_tokens + 63) // 64 * 64

def deduplicate_and_sort(lst):
    return sorted(set(lst))
class ModelRunner:
    """A CudaGraphRunner runs the forward pass of a model with CUDA graph and torch.compile."""

    model: KDeepseekV3ForCausalLM  | KQwen2MoeForCausalLM | KQwen3MoeForCausalLM 
    input: ForwardBatchInput | list[ForwardBatchInput]
    output: ForwardBatchOutput
    
    def __init__(self, model = None, device = None, use_cuda_graph = False, max_decode_batch_size = 1, max_chunk_size = 4096, num_mini_batches: int = 1, page_size = 256, block_num = 8):
        
        self.stream = torch.cuda.Stream(device=device)
        self.model = model
        self.device = device
        self.input = None
        self.features_buf = None
        self.output = None
        self.graph_memory_pool = None
        self.cuda_graphs = deduplicate_and_sort([1, 2, 3, Config().max_batch_size, 512]) # note 16/32/64 related to GPU Memory Size
        self.sub_graphs = [0,1,2,3,4,5,6]
        self.use_cuda_graph = use_cuda_graph
        self.model_time = 0
        self.page_size = page_size
        self.block_num = block_num
        # GPU timing for model execution
        self.start_model_event = torch.cuda.Event(enable_timing=True)
        self.end_model_event = torch.cuda.Event(enable_timing=True)

        self.graphs = [torch.cuda.CUDAGraph() for _ in range(len(self.cuda_graphs))]
        # Retained for compatibility with the subgraph replay interface.
        self.subgraphs = [[torch.cuda.CUDAGraph() for _ in range(len(self.sub_graphs))] if i < 4 else [] for i in range(len(self.cuda_graphs))]
        self.page_idx_buf = [torch.zeros([self.cuda_graphs[i]], dtype=torch.int32, device = self.device) for i in range(len(self.cuda_graphs))]
        self.page_offset_buf = [torch.zeros([self.cuda_graphs[i]], dtype=torch.int32, device = self.device) for i in range(len(self.cuda_graphs))]
 
        self.num_mini_batches = num_mini_batches

        self.max_chunk_size = max_chunk_size

        self.bsz_tensor_buf = torch.empty((1, ),dtype=torch.int32, device=device)
        self.num_tokens_tensor_buf = torch.empty((1, ),dtype=torch.int32, device=device)
        self.dynamic_topk = Config().rapidmoe_mode == "dynamic"
        self.rapidmoe_static_r = Config().rapidmoe_static_r
        
        
        if isinstance(self.model.model.layers[11].mlp.experts.generate_experts, KExpertsHybrid):
            if isinstance(self.model, KDeepseekV3ForCausalLM):
                self.layers_list = [self.model.model.layers[idx].mlp.experts.generate_experts for idx in range(3, 61)]
            elif isinstance(self.model, KQwen2MoeForCausalLM) or isinstance(self.model, KQwen3MoeForCausalLM):
                self.layers_list = [self.model.model.layers[idx].mlp.experts.generate_experts for idx in range(6, 94)]
            else:
                self.layers_list = None
        else:
            self.layers_list = None

        if not isinstance(self.model, KDeepseekV3ForCausalLM):
            raise ValueError("RapidMoE AE balance-serve supports only DeepSeek-V3")
        self.rapidmoe_deployment = configure_rapidmoe_layers(
            self.layers_list, Config().rapidmoe_mode, Config().rapidmoe_profile,
            self.rapidmoe_static_r,
        )
        print("[RapidMoE deployment] " + json.dumps(self.rapidmoe_deployment, sort_keys=True))

    def model_attn_plan(self, batch, cuda_graph_idx=0):
        if isinstance(self.model, KDeepseekV3ForCausalLM):
            self.model.flash_infer_attn_plan(batch, self.bsz_tensor_buf, self.num_tokens_tensor_buf,
                                             num_heads=self.model.config.num_attention_heads, head_dim_ckv=self.model.config.kv_lora_rank, 
                                             head_dim_kpe=self.model.config.qk_rope_head_dim, page_size=self.model.cache.page_size, causal=True,
                                             sm_scale=self.model.model.layers[0].self_attn.softmax_scale, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
            # DeepSeek-V3 uses the verified two-GPU hybrid configuration.
            self.all_cuda_device= ["cuda:0","cuda:1"]
        elif isinstance(self.model, KQwen2MoeForCausalLM) or isinstance(self.model, KQwen3MoeForCausalLM):
            self.model.flash_infer_attn_plan(batch, self.bsz_tensor_buf, self.num_tokens_tensor_buf,
                                             num_q_heads=self.model.config.num_attention_heads, num_kv_heads=self.model.config.num_key_value_heads,
                                             head_dim=self.model.config.head_dim if hasattr(self.model.config, 'head_dim') else self.model.config.hidden_size // self.model.config.num_attention_heads, 
                                             page_size=self.model.cache.page_size, causal=True,
                                             q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16, cuda_graph_idx=cuda_graph_idx)
            # The retained Qwen compatibility path uses one GPU.
            self.all_cuda_device= ["cuda:0"]
        else:
            assert False, "model type not supported"


    def set_subgraph_idx(self,dynamic_topk, sub_graph_idx = -1):
        """Apply one split-expert setting to every hybrid layer."""
        if self.layers_list is None:
            return
        for layer in self.layers_list:
            layer.dynamic_topk = dynamic_topk
            layer.threshold_enabled = bool(dynamic_topk)
            layer.subgraph_idx = sub_graph_idx
            layer.flex_decode_idx.copy_(torch.tensor([sub_graph_idx], device=layer.flex_decode_idx.device,dtype=torch.int32))
            layer.flex_decode_topk.copy_(torch.tensor([self.sub_graphs[sub_graph_idx]], device=layer.flex_decode_topk.device,dtype=torch.int32))

    def set_subgraph_idx_prefill(self, dynamic_topk, sub_graph_idx = -1):
        if self.layers_list is None:
            return
        for layer in self.layers_list:
            layer.dynamic_topk = dynamic_topk
            layer.threshold_enabled = bool(dynamic_topk)
            layer.subgraph_idx = sub_graph_idx
            if hasattr(layer, "flex_prefill_idx"):
                layer.flex_prefill_idx.copy_(torch.tensor([sub_graph_idx], device=layer.flex_prefill_idx.device, dtype=torch.int32))
            if hasattr(layer, "flex_prefill_topk"):
                layer.flex_prefill_topk.copy_(torch.tensor([self.sub_graphs[sub_graph_idx]], device=layer.flex_prefill_topk.device, dtype=torch.int32))
    
    def warmup(self):
        def capture_graphs(cuda_graph_idx):
            capture_r = 0 if self.dynamic_topk else self.rapidmoe_static_r
            self.set_subgraph_idx(dynamic_topk=self.dynamic_topk, sub_graph_idx=capture_r)
            with torch.cuda.graph(self.graphs[cuda_graph_idx], pool=self.graph_memory_pool, stream=self.stream):
                self.outputs_buf[cuda_graph_idx] = self.model(self.input[cuda_graph_idx], self.features_buf[cuda_graph_idx], self.bsz_tensor_buf, self.num_tokens_tensor_buf, self.page_idx_buf[cuda_graph_idx], self.page_offset_buf[cuda_graph_idx], cuda_graph_idx=cuda_graph_idx)   
                self.stream.wait_stream(torch.cuda.current_stream())
                torch.cuda.set_device(self.device)
                torch.cuda.set_stream(self.stream)
            self.graph_memory_pool = self.graphs[cuda_graph_idx].pool()

        def capture_subgraphs(cuda_graph_idx, sub_graph_idx):
            """Capture a graph for a fixed split-expert setting."""
            self.set_subgraph_idx(dynamic_topk=True, sub_graph_idx=sub_graph_idx)
            # Reuse the main graph's memory pool.
            with torch.cuda.graph(self.subgraphs[cuda_graph_idx][sub_graph_idx], pool=self.graph_memory_pool, stream=self.stream):
                self.outputs_buf[cuda_graph_idx] = self.model(self.input[cuda_graph_idx], self.features_buf[cuda_graph_idx], self.bsz_tensor_buf, self.num_tokens_tensor_buf, self.page_idx_buf[cuda_graph_idx], self.page_offset_buf[cuda_graph_idx], cuda_graph_idx=cuda_graph_idx)   
                self.stream.wait_stream(torch.cuda.current_stream())
                torch.cuda.set_device(self.device)
                torch.cuda.set_stream(self.stream)

        self.input = []
        self.features_buf = []
        self.outputs_buf = []
        self.bsz_tensor_buf = torch.tensor([0], dtype=torch.int32, device=self.device)
        self.num_tokens_tensor_buf = torch.tensor([0], dtype=torch.int32, device=self.device)
        for i in range(len(self.cuda_graphs)):
            prefill_query_length = (self.cuda_graphs[i] - Config().max_decode_batch_size) // Config().max_prefill_batch_size if self.cuda_graphs[i] > Config().max_decode_batch_size else 0
            self.input.append(ForwardBatchInput.gen_max_forward_batch(device=self.device, num_mini_batches = self.num_mini_batches, prefill_query_length=prefill_query_length, prefill_active_length=prefill_query_length, page_size=self.page_size, cuda_lens=self.cuda_graphs[i]))

            self.features_buf.append(self.model.batch_embeddings(self.input[i]))
            batch_size = self.input[i].minibatch.q_indptr.size(0)-1
            num_tokens = self.features_buf[i][0].size(0)
            print(f"[RapidMoE] Capturing CUDA Graph: batch={batch_size}, tokens={num_tokens}")

            if isinstance(self.model, KQwen2MoeForCausalLM) or isinstance(self.model, KQwen3MoeForCausalLM):
                self.model.init_wrapper(self.use_cuda_graph, self.device, num_tokens, batch_size, self.block_num, i)

            self.bsz_tensor_buf[0] = batch_size
            self.num_tokens_tensor_buf[0] = num_tokens

            self.model_attn_plan(self.input[i], i)
        
            page_idx, page_offset = self.model.cache.get_page_table(self.input[i].minibatch.position_ids, self.input[i].minibatch.q_indptr, self.input[i].minibatch.kv_indptr, self.input[i].minibatch.kv_indices, self.num_tokens_tensor_buf)

            
            self.page_idx_buf[i][:num_tokens].copy_(page_idx[:num_tokens])
            self.page_offset_buf[i][:num_tokens].copy_(page_offset[:num_tokens])

            self.page_idx_buf[i][num_tokens:].fill_(self.model.cache.max_cache_len // self.model.cache.page_size -1) 
        
            self.outputs_buf.append(None)
        
            torch.cuda.synchronize()
            for warm_up_iters in range(11):
                warmup_r = 0 if self.dynamic_topk else self.rapidmoe_static_r
                self.set_subgraph_idx(self.dynamic_topk, warmup_r)
                with torch.cuda.stream(self.stream):
                    self.outputs_buf[i] = self.model(self.input[i], self.features_buf[i], self.bsz_tensor_buf, self.num_tokens_tensor_buf, self.page_idx_buf[i], self.page_offset_buf[i], cuda_graph_idx=i)
                    self.stream.wait_stream(torch.cuda.current_stream())
                    torch.cuda.set_device(self.device)
                    torch.cuda.set_stream(self.stream)
            # Synchronize every GPU used by the hybrid configuration.
            for device in self.all_cuda_device:
                torch.cuda.synchronize(device)

            self.outputs_buf[i].num_batchs = batch_size
            
            capture_graphs(i)

            with torch.cuda.stream(self.stream):
                self.graphs[i].replay()
            self.sync(calc_time=False)
            print(f"[RapidMoE] CUDA Graph {i + 1}/{len(self.cuda_graphs)} ready")

        # Graph capture changes the routing scratch tensors. Restore the public
        # deployment contract before accepting the first API request.
        self.rapidmoe_deployment = configure_rapidmoe_layers(
            self.layers_list, Config().rapidmoe_mode, Config().rapidmoe_profile,
            self.rapidmoe_static_r,
        )
        

    def run(self, batch: sched_ext.BatchQueryTodo = None, query_manager: QueryManager = None):
        if batch is None or query_manager is None:
            return
        with torch.cuda.stream(self.stream):
            batch_size = len(batch.prefill_mini_batches)
            num_tokens = 0
            for i in range(len(batch.decode_mini_batches)):
                batch_size += len(batch.decode_mini_batches[i])
                num_tokens += len(batch.decode_mini_batches[i])

            for i in range(len(batch.prefill_mini_batches)):
                num_tokens += batch.prefill_mini_batches[i][2]

            # cuda graph idx equal to min idx i in self.cuda_graphs, that self.cuda_graphs[i] > num_tokens
            cuda_graph_idx = next((i for i, token in enumerate(self.cuda_graphs) if token >= num_tokens), len(self.cuda_graphs))
            if not self.use_cuda_graph:
                cuda_graph_idx = 0
    
            if self.use_cuda_graph:
                self.input[cuda_graph_idx].fill(batch, query_manager, self.page_size)
            else:
                self.input = [ForwardBatchInput(batch=batch, query_manager=query_manager, device=self.device)]
            if self.use_cuda_graph:
                self.features = self.model.batch_embeddings(self.input[cuda_graph_idx], device=self.device)
            else:
                self.features = self.model.batch_embeddings(self.input[cuda_graph_idx], device=self.device)


            self.bsz_tensor_buf.copy_(batch_size)
            self.num_tokens_tensor_buf.copy_(torch.tensor([num_tokens], dtype=torch.int32, device=self.device))

            if self.use_cuda_graph:
                self.features_buf[cuda_graph_idx][0].copy_(self.features[0], non_blocking=True)

            self.model_attn_plan(self.input[cuda_graph_idx], cuda_graph_idx)
            self.start_model_event.record(self.stream)
            page_idx, page_offset = self.model.cache.get_page_table(self.input[cuda_graph_idx].minibatch.position_ids, self.input[cuda_graph_idx].minibatch.q_indptr, self.input[cuda_graph_idx].minibatch.kv_indptr, self.input[cuda_graph_idx].minibatch.kv_indices, self.num_tokens_tensor_buf)
            if self.use_cuda_graph:
                self.page_idx_buf[cuda_graph_idx][:num_tokens].copy_(page_idx[:num_tokens])
                self.page_offset_buf[cuda_graph_idx][:num_tokens].copy_(page_offset[:num_tokens])

                self.page_idx_buf[cuda_graph_idx][num_tokens:].fill_(self.model.cache.max_cache_len // self.model.cache.page_size -1)
   
                self.replay(cuda_graph_idx)
                self.stream.synchronize()
                self.output = ForwardBatchOutput()
                self.output.top_ps.append(self.input[cuda_graph_idx].minibatch.top_ps)
                self.output.temperatures.append(self.input[cuda_graph_idx].minibatch.temperatures)

                self.output.logits.append(self.outputs_buf[cuda_graph_idx].logits[0][self.input[cuda_graph_idx].minibatch.logits_start].clone())
            else:
                self.output = self.model(self.input[cuda_graph_idx], self.features, self.bsz_tensor_buf, self.num_tokens_tensor_buf, page_idx, page_offset)
                self.output.logits[0] = self.output.logits[0][self.input[cuda_graph_idx].minibatch.logits_start]
                self.output.top_ps.append(self.input[cuda_graph_idx].minibatch.top_ps)
                self.output.temperatures.append(self.input[cuda_graph_idx].minibatch.temperatures)
            self.end_model_event.record(self.stream)
    def replay(self, cuda_graph_idx=-1):
        with torch.cuda.stream(self.stream):
            if cuda_graph_idx != -1:
                self.graphs[cuda_graph_idx].replay()
            else:
                self.graphs.replay()
    
    def sub_replay(self, cuda_graph_idx=-1, sub_graph_idx=-1):
        if cuda_graph_idx == 0:
            self.set_subgraph_idx(self.dynamic_topk,sub_graph_idx)
            self.replay(cuda_graph_idx)
        else:
            self.replay(cuda_graph_idx)

    def sync(self, calc_time = True):
        self.stream.synchronize()
        if calc_time:
            self.model_time = self.start_model_event.elapsed_time(self.end_model_event)  # In ms
