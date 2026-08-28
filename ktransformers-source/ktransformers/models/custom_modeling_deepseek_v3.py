"""
Date: 2024-11-06 10:05:11
LastEditors: djw
LastEditTime: 2024-11-13 07:50:51
"""

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ktransformers.server.balance_serve.inference.forward_batch import ForwardBatchInput, ForwardBatchOutput
from ktransformers.models.custom_cache import KDeepSeekV3Cache
from ktransformers.models.modeling_deepseek_v3 import DeepseekV3Model,  DeepseekV3PreTrainedModel
from ktransformers.models.configuration_deepseek_v3 import DeepseekV3Config


torch.set_grad_enabled(False)
torch.set_default_dtype(torch.bfloat16)
import flashinfer

class KDeepseekV3ForCausalLM(DeepseekV3PreTrainedModel):

    cache: KDeepSeekV3Cache
    use_cuda_graph = False
    def __init__(
        self,
        config: DeepseekV3Config,
        cache,
    ):
        super().__init__(config)
        self.model = DeepseekV3Model(config)
        self.config = config
        self.cache = cache
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    def init_wrapper(self, use_cuda_graph, device, max_batch_size, max_pages):
        self.use_cuda_graph = use_cuda_graph
        self.workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8).to(0)
        self.qo_indptr_buf = torch.empty((max_batch_size+2,), dtype=torch.int32, device=device)
        self.paged_kv_indptr_buf = torch.empty((max_batch_size+2,), dtype=torch.int32, device=device)
        self.paged_kv_indices_buf = torch.empty((max_pages,), dtype=torch.int32, device=device)
        self.paged_kv_len_buf = torch.empty((max_batch_size+1,), dtype=torch.int32, device=device)
        self.bsz_tensor_buf = torch.empty((1, ), dtype=torch.int32, device=device)
		

        self.wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
            self.workspace_buffer, use_cuda_graph=use_cuda_graph,
            qo_indptr=self.qo_indptr_buf,kv_indptr=self.paged_kv_indptr_buf,
            kv_indices=self.paged_kv_indices_buf,kv_len_arr=self.paged_kv_len_buf,
            bsz_tensor=self.bsz_tensor_buf,
            backend = "fa2",
        )
        
        # only for hybrid-2
        self.workspace_buffer1 = torch.empty(128 * 1024 * 1024, dtype=torch.int8).to(1)
        self.qo_indptr_buf1 = torch.empty((max_batch_size+2,), dtype=torch.int32, device="cuda:1")
        self.paged_kv_indptr_buf1 = torch.empty((max_batch_size+2,), dtype=torch.int32, device="cuda:1")
        self.paged_kv_indices_buf1 = torch.empty((max_pages,), dtype=torch.int32, device="cuda:1")
        self.paged_kv_len_buf1 = torch.empty((max_batch_size+1,), dtype=torch.int32, device="cuda:1")
        self.bsz_tensor_buf1 = torch.empty((1, ), dtype=torch.int32, device="cuda:1")
		

        self.wrapper1 = flashinfer.mla.BatchMLAPagedAttentionWrapper(
            self.workspace_buffer1, use_cuda_graph=use_cuda_graph,
            qo_indptr=self.qo_indptr_buf1,kv_indptr=self.paged_kv_indptr_buf1,
            kv_indices=self.paged_kv_indices_buf1,kv_len_arr=self.paged_kv_len_buf1,
            bsz_tensor=self.bsz_tensor_buf1
        )
        

    def batch_embeddings(self, batch: ForwardBatchInput, device="cuda:0"):
        features = []
        for i in range(batch.batch_size):
            tokens = batch.minibatch.tokens.contiguous()
            feature = (
                self.model.embed_tokens(tokens.to(torch.device('cpu')))
                .to(torch.bfloat16)
                .to(device=device)
            )
            features.append(feature)

        return features


    def forward(
        self,
        batch: ForwardBatchInput | None = None,
        features: List[torch.Tensor] | None = None,
        bsz_tensors: torch.Tensor | None = None,
        num_tokens_tensors: torch.Tensor | None = None,
        page_idx: torch.Tensor | None = None,
        page_offset: torch.Tensor | None = None,
        cuda_graph_idx: int | None = -1
    ) -> ForwardBatchOutput:
        current_stream = torch.cuda.current_stream()

        forward_batch_output = ForwardBatchOutput()

        
        hidden_states = features[0]
        position_ids = batch.minibatch.position_ids
        #with torch.cuda.stream(current_stream):
        residual = torch.zeros_like(hidden_states)
        cuda_graph_idx = cuda_graph_idx if cuda_graph_idx is not None else 0
        for i, decode_layer in enumerate(self.model.layers):
            # Expert Deferral: 在进入本层时，sync 上一层 MoE 的 phase1+phase2 并将 CPU 专家结果合并到 hidden_states（phase2 已在上一层 MoE 返回时 start_deferred，与本层 Attention 重叠）
            # 第3层做deferral, 而deferral的结果在第4层MoE拿到，因此第5层才需要add_pending
            if i > 0 and (i - 2) >= self.config.first_k_dense_replace:
                prev_mlp = self.model.layers[i - 2].mlp
                if hasattr(prev_mlp, "experts") and hasattr(prev_mlp.experts, "generate_experts"):
                    ge = prev_mlp.experts.generate_experts
                    if getattr(ge, "expert_deferral_enabled", False):
                        ge.add_pending_to(hidden_states, cuda_graph_idx)
            # can't use now, only one flashinfer wrapper
            if self.model.transfer_map is not None and i in self.model.transfer_map:
                prev_stream = torch.cuda.current_stream()
                cur_device = self.model.transfer_map[i]
                if cur_device not in self.model.stream_device_map:
                    self.model.stream_device_map[cur_device] = torch.cuda.Stream(cur_device)
                torch.cuda.set_device(cur_device)
                self.model.stream_device_map[cur_device].wait_stream(prev_stream)
                torch.cuda.set_stream(self.model.stream_device_map[cur_device])
                hidden_states = hidden_states.to(
                    self.model.transfer_map[i], non_blocking=True
                )
                # modify to support multi-gpu 
                # only for 
                residual = residual.to(
                    self.model.transfer_map[i], non_blocking=True
                )
                num_tokens_tensors = num_tokens_tensors.to(
                    self.model.transfer_map[i], non_blocking=True
                )

                position_ids = (
                    position_ids.to(self.model.transfer_map[i], non_blocking=True)
                    if position_ids is not None
                    else None
                )
                page_idx = (
                    page_idx.to(self.model.transfer_map[i], non_blocking=True)
                    if page_idx is not None
                    else None
                )
                page_offset = (
                    page_offset.to(self.model.transfer_map[i],non_blocking=True)
                    if page_offset is not None
                    else None
                )
            hidden_states, residual = decode_layer.input_layernorm(hidden_states, num_tokens_tensors, residual)
            # only for Hybrid-2
            wrapper = self.wrapper1 if i > 30 else self.wrapper
            #wrapper = self.wrapper
            
            hidden_states = decode_layer.self_attn(hidden_states, self.cache, 
                                                    position_ids=position_ids, 
                                                    wrapper=wrapper, num_tokens_tensors=num_tokens_tensors, 
                                                    page_idx=page_idx,
                                                    page_offset=page_offset
                                                    )
            #print(f"layer_{i} attn_out:")
            #print(hidden_states)
            #hidden_states = decode_layer.self_attn(hidden_states, self.cache, 
            #                                       position_ids=batch.minibatch.position_ids, 
            #                                       wrapper=self.wrapper, num_tokens_tensors=num_tokens_tensors, 
            #                                       page_idx=page_idx,
            #                                       page_offset=page_offset
            #                                       )
            hidden_states, residual = decode_layer.post_attention_layernorm(hidden_states, num_tokens_tensors, residual)
            if i > 0 and (i - 1) >= self.config.first_k_dense_replace:
                # sync for layer i-1 
                # 第3层做deferral, 第4层需要sync
                prev_mlp = self.model.layers[i - 1].mlp
                if hasattr(prev_mlp, "experts") and hasattr(prev_mlp.experts, "generate_experts"):
                    ge = prev_mlp.experts.generate_experts
                    if getattr(ge, "expert_deferral_enabled", False):
                        ge.sync_only(cuda_graph_idx)
            if i < self.config.first_k_dense_replace:
                hidden_states = decode_layer.mlp(hidden_states, num_tokens_tensors)
            else:
                hidden_states = decode_layer.mlp(hidden_states.unsqueeze(0), num_tokens_tensors, cuda_graph_idx)
                hidden_states = hidden_states.squeeze(0)
                #print(f"layer_{i} moe_out")
                #print(hidden_states)
        forward_batch_output = ForwardBatchOutput()
        calib_logit =self.model.norm(hidden_states, num_tokens_tensors, residual)[0]
        forward_batch_output.calib_logits.append(calib_logit)
        local_logit = self.lm_head(calib_logit, num_tokens_tensors)
        forward_batch_output.logits.append(local_logit)
        return forward_batch_output
    

               
    def flash_infer_attn_plan(self, batch: ForwardBatchInput, bsz_tensors, num_tokens_tensors,
        num_heads: int,
        head_dim_ckv: int,
        head_dim_kpe: int,
        page_size: int,
        causal: bool,
        sm_scale: float,
        q_data_type: torch.dtype,
        kv_data_type: torch.dtype,):
        minibatch = batch.minibatch
        self.wrapper.plan(minibatch.q_indptr, minibatch.kv_indptr, minibatch.kv_indices, 
                          minibatch.kv_len, num_heads, head_dim_ckv, head_dim_kpe, page_size, causal, sm_scale, q_data_type, kv_data_type, bsz_tensors)
        
        # only for hybrid-2
        bsz_tensors1 = bsz_tensors.to("cuda:1")
        q_indptr_cur = minibatch.q_indptr.to("cuda:1")
        kv_indptr_cur= minibatch.kv_indptr.to("cuda:1")
        kv_indices_cur= minibatch.kv_indices.to("cuda:1")
        kv_len_cur   = minibatch.kv_len.to("cuda:1")
        self.wrapper1.plan(q_indptr_cur, kv_indptr_cur, kv_indices_cur, 
                    kv_len_cur, num_heads, head_dim_ckv, head_dim_kpe, page_size, causal, sm_scale, q_data_type, kv_data_type, bsz_tensors1)
        