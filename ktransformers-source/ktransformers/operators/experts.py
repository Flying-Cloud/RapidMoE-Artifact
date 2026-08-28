#!/usr/bin/env python
# coding=utf-8
"""Expert backends and MoE wrappers used by KTransformers and RapidMoE."""

import numpy as np
from torch import Tensor, nn
import torch.nn.functional as F
import torch
import os
import sys
import triton
from ktransformers.operators.base_operator import BaseInjectedModule
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ktransformers_ext", "build"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ktransformers_ext", "build", "Release"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ktransformers_ext", "build", "Debug"))
from cpuinfer_ext.moe import MOEConfig, MOE
from cpuinfer_ext.flexmoe import FlexMOEConfig, FlexMOE
import ctypes
from ktransformers.util.custom_gguf import GGMLQuantizationType, GGUFLoader, offs_concate_experts
from ktransformers.util.utils import InferenceState
from ktransformers.server.config.config import Config
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from abc import ABC, abstractmethod
from ktransformers.operators.linear import KLinearMarlin
from ktransformers.operators.cpuinfer import CPUInfer
import KTransformersOps 


def deduplicate_and_sort(lst):
    return sorted(set(lst))
cuda_graphs  = deduplicate_and_sort([1, 2, 3, Config().max_batch_size,512])
sub_graphs   = [0,1,2,3,4,5,6]

class KExpertsBase(ABC):
    def __init__(self, key: str, gguf_loader: GGUFLoader, config: PretrainedConfig, orig_module: nn.Module, device: str = "cuda", **kwargs):
        self.key = key
        self.gguf_loader = gguf_loader
        self.config = config
        self.device = device
    
    @abstractmethod
    def forward(self, input_tensor, expert_ids, weights):
        pass

    @abstractmethod
    def load(self, w: dict | nn.Parameter | tuple | None = None, device: str = "cpu", warmup: bool = False):
        pass
    
    @abstractmethod
    def unload():
        pass

    def load_weights(self, override_key: str | None = None, device: str = "cpu"):
        res = {}
        if override_key is not None:
            keys = override_key
        else:
            keys = [self.key]

        gate = None
        up = None
        down = None
        gate_type = None
        up_type = None
        down_type = None

        for key in keys:
            if key + ".ffn_gate_exps.weight" in self.gguf_loader.tensor_info:
                targets = [".ffn_gate_exps.weight", ".ffn_up_exps.weight", ".ffn_down_exps.weight" ]
                tensors = self.load_multi(key, targets, device=device)
                gate = tensors[".ffn_gate_exps.weight"]
                up = tensors[".ffn_up_exps.weight"]
                down = tensors[".ffn_down_exps.weight"]
                gate_type = self.gguf_loader.tensor_info[key + ".ffn_gate_exps.weight"]["ggml_type"]
                up_type = self.gguf_loader.tensor_info[key + ".ffn_up_exps.weight"]["ggml_type"]
                down_type = self.gguf_loader.tensor_info[key + ".ffn_down_exps.weight"]["ggml_type"]
            elif key + ".ffn_down.0.weight" in self.gguf_loader.tensor_info:
                # for supporting  Mixtral-8x7B-Instuct  
                gate = []
                up = []
                down = []
                for i in range(8):
                    gatei, upi, downi = f".ffn_gate.{i}.weight", f".ffn_up.{i}.weight", f".ffn_down.{i}.weight"
                    targets = [gatei, upi, downi]
                    tensors = self.load_multi(key, targets, device=device)
                    gate_it, up_it, down_it = tensors[gatei], tensors[upi], tensors[downi]
                    gate.append(gate_it)
                    up.append(up_it)
                    down.append(down_it)
                gate = torch.stack(gate)
                up = torch.stack(up)
                down = torch.stack(down)
                gate_type = self.gguf_loader.tensor_info[key + ".ffn_gate.0.weight"]["ggml_type"]
                up_type = self.gguf_loader.tensor_info[key + ".ffn_up.0.weight"]["ggml_type"]
                down_type = self.gguf_loader.tensor_info[key + ".ffn_down.0.weight"]["ggml_type"]
            else:
                raise ValueError(f"Experts {key} not found in gguf_loader")
            res = {key:{"gate": gate, "up": up, "down": down, "gate_type": gate_type, "up_type": up_type, "down_type": down_type}}
        return res
    
    def load_multi(self, key: str, keys: list[str], device: str = "cpu"):
        tensors = {}
        for k in keys:
            tensors[k] = self.gguf_loader.load_gguf_tensor(key + k, device=device)
        return tensors


class KExpertsCPU(KExpertsBase):
    input_tensor_cpu:Tensor = None
    expert_ids_cpu:Tensor = None
    weights_cpu:Tensor = None
    output_cpu:Tensor = None
    output_gpu_map:dict = {}  # Output buffers keyed by GPU device.
    CPU_INFER = CPUInfer(Config().cpu_infer)
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        n_routed_experts: int,
        orig_module: nn.Module = None,
        device: str = "cpu",
        out_device: str = "cuda",
        **kwargs
    ):
        super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)
        assert device.lower() == "cpu", "KExpertsCPU can only be loaded on CPU"
        self.n_routed_experts = n_routed_experts
        self.out_device = out_device
        self.backend = kwargs.get("backend", "llamafile")

    def load(self, w: dict | nn.Parameter | tuple | None = None, device:str|None = None, warmup:bool = False):
        if device:
            assert device.lower() == "cpu", "KExpertsCPU can only be loaded on CPU, Parameter \"device\" can be cpu or None."
        if w is None: 
            w = self.load_weights()[self.key]
        self.gate = w["gate"]
        self.up = w["up"]
        self.down = w["down"]
        self.gate_type = w["gate_type"]
        self.up_type = w["up_type"]
        self.down_type = w["down_type"]
        gate_ptr = ctypes.addressof(
            ctypes.cast(self.gate.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents
        )
        up_ptr = ctypes.addressof(
            ctypes.cast(self.up.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents
        )
        down_ptr = ctypes.addressof(
            ctypes.cast(self.down.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents
        )
        n_routed_experts = self.n_routed_experts
        self.cpu_infer = KExpertsCPU.CPU_INFER
        if self.backend == "llamafile":
            moe_config = MOEConfig(
                n_routed_experts,
                self.config.num_experts_per_tok,
                self.config.hidden_size,
                self.config.moe_intermediate_size,
                64,
                10,
                1024,
                gate_ptr,
                up_ptr,
                down_ptr,
                self.gate_type,
                self.up_type,
                self.down_type,
                30,
            )
            self.moe = MOE(moe_config)
        elif self.backend == "AMXBF16":
            from cpuinfer_ext.moe import AMX_MOEConfig, AMXBF16_MOE
            assert self.gate_type == GGMLQuantizationType.BF16
            assert self.up_type == GGMLQuantizationType.BF16
            assert self.down_type == GGMLQuantizationType.BF16
            moe_config = AMX_MOEConfig(
                n_routed_experts,
                self.config.num_experts_per_tok,
                self.config.hidden_size,
                self.config.moe_intermediate_size,
                25600,
                gate_ptr,
                up_ptr,
                down_ptr,
            )
            self.moe = AMXBF16_MOE(moe_config)
            self.cpu_infer.submit(self.moe.load_weights())
            self.cpu_infer.sync()
        elif self.backend == "AMXInt8":
            from cpuinfer_ext.moe import AMX_MOEConfig, AMXInt8_MOE
            assert self.gate_type == GGMLQuantizationType.BF16
            assert self.up_type == GGMLQuantizationType.BF16
            assert self.down_type == GGMLQuantizationType.BF16
            moe_config = AMX_MOEConfig(
                n_routed_experts,
                self.config.num_experts_per_tok,
                self.config.hidden_size,
                self.config.moe_intermediate_size,
                25600,
                gate_ptr,
                up_ptr,
                down_ptr,
            )
            self.moe = AMXInt8_MOE(moe_config)
            self.cpu_infer.submit(self.moe.load_weights())
            self.cpu_infer.sync()
        num_experts_per_tok = self.config.num_experts_per_tok
        if warmup:
            self.cpu_infer.submit(self.moe.warm_up())
            self.cpu_infer.sync()
        if self.out_device not in KExpertsCPU.output_gpu_map:
            if isinstance(cuda_graphs, list):
                KExpertsCPU.output_gpu_map[self.out_device] = [torch.zeros((cuda_graphs[i], self.config.hidden_size), device=self.out_device,dtype=torch.bfloat16) for i in range(len(cuda_graphs))]
            else:
                KExpertsCPU.output_gpu_map[self.out_device] = torch.zeros((cuda_graphs, self.config.hidden_size), device=self.out_device,dtype=torch.bfloat16)
        if KExpertsCPU.input_tensor_cpu == None:
            if isinstance(cuda_graphs, list):
                KExpertsCPU.input_tensor_cpu = [torch.zeros((cuda_graphs[i], self.config.hidden_size), device="cpu", pin_memory=True,dtype=torch.bfloat16) for i in range(len(cuda_graphs))]
                KExpertsCPU.expert_ids_cpu = [torch.zeros((cuda_graphs[i], num_experts_per_tok), device="cpu", dtype=torch.long, pin_memory=True) for i in range(len(cuda_graphs))]
                KExpertsCPU.weights_cpu = [torch.zeros((cuda_graphs[i], num_experts_per_tok), device="cpu", dtype=torch.float32, pin_memory=True) for i in range(len(cuda_graphs))]
                KExpertsCPU.output_cpu = [torch.zeros((cuda_graphs[i], self.config.hidden_size), device="cpu", pin_memory=True, dtype=torch.bfloat16) for i in range(len(cuda_graphs))]
                KExpertsCPU.bsz_tensor_cpu = [torch.zeros((1), device="cpu", dtype=torch.int32, pin_memory=True) for i in range(len(cuda_graphs))]
            else:
                KExpertsCPU.input_tensor_cpu = torch.zeros((cuda_graphs, self.config.hidden_size), device="cpu", pin_memory=True,dtype=torch.bfloat16)
                KExpertsCPU.expert_ids_cpu = torch.zeros((cuda_graphs, num_experts_per_tok), device="cpu", dtype=torch.long, pin_memory=True)
                KExpertsCPU.weights_cpu = torch.zeros((cuda_graphs, num_experts_per_tok), device="cpu", dtype=torch.float32, pin_memory=True)
                KExpertsCPU.output_cpu = torch.zeros((cuda_graphs, self.config.hidden_size), device="cpu", pin_memory=True, dtype=torch.bfloat16)
                KExpertsCPU.bsz_tensor_cpu = torch.zeros((1), device="cpu", dtype=torch.int32, pin_memory=True)
            
    def submit_for_one_decode(self, input_tensor, expert_ids, weights, bsz_tensor=None, cuda_graph_idx=0):
        if bsz_tensor is None:
            bsz_tensor = torch.ones(1, device=input_tensor.device, dtype=torch.int32)
        if cuda_graph_idx != -1:
            KExpertsCPU.input_tensor_cpu[cuda_graph_idx].copy_(input_tensor, non_blocking=True)
            KExpertsCPU.expert_ids_cpu[cuda_graph_idx].copy_(expert_ids[:], non_blocking=True)
            KExpertsCPU.weights_cpu[cuda_graph_idx].copy_(weights[:], non_blocking=True)
            KExpertsCPU.bsz_tensor_cpu[cuda_graph_idx].copy_(bsz_tensor, non_blocking=True)
            self.cpu_infer.submit_with_cuda_stream(torch.cuda.current_stream(self.out_device).cuda_stream, self.moe.forward(1, expert_ids.size(-1), KExpertsCPU.expert_ids_cpu[cuda_graph_idx].data_ptr(), KExpertsCPU.weights_cpu[cuda_graph_idx].data_ptr(), KExpertsCPU.input_tensor_cpu[cuda_graph_idx].data_ptr(), KExpertsCPU.output_cpu[cuda_graph_idx].data_ptr(), KExpertsCPU.bsz_tensor_cpu[cuda_graph_idx].data_ptr()))
        else:
            KExpertsCPU.input_tensor_cpu.copy_(input_tensor, non_blocking=True)
            KExpertsCPU.expert_ids_cpu.copy_(expert_ids, non_blocking=True)
            KExpertsCPU.weights_cpu.copy_(weights, non_blocking=True)
            KExpertsCPU.bsz_tensor_cpu.copy_(bsz_tensor, non_blocking=True)
            self.cpu_infer.submit_with_cuda_stream(torch.cuda.current_stream(self.out_device).cuda_stream, self.moe.forward(1, expert_ids.size(-1), KExpertsCPU.expert_ids_cpu.data_ptr(), KExpertsCPU.weights_cpu.data_ptr(), KExpertsCPU.input_tensor_cpu.data_ptr(), KExpertsCPU.output_cpu.data_ptr(), KExpertsCPU.bsz_tensor_cpu.data_ptr()))
        

    def sync_for_one_decode(self, cuda_graph_idx=0):
        if cuda_graph_idx != -1:
            self.cpu_infer.sync_with_cuda_stream(torch.cuda.current_stream(self.out_device).cuda_stream)
            KExpertsCPU.output_gpu_map[self.out_device][cuda_graph_idx].copy_(KExpertsCPU.output_cpu[cuda_graph_idx], non_blocking=True)
            return KExpertsCPU.output_gpu_map[self.out_device][cuda_graph_idx]
        else:
            self.cpu_infer.sync_with_cuda_stream(torch.cuda.current_stream(self.out_device).cuda_stream)
            KExpertsCPU.output_gpu_map[self.out_device].copy_(KExpertsCPU.output_cpu, non_blocking=True)
            return KExpertsCPU.output_gpu_map[self.out_device]

    def forward(self, input_tensor, expert_ids, weights, bsz_tensor=None, cuda_graph_idx=0):
        if bsz_tensor is None:
            bsz_tensor = torch.tensor([input_tensor.size(0)], device=input_tensor.device, dtype=torch.int32)
        if torch.cuda.is_current_stream_capturing():
            if cuda_graph_idx != -1:
                KExpertsCPU.input_tensor_cpu[cuda_graph_idx].copy_(input_tensor, non_blocking=True)
                KExpertsCPU.expert_ids_cpu[cuda_graph_idx].copy_(expert_ids, non_blocking=True)
                KExpertsCPU.weights_cpu[cuda_graph_idx].copy_(weights, non_blocking=True)
                KExpertsCPU.bsz_tensor_cpu[cuda_graph_idx].copy_(bsz_tensor, non_blocking=True)
                self.cpu_infer.submit_with_cuda_stream(torch.cuda.current_stream().cuda_stream, self.moe.forward(expert_ids.size(0), expert_ids.size(-1), KExpertsCPU.expert_ids_cpu[cuda_graph_idx].data_ptr(), KExpertsCPU.weights_cpu[cuda_graph_idx].data_ptr(), KExpertsCPU.input_tensor_cpu[cuda_graph_idx].data_ptr(), KExpertsCPU.output_cpu[cuda_graph_idx].data_ptr(), KExpertsCPU.bsz_tensor_cpu[cuda_graph_idx].data_ptr()))
                self.cpu_infer.sync_with_cuda_stream(torch.cuda.current_stream().cuda_stream)
                KExpertsCPU.output_gpu_map[self.out_device][cuda_graph_idx].copy_(KExpertsCPU.output_cpu[cuda_graph_idx], non_blocking=True)
                return KExpertsCPU.output_gpu_map[self.out_device][cuda_graph_idx]

            else:
                KExpertsCPU.input_tensor_cpu.copy_(input_tensor, non_blocking=True)
                KExpertsCPU.expert_ids_cpu.copy_(expert_ids, non_blocking=True)
                KExpertsCPU.weights_cpu.copy_(weights, non_blocking=True)
                KExpertsCPU.bsz_tensor_cpu.copy_(bsz_tensor, non_blocking=True)
                self.cpu_infer.submit_with_cuda_stream(torch.cuda.current_stream().cuda_stream, self.moe.forward(expert_ids.size(0), expert_ids.size(-1), KExpertsCPU.expert_ids_cpu.data_ptr(), KExpertsCPU.weights_cpu.data_ptr(), KExpertsCPU.input_tensor_cpu.data_ptr(), KExpertsCPU.output_cpu.data_ptr(), KExpertsCPU.bsz_tensor_cpu.data_ptr()))
                self.cpu_infer.sync_with_cuda_stream(torch.cuda.current_stream().cuda_stream)
                KExpertsCPU.output_gpu_map[self.out_device].copy_(KExpertsCPU.output_cpu, non_blocking=True)
                return KExpertsCPU.output_gpu_map[self.out_device]
        else:
            input_tensor = input_tensor.contiguous().cpu()
            expert_ids = expert_ids.contiguous().cpu()
            weights = weights.contiguous().to(torch.float32).cpu()
            bsz_tensor = bsz_tensor.contiguous().cpu()
            output = torch.empty_like(input_tensor).contiguous()
            self.cpu_infer.submit(self.moe.forward(expert_ids.size(0), expert_ids.size(1), expert_ids.data_ptr(), weights.data_ptr(), input_tensor.data_ptr(), output.data_ptr(), bsz_tensor.data_ptr()))
            self.cpu_infer.sync()
            return output.to(device=object.__getattribute__(self, "out_device"))

    def unload(self):
        return

    def load_weights(self, override_key: str | None = None, device: str = "cpu"):
        res = {}
        if override_key is not None:
            keys = override_key
        else:
            keys = [self.key]

        gate = None
        up = None
        down = None
        gate_type = None
        up_type = None
        down_type = None

        for key in keys:
            if self.gguf_loader.safetensor_loader is not None:
                # Safetensor-backed weights are materialized as NumPy arrays.
                gate = self.gguf_loader.safetensor_loader.load_tensor(key + ".ffn_gate_exps.weight").numpy()
                up = self.gguf_loader.safetensor_loader.load_tensor(key + ".ffn_up_exps.weight").numpy()
                down = self.gguf_loader.safetensor_loader.load_tensor(key + ".ffn_down_exps.weight").numpy()
                gate_type = self.gguf_loader.safetensor_loader.load_tensor(key + ".ffn_gate_exps.ggml_type").item()
                up_type = self.gguf_loader.safetensor_loader.load_tensor(key + ".ffn_up_exps.ggml_type").item()
                down_type = self.gguf_loader.safetensor_loader.load_tensor(key + ".ffn_down_exps.ggml_type").item()
            
            elif key + ".ffn_gate_exps.weight" in self.gguf_loader.tensor_info:
                gate = self.gguf_loader.get_mmap_tensor(key + ".ffn_gate_exps.weight")
                up = self.gguf_loader.get_mmap_tensor(key + ".ffn_up_exps.weight")
                down = self.gguf_loader.get_mmap_tensor(key + ".ffn_down_exps.weight")
                gate_type = self.gguf_loader.tensor_info[key + ".ffn_gate_exps.weight"]["ggml_type"]
                up_type = self.gguf_loader.tensor_info[key + ".ffn_up_exps.weight"]["ggml_type"]
                down_type = self.gguf_loader.tensor_info[key + ".ffn_down_exps.weight"]["ggml_type"]
            elif key + ".ffn_down.0.weight" in self.gguf_loader.tensor_info:
                # for supporting  Mixtral-8x7B-Instuct  
                gate = []
                up = []
                down = []
                for i in range(8):
                    gate_it = self.gguf_loader.get_mmap_tensor(f"{key}.ffn_gate.{i}.weight")
                    up_it = self.gguf_loader.get_mmap_tensor(f"{key}.ffn_up.{i}.weight")
                    down_it = self.gguf_loader.get_mmap_tensor(f"{key}.ffn_down.{i}.weight")
                    gate.append(gate_it)
                    up.append(up_it)
                    down.append(down_it)
                gate = np.stack(gate)
                up = np.stack(up)
                down = np.stack(down)
                gate_type = self.gguf_loader.tensor_info[key + ".ffn_gate.0.weight"]["ggml_type"]
                up_type = self.gguf_loader.tensor_info[key + ".ffn_up.0.weight"]["ggml_type"]
                down_type = self.gguf_loader.tensor_info[key + ".ffn_down.0.weight"]["ggml_type"]
            else:
                raise ValueError(f"Experts {key} not found in gguf_loader")
            res = {key:{"gate": gate, "up": up, "down": down, "gate_type": gate_type, "up_type": up_type, "down_type": down_type}}
        return res
residual_dict = {
    348 : 'IQ1_S_R4_RES_Q2_K',
    350 : 'IQ1_M_R4_RES_Q2_K',
    351 : 'IQ1_M_PLUS_Q4_K_R4'
}
ori_dict = {
    348 : 219, # IQ1_S_R4
    350 : 229,
    351 : 229
}
res_dict = {
    348: 210, # Q2_K_R4
    350: 210,
    351: 351
}
def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the
        top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.
    - expert_map: A tensor of shape [num_experts] that maps the expert index
        from the global space to the local index space of the current
        expert parallel shard. If the expert is not in the current expert
        parallel shard, the mapping is set to -1.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according
        to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding,
        ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process
    so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions
    align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
    block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts,
        with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in
        the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible
        by block_size for proper block matrix operations.
    """
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.empty((max_num_tokens_padded, ),
                             dtype=torch.int32,
                             device=topk_ids.device)
    sorted_ids.fill_(topk_ids.numel())
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    # Expert ids must be zeroed out to prevent index out of bounds error while
    # mapping global expert ids to local expert ids in expert parallelism.
    expert_ids = torch.zeros((max_num_m_blocks, ),
                             dtype=torch.int32,
                             device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1),
                                      dtype=torch.int32,
                                      device=topk_ids.device)
    KTransformersOps.moe_align_block_size(topk_ids, num_experts, block_size, sorted_ids,
                                 expert_ids, num_tokens_post_pad)
    if expert_map is not None:
        expert_ids = expert_map[expert_ids]

    return sorted_ids, expert_ids, num_tokens_post_pad

class KExpertsHybrid(KExpertsBase):
    output_gpu_map:dict = {} # Manage output tensor buffer on different gpu
    out_hidden_states:dict = {}
    intermediate_output: dict= {}

    input_tensor_cpu:Tensor = None
    expert_ids_cpu:Tensor = None
    weights_cpu:Tensor = None
    output_cpu:Tensor = None
    w12_projs_cpu:Tensor = None
    w12_projs_prefill_cpu:Tensor =None
    input_tensor_prefill_cpu: Tensor = None
    expert_ids_prefill_cpu: Tensor = None
    weights_prefill_cpu: Tensor = None
    outputs_prefill_cpu: Tensor = None
    # CPU buffers for the deferred expert phase.
    w12_projs_deferred_cpu: Tensor = None
    expert_ids_deferred_cpu: Tensor = None
    weights_deferred_cpu: Tensor = None
    output_deferred_cpu: Tensor = None
    output_cpu_to_gpu: dict = {}
    out_hidden_states_2: dict = {}
    CPU_INFER = CPUInfer(Config().cpu_infer)
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        n_routed_experts: int,
        orig_module: nn.Module = None,
        device: str = "cuda",
        out_device: str = "cuda",
        **kwargs
    ):
        super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)
        self.gguf_loader = gguf_loader
        self.n_routed_experts = n_routed_experts
        self.out_device = out_device
        self.act = KTransformersOps.silu_and_mul if config.hidden_act == "silu" else None
        # Respect the selected deployment mode. configure_rapidmoe_layers()
        # reasserts this after all layers have loaded.
        self.dynamic_topk = kwargs.get(
            "dynamic_topk", Config().rapidmoe_mode == "dynamic"
        )
        self.threshold_enabled = False
        
    def load(self, w: dict | nn.Parameter | tuple | None = None, device:str|None = None, warmup:bool = False, flex_topk:int = 4, prefill_topk:int = 3, prefill_topk_phase1: int | None = None): 
        self.flex_topk = flex_topk
        self.prefill_topk = prefill_topk
        # Split CPU prefill work so its second phase can overlap attention.
        self.prefill_topk_phase1 = prefill_topk_phase1 if prefill_topk_phase1 is not None else 1
        self.prefill_topk_phase2 = self.prefill_topk - self.prefill_topk_phase1
        self.expert_deferral_enabled = False
        self.max_topk = sub_graphs[-1]
        self.subgraph_idx = 0
        self.flex_decode_topk = torch.tensor([1],device = self.out_device, dtype=torch.int32)
        self.flex_decode_topk_cpu = torch.tensor([1], device="cpu", dtype=torch.long, pin_memory=True)
        self.flex_decode_idx  = torch.tensor([1],device = self.out_device,dtype = torch.int32)
        self.flex_prefill_topk = torch.tensor([3],device = self.out_device,dtype = torch.int32)
        self.flex_prefill_topk_cpu = torch.tensor([3], device="cpu", dtype=torch.long, pin_memory=True)
        self.flex_prefill_idx = torch.tensor([3],device = self.out_device, dtype=torch.int32)
        # Device scalar used by the static CUDA Graph guard. copy_() from this
        # tensor is capturable and is replayed before every MoE split.
        self.static_r_value = torch.ones((1,), device=self.out_device, dtype=torch.int32)
        self.prefill_thre = torch.tensor([0.], device = self.out_device, dtype = torch.float32)
        self.decode_thre  = torch.tensor([0.], device = self.out_device, dtype = torch.float32)
        self.prefill_alpha= torch.tensor([1.], device = self.out_device, dtype = torch.float32)
        self.decode_alpha = torch.tensor([1.], device = self.out_device, dtype = torch.float32)
        self.expert_start = torch.tensor([0], device=self.out_device,dtype = torch.int32)
        self.expert_end = torch.tensor([self.config.num_experts_per_tok],device = self.flex_decode_topk.device,dtype = torch.int32)
        self.exp2 = False
        if device:
            if device.lower() == "cpu":
                raise ValueError("KExpertsHybrid can only be loaded with GPU")
            if device != self.device:
                raise ValueError("KExpertsHybrid can only be loaded with the same device as the model outputs")
        if w is None: 
            w = self.load_weights()[self.key]
            load_by_experts = True
        self.gate = w["gate"]
        self.up = w["up"]
        self.down = w["down"]
        self.gate_type = w["gate_type"]
        self.up_type = w["up_type"]
        self.down_type = w["down_type"]

        self.gate_type_gpu = ori_dict[self.gate_type]
        self.up_type_gpu = ori_dict[self.up_type]
        self.down_type_gpu = ori_dict[self.down_type]
        gate_ptr = ctypes.addressof(
            ctypes.cast(self.gate.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents
        )
        up_ptr = ctypes.addressof(
            ctypes.cast(self.up.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents
        )
        down_ptr = ctypes.addressof(
            ctypes.cast(self.down.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents
        )
        # Build combined GPU tensors from the base and residual layouts.
        if load_by_experts:
            shape13 = [self.config.hidden_size,self.config.moe_intermediate_size,self.n_routed_experts]
            type13 = residual_dict[self.gate_type]
            self.w13 = torch.tensor(offs_concate_experts(self.gate,shape13,type13,self.up),dtype=torch.uint8,device=self.out_device)
            shape2 = [self.config.moe_intermediate_size,self.config.hidden_size,self.n_routed_experts]
            type2 = residual_dict[self.down_type]
            self.w2 = torch.tensor(offs_concate_experts(self.down,shape2,type2),dtype=torch.uint8,device=self.out_device)
            
        n_routed_experts = self.n_routed_experts
        gate_ptr = ctypes.addressof(
            ctypes.cast(self.gate.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents
        )
        up_ptr = ctypes.addressof(
            ctypes.cast(self.up.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents
        )
        down_ptr = ctypes.addressof(
            ctypes.cast(self.down.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents
        )
        moe_config = FlexMOEConfig(
            n_routed_experts,
            self.config.num_experts_per_tok,
            self.max_topk, # Dynamic_topk
            self.config.hidden_size,
            self.config.moe_intermediate_size,
            32, # 16 > 32 > 64
            10,
            1024,
            gate_ptr,
            up_ptr,
            down_ptr,
            self.gate_type,
            self.gate_type_gpu,
            res_dict[self.gate_type],
            self.up_type,
            self.up_type_gpu,
            res_dict[self.up_type],
            self.down_type,
            self.down_type_gpu,
            res_dict[self.down_type],
            30,
        )

        self.moe = FlexMOE(moe_config)
        
        self.cpu_infer = KExpertsHybrid.CPU_INFER
        if self.out_device not in KExpertsHybrid.output_gpu_map:
            if isinstance(cuda_graphs, list):
                #dynamic_topk
                KExpertsHybrid.intermediate_output[self.out_device] = [torch.zeros((cuda_graphs[i]*(self.config.num_experts_per_tok),self.config.moe_intermediate_size),device=self.out_device,dtype=torch.bfloat16) for i in range(len(cuda_graphs))]
                
                KExpertsHybrid.output_gpu_map[self.out_device] = [torch.zeros((cuda_graphs[i], self.config.hidden_size), device=self.out_device,dtype=torch.bfloat16) for i in range(len(cuda_graphs))]
                KExpertsHybrid.out_hidden_states[self.out_device]   = [torch.zeros((cuda_graphs[i], self.config.hidden_size), dtype=torch.bfloat16, device=self.out_device) for i in range(len(cuda_graphs))]
                KExpertsHybrid.out_hidden_states_2[self.out_device] = [torch.zeros((cuda_graphs[i], self.config.hidden_size), dtype=torch.bfloat16, device=self.out_device) for i in range(len(cuda_graphs))]
                KExpertsHybrid.output_cpu_to_gpu[self.out_device] = [torch.zeros((cuda_graphs[i], self.config.hidden_size), device=self.out_device,dtype=torch.bfloat16) for i in range(len(cuda_graphs))]
            else:
                KExpertsHybrid.output_gpu_map[self.out_device] = torch.zeros((self.config.hidden_size),device=self.out_device,dtype=torch.bfloat16)
                KExpertsHybrid.out_hidden_states[self.out_device]   = torch.zeros((self.config.hidden_size), dtype=torch.bfloat16, device=self.out_device)
                KExpertsHybrid.intermediate_output[self.out_device] = torch.zeros(((self.config.num_experts_per_tok - self.flex_topk),self.config.moe_intermediate_size),device=self.out_device,dtype=torch.bfloat16)
        def gen_ids_p1(j, k, device="cuda"):
            base = torch.arange(j, dtype=torch.int32, device=device) * k
            return base.repeat_interleave(k)
            
        if isinstance(cuda_graphs, list):
            # dynamic_topk
            self.expert_ids_p1 = [torch.empty((cuda_graphs[i] * self.config.num_experts_per_tok), dtype=torch.int32, device=self.out_device) for i in range(len(cuda_graphs))]
            self.expert_ids_p2 = [torch.empty((cuda_graphs[i] * self.config.num_experts_per_tok), dtype=torch.int32, device=self.out_device) for i in range(len(cuda_graphs))]
            self.weights_p2 = [torch.empty((cuda_graphs[i] * self.config.num_experts_per_tok), dtype=torch.float32, device=self.out_device) for i in range(len(cuda_graphs))]
            self.sorted_token_ids_p2 = [[torch.arange((cuda_graphs[i] * (self.config.num_experts_per_tok - topk)), dtype=torch.int32, device=self.out_device) for topk in sub_graphs] for i in range(len(cuda_graphs))]
            self.sorted_token_ids_p1 = [[gen_ids_p1(cuda_graphs[i],topk,device=self.out_device) for topk in sub_graphs] for i in range(len(cuda_graphs))]
            self.num_tokens_post_padded_p1 = [[torch.tensor([cuda_graphs[i] * topk], dtype=torch.int32, device=self.out_device) for topk in sub_graphs] for i in range(len(cuda_graphs))]
            self.num_tokens_post_padded_p2 = [[torch.tensor([cuda_graphs[i] * (self.config.num_experts_per_tok - topk)], dtype=torch.int32, device=self.out_device) for topk in sub_graphs] for i in range(len(cuda_graphs))]
            #self.expert_ids_p1 = [torch.empty((cuda_graphs[i] * self.flex_topk), dtype=torch.int32, device=self.out_device) for i in range(len(cuda_graphs))]
            #self.expert_ids_p2 = [torch.empty((cuda_graphs[i] * (self.config.num_experts_per_tok - self.flex_topk)), dtype=torch.int32, device=self.out_device) for i in range(len(cuda_graphs))]
            #self.weights_p2 = [torch.empty((cuda_graphs[i] * (self.config.num_experts_per_tok - self.flex_topk)), dtype=torch.float32, device=self.out_device) for i in range(len(cuda_graphs))]
            #self.sorted_token_ids_p2 = [torch.arange((cuda_graphs[i]* (self.config.num_experts_per_tok - self.flex_topk)),dtype = torch.int32, device=self.out_device) for i in range(len(cuda_graphs))]
            self.sorted_token_ids_p1_all = [torch.cat(self.sorted_token_ids_p1[i],dim=0) for i in range(len(cuda_graphs))]
            self.num_tokens_post_padded_p1_all = [torch.cat(self.num_tokens_post_padded_p1[i],dim=0) for i in range(len(cuda_graphs))]
            self.sorted_token_ids_p2_all = [torch.cat(self.sorted_token_ids_p2[i],dim=0) for i in range(len(cuda_graphs))]
            self.num_tokens_post_padded_p2_all = [torch.cat(self.num_tokens_post_padded_p2[i],dim=0) for i in range(len(cuda_graphs))]
            #self.sorted_token_ids_p1 = [gen_ids_p1(cuda_graphs[i],self.flex_topk,device=self.out_device) for i in range(len(cuda_graphs))]
            #self.num_tokens_post_padded_p1 = [torch.tensor([cuda_graphs[i] * self.flex_topk], dtype=torch.int32, device=self.out_device) for i in range(len(cuda_graphs))]
            #self.num_tokens_post_padded_p2 = [torch.tensor([cuda_graphs[i] * (self.config.num_experts_per_tok - self.flex_topk)], dtype=torch.int32, device=self.out_device) for i in range(len(cuda_graphs))]
            self.sorted_token_reverse = [torch.empty((cuda_graphs[i] * (self.config.num_experts_per_tok - self.flex_topk)), dtype=torch.int32, device=self.out_device) for i in range(len(cuda_graphs))]

            self.sorted_token_ids_p1_prefill = [gen_ids_p1(cuda_graphs[i],self.prefill_topk,device=self.out_device) for i in range(len(cuda_graphs))]
            self.num_tokens_post_padded_p1_prefill = [torch.tensor([cuda_graphs[i] * self.prefill_topk], dtype=torch.int32, device=self.out_device) for i in range(len(cuda_graphs))]
            self.expert_ids_p1_prefill = [torch.empty((cuda_graphs[i] * self.prefill_topk), dtype=torch.int32, device=self.out_device) for i in range(len(cuda_graphs))]
            self.sorted_token_ids_p2_prefill = [torch.arange((cuda_graphs[i]* (self.config.num_experts_per_tok - self.prefill_topk)),dtype = torch.int32, device=self.out_device) for i in range(len(cuda_graphs))]
            self.num_tokens_post_padded_p2_prefill = [torch.tensor([cuda_graphs[i] * (self.config.num_experts_per_tok - self.prefill_topk)], dtype=torch.int32, device=self.out_device) for i in range(len(cuda_graphs))]
            self.weights_p2_prefill = [torch.empty((cuda_graphs[i] * (self.config.num_experts_per_tok - self.prefill_topk)), dtype=torch.float32, device=self.out_device) for i in range(len(cuda_graphs))]
            self.expert_ids_p2_prefill = [torch.empty((cuda_graphs[i] * (self.config.num_experts_per_tok - self.prefill_topk)), dtype=torch.int32, device=self.out_device) for i in range(len(cuda_graphs))]
            self.sorted_token_reverse_prefill = [torch.empty((cuda_graphs[i] * (self.config.num_experts_per_tok - self.prefill_topk)), dtype=torch.int32, device=self.out_device) for i in range(len(cuda_graphs))]
        
            self.sorted_token_reverse_prefill_2 = [torch.empty((cuda_graphs[i] * (self.prefill_topk)), dtype=torch.int32, device=self.out_device) for i in range(len(cuda_graphs))]
            self.weights_p2_log = [torch.empty((cuda_graphs[i] * (self.config.num_experts_per_tok)), dtype=torch.float32, device=self.out_device) for i in range(len(cuda_graphs))]
            self.weights_p2_log_prefill = [torch.empty((cuda_graphs[i] * (self.config.num_experts_per_tok)), dtype=torch.float32, device=self.out_device) for i in range(len(cuda_graphs))]
            self.weights_p2_prefill_2 = [torch.empty((cuda_graphs[i] * (self.prefill_topk)), dtype=torch.float32, device=self.out_device) for i in range(len(cuda_graphs))]
            self.expert_ids_p2_prefill_2 = [torch.empty((cuda_graphs[i] * (self.prefill_topk)), dtype=torch.int32, device=self.out_device) for i in range(len(cuda_graphs))]
        else:
            self.sorted_token_ids_p1 = torch.tensor([0] * self.flex_topk, dtype=torch.int32, device=self.out_device)
            self.num_tokens_post_padded_p1 = torch.tensor([self.flex_topk], dtype=torch.int32, device=self.out_device)
            self.expert_ids_p1 = torch.empty((self.flex_topk), dtype=torch.int32, device=self.out_device)

            self.sorted_token_ids_p2 = torch.arange((self.config.num_experts_per_tok - self.flex_topk),dtype = torch.int32, device=self.out_device)
            #self.sorted_token_ids_p2 = torch.tensor([0] * (self.config.num_experts_per_tok - self.flex_topk), dtype=torch.int32, device=self.out_device)
            self.num_tokens_post_padded_p2 = torch.tensor([self.config.num_experts_per_tok - self.flex_topk], dtype=torch.int32, device=self.out_device)
            
            self.weights_p2 = torch.empty((self.config.num_experts_per_tok - self.flex_topk), dtype=torch.float32, device=self.out_device)
            self.expert_ids_p2 = torch.empty((self.config.num_experts_per_tok - self.flex_topk), dtype=torch.int32, device=self.out_device)
            self.sorted_token_reverse = torch.empty((self.config.num_experts_per_tok - self.flex_topk), dtype=torch.int32, device=self.out_device)

        # p3: [0,1*topk,2*topk,3*topk,...,(num_experts_per_tok-1)*topk]

        if KExpertsHybrid.input_tensor_cpu is None:
            if isinstance(cuda_graphs, list):
                # dynamic_topk
                KExpertsHybrid.expert_ids_cpu = [torch.zeros((cuda_graphs[i] * self.config.num_experts_per_tok), device="cpu", dtype=torch.long, pin_memory=True) for i in range(len(cuda_graphs))]
                KExpertsHybrid.weights_cpu = [torch.zeros((cuda_graphs[i] * self.config.num_experts_per_tok), device="cpu", dtype=torch.float32, pin_memory=True) for i in range(len(cuda_graphs))]

                KExpertsHybrid.input_tensor_cpu = [torch.zeros((cuda_graphs[i],self.config.hidden_size),device = "cpu", dtype=torch.bfloat16,pin_memory=True) for i in range(len(cuda_graphs))]
                KExpertsHybrid.w12_projs_cpu    = [torch.zeros((cuda_graphs[i]*self.max_topk,self.config.moe_intermediate_size * 2), dtype=torch.bfloat16,device="cpu", pin_memory=True) for i in range(len(cuda_graphs))] # invalid when using concurrent
                #KExpertsHybrid.expert_ids_cpu = [torch.zeros((cuda_graphs[i], self.flex_topk), device="cpu", dtype=torch.long, pin_memory=True) for i in range(len(cuda_graphs))]
                #KExpertsHybrid.weights_cpu = [torch.zeros((cuda_graphs[i], self.flex_topk), device="cpu", dtype=torch.float32, pin_memory=True) for i in range(len(cuda_graphs))]
                
                KExpertsHybrid.w12_projs_prefill_cpu = [torch.zeros((cuda_graphs[i]*self.prefill_topk,self.config.moe_intermediate_size * 2), dtype=torch.float32,device="cpu", pin_memory=True) for i in range(len(cuda_graphs))]
                KExpertsHybrid.expert_ids_prefill_cpu = [torch.zeros((cuda_graphs[i], self.prefill_topk), device="cpu", dtype=torch.long, pin_memory=True) for i in range(len(cuda_graphs))]
                KExpertsHybrid.weights_prefill_cpu = [torch.zeros((cuda_graphs[i], self.prefill_topk), device="cpu", dtype=torch.float32, pin_memory=True) for i in range(len(cuda_graphs))]
                

                # dynamic_prefill_topk
                KExpertsHybrid.w12_projs_prefill_cpu = [torch.zeros((cuda_graphs[i]*self.max_topk,self.config.moe_intermediate_size * 2), dtype=torch.float32,device="cpu", pin_memory=True) for i in range(len(cuda_graphs))]
                KExpertsHybrid.expert_ids_prefill_cpu = [torch.zeros((cuda_graphs[i], self.max_topk), device="cpu", dtype=torch.long, pin_memory=True) for i in range(len(cuda_graphs))]
                KExpertsHybrid.weights_prefill_cpu = [torch.zeros((cuda_graphs[i], self.max_topk), device="cpu", dtype=torch.float32, pin_memory=True) for i in range(len(cuda_graphs))]


                if self.expert_deferral_enabled:
                    KExpertsHybrid.output_deferred_cpu = [torch.zeros((cuda_graphs[i], self.config.hidden_size), device="cpu", pin_memory=True, dtype=torch.bfloat16) for i in range(len(cuda_graphs))]
                KExpertsHybrid.output_cpu = [torch.zeros((cuda_graphs[i], self.config.hidden_size), device="cpu", pin_memory=True, dtype=torch.bfloat16) for i in range(len(cuda_graphs))]
                KExpertsHybrid.bsz_tensor_cpu = [torch.zeros((1), device="cpu", dtype=torch.int32, pin_memory=True) for i in range(len(cuda_graphs))]
            else:
                KExpertsHybrid.input_tensor_cpu = torch.zeros((self.config.hidden_size), device="cpu", dtype=torch.bfloat16,pin_memory=True)
                KExpertsHybrid.w12_projs_cpu    = torch.zeros((self.flex_topk,self.config.moe_intermediate_size * 2), dtype=torch.bfloat16,device="cpu", pin_memory=True)
                KExpertsHybrid.expert_ids_cpu = torch.zeros((self.flex_topk), device="cpu", dtype=torch.long, pin_memory=True)
                KExpertsHybrid.weights_cpu = torch.zeros((self.flex_topk), device="cpu", dtype=torch.float32, pin_memory=True)

                KExpertsHybrid.w12_projs_prefill_cpu = torch.zeros((self.prefill_topk,self.config.moe_intermediate_size * 2), dtype=torch.float32,device="cpu", pin_memory=True)
                KExpertsHybrid.expert_ids_prefill_cpu = torch.zeros((self.prefill_topk), device="cpu", dtype=torch.long, pin_memory=True)
                KExpertsHybrid.weights_prefill_cpu = torch.zeros((self.prefill_topk), device="cpu", dtype=torch.float32, pin_memory=True)
                if self.expert_deferral_enabled:
                    KExpertsHybrid.w12_projs_deferred_cpu = torch.zeros((self.prefill_topk_phase2,self.config.moe_intermediate_size * 2), dtype=torch.float32,device="cpu", pin_memory=True)
                    KExpertsHybrid.expert_ids_deferred_cpu = torch.zeros((self.prefill_topk_phase2), device="cpu", dtype=torch.long, pin_memory=True)
                    KExpertsHybrid.weights_deferred_cpu = torch.zeros((self.prefill_topk_phase2), device="cpu", dtype=torch.float32, pin_memory=True)
                    KExpertsHybrid.output_deferred_cpu = torch.zeros((self.config.hidden_size), device="cpu", pin_memory=True, dtype=torch.bfloat16)

                KExpertsHybrid.output_cpu = torch.zeros((self.config.hidden_size), device="cpu", pin_memory=True, dtype=torch.bfloat16)
                KExpertsHybrid.bsz_tensor_cpu = torch.zeros((1), device="cpu", dtype=torch.int32, pin_memory=True)
        
    def force_static_r(self, cuda_graph_idx: int) -> None:
        """Pin the current stage's split-kernel inputs to the configured r."""
        if cuda_graph_idx in [0, 1, 2, 3]:
            self.flex_decode_topk.copy_(self.static_r_value)
            self.flex_decode_idx.copy_(self.static_r_value)
        else:
            self.flex_prefill_topk.copy_(self.static_r_value)
            self.flex_prefill_idx.copy_(self.static_r_value)

    def submit_for_one_decode(self, input_tensor, expert_ids, weights, bsz_tensor = None, cuda_graph_idx=0):
        if bsz_tensor is None:
            bsz_tensor = torch.ones(1, device=input_tensor.device, dtype = torch.int32)
        if cuda_graph_idx != -1 and cuda_graph_idx in [0,1,2,3]:

            # cuda_graph_idx == 0 valid
            max_topk_cpu = self.max_topk
            max_topk_gpu = self.config.num_experts_per_tok
            # Both modes use the configured tensor. Dynamic V3 updates it at
            # runtime; static deployment pins it before graph capture.
            flex_topk = self.flex_decode_topk
            self.flex_decode_topk_cpu.copy_(self.flex_decode_topk,non_blocking=True)
            tokens = cuda_graphs[cuda_graph_idx]
            sorted_weights, sorted_indices = weights.sort(dim=-1, descending=True)

            weights = sorted_weights
            expert_ids = expert_ids.gather(dim=-1, index=sorted_indices)
            input_tensor = input_tensor.reshape(-1, self.config.hidden_size)
            weights = weights.reshape(-1,self.config.num_experts_per_tok).to(torch.float32)
            expert_ids = expert_ids.reshape(-1,self.config.num_experts_per_tok)
            self.weights_p2[cuda_graph_idx][:cuda_graphs[cuda_graph_idx] * max_topk_gpu].copy_(weights[:,:].reshape(-1))

            self.weights_p2_log[cuda_graph_idx].copy_(weights[:,:].reshape(-1))
            # Both split kernels consume the full sorted expert-id buffer.
            self.expert_ids_p1[cuda_graph_idx][:].copy_(expert_ids[:,:].reshape(-1))
            self.expert_ids_p2[cuda_graph_idx][:].copy_(expert_ids[:,:].reshape(-1))
            KExpertsHybrid.input_tensor_cpu[cuda_graph_idx].copy_(input_tensor, non_blocking=True)
            # The CPU backend reads at most max_topk_cpu entries per token.
            KExpertsHybrid.expert_ids_cpu[cuda_graph_idx][:max_topk_cpu*tokens].copy_(expert_ids[:,:max_topk_cpu].reshape(-1), non_blocking=True)
            KExpertsHybrid.weights_cpu[cuda_graph_idx][:max_topk_cpu*tokens].copy_(weights[:,:max_topk_cpu].reshape(-1), non_blocking=True)
            KExpertsHybrid.bsz_tensor_cpu[cuda_graph_idx].copy_(bsz_tensor, non_blocking=True)
            #[bs*topk,self.config.moe_intermediate_size * 2]
            gpu_topk_projs = KTransformersOps.moe_gemm(
                input_tensor, self.w13,
                self.sorted_token_ids_p1_all[cuda_graph_idx],
                self.expert_ids_p1[cuda_graph_idx],
                self.num_tokens_post_padded_p1_all[cuda_graph_idx],
                self.up_type_gpu,self.config.moe_intermediate_size * 2,
                self.flex_decode_idx, self.flex_decode_topk, tokens * self.flex_decode_idx * (self.flex_decode_idx - 1) // 2,
                self.expert_start, self.flex_decode_topk,
                self.config.num_experts_per_tok, max_topk_cpu*tokens
            )
            
            t_shape = (tokens * max_topk_cpu, self.config.moe_intermediate_size * 2)
            KExpertsHybrid.w12_projs_cpu[cuda_graph_idx][:t_shape[0],:].copy_(gpu_topk_projs.reshape(t_shape),non_blocking=True)
            if not hasattr(self, "_nvtx_decode_pending"):
                self._nvtx_decode_pending = {}
            torch.cuda.nvtx.range_push(f"moe.forward_flex_many_v1[idx={cuda_graph_idx}]")
            try:
                cpu_task = self.moe.forward_flex_many_v1(
                    1,
                    self.flex_decode_topk_cpu.data_ptr(),
                    max_topk_cpu,
                    KExpertsHybrid.expert_ids_cpu[cuda_graph_idx].data_ptr(),
                    KExpertsHybrid.weights_cpu[cuda_graph_idx].data_ptr(),
                    KExpertsHybrid.input_tensor_cpu[cuda_graph_idx].data_ptr(),
                    KExpertsHybrid.w12_projs_cpu[cuda_graph_idx].data_ptr(),
                    KExpertsHybrid.output_cpu[cuda_graph_idx].data_ptr(),
                    KExpertsHybrid.bsz_tensor_cpu[cuda_graph_idx].data_ptr(),
                )
                self.cpu_infer.submit_with_cuda_stream(
                    torch.cuda.current_stream(self.out_device).cuda_stream,
                    cpu_task,
                )
                # Keep NVTX range open until sync_for_one_decode() confirms completion.
                self._nvtx_decode_pending[cuda_graph_idx] = True
            except Exception:
                torch.cuda.nvtx.range_pop()
                raise
            
            # Execute the remaining experts on the GPU.
            
            p2_projs = KTransformersOps.moe_gemm(
                input_tensor,self.w13,
                self.sorted_token_ids_p2_all[cuda_graph_idx],
                self.expert_ids_p1[cuda_graph_idx],
                self.num_tokens_post_padded_p2_all[cuda_graph_idx],
                self.up_type_gpu, self.config.moe_intermediate_size * 2,
                self.flex_decode_idx, self.config.num_experts_per_tok - flex_topk, tokens * (self.config.num_experts_per_tok * self.flex_decode_idx - self.flex_decode_idx * (self.flex_decode_idx - 1) // 2),
                flex_topk, self.expert_end, 
                max_topk_gpu, max_topk_gpu * tokens
            )
            # [max_topk_gpu * tokens, self.config.moe_intermediate_size * 2]
            KExpertsHybrid.intermediate_output[self.out_device][cuda_graph_idx] = self.act(KExpertsHybrid.intermediate_output[self.out_device][cuda_graph_idx],p2_projs)
            # [bs*(8-topk), self.config.hidden_size]
            # TODO: 
            out = KTransformersOps.moe_gemm_w2(
                KExpertsHybrid.intermediate_output[self.out_device][cuda_graph_idx][:cuda_graphs[cuda_graph_idx] * max_topk_gpu], self.w2,
                self.sorted_token_ids_p2_all[cuda_graph_idx],
                self.expert_ids_p2[cuda_graph_idx], 
                self.num_tokens_post_padded_p2_all[cuda_graph_idx],
                self.down_type_gpu, self.config.hidden_size,
                1, 
                self.flex_decode_idx,flex_topk, tokens * (self.config.num_experts_per_tok * self.flex_decode_idx - self.flex_decode_idx * (self.flex_decode_idx - 1) // 2),
                max_topk_gpu, max_topk_gpu * tokens
            )
            # out: [:tokens*(self.config.num_experts_per_tok - flex_topk) * self.config.hidden_size] valid
            out = out.reshape(tokens * max_topk_gpu, self.config.hidden_size)
            # dynamic_topk
            KTransformersOps.moe_weight_sum_v1(
                out, self.weights_p2[cuda_graph_idx],
                KExpertsHybrid.out_hidden_states[self.out_device][cuda_graph_idx],
                flex_topk, max_topk_gpu, tokens
            )
            
        
        elif cuda_graph_idx != -1 and True:
            
            max_topk_cpu = self.max_topk
            max_topk_gpu = self.config.num_experts_per_tok
            tokens = expert_ids.shape[0]    
            sorted_weights, sorted_indices = weights.sort(dim=-1, descending=True)
            
            weights = sorted_weights
            expert_ids = expert_ids.gather(dim=-1, index=sorted_indices)
            input_tensor = input_tensor.reshape(-1, self.config.hidden_size)
            weights = weights.reshape(-1,self.config.num_experts_per_tok).to(torch.float32)
            expert_ids = expert_ids.reshape(-1,self.config.num_experts_per_tok)

            self.flex_prefill_topk_cpu.copy_(self.flex_prefill_topk,non_blocking=True)
            KExpertsHybrid.expert_ids_prefill_cpu[cuda_graph_idx].copy_(expert_ids[:,:max_topk_cpu], non_blocking=True)
            KExpertsHybrid.weights_prefill_cpu[cuda_graph_idx].copy_(weights[:,:max_topk_cpu], non_blocking=True)
            KExpertsHybrid.bsz_tensor_cpu[cuda_graph_idx].copy_(bsz_tensor, non_blocking=True)
            KExpertsHybrid.input_tensor_cpu[cuda_graph_idx].copy_(input_tensor, non_blocking=True)


            expert_ids_tops = [expert_ids[:,:sub_graphs[i]].reshape(-1).contiguous() for i in range(len(sub_graphs))]

            BLOCK_SIZE = 1
            sorted_token_ids_p1, expert_ids_p1, num_tokens_post_padded_p1 = [],[],[]
            for i in range(len(sub_graphs)):
                s,e,n = moe_align_block_size(expert_ids_tops[i],BLOCK_SIZE,self.n_routed_experts)
                sorted_token_ids_p1.append(s)
                expert_ids_p1.append(e)
                num_tokens_post_padded_p1.append(n)
            sorted_token_ids_p1_all = torch.cat(sorted_token_ids_p1,dim=0)
            expert_ids_p1_all       = torch.cat(expert_ids_p1,dim=0)
            num_tokens_post_padded_p1_all = torch.cat(num_tokens_post_padded_p1,dim=0)
            # tokens_dev = tokens*(8*idx-idx*(idx+1) // 2)
            
            gpu_topk_projs = KTransformersOps.moe_gemm_prefill(
                input_tensor, self.w13, 
                sorted_token_ids_p1_all,
                expert_ids_p1_all,
                num_tokens_post_padded_p1_all,
                self.up_type_gpu, self.config.moe_intermediate_size * 2,
                self.flex_prefill_idx, self.flex_prefill_topk, 
                tokens*(self.flex_prefill_idx * (self.flex_prefill_idx - 1) // 2),
                tokens*(self.flex_prefill_idx * (self.flex_prefill_idx - 1) // 2),
                max_topk_gpu, max_topk_cpu * tokens
            )
            gpu_topk_projs = gpu_topk_projs.to(torch.float32).contiguous()
            KExpertsHybrid.w12_projs_prefill_cpu[cuda_graph_idx].copy_(gpu_topk_projs, non_blocking=True)
            
            self.cpu_infer.submit_with_cuda_stream(torch.cuda.current_stream(self.out_device).cuda_stream,
                                                    self.moe.forward_many_flex_v1(tokens,self.flex_prefill_topk_cpu.data_ptr(),max_topk_cpu,
                                                    KExpertsHybrid.expert_ids_prefill_cpu[cuda_graph_idx].data_ptr(),
                                                    KExpertsHybrid.weights_prefill_cpu[cuda_graph_idx].data_ptr(),
                                                    KExpertsHybrid.input_tensor_cpu[cuda_graph_idx].data_ptr(),
                                                    KExpertsHybrid.w12_projs_prefill_cpu[cuda_graph_idx].data_ptr(),
                                                    KExpertsHybrid.output_cpu[cuda_graph_idx].data_ptr(),
                                                    KExpertsHybrid.bsz_tensor_cpu[cuda_graph_idx].data_ptr()))

            # GPU Branch
            expert_ids_tops = [expert_ids[:,sub_graphs[i]:].contiguous() for i in range(len(sub_graphs))]
            sorted_token_ids_p2, expert_ids_p2, num_tokens_post_padded_p2 = [], [], []
            for i in range(len(sub_graphs)):
                s,e,n = moe_align_block_size(expert_ids_tops[i],1,self.n_routed_experts)
                sorted_token_ids_p2.append(s)
                expert_ids_p2.append(e)
                num_tokens_post_padded_p2.append(n)
            sorted_token_ids_p2_all = torch.cat(sorted_token_ids_p2,dim=0)
            expert_ids_p2_all = torch.cat(expert_ids_p2,dim=0)
            num_tokens_post_padded_p2_all = torch.cat(num_tokens_post_padded_p2,dim=0)

            p2_projs = KTransformersOps.moe_gemm_prefill(
                input_tensor, self.w13,
                sorted_token_ids_p2_all,
                expert_ids_p2_all,
                num_tokens_post_padded_p2_all,
                self.up_type_gpu, self.config.moe_intermediate_size * 2,
                self.flex_prefill_idx, self.config.num_experts_per_tok - self.flex_prefill_topk,
                tokens * (self.config.num_experts_per_tok * self.flex_prefill_idx - self.flex_prefill_idx * (self.flex_prefill_idx - 1) // 2),
                tokens * (self.config.num_experts_per_tok * self.flex_prefill_idx - self.flex_prefill_idx * (self.flex_prefill_idx - 1) // 2),
                max_topk_gpu, max_topk_gpu * tokens
            )

            d = p2_projs.shape[1] // 2
            output_shape = (p2_projs.shape[:-1] + (d, ))
            out = torch.empty(output_shape, dtype=p2_projs.dtype, device=p2_projs.device)
            out = self.act(out,p2_projs)
            KExpertsHybrid.intermediate_output[self.out_device][cuda_graph_idx] = out

            sorted_reverse_prefill = []
            expert_ids_p2_reverse = []
            for i in range(len(sub_graphs)):
                s = torch.empty(tokens*(self.config.num_experts_per_tok - sub_graphs[i]),dtype = torch.int32,device=self.out_device)
                s[sorted_token_ids_p2[i]] = torch.arange(len(sorted_token_ids_p2[i]),dtype = torch.int32,device=self.out_device)
                e = expert_ids_p2[i][s]
                sorted_reverse_prefill.append(s)
                expert_ids_p2_reverse.append(e)
            sorted_reverse_prefill_all = torch.cat(sorted_reverse_prefill,dim=0)
            expert_ids_p2_reverse_all  = torch.cat(expert_ids_p2_reverse,dim=0)
            out = KTransformersOps.moe_gemm_w2_prefill(
                out, self.w2,
                sorted_reverse_prefill_all,
                expert_ids_p2_reverse_all,
                num_tokens_post_padded_p2_all,
                self.down_type_gpu, self.config.hidden_size,
                1,
                self.flex_prefill_idx,
                tokens * (self.config.num_experts_per_tok * self.flex_prefill_idx - self.flex_prefill_idx * (self.flex_prefill_idx - 1) // 2),  
                tokens * (self.config.num_experts_per_tok * self.flex_prefill_idx - self.flex_prefill_idx * (self.flex_prefill_idx - 1) // 2),
                max_topk_gpu, max_topk_gpu * tokens
            )
            # [tokens, 8, hidden_size]
            out = out.reshape(tokens, max_topk_gpu, self.config.hidden_size)
            KTransformersOps.moe_weight_sum_v1(
                out,
                weights,
                KExpertsHybrid.out_hidden_states[self.out_device][cuda_graph_idx],
                self.flex_prefill_topk,
                max_topk_gpu,
                tokens
            )
            #KExpertsHybrid.out_hidden_states[self.out_device][cuda_graph_idx]= KTransformersOps.moe_sum(out, KExpertsHybrid.out_hidden_states[self.out_device][cuda_graph_idx])
        
        elif cuda_graph_idx != -1:
            tokens = expert_ids.shape[0]
            sorted_weights, sorted_indices = weights.sort(dim=-1, descending=True)
            weights = sorted_weights
            expert_ids = expert_ids.gather(dim=-1, index=sorted_indices)
            input_tensor = input_tensor.reshape(-1, self.config.hidden_size)
            weights = weights.reshape(-1,self.config.num_experts_per_tok).to(torch.float32)
            expert_ids = expert_ids.reshape(-1,self.config.num_experts_per_tok)
            
            self.weights_p2_prefill[cuda_graph_idx].copy_(weights[:,self.prefill_topk:].reshape(-1))
            self.expert_ids_p2_prefill[cuda_graph_idx].copy_(expert_ids[:,self.prefill_topk:].reshape(-1))
            self.weights_p2_prefill_2[cuda_graph_idx].copy_(weights[:,:self.prefill_topk].reshape(-1))    
            self.expert_ids_p2_prefill_2[cuda_graph_idx].copy_(expert_ids[:,:self.prefill_topk].reshape(-1))
            
            KExpertsHybrid.input_tensor_cpu[cuda_graph_idx].copy_(input_tensor, non_blocking=True)
            KExpertsHybrid.bsz_tensor_cpu[cuda_graph_idx].copy_(bsz_tensor, non_blocking=True)

            BLOCK_SIZE = 1
            if self.expert_deferral_enabled:
                # Submit phase 1 now; phase 2 may overlap the following attention.
                phase1 = self.prefill_topk
                KExpertsHybrid.expert_ids_prefill_cpu[cuda_graph_idx].copy_(expert_ids[:, :phase1], non_blocking=True)
                KExpertsHybrid.weights_prefill_cpu[cuda_graph_idx].copy_(weights[:, :phase1], non_blocking=True)
                sorted_token_ids_p1, expert_ids_p1, num_tokens_post_padded_p1 = moe_align_block_size(expert_ids[:, :phase1].contiguous(), BLOCK_SIZE, self.n_routed_experts)
                gpu_topk_projs_p1 = KTransformersOps.ggml_moe_vec_q8_k128(input_tensor, self.w13, sorted_token_ids_p1,
                                                                          expert_ids_p1, num_tokens_post_padded_p1,
                                                                          self.up_type_gpu, self.config.moe_intermediate_size * 2,
                                                                          phase1, phase1 * tokens)
                gpu_topk_projs_p1 = gpu_topk_projs_p1.to(torch.float32).contiguous()
                KExpertsHybrid.w12_projs_prefill_cpu[cuda_graph_idx][:phase1 * tokens].copy_(gpu_topk_projs_p1, non_blocking=True)
                self.cpu_infer.submit_with_cuda_stream(torch.cuda.current_stream(self.out_device).cuda_stream,
                                                       self.moe.forward_many_flex(tokens, phase1, KExpertsHybrid.expert_ids_prefill_cpu[cuda_graph_idx].data_ptr(),
                                                                                 KExpertsHybrid.weights_prefill_cpu[cuda_graph_idx].data_ptr(),
                                                                                 KExpertsHybrid.input_tensor_cpu[cuda_graph_idx].data_ptr(),
                                                                                 KExpertsHybrid.w12_projs_prefill_cpu[cuda_graph_idx].data_ptr(),
                                                                                 KExpertsHybrid.output_cpu[cuda_graph_idx].data_ptr(),
                                                                                 KExpertsHybrid.bsz_tensor_cpu[cuda_graph_idx].data_ptr()))
                # GPU Branch for num_experts_per_tok - prefill_topk Experts
                sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(self.expert_ids_p2_prefill[cuda_graph_idx], 1, self.n_routed_experts)
                p2_projs = KTransformersOps.ggml_moe_vec_q8_k128(input_tensor, self.w13, sorted_token_ids, 
                                                            expert_ids, num_tokens_post_padded,
                                                            self.up_type_gpu, self.config.moe_intermediate_size * 2,
                                                            self.config.num_experts_per_tok - self.prefill_topk, (self.config.num_experts_per_tok - self.prefill_topk) * tokens)
                d = p2_projs.shape[1] // 2
                output_shape = (p2_projs.shape[:-1] + (d, ))
                out = torch.empty(output_shape, dtype=p2_projs.dtype, device=p2_projs.device)
                out = self.act(out,p2_projs)
                
                self.sorted_token_reverse_prefill[cuda_graph_idx][sorted_token_ids] = torch.arange(len(sorted_token_ids),dtype =torch.int32,device=self.out_device)
                self.expert_ids_p2_prefill[cuda_graph_idx].copy_(expert_ids[self.sorted_token_reverse_prefill[cuda_graph_idx]])
                

                out = KTransformersOps.ggml_moe_vec_q8_k128(out, self.w2, self.sorted_token_reverse_prefill[cuda_graph_idx],
                                                            self.expert_ids_p2_prefill[cuda_graph_idx], num_tokens_post_padded,
                                                            self.down_type_gpu, self.config.hidden_size,
                                                            1, (self.config.num_experts_per_tok - self.prefill_topk) * tokens)
                out = out.reshape(tokens,self.config.num_experts_per_tok - self.prefill_topk,self.config.hidden_size).mul_(
                    weights[:, self.prefill_topk:].view(tokens,self.config.num_experts_per_tok - self.prefill_topk,1))
                #KExpertsHybrid.out_hidden_states[self.out_device][cuda_graph_idx].copy_(out, non_blocking=True)
                KExpertsHybrid.out_hidden_states[self.out_device][cuda_graph_idx]= KTransformersOps.moe_sum(out, KExpertsHybrid.out_hidden_states[self.out_device][cuda_graph_idx])
                # GPU Branch 2 for prefill_topk Experts
                sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(self.expert_ids_p2_prefill_2[cuda_graph_idx], 1, self.n_routed_experts)
                p2_projs = KTransformersOps.ggml_moe_vec_q8_k128(input_tensor, self.w13, sorted_token_ids, 
                                                            expert_ids, num_tokens_post_padded,
                                                            self.up_type_gpu, self.config.moe_intermediate_size * 2,
                                                            self.prefill_topk, self.prefill_topk * tokens)
                d = p2_projs.shape[1] // 2
                output_shape = (p2_projs.shape[:-1] + (d, ))
                out = torch.empty(output_shape, dtype=p2_projs.dtype, device=p2_projs.device)
                out = self.act(out,p2_projs)
                self.sorted_token_reverse_prefill_2[cuda_graph_idx][sorted_token_ids] = torch.arange(len(sorted_token_ids),dtype =torch.int32,device=self.out_device)
                self.expert_ids_p2_prefill_2[cuda_graph_idx].copy_(expert_ids[self.sorted_token_reverse_prefill_2[cuda_graph_idx]])
                
                out = KTransformersOps.ggml_moe_vec_q8_k128(out, self.w2, self.sorted_token_reverse_prefill_2[cuda_graph_idx],
                                                            self.expert_ids_p2_prefill_2[cuda_graph_idx], num_tokens_post_padded,
                                                            self.down_type_gpu, self.config.hidden_size,
                                                            1, self.prefill_topk * tokens)
                out = out.reshape(tokens,self.prefill_topk,self.config.hidden_size).mul_(
                    weights[:,:self.prefill_topk].view(tokens,self.prefill_topk,1))
                #KExpertsHybrid.out_hidden_states[self.out_device][cuda_graph_idx].copy_(out, non_blocking=True)
                KExpertsHybrid.out_hidden_states_2[self.out_device][cuda_graph_idx]= KTransformersOps.moe_sum(out, KExpertsHybrid.out_hidden_states_2[self.out_device][cuda_graph_idx])
                KExpertsHybrid.out_hidden_states[self.out_device][cuda_graph_idx].add_(KExpertsHybrid.out_hidden_states_2[self.out_device][cuda_graph_idx])
            else:
                KExpertsHybrid.expert_ids_prefill_cpu[cuda_graph_idx].copy_(expert_ids[:,:self.prefill_topk], non_blocking=True)
                KExpertsHybrid.weights_prefill_cpu[cuda_graph_idx].copy_(weights[:,:self.prefill_topk], non_blocking=True)
                sorted_token_ids_p1, expert_ids_p1, num_tokens_post_padded_p1 = moe_align_block_size(expert_ids[:,:self.prefill_topk].contiguous(), BLOCK_SIZE, self.n_routed_experts)
                gpu_topk_projs = KTransformersOps.ggml_moe_vec_q8_k128(input_tensor, self.w13, sorted_token_ids_p1,
                                                            expert_ids_p1, num_tokens_post_padded_p1,
                                                            self.up_type_gpu, self.config.moe_intermediate_size * 2,
                                                            self.prefill_topk, self.prefill_topk * tokens)
                gpu_topk_projs = gpu_topk_projs.to(torch.float32).contiguous()
                KExpertsHybrid.w12_projs_prefill_cpu[cuda_graph_idx].copy_(gpu_topk_projs, non_blocking=True)
                self.cpu_infer.submit_with_cuda_stream(torch.cuda.current_stream(self.out_device).cuda_stream,
                                                    self.moe.forward_many_flex(tokens, self.prefill_topk, KExpertsHybrid.expert_ids_prefill_cpu[cuda_graph_idx].data_ptr(),
                                                KExpertsHybrid.weights_prefill_cpu[cuda_graph_idx].data_ptr(),
                                                KExpertsHybrid.input_tensor_cpu[cuda_graph_idx].data_ptr(),
                                                KExpertsHybrid.w12_projs_prefill_cpu[cuda_graph_idx].data_ptr(),
                                                KExpertsHybrid.output_cpu[cuda_graph_idx].data_ptr(),
                                                KExpertsHybrid.bsz_tensor_cpu[cuda_graph_idx].data_ptr()))
                # GPU Branch
                sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(self.expert_ids_p2_prefill[cuda_graph_idx], 1, self.n_routed_experts)
                p2_projs = KTransformersOps.ggml_moe_vec_q8_k128(input_tensor, self.w13, sorted_token_ids, 
                                                            expert_ids, num_tokens_post_padded,
                                                            self.up_type_gpu, self.config.moe_intermediate_size * 2,
                                                            self.config.num_experts_per_tok - self.prefill_topk, (self.config.num_experts_per_tok - self.prefill_topk) * tokens)
                d = p2_projs.shape[1] // 2
                output_shape = (p2_projs.shape[:-1] + (d, ))
                out = torch.empty(output_shape, dtype=p2_projs.dtype, device=p2_projs.device)
                out = self.act(out,p2_projs)
                
                self.sorted_token_reverse_prefill[cuda_graph_idx][sorted_token_ids] = torch.arange(len(sorted_token_ids),dtype =torch.int32,device=self.out_device)
                self.expert_ids_p2_prefill[cuda_graph_idx].copy_(expert_ids[self.sorted_token_reverse_prefill[cuda_graph_idx]])
                

                out = KTransformersOps.ggml_moe_vec_q8_k128(out, self.w2, self.sorted_token_reverse_prefill[cuda_graph_idx],
                                                            self.expert_ids_p2_prefill[cuda_graph_idx], num_tokens_post_padded,
                                                            self.down_type_gpu, self.config.hidden_size,
                                                            1, (self.config.num_experts_per_tok - self.prefill_topk) * tokens)
                out = out.reshape(tokens,self.config.num_experts_per_tok - self.prefill_topk,self.config.hidden_size).mul_(
                    weights[:, self.prefill_topk:].view(tokens,self.config.num_experts_per_tok - self.prefill_topk,1))
                #KExpertsHybrid.out_hidden_states[self.out_device][cuda_graph_idx].copy_(out, non_blocking=True)
                KExpertsHybrid.out_hidden_states[self.out_device][cuda_graph_idx]= KTransformersOps.moe_sum(out, KExpertsHybrid.out_hidden_states[self.out_device][cuda_graph_idx])
        else:
            sorted_weights, sorted_indices = weights.sort(dim=-1, descending=True)
            weights = sorted_weights
            self.weights_p2 = self.weights_p2.copy_(weights[self.flex_topk:])
            expert_ids = expert_ids.gather(dim=-1, index=sorted_indices)
            self.expert_ids_p1 = self.expert_ids_p1.copy_(expert_ids[:self.flex_topk])
            self.expert_ids_p2 = self.expert_ids_p2.copy_(expert_ids[self.flex_topk:])
            #input_tensor_copy = torch.empty_like(input_tensor, memory_format = torch.contiguous_format)
            #input_tensor_copy.copy_(input_tensor)
            KExpertsHybrid.input_tensor_cpu.copy_(input_tensor, non_blocking=True)
            KExpertsHybrid.expert_ids_cpu.copy_(expert_ids[:self.flex_topk], non_blocking=True)
            KExpertsHybrid.weights_cpu.copy_(weights[:self.flex_topk], non_blocking=True)
            # Decode the CPU-selected experts' projections on the GPU.
            gpu_topk_projs = KTransformersOps.ggml_moe_vec_q8_k128(input_tensor.unsqueeze(dim=0), self.w13, self.sorted_token_ids_p1, 
                                                                self.expert_ids_p1, self.num_tokens_post_padded_p1, 
                                                                self.up_type_gpu, self.config.moe_intermediate_size * 2,
                                                                    self.flex_topk, self.flex_topk)    
            KExpertsHybrid.w12_projs_cpu.copy_(gpu_topk_projs,non_blocking=True)
            self.cpu_infer.submit_with_cuda_stream(torch.cuda.current_stream(self.out_device).cuda_stream, 
                                                self.moe.forward_flex(1, self.flex_topk, KExpertsHybrid.expert_ids_cpu.data_ptr(), 
                                                                    KExpertsHybrid.weights_cpu.data_ptr(), KExpertsHybrid.input_tensor_cpu.data_ptr(), 
                                                                    KExpertsHybrid.w12_projs_cpu.data_ptr(), KExpertsHybrid.output_cpu.data_ptr()))
            # Execute the remaining experts on the GPU.
            p2_projs = KTransformersOps.ggml_moe_vec_q8_k128(input_tensor.unsqueeze(dim=0), self.w13, self.sorted_token_ids_p2, 
                                                            self.expert_ids_p2, self.num_tokens_post_padded_p2, 
                                                            self.up_type_gpu, self.config.moe_intermediate_size * 2,
                                                            self.config.num_experts_per_tok - self.flex_topk, self.config.num_experts_per_tok - self.flex_topk)
            
            KExpertsHybrid.intermediate_output[self.out_device] = self.act(KExpertsHybrid.intermediate_output[self.out_device],p2_projs)
            out = KTransformersOps.ggml_moe_vec_q8_k128(KExpertsHybrid.intermediate_output[self.out_device], self.w2, self.sorted_token_ids_p2, 
                                                            self.expert_ids_p2, self.num_tokens_post_padded_p2, 
                                                            self.down_type_gpu, self.config.hidden_size,
                                                            1, (self.config.num_experts_per_tok - self.flex_topk))
            out = out.reshape(1,self.config.num_experts_per_tok - self.flex_topk, self.config.hidden_size)
            KTransformersOps.moe_weight_sum(out,self.weights_p2,KExpertsHybrid.out_hidden_states[self.out_device])
        

    def start_deferred_experts(self, cuda_graph_idx=0):
        """Submit deferred CPU experts so their execution overlaps attention."""
        if not self.expert_deferral_enabled:
            return
        if cuda_graph_idx != -1:
            tokens = KExpertsHybrid.expert_ids_deferred_cpu[cuda_graph_idx].shape[0]
            self.cpu_infer.submit_with_cuda_stream(torch.cuda.current_stream(self.out_device).cuda_stream,
                                                   self.moe.forward_many_flex(tokens, self.prefill_topk_phase2,
                                                                              KExpertsHybrid.expert_ids_deferred_cpu[cuda_graph_idx].data_ptr(),
                                                                              KExpertsHybrid.weights_deferred_cpu[cuda_graph_idx].data_ptr(),
                                                                              KExpertsHybrid.input_tensor_cpu[cuda_graph_idx].data_ptr(),
                                                                              KExpertsHybrid.w12_projs_deferred_cpu[cuda_graph_idx].data_ptr(),
                                                                              KExpertsHybrid.output_deferred_cpu[cuda_graph_idx].data_ptr(),
                                                                              KExpertsHybrid.bsz_tensor_cpu[cuda_graph_idx].data_ptr()))
        else:
            tokens = KExpertsHybrid.expert_ids_deferred_cpu.shape[0]
            self.cpu_infer.submit_with_cuda_stream(torch.cuda.current_stream(self.out_device).cuda_stream,
                                                   self.moe.forward_many_flex(tokens, self.prefill_topk_phase2,
                                                                              KExpertsHybrid.expert_ids_deferred_cpu.data_ptr(),
                                                                              KExpertsHybrid.weights_deferred_cpu.data_ptr(),
                                                                              KExpertsHybrid.input_tensor_cpu.data_ptr(),
                                                                              KExpertsHybrid.w12_projs_deferred_cpu.data_ptr(),
                                                                              KExpertsHybrid.output_deferred_cpu.data_ptr(),
                                                                              KExpertsHybrid.bsz_tensor_cpu.data_ptr()))

    def sync_for_one_decode(self, cuda_graph_idx=0):
        if cuda_graph_idx != -1:
            self.cpu_infer.sync_with_cuda_stream(torch.cuda.current_stream(self.out_device).cuda_stream)
            if getattr(self, "_nvtx_decode_pending", {}).pop(cuda_graph_idx, False):
                torch.cuda.nvtx.range_pop()
            KExpertsHybrid.output_gpu_map[self.out_device][cuda_graph_idx].copy_(KExpertsHybrid.output_cpu[cuda_graph_idx], non_blocking=True)
            if cuda_graph_idx in [0,1,2,3]:
                KTransformersOps.dynamic_add(KExpertsHybrid.output_gpu_map[self.out_device][cuda_graph_idx],KExpertsHybrid.out_hidden_states[self.out_device][cuda_graph_idx], self.flex_decode_topk)
            else:
                KTransformersOps.dynamic_add(KExpertsHybrid.output_gpu_map[self.out_device][cuda_graph_idx],KExpertsHybrid.out_hidden_states[self.out_device][cuda_graph_idx], self.flex_prefill_topk)
            return KExpertsHybrid.output_gpu_map[self.out_device][cuda_graph_idx]
        else:
            self.cpu_infer.sync_with_cuda_stream(torch.cuda.current_stream(self.out_device).cuda_stream)
            KExpertsHybrid.output_gpu_map[self.out_device].copy_(KExpertsHybrid.output_cpu, non_blocking=True)
            return KExpertsHybrid.output_gpu_map[self.out_device].add_(KExpertsHybrid.out_hidden_states[self.out_device])

    def get_gpu_part_only(self, cuda_graph_idx=0):
        """Return the GPU contribution without synchronizing deferred CPU work."""
        if cuda_graph_idx != -1:
            return KExpertsHybrid.out_hidden_states[self.out_device][cuda_graph_idx].clone()
        return KExpertsHybrid.out_hidden_states[self.out_device].clone()

    def sync_only(self, cuda_graph_idx=0):
        if not self.expert_deferral_enabled or cuda_graph_idx in [0,1,2,3]:
            return
        self.cpu_infer.sync_with_cuda_stream(torch.cuda.current_stream(self.out_device).cuda_stream)
        KExpertsHybrid.output_cpu_to_gpu[self.out_device][cuda_graph_idx].copy_(KExpertsHybrid.output_cpu[cuda_graph_idx], non_blocking=True)
        return
    def add_pending_to(self, target_tensor: torch.Tensor, cuda_graph_idx=0):
        """Merge the deferred CPU contribution into the preceding MoE output."""
        if not self.expert_deferral_enabled or cuda_graph_idx in [0,1,2,3]:
            return
        dev = target_tensor.device
        if cuda_graph_idx != -1:
            target_tensor.add_(KExpertsHybrid.output_cpu_to_gpu[self.out_device][cuda_graph_idx].to(dev, non_blocking=True))
            target_tensor.sub_(KExpertsHybrid.out_hidden_states_2[self.out_device][cuda_graph_idx].to(dev, non_blocking=True))


    def forward(self, input_tensor, expert_ids, weights, bsz_tensor=None, cuda_graph_idx=0):
        # input_tensor: [tokens, hidden_size]
        # expert_ids: [tokens, num_experts_per_tok]
        # weights:    [tokens, num_experts_per_tok]
        if bsz_tensor is None:
            bsz_tensor = torch.tensor([input_tensor.size(0)], device=input_tensor.device, dtype=torch.int32)
        if torch.cuda.is_current_stream_capturing():
            # R1 use 2, v3 use 3
            tokens = expert_ids.shape[0]
            sorted_weights, sorted_indices = weights.sort(dim=-1, descending=True)
            weights = sorted_weights
            expert_ids = expert_ids.gather(dim=-1, index=sorted_indices)
            if cuda_graph_idx != -1:
                # valid: bs = 1,2,3,4,64,512
                self.expert_ids_p2[cuda_graph_idx].copy_(expert_ids[:,self.prefill_topk:].reshape(-1))
                KExpertsHybrid.input_tensor_cpu[cuda_graph_idx].copy_(input_tensor, non_blocking=True)
                KExpertsHybrid.expert_ids_prefill_cpu[cuda_graph_idx].copy_(expert_ids[:,:self.prefill_topk], non_blocking=True)
                KExpertsHybrid.weights_prefill_cpu[cuda_graph_idx].copy_(weights[:,:self.prefill_topk], non_blocking=True)
                KExpertsHybrid.bsz_tensor_cpu[cuda_graph_idx].copy_(bsz_tensor, non_blocking=True)

                BLOCK_SIZE = 1
                sorted_token_ids_p1, expert_ids_p1, num_tokens_post_padded_p1 = moe_align_block_size(expert_ids[:,:self.prefill_topk].contiguous(), BLOCK_SIZE, self.n_routed_experts)
                gpu_topk_projs = KTransformersOps.ggml_moe_vec_q8_k128(input_tensor, self.w13, sorted_token_ids_p1, 
                                                               expert_ids_p1, num_tokens_post_padded_p1, 
                                                               self.up_type_gpu, self.config.moe_intermediate_size * 2,
                                                                self.prefill_topk, self.prefill_topk * tokens)
                gpu_topk_projs = gpu_topk_projs.to(torch.float32).contiguous()
                KExpertsHybrid.w12_projs_prefill_cpu[cuda_graph_idx].copy_(gpu_topk_projs,non_blocking=True)
                self.cpu_infer.submit_with_cuda_stream(torch.cuda.current_stream(self.out_device).cuda_stream,
                                                    self.moe.forward_many_flex(tokens, self.prefill_topk, KExpertsHybrid.expert_ids_prefill_cpu[cuda_graph_idx].data_ptr(),
                                                    KExpertsHybrid.weights_prefill_cpu[cuda_graph_idx].data_ptr(),
                                                    KExpertsHybrid.input_tensor_cpu[cuda_graph_idx].data_ptr(),
                                                    KExpertsHybrid.w12_projs_prefill_cpu[cuda_graph_idx].data_ptr(),
                                                    KExpertsHybrid.output_cpu[cuda_graph_idx].data_ptr(),
                                                    KExpertsHybrid.bsz_tensor_cpu[cuda_graph_idx].data_ptr()))

                sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(self.expert_ids_p2[cuda_graph_idx], BLOCK_SIZE, self.n_routed_experts)
                p2_projs = KTransformersOps.ggml_moe_vec_q8_k128(input_tensor, self.w13, sorted_token_ids, 
                                                            expert_ids, num_tokens_post_padded,
                                                            self.up_type_gpu, self.config.moe_intermediate_size * 2,
                                                            self.config.num_experts_per_tok - self.prefill_topk, (self.config.num_experts_per_tok - self.prefill_topk) * tokens)
                d = p2_projs.shape[1] // 2
                output_shape = (p2_projs.shape[:-1] + (d, ))
                out = torch.empty(output_shape, dtype=p2_projs.dtype, device=p2_projs.device)
                out = self.act(out,p2_projs)
                self.sorted_token_reverse[cuda_graph_idx][sorted_token_ids] = torch.arange(len(sorted_token_ids),dtype =torch.int32,device=self.out_device)
                self.expert_ids_p2[cuda_graph_idx].copy_(expert_ids[self.sorted_token_reverse[cuda_graph_idx]])
            
                out = KTransformersOps.ggml_moe_vec_q8_k128(out, self.w2, self.sorted_token_reverse[cuda_graph_idx],
                                                            self.expert_ids_p2[cuda_graph_idx], num_tokens_post_padded,
                                                            self.down_type_gpu, self.config.hidden_size,
                                                            1, (self.config.num_experts_per_tok - self.prefill_topk) * tokens)
                out = out.reshape(tokens,self.config.num_experts_per_tok - self.prefill_topk,self.config.hidden_size).mul_(
                    weights[:, self.prefill_topk:].view(tokens,self.config.num_experts_per_tok - self.prefill_topk,1))
                
                KExpertsHybrid.out_hidden_states[self.out_device][cuda_graph_idx]= KTransformersOps.moe_sum(out, KExpertsHybrid.out_hidden_states[self.out_device][cuda_graph_idx])
                self.cpu_infer.sync_with_cuda_stream(torch.cuda.current_stream(self.out_device).cuda_stream)
                KExpertsHybrid.output_gpu_map[self.out_device][cuda_graph_idx].copy_(KExpertsHybrid.output_cpu[cuda_graph_idx], non_blocking=True)
                return KExpertsHybrid.output_gpu_map[self.out_device][cuda_graph_idx].add_(KExpertsHybrid.out_hidden_states[self.out_device][cuda_graph_idx])
            else:
                # valid
                tokens = expert_ids.shape[0]
                self.expert_ids_p2.copy_(expert_ids[:,self.prefill_topk:].reshape(-1))
                KExpertsHybrid.input_tensor_cpu.copy_(input_tensor, non_blocking=True)
                KExpertsHybrid.expert_ids_cpu.copy_(expert_ids[:,:self.prefill_topk], non_blocking=True)
                KExpertsHybrid.weights_cpu.copy_(weights[:,:self.prefill_topk], non_blocking=True)
                KExpertsHybrid.bsz_tensor_cpu.copy_(bsz_tensor, non_blocking=True)
                self.cpu_infer.submit_with_cuda_stream(torch.cuda.current_stream(self.out_device).cuda_stream,
                                                    self.moe.forward(expert_ids.size(0), self.prefill_topk, KExpertsHybrid.expert_ids_cpu.data_ptr(),
                                                    KExpertsHybrid.weights_cpu.data_ptr(),
                                                    KExpertsHybrid.input_tensor_cpu.data_ptr(),
                                                    KExpertsHybrid.output_cpu.data_ptr(),
                                                    KExpertsHybrid.bsz_tensor_cpu.data_ptr()))
                BLOCK_SIZE = 1
                sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(self.expert_ids_p2, BLOCK_SIZE, self.n_routed_experts)
                p2_projs = KTransformersOps.ggml_moe_vec_q8_k128(input_tensor, self.w13, sorted_token_ids, 
                                                            expert_ids, num_tokens_post_padded,
                                                            self.up_type_gpu, self.config.moe_intermediate_size * 2,
                                                            self.config.num_experts_per_tok - self.prefill_topk, (self.config.num_experts_per_tok - self.prefill_topk) * tokens)
                d = p2_projs.shape[1] // 2
                output_shape = (p2_projs.shape[:-1] + (d, ))
                out = torch.empty(output_shape, dtype=p2_projs.dtype, device=p2_projs.device)
                out = self.act(out,p2_projs)
                self.sorted_token_reverse[sorted_token_ids] = torch.arange(len(sorted_token_ids),dtype =torch.int32,device=self.out_device)
                self.expert_ids_p2.copy_(expert_ids[self.sorted_token_reverse])
                out = KTransformersOps.ggml_moe_vec_q8_k128(out, self.w2, self.sorted_token_reverse,
                                                            self.expert_ids_p2, num_tokens_post_padded,
                                                            self.down_type_gpu, self.config.hidden_size,
                                                            1, (self.config.num_experts_per_tok - self.prefill_topk) * tokens)
                out = out.reshape(tokens,self.config.num_experts_per_tok - self.prefill_topk,self.config.hidden_size).mul_(
                    weights[:, self.prefill_topk:].view(tokens,self.config.num_experts_per_tok - self.prefill_topk,1))
                
                KExpertsHybrid.out_hidden_states[self.out_device]= KTransformersOps.moe_sum(out, KExpertsHybrid.out_hidden_states[self.out_device])
                self.cpu_infer.sync_with_cuda_stream(torch.cuda.current_stream(self.out_device).cuda_stream)
                KExpertsHybrid.output_gpu_map[self.out_device].copy_(KExpertsHybrid.output_cpu, non_blocking=True)
                return KExpertsHybrid.output_gpu_map[self.out_device].add_(KExpertsHybrid.out_hidden_states[self.out_device])

        else:
            # Eager prefill path used by AE correctness/performance checks.
            # Keep it consistent with the configured static/dynamic split.
            prefill_topk = int(self.flex_prefill_topk.item())
            tokens = input_tensor.shape[0]
            input_tensor = input_tensor.contiguous() # [tokens , hidden_size]
            output = torch.empty_like(input_tensor).contiguous()
            sorted_weights, sorted_indices = weights[:,:].sort(dim=-1, descending=True)
            weights = sorted_weights.contiguous().to(torch.float32)
            expert_ids = expert_ids[:,:].gather(dim=-1, index=sorted_indices).contiguous()
            input_tensor_slice = input_tensor.cpu()
            output1 = torch.empty_like(input_tensor_slice).contiguous()
            expert_ids_slice = expert_ids[:,:prefill_topk].contiguous().cpu()
            weights_slice = weights[:,:prefill_topk].contiguous().cpu()
            bsz_tensor = bsz_tensor.contiguous().cpu()
            
            BLOCK_SIZE = 1
            sorted_token_ids_p1, expert_ids_p1, num_tokens_post_padded_p1 = moe_align_block_size(expert_ids[:tokens,:prefill_topk].contiguous(), BLOCK_SIZE, self.n_routed_experts)
            gpu_topk_projs = KTransformersOps.ggml_moe_vec_q8_k128(input_tensor, self.w13, sorted_token_ids_p1, 
                                                            expert_ids_p1, num_tokens_post_padded_p1, 
                                                            self.up_type_gpu, self.config.moe_intermediate_size * 2,
                                                            prefill_topk, prefill_topk * tokens)
            gpu_topk_projs = gpu_topk_projs.to(torch.float32).contiguous().cpu()
            self.cpu_infer.submit(self.moe.forward(tokens, expert_ids_slice.size(1), expert_ids_slice.data_ptr(), weights_slice.data_ptr(), input_tensor_slice.data_ptr(), output1.data_ptr(),bsz_tensor.data_ptr()))
            
            BLOCK_SIZE = 1
            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(expert_ids[:,prefill_topk:].contiguous(), BLOCK_SIZE, self.n_routed_experts)
            output = torch.empty_like(input_tensor).to(device=object.__getattribute__(self, "out_device"))
            p2_projs = KTransformersOps.ggml_moe_vec_q8_k128(input_tensor, self.w13, sorted_token_ids, 
                                                          expert_ids, num_tokens_post_padded,
                                                          self.up_type_gpu, self.config.moe_intermediate_size * 2,
                                                          self.config.num_experts_per_tok - prefill_topk, (self.config.num_experts_per_tok - prefill_topk) * tokens)
            d = p2_projs.shape[1] // 2
            output_shape = (p2_projs.shape[:-1] + (d, ))
            out = torch.empty(output_shape, dtype=p2_projs.dtype, device=p2_projs.device)
            out = self.act(out,p2_projs)
            sorted_token_reverse = torch.empty_like(sorted_token_ids,dtype=torch.int32,device=p2_projs.device)
            sorted_token_reverse[sorted_token_ids] = torch.arange(len(sorted_token_ids),dtype =torch.int32,device=self.out_device)
            expert_ids_p2 = expert_ids[sorted_token_reverse]
            out = KTransformersOps.ggml_moe_vec_q8_k128(out, self.w2, sorted_token_reverse,
                                                          expert_ids_p2, num_tokens_post_padded,
                                                          self.down_type_gpu, self.config.hidden_size,
                                                          1, (self.config.num_experts_per_tok - prefill_topk) * tokens)
            out = out.reshape(tokens,self.config.num_experts_per_tok - prefill_topk,self.config.hidden_size).mul_(
                weights[:tokens, prefill_topk:].view(tokens,self.config.num_experts_per_tok - prefill_topk,1))
            output = KTransformersOps.moe_sum(out, output)
            self.cpu_infer.sync()
            return output.add_(output1.to(device=object.__getattribute__(self, "out_device")))

    def unload(self):
        return

    def load_weights(self, override_key: str | None = None, device: str = "cpu"):
        # TODO: support Bias
        res = {}
        if override_key is not None:
            keys = override_key
        else:
            keys = [self.key]

        gate = None
        up = None
        down = None
        gate_type = None
        up_type = None
        down_type = None
        for key in keys:
            if self.gguf_loader.safetensor_loader is not None:
                # Safetensor-backed weights are materialized as NumPy arrays.
                gate = self.gguf_loader.safetensor_loader.load_tensor(key + ".ffn_gate_exps.weight").numpy()
                up = self.gguf_loader.safetensor_loader.load_tensor(key + ".ffn_up_exps.weight").numpy()
                down = self.gguf_loader.safetensor_loader.load_tensor(key + ".ffn_down_exps.weight").numpy()
                gate_type = self.gguf_loader.safetensor_loader.load_tensor(key + ".ffn_gate_exps.ggml_type").item()
                up_type = self.gguf_loader.safetensor_loader.load_tensor(key + ".ffn_up_exps.ggml_type").item()
                down_type = self.gguf_loader.safetensor_loader.load_tensor(key + ".ffn_down_exps.ggml_type").item()
            
            elif key + ".ffn_gate_exps.weight" in self.gguf_loader.tensor_info:
                gate = self.gguf_loader.get_mmap_tensor(key + ".ffn_gate_exps.weight")
                up = self.gguf_loader.get_mmap_tensor(key + ".ffn_up_exps.weight")
                down = self.gguf_loader.get_mmap_tensor(key + ".ffn_down_exps.weight")
                gate_type = self.gguf_loader.tensor_info[key + ".ffn_gate_exps.weight"]["ggml_type"]
                up_type = self.gguf_loader.tensor_info[key + ".ffn_up_exps.weight"]["ggml_type"]
                down_type = self.gguf_loader.tensor_info[key + ".ffn_down_exps.weight"]["ggml_type"]
            elif key + ".ffn_down.0.weight" in self.gguf_loader.tensor_info:
                # for supporting  Mixtral-8x7B-Instuct  
                gate = []
                up = []
                down = []
                for i in range(8):
                    gate_it = self.gguf_loader.get_mmap_tensor(f"{key}.ffn_gate.{i}.weight")
                    up_it = self.gguf_loader.get_mmap_tensor(f"{key}.ffn_up.{i}.weight")
                    down_it = self.gguf_loader.get_mmap_tensor(f"{key}.ffn_down.{i}.weight")
                    gate.append(gate_it)
                    up.append(up_it)
                    down.append(down_it)
                gate = np.stack(gate)
                up = np.stack(up)
                down = np.stack(down)
                gate_type = self.gguf_loader.tensor_info[key + ".ffn_gate.0.weight"]["ggml_type"]
                up_type = self.gguf_loader.tensor_info[key + ".ffn_up.0.weight"]["ggml_type"]
                down_type = self.gguf_loader.tensor_info[key + ".ffn_down.0.weight"]["ggml_type"]
            else:
                raise ValueError(f"Experts {key} not found in gguf_loader")
            res = {key:{"gate": gate, "up": up, "down": down, "gate_type": gate_type, "up_type": up_type, "down_type": down_type}}
        return res
    
class KExpertsMarlin(KExpertsBase):
    expert_num: int
    loaded_experts_idx: list[int]
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        n_routed_experts: int,
        orig_module: nn.Module = None,
        device: str = "cuda",
        **kwargs
    ):
        super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)
        self.expert_num = n_routed_experts
        self.loaded_experts_idx = []
        self.act_fn = ACT2FN[config.hidden_act]
        assert device.lower() != "cpu", "Marlin experts can only be loaded on GPU"
        self.device = device
        self.elements_per_tensor = config.moe_intermediate_size * config.hidden_size

        # create empty marlin experts according to the number of experts per token
        # up
        self.up_projs = [KLinearMarlin(key+ "." + "ffn_up_exps", gguf_loader, config, device=device) for i in range(self.expert_num)]
        # gate
        self.gate_projs = [KLinearMarlin(key+ "." + "ffn_gate_exps", gguf_loader, config, device=device) for i in range(self.expert_num)]
        # down
        self.down_projs = [KLinearMarlin(key+ "." + "ffn_down_exps", gguf_loader, config, device=device) for i in range(self.expert_num)]

    def load(self, w: dict | nn.Parameter | tuple | None = None, device: str | None = None, warmup: bool = False):
        if device is None: device = self.device
        assert device.lower() != "cpu", "Marlin experts can only be loaded on GPU"
        if w is None:
            w = self.load_weights()
            load_by_experts = True

        if load_by_experts:
            if isinstance(w, dict):
                self.gate = w["gate"]
                self.up = (w["up"])
                self.down = (w["down"])
                for i in tqdm(range(self.expert_num), desc=f"Dequanting and quanting for KExpertsMarlin {self.key}"):
                    up_weights = self.gguf_loader.load_expert_tensor(self.key + ".ffn_up_exps.weight", self.up, i, self.elements_per_tensor, device=self.device)
                    gate_weights = self.gguf_loader.load_expert_tensor(self.key + ".ffn_gate_exps.weight", self.gate, i, self.elements_per_tensor, device=self.device)
                    down_weights = self.gguf_loader.load_expert_tensor(self.key + ".ffn_down_exps.weight", self.down, i, self.elements_per_tensor, device=self.device)
                    
                    self.up_projs[i].load(nn.Parameter(up_weights), device=device)
                    self.gate_projs[i].load(nn.Parameter(gate_weights), device=device)
                    self.down_projs[i].load(nn.Parameter(down_weights), device=device)
                    self.loaded_experts_idx.append(i)
        else:
            if isinstance(w, dict):
                self.gate = w["gate"]
                self.up = (w["up"])
                self.down = (w["down"])
                for i in range(self.expert_num):
                    self.up_projs[i].load(nn.Parameter(self.up[i,...]), device=device)
                    self.gate_projs[i].load(nn.Parameter(self.gate[i,...]), device=device)
                    self.down_projs[i].load(nn.Parameter(self.down[i,...]), device=device)
                    self.loaded_experts_idx.append(i)
        return 

    def unload(self):
        for i in self.loaded_experts_idx:
            self.up_projs[i].unload()
            self.gate_projs[i].unload()
            self.down_projs[i].unload()
        self.loaded_experts_idx = []

    def load_weights(self, override_key: str | None = None):
        res = {}
        if override_key is not None:
            keys = override_key
        else:
            keys = [self.key]

        gate = None
        up = None
        down = None

        for key in keys:
            if key + ".ffn_gate_exps.weight" in self.gguf_loader.tensor_info:
                gate = self.gguf_loader.get_mmap_tensor(key + ".ffn_gate_exps.weight")
                up = self.gguf_loader.get_mmap_tensor(key + ".ffn_up_exps.weight")
                down = self.gguf_loader.get_mmap_tensor(key + ".ffn_down_exps.weight")
            res = {"gate": gate, "up": up, "down": down}
        return res

    def forward(self, hidden_states_cpu: torch.Tensor, selected_experts_cpu: torch.Tensor, routing_weights_cpu: torch.Tensor) -> torch.Tensor:
        org_dtype = hidden_states_cpu.dtype
        org_device = hidden_states_cpu.device
        hidden_states_cpu = hidden_states_cpu.to(self.device)
        selected_experts_cpu = selected_experts_cpu.to(self.device)
        routing_weights_cpu = routing_weights_cpu.to(self.device).to(org_dtype)
        
        batch_sequence_length, hidden_dim = hidden_states_cpu.size()

        final_hidden_states = torch.zeros(
            (batch_sequence_length, hidden_dim), dtype=hidden_states_cpu.dtype, device=hidden_states_cpu.device
        )
        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts_cpu, num_classes=self.expert_num).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.expert_num):
            if not expert_mask[expert_idx].any():
                continue
            idx, top_x = torch.where(expert_mask[expert_idx])
            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states_cpu[None, top_x].reshape(-1, hidden_dim)
            G = self.gate_projs[expert_idx].forward(current_state)
            A = self.act_fn(G)
            U = self.up_projs[expert_idx].forward(current_state)
            H = A * U  # Element-wise multiplication
            current_hidden_states = self.down_projs[expert_idx].forward(H) * routing_weights_cpu[top_x, idx, None]
            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states)
        
        return final_hidden_states.to(dtype=org_dtype, device=org_device)
    
# untested, CUDA OOM
class KExpertsTorch(KExpertsBase):
    expert_num: int
    loaded_experts_idx: list[int]
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        n_routed_experts: int,
        orig_module: nn.Module = None,
        device: str = "cpu",
        **kwargs
    ):
        super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)
        self.expert_num = n_routed_experts
        # self.loaded_experts_idx = []
        self.act_fn = ACT2FN[config.hidden_act]
        self.device = device
        self.elements_per_tensor = config.moe_intermediate_size * config.hidden_size
        self.gate = [None for _ in range(self.expert_num)]
        self.up = [None for _ in range(self.expert_num)]
        self.down = [None for _ in range(self.expert_num)]
        self.dtype = torch.get_default_dtype()

    def load(self, w: dict | nn.Parameter | tuple | None = None, device: str | None = None, warmup: bool = False):
        if device is None: device = self.device
        if w is None:
            w = self.load_weights()
            load_by_experts = True

        if load_by_experts:
            if isinstance(w, dict):
                for i in tqdm(range(self.expert_num), desc=f"Dequanting for KExpertsTorch {self.key}"):
                    up_weights = self.gguf_loader.load_expert_tensor(self.key + ".ffn_up_exps.weight", w["up"], i, self.elements_per_tensor, device=self.device)
                    gate_weights = self.gguf_loader.load_expert_tensor(self.key + ".ffn_gate_exps.weight", w["gate"], i, self.elements_per_tensor, device=self.device)
                    down_weights = self.gguf_loader.load_expert_tensor(self.key + ".ffn_down_exps.weight", w["down"], i, self.elements_per_tensor, device=self.device)
                    
                    self.up[i] = up_weights
                    self.gate[i] = gate_weights
                    self.down[i] = down_weights
        else:
            if isinstance(w, dict):
                for i in range(self.expert_num):
                    self.gate[i] = w["gate"][i, ...].to(device=device, dtype=self.dtype)
                    self.up[i] = w["up"][i, ...].to(device=device, dtype=self.dtype)
                    self.down[i] = w["down"][i, ...].to(device=device, dtype=self.dtype)
        
        self.up = torch.stack(self.up, dim=0)
        self.gate = torch.stack(self.gate, dim=0)
        self.down = torch.stack(self.down, dim=0)
        return 

    def unload(self):
        if self.gate is not None:
            self.gate = None
            self.up = None
            self.down = None

    def load_weights(self, override_key: str | None = None):
        res = {}
        if override_key is not None:
            keys = override_key
        else:
            keys = [self.key]

        gate = None
        up = None
        down = None

        for key in keys:
            if key + ".ffn_gate_exps.weight" in self.gguf_loader.tensor_info:
                gate = self.gguf_loader.get_mmap_tensor(key + ".ffn_gate_exps.weight")
                up = self.gguf_loader.get_mmap_tensor(key + ".ffn_up_exps.weight")
                down = self.gguf_loader.get_mmap_tensor(key + ".ffn_down_exps.weight")
            res = {"gate": gate, "up": up, "down": down}
        return res

    def forward(self, hidden_states_cpu: torch.Tensor, selected_experts_cpu: torch.Tensor, routing_weights_cpu: torch.Tensor) -> torch.Tensor:

        org_device = hidden_states_cpu.device
        hidden_states_cpu = hidden_states_cpu.to(self.device)
        selected_experts_cpu = selected_experts_cpu.to(self.device)
        routing_weights_cpu = routing_weights_cpu.to(self.device)
        
        batch_sequence_length, hidden_dim = hidden_states_cpu.size()

        final_hidden_states = torch.zeros(
            (batch_sequence_length, hidden_dim), dtype=self.gate.dtype, device=hidden_states_cpu.device
        )
        org_dtype = hidden_states_cpu.dtype
        hidden_states_cpu = hidden_states_cpu.to(self.gate.dtype)
        routing_weights_cpu = routing_weights_cpu.to(self.gate.dtype)
        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts_cpu, num_classes=self.expert_num).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.expert_num):
            idx, top_x = torch.where(expert_mask[expert_idx])
            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states_cpu[None, top_x].reshape(-1, hidden_dim)
            G = current_state @ self.gate[expert_idx,...].T
            A = self.act_fn(G)
            U = current_state @ self.up[expert_idx,...].T
            H = A * U  # Element-wise multiplication
            current_hidden_states = H @ self.down[expert_idx,...].T * routing_weights_cpu[top_x, idx, None]
            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states)


        return final_hidden_states.to(dtype=org_dtype, device=org_device)

EXPERTS_MAP = {
    "KExpertsCPU": KExpertsCPU,
    "KExpertsTorch": KExpertsTorch,
    "KExpertsMarlin": KExpertsMarlin,
    "KExpertsHybrid": KExpertsHybrid
}

class KTransformersExperts(BaseInjectedModule, KExpertsBase):
    def __init__(self,
                 key: str,
                 gguf_loader: GGUFLoader,
                 config: PretrainedConfig,
                 orig_module: nn.Module,
                #  device: str = "cuda",
                 prefill_device:str = "cuda",
                 prefill_op: str | None = "KExpertsTorch",
                 generate_device: str = "cpu",
                 generate_op: str | None = "KExpertsCPU",
                 **kwargs):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, prefill_device, generate_device, **kwargs)
        KExpertsBase.__init__(self, key, gguf_loader, config, orig_module, generate_device, **kwargs)
        if generate_op is not None:
            self.generate_experts = EXPERTS_MAP[generate_op](key, gguf_loader, config, len(orig_module), device=generate_device, **kwargs)
        else:
            self.generate_experts = None
        if prefill_op is not None:
            self.prefill_experts = EXPERTS_MAP[prefill_op](key, gguf_loader, config, len(orig_module), device=prefill_device, **kwargs)
        else:
            self.prefill_experts = None
        self.gpu_mlp_type = prefill_op
        self.cpu_mlp_type = generate_op
        self.mode = InferenceState.UNLOAD

    def load(self, w: dict = None,  mode: InferenceState = None, warmup: bool = True):
        # TODO support w as input
        if not mode: mode = InferenceState.GENERATE
        if mode == InferenceState.GENERATE:
            # Generate is the default; prefill is selected explicitly.
            self.prefill_experts.unload()
            self.generate_experts.load(w, warmup=warmup)
            self.device = self.generate_experts.device
            self.mode = mode
        elif mode == InferenceState.PREFILL:
            self.generate_experts.unload()
            self.prefill_experts.load(w, warmup=warmup)
            self.device = self.prefill_experts.device
            self.mode = mode
        elif mode == InferenceState.UNLOAD:
            self.unload()
            self.mode = mode
            self.device = self.generate_experts.device
        else:
            raise ValueError("mode must be either InferenceState.GENERATE, InferenceState.PREFILL or InferenceState.UNLOAD")

    def unload(self):
        if self.generate_experts is not None:
            self.generate_experts.unload()
        if self.prefill_experts is not None:
            self.prefill_experts.unload()
        self.device = self.generate_experts.device

    def forward(self, input_tensor, expert_ids, weights):
        if self.mode == InferenceState.GENERATE:
            assert self.generate_experts is not None, "generate_experts is None"
            return self.generate_experts.forward(input_tensor, expert_ids, weights)
        elif self.mode == InferenceState.PREFILL:
            assert self.prefill_experts is not None, "prefill_experts is None"
            return self.prefill_experts.forward(input_tensor, expert_ids, weights)
        else:
            raise ValueError("load or set_inference_mode before forward")

    def set_inference_mode(self, mode: InferenceState):
        if mode == InferenceState.GENERATE:
            self.load(mode=InferenceState.GENERATE, warmup=False)
        elif mode == InferenceState.PREFILL:
            self.load(mode=InferenceState.PREFILL, warmup=False)
        elif mode == InferenceState.UNLOAD:
            self.unload()
        else:
            raise ValueError("mode must be either InferenceState.GENERATE, InferenceState.PREFILL or InferenceState.UNLOAD")


from ktransformers.models.modeling_deepseek import DeepseekV2MoE
from ktransformers.models.modeling_deepseek_v3 import DeepseekV3MoE
from ktransformers.models.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock
from ktransformers.models.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
from ktransformers.models.modeling_mixtral import MixtralSparseMoeBlock


class KQwen2MoeSparseMoeBlock(BaseInjectedModule, Qwen2MoeSparseMoeBlock):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        orig_shape = hidden_states.shape
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)
        
        if sequence_length == 1 and hasattr(self.experts.generate_experts, "submit_for_one_decode"):
            self.experts.generate_experts.submit_for_one_decode(hidden_states[0], selected_experts[0], routing_weights[0])
            shared_expert_output = self.shared_expert(hidden_states)
            shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
            y = self.experts.generate_experts.sync_for_one_decode().unsqueeze(0)
            y += shared_expert_output
            y.resize_(*orig_shape)
            return y, router_logits
        
        hidden_states_expert = hidden_states.to(self.experts.device)  if isinstance(self.experts, KExpertsBase) else hidden_states.cpu()
        selected_experts_expert = selected_experts.to(self.experts.device) if isinstance(self.experts, KExpertsBase) else selected_experts.cpu()
        routing_weights_expert = routing_weights.to(self.experts.device) if isinstance(self.experts, KExpertsBase) else routing_weights.cpu()

        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = (
            F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
        )

        if isinstance(self.experts, KExpertsBase):
            y = (
                self.moe_kexperts(
                    hidden_states_expert, selected_experts_expert, routing_weights_expert
                )
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
        elif hidden_states_expert.size(0) > 10:
            y = self.moe_infer(
                hidden_states_expert, selected_experts_expert, routing_weights_expert, orig_shape
            ).to(device=hidden_states.device)
        else:
            y = self.moe_infer_simple(
                hidden_states_expert, selected_experts_expert, routing_weights_expert
            ).to(device=hidden_states.device)
        y += shared_expert_output
        y.resize_(*orig_shape)
        return y, router_logits
    
    @torch.no_grad()
    def moe_kexperts(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
        outs = self.experts(x, topk_ids, topk_weight)
        return outs

    @torch.no_grad()
    def moe_infer_simple(self, hidden_states_cpu: torch.Tensor, selected_experts_cpu: torch.Tensor, routing_weights_cpu: torch.Tensor) -> torch.Tensor:
        '''
        hidden_states_cpu: [num_tokens, hidden_size]
        topk_ids, topk_weight: [num_tokens, num_selected_experts]
        '''
        outs = torch.zeros_like(hidden_states_cpu)
        for token_idx in range(selected_experts_cpu.size(0)):
            for expert_idx in range(selected_experts_cpu.size(1)):
                expert = self.experts[selected_experts_cpu[token_idx, expert_idx]]
                outs[token_idx] += expert.forward(hidden_states_cpu[token_idx]) * routing_weights_cpu[token_idx, expert_idx]
        return outs
    
    @torch.no_grad()
    def moe_infer(self, hidden_states_cpu: torch.Tensor, selected_experts_cpu: torch.Tensor, routing_weights_cpu: torch.Tensor, orig_shape: tuple) -> torch.Tensor:
        
        batch_size, sequence_length, hidden_dim = orig_shape

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states_cpu.dtype, device=hidden_states_cpu.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts_cpu, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states_cpu[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer.forward(current_state) * routing_weights_cpu[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states_cpu.dtype))

        return final_hidden_states

class KDeepseekV2MoE(BaseInjectedModule, DeepseekV2MoE):
    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        sequence_length = orig_shape[1]
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        
        if sequence_length == 1 and hasattr(self.experts.generate_experts, "submit_for_one_decode") and torch.cuda.is_current_stream_capturing():
            # The hybrid backend sorts routing weights and expert ids together.
            self.experts.generate_experts.submit_for_one_decode(hidden_states[0], topk_idx[0], topk_weight[0])
            if self.config.n_shared_experts is not None:
                y_ = self.shared_experts(identity).squeeze(0)
            y = self.experts.generate_experts.sync_for_one_decode().unsqueeze(0)
            y += y_
            y.resize_(*orig_shape)
            return y
        if self.config.n_shared_experts is not None:
            y_ = self.shared_experts(identity).squeeze(0)
        if isinstance(self.experts, KExpertsBase):
            # this branch valid
            y = self.moe_kexperts(hidden_states, topk_idx, topk_weight).view(*orig_shape).to(device=hidden_states.device)
        elif hidden_states.size(0) > 10:
            y = (
                self.moe_infer(hidden_states, topk_idx, topk_weight)
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
        else:
            y = (
                self.moe_infer_simple(hidden_states, topk_idx, topk_weight)
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
        if self.config.n_shared_experts is not None:
            y += y_
        return y

    @torch.no_grad()
    def moe_kexperts(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
        outs = self.experts(x, topk_ids, topk_weight)
        return outs

    @torch.no_grad()
    def moe_infer_simple(
        self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor
    ) -> torch.Tensor:
        """
        x: [num_tokens, hidden_size]
        topk_ids, topk_weight: [num_tokens, num_selected_experts]
        """
        outs = torch.zeros_like(x)
        for token_idx in range(topk_ids.size(0)):
            for expert_idx in range(topk_ids.size(1)):
                expert = self.experts[topk_ids[token_idx, expert_idx]]
                outs[token_idx] += (
                    expert.forward(x[token_idx]) * topk_weight[token_idx, expert_idx]
                )
        return outs

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert.forward(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out

class KDeepseekV3MoE(BaseInjectedModule, DeepseekV3MoE):
    
    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        sequence_length = orig_shape[1]
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        # only for generate phase
        if sequence_length == 1 and hasattr(self.experts.generate_experts, "submit_for_one_decode") and torch.cuda.is_current_stream_capturing():
            self.experts.generate_experts.submit_for_one_decode(hidden_states[0], topk_idx[0], topk_weight[0])
            if self.config.n_shared_experts is not None:
                y_ = self.shared_experts(identity).squeeze(0)
            ge = self.experts.generate_experts
            if getattr(ge, "expert_deferral_enabled", False):
                ge.start_deferred_experts(0)
                y = identity + ge.get_gpu_part_only(0).unsqueeze(0)
            else:
                y = ge.sync_for_one_decode().unsqueeze(0)
            y += y_
            y.resize_(*orig_shape)
            return y
        #if sequence_length == 1 and (torch.cuda.is_current_stream_capturing()==False):
        if self.config.n_shared_experts is not None:
            y_ = self.shared_experts(identity).squeeze(0)
            
        if isinstance(self.experts, KExpertsBase):
            y = self.moe_kexperts(hidden_states, topk_idx, topk_weight).view(*orig_shape).to(device=hidden_states.device)
        elif hidden_states.size(0) > 10:
            y = (
                self.moe_infer(hidden_states, topk_idx, topk_weight)
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
        else:
            y = (
                self.moe_infer_simple(hidden_states, topk_idx, topk_weight)
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
        if self.config.n_shared_experts is not None:
            y += y_
        return y



    @torch.no_grad()
    def moe_kexperts(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
        outs = self.experts(x, topk_ids, topk_weight)
        return outs

    @torch.no_grad()
    def moe_infer_simple(
        self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor
    ) -> torch.Tensor:
        """
        x: [num_tokens, hidden_size]
        topk_ids, topk_weight: [num_tokens, num_selected_experts]
        """
        outs = torch.zeros_like(x)
        for token_idx in range(topk_ids.size(0)):
            for expert_idx in range(topk_ids.size(1)):
                expert = self.experts[topk_ids[token_idx, expert_idx]]
                outs[token_idx] += (
                    expert.forward(x[token_idx]) * topk_weight[token_idx, expert_idx]
                )
        return outs

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert.forward(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out

class KMistralSparseMoEBlock(BaseInjectedModule, MixtralSparseMoeBlock):
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        orig_shape = hidden_states.shape
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)
        
        if sequence_length == 1 and hasattr(self.experts.generate_experts, "submit_for_one_decode"):
            self.experts.generate_experts.submit_for_one_decode(hidden_states[0], selected_experts[0], routing_weights[0])
            y = self.experts.generate_experts.sync_for_one_decode().unsqueeze(0)
            y.resize_(*orig_shape)
            return y, router_logits
        
        hidden_states_expert = hidden_states.to(self.experts.device)  if isinstance(self.experts, KExpertsBase) else hidden_states_expert.cpu()
        selected_experts_expert = selected_experts.to(self.experts.device) if isinstance(self.experts, KExpertsBase) else selected_experts_expert.cpu()
        routing_weights_expert = routing_weights.to(self.experts.device) if isinstance(self.experts, KExpertsBase) else routing_weights_expert.cpu()

        if isinstance(self.experts, KExpertsBase):
            y = (
                self.moe_kexperts(
                    hidden_states_expert, selected_experts_expert, routing_weights_expert
                )
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
        elif hidden_states_expert.size(0) > 10:
            y = self.moe_infer(
                hidden_states_expert, selected_experts_expert, routing_weights_expert, orig_shape
            ).to(device=hidden_states.device)
        else:
            y = self.moe_infer_simple(
                hidden_states_expert, selected_experts_expert, routing_weights_expert
            ).to(device=hidden_states.device)
            
        y.resize_(*orig_shape)
        return y, router_logits
    
    @torch.no_grad()
    def moe_kexperts(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
        outs = self.experts(x, topk_ids, topk_weight)
        return outs

    @torch.no_grad()
    def moe_infer_simple(self, hidden_states_cpu: torch.Tensor, selected_experts_cpu: torch.Tensor, routing_weights_cpu: torch.Tensor) -> torch.Tensor:
        '''
        hidden_states_cpu: [num_tokens, hidden_size]
        topk_ids, topk_weight: [num_tokens, num_selected_experts]
        '''
        outs = torch.zeros_like(hidden_states_cpu)
        for token_idx in range(selected_experts_cpu.size(0)):
            for expert_idx in range(selected_experts_cpu.size(1)):
                expert = self.experts[selected_experts_cpu[token_idx, expert_idx]]
                outs[token_idx] += expert.forward(hidden_states_cpu[token_idx]) * routing_weights_cpu[token_idx, expert_idx]
        return outs
    
    @torch.no_grad()
    def moe_infer(self, hidden_states_cpu: torch.Tensor, selected_experts_cpu: torch.Tensor, routing_weights_cpu: torch.Tensor, orig_shape: tuple) -> torch.Tensor:
        
        batch_size, sequence_length, hidden_dim = orig_shape

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states_cpu.dtype, device=hidden_states_cpu.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts_cpu, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states_cpu[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer.forward(current_state) * routing_weights_cpu[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states_cpu.dtype))

        return final_hidden_states

class KDeepseekV3MoEV2(BaseInjectedModule, DeepseekV3MoE):
    def forward(self, hidden_states, bsz_tensor, cuda_graph_idx=0):
        identity = hidden_states
        orig_shape = hidden_states.shape
        sequence_length = orig_shape[1]
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        
        
        generate_experts = self.experts.generate_experts
        # Static mode never invokes the dynamic decision kernel.  The in-place
        # device copies are captured into CUDA Graphs and therefore execute on
        # every replay immediately before the split kernels consume topk/idx.
        if isinstance(generate_experts, KExpertsHybrid) and not generate_experts.dynamic_topk:
            generate_experts.force_static_r(cuda_graph_idx)

        # Select the CPU expert count from sorted routing weights and the frozen profile.
        if cuda_graph_idx in [0,1,2,3] and generate_experts.dynamic_topk and generate_experts.threshold_enabled:
            sorted_weight,_ = topk_weight.view(-1, topk_weight.shape[-1]).sort(dim=-1, descending=True)
            KTransformersOps.dynamic_threshold(
                sorted_weight,
                self.experts.generate_experts.decode_alpha,
                self.experts.generate_experts.decode_thre,
                self.experts.generate_experts.flex_decode_topk,
                self.experts.generate_experts.flex_decode_idx,
                bsz_tensor
            )
        elif cuda_graph_idx == 4 and generate_experts.dynamic_topk and generate_experts.threshold_enabled:
            sorted_weight,_ = topk_weight.view(-1, topk_weight.shape[-1]).sort(dim=-1, descending=True)
            KTransformersOps.dynamic_threshold(
                sorted_weight,
                self.experts.generate_experts.prefill_alpha,
                self.experts.generate_experts.prefill_thre,
                self.experts.generate_experts.flex_prefill_topk,
                self.experts.generate_experts.flex_prefill_idx,
                bsz_tensor
            )
        
        if hasattr(generate_experts, "submit_for_one_decode"):
            self.experts.generate_experts.submit_for_one_decode(hidden_states, topk_idx, topk_weight, bsz_tensor, cuda_graph_idx)
            if self.config.n_shared_experts is not None:
                y_ = self.shared_experts(identity, bsz_tensor).squeeze(0)
            ge = self.experts.generate_experts
            if getattr(ge, "expert_deferral_enabled", False) and cuda_graph_idx not in [0,1,2,3]:
                # Return the GPU contribution while deferred CPU work overlaps attention.
                y = ge.get_gpu_part_only(cuda_graph_idx).unsqueeze(0)
            else:
                y = ge.sync_for_one_decode(cuda_graph_idx).unsqueeze(0)
            y += y_
            y.resize_(*orig_shape)
            return y
        if self.config.n_shared_experts is not None:
            y_ = self.shared_experts(identity, bsz_tensor).squeeze(0)
            
        if isinstance(self.experts, KExpertsBase):
            y = self.moe_on_cpuinfer(hidden_states, topk_idx, topk_weight, bsz_tensor, cuda_graph_idx).view(*orig_shape).to(device=hidden_states.device)
        elif hidden_states.size(0) > 10:
            y = (
                self.moe_infer(hidden_states, topk_idx, topk_weight)
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
        else:
            y = (
                self.moe_infer_simple(hidden_states, topk_idx, topk_weight)
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
        if self.config.n_shared_experts is not None:
            y += y_
        return y

    @torch.no_grad()
    def moe_on_cpuinfer(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor, bsz_tensor, cuda_graph_idx=0) -> torch.Tensor:
        outs = torch.empty_like(x)
        outs = self.experts(x, topk_ids, topk_weight, bsz_tensor, cuda_graph_idx)
        return outs

    @torch.no_grad()
    def moe_infer_simple(
        self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor
    ) -> torch.Tensor:
        """
        x: [num_tokens, hidden_size]
        topk_ids, topk_weight: [num_tokens, num_selected_experts]
        """
        outs = torch.zeros_like(x)
        for token_idx in range(topk_ids.size(0)):
            for expert_idx in range(topk_ids.size(1)):
                expert = self.experts[topk_ids[token_idx, expert_idx]]
                outs[token_idx] += (
                    expert.forward(x[token_idx]) * topk_weight[token_idx, expert_idx]
                )
        return outs

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert.forward(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out

class KTransformersExpertsV2(BaseInjectedModule, KExpertsBase):
    def __init__(self,
                 key: str,
                 gguf_loader: GGUFLoader,
                 config: PretrainedConfig,
                 orig_module: nn.Module,
                #  device: str = "cuda",
                 prefill_device:str = "cuda",
                 prefill_op: str | None = "KExpertsTorch",
                 generate_device: str = "cpu",
                 generate_op: str | None = "KExpertsCPU",
                 **kwargs):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, generate_device, **kwargs)
        KExpertsBase.__init__(self, key, gguf_loader, config, orig_module, generate_device, **kwargs)
        if generate_op is not None:
            self.generate_experts = EXPERTS_MAP[generate_op](key, gguf_loader, config, len(orig_module), device=generate_device, **kwargs)
        else:
            self.generate_experts = None
        if prefill_op is not None:
            self.prefill_experts = EXPERTS_MAP[prefill_op](key, gguf_loader, config, len(orig_module), device=prefill_device, **kwargs)
        else:
            self.prefill_experts = None
        self.gpu_mlp_type = prefill_op
        self.cpu_mlp_type = generate_op
        self.mode = InferenceState.UNLOAD

    def load(self, w: dict = None,  mode: InferenceState = None, warmup: bool = True):
        # TODO support w as input
        if not mode: mode = InferenceState.GENERATE
        if mode == InferenceState.GENERATE:
            self.prefill_experts.unload()
            self.generate_experts.load(w, warmup=warmup)
            self.device = self.generate_experts.device
            self.mode = mode
        elif mode == InferenceState.PREFILL:
            self.generate_experts.unload()
            self.prefill_experts.load(w, warmup=warmup)
            self.device = self.prefill_experts.device
            self.mode = mode
        elif mode == InferenceState.UNLOAD:
            self.unload()
            self.mode = mode
            self.device = self.generate_experts.device
        else:
            raise ValueError("mode must be either InferenceState.GENERATE, InferenceState.PREFILL or InferenceState.UNLOAD")

    def unload(self):
        if self.generate_experts is not None:
            self.generate_experts.unload()
        if self.prefill_experts is not None:
            self.prefill_experts.unload()
        self.device = self.generate_experts.device

    def forward(self, input_tensor, expert_ids, weights, bsz_tensor, cuda_graph_idx=0):
        if self.mode == InferenceState.GENERATE:
            assert self.generate_experts is not None, "generate_experts is None"
            return self.generate_experts.forward(input_tensor, expert_ids, weights, bsz_tensor, cuda_graph_idx)
        elif self.mode == InferenceState.PREFILL:
            assert self.prefill_experts is not None, "prefill_experts is None"
            return self.prefill_experts.forward(input_tensor, expert_ids, weights, bsz_tensor, cuda_graph_idx)
        else:
            raise ValueError("load or set_inference_mode before forward")

    def set_inference_mode(self, mode: InferenceState):
        if mode == InferenceState.GENERATE:
            self.load(mode=InferenceState.GENERATE, warmup=False)
        elif mode == InferenceState.PREFILL:
            self.load(mode=InferenceState.PREFILL, warmup=False)
        elif mode == InferenceState.UNLOAD:
            self.unload()
        else:
            raise ValueError("mode must be either InferenceState.GENERATE, InferenceState.PREFILL or InferenceState.UNLOAD")

class KQwen2MoeSparseMoeBlockV2(BaseInjectedModule, Qwen2MoeSparseMoeBlock):
    def forward(self, hidden_states, bsz_tensor, cuda_graph_idx=0):

        orig_shape = hidden_states.shape
        sequence_length = orig_shape[1]

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        router_logits = self.gate(hidden_states, bsz_tensor)        

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        # only for generate phase
        if hasattr(self.experts.generate_experts, "submit_for_one_decode") and torch.cuda.is_current_stream_capturing():
            self.experts.generate_experts.submit_for_one_decode(hidden_states, selected_experts, routing_weights, bsz_tensor, cuda_graph_idx)
            y_ = self.shared_expert(hidden_states, bsz_tensor).squeeze(0)
            y_ = F.sigmoid(self.shared_expert_gate(hidden_states)) * y_    

            y = self.experts.generate_experts.sync_for_one_decode(cuda_graph_idx).unsqueeze(0)
            
            y += y_
            y.resize_(*orig_shape)
            return y

        y_ = self.shared_expert(hidden_states, bsz_tensor).squeeze(0)
        y_ = (
            F.sigmoid(self.shared_expert_gate(hidden_states)) * y_
        )


        if isinstance(self.experts, KExpertsBase):
            y = self.moe_on_cpuinfer(hidden_states, selected_experts, routing_weights, bsz_tensor, cuda_graph_idx).view(*orig_shape).to(device=hidden_states.device)
        elif hidden_states.size(0) > 10:
            y = (
                self.moe_infer(hidden_states, selected_experts, routing_weights)
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
        else:
            y = (
                self.moe_infer_simple(hidden_states, selected_experts, routing_weights)
                .view(*orig_shape)
                .to(device=hidden_states.device)
            ) 
        y += y_
        return y

    @torch.no_grad()
    def moe_on_cpuinfer(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor, bsz_tensor, cuda_graph_idx=0) -> torch.Tensor:
        outs = torch.empty_like(x)
        outs = self.experts(x, topk_ids, topk_weight, bsz_tensor, cuda_graph_idx)
        return outs

    @torch.no_grad()
    def moe_infer_simple(
        self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor
    ) -> torch.Tensor:
        """
        x: [num_tokens, hidden_size]
        topk_ids, topk_weight: [num_tokens, num_selected_experts]
        """
        outs = torch.zeros_like(x)
        for token_idx in range(topk_ids.size(0)):
            for expert_idx in range(topk_ids.size(1)):
                expert = self.experts[topk_ids[token_idx, expert_idx]]
                outs[token_idx] += (
                    expert.forward(x[token_idx]) * topk_weight[token_idx, expert_idx]
                )
        return outs

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert.forward(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out

class KQwen3MoeSparseMoeBlockV2(BaseInjectedModule, Qwen3MoeSparseMoeBlock):
    def forward(self, hidden_states, bsz_tensor, cuda_graph_idx=0):

        orig_shape = hidden_states.shape
        sequence_length = orig_shape[1]

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        router_logits = self.gate(hidden_states, bsz_tensor)        

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        _, min_indices = torch.topk(routing_weights, k=7, dim=-1, largest=False)
        routing_weights = routing_weights.scatter_(-1, min_indices, 0.0)
        if isinstance(self.experts.generate_experts, KExpertsHybrid):
            if cuda_graph_idx in [0,1,2,3] and self.experts.generate_experts.threshold_enabled:
                sorted_weight = routing_weights.to(torch.float32)
                KTransformersOps.dynamic_threshold(
                    sorted_weight,
                    self.experts.generate_experts.decode_alpha,
                    self.experts.generate_experts.decode_thre,
                    self.experts.generate_experts.flex_decode_topk,
                    self.experts.generate_experts.flex_decode_idx,
                    bsz_tensor
                )
            elif cuda_graph_idx == 4 and self.experts.generate_experts.threshold_enabled:
                sorted_weight = routing_weights.to(torch.float32)
                KTransformersOps.dynamic_threshold(
                    sorted_weight,
                    self.experts.generate_experts.prefill_alpha,
                    self.experts.generate_experts.prefill_thre,
                    self.experts.generate_experts.flex_prefill_topk,
                    self.experts.generate_experts.flex_prefill_idx,
                    bsz_tensor
                )
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)
        # only for generate phase
        if hasattr(self.experts.generate_experts, "submit_for_one_decode") and torch.cuda.is_current_stream_capturing():
            self.experts.generate_experts.submit_for_one_decode(hidden_states, selected_experts, routing_weights, bsz_tensor, cuda_graph_idx)

            y = self.experts.generate_experts.sync_for_one_decode(cuda_graph_idx).unsqueeze(0)
            
            y.resize_(*orig_shape)
            return y



        if isinstance(self.experts, KExpertsBase):
            y = self.moe_on_cpuinfer(hidden_states, selected_experts, routing_weights, bsz_tensor, cuda_graph_idx).view(*orig_shape).to(device=hidden_states.device)
        elif hidden_states.size(0) > 10:
            y = (
                self.moe_infer(hidden_states, selected_experts, routing_weights)
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
        else:
            y = (
                self.moe_infer_simple(hidden_states, selected_experts, routing_weights)
                .view(*orig_shape)
                .to(device=hidden_states.device)
            ) 
        return y

    @torch.no_grad()
    def moe_on_cpuinfer(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor, bsz_tensor, cuda_graph_idx=0) -> torch.Tensor:
        outs = torch.empty_like(x)
        outs = self.experts(x, topk_ids, topk_weight, bsz_tensor, cuda_graph_idx)
        return outs

    @torch.no_grad()
    def moe_infer_simple(
        self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor
    ) -> torch.Tensor:
        """
        x: [num_tokens, hidden_size]
        topk_ids, topk_weight: [num_tokens, num_selected_experts]
        """
        outs = torch.zeros_like(x)
        for token_idx in range(topk_ids.size(0)):
            for expert_idx in range(topk_ids.size(1)):
                expert = self.experts[topk_ids[token_idx, expert_idx]]
                outs[token_idx] += (
                    expert.forward(x[token_idx]) * topk_weight[token_idx, expert_idx]
                )
        return outs

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert.forward(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out
