#!/usr/bin/env python
# coding=utf-8
'''
Description  :  
Author       : Azure-Tang, Boxin Zhang, chenht2022
Date         : 2024-07-26 08:48:54
Version      : 1.0.0
LastEditors  : kkk1nak0
LastEditTime : 2024-08-14 08:20:45
Adapted from https://github.com/99991/pygguf/blob/main/gguf.py
Copyright (c) 2023-2024 The ggml authors
Copyright (c) 2024 Thomas Germer
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
'''
# copied from llama.cpp/gguf-py/gguf/constants.py to satisfy dependence of gguf
# GGUF specification
# https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
import struct
import warnings
import numpy as np
import re
import numpy.typing as npt
from typing import Sequence
import os
from enum import IntEnum
import torch
import KTransformersOps
'''
grid_int8 = np.zeros((len(iq1s_grid), 8), dtype=np.int8)
for i, grid_val in enumerate(iq1s_grid):
    grid_bytes = grid_val.tobytes()
    for j in range(min(8, len(grid_bytes))):
        grid_int8[i, j] = int(grid_bytes[j])
# 将grid_int8数组写入到txt文件中
output_file_path = os.path.join(os.path.dirname(__file__), "grid_int8_values.txt")
with open(output_file_path, "w") as f:
    f.write("grid_int8数组的值：\n")
    for row in grid_int8:
        f.write("{" + ", ".join(str(x) for x in row) + "}, \ \n")
print(f"已将grid_int8数组的值写入到文件: {output_file_path}")
'''
from .custom_loader import SafeTensorLoader
import ctypes
import math

class GGMLQuantizationType(IntEnum):
    F32     = 0
    F16     = 1
    Q4_0    = 2
    Q4_1    = 3
    Q5_0    = 6
    Q5_1    = 7
    Q8_0    = 8
    Q8_1    = 9
    Q2_K    = 10
    Q3_K    = 11
    Q4_K    = 12
    Q5_K    = 13
    Q6_K    = 14
    Q8_K    = 15
    IQ2_XXS = 16
    IQ2_XS  = 17
    IQ3_XXS = 18
    IQ1_S   = 19
    IQ4_NL  = 20
    IQ3_S   = 21
    IQ2_S   = 22
    IQ4_XS  = 23
    I8      = 24
    I16     = 25
    I32     = 26
    I64     = 27
    F64     = 28
    IQ1_M   = 29
    BF16    = 30
    IQ1_S_R4= 219
    IQ1_S_R4_RES_Q2_K = 348
    IQ1_M_R4_RES_Q2_K = 350
    IQ1_M_PLUS_Q4_K_R4 = 351

QK_K = 256
GGML_QUANT_SIZES: dict[GGMLQuantizationType, tuple[int, int]] = {
    GGMLQuantizationType.F32:     (1, 4),
    GGMLQuantizationType.F16:     (1, 2),
    GGMLQuantizationType.Q4_0:    (32, 2 + 16),
    GGMLQuantizationType.Q4_1:    (32, 2 + 2 + 16),
    GGMLQuantizationType.Q5_0:    (32, 2 + 4 + 16),
    GGMLQuantizationType.Q5_1:    (32, 2 + 2 + 4 + 16),
    GGMLQuantizationType.Q8_0:    (32, 2 + 32),
    GGMLQuantizationType.Q8_1:    (32, 4 + 4 + 32),
    GGMLQuantizationType.Q2_K:    (256, 2 + 2 + QK_K // 16 + QK_K // 4),
    GGMLQuantizationType.Q3_K:    (256, 2 + QK_K // 4 + QK_K // 8 + 12),
    GGMLQuantizationType.Q4_K:    (256, 2 + 2 + QK_K // 2 + 12),
    GGMLQuantizationType.Q5_K:    (256, 2 + 2 + QK_K // 2 + QK_K // 8 + 12),
    GGMLQuantizationType.Q6_K:    (256, 2 + QK_K // 2 + QK_K // 4 + QK_K // 16),
    GGMLQuantizationType.Q8_K:    (256, 4 + QK_K + QK_K // 8),
    GGMLQuantizationType.IQ2_XXS: (256, 2 + QK_K // 4),
    GGMLQuantizationType.IQ2_XS:  (256, 2 + QK_K // 4 + QK_K // 32),
    GGMLQuantizationType.IQ3_XXS: (256, 2 + QK_K // 4 + QK_K // 8),
    GGMLQuantizationType.IQ1_S:   (256, 2 + QK_K // 8 + QK_K // 16),
    GGMLQuantizationType.IQ4_NL:  (32, 2 + 16),
    GGMLQuantizationType.IQ3_S:   (256, 2 + QK_K // 4 + QK_K // 8 + QK_K // 32 + 4),
    GGMLQuantizationType.IQ2_S:   (256, 2 + QK_K // 4 + QK_K // 16),
    GGMLQuantizationType.IQ4_XS:  (256, 2 + 2 + QK_K // 2 + QK_K // 64),
    GGMLQuantizationType.I8:      (1, 1),
    GGMLQuantizationType.I16:     (1, 2),
    GGMLQuantizationType.I32:     (1, 4),
    GGMLQuantizationType.I64:     (1, 8),
    GGMLQuantizationType.F64:     (1, 8),
    GGMLQuantizationType.IQ1_M:   (256, QK_K // 8 + QK_K // 16  + QK_K // 32),
    GGMLQuantizationType.BF16:    (1, 2),
    GGMLQuantizationType.IQ1_S_R4: (32,16 // 4 + 8 // 4),
    GGMLQuantizationType.IQ1_S_R4_RES_Q2_K: (256 , (16 // 4 + 8 // 4) * 8 + 2 + 2 + QK_K // 16 + QK_K // 4),
    GGMLQuantizationType.IQ1_M_R4_RES_Q2_K: (256 , (16 // 4 + 8 // 4 + 4 // 4) * 8 + 2 + 2 + QK_K // 16 + QK_K // 4),
    GGMLQuantizationType.IQ1_M_PLUS_Q4_K_R4: (256,  (16 // 4 + 8 // 4 + 4 // 4) * 8 + 2 + 2 + QK_K // 2 + 12)
}
# 定义 GGML_ROW_META_SIZE
GGML_ROW_META_SIZES: dict[GGMLQuantizationType, int] = {
    GGMLQuantizationType.F32:     0,
    GGMLQuantizationType.F16:     0,
    GGMLQuantizationType.Q4_0:    0,
    GGMLQuantizationType.Q4_1:    0,
    GGMLQuantizationType.Q5_0:    0,
    GGMLQuantizationType.Q5_1:    0,
    GGMLQuantizationType.Q8_0:    0,
    GGMLQuantizationType.Q8_1:    0,
    GGMLQuantizationType.Q2_K:    0,
    GGMLQuantizationType.Q3_K:    0,
    GGMLQuantizationType.Q4_K:    0,
    GGMLQuantizationType.Q5_K:    0,
    GGMLQuantizationType.Q6_K:    0,
    GGMLQuantizationType.Q8_K:    0,
    GGMLQuantizationType.IQ2_XXS: 0,
    GGMLQuantizationType.IQ2_XS:  0,
    GGMLQuantizationType.IQ3_XXS: 0,
    GGMLQuantizationType.IQ1_S:   0,
    GGMLQuantizationType.IQ4_NL:  0,
    GGMLQuantizationType.IQ3_S:   0,
    GGMLQuantizationType.IQ2_S:   0,
    GGMLQuantizationType.IQ4_XS:  0,
    GGMLQuantizationType.I8:      0,
    GGMLQuantizationType.I16:     0,
    GGMLQuantizationType.I32:     0,
    GGMLQuantizationType.I64:     0,
    GGMLQuantizationType.F64:     0,
    GGMLQuantizationType.IQ1_M:   0,
    GGMLQuantizationType.BF16:    0,
    GGMLQuantizationType.IQ1_S_R4: 2,
    GGMLQuantizationType.IQ1_S_R4_RES_Q2_K: 2,
    GGMLQuantizationType.IQ1_M_R4_RES_Q2_K: 2,
    GGMLQuantizationType.IQ1_M_PLUS_Q4_K_R4: 2,
}
# copied from llama.cpp/gguf-py/gguf/quants.py to avoid dependence of gguf
def quant_shape_to_byte_shape(shape: Sequence[int], quant_type: GGMLQuantizationType):
    block_size, type_size = GGML_QUANT_SIZES[quant_type]
    if shape[-1] % block_size != 0:
        raise ValueError(f"Quantized tensor row size ({shape[-1]}) is not a multiple of {quant_type.name} block size ({block_size})")
    return (*shape[:-1], shape[-1] // block_size * type_size)

GGML_TYPES = {
    "F32": 0,
    "F16": 1,
    "Q4_0": 2,
    "Q5_0": 6,
    "Q8_0": 8,
    "Q2_K": 10,
    "Q3_K": 11,
    "Q4_K": 12,
    "Q5_K": 13,
    "Q6_K": 14,
    "IQ4_XS": 23,
    "BF16": 30,
    "IQ1_S_R4": 219,
    "IQ1_S_R4_RES_Q2_K": 348,
    "IQ1_M_R4_RES_Q2_K": 350,
    "IQ1_M_PLUS_Q4_K_R4": 351,
}

GGML_NAMES = {ggml_type: name for name, ggml_type in GGML_TYPES.items()}

GGML_BLOCK_SIZES = {
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "Q4_0": 2 + 16,
    "Q5_0": 2 + 4 + 16,
    "Q8_0": 2 + 32,
    "Q2_K": 256 // 16 + 256 // 4 + 2 + 2,
    "Q3_K": 256 // 8 + 256 // 4 + 12 + 2,
    "Q4_K": 2 + 2 + 12 + 256 // 2,
    "Q5_K": 2 + 2 + 12 + 256 // 8 + 256 // 2,
    "Q6_K": 256 // 2 + 256 // 4 + 256 // 16 + 2,
    "IQ4_XS": 2 + 2 + 256 // 2 + 256 // 64,
    "FP8": 1,
    "IQ1_S_R4": 16 // 4 + 8 // 4,
    "IQ1_S_R4_RES": (16 // 4 + 8 // 4) * 2,
    "IQ1_M_R4": 16 // 4 + 8 // 4 + 4 // 4,
}

GGML_RES_RATIO = {
    "IQ1_S_R4_RES": 1/2
}

GGML_ROW_META_SIZE = {
    "F32": 0,
    "F16": 0,
    "BF16": 0,
    "Q4_0": 0,
    "Q5_0": 0,
    "Q8_0": 0,
    "Q2_K": 0,
    "Q3_K": 0,
    "Q4_K": 0,
    "Q5_K": 0,
    "Q6_K": 0,
    "IQ4_XS": 0,
    "FP8": 0,
    "IQ1_S_R4": 2,
    "IQ1_M_R4": 2,
    "IQ1_S_R4_RES": 4,
}

GGML_ELEMENTS_PER_BLOCK = {
    "F32": 1,
    "F16": 1,
    "BF16": 1,
    "Q4_0": 32,
    "Q5_0": 32,
    "Q8_0": 32,
    "Q2_K": 256,
    "Q3_K": 256,
    "Q4_K": 256,
    "Q5_K": 256,
    "Q6_K": 256,
    "IQ4_XS": 256,
    "FP8": 1,
    "IQ1_S_R4": 32,
    "IQ1_S_R4_RES": 32,
}

DATA_TYPES = {
    "uint8": 0,
    "int8": 1,
    "uint16": 2,
    "int16": 3,
    "uint32": 4,
    "int32": 5,
    "float32": 6,
    "bool": 7,
    "string": 8,
    "array": 9,
    "uint64": 10,
    "int64": 11,
    "float64": 12,
    "FP8": 13,
    "IQ1_S_R4": 14,
}

class GGUFLoader:
    tensor_info: dict
    gguf_path: str
    tensor_file_map: dict # {tensor_name: tensor_file_path}
    gguf_file_meta: dict
    safetensor_loader: SafeTensorLoader
    def __init__(self, gguf_path: str):
        # Check dir exist
        if not os.path.exists(gguf_path):
            raise FileNotFoundError(f"GGUF dir not found: {gguf_path}")
        if os.path.isfile(gguf_path):
            gguf_path = os.path.dirname(gguf_path)

        self.safetensor_loader = None
        
        self.tensor_info = {}
        self.gguf_path = gguf_path
        self.tensor_file_map = {}
        self.file_data_map = {}
        self.gguf_file_meta = {}
        self.tensor_device_map = {}

        # I know this is ugly, but I don't want to change the original code too much
        # TODO: merge gguf load and other loads.
        #safetensor_loader = SafeTensorLoader(gguf_path)
        #if safetensor_loader.tensor_file_map:
        #    self.safetensor_loader = safetensor_loader
        #    return
        # Walk through all the .gguf files in the directory
        found_gguf = False
        for root, dirs, files in os.walk(gguf_path):
            for file in files:
                # wwx: add q4_k_m
                if file.endswith(".gguf"):
                    found_gguf = True
                    file_name = os.path.join(root, file)
                    with open(file_name, "rb") as f:
                        self.load_gguf(f)
                        if file_name not in self.file_data_map:
                            self.file_data_map[file_name] = np.memmap(file_name, mode = 'r')
        if not found_gguf:
            raise FileNotFoundError(f"Cannot find any .gguf files in: {gguf_path}")
                            
    def load_gguf(self, f):
        f.seek(0)
        assert f.read(4) == b'GGUF'
        values = struct.unpack("<IQQ", f.read(4+8+8))
        version, n_tensors, n_kv = values
        if version != 3:
            warnings.warn(f"Version {version} has never been tested, might not work")

        info = {}
        for _ in range(n_kv):
            name = read_value(f, DATA_TYPES["string"])

            data_type = struct.unpack("<I", f.read(4))[0]

            info[name] = read_value(f, data_type)

        tensor_info = {}
        for _ in range(n_tensors):
            name = read_value(f, DATA_TYPES["string"])
            shape_len = read_value(f, DATA_TYPES["uint32"])
            shape = [read_value(f, DATA_TYPES["uint64"]) for _ in range(shape_len)]
            ggml_type = read_value(f, DATA_TYPES["uint32"])
            bad_offset = read_value(f, DATA_TYPES["uint64"])
            n_elems = int(math.prod(shape))
            block_size, type_size = GGML_QUANT_SIZES[ggml_type]
            row_meta_size = GGML_ROW_META_SIZES[ggml_type]
            n_bytes = n_elems * type_size // block_size
            if shape_len == 1:
                n_bytes = (row_meta_size + shape[0]*type_size // block_size)
            elif shape_len == 2:
                n_bytes = (row_meta_size + shape[0]*type_size // block_size ) * shape[1]
            elif shape_len == 3:
                n_bytes = (row_meta_size + shape[0]* type_size // block_size ) * shape[1] * shape[2]
            np_dims = tuple(reversed(shape))
        
            item_type: npt.DTypeLike
            if ggml_type == GGMLQuantizationType.F16:
                item_count = n_elems
                item_type = np.float16
            elif ggml_type == GGMLQuantizationType.F32:
                item_count = n_elems
                item_type = np.float32
            elif ggml_type == GGMLQuantizationType.F64:
                item_count = n_elems
                item_type = np.float64
            elif ggml_type == GGMLQuantizationType.I8:
                item_count = n_elems
                item_type = np.int8
            elif ggml_type == GGMLQuantizationType.I16:
                item_count = n_elems
                item_type = np.int16
            elif ggml_type == GGMLQuantizationType.I32:
                item_count = n_elems
                item_type = np.int32
            elif ggml_type == GGMLQuantizationType.I64:
                item_count = n_elems
                item_type = np.int64
            else:
                item_count = n_bytes
                item_type = np.uint8
                np_dims = quant_shape_to_byte_shape(np_dims, ggml_type)

            tensor_info[name] = {
                "ggml_type": ggml_type,
                "shape": shape,
                "bad_offset": bad_offset,
                "item_type": item_type,
                "item_count": item_count,
                "np_dims": np_dims
            }

        start = f.tell()
        # Alignment is 32 by default.
        # https://github.com/ggerganov/ggml/blob/e1daebbf9d38d510ba456c4d50b4500a73ac2b14/docs/gguf.md?plain=1#L253
        alignment = info.get("general.alignment", 32)

        # Inconveniently, the offset defined in gguf files is relative to the
        # end of the header and is unaligned.
        # We need to compute the absolute file offset ourselves instead.
        for t in tensor_info.values():
            offset = start + t["bad_offset"]
            offset += (alignment - offset % alignment) % alignment
            t["offset"] = offset
            
        for name in tensor_info:
            self.tensor_file_map[name] = f.name
        self.tensor_info.update(tensor_info)
        self.gguf_file_meta.update(info)
    
    def get_mmap_tensor(self, name):
        t = self.tensor_info[name]
        mmap_data = self.file_data_map[ self.tensor_file_map[name] ]
        offset = t["offset"]
        item_type = t["item_type"]
        item_count = t["item_count"]
        itemsize = int(np.empty([], dtype = item_type).itemsize)
        return mmap_data[offset : offset + itemsize * item_count]
    
    def get_undequanted_tensor_and_ggml_type(self, name):
        t = self.tensor_info[name]
        data = self.get_mmap_tensor(name)
        ggml_type = t["ggml_type"]
        data = torch.from_numpy(data)
        return data, ggml_type

    def load_expert_tensor(self, name, data, expert_id, elements_per_expert, device = "cuda", target_dtype = torch.get_default_dtype())->torch.Tensor:
        t = self.tensor_info[name]
        if device.lower() == "cpu":
            print(f"loading expert {expert_id} of {name} with CPU")
        shape = t["shape"]
        ggml_type = t["ggml_type"]
        if ggml_type not in GGML_NAMES:
            raise NotImplementedError(f"ggml_type {ggml_type} not implemented")
        ggml_name = GGML_NAMES[ggml_type]

        # TODO: experts may fused in quant block, split it
        assert elements_per_expert % GGML_ELEMENTS_PER_BLOCK[ggml_name] == 0, "experts may fused in quant block, please use CPU dequant"

        blocks_per_experts = elements_per_expert // GGML_ELEMENTS_PER_BLOCK[ggml_name]
        block_size = GGML_BLOCK_SIZES[ggml_name]
        offset = expert_id * block_size * blocks_per_experts
        data = data[offset: offset + block_size * blocks_per_experts]
        
        if "cuda" in device.lower():
            values = GGML_DEQUANTIZE_GPU[ggml_name](data, device, target_dtype,shape=shape)
        else:
            values = GGML_DEQUANTIZE[ggml_name](data,shape)
            values = torch.from_numpy(values.copy())
        if ggml_name == "BF16":
            values = values.view(torch.bfloat16)
        values = values.view(shape[-2::-1])

        return values

    def load_gguf_tensor(self, name: str, device:str = "cpu", target_dtype = None)->torch.Tensor:
        t = self.tensor_info[name]
        if device.lower() == "cpu":
            print(f"loading {name} with CPU")
        if target_dtype == None:
            target_dtype = torch.get_default_dtype()
        
        shape = t["shape"]
        ggml_type = t["ggml_type"]
        if ggml_type not in GGML_NAMES:
            raise NotImplementedError(f"ggml_type {ggml_type} not implemented")

        ggml_name = GGML_NAMES[ggml_type]

        data = self.get_mmap_tensor(name)
        block_size = GGML_BLOCK_SIZES[ggml_name]
        elements_per_block = GGML_ELEMENTS_PER_BLOCK[ggml_name]
        num_elements = int(np.prod(shape))
        num_blocks = num_elements // elements_per_block
        
        blocks_per_iter = 16384
        
        if num_blocks > blocks_per_iter and "R4" not in ggml_name: # dequant large tensor
            values = torch.empty((num_blocks, elements_per_block), dtype=target_dtype, device=device)
            for i in range( (num_blocks + blocks_per_iter - 1) // blocks_per_iter):
                blocks_begin = i * blocks_per_iter
                blocks_end = min(blocks_begin + blocks_per_iter, num_blocks)
                if "cuda" in device.lower():
                    cur_values = GGML_DEQUANTIZE_GPU[ggml_name](data[blocks_begin*block_size : blocks_end*block_size], device, target_dtype,shape=shape)
                else:
                    cur_values = GGML_DEQUANTIZE[ggml_name](data[blocks_begin*block_size : blocks_end*block_size],shape)
                    cur_values = torch.from_numpy(cur_values.copy())
                
                cur_values = cur_values.view(-1, elements_per_block)
                if ggml_name == "BF16":
                    cur_values = cur_values.view(torch.bfloat16)
                values[blocks_begin : blocks_end] = cur_values
        else:
            if "cuda" in device.lower():
                values = GGML_DEQUANTIZE_GPU[ggml_name](data, device,shape=shape)
            else:
                values = GGML_DEQUANTIZE[ggml_name](data,shape)
                values = torch.from_numpy(values)
        #if len(shape) == 3:
            # [2048,10944]
            #torch.save(values.cpu().view(shape[::-1]), "2048x10944_iq.pt")
        #    import pdb;pdb.set_trace()  
        if ggml_name == "BF16":
            values = values.view(torch.bfloat16)
        # wwx: not   
        # values = torch.tensor(values)
        values = values.view(shape[::-1])
        if "attn_q" in name and self.gguf_file_meta['general.architecture'] in ["llama"]:
            n_head = self.gguf_file_meta['llama.attention.head_count']
            values = (values.reshape(n_head, values.shape[0] // n_head // 2, 2, *values.shape[1:])
            .swapaxes(1, 2)
            .reshape(values.shape))
        elif "attn_k" in name and self.gguf_file_meta['general.architecture'] in ["llama"]:
            n_head = self.gguf_file_meta['llama.attention.head_count_kv'] 
            values = (values.reshape(n_head, values.shape[0] // n_head // 2, 2, *values.shape[1:])
            .swapaxes(1, 2)
            .reshape(values.shape))
        return values

def read_value(f, data_type):
    if data_type == DATA_TYPES["string"]:
        length = struct.unpack("<Q", f.read(8))[0]
        return f.read(length).decode("utf-8")

    elif data_type == DATA_TYPES["bool"]:
        return bool(struct.unpack("<?", f.read(1))[0])

    elif data_type == DATA_TYPES["uint8"]:
        return struct.unpack("<B", f.read(1))[0]

    elif data_type == DATA_TYPES["int8"]:
        return struct.unpack("<b", f.read(1))[0]

    elif data_type == DATA_TYPES["uint16"]:
        return struct.unpack("<H", f.read(2))[0]

    elif data_type == DATA_TYPES["int16"]:
        return struct.unpack("<h", f.read(2))[0]

    elif data_type == DATA_TYPES["uint32"]:
        return struct.unpack("<I", f.read(4))[0]

    elif data_type == DATA_TYPES["int32"]:
        return struct.unpack("<i", f.read(4))[0]

    elif data_type == DATA_TYPES["float32"]:
        return struct.unpack("<f", f.read(4))[0]

    elif data_type == DATA_TYPES["uint64"]:
        return struct.unpack("<Q", f.read(8))[0]

    elif data_type == DATA_TYPES["int64"]:
        return struct.unpack("<q", f.read(8))[0]

    elif data_type == DATA_TYPES["float64"]:
        return struct.unpack("<d", f.read(8))[0]

    elif data_type == DATA_TYPES["array"]:
        elem_type, count = struct.unpack("<IQ", f.read(4 + 8))
        return [read_value(f, elem_type) for _ in range(count)]

    elif data_type == DATA_TYPES["FP8"]:
        return struct.unpack("<B", f.read(1))[0]

    else:
        raise NotImplementedError(f"Data type {data_type} not implemented")

def dequantize_q2_k(data,shape=None):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L1547
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L74
    block_size = GGML_BLOCK_SIZES["Q2_K"]
    num_blocks = len(data) // block_size

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, block_size // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, block_size)

    dmin = data_f16[:, -1].reshape(num_blocks, 1, 1).astype(np.float32)
    d = data_f16[:, -2].reshape(num_blocks, 1, 1).astype(np.float32)
    scales = data_u8[:, :16].reshape(num_blocks, 16, 1)
    qs = data_u8[:, 16:80].reshape(num_blocks, 64)

    tmp = np.stack([
        qs[:, 00:16] >> 0,
        qs[:, 16:32] >> 0,
        qs[:, 00:16] >> 2,
        qs[:, 16:32] >> 2,
        qs[:, 00:16] >> 4,
        qs[:, 16:32] >> 4,
        qs[:, 00:16] >> 6,
        qs[:, 16:32] >> 6,
        qs[:, 32:48] >> 0,
        qs[:, 48:64] >> 0,
        qs[:, 32:48] >> 2,
        qs[:, 48:64] >> 2,
        qs[:, 32:48] >> 4,
        qs[:, 48:64] >> 4,
        qs[:, 32:48] >> 6,
        qs[:, 48:64] >> 6,
    ], axis=1)

    return d * (scales & 15) * (tmp & 3) - dmin * (scales >> 4)

def dequantize_q2_k_gpu(data, device:str ="cuda", target_dtype = torch.get_default_dtype(),shape = None):
    block_size = GGML_BLOCK_SIZES["Q2_K"]
    ele_per_blk = GGML_ELEMENTS_PER_BLOCK["Q2_K"]
    data = np.frombuffer(data, dtype=data.dtype)
    device = torch.device(device)
    # TODO: this and from_numpy in other functions will cause a warning saying that numpy is not writable, 
    # the best way to fix this is transfer ptr to KTransformersOps instead of Tensor.
    c_pointer = ctypes.addressof(ctypes.cast(data.ctypes.data, ctypes.POINTER(ctypes.c_int8)).contents)
    return KTransformersOps.dequantize_q2_k(c_pointer, data.size, block_size, ele_per_blk, device, target_dtype)

def dequantize_q3_k(data,shape=None):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L1723C32-L1723C42
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L95
    block_size = GGML_BLOCK_SIZES["Q3_K"]
    num_blocks = len(data) // block_size

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, block_size // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, block_size)

    d = data_f16[:, -1].reshape(num_blocks, 1, 1).astype(np.float32)
    bits = np.unpackbits(data_u8[:, :32].reshape(num_blocks, 32, 1), axis=-1, bitorder="little")
    bits = 4 ^ (bits << 2)
    qs = data_u8[:, 32:32 + 64].astype(np.int16)
    a, b, c = data_u8[:, 96: 96 + 12].reshape(num_blocks, 3, 4).transpose(1, 0, 2)
    scales = np.zeros((num_blocks, 4, 4), dtype=np.uint8)
    scales[:, 0] = (a & 15) | ((c & 3) << 4)
    scales[:, 1] = (b & 15) | (((c >> 2) & 3) << 4)
    scales[:, 2] = (a >> 4) | (((c >> 4) & 3) << 4)
    scales[:, 3] = (b >> 4) | ((c >> 6) << 4)
    scales = scales.reshape(num_blocks, 16, 1).astype(np.int16)

    return d * (scales - 32) * np.stack([
        (((qs[:, 00:16] >> 0) & 3) - bits[:, :16, 0]),
        (((qs[:, 16:32] >> 0) & 3) - bits[:, 16:, 0]),
        (((qs[:, 00:16] >> 2) & 3) - bits[:, :16, 1]),
        (((qs[:, 16:32] >> 2) & 3) - bits[:, 16:, 1]),
        (((qs[:, 00:16] >> 4) & 3) - bits[:, :16, 2]),
        (((qs[:, 16:32] >> 4) & 3) - bits[:, 16:, 2]),
        (((qs[:, 00:16] >> 6) & 3) - bits[:, :16, 3]),
        (((qs[:, 16:32] >> 6) & 3) - bits[:, 16:, 3]),
        (((qs[:, 32:48] >> 0) & 3) - bits[:, :16, 4]),
        (((qs[:, 48:64] >> 0) & 3) - bits[:, 16:, 4]),
        (((qs[:, 32:48] >> 2) & 3) - bits[:, :16, 5]),
        (((qs[:, 48:64] >> 2) & 3) - bits[:, 16:, 5]),
        (((qs[:, 32:48] >> 4) & 3) - bits[:, :16, 6]),
        (((qs[:, 48:64] >> 4) & 3) - bits[:, 16:, 6]),
        (((qs[:, 32:48] >> 6) & 3) - bits[:, :16, 7]),
        (((qs[:, 48:64] >> 6) & 3) - bits[:, 16:, 7])
    ], axis=1)

def dequantize_q3_k_gpu(data, device:str ="cuda", target_dtype = torch.get_default_dtype(),shape = None):
    block_size = GGML_BLOCK_SIZES["Q3_K"]
    ele_per_blk = GGML_ELEMENTS_PER_BLOCK["Q3_K"]
    data = np.frombuffer(data, dtype=data.dtype)
    device = torch.device(device)
    # TODO: this and from_numpy in other functions will cause a warning saying that numpy is not writable, 
    # the best way to fix this is transfer ptr to KTransformersOps instead of Tensor.
    c_pointer = ctypes.addressof(ctypes.cast(data.ctypes.data, ctypes.POINTER(ctypes.c_int8)).contents)
    return KTransformersOps.dequantize_q3_k(c_pointer, data.size, block_size, ele_per_blk, device, target_dtype)

def dequantize_q4_k(data,shape=None):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L1929
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L116
    block_size = GGML_BLOCK_SIZES["Q4_K"]
    num_blocks = len(data) // block_size
    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, block_size // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, block_size)
    # Casting to float32 because float16 is very slow on CPU
    scale_factors = data_f16[:, 0].reshape(num_blocks, 1, 1).astype(np.float32)
    scale_offsets = data_f16[:, 1].reshape(num_blocks, 1, 1).astype(np.float32)
    qs1 = data_u8[:, 4:16].reshape(num_blocks, 12, 1)
    qs2 = data_u8[:, 16:].reshape(num_blocks, 4, 32)
    # Dequantize scales and offsets (6 bits and 4 + 2 bits)
    factors = scale_factors * np.concatenate([qs1[:, 0:4] & 0b111111, (qs1[:, 8:] & 15) | ((qs1[:, 0:4] >> 6) << 4)], axis=1)
    offsets = scale_offsets * np.concatenate([qs1[:, 4:8] & 0b111111, (qs1[:, 8:] >> 4) | ((qs1[:, 4:8] >> 6) << 4)], axis=1)
    # Interleave low and high quantized bits
    qs2 = np.stack([qs2 & 0xf, qs2 >> 4], axis=2).reshape(num_blocks, 8, 32)
    # Dequantize final weights using scales and offsets
    return factors * qs2 - offsets

def dequantize_q4_k_gpu(data, device:str ="cuda", target_dtype = torch.get_default_dtype(),shape = None):
    block_size = GGML_BLOCK_SIZES["Q4_K"]
    ele_per_blk = GGML_ELEMENTS_PER_BLOCK["Q4_K"]
    data = np.frombuffer(data, dtype=data.dtype)
    device = torch.device(device)
    # TODO: this and from_numpy in other functions will cause a warning saying that numpy is not writable, 
    # the best way to fix this is transfer ptr to KTransformersOps instead of Tensor.
    c_pointer = ctypes.addressof(ctypes.cast(data.ctypes.data, ctypes.POINTER(ctypes.c_int8)).contents)
    return KTransformersOps.dequantize_q4_k(c_pointer, data.size, block_size, ele_per_blk, device, target_dtype)

def dequantize_q5_k(data,shape=None):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L2129
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L138
    block_size = GGML_BLOCK_SIZES["Q5_K"]
    num_blocks = len(data) // block_size

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, block_size // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, block_size)

    d = data_f16[:, 0].reshape(num_blocks, 1).astype(np.float32)
    dmin = data_f16[:, 1].reshape(num_blocks, 1).astype(np.float32)
    scales = data_u8[:, 4:16].reshape(num_blocks, 12, 1)
    qh = data_u8[:, 16: 16 + 32].reshape(num_blocks, 32, 1)
    qs = data_u8[:, 48: 48 + 128].reshape(num_blocks, 4, 32)

    bits = np.unpackbits(qh, axis=-1, bitorder="little")

    qs_hi_4 = qs >> 4
    qs_lo_4 = qs & 15

    scales_lo_6 = scales[:, :8] & 63
    scales_hi_6 = scales[:, :8] >> 6
    scales_lo_4 = scales[:, 8:] & 15
    scales_hi_4 = scales[:, 8:] >> 4

    m1 = dmin * scales_lo_6[:, 4]
    m2 = dmin * scales_lo_6[:, 5]
    m3 = dmin * scales_lo_6[:, 6]
    m4 = dmin * scales_lo_6[:, 7]
    m5 = dmin * (scales_hi_4[:, 0] | (scales_hi_6[:, 4] << 4))
    m6 = dmin * (scales_hi_4[:, 1] | (scales_hi_6[:, 5] << 4))
    m7 = dmin * (scales_hi_4[:, 2] | (scales_hi_6[:, 6] << 4))
    m8 = dmin * (scales_hi_4[:, 3] | (scales_hi_6[:, 7] << 4))

    d1 = d * scales_lo_6[:, 0]
    d2 = d * scales_lo_6[:, 1]
    d3 = d * scales_lo_6[:, 2]
    d4 = d * scales_lo_6[:, 3]
    d5 = d * (scales_lo_4[:, 0] | (scales_hi_6[:, 0] << 4))
    d6 = d * (scales_lo_4[:, 1] | (scales_hi_6[:, 1] << 4))
    d7 = d * (scales_lo_4[:, 2] | (scales_hi_6[:, 2] << 4))
    d8 = d * (scales_lo_4[:, 3] | (scales_hi_6[:, 3] << 4))

    return np.concatenate([
        d1 * (qs_lo_4[:, 0] + (bits[:, :, 0] << 4)) - m1,
        d2 * (qs_hi_4[:, 0] + (bits[:, :, 1] << 4)) - m2,
        d3 * (qs_lo_4[:, 1] + (bits[:, :, 2] << 4)) - m3,
        d4 * (qs_hi_4[:, 1] + (bits[:, :, 3] << 4)) - m4,
        d5 * (qs_lo_4[:, 2] + (bits[:, :, 4] << 4)) - m5,
        d6 * (qs_hi_4[:, 2] + (bits[:, :, 5] << 4)) - m6,
        d7 * (qs_lo_4[:, 3] + (bits[:, :, 6] << 4)) - m7,
        d8 * (qs_hi_4[:, 3] + (bits[:, :, 7] << 4)) - m8,
    ], axis=1)

def dequantize_q5_k_gpu(data, device:str ="cuda", target_dtype = torch.get_default_dtype(),shape = None):
    block_size = GGML_BLOCK_SIZES["Q5_K"]
    ele_per_blk = GGML_ELEMENTS_PER_BLOCK["Q5_K"]
    data = np.frombuffer(data, dtype=data.dtype)
    device = torch.device(device)
    # TODO: this and from_numpy in other functions will cause a warning saying that numpy is not writable, 
    # the best way to fix this is transfer ptr to KTransformersOps instead of Tensor.
    c_pointer = ctypes.addressof(ctypes.cast(data.ctypes.data, ctypes.POINTER(ctypes.c_int8)).contents)
    return KTransformersOps.dequantize_q5_k(c_pointer, data.size, block_size, ele_per_blk, device, target_dtype)

def dequantize_q6_k(data,shape=None):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L2275
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L152
    block_size = GGML_BLOCK_SIZES["Q6_K"]
    num_blocks = len(data) // block_size

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, block_size // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, block_size)
    data_i8 = np.frombuffer(data, dtype=np.int8).reshape(num_blocks, block_size)

    scales = data_f16[:, -1].reshape(num_blocks, 1).astype(np.float32)
    # TODO use uint8 and cast later?
    ql = data_u8[:, :128].astype(np.int16)
    qh = data_u8[:, 128:192].astype(np.int16)
    sc = data_i8[:, 192:208, np.newaxis].astype(np.float32)

    # Unpack bits, subtraction requires signed data type
    q1 = (ql[:,   :32 ] & 0xF) | (((qh[:, :32] >> 0) & 3) << 4) - 32
    q2 = (ql[:, 32:64 ] & 0xF) | (((qh[:, :32] >> 2) & 3) << 4) - 32
    q3 = (ql[:,   :32 ] >>  4) | (((qh[:, :32] >> 4) & 3) << 4) - 32
    q4 = (ql[:, 32:64 ] >>  4) | (((qh[:, :32] >> 6) & 3) << 4) - 32
    q5 = (ql[:, 64:96 ] & 0xF) | (((qh[:, 32:] >> 0) & 3) << 4) - 32
    q6 = (ql[:, 96:128] & 0xF) | (((qh[:, 32:] >> 2) & 3) << 4) - 32
    q7 = (ql[:, 64:96 ] >>  4) | (((qh[:, 32:] >> 4) & 3) << 4) - 32
    q8 = (ql[:, 96:128] >>  4) | (((qh[:, 32:] >> 6) & 3) << 4) - 32

    # Dequantize
    return scales * np.concatenate([
        sc[:,  0] * q1[:, :16],
        sc[:,  1] * q1[:, 16:],
        sc[:,  2] * q2[:, :16],
        sc[:,  3] * q2[:, 16:],
        sc[:,  4] * q3[:, :16],
        sc[:,  5] * q3[:, 16:],
        sc[:,  6] * q4[:, :16],
        sc[:,  7] * q4[:, 16:],
        sc[:,  8] * q5[:, :16],
        sc[:,  9] * q5[:, 16:],
        sc[:, 10] * q6[:, :16],
        sc[:, 11] * q6[:, 16:],
        sc[:, 12] * q7[:, :16],
        sc[:, 13] * q7[:, 16:],
        sc[:, 14] * q8[:, :16],
        sc[:, 15] * q8[:, 16:],
    ], axis=1) 

# @torch.jit.script
def dequantize_q6_k_gpu(data: np.ndarray, device:str = "cuda", target_dtype = torch.get_default_dtype(),shape = None):
    block_size = GGML_BLOCK_SIZES["Q6_K"]
    ele_per_blk = GGML_ELEMENTS_PER_BLOCK["Q6_K"]
    device = torch.device(device)
    num_blocks = len(data) // block_size
    data = np.frombuffer(data, dtype=data.dtype)
    c_pointer = ctypes.addressof(ctypes.cast(data.ctypes.data, ctypes.POINTER(ctypes.c_int8)).contents)
    return KTransformersOps.dequantize_q6_k(c_pointer, data.size, block_size, ele_per_blk, device, target_dtype)

kvalues_iq4nl = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], dtype=np.int8)

def dequantize_iq4_xs(data,shape=None):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/21d3a308fcb7f31cb9beceaeebad4fb622f3c337/src/ggml-quants.c#L3568
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/21d3a308fcb7f31cb9beceaeebad4fb622f3c337/src/ggml-common.h#L393
    block_size = GGML_BLOCK_SIZES["IQ4_XS"]
    num_blocks = len(data) // block_size

    d = np.frombuffer(data, dtype=np.float16)[0::block_size//2].astype(np.float32).reshape(num_blocks, 1)
    scales_h = np.frombuffer(data, dtype=np.uint16)[1::block_size//2].reshape(num_blocks, 1)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, block_size)[:, 4:]
    scales_l = data_u8[:, :4].reshape(num_blocks, 4)
    qs = data_u8[:, 4:].reshape(num_blocks, block_size - 8)

    ls = np.zeros((num_blocks, QK_K // 32), dtype=np.int8)
    for ib in range(QK_K // 32):
        ls[:, ib] = ((scales_l[:, ib // 2] >> 4 * (ib % 2)) & 0xf) | (((scales_h[:, 0] >> 2 * ib) & 3) << 4)

    dl = (d * (ls - 32)).reshape(num_blocks, -1, 1)

    qs_lo_4 = qs[:, :QK_K // 2].reshape(num_blocks, -1, 16) & 0xf
    qs_hi_4 = qs[:, :QK_K // 2].reshape(num_blocks, -1, 16) >> 4

    y = np.zeros((num_blocks, QK_K), dtype=np.float32)
    for ib in range(QK_K // 32):
        y[:, ib*32:(ib*32)+16] = dl[:, ib] * kvalues_iq4nl[qs_lo_4[:, ib]]
        y[:, (ib*32)+16:(ib*32)+32] = dl[:, ib] * kvalues_iq4nl[qs_hi_4[:, ib]]

    return y.flatten()

iq1s_grid = np.array([
    0xffffffffffffffff, 0xffffffffffffff01, 0xffffffffffff0000, 0xffffffffffff01ff,
    0xffffffffffff0101, 0xffffffffff00ff00, 0xffffffffff000000, 0xffffffffff01ffff,
    0xffffffffff01ff01, 0xffffffffff0101ff, 0xffffffffff010101, 0xffffffff00ff0000,
    0xffffffff0000ff00, 0xffffffff000000ff, 0xffffffff00000001, 0xffffffff00010000,
    0xffffffff01ffffff, 0xffffffff01ffff01, 0xffffffff01ff01ff, 0xffffffff01ff0101,
    0xffffffff01000000, 0xffffffff0101ffff, 0xffffffff0101ff01, 0xffffffff010101ff,
    0xffffffff01010101, 0xffffff00ffff00ff, 0xffffff00ffff0000, 0xffffff00ff00ff00,
    0xffffff00ff0000ff, 0xffffff00ff000001, 0xffffff00ff000100, 0xffffff00ff000101,
    0xffffff00ff010000, 0xffffff0000ffff00, 0xffffff0000ff0001, 0xffffff0000ff0100,
    0xffffff000000ff01, 0xffffff0000000000, 0xffffff0000000101, 0xffffff000001ff00,
    0xffffff00000100ff, 0xffffff0000010001, 0xffffff00000101ff, 0xffffff0001ff0000,
    0xffffff000100ff00, 0xffffff00010000ff, 0xffffff0001000001, 0xffffff0001010000,
    0xffffff01ffffffff, 0xffffff01ffffff01, 0xffffff01ffff01ff, 0xffffff01ffff0101,
    0xffffff01ff000000, 0xffffff01ff01ffff, 0xffffff01ff01ff01, 0xffffff01ff0101ff,
    0xffffff01ff010101, 0xffffff0100ff0000, 0xffffff010000ff00, 0xffffff0100000100,
    0xffffff01000100ff, 0xffffff0100010100, 0xffffff0101ffffff, 0xffffff0101ffff01,
    0xffffff0101ff01ff, 0xffffff0101ff0101, 0xffffff010100ff00, 0xffffff0101000000,
    0xffffff0101000100, 0xffffff010101ffff, 0xffffff010101ff01, 0xffffff01010101ff,
    0xffffff0101010101, 0xffff00ffff00ff00, 0xffff00ffff0000ff, 0xffff00ffff000001,
    0xffff00ffff010000, 0xffff00ff00ffff00, 0xffff00ff00ff0100, 0xffff00ff00000000,
    0xffff00ff00000101, 0xffff00ff000100ff, 0xffff00ff00010000, 0xffff00ff0100ff00,
    0xffff00ff01000100, 0xffff00ff01010000, 0xffff0000ffffff00, 0xffff0000ffff00ff,
    0xffff0000ffff0000, 0xffff0000ffff0001, 0xffff0000ff000000, 0xffff0000ff0001ff,
    0xffff0000ff000101, 0xffff0000ff010100, 0xffff000000ffffff, 0xffff000000ff0000,
    0xffff000000ff0101, 0xffff00000000ffff, 0xffff00000000ff00, 0xffff0000000000ff,
    0xffff000000000000, 0xffff000000000001, 0xffff000000000100, 0xffff00000001ffff,
    0xffff00000001ff01, 0xffff000000010000, 0xffff0000000101ff, 0xffff000000010101,
    0xffff000001ffff00, 0xffff00000100ff00, 0xffff000001000000, 0xffff0000010001ff,
    0xffff000001000101, 0xffff00000101ff00, 0xffff0000010100ff, 0xffff000001010000,
    0xffff000001010001, 0xffff000001010100, 0xffff0001ff0000ff, 0xffff0001ff000100,
    0xffff000100ffff00, 0xffff000100ff00ff, 0xffff00010000ffff, 0xffff00010000ff01,
    0xffff000100000000, 0xffff0001000001ff, 0xffff00010001ffff, 0xffff00010001ff00,
    0xffff000100010001, 0xffff000100010100, 0xffff000101ff0000, 0xffff00010100ff00,
    0xffff0001010000ff, 0xffff000101000100, 0xffff01ffffffffff, 0xffff01ffffffff01,
    0xffff01ffffff01ff, 0xffff01ffffff0101, 0xffff01ffff000000, 0xffff01ffff01ffff,
    0xffff01ffff01ff01, 0xffff01ffff0101ff, 0xffff01ffff010101, 0xffff01ff00ff0000,
    0xffff01ff0000ff00, 0xffff01ff00000001, 0xffff01ff00010000, 0xffff01ff01ffffff,
    0xffff01ff01ffff01, 0xffff01ff01ff01ff, 0xffff01ff01ff0101, 0xffff01ff01000000,
    0xffff01ff0101ffff, 0xffff01ff0101ff01, 0xffff01ff010101ff, 0xffff01ff01010101,
    0xffff0100ffff0000, 0xffff0100ff00ff00, 0xffff0100ff0000ff, 0xffff0100ff000100,
    0xffff0100ff0100ff, 0xffff0100ff010000, 0xffff010000ffff00, 0xffff01000000ffff,
    0xffff01000000ff00, 0xffff010000000000, 0xffff01000001ff00, 0xffff0100000100ff,
    0xffff010000010100, 0xffff01000100ff00, 0xffff0100010000ff, 0xffff010001000001,
    0xffff010001000100, 0xffff010001010000, 0xffff0101ffffffff, 0xffff0101ffffff01,
    0xffff0101ffff01ff, 0xffff0101ffff0101, 0xffff0101ff000000, 0xffff0101ff01ffff,
    0xffff0101ff01ff01, 0xffff0101ff0101ff, 0xffff0101ff010101, 0xffff010100ff0000,
    0xffff01010000ff00, 0xffff010100000100, 0xffff01010001ff00, 0xffff010100010000,
    0xffff010101ffffff, 0xffff010101ffff01, 0xffff010101ff0000, 0xffff010101ff01ff,
    0xffff010101ff0101, 0xffff010101000000, 0xffff01010101ffff, 0xffff01010101ff01,
    0xffff0101010101ff, 0xffff010101010101, 0xff00ffffff00ffff, 0xff00ffffff00ff00,
    0xff00ffffff0000ff, 0xff00ffffff000100, 0xff00ffffff0100ff, 0xff00ffffff010000,
    0xff00ffff00ffff00, 0xff00ffff00ff00ff, 0xff00ffff0000ffff, 0xff00ffff00000000,
    0xff00ffff000001ff, 0xff00ffff0001ff00, 0xff00ffff000100ff, 0xff00ffff00010000,
    0xff00ffff00010100, 0xff00ffff0100ff00, 0xff00ffff010000ff, 0xff00ffff01000001,
    0xff00ffff0101ff00, 0xff00ffff01010000, 0xff00ff00ffffff00, 0xff00ff00ffff00ff,
    0xff00ff00ffff0001, 0xff00ff00ffff0100, 0xff00ff00ff00ffff, 0xff00ff00ff00ff01,
    0xff00ff00ff000000, 0xff00ff00ff0001ff, 0xff00ff00ff01ff00, 0xff00ff00ff0100ff,
    0xff00ff00ff010100, 0xff00ff0000ff0000, 0xff00ff0000ff0101, 0xff00ff000000ffff,
    0xff00ff000000ff00, 0xff00ff000000ff01, 0xff00ff00000000ff, 0xff00ff0000000000,
    0xff00ff0000000001, 0xff00ff0000000100, 0xff00ff000001ffff, 0xff00ff0000010000,
    0xff00ff0001ff00ff, 0xff00ff000100ff01, 0xff00ff0001000000, 0xff00ff000101ff00,
    0xff00ff00010100ff, 0xff00ff01ff00ff00, 0xff00ff01ff0000ff, 0xff00ff01ff000001,
    0xff00ff01ff010000, 0xff00ff0100ffffff, 0xff00ff0100ff0001, 0xff00ff0100ff0100,
    0xff00ff010000ff01, 0xff00ff0100000000, 0xff00ff01000001ff, 0xff00ff0100000101,
    0xff00ff01000100ff, 0xff00ff0100010001, 0xff00ff0101ff0000, 0xff00ff010100ff00,
    0xff00ff01010000ff, 0xff00ff0101000001, 0xff00ff0101010000, 0xff0000ffffffff00,
    0xff0000ffffff0001, 0xff0000ffffff0100, 0xff0000ffff0000ff, 0xff0000ffff000000,
    0xff0000ffff0001ff, 0xff0000ffff000100, 0xff0000ffff01ff00, 0xff0000ffff010001,
    0xff0000ff00ffff00, 0xff0000ff00ff0000, 0xff0000ff00ff0001, 0xff0000ff00ff01ff,
    0xff0000ff00ff0101, 0xff0000ff0000ff00, 0xff0000ff000000ff, 0xff0000ff00000000,
    0xff0000ff00000001, 0xff0000ff00000100, 0xff0000ff0001ff01, 0xff0000ff00010000,
    0xff0000ff000101ff, 0xff0000ff01ff00ff, 0xff0000ff01ff0100, 0xff0000ff0100ffff,
    0xff0000ff010000ff, 0xff0000ff01000000, 0xff0000ff010001ff, 0xff0000ff01000100,
    0xff0000ff01000101, 0xff0000ff0101ff00, 0xff0000ff010100ff, 0xff0000ff01010000,
    0xff0000ff01010100, 0xff000000ffffff01, 0xff000000ffff0000, 0xff000000ffff0101,
    0xff000000ff00ff00, 0xff000000ff0000ff, 0xff000000ff000000, 0xff000000ff000001,
    0xff000000ff000100, 0xff000000ff01ffff, 0xff000000ff01ff01, 0xff000000ff010000,
    0xff000000ff0101ff, 0xff000000ff010101, 0xff00000000ffff00, 0xff00000000ff00ff,
    0xff00000000ff0000, 0xff00000000ff0001, 0xff0000000000ff00, 0xff0000000000ff01,
    0xff000000000000ff, 0xff00000000000000, 0xff00000000000001, 0xff00000000000100,
    0xff00000000000101, 0xff0000000001ff00, 0xff000000000100ff, 0xff00000000010000,
    0xff00000000010001, 0xff00000000010100, 0xff00000001ffffff, 0xff00000001ffff01,
    0xff00000001ff00ff, 0xff00000001ff0000, 0xff00000001ff01ff, 0xff00000001ff0101,
    0xff0000000100ffff, 0xff0000000100ff00, 0xff000000010000ff, 0xff00000001000000,
    0xff00000001000001, 0xff00000001000100, 0xff00000001000101, 0xff0000000101ffff,
    0xff0000000101ff01, 0xff00000001010000, 0xff000001ffffff00, 0xff000001ffff00ff,
    0xff000001ffff0000, 0xff000001ffff0001, 0xff000001ff000000, 0xff000001ff000001,
    0xff000001ff0001ff, 0xff000001ff000101, 0xff000001ff01ff00, 0xff000001ff010001,
    0xff00000100ffffff, 0xff00000100ffff01, 0xff00000100ff00ff, 0xff00000100ff0000,
    0xff00000100ff01ff, 0xff00000100ff0101, 0xff0000010000ff00, 0xff00000100000000,
    0xff00000100000001, 0xff000001000001ff, 0xff00000100000100, 0xff0000010001ff00,
    0xff000001000100ff, 0xff00000100010000, 0xff000001000101ff, 0xff00000100010100,
    0xff00000100010101, 0xff00000101ff0001, 0xff00000101ff0101, 0xff0000010100ff01,
    0xff00000101000000, 0xff000001010100ff, 0xff00000101010100, 0xff0001ffff00ff00,
    0xff0001ffff000001, 0xff0001ffff010000, 0xff0001ff00ffff00, 0xff0001ff00ff00ff,
    0xff0001ff00ff0001, 0xff0001ff00ff0100, 0xff0001ff0000ffff, 0xff0001ff00000000,
    0xff0001ff000001ff, 0xff0001ff00000101, 0xff0001ff0001ffff, 0xff0001ff0001ff00,
    0xff0001ff000100ff, 0xff0001ff00010001, 0xff0001ff00010100, 0xff0001ff01ff0000,
    0xff0001ff0100ff00, 0xff0001ff010000ff, 0xff0001ff01010000, 0xff000100ff00ffff,
    0xff000100ff00ff01, 0xff000100ff000000, 0xff000100ff000101, 0xff000100ff01ff00,
    0xff000100ff010000, 0xff00010000ffff01, 0xff00010000ff00ff, 0xff00010000ff0000,
    0xff00010000ff01ff, 0xff0001000000ff00, 0xff000100000000ff, 0xff00010000000000,
    0xff00010000000001, 0xff00010000000100, 0xff00010000000101, 0xff0001000001ffff,
    0xff00010000010000, 0xff00010000010101, 0xff00010001ff0100, 0xff0001000100ff00,
    0xff0001000100ff01, 0xff00010001000000, 0xff000100010001ff, 0xff0001000101ff00,
    0xff00010001010001, 0xff00010001010100, 0xff000101ffff0100, 0xff000101ff000001,
    0xff000101ff0100ff, 0xff000101ff010001, 0xff00010100ff00ff, 0xff00010100ff0001,
    0xff00010100ff0100, 0xff0001010000ffff, 0xff0001010000ff01, 0xff00010100000000,
    0xff000101000001ff, 0xff0001010001ff00, 0xff00010100010001, 0xff00010100010100,
    0xff00010101ff0000, 0xff0001010100ff00, 0xff00010101000001, 0xff00010101000101,
    0xff01ffffffffffff, 0xff01ffffffffff01, 0xff01ffffffff01ff, 0xff01ffffffff0101,
    0xff01ffffff000000, 0xff01ffffff01ffff, 0xff01ffffff01ff01, 0xff01ffffff010000,
    0xff01ffffff0101ff, 0xff01ffffff010101, 0xff01ffff00ff0000, 0xff01ffff0000ff00,
    0xff01ffff00000100, 0xff01ffff0001ff00, 0xff01ffff00010000, 0xff01ffff01ffffff,
    0xff01ffff01ffff01, 0xff01ffff01ff01ff, 0xff01ffff01ff0101, 0xff01ffff01000000,
    0xff01ffff0101ffff, 0xff01ffff0101ff01, 0xff01ffff01010000, 0xff01ffff010101ff,
    0xff01ffff01010101, 0xff01ff00ffff0000, 0xff01ff00ff00ff00, 0xff01ff00ff0000ff,
    0xff01ff00ff000100, 0xff01ff00ff010000, 0xff01ff0000ffff01, 0xff01ff0000ff00ff,
    0xff01ff0000ff0100, 0xff01ff0000000000, 0xff01ff00000001ff, 0xff01ff0000000101,
    0xff01ff000001ff00, 0xff01ff00000100ff, 0xff01ff0000010000, 0xff01ff0000010001,
    0xff01ff0001ff0000, 0xff01ff000100ffff, 0xff01ff0001000001, 0xff01ff0001000100,
    0xff01ff0001010000, 0xff01ff01ffffff00, 0xff01ff01ffff01ff, 0xff01ff01ffff0101,
    0xff01ff01ff00ff00, 0xff01ff01ff000000, 0xff01ff01ff01ffff, 0xff01ff01ff01ff01,
    0xff01ff01ff0101ff, 0xff01ff01ff010101, 0xff01ff0100ff0000, 0xff01ff010000ff00,
    0xff01ff0100000001, 0xff01ff0100000100, 0xff01ff0100010000, 0xff01ff0101ffff00,
    0xff01ff0101ff01ff, 0xff01ff0101ff0101, 0xff01ff010100ff00, 0xff01ff0101000000,
    0xff01ff010101ffff, 0xff01ff010101ff01, 0xff01ff01010101ff, 0xff01ff0101010101,
    0xff0100ffffff0000, 0xff0100ffff0000ff, 0xff0100ffff000001, 0xff0100ffff000100,
    0xff0100ffff010000, 0xff0100ff00ff00ff, 0xff0100ff00ff0000, 0xff0100ff00ff0001,
    0xff0100ff00ff0100, 0xff0100ff0000ff01, 0xff0100ff00000000, 0xff0100ff000001ff,
    0xff0100ff00000101, 0xff0100ff00010001, 0xff0100ff01ff0000, 0xff0100ff0100ff00,
    0xff0100ff010000ff, 0xff0100ff01000100, 0xff0100ff0101ff00, 0xff0100ff01010000,
    0xff010000ffff0100, 0xff010000ff000000, 0xff010000ff01ff00, 0xff010000ff010100,
    0xff01000000ffffff, 0xff01000000ff0000, 0xff01000000ff01ff, 0xff0100000000ff00,
    0xff010000000000ff, 0xff01000000000000, 0xff01000000000100, 0xff0100000001ff01,
    0xff01000000010000, 0xff010000000101ff, 0xff01000001ff0100, 0xff0100000100ffff,
    0xff010000010000ff, 0xff01000001000000, 0xff010000010001ff, 0xff01000001000101,
    0xff0100000101ff00, 0xff010000010100ff, 0xff01000001010001, 0xff01000001010100,
    0xff010001ffff0000, 0xff010001ff00ffff, 0xff010001ff00ff01, 0xff010001ff000100,
    0xff010001ff010000, 0xff01000100ffff00, 0xff01000100ff0100, 0xff01000100000000,
    0xff0100010001ffff, 0xff0100010001ff00, 0xff01000100010100, 0xff01000101ff00ff,
    0xff01000101ff0001, 0xff0100010100ffff, 0xff01000101000101, 0xff0101ffffffffff,
    0xff0101ffffffff01, 0xff0101ffffff01ff, 0xff0101ffffff0101, 0xff0101ffff000000,
    0xff0101ffff01ffff, 0xff0101ffff01ff01, 0xff0101ffff0101ff, 0xff0101ffff010101,
    0xff0101ff00ff0000, 0xff0101ff0000ff00, 0xff0101ff000000ff, 0xff0101ff00010000,
    0xff0101ff01ffffff, 0xff0101ff01ffff01, 0xff0101ff01ff01ff, 0xff0101ff01ff0101,
    0xff0101ff0101ffff, 0xff0101ff0101ff01, 0xff0101ff010101ff, 0xff0101ff01010101,
    0xff010100ffff0100, 0xff010100ff00ff00, 0xff010100ff0000ff, 0xff010100ff000100,
    0xff010100ff010000, 0xff01010000ff0001, 0xff01010000ff0100, 0xff0101000000ff01,
    0xff01010000000000, 0xff0101000001ff00, 0xff010100000100ff, 0xff01010000010001,
    0xff01010000010100, 0xff01010001ff0000, 0xff0101000100ffff, 0xff01010001000001,
    0xff01010001000100, 0xff010100010100ff, 0xff01010001010000, 0xff010101ffffffff,
    0xff010101ffffff01, 0xff010101ffff01ff, 0xff010101ffff0101, 0xff010101ff01ffff,
    0xff010101ff01ff01, 0xff010101ff0101ff, 0xff010101ff010101, 0xff01010100ff0000,
    0xff0101010000ff00, 0xff01010100000001, 0xff01010100000100, 0xff01010100010000,
    0xff01010101ffffff, 0xff01010101ffff01, 0xff01010101ff01ff, 0xff01010101ff0101,
    0xff01010101000000, 0xff0101010101ffff, 0xff0101010101ff01, 0xff010101010101ff,
    0xff01010101010101, 0x00ffffffffff0000, 0x00ffffffff00ff00, 0x00ffffffff000001,
    0x00ffffffff010000, 0x00ffffff00ff0100, 0x00ffffff0000ff01, 0x00ffffff00000000,
    0x00ffffff000001ff, 0x00ffffff00000101, 0x00ffffff0001ff00, 0x00ffffff000100ff,
    0x00ffffff00010001, 0x00ffffff010000ff, 0x00ffffff01000100, 0x00ffffff0101ff00,
    0x00ffffff01010001, 0x00ffff00ffffffff, 0x00ffff00ffffff00, 0x00ffff00ffff00ff,
    0x00ffff00ffff0001, 0x00ffff00ffff0100, 0x00ffff00ff00ff01, 0x00ffff00ff000000,
    0x00ffff00ff000001, 0x00ffff00ff0001ff, 0x00ffff00ff000101, 0x00ffff00ff01ff00,
    0x00ffff00ff010001, 0x00ffff00ff010100, 0x00ffff0000ff0000, 0x00ffff0000ff01ff,
    0x00ffff0000ff0101, 0x00ffff000000ff00, 0x00ffff00000000ff, 0x00ffff0000000000,
    0x00ffff0000000001, 0x00ffff0000000100, 0x00ffff0000000101, 0x00ffff0000010000,
    0x00ffff00000101ff, 0x00ffff0000010101, 0x00ffff0001ffff00, 0x00ffff0001ff00ff,
    0x00ffff0001ff0001, 0x00ffff000100ffff, 0x00ffff000100ff01, 0x00ffff0001000000,
    0x00ffff000101ffff, 0x00ffff000101ff00, 0x00ffff000101ff01, 0x00ffff01ffff0000,
    0x00ffff01ff00ff00, 0x00ffff01ff0000ff, 0x00ffff01ff000001, 0x00ffff01ff010000,
    0x00ffff0100ffff00, 0x00ffff010000ff01, 0x00ffff0100000000, 0x00ffff0100000101,
    0x00ffff01000100ff, 0x00ffff0100010100, 0x00ffff0101ff0100, 0x00ffff01010000ff,
    0x00ffff0101010000, 0x00ff00ffffffff00, 0x00ff00ffff000000, 0x00ff00ffff000100,
    0x00ff00ffff010100, 0x00ff00ff00ff0000, 0x00ff00ff00ff01ff, 0x00ff00ff00ff0101,
    0x00ff00ff0000ff00, 0x00ff00ff000000ff, 0x00ff00ff00000000, 0x00ff00ff00000001,
    0x00ff00ff0001ff00, 0x00ff00ff0001ff01, 0x00ff00ff00010000, 0x00ff00ff000101ff,
    0x00ff00ff00010101, 0x00ff00ff01ffff00, 0x00ff00ff01ff0001, 0x00ff00ff01ff0100,
    0x00ff00ff0100ffff, 0x00ff00ff0100ff01, 0x00ff00ff01000000, 0x00ff00ff0101ffff,
    0x00ff00ff0101ff00, 0x00ff00ff01010100, 0x00ff0000ffffff00, 0x00ff0000ffffff01,
    0x00ff0000ffff0000, 0x00ff0000ffff0101, 0x00ff0000ff00ff00, 0x00ff0000ff0000ff,
    0x00ff0000ff000000, 0x00ff0000ff000001, 0x00ff0000ff000100, 0x00ff0000ff01ffff,
    0x00ff0000ff010000, 0x00ff0000ff010101, 0x00ff000000ffff00, 0x00ff000000ff00ff,
    0x00ff000000ff0000, 0x00ff000000ff0001, 0x00ff000000ff0100, 0x00ff00000000ffff,
    0x00ff00000000ff00, 0x00ff0000000000ff, 0x00ff000000000000, 0x00ff000000000001,
    0x00ff0000000001ff, 0x00ff000000000100, 0x00ff00000001ff00, 0x00ff0000000100ff,
    0x00ff000000010000, 0x00ff000000010001, 0x00ff000000010100, 0x00ff000001ffff01,
    0x00ff000001ff00ff, 0x00ff000001ff0000, 0x00ff000001ff01ff, 0x00ff00000100ff00,
    0x00ff0000010000ff, 0x00ff000001000000, 0x00ff000001000001, 0x00ff000001000100,
    0x00ff000001000101, 0x00ff000001010000, 0x00ff0000010101ff, 0x00ff000001010101,
    0x00ff0001ffffff00, 0x00ff0001ffff0000, 0x00ff0001ffff0100, 0x00ff0001ff0000ff,
    0x00ff0001ff000000, 0x00ff0001ff0001ff, 0x00ff0001ff000101, 0x00ff0001ff01ff00,
    0x00ff0001ff0100ff, 0x00ff0001ff010100, 0x00ff000100ffffff, 0x00ff000100ffff01,
    0x00ff000100ff0000, 0x00ff000100ff01ff, 0x00ff00010000ffff, 0x00ff00010000ff00,
    0x00ff00010000ff01, 0x00ff000100000000, 0x00ff000100000001, 0x00ff000100000100,
    0x00ff00010001ff01, 0x00ff000100010000, 0x00ff0001000101ff, 0x00ff000101ffff00,
    0x00ff000101ff0000, 0x00ff000101ff0101, 0x00ff0001010000ff, 0x00ff000101000000,
    0x00ff00010101ff00, 0x00ff0001010100ff, 0x00ff000101010001, 0x00ff01ffffff0000,
    0x00ff01ffff00ff00, 0x00ff01ffff000000, 0x00ff01ffff000101, 0x00ff01ffff010000,
    0x00ff01ff00ffff01, 0x00ff01ff00ff0100, 0x00ff01ff0000ffff, 0x00ff01ff00000000,
    0x00ff01ff000001ff, 0x00ff01ff0001ff00, 0x00ff01ff000100ff, 0x00ff01ff00010001,
    0x00ff01ff00010100, 0x00ff01ff01ff0000, 0x00ff01ff0100ff00, 0x00ff01ff010000ff,
    0x00ff01ff01000001, 0x00ff01ff01000100, 0x00ff01ff01010000, 0x00ff0100ffffff00,
    0x00ff0100ffff0000, 0x00ff0100ffff0001, 0x00ff0100ffff0101, 0x00ff0100ff00ffff,
    0x00ff0100ff0000ff, 0x00ff0100ff000000, 0x00ff0100ff0001ff, 0x00ff0100ff01ff00,
    0x00ff0100ff0100ff, 0x00ff0100ff010001, 0x00ff010000ffffff, 0x00ff010000ff0000,
    0x00ff010000ff0101, 0x00ff01000000ff00, 0x00ff01000000ff01, 0x00ff0100000000ff,
    0x00ff010000000000, 0x00ff010000000001, 0x00ff010000000100, 0x00ff01000001ffff,
    0x00ff01000001ff01, 0x00ff010000010000, 0x00ff010000010001, 0x00ff010000010101,
    0x00ff010001ff0001, 0x00ff010001ff0100, 0x00ff01000100ff01, 0x00ff010001000000,
    0x00ff010001000001, 0x00ff0100010001ff, 0x00ff01000101ff00, 0x00ff0100010100ff,
    0x00ff010001010001, 0x00ff010001010100, 0x00ff0101ff000001, 0x00ff010100ff00ff,
    0x00ff010100ff0001, 0x00ff010100ff0100, 0x00ff010100000000, 0x00ff0101000001ff,
    0x00ff010100000101, 0x00ff0101000100ff, 0x00ff010100010100, 0x00ff0101010000ff,
    0x00ff010101010000, 0x0000ffffffffff00, 0x0000ffffffff00ff, 0x0000ffffffff0000,
    0x0000ffffffff0001, 0x0000ffffffff0100, 0x0000ffffff00ff01, 0x0000ffffff000000,
    0x0000ffffff000101, 0x0000ffffff01ff00, 0x0000ffffff0100ff, 0x0000ffffff010100,
    0x0000ffff00ffffff, 0x0000ffff00ff0000, 0x0000ffff00ff01ff, 0x0000ffff0000ff00,
    0x0000ffff000000ff, 0x0000ffff00000000, 0x0000ffff00000001, 0x0000ffff00000100,
    0x0000ffff00010000, 0x0000ffff000101ff, 0x0000ffff01ff0001, 0x0000ffff01ff0100,
    0x0000ffff01000000, 0x0000ffff010001ff, 0x0000ffff0101ffff, 0x0000ffff0101ff00,
    0x0000ffff01010001, 0x0000ffff01010100, 0x0000ff00ffff0000, 0x0000ff00ffff01ff,
    0x0000ff00ffff0100, 0x0000ff00ffff0101, 0x0000ff00ff00ff00, 0x0000ff00ff0000ff,
    0x0000ff00ff000000, 0x0000ff00ff000001, 0x0000ff00ff0001ff, 0x0000ff00ff000100,
    0x0000ff00ff01ffff, 0x0000ff00ff010000, 0x0000ff00ff010001, 0x0000ff00ff0101ff,
    0x0000ff00ff010101, 0x0000ff0000ffff00, 0x0000ff0000ff00ff, 0x0000ff0000ff0000,
    0x0000ff0000ff0001, 0x0000ff0000ff0100, 0x0000ff000000ffff, 0x0000ff000000ff00,
    0x0000ff000000ff01, 0x0000ff00000000ff, 0x0000ff0000000000, 0x0000ff0000000001,
    0x0000ff00000001ff, 0x0000ff0000000100, 0x0000ff0000000101, 0x0000ff000001ff00,
    0x0000ff00000100ff, 0x0000ff0000010000, 0x0000ff0000010001, 0x0000ff0000010100,
    0x0000ff0001ffff01, 0x0000ff0001ff0000, 0x0000ff000100ff00, 0x0000ff00010000ff,
    0x0000ff0001000000, 0x0000ff0001000001, 0x0000ff0001000100, 0x0000ff000101ffff,
    0x0000ff0001010000, 0x0000ff0001010101, 0x0000ff01ffffff00, 0x0000ff01ffff0001,
    0x0000ff01ff00ff01, 0x0000ff01ff000000, 0x0000ff01ff000101, 0x0000ff01ff01ff00,
    0x0000ff01ff0100ff, 0x0000ff0100ffff01, 0x0000ff0100ff0000, 0x0000ff0100ff0101,
    0x0000ff010000ff00, 0x0000ff01000000ff, 0x0000ff0100000000, 0x0000ff0100000001,
    0x0000ff0100000100, 0x0000ff010001ff01, 0x0000ff0100010000, 0x0000ff0101ff0000,
    0x0000ff010100ffff, 0x0000ff010100ff01, 0x0000ff0101000000, 0x0000ff0101000100,
    0x0000ff0101000101, 0x0000ff01010100ff, 0x000000ffffff00ff, 0x000000ffffff0000,
    0x000000ffff00ff00, 0x000000ffff0000ff, 0x000000ffff000000, 0x000000ffff000001,
    0x000000ffff0001ff, 0x000000ffff000100, 0x000000ffff01ff00, 0x000000ffff010000,
    0x000000ffff0101ff, 0x000000ffff010101, 0x000000ff00ffff00, 0x000000ff00ff00ff,
    0x000000ff00ff0000, 0x000000ff00ff0001, 0x000000ff00ff0100, 0x000000ff00ff0101,
    0x000000ff0000ffff, 0x000000ff0000ff00, 0x000000ff000000ff, 0x000000ff00000000,
    0x000000ff00000001, 0x000000ff000001ff, 0x000000ff00000100, 0x000000ff00000101,
    0x000000ff0001ff00, 0x000000ff0001ff01, 0x000000ff000100ff, 0x000000ff00010000,
    0x000000ff00010001, 0x000000ff00010100, 0x000000ff01ffffff, 0x000000ff01ff01ff,
    0x000000ff01ff0101, 0x000000ff0100ff00, 0x000000ff010000ff, 0x000000ff01000000,
    0x000000ff01000001, 0x000000ff01000100, 0x000000ff0101ff00, 0x000000ff010100ff,
    0x000000ff01010000, 0x000000ff01010101, 0x00000000ffffff00, 0x00000000ffffff01,
    0x00000000ffff00ff, 0x00000000ffff0000, 0x00000000ffff0001, 0x00000000ffff0100,
    0x00000000ff00ffff, 0x00000000ff00ff00, 0x00000000ff00ff01, 0x00000000ff0000ff,
    0x00000000ff000000, 0x00000000ff000001, 0x00000000ff000100, 0x00000000ff000101,
    0x00000000ff01ff00, 0x00000000ff0100ff, 0x00000000ff010000, 0x00000000ff010001,
    0x00000000ff010100, 0x0000000000ffffff, 0x0000000000ffff00, 0x0000000000ffff01,
    0x0000000000ff00ff, 0x0000000000ff0000, 0x0000000000ff0001, 0x0000000000ff01ff,
    0x0000000000ff0100, 0x000000000000ffff, 0x000000000000ff00, 0x000000000000ff01,
    0x00000000000000ff, 0x0000000000000000, 0x0000000000000001, 0x00000000000001ff,
    0x0000000000000100, 0x0000000000000101, 0x000000000001ffff, 0x000000000001ff00,
    0x00000000000100ff, 0x0000000000010000, 0x0000000000010001, 0x00000000000101ff,
    0x0000000000010100, 0x0000000000010101, 0x0000000001ffff00, 0x0000000001ff00ff,
    0x0000000001ff0000, 0x0000000001ff0100, 0x0000000001ff0101, 0x000000000100ffff,
    0x000000000100ff00, 0x00000000010000ff, 0x0000000001000000, 0x0000000001000001,
    0x00000000010001ff, 0x0000000001000100, 0x000000000101ff00, 0x00000000010100ff,
    0x0000000001010000, 0x0000000001010001, 0x0000000001010100, 0x00000001ffffffff,
    0x00000001ffffff00, 0x00000001ffffff01, 0x00000001ffff00ff, 0x00000001ffff0001,
    0x00000001ffff01ff, 0x00000001ffff0100, 0x00000001ff00ff00, 0x00000001ff0000ff,
    0x00000001ff000000, 0x00000001ff0001ff, 0x00000001ff000100, 0x00000001ff01ffff,
    0x00000001ff01ff00, 0x00000001ff01ff01, 0x00000001ff0100ff, 0x00000001ff010000,
    0x00000001ff010001, 0x00000001ff0101ff, 0x00000001ff010100, 0x0000000100ffff00,
    0x0000000100ff0000, 0x0000000100ff0001, 0x0000000100ff01ff, 0x0000000100ff0100,
    0x0000000100ff0101, 0x000000010000ffff, 0x000000010000ff00, 0x000000010000ff01,
    0x00000001000000ff, 0x0000000100000000, 0x0000000100000001, 0x00000001000001ff,
    0x0000000100000100, 0x0000000100000101, 0x000000010001ff00, 0x00000001000100ff,
    0x0000000100010000, 0x0000000100010100, 0x0000000101ffff01, 0x0000000101ff0000,
    0x0000000101ff0001, 0x0000000101ff01ff, 0x0000000101ff0100, 0x0000000101ff0101,
    0x000000010100ff00, 0x0000000101000000, 0x0000000101000101, 0x000000010101ff01,
    0x0000000101010000, 0x0000000101010001, 0x00000001010101ff, 0x0000000101010100,
    0x000001ffffff00ff, 0x000001ffffff0000, 0x000001ffffff0001, 0x000001ffffff0100,
    0x000001ffff00ffff, 0x000001ffff000000, 0x000001ffff0001ff, 0x000001ffff01ff00,
    0x000001ffff010101, 0x000001ff00ff0000, 0x000001ff00ff01ff, 0x000001ff00ff0101,
    0x000001ff0000ff00, 0x000001ff000000ff, 0x000001ff00000000, 0x000001ff00000001,
    0x000001ff000001ff, 0x000001ff00000100, 0x000001ff0001ffff, 0x000001ff0001ff01,
    0x000001ff000100ff, 0x000001ff00010000, 0x000001ff01ffff01, 0x000001ff01ff0100,
    0x000001ff0100ffff, 0x000001ff0100ff01, 0x000001ff01000000, 0x000001ff010001ff,
    0x000001ff0101ff00, 0x000001ff01010100, 0x00000100ffffff00, 0x00000100ffffff01,
    0x00000100ffff0000, 0x00000100ffff0101, 0x00000100ff00ff00, 0x00000100ff0000ff,
    0x00000100ff000000, 0x00000100ff000001, 0x00000100ff000100, 0x00000100ff010000,
    0x0000010000ffff00, 0x0000010000ff00ff, 0x0000010000ff0000, 0x0000010000ff0001,
    0x0000010000ff0100, 0x000001000000ffff, 0x000001000000ff00, 0x000001000000ff01,
    0x00000100000000ff, 0x0000010000000000, 0x0000010000000001, 0x00000100000001ff,
    0x0000010000000100, 0x0000010000000101, 0x000001000001ff00, 0x00000100000100ff,
    0x0000010000010000, 0x0000010000010001, 0x0000010000010100, 0x0000010001ffff00,
    0x0000010001ff0000, 0x0000010001ff0100, 0x000001000100ff00, 0x00000100010000ff,
    0x0000010001000000, 0x0000010001000001, 0x00000100010001ff, 0x0000010001000100,
    0x0000010001010000, 0x00000101ffff00ff, 0x00000101ffff01ff, 0x00000101ff000000,
    0x00000101ff000101, 0x00000101ff01ffff, 0x00000101ff010000, 0x00000101ff010001,
    0x00000101ff010100, 0x0000010100ff0000, 0x0000010100ff01ff, 0x0000010100ff0100,
    0x000001010000ff00, 0x0000010100000000, 0x0000010100000001, 0x00000101000001ff,
    0x0000010100000100, 0x000001010001ff01, 0x0000010100010000, 0x00000101000101ff,
    0x0000010100010101, 0x0000010101ffff00, 0x0000010101ff0101, 0x000001010100ff01,
    0x0000010101000000, 0x0000010101000001, 0x00000101010001ff, 0x0000010101000101,
    0x000001010101ff00, 0x0001ffffffff0000, 0x0001ffffff0000ff, 0x0001ffffff000001,
    0x0001ffffff000100, 0x0001ffffff010000, 0x0001ffff00ff00ff, 0x0001ffff0000ffff,
    0x0001ffff00000000, 0x0001ffff00000001, 0x0001ffff000001ff, 0x0001ffff00000101,
    0x0001ffff0001ff00, 0x0001ffff000100ff, 0x0001ffff00010001, 0x0001ffff00010100,
    0x0001ffff01ffff00, 0x0001ffff01000001, 0x0001ffff01010000, 0x0001ff00ffffff00,
    0x0001ff00ffff00ff, 0x0001ff00ffff0001, 0x0001ff00ffff0100, 0x0001ff00ff00ff01,
    0x0001ff00ff000000, 0x0001ff00ff01ff00, 0x0001ff00ff01ff01, 0x0001ff00ff010001,
    0x0001ff00ff010100, 0x0001ff0000ff0000, 0x0001ff0000ff0100, 0x0001ff000000ff00,
    0x0001ff0000000000, 0x0001ff0000000001, 0x0001ff0000000100, 0x0001ff0000010000,
    0x0001ff0000010001, 0x0001ff0000010101, 0x0001ff0001ff00ff, 0x0001ff0001ff0101,
    0x0001ff000100ff01, 0x0001ff0001000000, 0x0001ff000101ff00, 0x0001ff0001010001,
    0x0001ff0001010100, 0x0001ff01ff00ff00, 0x0001ff01ff000001, 0x0001ff01ff000100,
    0x0001ff0100ffffff, 0x0001ff0100ffff00, 0x0001ff0100ff0001, 0x0001ff0100000000,
    0x0001ff0100000001, 0x0001ff01000001ff, 0x0001ff010001ffff, 0x0001ff0101ff0000,
    0x0001ff010100ff00, 0x0001ff0101000001, 0x0001ff0101010000, 0x000100ffff00ff00,
    0x000100ffff00ff01, 0x000100ffff000000, 0x000100ffff000001, 0x000100ffff000101,
    0x000100ffff01ff00, 0x000100ffff010001, 0x000100ffff010100, 0x000100ff00ffffff,
    0x000100ff00ffff01, 0x000100ff00ff0000, 0x000100ff00ff01ff, 0x000100ff00ff0101,
    0x000100ff0000ff00, 0x000100ff000000ff, 0x000100ff00000000, 0x000100ff00000001,
    0x000100ff00000100, 0x000100ff00000101, 0x000100ff0001ffff, 0x000100ff0001ff01,
    0x000100ff00010000, 0x000100ff01ff00ff, 0x000100ff01ff0000, 0x000100ff01ff0100,
    0x000100ff0100ffff, 0x000100ff0100ff01, 0x000100ff010000ff, 0x000100ff01000000,
    0x000100ff01000001, 0x000100ff010001ff, 0x000100ff01000101, 0x000100ff0101ff00,
    0x000100ff010100ff, 0x000100ff01010100, 0x00010000ffff0000, 0x00010000ffff01ff,
    0x00010000ffff0101, 0x00010000ff00ff00, 0x00010000ff000000, 0x00010000ff000001,
    0x00010000ff000100, 0x0001000000ff00ff, 0x0001000000ff0000, 0x0001000000ff0001,
    0x0001000000ff0100, 0x000100000000ffff, 0x000100000000ff00, 0x00010000000000ff,
    0x0001000000000000, 0x0001000000000001, 0x0001000000000100, 0x000100000001ff00,
    0x00010000000100ff, 0x0001000000010000, 0x0001000000010001, 0x0001000000010100,
    0x0001000001ff0001, 0x0001000001ff0100, 0x0001000001ff0101, 0x000100000100ff00,
    0x0001000001000000, 0x0001000001000001, 0x0001000001000100, 0x0001000001000101,
    0x000100000101ff01, 0x0001000001010000, 0x0001000001010001, 0x00010000010101ff,
    0x00010001ffffff01, 0x00010001ffff0100, 0x00010001ff000000, 0x00010001ff01ffff,
    0x00010001ff010001, 0x00010001ff0101ff, 0x00010001ff010100, 0x0001000100ffffff,
    0x0001000100ff0000, 0x0001000100ff01ff, 0x0001000100ff0101, 0x000100010000ff00,
    0x00010001000000ff, 0x0001000100000000, 0x0001000100000001, 0x00010001000001ff,
    0x0001000100000101, 0x000100010001ffff, 0x0001000100010000, 0x00010001000101ff,
    0x0001000101ffffff, 0x0001000101ffff01, 0x0001000101ff0000, 0x0001000101ff0101,
    0x00010001010000ff, 0x0001000101000001, 0x00010001010001ff, 0x0001000101000100,
    0x000100010101ffff, 0x00010001010100ff, 0x0001000101010001, 0x0001000101010101,
    0x000101ffff000001, 0x000101ffff000100, 0x000101ffff010000, 0x000101ff00ffff00,
    0x000101ff0000ff01, 0x000101ff00000000, 0x000101ff00000101, 0x000101ff0001ff00,
    0x000101ff00010100, 0x000101ff01ff0000, 0x000101ff0100ff00, 0x000101ff010001ff,
    0x000101ff01010001, 0x00010100ffffff00, 0x00010100ffff00ff, 0x00010100ff00ffff,
    0x00010100ff000000, 0x00010100ff01ff00, 0x00010100ff0100ff, 0x00010100ff010001,
    0x00010100ff010100, 0x0001010000ffffff, 0x0001010000ffff00, 0x0001010000ff0000,
    0x0001010000ff0001, 0x0001010000ff01ff, 0x000101000000ff00, 0x00010100000000ff,
    0x0001010000000000, 0x0001010000000001, 0x0001010000000100, 0x000101000001ffff,
    0x0001010000010000, 0x0001010000010101, 0x0001010001ffff01, 0x0001010001ff00ff,
    0x0001010001ff0101, 0x0001010001000000, 0x000101000101ff00, 0x00010100010100ff,
    0x0001010001010000, 0x0001010001010100, 0x00010101ff00ff00, 0x00010101ff000001,
    0x00010101ff0001ff, 0x0001010100ffff00, 0x0001010100ff00ff, 0x0001010100ff0100,
    0x000101010000ffff, 0x0001010100000000, 0x00010101000001ff, 0x0001010100000101,
    0x00010101000100ff, 0x0001010100010000, 0x0001010100010100, 0x0001010101ff0001,
    0x00010101010000ff, 0x00010101010001ff, 0x0001010101000101, 0x0001010101010001,
    0x01ffffffffffffff, 0x01ffffffffffff01, 0x01ffffffffff01ff, 0x01ffffffffff0101,
    0x01ffffffff01ffff, 0x01ffffffff01ff01, 0x01ffffffff0101ff, 0x01ffffffff010101,
    0x01ffffff00ff0000, 0x01ffffff0000ffff, 0x01ffffff0000ff00, 0x01ffffff000000ff,
    0x01ffffff00000001, 0x01ffffff00000100, 0x01ffffff00010000, 0x01ffffff01ffffff,
    0x01ffffff01ffff01, 0x01ffffff01ff01ff, 0x01ffffff01ff0101, 0x01ffffff01000000,
    0x01ffffff0101ffff, 0x01ffffff0101ff01, 0x01ffffff010101ff, 0x01ffffff01010101,
    0x01ffff00ffff0000, 0x01ffff00ff00ff00, 0x01ffff00ff0000ff, 0x01ffff00ff000001,
    0x01ffff00ff000100, 0x01ffff00ff010000, 0x01ffff0000ffff00, 0x01ffff0000ff00ff,
    0x01ffff0000ff0100, 0x01ffff000000ffff, 0x01ffff000000ff01, 0x01ffff0000000000,
    0x01ffff0000000001, 0x01ffff00000001ff, 0x01ffff0000000100, 0x01ffff00000100ff,
    0x01ffff0000010001, 0x01ffff0000010100, 0x01ffff0001ff0000, 0x01ffff0001ff0100,
    0x01ffff00010000ff, 0x01ffff0001000001, 0x01ffff0001000100, 0x01ffff0001010000,
    0x01ffff01ffffffff, 0x01ffff01ffffff01, 0x01ffff01ffff01ff, 0x01ffff01ffff0101,
    0x01ffff01ff000000, 0x01ffff01ff01ffff, 0x01ffff01ff01ff01, 0x01ffff01ff0101ff,
    0x01ffff01ff010101, 0x01ffff010000ff00, 0x01ffff01000000ff, 0x01ffff0100000100,
    0x01ffff0100010000, 0x01ffff0101ffffff, 0x01ffff0101ffff01, 0x01ffff0101ff01ff,
    0x01ffff0101ff0101, 0x01ffff0101000000, 0x01ffff010101ffff, 0x01ffff010101ff01,
    0x01ffff01010101ff, 0x01ffff0101010101, 0x01ff00ffff0000ff, 0x01ff00ffff000100,
    0x01ff00ff00ffff00, 0x01ff00ff00ff00ff, 0x01ff00ff0000ff00, 0x01ff00ff00000000,
    0x01ff00ff00000101, 0x01ff00ff0001ff00, 0x01ff00ff000100ff, 0x01ff00ff00010100,
    0x01ff00ff010000ff, 0x01ff00ff01000100, 0x01ff0000ffffff00, 0x01ff0000ffff0100,
    0x01ff0000ff00ff01, 0x01ff0000ff000000, 0x01ff0000ff000101, 0x01ff0000ff010001,
    0x01ff0000ff010100, 0x01ff000000ffffff, 0x01ff000000ffff00, 0x01ff000000ff0000,
    0x01ff000000ff01ff, 0x01ff00000000ff00, 0x01ff0000000000ff, 0x01ff000000000000,
    0x01ff000000000001, 0x01ff000000000100, 0x01ff000000000101, 0x01ff000000010000,
    0x01ff000000010001, 0x01ff0000000101ff, 0x01ff000000010101, 0x01ff000001ffff00,
    0x01ff000001ff00ff, 0x01ff000001ff0001, 0x01ff000001ff0100, 0x01ff00000100ffff,
    0x01ff00000100ff01, 0x01ff000001000000, 0x01ff0000010001ff, 0x01ff000001010001,
    0x01ff0001ff00ff00, 0x01ff0001ff000001, 0x01ff0001ff000100, 0x01ff0001ff010000,
    0x01ff000100ffff00, 0x01ff000100ff00ff, 0x01ff000100ff0100, 0x01ff000100ff0101,
    0x01ff00010000ffff, 0x01ff000100000000, 0x01ff000100000100, 0x01ff000100000101,
    0x01ff00010001ff00, 0x01ff000100010001, 0x01ff000100010101, 0x01ff000101ff0000,
    0x01ff00010100ff00, 0x01ff000101000101, 0x01ff0001010100ff, 0x01ff01ffffffffff,
    0x01ff01ffffffff01, 0x01ff01ffffff01ff, 0x01ff01ffffff0101, 0x01ff01ffff000000,
    0x01ff01ffff01ffff, 0x01ff01ffff01ff01, 0x01ff01ffff0101ff, 0x01ff01ffff010101,
    0x01ff01ff00ffff00, 0x01ff01ff00ff0000, 0x01ff01ff0000ff00, 0x01ff01ff000000ff,
    0x01ff01ff00000100, 0x01ff01ff00010000, 0x01ff01ff00010100, 0x01ff01ff01ffffff,
    0x01ff01ff01ffff01, 0x01ff01ff01ff01ff, 0x01ff01ff01ff0101, 0x01ff01ff01000000,
    0x01ff01ff0101ffff, 0x01ff01ff0101ff01, 0x01ff01ff010101ff, 0x01ff01ff01010101,
    0x01ff0100ffff0000, 0x01ff0100ffff0001, 0x01ff0100ff00ff00, 0x01ff0100ff0000ff,
    0x01ff0100ff000001, 0x01ff0100ff010000, 0x01ff010000ffff00, 0x01ff010000ff00ff,
    0x01ff010000ff0001, 0x01ff010000ff0100, 0x01ff01000000ffff, 0x01ff01000000ff01,
    0x01ff010000000000, 0x01ff010000000101, 0x01ff01000001ff00, 0x01ff0100000100ff,
    0x01ff010001ff0000, 0x01ff010001000001, 0x01ff010001000100, 0x01ff010001010000,
    0x01ff0101ffffffff, 0x01ff0101ffffff01, 0x01ff0101ffff01ff, 0x01ff0101ffff0101,
    0x01ff0101ff000000, 0x01ff0101ff01ffff, 0x01ff0101ff01ff01, 0x01ff0101ff0101ff,
    0x01ff0101ff010101, 0x01ff010100ff0000, 0x01ff01010000ff00, 0x01ff0101000000ff,
    0x01ff010100000001, 0x01ff010101ffffff, 0x01ff010101ffff01, 0x01ff010101ff01ff,
    0x01ff010101ff0101, 0x01ff010101000000, 0x01ff01010101ffff, 0x01ff01010101ff01,
    0x01ff0101010101ff, 0x01ff010101010101, 0x0100ffffffff0000, 0x0100ffffff00ff00,
    0x0100ffffff000001, 0x0100ffffff0001ff, 0x0100ffffff000100, 0x0100ffffff010000,
    0x0100ffff00ffff00, 0x0100ffff00ff0001, 0x0100ffff00ff0100, 0x0100ffff00000000,
    0x0100ffff000001ff, 0x0100ffff00000101, 0x0100ffff00010100, 0x0100ffff00010101,
    0x0100ffff01ff0000, 0x0100ffff0100ff00, 0x0100ffff010000ff, 0x0100ffff01000001,
    0x0100ffff01000100, 0x0100ffff01010000, 0x0100ff00ffffff00, 0x0100ff00ffff00ff,
    0x0100ff00ffff0001, 0x0100ff00ffff0100, 0x0100ff00ff00ffff, 0x0100ff00ff000000,
    0x0100ff00ff0001ff, 0x0100ff00ff000101, 0x0100ff00ff01ff00, 0x0100ff00ff0100ff,
    0x0100ff00ff010001, 0x0100ff00ff010100, 0x0100ff0000ffffff, 0x0100ff0000ff0000,
    0x0100ff000000ffff, 0x0100ff000000ff00, 0x0100ff00000000ff, 0x0100ff0000000000,
    0x0100ff0000000001, 0x0100ff0000000100, 0x0100ff000001ff01, 0x0100ff0000010000,
    0x0100ff0001ff00ff, 0x0100ff0001ff0001, 0x0100ff000100ff01, 0x0100ff0001000000,
    0x0100ff00010001ff, 0x0100ff000101ff00, 0x0100ff00010100ff, 0x0100ff0001010001,
    0x0100ff0001010100, 0x0100ff01ffff0000, 0x0100ff01ff00ff00, 0x0100ff01ff0000ff,
    0x0100ff01ff000100, 0x0100ff01ff010000, 0x0100ff0100ff00ff, 0x0100ff0100ff0001,
    0x0100ff0100ff0100, 0x0100ff010000ffff, 0x0100ff010000ff01, 0x0100ff0100000000,
    0x0100ff01000001ff, 0x0100ff0100010001, 0x0100ff0100010100, 0x0100ff0101ff0000,
    0x0100ff01010000ff, 0x0100ff0101000001, 0x0100ff0101010100, 0x010000ffffffff00,
    0x010000ffffff00ff, 0x010000ffffff0001, 0x010000ffff00ffff, 0x010000ffff000000,
    0x010000ffff0001ff, 0x010000ffff010001, 0x010000ff00ffffff, 0x010000ff00ff0101,
    0x010000ff0000ff00, 0x010000ff000000ff, 0x010000ff00000000, 0x010000ff00000001,
    0x010000ff000001ff, 0x010000ff00000100, 0x010000ff0001ffff, 0x010000ff0001ff00,
    0x010000ff0001ff01, 0x010000ff00010000, 0x010000ff01ff00ff, 0x010000ff01ff0001,
    0x010000ff0100ff01, 0x010000ff010000ff, 0x010000ff01000000, 0x010000ff010001ff,
    0x010000ff0101ff00, 0x010000ff01010100, 0x01000000ffffffff, 0x01000000ffff0000,
    0x01000000ffff01ff, 0x01000000ffff0101, 0x01000000ff00ffff, 0x01000000ff00ff00,
    0x01000000ff0000ff, 0x01000000ff000000, 0x01000000ff000001, 0x01000000ff000100,
    0x01000000ff01ff00, 0x01000000ff010000, 0x01000000ff010100, 0x01000000ff010101,
    0x0100000000ffff00, 0x0100000000ff00ff, 0x0100000000ff0000, 0x0100000000ff0001,
    0x0100000000ff0100, 0x010000000000ffff, 0x010000000000ff00, 0x010000000000ff01,
    0x01000000000000ff, 0x0100000000000000, 0x0100000000000001, 0x01000000000001ff,
    0x0100000000000100, 0x0100000000000101, 0x010000000001ff00, 0x01000000000100ff,
    0x0100000000010000, 0x0100000000010001, 0x0100000000010100, 0x0100000001ffff00,
    0x0100000001ff0000, 0x0100000001ff01ff, 0x010000000100ff00, 0x010000000100ff01,
    0x01000000010000ff, 0x0100000001000000, 0x0100000001000001, 0x0100000001000100,
    0x0100000001000101, 0x010000000101ffff, 0x010000000101ff01, 0x0100000001010000,
    0x01000000010101ff, 0x0100000001010101, 0x01000001ffffff00, 0x01000001ffff00ff,
    0x01000001ff00ffff, 0x01000001ff000000, 0x01000001ff000100, 0x01000001ff01ffff,
    0x01000001ff010001, 0x01000001ff010100, 0x0100000100ff0000, 0x0100000100ff01ff,
    0x0100000100ff0100, 0x010000010000ff00, 0x010000010000ff01, 0x0100000100000000,
    0x0100000100000001, 0x0100000100000100, 0x0100000100010000, 0x01000001000101ff,
    0x0100000101ffff01, 0x0100000101ff00ff, 0x0100000101ff0100, 0x0100000101ff0101,
    0x010000010100ff01, 0x01000001010000ff, 0x0100000101000000, 0x01000001010100ff,
    0x0100000101010001, 0x0100000101010100, 0x010001ffffff0000, 0x010001ffff000001,
    0x010001ffff000100, 0x010001ffff010000, 0x010001ff00ffff00, 0x010001ff00ff0001,
    0x010001ff0000ffff, 0x010001ff0000ff01, 0x010001ff00000000, 0x010001ff00000001,
    0x010001ff00000101, 0x010001ff000100ff, 0x010001ff00010000, 0x010001ff01ff0000,
    0x010001ff0100ff00, 0x010001ff01000001, 0x010001ff01000100, 0x010001ff01010000,
    0x01000100ffff00ff, 0x01000100ffff0001, 0x01000100ffff0100, 0x01000100ff00ffff,
    0x01000100ff00ff01, 0x01000100ff000000, 0x01000100ff0001ff, 0x01000100ff000101,
    0x01000100ff01ffff, 0x01000100ff01ff00, 0x01000100ff0100ff, 0x01000100ff010001,
    0x0100010000ffffff, 0x0100010000ffff01, 0x0100010000ff0000, 0x0100010000ff01ff,
    0x0100010000ff0101, 0x010001000000ff00, 0x01000100000000ff, 0x0100010000000000,
    0x0100010000000001, 0x0100010000000100, 0x010001000001ff01, 0x0100010000010000,
    0x0100010000010001, 0x0100010000010101, 0x0100010001ffff00, 0x0100010001ff00ff,
    0x010001000100ffff, 0x010001000100ff01, 0x0100010001000000, 0x0100010001000101,
    0x010001000101ff00, 0x0100010001010001, 0x01000101ffff0000, 0x01000101ff000000,
    0x01000101ff010000, 0x0100010100ff00ff, 0x0100010100ff0001, 0x0100010100ff0100,
    0x010001010000ffff, 0x0100010100000000, 0x01000101000001ff, 0x010001010001ff00,
    0x0100010101ff0000, 0x010001010100ff00, 0x01000101010000ff, 0x0100010101000000,
    0x0100010101000001, 0x0101ffffffffffff, 0x0101ffffffffff01, 0x0101ffffffff01ff,
    0x0101ffffffff0101, 0x0101ffffff000000, 0x0101ffffff01ffff, 0x0101ffffff01ff01,
    0x0101ffffff0101ff, 0x0101ffffff010101, 0x0101ffff00ff0000, 0x0101ffff0000ff00,
    0x0101ffff000000ff, 0x0101ffff00000001, 0x0101ffff00000100, 0x0101ffff01ffffff,
    0x0101ffff01ffff01, 0x0101ffff01ff01ff, 0x0101ffff01ff0101, 0x0101ffff01000000,
    0x0101ffff0101ffff, 0x0101ffff0101ff01, 0x0101ffff010101ff, 0x0101ffff01010101,
    0x0101ff00ffff0000, 0x0101ff00ffff0100, 0x0101ff00ff00ff00, 0x0101ff00ff0000ff,
    0x0101ff00ff000001, 0x0101ff00ff000100, 0x0101ff00ff000101, 0x0101ff0000ff0001,
    0x0101ff0000ff0100, 0x0101ff000000ff00, 0x0101ff0000000000, 0x0101ff00000001ff,
    0x0101ff0000000101, 0x0101ff000001ff00, 0x0101ff00000100ff, 0x0101ff0001ff0000,
    0x0101ff000100ffff, 0x0101ff000100ff01, 0x0101ff0001000001, 0x0101ff0001000100,
    0x0101ff01ffffff01, 0x0101ff01ffff01ff, 0x0101ff01ffff0101, 0x0101ff01ff00ffff,
    0x0101ff01ff000100, 0x0101ff01ff01ff01, 0x0101ff01ff0101ff, 0x0101ff01ff010101,
    0x0101ff0100ff0000, 0x0101ff010000ff00, 0x0101ff0100000001, 0x0101ff0100000100,
    0x0101ff0100010000, 0x0101ff0101ffffff, 0x0101ff0101ffff01, 0x0101ff0101ff01ff,
    0x0101ff0101ff0101, 0x0101ff0101000000, 0x0101ff010101ffff, 0x0101ff010101ff01,
    0x0101ff01010101ff, 0x0101ff0101010101, 0x010100ffff000100, 0x010100ffff010000,
    0x010100ff00ffff00, 0x010100ff00ff00ff, 0x010100ff0000ffff, 0x010100ff000000ff,
    0x010100ff00000000, 0x010100ff000001ff, 0x010100ff00000101, 0x010100ff0001ff00,
    0x010100ff00010000, 0x010100ff00010001, 0x010100ff000101ff, 0x010100ff00010100,
    0x010100ff01ff0000, 0x01010000ffff0001, 0x01010000ffff0100, 0x01010000ff00ffff,
    0x01010000ff00ff01, 0x01010000ff000000, 0x01010000ff0001ff, 0x01010000ff010001,
    0x01010000ff010100, 0x0101000000ffff01, 0x0101000000ff0000, 0x010100000000ff00,
    0x01010000000000ff, 0x0101000000000000, 0x0101000000000001, 0x0101000000000100,
    0x0101000000010000, 0x0101000000010101, 0x0101000001ffff00, 0x0101000001ff00ff,
    0x0101000001ff0000, 0x0101000001ff0001, 0x0101000001ff0100, 0x010100000100ff01,
    0x0101000001000000, 0x01010000010001ff, 0x01010001ffff0000, 0x01010001ff00ff00,
    0x01010001ff000001, 0x01010001ff000101, 0x01010001ff01ff00, 0x01010001ff010000,
    0x0101000100ff00ff, 0x0101000100ff0001, 0x0101000100ff0101, 0x010100010000ff01,
    0x0101000100000000, 0x0101000100000001, 0x01010001000001ff, 0x010100010001ffff,
    0x010100010001ff01, 0x0101000101ff0001, 0x010100010100ffff, 0x0101000101000000,
    0x0101000101000001, 0x0101000101000100, 0x010100010101ff00, 0x01010001010100ff,
    0x0101000101010001, 0x010101ffffffffff, 0x010101ffffffff01, 0x010101ffffff01ff,
    0x010101ffffff0101, 0x010101ffff01ffff, 0x010101ffff01ff01, 0x010101ffff0101ff,
    0x010101ffff010101, 0x010101ff0000ff00, 0x010101ff000000ff, 0x010101ff00000001,
    0x010101ff00000100, 0x010101ff01ffffff, 0x010101ff01ffff01, 0x010101ff01ff01ff,
    0x010101ff01ff0101, 0x010101ff01000000, 0x010101ff0101ffff, 0x010101ff0101ff01,
    0x010101ff010101ff, 0x010101ff01010101, 0x01010100ffff0000, 0x01010100ff0000ff,
    0x01010100ff000100, 0x01010100ff01ff00, 0x01010100ff010000, 0x0101010000ffff00,
    0x010101000000ffff, 0x0101010000000000, 0x0101010000000101, 0x010101000001ff00,
    0x0101010000010001, 0x0101010000010100, 0x010101000100ffff, 0x0101010001000001,
    0x01010101ffffffff, 0x01010101ffffff01, 0x01010101ffff01ff, 0x01010101ffff0101,
    0x01010101ff01ffff, 0x01010101ff01ff01, 0x01010101ff0101ff, 0x01010101ff010101,
    0x010101010000ff00, 0x01010101000000ff, 0x0101010100000001, 0x0101010101ffffff,
    0x0101010101ffff01, 0x0101010101ff01ff, 0x0101010101ff0101, 0x0101010101000000,
    0x010101010101ffff, 0x010101010101ff01, 0x01010101010101ff, 0x0101010101010101,
],dtype = np.uint64)
grid_int8 = np.zeros((len(iq1s_grid), 8), dtype=np.int8)
for i, grid_val in enumerate(iq1s_grid):
    grid_bytes = grid_val.tobytes()
    for j in range(min(8, len(grid_bytes))):
        #grid_int8[i, j] = int(grid_bytes[j])
        grid_int8[i,j] = np.int8(np.uint8(grid_bytes[j]))
def offs_concate_experts(w1,shape,type,w3=None):
    n_per_row = shape[0]
    row = shape[1]
    e = shape[2]
    if type == "IQ1_S_R4_RES_Q2_K":
        block_size, type_size = GGML_QUANT_SIZES[348]
        row_size = (shape[0] // block_size) * type_size + GGML_ROW_META_SIZE["IQ1_S_R4"]
        res_ratio = shape[1] * ((shape[0] // 32) * GGML_BLOCK_SIZES["IQ1_S_R4"] + GGML_ROW_META_SIZE["IQ1_S_R4"]) / (row_size * shape[1])
    elif type == "IQ1_M_R4_RES_Q2_K":
        block_size, type_size = GGML_QUANT_SIZES[350]
        row_size = (shape[0] // block_size) * type_size + GGML_ROW_META_SIZE["IQ1_M_R4"]
        res_ratio = shape[1] * ((shape[0] // 32) * GGML_BLOCK_SIZES["IQ1_M_R4"] + GGML_ROW_META_SIZE["IQ1_M_R4"]) / (row_size * shape[1])
    elif type == 'IQ1_M_PLUS_Q4_K_R4':
        block_size, type_size = GGML_QUANT_SIZES[351]
        row_size = (shape[0] // block_size) * type_size + GGML_ROW_META_SIZE["IQ1_M_R4"]
        res_ratio = shape[1] * ((shape[0] // 32) * GGML_BLOCK_SIZES["IQ1_M_R4"] + GGML_ROW_META_SIZE["IQ1_M_R4"]) / (row_size * shape[1])
    else:
        block_size = 32
        type_size =  GGML_BLOCK_SIZES[type]
    expert_stride = row * row_size
    offs = round(res_ratio * expert_stride)
    result = []
    for i in range(e):
        if w3 is not None:
            d1 = w1[i*expert_stride:i*expert_stride + offs]
            d3 = w3[i*expert_stride:i*expert_stride + offs]
            d1 = np.concatenate([d1,d3])
        else:
            d1 = w1[i*expert_stride:i*expert_stride + offs]
        result.append(d1)
    result = np.concatenate(result)
    return result
def offs_concate_experts_q2(w1,shape,type,w3=None):
    n_per_row = shape[0]
    row = shape[1]
    e = shape[2]
    if type == "IQ1_S_R4_RES_Q2_K":
        block_size, type_size = GGML_QUANT_SIZES[348]
    elif type == "IQ1_M_R4_RES_Q2_K":
        block_size, type_size = GGML_QUANT_SIZES[350]
    else:
        block_size = 32
        type_size =  GGML_BLOCK_SIZES[type]
    row_size = (shape[0] // block_size) * type_size + GGML_ROW_META_SIZE["IQ1_S_R4"]
    res_ratio = shape[1] * ((shape[0] // 32) * GGML_BLOCK_SIZES["IQ1_S_R4"] + GGML_ROW_META_SIZE["IQ1_S_R4"]) / (row_size * shape[1])
    expert_stride = row * row_size
    offs = int(res_ratio * expert_stride)
    result = []
    for i in range(8):
        if w3 is not None:
            d1 = w1[i*expert_stride + offs :(i+1)*expert_stride]
            d3 = w3[i*expert_stride + offs :(i+1)*expert_stride]
            d1 = np.concatenate([d1,d3])
        else:
            d1 = w1[i*expert_stride + offs :(i+1)*expert_stride]
        result.append(d1)
    result = np.concatenate(result)
    return result
def dequantize_experts(tensor,shape,target_dtype = torch.float32,device="cpu"):
    e = shape[-1]
    result= []
    block_size = GGML_BLOCK_SIZES["IQ1_S_R4"]
    num_rows = shape[1]
    n_per_row = shape[0]
    row_size = (n_per_row // 32) * block_size + GGML_ROW_META_SIZE["IQ1_S_R4"]
    expert_stride = shape[1] * row_size
    for i in range(e):
        data = tensor[i*expert_stride:]
        if device!="cpu":
            res = dequantize_iq1_s_r4_gpu(data,device=device,target_dtype=target_dtype,shape=shape[:2])
        else:
            res = dequantize_iq1_s_r4(data, shape[:2])
        # 拼接res
        result.append(res)
    return result
# 没有验证精度结果，从数值上来说大概率没问题
def dequantize_iq1_s_r4(data, shape=None):
    IQ1S_DELTA = 0.125
    # 参照 third_party/llama.cpp/iqk_quantize.cpp 中的 dequantize_row_iq1_s_r4_q1 函数实现
    block_size = GGML_BLOCK_SIZES["IQ1_S_R4"]
    num_rows = shape[1]
    n_per_row = shape[0]
    
    # 确保行数是4的倍数，每行的元素数是32的倍数
    assert num_rows % 4 == 0, "行数必须是4的倍数"
    assert n_per_row % 32 == 0, "每行的元素数必须是32的倍数"
    
    row_size = (n_per_row // 32) * block_size + GGML_ROW_META_SIZE["IQ1_S_R4"]
    result = np.zeros((num_rows,n_per_row), dtype=np.float32)
    
    cx = data
    for row in range(0, num_rows, 4):
        # 获取当前4行的scale值
        dptr = np.frombuffer(cx, dtype=np.float16)[:4]
        # 0.001924,0.001807,0.001947,0.001556
        x_offset = 8  # 4个float16值后的偏移
        x = np.frombuffer(cx[x_offset:], dtype=np.uint8).reshape(-1)
        # 转换scale值为float32
        d = dptr.astype(np.float32)
        
        nblock = n_per_row // 32
        
        # 对每个block进行解量化
        for ib in range(nblock):
            block_offset = ib * block_size * 4
            for k in range(4):
                # 获取高位信息
                qh = (np.uint16(x[block_offset + 16 + 2*k + 1]) << 8) | np.uint16(x[block_offset + 16 + 2*k])
                
                # 根据最高位判断shift方向
                shift = -IQ1S_DELTA if (qh & 0x8000) != 0 else IQ1S_DELTA
                # 计算缩放因子
                dl = d[k] * (2.0 * ((qh >> 12) & 7) + 1.0)
                
                for i in range(4):
                    # 8个数对应8b
                    # 计算索引
                    # x[block_offset + 4*i + k]
                    idx = x[block_offset + 4*i + k] | (((qh >> 3*i) & 7) << 8)
                    # 直接使用预计算的grid值
                    grid = grid_int8[idx]
                    
                    # 向量化计算并写入结果
                    result[ row + k,32*ib + 8*i:32*ib + 8*i + 8] = dl * (grid + shift)
        
        # 移动到下一组4行
        cx = cx[4 * row_size:]
    
    # 移除调试断点
    return result.flatten()

def dequantize_q2_k_r4(data, shape=None):
    # 常量定义
    QK_K = 256  # 每个块的大小
    block_size = 84 * 4
    n_per_row = shape[0]# 每行的元素数除以4
    num_rows = shape[1]
    # 初始化输出数组
    y = np.zeros((num_rows,n_per_row), dtype=np.float32)
    y4 = [y[i * n_per_row:(i + 1) * n_per_row] for i in range(4)]  # 将输出分为4部分

    # 计算块数
    nblock = n_per_row // QK_K
    for r4 in range(0, num_rows, 4):
        rt = r4
        offs = r4 * nblock * block_size // 4
        # 遍历每个块
        for ibl in range(nblock):
            # 获取当前块的指针
            x = data[offs + ibl * block_size:offs + (ibl + 1) * block_size]

            # 遍历4个部分
            for k in range(4):
                # 获取d和m值
                d = np.frombuffer(x[k * 2:(k + 1) * 2], dtype=np.float16)[0]
                m = np.frombuffer(x[(k + 4) * 2:(k + 5) * 2], dtype=np.float16)[0]

                # 获取量化数据
                ql = x[80:80 + 256]

                # 遍历每个子块
                for ib in range(QK_K // 32):
                    # 计算dl1, ml1, dl2, ml2
                    dl1 = d * (x[16 + ib * 8 + k] & 0xf)
                    ml1 = m * (x[16 + ib * 8 + k] >> 4)
                    dl2 = d * (x[16 + ib * 8 + k + 4] & 0xf)
                    ml2 = m * (x[16 + ib * 8 + k + 4] >> 4)

                    # 解量化并写入结果
                    for i in range(4):
                        y[rt + k][ibl * QK_K + ib * 32 + i + 0] = dl1 * ((ql[ib*32+4*k+i] >> 0) & 3) - ml1
                        y[rt + k][ibl * QK_K + ib * 32 + i + 4] = dl1 * ((ql[ib*32+4*k+i] >> 2) & 3) - ml1
                        y[rt + k][ibl * QK_K + ib * 32 + i + 8] = dl1 * ((ql[ib*32+4*k+i] >> 4) & 3) - ml1
                        y[rt + k][ibl * QK_K + ib * 32 + i + 12] = dl1 * ((ql[ib*32+4*k+i] >> 6) & 3) - ml1
                        y[rt + k][ibl * QK_K + ib * 32 + i + 16] = dl2 * ((ql[ib*32+4*k+i + 16] >> 0) & 3) - ml2
                        y[rt + k][ibl * QK_K + ib * 32 + i + 20] = dl2 * ((ql[ib*32+4*k+i + 16] >> 2) & 3) - ml2
                        y[rt + k][ibl * QK_K + ib * 32 + i + 24] = dl2 * ((ql[ib*32+4*k+i + 16] >> 4) & 3) - ml2
                        y[rt + k][ibl * QK_K + ib * 32 + i + 28] = dl2 * ((ql[ib*32+4*k+i + 16] >> 6) & 3) - ml2

    return y.flatten()
def dequantize_iq4_xs_gpu(data: np.ndarray, device:str = "cuda", target_dtype = torch.get_default_dtype(),shape = None):
    block_size = GGML_BLOCK_SIZES["IQ4_XS"]
    ele_per_blk = GGML_ELEMENTS_PER_BLOCK["IQ4_XS"]
    device = torch.device(device)
    num_blocks = len(data) // block_size
    data = np.frombuffer(data, dtype=data.dtype)
    c_pointer = ctypes.addressof(ctypes.cast(data.ctypes.data, ctypes.POINTER(ctypes.c_int8)).contents)
    return KTransformersOps.dequantize_iq4_xs(c_pointer, data.size, block_size, ele_per_blk, device, target_dtype)
def dequantize_iq1_s_r4_gpu(data: np.ndarray, device:str = "cuda", target_dtype = torch.get_default_dtype(),shape = None):
    ele_per_blk = GGML_ELEMENTS_PER_BLOCK["IQ1_S_R4"]
    num_rows = shape[1]
    n_per_row = shape[0]
    row_meta_size = GGML_ROW_META_SIZE["IQ1_S_R4"]
    block_size = GGML_BLOCK_SIZES["IQ1_S_R4"]
    row_size = (n_per_row // 32) * block_size + row_meta_size
    data = np.frombuffer(data, dtype=data.dtype)
    device = torch.device(device)
    c_pointer = ctypes.addressof(ctypes.cast(data.ctypes.data, ctypes.POINTER(ctypes.c_int8)).contents)
    #result = dequantize_iq1_s_r4(data, shape)
    return KTransformersOps.dequantize_iq1_s_r4(c_pointer, data.size, block_size, ele_per_blk,  n_per_row,num_rows, row_size, device, target_dtype)
    #return result
def dequantize_q4_0(data,shape=None):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/a3c0188a4b5d3dec052ff87c9f773baa53631d70/src/ggml-quants.c#L1515
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/a3c0188a4b5d3dec052ff87c9f773baa53631d70/src/ggml-common.h#L141
    num_blocks = len(data) // GGML_BLOCK_SIZES["Q4_0"]

    scales = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, 1 + 8)[:, :1].astype(np.float32)
    qs = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, 2 + 16)[:, 2:]

    return np.concatenate([
        scales * ((qs & 0xf).astype(np.int8) - 8),
        scales * ((qs >> 4).astype(np.int8) - 8),
    ], axis=1)

def dequantize_q4_0_gpu(data, device:str = "cuda", target_dtype = torch.get_default_dtype(),shape = None):
    raise NotImplementedError()

def dequantize_q5_0(data,shape=None ):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/a3c0188a4b5d3dec052ff87c9f773baa53631d70/src/ggml-quants.c#L1556
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/a3c0188a4b5d3dec052ff87c9f773baa53631d70/src/ggml-common.h#L161
    num_blocks = len(data) // GGML_BLOCK_SIZES["Q5_0"]

    scales = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, 1 + 2 + 8)[:, :1].astype(np.float32)
    qh = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, 2 + 4 + 16)[:, 2:2 + 4]
    qs = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, 2 + 4 + 16)[:, 2 + 4:]

    bits = np.unpackbits(qh, axis=-1, bitorder="little")

    x0 = ((qs & 0xf).astype(np.int8) | (bits[:, :16] << 4)) - 16
    x1 = ((qs >> 4).astype(np.int8) | (bits[:, 16:] << 4)) - 16

    return np.concatenate([
        scales * x0,
        scales * x1,
    ], axis=1)

def dequantize_q5_0_gpu(data, device:str = "cuda", target_dtype = torch.get_default_dtype(),shape = None):
    result = dequantize_q5_0(data,shape)
    result = torch.from_numpy(result)
    return result
    raise NotImplementedError()

def dequantize_q8_0(data,shape=None):
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L43
    num_blocks = len(data) // GGML_BLOCK_SIZES["Q8_0"]

    scales = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, 1 + 16)[:, :1].astype(np.float32)
    qs = np.frombuffer(data, dtype=np.int8).reshape(num_blocks, 2 + 32)[:, 2:]
    return scales * qs

def dequantize_q8_0_gpu(data, device:str = "cuda", target_dtype = torch.get_default_dtype(),shape = None):
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L43
    
    block_size = GGML_BLOCK_SIZES["Q8_0"]
    ele_per_blk = GGML_ELEMENTS_PER_BLOCK["Q8_0"]
    device = torch.device(device)
    data = np.frombuffer(data, dtype=data.dtype)
    c_pointer = ctypes.addressof(ctypes.cast(data.ctypes.data, ctypes.POINTER(ctypes.c_int8)).contents)
    return KTransformersOps.dequantize_q8_0(c_pointer, data.size, block_size, ele_per_blk, device, target_dtype)


def dequantize_f32(data,shape=None):
    return np.frombuffer(data, dtype=np.float32)

def dequantize_f32_gpu(data, device, target_dtype = torch.get_default_dtype(),shape = None):
    data = np.frombuffer(data, dtype=np.float32)
    res = torch.from_numpy(data.copy())
    res_gpu = torch.empty_like(res, device=device, dtype=target_dtype)
    res_gpu.copy_(res)
    return res_gpu

def dequantize_f16(data,shape=None):
    return np.frombuffer(data, dtype=np.float16)

def dequantize_f16_gpu(data, device, target_dtype = torch.get_default_dtype(),shape = None):
    data = np.frombuffer(data, dtype=np.float16)
    res = torch.from_numpy(data.copy())
    res_gpu = torch.empty_like(res, device=device, dtype=target_dtype)
    res_gpu.copy_(res)
    return res_gpu

def dequantize_bf16_gpu(data, device, target_dtype = torch.get_default_dtype(),shape = None):
    data = np.frombuffer(data, dtype=np.float16)
    res = torch.from_numpy(data.copy())
    res_gpu = torch.empty_like(res, device=device)
    res_gpu.copy_(res)
    return res_gpu

GGML_DEQUANTIZE = {
    "F32": dequantize_f32,
    "F16": dequantize_f16,
    "BF16": dequantize_f16,
    "Q4_0": dequantize_q4_0,
    "Q5_0": dequantize_q5_0,
    "Q8_0": dequantize_q8_0,
    "Q2_K": dequantize_q2_k,
    "Q3_K": dequantize_q3_k,
    "Q4_K": dequantize_q4_k,
    "Q5_K": dequantize_q5_k,
    "Q6_K": dequantize_q6_k,
    "IQ4_XS": dequantize_iq4_xs,
    "IQ1_S_R4": dequantize_iq1_s_r4,
}

GGML_DEQUANTIZE_GPU = {
    "F32": dequantize_f32_gpu,
    "F16": dequantize_f16_gpu,
    "BF16": dequantize_bf16_gpu,
    "Q4_0": dequantize_q4_0_gpu,
    "Q5_0": dequantize_q5_0_gpu,
    "Q8_0": dequantize_q8_0_gpu,
    "Q2_K": dequantize_q2_k_gpu,
    "Q3_K": dequantize_q3_k_gpu,
    "Q4_K": dequantize_q4_k_gpu,
    "Q5_K": dequantize_q5_k_gpu,
    "Q6_K": dequantize_q6_k_gpu,
    "IQ4_XS": dequantize_iq4_xs_gpu,
    "IQ1_S_R4": dequantize_iq1_s_r4_gpu,
}


def translate_name_to_gguf_mixtral(name):
    
    replacement_template = {
        "w1.weight": "ffn_gate",
        "w2.weight": "ffn_down",
        "w3.weight": "ffn_up"
    }  

    pattern = re.compile(r"model.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w\d\.weight)")

    def replace_match(match):
        blk_id = match.group(1)
        expert_id = match.group(2)
        weight_type = match.group(3)
        if weight_type in replacement_template:
            return f"blk.{blk_id}.{replacement_template[weight_type]}.{expert_id}.weight"
        else:
            return match.group(0)

    new_name = re.sub(pattern, replace_match, name)
    
    return new_name

def translate_name_to_gguf(name):

    name = translate_name_to_gguf_mixtral(name)

    name = name.replace("lm_head.", "output.")
    name = name.replace("model.embed_tokens.", "token_embd.")
    name = name.replace("model.norm.", "output_norm.")
    
    name = name.replace("model.layers.", "blk.")
    name = name.replace(".input_layernorm", ".attn_norm")
    name = name.replace(".mlp.down_proj", ".ffn_down")
    name = name.replace(".mlp.gate_proj", ".ffn_gate")
    name = name.replace(".mlp.up_proj", ".ffn_up")
    name = name.replace(".post_attention_layernorm", ".ffn_norm")
    name = name.replace(".self_attn.q_proj", ".attn_q")
    name = name.replace(".self_attn.k_proj", ".attn_k")
    name = name.replace(".self_attn.v_proj", ".attn_v")
    name = name.replace(".self_attn.o_proj", ".attn_output")
    name = name.replace(".self_attn.qkv_proj", ".attn_qkv")
    name = name.replace(".self_attn.kv_a_proj_with_mqa", ".attn_kv_a_mqa")
    name = name.replace(".self_attn.kv_a_layernorm", ".attn_kv_a_norm")
    name = name.replace(".self_attn.kv_b_proj", ".attn_kv_b")
    name = name.replace(".self_attn.q_a_proj", ".attn_q_a")
    name = name.replace(".self_attn.q_a_layernorm", ".attn_q_a_norm")
    name = name.replace(".self_attn.q_b_proj", ".attn_q_b")

    name = name.replace(".self_attn.q_norm", ".attn_q_norm")
    name = name.replace(".self_attn.k_norm", ".attn_k_norm")
    
    name = name.replace(".shared_expert.", ".shared_experts.")
    name = name.replace(".shared_expert_", ".shared_experts_")
    name = name.replace(".gate_up_proj.", ".up_proj")
    
    name = name.replace(".mlp.shared_experts.down_proj", ".ffn_down_shexp")
    name = name.replace(".mlp.gate", ".ffn_gate_inp")
    name = name.replace(".mlp.shared_experts.gate_proj", ".ffn_gate_shexp")
    name = name.replace(".mlp.shared_experts.up_proj", ".ffn_up_shexp")
    name = name.replace(".mlp.shared_experts_gate", ".ffn_gate_inp_shexp")


    name = name.replace(".mlp.experts", "")

    
    name = name.replace(".block_sparse_moe.gate.", ".ffn_gate_inp.")
    name = name.replace(".block_sparse_moe.experts", "")
    
    name = name.replace(".feed_forward.experts", "")
    name = name.replace(".feed_forward.router", ".ffn_gate_inp")
    name = name.replace(".feed_forward.shared_experts.down_proj", ".ffn_down_shexp")
    name = name.replace(".feed_forward.shared_experts.gate_proj", ".ffn_gate_shexp")
    name = name.replace(".feed_forward.shared_experts.up_proj", ".ffn_up_shexp")

    return name


if __name__ == '__main__':
    gguf_path = '/mnt/data/model/DeepSeek-Coder-V2-GGUF-WJH'
    loader = GGUFLoader(gguf_path)
    loader.load_gguf_tensor('token_embd.weight')
