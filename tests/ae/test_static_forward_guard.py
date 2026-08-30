#!/usr/bin/env python3
"""Verify the capture-safe static r=2 device-copy guard."""

import torch

from ktransformers.operators.experts import KExpertsHybrid


if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required; static graph guard must not silently fall back")

device = torch.device("cuda:0")
expert = object.__new__(KExpertsHybrid)
expert.static_r_value = torch.full((1,), 2, dtype=torch.int32, device=device)
expert.flex_decode_topk = torch.tensor([6], dtype=torch.int32, device=device)
expert.flex_decode_idx = torch.tensor([6], dtype=torch.int32, device=device)
expert.flex_prefill_topk = torch.tensor([6], dtype=torch.int32, device=device)
expert.flex_prefill_idx = torch.tensor([6], dtype=torch.int32, device=device)

expert.force_static_r(0)
assert expert.flex_decode_topk.item() == 2
assert expert.flex_decode_idx.item() == 2
assert expert.flex_prefill_topk.item() == 6

expert.force_static_r(4)
assert expert.flex_prefill_topk.item() == 2
assert expert.flex_prefill_idx.item() == 2

graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    expert.force_static_r(0)
torch.cuda.synchronize(device)
expert.flex_decode_topk.fill_(6)
expert.flex_decode_idx.fill_(6)
graph.replay()
torch.cuda.synchronize(device)
assert expert.flex_decode_topk.item() == 2
assert expert.flex_decode_idx.item() == 2

print("[PASS] static decode/prefill guards copy device-side r=2 in place")
print("[PASS] static r=2 device copies execute on every CUDA Graph replay")
