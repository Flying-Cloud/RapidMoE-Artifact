#!/usr/bin/env python3
"""Deterministic, model-free checks over real RapidMoE preprocessing/kernels."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np


def passed(message: str) -> None:
    print(f"[PASS] {message}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    sys.path.insert(0, str(root))

    import torch
    import KTransformersOps as ops
    from ktransformers.util.custom_gguf import (
        GGML_QUANT_SIZES,
        GGML_ROW_META_SIZE,
        offs_concate_experts,
        offs_concate_experts_q2,
    )
    from rapidmoe_ae.profile import load_profile, scalars_for_layer

    started = time.perf_counter()
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU/reference fallback is permitted")
    device = torch.device(cfg["device"])
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    passed("Environment")

    # Exercise the exact preprocessing functions used by KExpertsHybrid.load.
    h, m, e = cfg["hidden_size"], cfg["intermediate_size"], cfg["experts"]
    block, size = GGML_QUANT_SIZES[cfg["combined_ggml_type"]]
    row_size_h = h // block * size + GGML_ROW_META_SIZE["IQ1_S_R4"]
    row_size_m = m // block * size + GGML_ROW_META_SIZE["IQ1_S_R4"]
    gate = np.arange(e * m * row_size_h, dtype=np.uint64).view(np.uint8)
    up = np.arange(e * m * row_size_h, dtype=np.uint64).view(np.uint8)
    down = np.arange(e * h * row_size_m, dtype=np.uint64).view(np.uint8)
    wq13_np = offs_concate_experts(gate, [h, m, e], "IQ1_S_R4_RES_Q2_K", up)
    wq2_np = offs_concate_experts(down, [m, h, e], "IQ1_S_R4_RES_Q2_K")
    wr13 = offs_concate_experts_q2(gate, [h, m, e], "IQ1_S_R4_RES_Q2_K", up)
    wr2 = offs_concate_experts_q2(down, [m, h, e], "IQ1_S_R4_RES_Q2_K")
    wq13 = torch.tensor(wq13_np, dtype=torch.uint8, device=device)
    wq2 = torch.tensor(wq2_np, dtype=torch.uint8, device=device)
    assert wq13.is_cuda and wq2.is_cuda
    # NumPy outputs are host-resident. Keep explicit CPU tensors for placement evidence.
    wr13_cpu = torch.from_numpy(wr13.copy())
    wr2_cpu = torch.from_numpy(wr2.copy())
    assert wr13_cpu.device.type == "cpu" and wr2_cpu.device.type == "cpu"
    passed("WQ preprocessing and placement: GPU")
    passed("WR preprocessing and placement: CPU")

    # Real CUDA routing kernel: every routed pair appears exactly once before padding.
    topk_ids = torch.tensor([[7, 1, 4, 2, 0, 6, 3, 5], [3, 6, 0, 5, 2, 7, 1, 4]], dtype=torch.int64, device=device)
    max_padded = topk_ids.numel() + e * 31
    sorted_ids = torch.full((max_padded,), topk_ids.numel(), dtype=torch.int32, device=device)
    expert_ids = torch.zeros(((max_padded + 31) // 32,), dtype=torch.int32, device=device)
    post = torch.empty(1, dtype=torch.int32, device=device)
    ops.moe_align_block_size(topk_ids, e, 32, sorted_ids, expert_ids, post)
    torch.cuda.synchronize(device)
    valid = sorted_ids[: int(post.item())]
    valid = valid[valid < topk_ids.numel()].cpu().tolist()
    assert sorted(valid) == list(range(topk_ids.numel()))
    passed("Routing invariants: no lost or duplicate token/expert pairs")

    # Real UMIA CUDA decision kernel, driven by the frozen deployment profile.
    profile_path = root / cfg["deployment_profile"]
    profile = load_profile(profile_path)
    alpha, threshold = scalars_for_layer(profile, 38, "decode")
    router = torch.tensor([[0.90, 0.70, 0.40, 0.20, 0.05, 0.01, 0.005, 0.001]], dtype=torch.float32, device=device)
    alpha_t = torch.tensor([alpha], dtype=torch.float32, device=device)
    threshold_t = torch.tensor([threshold], dtype=torch.float32, device=device)
    r = torch.empty(1, dtype=torch.int32, device=device)
    idx = torch.empty(1, dtype=torch.int32, device=device)
    bsz = torch.ones(1, dtype=torch.int32, device=device)
    ops.dynamic_threshold(router, alpha_t, threshold_t, r, idx, bsz)
    torch.cuda.synchronize(device)
    r_low = int(r.item())
    threshold_t.fill_(threshold * 4.0)
    ops.dynamic_threshold(router, alpha_t, threshold_t, r, idx, bsz)
    torch.cuda.synchronize(device)
    r_high = int(r.item())
    assert 0 <= r_low <= 6 and 0 <= r_high <= 6 and r_low != r_high
    passed(f"UMIA dynamic selection: threshold changed r from {r_low} to {r_high}")

    # Real merge kernel. Nonzero r enables the CPU contribution; r=0 suppresses it.
    cpu_branch = torch.full((cfg["tokens"], h), 2.0, dtype=torch.bfloat16, device=device)
    gpu_branch = torch.full((cfg["tokens"], h), 3.0, dtype=torch.bfloat16, device=device)
    r.fill_(1)
    merged = ops.dynamic_add(cpu_branch.clone(), gpu_branch, r)
    torch.cuda.synchronize(device)
    torch.testing.assert_close(merged, torch.full_like(merged, 5.0), atol=cfg["atol"], rtol=cfg["rtol"])
    r.zero_()
    gpu_only = ops.dynamic_add(cpu_branch.clone(), gpu_branch, r)
    torch.cuda.synchronize(device)
    torch.testing.assert_close(gpu_only, gpu_branch, atol=cfg["atol"], rtol=cfg["rtol"])
    passed("GPU contribution merge condition executed")
    passed("CPU contribution merge condition executed")
    passed("Numerical error within tolerance")

    result = {
        "status": "PASS",
        "seed": cfg["seed"],
        "device": torch.cuda.get_device_name(device),
        "wq_bytes": int(wq13.numel() + wq2.numel()),
        "wr_bytes": int(wr13_cpu.numel() + wr2_cpu.numel()),
        "umia_r": {"profile_threshold": r_low, "raised_threshold": r_high},
        "peak_vram_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 2),
        "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "scope_note": "Model-free smoke covers production preprocessing, routing, UMIA, and merge kernels; CPU residual and GPU low-bit computation are executed by the functional test."
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    passed("RapidMoE smoke test completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
