#!/usr/bin/env python3
"""One real DeepSeek-V3 KExpertsHybrid forward using released combined GGUF."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path


def passed(message: str) -> None:
    print(f"[PASS] {message}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--gguf-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    sys.path.insert(0, str(root))

    import torch
    import cpuinfer_ext
    from transformers import AutoConfig
    from ktransformers.operators.experts import KExpertsHybrid
    from ktransformers.util.custom_gguf import GGUFLoader

    started = time.perf_counter()
    torch.manual_seed(cfg["seed"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; the test will not silently fall back")
    device = torch.device(cfg["device"])
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    model_path = Path(args.model_path)
    gguf_path = Path(args.gguf_path)
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"model config missing: {model_path / 'config.json'}")
    if not gguf_path.exists():
        raise FileNotFoundError(f"GGUF path missing: {gguf_path}")

    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    loader = GGUFLoader(str(gguf_path))
    if cfg["tensor_key"] + ".ffn_gate_exps.weight" not in loader.tensor_info:
        raise KeyError(f"{cfg['tensor_key']} expert tensors are absent")
    KExpertsHybrid.CPU_INFER = cpuinfer_ext.CPUInfer(cfg["cpu_threads"])
    expert = KExpertsHybrid(
        key=cfg["tensor_key"], gguf_loader=loader, config=config,
        n_routed_experts=config.n_routed_experts, device=str(device), out_device=str(device)
    )
    expert.load(device=str(device), flex_topk=1, prefill_topk=3)
    if cfg["routing_mode"] != "static":
        raise ValueError("the DeepSeek-V3 functional test only accepts static mode")
    assert expert.w13.is_cuda and expert.w2.is_cuda
    assert expert.gate.__class__.__module__.startswith("numpy")
    passed("Combined GGUF loaded; WQ is on GPU and combined/WR backing is CPU mmap")

    expert.dynamic_topk = False
    expert.threshold_enabled = False
    expert.flex_decode_topk.fill_(6)
    expert.flex_decode_idx.fill_(6)
    expert.force_static_r(cuda_graph_idx=0)
    assert int(expert.flex_decode_topk.item()) == 1
    assert int(expert.flex_decode_idx.item()) == 1
    passed("Static forward guard restored decode topk/idx to r=1")

    tokens, top_k = cfg["tokens"], cfg["top_k"]
    x = torch.randn((tokens, config.hidden_size), dtype=torch.bfloat16, device=device)
    ids = torch.arange(top_k, dtype=torch.int64, device=device).repeat(tokens, 1)
    weights = torch.tensor(
        [[0.90, 0.70, 0.40, 0.20, 0.05, 0.01, 0.005, 0.001]],
        dtype=torch.float32, device=device
    ).repeat(tokens, 1)
    bsz = torch.tensor([tokens], dtype=torch.int32, device=device)
    def run_once(r_value: int):
        for name in ("flex_decode_topk", "flex_decode_idx"):
            tensor = getattr(expert, name)
            tensor.copy_(torch.tensor([r_value], device=tensor.device, dtype=tensor.dtype))
        expert.submit_for_one_decode(x, ids, weights, bsz, cuda_graph_idx=0)
        out = expert.sync_for_one_decode(cuda_graph_idx=0).clone()
        torch.cuda.synchronize(device)
        return out

    observations = {}
    for selected_r in cfg["static_r_values"]:
        t0 = time.perf_counter()
        output1 = run_once(selected_r)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        cpu_norm = float(KExpertsHybrid.output_cpu[0][:tokens].float().norm().item())
        gpu_norm = float(KExpertsHybrid.out_hidden_states[str(device)][0][:tokens].float().norm().item())
        output2 = run_once(selected_r)
        assert cpu_norm > 0 and gpu_norm > 0
        assert torch.isfinite(output1).all() and torch.isfinite(output2).all()
        torch.testing.assert_close(output1, output2, atol=0.0, rtol=0.0)
        observations[str(selected_r)] = {
            "critical_experts": selected_r,
            "gpu_experts": top_k - selected_r,
            "cpu_branch_norm": cpu_norm,
            "gpu_branch_norm": gpu_norm,
            "output_norm": float(output1.float().norm().item()),
            "single_forward_ms_observation": round(elapsed_ms, 3),
        }
        passed(f"Static Split Expert Routing r={selected_r}: {top_k-selected_r} GPU experts")
    passed("Critical CPU residual branch executed")
    passed("Non-critical GPU low-bit branch executed")
    passed("CPU/GPU outputs merged without NaN or silent fallback")
    passed("Repeated forward is deterministic")

    result = {
        "status": "PASS", "model": cfg["model"], "layer": cfg["layer"],
        "routing_mode": "static", "observations": observations,
        "peak_vram_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 2),
        "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "performance_is_acceptance_criterion": False
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    passed("RapidMoE functional test completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
