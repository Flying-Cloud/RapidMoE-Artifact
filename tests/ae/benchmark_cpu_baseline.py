#!/usr/bin/env python3
"""Layer-38 KExpertsCPU versus RapidMoE CUDA Graph speed benchmark."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--rapidmoe-gguf-path", required=True)
    parser.add_argument("--baseline-gguf-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    cfg = json.loads(Path(args.config).read_text())
    sys.path.insert(0, str(root / "ktransformers-source"))

    import torch
    import cpuinfer_ext
    from transformers import AutoConfig
    from ktransformers.operators.experts import KExpertsCPU, KExpertsHybrid
    from ktransformers.util.custom_gguf import GGUFLoader

    torch.manual_seed(cfg["seed"])
    device = torch.device(cfg["device"])
    torch.cuda.set_device(device)
    model_cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    baseline_loader = GGUFLoader(args.baseline_gguf_path)
    rapidmoe_loader = GGUFLoader(args.rapidmoe_gguf_path)
    pool = cpuinfer_ext.CPUInfer(cfg["cpu_threads"])
    KExpertsCPU.CPU_INFER = pool
    KExpertsHybrid.CPU_INFER = pool
    baseline = KExpertsCPU(
        key=cfg["tensor_key"], gguf_loader=baseline_loader, config=model_cfg,
        n_routed_experts=model_cfg.n_routed_experts, device="cpu", out_device=str(device),
    )
    baseline.load(device="cpu")
    hybrid = KExpertsHybrid(
        key=cfg["tensor_key"], gguf_loader=rapidmoe_loader, config=model_cfg,
        n_routed_experts=model_cfg.n_routed_experts, device=str(device), out_device=str(device),
    )
    hybrid.load(device=str(device), flex_topk=1, prefill_topk=1)

    def inputs(tokens: int):
        x = torch.randn((tokens, model_cfg.hidden_size), dtype=torch.bfloat16, device=device)
        ids = torch.arange(cfg["top_k"], dtype=torch.int64, device=device).repeat(tokens, 1)
        weights = torch.tensor(
            [[0.90, 0.70, 0.40, 0.20, 0.05, 0.01, 0.005, 0.001]],
            dtype=torch.float32, device=device,
        ).repeat(tokens, 1)
        bsz = torch.tensor([tokens], dtype=torch.int32, device=device)
        return x, ids, weights, bsz

    def set_r(r_value: int, phase: str):
        names = ("flex_decode_topk", "flex_decode_idx") if phase == "decode" else ("flex_prefill_topk", "flex_prefill_idx")
        for name in names:
            tensor = getattr(hybrid, name)
            tensor.copy_(torch.tensor([r_value], device=tensor.device, dtype=tensor.dtype))

    def cpu_decode(data):
        baseline.submit_for_one_decode(*data, cuda_graph_idx=0)
        return baseline.sync_for_one_decode(cuda_graph_idx=0)

    def hybrid_decode(data):
        hybrid.submit_for_one_decode(*data, cuda_graph_idx=0)
        out = hybrid.sync_for_one_decode(cuda_graph_idx=0).clone()
        return out[: data[0].shape[0]]

    def capture_and_measure(call, replays_per_trial: int, trials: int, warmups: int):
        for _ in range(warmups):
            call()
        torch.cuda.synchronize(device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        torch.cuda.synchronize(device)

        # Prime the captured graph before collecting independent trial means.
        graph.replay()
        torch.cuda.synchronize(device)
        trial_mean_ms = []
        for _ in range(trials):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(replays_per_trial):
                graph.replay()
            end_event.record()
            end_event.synchronize()
            trial_mean_ms.append(start_event.elapsed_time(end_event) / replays_per_trial)

        return trial_mean_ms

    def graph_stats(samples):
        mean_ms = statistics.fmean(samples)
        return {
            "mean_ms": mean_ms,
            "stddev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
            "min_ms": min(samples),
            "max_ms": max(samples),
            "trial_mean_ms": samples,
            "tokens_per_second": cfg["decode_tokens"] / (mean_ms / 1000),
        }

    report = {
        "status": "PASS", "model": cfg["model"], "layer": cfg["layer"],
        "weights": "AE layer-38 GGUFs (Q4_K_M and RESplit)",
        "baseline": "KExpertsCPU with layer-38 Q4_K_M",
        "candidate": "RapidMoE with layer-38 RESplit",
        "protocol": {
            "timed_phase": "decode",
            "timing": "CUDA events around CUDA Graph replay",
            "warmup_iterations": cfg["warmup_iterations"],
            "graph_replays_per_trial": cfg["graph_replays_per_trial"],
            "graph_replay_trials": cfg["graph_replay_trials"],
            "estimator": "arithmetic mean of per-replay trial means",
            "reference": "gguf_test.py::test_kexpertshybrid_decode_time",
        },
        "phases": {},
    }

    decode_data = inputs(cfg["decode_tokens"])
    cpu_samples = capture_and_measure(
        lambda: cpu_decode(decode_data), cfg["graph_replays_per_trial"],
        cfg["graph_replay_trials"], cfg["warmup_iterations"],
    )
    cpu_stats = graph_stats(cpu_samples)
    decode_report = {
        "tokens": cfg["decode_tokens"],
        "KExpertsCPU": cpu_stats,
        "RapidMoE": {},
    }
    for r_value in cfg["static_r_values"]:
        # Routing configuration is fixed before capture. Tensor creation or
        # mutation inside a CUDA Graph capture is intentionally forbidden.
        set_r(r_value, "decode")
        samples = capture_and_measure(
            lambda: hybrid_decode(decode_data),
            cfg["graph_replays_per_trial"], cfg["graph_replay_trials"],
            cfg["warmup_iterations"],
        )
        stats = graph_stats(samples)
        stats.update({
            "speedup_vs_cpu_mean": cpu_stats["mean_ms"] / stats["mean_ms"],
        })
        decode_report["RapidMoE"][str(r_value)] = stats
    report["phases"]["decode"] = decode_report

    report["performance_is_acceptance_criterion"] = cfg["performance_is_acceptance_criterion"]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("[PASS] repeated CUDA Graph replay decode comparison recorded")
    print("[PASS] KExpertsCPU and RapidMoE speed statistics recorded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
