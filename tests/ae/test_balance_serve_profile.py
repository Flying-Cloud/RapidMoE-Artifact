#!/usr/bin/env python3
"""Unit checks for automatic balance-serve deployment configuration."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch


class MockExpert:
    def __init__(self):
        for name in (
            "flex_decode_topk", "flex_decode_idx", "flex_prefill_topk", "flex_prefill_idx",
        ):
            setattr(self, name, torch.zeros(1, dtype=torch.int32))
        self.static_r_value = torch.ones(1, dtype=torch.int32)
        for name in ("decode_alpha", "prefill_alpha", "decode_thre", "prefill_thre"):
            setattr(self, name, torch.zeros(1, dtype=torch.float32))
        self.dynamic_topk = None
        self.threshold_enabled = None


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    module_path = root / "ktransformers-source/ktransformers/server/balance_serve/inference/rapidmoe_deployment.py"
    spec = importlib.util.spec_from_file_location("rapidmoe_deployment", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    static_layers = [MockExpert() for _ in range(58)]
    record = module.configure_rapidmoe_layers(static_layers, "static", None)
    assert record["r"] == 1
    assert record["dynamic_topk"] is False
    assert record["threshold_enabled"] is False
    assert record["static_guard"] == "per-forward-device-copy"
    assert record["model"] == "DeepSeek-V3"
    for expert in static_layers:
        assert not expert.dynamic_topk and not expert.threshold_enabled
        assert expert.flex_topk == expert.prefill_topk == expert.prefill_topk_phase1 == 1
        assert expert.prefill_topk_phase2 == 0
        assert int(expert.static_r_value.item()) == 1
        assert all(int(getattr(expert, name).item()) == 1 for name in (
            "flex_decode_topk", "flex_decode_idx", "flex_prefill_topk", "flex_prefill_idx",
        ))

    path = root / "configs/ae/deepseek_v3_deployment_profile.json"
    profile = json.loads(path.read_text())
    dynamic_layers = [MockExpert() for _ in range(58)]
    record = module.configure_rapidmoe_layers(dynamic_layers, "dynamic", str(path))
    assert record["layer_range"] == [3, 60]
    for offset, expert in enumerate(dynamic_layers):
        coefficient = profile["layer_coefficient"][offset]
        assert expert.dynamic_topk and expert.threshold_enabled
        for phase in ("prefill", "decode"):
            want_alpha = coefficient * profile["phase_scale"][phase]
            assert abs(float(getattr(expert, f"{phase}_alpha").item()) - want_alpha) < 1e-7
            assert abs(float(getattr(expert, f"{phase}_thre").item()) - profile["threshold"][phase]) < 1e-7
    print("[PASS] balance-serve V3 static configuration")
    print("[PASS] balance-serve V3 profile auto-application for 58 layers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
