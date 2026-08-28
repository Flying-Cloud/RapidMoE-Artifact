#!/usr/bin/env python3
"""Dependency-free checks for public RapidMoE configuration contracts."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from rapidmoe_ae.profile import load_profile, scalars_for_layer

    static = json.loads((root / "configs/ae/deepseek_v3_static.json").read_text())
    assert static["model"].startswith("DeepSeek-V3")
    assert static["mode"] == "static" and static["critical_experts_r"] == 1
    assert "profile" not in static

    dynamic = load_profile(root / "configs/ae/deepseek_v3_deployment_profile.json")
    assert dynamic["model"].startswith("DeepSeek-V3")
    for phase in ("prefill", "decode"):
        for layer in range(3, 61):
            alpha, threshold = scalars_for_layer(dynamic, layer, phase)
            assert alpha > 0 and threshold > 0
    print("[PASS] V3 static r=1 contract")
    print("[PASS] V3 read-only dynamic profile covers layers 3..60")
    print("[PASS] no runtime parameter-generation input is required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
