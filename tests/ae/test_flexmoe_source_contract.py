#!/usr/bin/env python3
"""Dependency-free regression check for the hybrid decode residual contract."""

from __future__ import annotations

import re
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    source = root / "ktransformers-source/csrc/ktransformers_ext/operators/llamafile/flexmoe.cpp"
    text = source.read_text(encoding="utf-8")
    marker = "void FlexMOE::forward_flex("
    assert text.count(marker) == 1, "expected exactly one FlexMOE::forward_flex definition"
    body = text.split(marker, 1)[1].split("\nvoid FlexMOE::", 1)[0]
    residual_modes = re.findall(r"(?:int\s+)?res\s*=\s*([01])\s*;", body)
    assert residual_modes == ["1", "0"], residual_modes
    assert body.count("iqk_mul_mat_ik_offs") == 3
    assert "s_local_gate_gpu_input_" in body and "s_local_up_gpu_input_" in body
    print("[PASS] FlexMOE decode computes gate/up residual only and full down projection")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
