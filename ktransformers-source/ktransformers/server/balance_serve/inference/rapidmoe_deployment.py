"""Read-only RapidMoE deployment configuration for balance-serve.

The sole public model is DeepSeek-V3.  It can run either static r=1 or dynamic
selection from a frozen profile.  This module contains no profile generation,
quality comparison, search, or dataset handling.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


def load_v3_profile(path: str | Path) -> dict:
    profile_path = Path(path).expanduser().resolve()
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "model", "mode", "layer_range", "threshold",
        "phase_scale", "layer_coefficient", "mutable",
    }
    missing = required.difference(profile)
    if missing:
        raise ValueError(f"RapidMoE profile is missing fields: {sorted(missing)}")
    if profile["mode"] != "dynamic" or not profile["model"].lower().startswith("deepseek-v3"):
        raise ValueError("dynamic mode accepts only a DeepSeek-V3 deployment profile")
    if profile["mutable"] is not False:
        raise ValueError("deployment profile must declare mutable=false")
    lo, hi = profile["layer_range"]
    if [lo, hi] != [3, 60] or len(profile["layer_coefficient"]) != hi - lo + 1:
        raise ValueError("DeepSeek-V3 profile must cover exactly layers 3..60")
    for phase in ("prefill", "decode"):
        if float(profile["threshold"][phase]) <= 0 or float(profile["phase_scale"][phase]) <= 0:
            raise ValueError(f"invalid {phase} deployment scalar")
    return profile


def _copy_scalar(tensor: torch.Tensor, value: float | int) -> None:
    tensor.copy_(torch.tensor([value], device=tensor.device, dtype=tensor.dtype))


def configure_rapidmoe_layers(layers: list, mode: str, profile_path: str | None) -> dict:
    """Configure already-loaded KExpertsHybrid layers and return an audit record."""
    if len(layers) != 58:
        raise ValueError(f"expected 58 DeepSeek MoE layers, got {len(layers)}")
    if mode == "static":
        if profile_path:
            raise ValueError("static V3 mode must not receive a deployment profile")
        for expert in layers:
            for name in ("flex_decode_topk", "flex_decode_idx", "flex_prefill_topk", "flex_prefill_idx"):
                _copy_scalar(getattr(expert, name), 1)
            _copy_scalar(expert.static_r_value, 1)
            # The prefill CUDA Graph path reads these Python-side split values
            # while capturing, so bind them to the same static r=1 contract.
            expert.flex_topk = 1
            expert.prefill_topk = 1
            expert.prefill_topk_phase1 = 1
            expert.prefill_topk_phase2 = 0
            expert.dynamic_topk = False
            expert.threshold_enabled = False
        return {
            "mode": "static", "model": "DeepSeek-V3", "r": 1, "layers": 58,
            "dynamic_topk": False, "threshold_enabled": False,
            "static_guard": "per-forward-device-copy",
        }

    if mode != "dynamic":
        raise ValueError("rapidmoe_mode must be static or dynamic")
    if not profile_path:
        raise ValueError("dynamic V3 mode requires --rapidmoe_profile")
    profile = load_v3_profile(profile_path)
    lo, hi = profile["layer_range"]
    for layer_id, expert in zip(range(lo, hi + 1), layers):
        coefficient = float(profile["layer_coefficient"][layer_id - lo])
        for phase in ("prefill", "decode"):
            alpha = coefficient * float(profile["phase_scale"][phase])
            _copy_scalar(getattr(expert, f"{phase}_alpha"), alpha)
            _copy_scalar(getattr(expert, f"{phase}_thre"), float(profile["threshold"][phase]))
        expert.dynamic_topk = True
        expert.threshold_enabled = True
    return {
        "mode": "dynamic", "model": profile["model"], "layers": 58,
        "layer_range": [lo, hi], "threshold": profile["threshold"],
        "profile": str(Path(profile_path).expanduser().resolve()),
    }
