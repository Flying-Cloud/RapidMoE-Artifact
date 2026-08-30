"""RapidMoE deployment modes and read-only UMIA profile application.

This module intentionally contains no calibration, KL-divergence measurement,
or frontier-search implementation.  It only consumes a frozen deployment
profile produced before artifact packaging.
"""

from __future__ import annotations

import json
from pathlib import Path


STATIC_MODE = "static"
DYNAMIC_MODE = "dynamic"


def load_profile(path: str | Path) -> dict:
    profile = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "schema_version", "model", "mode", "layer_range", "threshold",
        "phase_scale", "layer_coefficient", "mutable",
    }
    missing = required.difference(profile)
    if missing:
        raise ValueError(f"deployment profile is missing: {sorted(missing)}")
    lo, hi = profile["layer_range"]
    if len(profile["layer_coefficient"]) != hi - lo + 1:
        raise ValueError("layer_coefficient length does not match layer_range")
    if profile.get("mutable", True):
        raise ValueError("AE accepts only read-only deployment profiles (mutable=false)")
    if profile["mode"] != DYNAMIC_MODE:
        raise ValueError("a UMIA deployment profile must use mode=dynamic")
    if not str(profile["model"]).lower().startswith("deepseek-v3"):
        raise ValueError("the AE dynamic profile is restricted to DeepSeek-V3")
    for phase in ("prefill", "decode"):
        if phase not in profile["threshold"] or phase not in profile["phase_scale"]:
            raise ValueError(f"deployment profile has no {phase} scalar")
    return profile


def scalars_for_layer(profile: dict, layer: int, phase: str) -> tuple[float, float]:
    lo, hi = profile["layer_range"]
    if not lo <= layer <= hi:
        raise ValueError(f"layer {layer} is outside [{lo}, {hi}]")
    if phase not in ("prefill", "decode"):
        raise ValueError("phase must be prefill or decode")
    alpha = float(profile["layer_coefficient"][layer - lo]) * float(profile["phase_scale"][phase])
    threshold = float(profile["threshold"][phase])
    return alpha, threshold


def apply_to_expert(expert, profile: dict, layer: int) -> None:
    """Copy profile scalars to an already-loaded production KExpertsHybrid."""
    import torch

    for phase in ("prefill", "decode"):
        alpha, threshold = scalars_for_layer(profile, layer, phase)
        getattr(expert, f"{phase}_alpha").copy_(
            torch.tensor([alpha], device=expert.out_device, dtype=torch.float32)
        )
        getattr(expert, f"{phase}_thre").copy_(
            torch.tensor([threshold], device=expert.out_device, dtype=torch.float32)
        )
    expert.threshold_enabled = True


def apply_static(expert, r: int = 2) -> None:
    """Configure the DeepSeek-V3 AE path with a fixed split point."""
    if r != 2:
        raise ValueError("the public DeepSeek-V3 static mode is fixed at r=2")
    import torch

    for name in ("flex_decode_topk", "flex_decode_idx", "flex_prefill_topk", "flex_prefill_idx"):
        tensor = getattr(expert, name)
        tensor.copy_(torch.tensor([r], device=tensor.device, dtype=tensor.dtype))
    expert.threshold_enabled = False


def configure_layer_experts(experts: list, mode: str, profile: dict | None = None) -> None:
    """Apply exactly one public AE mode to an ordered list of MoE layers."""
    if mode == STATIC_MODE:
        if profile is not None:
            raise ValueError("DeepSeek-V3 static mode must not load a UMIA profile")
        for expert in experts:
            apply_static(expert)
        return
    if mode != DYNAMIC_MODE:
        raise ValueError("mode must be static or dynamic")
    if profile is None:
        raise ValueError("DeepSeek-V3 dynamic mode requires a frozen profile")
    lo, hi = profile["layer_range"]
    if len(experts) != hi - lo + 1:
        raise ValueError("expert-layer count does not match deployment profile")
    for layer, expert in zip(range(lo, hi + 1), experts):
        apply_to_expert(expert, profile, layer)
