#!/usr/bin/env python3
"""Branch tests for the production six-argument dynamic-threshold binding."""

from __future__ import annotations

import torch
import KTransformersOps as ops


def expected(weights: list[float], alpha: float, threshold: float) -> int:
    scores = [alpha * value for value in weights]
    if scores[0] < threshold:
        return 0
    if scores[4] > threshold:
        return 6
    for result, index in ((1, 1), (2, 2), (3, 3), (4, 4)):
        if scores[index - 1] > threshold and scores[index] < threshold:
            return result
    return 1


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    cases = [
        ([0.90, 0.70, 0.40, 0.20, 0.05, 0.01], 0.01, 0.02),
        ([0.90, 0.70, 0.40, 0.20, 0.05, 0.01], 0.10, 0.08),
        ([0.90, 0.70, 0.40, 0.20, 0.05, 0.01], 0.10, 0.06),
        ([0.90, 0.70, 0.40, 0.20, 0.05, 0.01], 0.10, 0.03),
        ([0.90, 0.70, 0.40, 0.20, 0.05, 0.01], 0.10, 0.015),
        ([0.90, 0.70, 0.40, 0.20, 0.05, 0.01], 0.10, 0.004),
    ]
    for weights, alpha, threshold in cases:
        topk = torch.tensor([weights], dtype=torch.float32, device="cuda")
        out_r = torch.empty(1, dtype=torch.int32, device="cuda")
        out_idx = torch.empty(1, dtype=torch.int32, device="cuda")
        bsz = torch.ones(1, dtype=torch.int32, device="cuda")
        ops.dynamic_threshold(
            topk,
            torch.tensor([alpha], dtype=torch.float32, device="cuda"),
            torch.tensor([threshold], dtype=torch.float32, device="cuda"),
            out_r, out_idx, bsz,
        )
        torch.cuda.synchronize()
        want = expected(weights, alpha, threshold)
        assert int(out_r.item()) == want and int(out_idx.item()) == want
    print("[PASS] dynamic threshold binding accepts bsz_tensor and covers r={0,1,2,3,4,6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
