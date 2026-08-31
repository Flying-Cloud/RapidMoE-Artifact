#!/usr/bin/env bash
set -euo pipefail

AE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -d "$AE_ROOT/ktransformers-source" ]]; then
  default_kt_root="$AE_ROOT/ktransformers-source"
else
  default_kt_root="$(cd "$AE_ROOT/../../llm/ktransformers" 2>/dev/null && pwd || true)"
fi
KT_ROOT="${RAPIDMOE_KTRANSFORMERS_ROOT:-$default_kt_root}"
AE_PYTHON="${RAPIDMOE_PYTHON:-python3}"

if [[ -n "${RAPIDMOE_CUDA_HOME:-}" ]]; then
  export CUDA_HOME="$RAPIDMOE_CUDA_HOME"
  export PATH="$CUDA_HOME/bin:$PATH"
fi

if [[ -n "${RAPIDMOE_LD_PRELOAD:-}" ]]; then
  export LD_PRELOAD="$RAPIDMOE_LD_PRELOAD${LD_PRELOAD:+:$LD_PRELOAD}"
fi

export PYTHONPATH="$AE_ROOT${KT_ROOT:+:$KT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
