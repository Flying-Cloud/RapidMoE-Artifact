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

rapidmoe_visible_physical_cores() {
  lscpu -p=CORE,SOCKET 2>/dev/null \
    | awk -F, '!/^#/ {seen[$1 FS $2]=1} END {print length(seen)}'
}

rapidmoe_default_cpu_threads() {
  local physical_cores threads
  physical_cores="$(rapidmoe_visible_physical_cores)"
  [[ "$physical_cores" =~ ^[1-9][0-9]*$ ]] || return 1
  threads=$((physical_cores - 8))
  (( threads > 0 )) || threads=1
  printf '%d\n' "$threads"
}

if [[ -n "${RAPIDMOE_CUDA_HOME:-}" ]]; then
  export CUDA_HOME="$RAPIDMOE_CUDA_HOME"
  export PATH="$CUDA_HOME/bin:$PATH"
fi

if [[ -n "${RAPIDMOE_LD_PRELOAD:-}" ]]; then
  export LD_PRELOAD="$RAPIDMOE_LD_PRELOAD${LD_PRELOAD:+:$LD_PRELOAD}"
fi

export PYTHONPATH="$AE_ROOT${KT_ROOT:+:$KT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
