#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
: "${RAPIDMOE_MODEL_PATH:?Set RAPIDMOE_MODEL_PATH to the DeepSeek config/tokenizer directory}"
: "${RAPIDMOE_GGUF_PATH:?Set RAPIDMOE_GGUF_PATH to the layer-38 RESplit GGUF}"
: "${RAPIDMOE_BASELINE_GGUF_PATH:?Set RAPIDMOE_BASELINE_GGUF_PATH to the layer-38 Q4_K_M GGUF}"
mkdir -p "$AE_ROOT/results"
"$AE_PYTHON" "$AE_ROOT/tests/ae/benchmark_cpu_baseline.py" \
  --config "$AE_ROOT/configs/ae/cpu_baseline_benchmark.yaml" \
  --model-path "$RAPIDMOE_MODEL_PATH" \
  --rapidmoe-gguf-path "$RAPIDMOE_GGUF_PATH" \
  --baseline-gguf-path "$RAPIDMOE_BASELINE_GGUF_PATH" \
  --output "$AE_ROOT/results/cpu_baseline_benchmark.json"
