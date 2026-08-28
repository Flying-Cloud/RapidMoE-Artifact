#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
: "${RAPIDMOE_MODEL_PATH:?Set RAPIDMOE_MODEL_PATH to a DeepSeek-V3 config/tokenizer directory}"
: "${RAPIDMOE_GGUF_PATH:?Set RAPIDMOE_GGUF_PATH to the layer-38 RESplit GGUF}"
mkdir -p "$AE_ROOT/results"
"$AE_PYTHON" "$AE_ROOT/tests/ae/test_functional.py" \
  --config "$AE_ROOT/configs/ae/functional.yaml" \
  --model-path "$RAPIDMOE_MODEL_PATH" \
  --gguf-path "$RAPIDMOE_GGUF_PATH" \
  --output "$AE_ROOT/results/functional.json"
