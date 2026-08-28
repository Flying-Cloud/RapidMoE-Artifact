#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
: "${RAPIDMOE_MODEL_PATH:?Set RAPIDMOE_MODEL_PATH to the DeepSeek-V3 config/tokenizer directory}"
: "${RAPIDMOE_GGUF_PATH:?Set RAPIDMOE_GGUF_PATH to the DeepSeek-V3 RESplit GGUF directory}"
export RAPIDMOE_STATE_DIR="${RAPIDMOE_STATE_DIR:-$AE_ROOT/results/server_state_v3}"
mkdir -p "$RAPIDMOE_STATE_DIR"
exec "$AE_PYTHON" -m ktransformers.server.main \
  --model_name deepseek-v3 \
  --port "${RAPIDMOE_PORT:-10002}" \
  --model_path "$RAPIDMOE_MODEL_PATH" \
  --gguf_path "$RAPIDMOE_GGUF_PATH" \
  --optimize_config_path "$KT_ROOT/ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat-Hybrid-multi-gpu-serve.yaml" \
  --max_new_tokens "${RAPIDMOE_MAX_NEW_TOKENS:-64}" \
  --cpu_infer "${RAPIDMOE_CPU_THREADS:-48}" \
  --cache_lens "${RAPIDMOE_CACHE_LENS:-4096}" \
  --chunk_size "${RAPIDMOE_CHUNK_SIZE:-256}" \
  --max_batch_size "${RAPIDMOE_MAX_BATCH_SIZE:-4}" \
  --backend_type balance_serve \
  --rapidmoe_mode dynamic \
  --rapidmoe_profile "$AE_ROOT/configs/ae/deepseek_v3_deployment_profile.json"
