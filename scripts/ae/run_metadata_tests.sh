#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
"$AE_PYTHON" "$AE_ROOT/tests/ae/test_profile.py"
"$AE_PYTHON" "$AE_ROOT/tests/ae/test_model_deposit.py"
"$AE_PYTHON" "$AE_ROOT/tests/ae/test_model_downloader.py"
"$AE_PYTHON" "$AE_ROOT/tests/ae/test_one_layer_downloader.py"
if [[ -n "${RAPIDMOE_GGUF_FILE:-}" ]]; then
  "$AE_PYTHON" "$AE_ROOT/scripts/ae/inspect_checkpoint.py" "$RAPIDMOE_GGUF_FILE"
else
  echo "[SKIP] checkpoint header: set RAPIDMOE_GGUF_FILE to enable"
fi
