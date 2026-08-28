#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
mkdir -p "$AE_ROOT/results"
"$AE_PYTHON" "$AE_ROOT/tests/ae/test_smoke.py" \
  --config "$AE_ROOT/configs/ae/smoke.yaml" \
  --output "$AE_ROOT/results/smoke.json"
