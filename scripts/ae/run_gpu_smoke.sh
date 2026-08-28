#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

output_dir="$AE_ROOT/results/gpu-smoke"
mkdir -p "$output_dir"

"$AE_PYTHON" "$AE_ROOT/scripts/ae/capture_environment.py" \
  --check gpu-smoke --output "$output_dir/environment.json"
"$AE_ROOT/scripts/ae/check_environment.sh"
"$AE_ROOT/scripts/ae/run_metadata_tests.sh"
"$AE_ROOT/scripts/ae/run_kernel_tests.sh"
"$AE_ROOT/scripts/ae/run_smoke_test.sh"

"$AE_PYTHON" - "$output_dir/summary.json" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
output.write_text(
    json.dumps({"status": "PASS", "scope": "synthetic GPU smoke test"}, indent=2) + "\n",
    encoding="utf-8",
)
print("[PASS] synthetic GPU smoke test")
PY
