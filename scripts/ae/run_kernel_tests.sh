#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
"$AE_PYTHON" "$AE_ROOT/tests/ae/test_dynamic_threshold_kernel.py"
"$AE_PYTHON" "$AE_ROOT/tests/ae/test_static_forward_guard.py"
