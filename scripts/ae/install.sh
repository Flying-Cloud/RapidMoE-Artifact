#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
: "${KT_ROOT:?Set RAPIDMOE_KTRANSFORMERS_ROOT to the KTransformers source root}"
cd "$KT_ROOT"
# The end-to-end API exposes only the DeepSeek balance-serve path, but requires its native
# scheduler/cache extension in addition to the core RapidMoE operators.
export USE_BALANCE_SERVE=1
export KTRANSFORMERS_FORCE_BUILD=TRUE
# Some cluster modules export DEBUG=release, while setuptools expects an integer.
# Pin the reproducible release-build setting rather than inheriting host state.
export DEBUG=0
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.2}"
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/include${CPATH:+:$CPATH}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$AE_PYTHON" -m pip install -v . --no-build-isolation --no-deps
