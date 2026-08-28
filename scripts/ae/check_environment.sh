#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

fail=0
required() { if "$@" >/dev/null 2>&1; then echo "[PASS] required: $*"; else echo "[FAIL] required: $*"; fail=1; fi; }
advisory() { if "$@" >/dev/null 2>&1; then echo "[PASS] recommended: $*"; else echo "[WARN] recommended condition not met: $*"; fi; }

[[ "$(uname -s)" == Linux ]] && echo "[PASS] required: Linux" || { echo "[FAIL] Linux is required"; fail=1; }
required command -v "$AE_PYTHON"
required command -v gcc
required command -v g++
required command -v nvcc
if [[ "${RAPIDMOE_REQUIRE_GPU:-1}" == 1 ]]; then
  required command -v nvidia-smi
  required "$AE_PYTHON" -c 'import torch; assert torch.cuda.is_available(); assert torch.version.cuda'
  required "$AE_PYTHON" -c 'from ktransformers.server.balance_serve.inference import model_runner'
else
  echo "[INFO] GPU checks disabled for metadata-only rehearsal"
fi
required "$AE_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version'
required "$AE_PYTHON" -c 'import torch; assert torch.__version__ == "2.5.1+cu121", torch.__version__'
required "$AE_PYTHON" -c 'import transformers, triton; assert transformers.__version__ == "4.51.3"; assert triton.__version__ == "3.1.0"'
required "$AE_PYTHON" -c 'import importlib.metadata as m; assert m.version("flashinfer-python") == "0.2.3"'
required "$AE_PYTHON" -c 'import importlib.metadata as m; from openai.types.chat.chat_completion_chunk import Choice; assert m.version("openai") == "1.39.0"'
required "$AE_PYTHON" -c 'import torch, KTransformersOps'
required "$AE_PYTHON" -c 'import torch, cpuinfer_ext'
required "$AE_PYTHON" -c 'import transformers, triton, numpy'
grep -qw avx2 /proc/cpuinfo && echo "[PASS] required: AVX2" || { echo "[FAIL] AVX2 is required"; fail=1; }
grep -qw avx512f /proc/cpuinfo && echo "[PASS] recommended: AVX-512" || echo "[WARN] AVX-512 unavailable; performance may be lower"
advisory command -v numactl

mem_gib=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)
(( mem_gib >= 32 )) && echo "[PASS] required: DRAM ${mem_gib} GiB" || { echo "[FAIL] at least 32 GiB DRAM is required"; fail=1; }
(( mem_gib >= 512 )) && echo "[PASS] paper-scale DRAM: ${mem_gib} GiB" || echo "[WARN] paper-scale run recommends 512 GiB DRAM"

if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
fi
if command -v nvcc >/dev/null; then
  echo "[INFO] selected nvcc: $(command -v nvcc)"
  nvcc --version | tail -1
fi
if [[ -n "${RAPIDMOE_MODEL_PATH:-}" && ! -f "$RAPIDMOE_MODEL_PATH/config.json" ]]; then
  echo "[FAIL] RAPIDMOE_MODEL_PATH lacks config.json"; fail=1
fi
if [[ -n "${RAPIDMOE_GGUF_PATH:-}" && ! -e "$RAPIDMOE_GGUF_PATH" ]]; then
  echo "[FAIL] RAPIDMOE_GGUF_PATH does not exist"; fail=1
fi
disk_probe="${RAPIDMOE_OUTPUT_DIR:-$AE_ROOT/results}"
mkdir -p "$disk_probe"
disk_gib=$(df -Pk "$disk_probe" | awk 'NR==2 {printf "%d", $4/1024/1024}')
(( disk_gib >= 10 )) && echo "[PASS] required: free disk ${disk_gib} GiB" || { echo "[FAIL] at least 10 GiB free disk is required"; fail=1; }

if (( fail )); then
  echo "[FAIL] Environment. See artifact/TROUBLESHOOTING.md."
  exit 1
fi
echo "[PASS] Environment"
