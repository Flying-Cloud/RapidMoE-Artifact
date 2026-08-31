#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

experiment="${RAPIDMOE_EXPERIMENT:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiment)
      [[ $# -ge 2 ]] || { echo "--experiment requires a number" >&2; exit 2; }
      experiment="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--experiment 1|2|3|4]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done
if [[ -n "$experiment" && ! "$experiment" =~ ^[1-4]$ ]]; then
  echo "[FAIL] --experiment must be 1, 2, 3, or 4" >&2
  exit 2
fi

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
(( mem_gib >= 16 )) && echo "[PASS] required: DRAM ${mem_gib} GiB" || { echo "[FAIL] at least 16 GiB DRAM is required"; fail=1; }
if [[ "$experiment" == 4 ]]; then
  (( mem_gib >= 500 )) && echo "[PASS] Experiment 4 DRAM: ${mem_gib} GiB visible" || { echo "[FAIL] Experiment 4 requires a 512 GB host (at least 500 GiB visible)"; fail=1; }
  required "$AE_PYTHON" -c 'import torch; assert torch.cuda.device_count() >= 2; assert all(torch.cuda.get_device_properties(i).total_memory >= 79 * 1024**3 for i in range(2))'
  "$AE_PYTHON" - <<'PY'
import torch
for index in range(min(2, torch.cuda.device_count())):
    props = torch.cuda.get_device_properties(index)
    print(f"[INFO] Experiment 4 GPU {index}: {props.name}; {props.total_memory / 1024**3:.1f} GiB; sm_{props.major}{props.minor}")
    if "A800" not in props.name:
        print("[WARN] Experiment 4 was validated on A800-80GB; this GPU model is untested")
PY

  cpu_threads="${RAPIDMOE_CPU_THREADS:-$(rapidmoe_default_cpu_threads)}"
  [[ "$cpu_threads" =~ ^[1-9][0-9]*$ ]] || { echo "[FAIL] RAPIDMOE_CPU_THREADS must be a positive integer"; fail=1; cpu_threads=1; }
  physical_cores="$(rapidmoe_visible_physical_cores)"
  if [[ "$physical_cores" =~ ^[1-9][0-9]*$ ]]; then
    (( cpu_threads <= physical_cores )) && echo "[PASS] CPU threads: ${cpu_threads} requested, ${physical_cores} physical cores visible" || { echo "[FAIL] RAPIDMOE_CPU_THREADS=${cpu_threads} oversubscribes ${physical_cores} physical cores"; fail=1; }
  else
    echo "[WARN] unable to determine physical core count"
  fi
fi

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
if [[ "$experiment" == 4 ]]; then
  [[ -n "${RAPIDMOE_MODEL_PATH:-}" ]] || { echo "[FAIL] Experiment 4 requires RAPIDMOE_MODEL_PATH"; fail=1; }
  [[ -n "${RAPIDMOE_GGUF_PATH:-}" ]] || { echo "[FAIL] Experiment 4 requires RAPIDMOE_GGUF_PATH"; fail=1; }
  if [[ -f "${RAPIDMOE_GGUF_PATH:-}" ]]; then
    gguf_bytes=$(stat -c %s "$RAPIDMOE_GGUF_PATH")
    (( gguf_bytes == 427535921888 )) && echo "[PASS] Experiment 4 checkpoint byte length" || { echo "[FAIL] unexpected Experiment 4 checkpoint byte length: ${gguf_bytes}"; fail=1; }
  fi
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
