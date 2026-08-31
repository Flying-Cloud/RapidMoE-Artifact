#!/usr/bin/env bash
set -euo pipefail

AE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${RAPIDMOE_AE_IMAGE:-rapidmoe-ae:eurosys27}"
MODELS_DIR="${RAPIDMOE_MODELS_DIR:-$AE_ROOT/models}"
RESULTS_DIR="${RAPIDMOE_RESULTS_DIR:-$AE_ROOT/results}"
CONFIG_DIR=/models/DeepSeek-V3-0324-config
LAYER_DIR=/models/DeepSeek-V3-0324-Layer38
FULL_DIR=/models/DeepSeek-V3-0324-Full
FULL_GGUF="$FULL_DIR/DeepSeek-V3-0324-RES.gguf"

usage() {
  cat <<'EOF'
Usage: ./scripts/ae/container_ae.sh COMMAND [ARG]

Commands:
  build                 Build the recommended container from Dockerfile.ae
  pull                  Pull and tag the optional GHCR reference image
  prepare-config        Download the pinned V3 configuration and tokenizer
  prepare-one-layer     Prepare configuration and Experiment 2/3 weights
  prepare-full          Prepare configuration and the full Experiment 4 weight
  exp1                  Run Metadata Check and Smoke Test
  exp2                  Run One-layer Functional
  exp3                  Run CPU/GPU Kernel Benchmark
  mwe                   Prepare and run Experiments 1-3
  exp4 [dynamic|static-r2]
                        Run preflight, server and API check for one V3 mode

Optional environment variables:
  RAPIDMOE_AE_IMAGE, RAPIDMOE_MODELS_DIR, RAPIDMOE_RESULTS_DIR,
  RAPIDMOE_CPU_THREADS (default: 48), RAPIDMOE_STARTUP_TIMEOUT (default: 1800)
EOF
}

require_image() {
  docker image inspect "$IMAGE" >/dev/null 2>&1 || {
    echo "[FAIL] Container image '$IMAGE' is unavailable; run '$0 build' first." >&2
    exit 1
  }
}

prepare_dirs() {
  mkdir -p "$MODELS_DIR" "$RESULTS_DIR"
}

prepare_config() {
  require_image
  prepare_dirs
  local required=(config.json configuration_deepseek.py modeling_deepseek.py tokenizer.json tokenizer_config.json)
  local file missing=0
  for file in "${required[@]}"; do
    [[ -f "$MODELS_DIR/DeepSeek-V3-0324-config/$file" ]] || missing=1
  done
  if (( missing == 0 )); then
    echo "[PASS] Configuration and tokenizer already present."
    return
  fi
  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$MODELS_DIR:/models" \
    "$IMAGE" \
    ms-hub download deepseek-ai/DeepSeek-V3-0324 \
      "${required[@]}" \
      --revision 1c22c4cbeb9aa228df82f8115008c38f046224c1 \
      --local-dir "$CONFIG_DIR"
}

prepare_one_layer() {
  prepare_config
  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$MODELS_DIR:/models" \
    "$IMAGE" \
    python scripts/ae/download_one_layer_weights.py --output-dir "$LAYER_DIR"
}

prepare_full() {
  prepare_config
  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$MODELS_DIR:/models" \
    "$IMAGE" \
    python scripts/ae/download_modelscope_checkpoint.py --output-dir "$FULL_DIR"
}

run_exp1() {
  require_image
  prepare_dirs
  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
    --gpus device=0 --ipc=host --network=none \
    -v "$RESULTS_DIR:/opt/rapidmoe/results" \
    "$IMAGE" ./scripts/ae/run_gpu_smoke.sh
}

run_exp2() {
  require_image
  prepare_dirs
  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
    --gpus device=0 --ipc=host --network=none \
    -e RAPIDMOE_MODEL_PATH="$CONFIG_DIR" \
    -e RAPIDMOE_GGUF_PATH="$LAYER_DIR/RES/DeepSeek-V3-0324-Layer38-RES.gguf" \
    -v "$MODELS_DIR:/models:ro" \
    -v "$RESULTS_DIR:/opt/rapidmoe/results" \
    "$IMAGE" ./scripts/ae/run_functional_test.sh
}

run_exp3() {
  require_image
  prepare_dirs
  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
    --gpus device=0 --ipc=host --network=none \
    -e RAPIDMOE_MODEL_PATH="$CONFIG_DIR" \
    -e RAPIDMOE_GGUF_PATH="$LAYER_DIR/RES/DeepSeek-V3-0324-Layer38-RES.gguf" \
    -e RAPIDMOE_BASELINE_GGUF_PATH="$LAYER_DIR/Q4_K_M/DeepSeek-V3-0324-Layer38-Q4_K_M.gguf" \
    -v "$MODELS_DIR:/models:ro" \
    -v "$RESULTS_DIR:/opt/rapidmoe/results" \
    "$IMAGE" ./scripts/ae/run_cpu_baseline_benchmark.sh
}

run_exp4() {
  local mode="${1:-dynamic}"
  [[ "$mode" == dynamic || "$mode" == static-r2 ]] || {
    echo "[FAIL] Experiment 4 mode must be dynamic or static-r2." >&2
    exit 2
  }
  require_image
  prepare_dirs
  local cpu_threads="${RAPIDMOE_CPU_THREADS:-48}"
  local timeout="${RAPIDMOE_STARTUP_TIMEOUT:-1800}"
  local result="$RESULTS_DIR/api_v3_${mode//-/_}.json"
  local container="rapidmoe-ae-v3-${mode//-/_}-$$"
  local common=(
    --gpus '"device=0,1"' --ipc=host
    -e HOME=/tmp
    -e RAPIDMOE_CPU_THREADS="$cpu_threads"
    -e RAPIDMOE_MODEL_PATH="$CONFIG_DIR"
    -e RAPIDMOE_GGUF_PATH="$FULL_GGUF"
    -v "$MODELS_DIR:/models:ro"
    -v "$RESULTS_DIR:/opt/rapidmoe/results"
  )

  docker run --rm --user "$(id -u):$(id -g)" \
    "${common[@]}" --network=none \
    "$IMAGE" ./scripts/ae/check_environment.sh --experiment 4

  RAPIDMOE_EXP4_CONTAINER="$container"
  cleanup() {
    if [[ -n "${RAPIDMOE_EXP4_CONTAINER:-}" ]]; then
      docker stop -t 20 "$RAPIDMOE_EXP4_CONTAINER" >/dev/null 2>&1 || true
      RAPIDMOE_EXP4_CONTAINER=""
    fi
  }
  trap cleanup EXIT INT TERM
  docker run -d --rm --name "$container" --user "$(id -u):$(id -g)" \
    "${common[@]}" --network=host --ulimit memlock=-1:-1 \
    "$IMAGE" ./scripts/ae/run_deepseek_v3.sh --mode "$mode" >/dev/null

  echo "[INFO] Waiting up to ${timeout}s for the $mode server."
  local deadline=$((SECONDS + timeout))
  while ! docker logs "$container" 2>&1 | grep -q 'Application startup complete'; do
    if [[ "$(docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null || true)" != true ]]; then
      docker logs "$container" >&2 || true
      echo "[FAIL] Experiment 4 server exited before startup." >&2
      exit 1
    fi
    if (( SECONDS >= deadline )); then
      docker logs "$container" >&2 || true
      echo "[FAIL] Experiment 4 startup exceeded ${timeout}s." >&2
      exit 1
    fi
    sleep 10
  done

  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
    --network=host \
    -v "$RESULTS_DIR:/opt/rapidmoe/results" \
    "$IMAGE" python scripts/ae/smoke_api.py \
      --base-url http://127.0.0.1:10002 \
      --model deepseek-v3 \
      --output "/opt/rapidmoe/results/$(basename "$result")"
  echo "[PASS] Experiment 4 $mode: $result"
  cleanup
  trap - EXIT INT TERM
}

command="${1:-}"
case "$command" in
  build)
    exec "$AE_ROOT/scripts/ae/build_clean_image.sh"
    ;;
  pull)
    docker pull ghcr.io/flying-cloud/rapidmoe-ae:eurosys27
    docker tag ghcr.io/flying-cloud/rapidmoe-ae:eurosys27 "$IMAGE"
    ;;
  prepare-config) prepare_config ;;
  prepare-one-layer) prepare_one_layer ;;
  prepare-full) prepare_full ;;
  exp1) run_exp1 ;;
  exp2) run_exp2 ;;
  exp3) run_exp3 ;;
  mwe)
    prepare_one_layer
    run_exp1
    run_exp2
    run_exp3
    ;;
  exp4) run_exp4 "${2:-dynamic}" ;;
  -h|--help|help|'') usage ;;
  *)
    echo "[FAIL] Unknown command: $command" >&2
    usage >&2
    exit 2
    ;;
esac
