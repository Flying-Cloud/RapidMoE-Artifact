#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

image="${RAPIDMOE_AE_IMAGE:-rapidmoe-ae:eurosys27}"
log="$AE_ROOT/results/clean-rehearsal/image-build.log"
mkdir -p "$(dirname "$log")"
pull_args=(--pull)
if [[ "${RAPIDMOE_DOCKER_PULL:-1}" == 0 ]]; then
  pull_args=()
fi

daemon_proxy=$(docker info 2>/dev/null | awk '/HTTPS Proxy:/ {print $3; exit}')
if [[ "$daemon_proxy" =~ ^https?://127\.0\.0\.1:([0-9]+)$ ]]; then
  proxy_port="${BASH_REMATCH[1]}"
  if ! timeout 1 bash -c "</dev/tcp/127.0.0.1/$proxy_port" 2>/dev/null; then
    echo "[FAIL] Docker daemon proxy $daemon_proxy is not reachable." >&2
    echo "[FAIL] Fix the daemon proxy or use an external clean host; no build was started." >&2
    exit 1
  fi
fi

docker build "${pull_args[@]}" --no-cache --progress=plain \
  -f "$AE_ROOT/environment/Dockerfile.ae" -t "$image" "$AE_ROOT" \
  2>&1 | tee "$log"
docker image inspect "$image" --format '{{.Id}}'
