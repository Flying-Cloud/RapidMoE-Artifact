# RapidMoE: EuroSys 2027 Artifact

This repository is the evaluation artifact for **“RapidMoE: Exploiting
Cross-Asymmetry via Adaptive Residual Offloading for Large-Scale MoE
Inference.”**

RapidMoE splits routed MoE experts into a CPU residual path and a GPU low-bit
path. The artifact provides static split routing, dynamic expert selection,
real one-layer execution, a CPU/GPU kernel benchmark, and full end-to-end
DeepSeek-V3-0324 dynamic and fixed-`r=2` endpoints.

All commands below are run from the repository root.

## Hardware and software

- An **Intel x86-64 CPU** is recommended. **AVX2, FMA, and F16C** are
  required; AVX-512 is optional. The artifact was validated on Intel CPUs
  with AVX2 support.
- NVIDIA **Ampere** GPUs are recommended. The artifact was validated on A800/A100 GPUs.
- Verified software: Python 3.11, PyTorch 2.5.1+cu121, CUDA toolkit 12.2,
  and GCC/G++ 11.
- The pinned environment is defined in
  [`environment/Dockerfile.ae`](environment/Dockerfile.ae).

## Evaluation workflow

**Experiments 1–3 form the Minimal Working Example (MWE).** Experiment 1 is
weight-free, while Experiments 2 and 3 use compact single-layer checkpoints.
The complete 398.17 GiB checkpoint is used only by Experiment 4.

| Experiment | Purpose | Recommended resources | Weights |
|---|---|---|---|
| **1 — Metadata Check and Smoke Test (MWE)** | Validate metadata, frozen profiles, routing, dynamic selection, preprocessing and CPU/GPU merge | 1 Ampere GPU; <1 GiB test memory | Weight-free |
| **2 — One-layer Functional (MWE)** | Execute one real DeepSeek-V3 MoE layer at static `r=1/2` | 1 Ampere GPU, 8 GiB VRAM, 16 GiB host memory | 6.57 GiB RESplit layer |
| **3 — CPU/GPU Kernel Benchmark (MWE)** | Compare KExpertsCPU/Q4_K_M and RapidMoE/RESplit with CUDA Graph replay | 1 Ampere GPU, 8 GiB VRAM, 16 GiB host memory | 12.48 GiB layer pair |
| **4 — End-to-end Model** | Validate dynamic and fixed-`r=2` selection through the OpenAI-compatible V3 API | 2×A800-80GB GPUs, 512 GB host memory; reserve 8–16 visible physical CPU cores for the system | 398.17 GiB RESplit |

## Environment setup

Building from the pinned Dockerfile is the recommended and default setup:

```bash
docker build --pull --no-cache \
  -f environment/Dockerfile.ae \
  -t rapidmoe-ae:eurosys27 .
```

As an optional reference, a prebuilt image is published on GHCR:

```bash
docker pull ghcr.io/flying-cloud/rapidmoe-ae:eurosys27
docker tag ghcr.io/flying-cloud/rapidmoe-ae:eurosys27 rapidmoe-ae:eurosys27
```

The prebuilt image targets **linux/amd64**, requires an x86-64 CPU with
**AVX2, FMA, and F16C**, and contains CUDA kernels for NVIDIA **A800/A100
(`sm_80`)** GPUs. It is provided only as a convenience and reference; building
from [`environment/Dockerfile.ae`](environment/Dockerfile.ae) remains the
recommended evaluation path.

The equivalent Apptainer workflow is documented in
[`environment/README.md`](environment/README.md).

For Experiments 2–4, set the DeepSeek-V3 configuration/tokenizer path:

```bash
export RAPIDMOE_PYTHON=/path/to/python3
export RAPIDMOE_CUDA_HOME=/path/to/cuda-12.2
export RAPIDMOE_MODEL_PATH=/path/to/DeepSeek-V3-config-and-tokenizer
```

## 1. Metadata Check and Smoke Test (MWE)

This single experiment runs the metadata checks followed by the production
CUDA routing, dynamic-threshold, RESplit preprocessing and merge kernels on
small synthetic tensors.

```bash
mkdir -p results
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
  --gpus 'device=0' --ipc=host --network=none \
  -v "$PWD/results:/opt/rapidmoe/results" \
  rapidmoe-ae:eurosys27 \
  ./scripts/ae/run_gpu_smoke.sh
```

Successful execution ends with:

```text
[PASS] V3 read-only dynamic profile covers layers 3..60
[PASS] Routing invariants: no lost or duplicate token/expert pairs
[PASS] UMIA dynamic selection: threshold changed r from ... to ...
[PASS] GPU contribution merge condition executed
[PASS] CPU contribution merge condition executed
[PASS] synthetic GPU smoke test
```

Results are written under `results/gpu-smoke/` and `results/smoke.json`.

## 2. One-layer Functional (MWE)

Download the DeepSeek-V3 layer-38 RESplit checkpoint:

```bash
python3 -m pip install 'modelscope-hub==0.2.0'
python3 scripts/ae/download_one_layer_weights.py \
  --output-dir models/DeepSeek-V3-0324-Layer38

export RAPIDMOE_GGUF_PATH="$PWD/models/DeepSeek-V3-0324-Layer38/RES/DeepSeek-V3-0324-Layer38-RES.gguf"
```

Run:

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/ae/run_functional_test.sh
```

The test executes the real CPU residual and GPU low-bit branches at static
`r=1` and `r=2`, merges their outputs, and writes
`results/functional.json`.

## 3. CPU/GPU Kernel Benchmark (MWE)

The one-layer downloader also provides the Q4_K_M checkpoint used by the CPU
baseline:

```bash
export RAPIDMOE_GGUF_PATH="$PWD/models/DeepSeek-V3-0324-Layer38/RES/DeepSeek-V3-0324-Layer38-RES.gguf"
export RAPIDMOE_BASELINE_GGUF_PATH="$PWD/models/DeepSeek-V3-0324-Layer38/Q4_K_M/DeepSeek-V3-0324-Layer38-Q4_K_M.gguf"

CUDA_VISIBLE_DEVICES=0 ./scripts/ae/run_cpu_baseline_benchmark.sh
```

The benchmark reports mean latency, standard deviation, throughput and speedup
from five trials of 1,000 CUDA Graph replays after five warmups.

## 4. End-to-end Model

Experiment 4 requires exclusive access to **2×A800-80GB GPUs** and a host with
**512 GB RAM** (at least 500 GiB visible inside the container). Set
`RAPIDMOE_CPU_THREADS` to the number of visible physical CPU cores minus 8;
subtract 16 instead when more host capacity should be reserved for the OS and
GPU runtime. When unset, the launcher uses the minus-8 setting automatically.
The default `cache_lens=4096`, `chunk_size=256`,
`max_batch_size=4`, and `max_new_tokens=64` are the recommended AE settings.
Allow 480 GiB of free disk for download/reconstruction, or about 550 GiB when
retaining the container image and results alongside the checkpoint.

Download and reconstruct the full RESplit checkpoint:

```bash
python3 scripts/ae/download_modelscope_checkpoint.py \
  --output-dir models/DeepSeek-V3-0324-Full

export RAPIDMOE_GGUF_PATH="$PWD/models/DeepSeek-V3-0324-Full/DeepSeek-V3-0324-RES.gguf"
PHYSICAL_CORES=$(lscpu -p=CORE,SOCKET | awk -F, '!/^#/ {seen[$1 FS $2]=1} END {print length(seen)}')
export RAPIDMOE_CPU_THREADS=$((PHYSICAL_CORES - 8))  # use -16 for a larger reserve
```

Run the strict Experiment 4 preflight:

```bash
./scripts/ae/check_environment.sh --experiment 4
```

Start one DeepSeek-V3 mode at a time:

```bash
# Frozen-profile dynamic selection
CUDA_VISIBLE_DEVICES=0,1 ./scripts/ae/run_deepseek_v3.sh --mode dynamic

# Fixed split: two critical experts on CPU
CUDA_VISIBLE_DEVICES=0,1 ./scripts/ae/run_deepseek_v3.sh --mode static-r2
```

After `Application startup complete`, run from another shell:

```bash
$RAPIDMOE_PYTHON scripts/ae/smoke_api.py \
  --base-url http://127.0.0.1:10002 \
  --model deepseek-v3 \
  --output results/api_v3.json
```

The API check passes when `/v1/chat/completions` returns exactly
`RapidMoE AE OK` after surrounding whitespace is removed. Stop the first
server before launching the other mode and use a distinct `--output` filename
for each result.

## Code and experiment map

| Implementation | Evaluator experiment |
|---|---|
| `rapidmoe_ae/profile.py`, frozen V3 profile | Experiment 1 |
| `tests/ae/test_smoke.py` and production CUDA kernels | Experiment 1 |
| `tests/ae/test_functional.py` | Experiment 2 |
| `tests/ae/benchmark_cpu_baseline.py` | Experiment 3 |
| `ktransformers-source/ktransformers/operators/experts.py` | Experiments 1–3 |
| `ktransformers-source/ktransformers/server/balance_serve/` | Experiment 4 |

Detailed expected results are in
[`artifact/EXPECTED_RESULTS.md`](artifact/EXPECTED_RESULTS.md), the
paper-to-code mapping is in [`artifact/PAPER_MAP.md`](artifact/PAPER_MAP.md),
and troubleshooting guidance is in
[`artifact/TROUBLESHOOTING.md`](artifact/TROUBLESHOOTING.md).

## License, model access and citation

RapidMoE AE code is licensed under **Apache-2.0**; see [`LICENSE`](LICENSE).
Bundled components retain their notices in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

The RESplit weights are hosted at
[`FlyingCloud/DeepSeek-V3-0324-RapidMoE`](https://modelscope.cn/models/FlyingCloud/DeepSeek-V3-0324-RapidMoE)
and pinned by revision and checksums in
[`artifact/model-deposit-manifest.json`](artifact/model-deposit-manifest.json).

Citation metadata and author contact information are provided in
[`CITATION.cff`](CITATION.cff).
