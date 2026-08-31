# Reproducible environment

The primary AE environment is the pinned OCI image described by
`Dockerfile.ae`. It fixes the Linux user space, CUDA 12.2.2 development
toolchain, Python 3.11 bootstrap, PyTorch 2.5.1 cu121, Python dependency
versions, target GPU architecture (`sm_80`) and portable AVX2 CPU build.

The external host supplies an x86-64 Linux kernel, Docker Engine, NVIDIA
Container Toolkit, and a compatible NVIDIA driver (525.60.13 or newer). The
artifact was validated on A800 (`sm_80`) GPUs; other GPU architectures have not
been tested. Model configuration and GGUF weights are mounted read-only and are
never copied into the image.

Build from the artifact root:

```bash
docker build --pull --no-cache \
  -f environment/Dockerfile.ae \
  -t rapidmoe-ae:eurosys27 .
```

The native `cpuinfer_ext` extension embeds `flexmoe.cpp`, so every source
revision must be rebuilt. One image supports both dynamic and static-`r=2`
Experiment 4 modes; a separate image per mode is not needed.

Run the dependency and kernel rehearsal with one GPU:

```bash
mkdir -p results
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
  --gpus 'device=0' --ipc=host --network=none \
  -v "$PWD/results:/opt/rapidmoe/results" \
  rapidmoe-ae:eurosys27 \
  ./scripts/ae/run_gpu_smoke.sh
```

For HPC, convert the exact OCI image to an immutable Apptainer SIF and record
both digests:

```bash
apptainer build rapidmoe-ae.sif docker-daemon:rapidmoe-ae:eurosys27
APPTAINERENV_CUDA_VISIBLE_DEVICES=0 apptainer exec --nv --cleanenv \
  --bind "$PWD/results:/opt/rapidmoe/results" \
  rapidmoe-ae.sif ./scripts/ae/run_gpu_smoke.sh
sha256sum rapidmoe-ae.sif
```

For Experiment 4, expose two A800-80GB GPUs and mount the config, checkpoint,
and results explicitly. The strict preflight checks GPU capacity, visible host
memory, checkpoint byte length, and CPU-thread oversubscription:

```bash
PHYSICAL_CORES=$(lscpu -p=CORE,SOCKET | awk -F, '!/^#/ {seen[$1 FS $2]=1} END {print length(seen)}')
export RAPIDMOE_CPU_THREADS=$((PHYSICAL_CORES - 8))  # use -16 for a larger reserve
docker run --rm --gpus '"device=0,1"' --ipc=host --network=host \
  --ulimit memlock=-1:-1 \
  -e HOME=/tmp -e RAPIDMOE_CPU_THREADS \
  -e RAPIDMOE_MODEL_PATH=/models/config \
  -e RAPIDMOE_GGUF_PATH=/models/DeepSeek-V3-0324-RES.gguf \
  -v /absolute/path/to/config:/models/config:ro \
  -v /absolute/path/to/DeepSeek-V3-0324-RES.gguf:/models/DeepSeek-V3-0324-RES.gguf:ro \
  -v "$PWD/results:/opt/rapidmoe/results" \
  rapidmoe-ae:eurosys27 \
  ./scripts/ae/check_environment.sh --experiment 4
```

The dependency lock is intentionally platform-specific. It is an input to the
pinned container, not a claim of cross-platform portability.

The AE snapshot replaces balance-serve's former `-march=native` flags with an
explicit AVX2/FMA/F16C baseline. The runtime preloads the Ubuntu GCC 11
`libstdc++` to match the native extensions instead of Miniconda's older copy.
