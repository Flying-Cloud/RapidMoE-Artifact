# Reproducible environment

The primary AE environment is the pinned OCI image described by
`Dockerfile.ae`. It fixes the Linux user space, CUDA 12.2.2 development
toolchain, Python 3.11 bootstrap, PyTorch 2.5.1 cu121, Python dependency
versions, target GPU architecture (`sm_80`) and portable AVX2 CPU build.

The external host supplies only an x86-64 Linux kernel, an NVIDIA Ampere GPU or
newer, and NVIDIA Linux driver 525.60.13 or newer. Model configuration and GGUF
weights are mounted read-only and are never copied into the image.

Build from the artifact root:

```bash
docker build --pull \
  -f environment/Dockerfile.ae \
  -t rapidmoe-ae:eurosys27 .
```

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

The dependency lock is intentionally platform-specific. It is an input to the
pinned container, not a claim of cross-platform portability.

The AE snapshot replaces balance-serve's former `-march=native` flags with an
explicit AVX2/FMA/F16C baseline. The runtime preloads the Ubuntu GCC 11
`libstdc++` to match the native extensions instead of Miniconda's older copy.
