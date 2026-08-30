# Troubleshooting

- `GLIBCXX_3.4.30 not found`: the extension was linked against a newer C++ runtime than the Conda environment provides. Rebuild inside the target environment. For diagnosis of an existing local build only, set `RAPIDMOE_LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6` if that exact library provides the required symbol.
- `KTransformersOps` or `cpuinfer_ext` missing: run `scripts/ae/install.sh` from the intended Python environment with CUDA 12.2 available.
- CUDA/PyTorch mismatch: use the pinned verified environment (Python 3.11, torch 2.5.1+cu121, CUDA toolkit 12.2) and a compatible NVIDIA driver. `install.sh` defaults `CUDA_HOME` to `/usr/local/cuda-12.2`; override `CUDA_HOME` when the toolkit is installed elsewhere.
- `invalid literal for int() with base 10: 'release'`: the host exported `DEBUG=release`, while setuptools expects `0` or `1`. The AE installer pins `DEBUG=0`; use the bundled installer rather than invoking `pip` directly.
- FlashInfer JIT selects an old `/usr/bin/nvcc`: set `RAPIDMOE_CUDA_HOME=/path/to/cuda-12.2`; `common.sh` prepends its `bin` directory and `check_environment.sh` reports the compiler actually selected.
- HTTP 502 from `smoke_api.py` with host proxy variables: the bundled client explicitly bypasses proxies for its loopback request. Third-party clients should set `NO_PROXY=127.0.0.1,localhost`.
- OOM: Experiments 2 and 3 use compact layer-38 files and were observed below
  3 GiB allocated VRAM; an idle 8 GiB Ampere GPU is recommended. Experiment 4
  requires exclusive 2×A800-80GB GPUs and a 512 GB host. Keep
  `RAPIDMOE_CPU_THREADS` at or below the visible physical-core count; use up
  to 48 threads.
- Missing expert tensor: the released combined GGUF must contain `blk.38.ffn_{gate,up,down}_exps.weight`.
- NUMA placement can change Experiment 3 timing, but NUMA-aware hardware or
  explicit binding is not required and no fixed performance threshold is used.
- A failure is not converted to SKIP and there is no pure-CPU/reference fallback. Inspect `results/*.json` only after a zero exit status.
