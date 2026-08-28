# External model materials

Model weights are not committed to GitHub. The code archive is Apache-2.0;
DeepSeek-derived weights retain the upstream MIT license. Always verify the
revision, byte length and SHA-256 values in `model-deposit-manifest.json`.

## 1. DeepSeek-V3 configuration and tokenizer

The verified small model files can be obtained from the official ModelScope
repository at the pinned revision recorded in the manifest:

```bash
python3 -m pip install 'modelscope-hub==0.2.0'
ms-hub download deepseek-ai/DeepSeek-V3-0324 \
  config.json configuration_deepseek.py modeling_deepseek.py \
  tokenizer.json tokenizer_config.json \
  --revision 1c22c4cbeb9aa228df82f8115008c38f046224c1 \
  --local-dir models/DeepSeek-V3-0324-config
```

Set `RAPIDMOE_MODEL_PATH` to that directory.

## 2. Layer-38 weights for Experiments 2 and 3

Experiments 2 and 3 do not require a complete DeepSeek-V3 checkpoint. Download
the two AE-specific GGUFs from the RapidMoE ModelScope deposit:

```bash
python3 -m pip install 'modelscope-hub==0.2.0'
python3 scripts/ae/download_one_layer_weights.py \
  --output-dir models/DeepSeek-V3-0324-Layer38
```

The downloader verifies the pinned repository revision, exact byte lengths and
SHA-256 values from `model-deposit-manifest.json`. It produces:

```text
models/DeepSeek-V3-0324-Layer38/
├── RES/DeepSeek-V3-0324-Layer38-RES.gguf          7,052,198,528 bytes
└── Q4_K_M/DeepSeek-V3-0324-Layer38-Q4_K_M.gguf  6,341,788,288 bytes
```

These files contain only `blk.38` routed-expert gate/up/down tensors and are
not complete language models. Keep them in separate directories: the current
`GGUFLoader` scans all `.gguf` files in the supplied directory, and both files
intentionally use the same three tensor names.

Set:

```bash
export RAPIDMOE_GGUF_PATH="$PWD/models/DeepSeek-V3-0324-Layer38/RES/DeepSeek-V3-0324-Layer38-RES.gguf"
export RAPIDMOE_BASELINE_GGUF_PATH="$PWD/models/DeepSeek-V3-0324-Layer38/Q4_K_M/DeepSeek-V3-0324-Layer38-Q4_K_M.gguf"
```

Experiment 2 uses only `RAPIDMOE_GGUF_PATH`. Experiment 3 uses both variables.
Neither experiment downloads the full checkpoint.

## 3. Complete RapidMoE RESplit checkpoint for Experiment 4

Public deposit:
<https://modelscope.cn/models/FlyingCloud/DeepSeek-V3-0324-RapidMoE>

The canonical checkpoint is logically one file:

```text
DeepSeek-V3-0324-RES.gguf
427535921888 bytes (398.17 GiB)
```

ModelScope stores five deterministic 80-GiB-or-smaller transport parts. The
bundled downloader verifies and reconstructs the complete file:

```bash
python3 scripts/ae/download_modelscope_checkpoint.py \
  --output-dir models/DeepSeek-V3-0324-Full
```

The exact public revision is recorded in the manifest. Anonymous access was
verified at `https://modelscope.cn`; the `.ai` endpoint was not mirrored when
tested. Reconstruction requires about 480 GiB free disk space. Set
`RAPIDMOE_GGUF_PATH` to the reconstructed file. This complete checkpoint is
needed only for Experiment 4.

## Integrity and failure policy

- A missing, private or mutable deposit is an error, not a skipped Functional test.
- Do not place the layer-38 RESplit and Q4_K_M files in the same directory.
- Do not rename transport parts or concatenate them without checking each hash.
- Do not use `master`/`main` when an immutable revision is available.
- The full checkpoint hash, not a repository commit alone, is the final identity.
