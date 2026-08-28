# RapidMoE AE manifest

This artifact targets the EuroSys 2027 **Available + Functional** badges. It does not target Reproduced.

| Paper component | Production implementation | AE evidence |
|---|---|---|
| RESplit WQ/WR layout | `custom_gguf.offs_concate_experts*`, `KExpertsHybrid.load` | smoke preprocessing placement; functional mmap/GPU placement |
| Split Expert Routing | `KExpertsHybrid.submit_for_one_decode` | V3 fixed `r=1/2` one-layer paths and routing invariants |
| Preconfigured UMIA Runtime | `dynamic_threshold` CUDA kernel | V3-only frozen profile plus all-branch kernel test |
| CPU/GPU overlap and merge | `FlexMOE.forward_*`, CUDA MoE kernels, `dynamic_add` | smoke merge conditions; functional nonzero branch evidence and merged finite output |
| End-to-end generation | balance-serve plus DeepSeek multi-GPU YAML | five CUDA Graph captures and V3 `/v1/chat/completions` PASS |

The package allowlist is `artifact/ALLOWLIST.txt`; the generated exact manifest and checksums are `artifact/FILES.txt` and `artifact/SHA256SUMS`.

The release includes the scoped balance-serve server path required by the DeepSeek endpoint. The compatibility patch remains available for source variants that evaluate core operators without a built scheduler extension.

Excluded: calibration/profile-generation/frontier-search implementation and inputs, Qwen3 AE entries, full quality suites, ShareGPT, weights, build products, logs/databases, notebooks, duplicate extension sources, and unrelated KTransformers entry points. Vendored third-party source is retained unchanged for reliable builds, but no third-party example or test is an AE entry point. A small number of internal Qwen model definitions remain only because production modules import them unconditionally; there is no Qwen configuration, launcher, or test.
