# Paper-to-artifact map

| Paper claim | Artifact implementation | Evaluator action | Expected result |
|---|---|---|---|
| Split Expert Routing | `KDeepseekV3MoEV2.forward`, `KExpertsHybrid.submit_for_one_decode` | `run_functional_test.sh` with V3 `r=1/2` | finite deterministic layer-38 output |
| RESplit representation | GGUF loader types 350/351 and `offs_concate_experts*` | `inspect_checkpoint.py`, then functional load assertion | canonical RESplit header and successful load |
| Residual CPU execution | `FlexMOE` in `csrc/.../llamafile/` | inspect the functional result | nonzero CPU branch norm |
| GPU low-bit execution | custom GGUF MoE CUDA kernels | inspect the functional result | nonzero GPU branch norm |
| CPU/GPU merge | `dynamic_add` and hybrid synchronization | GPU smoke test plus deterministic forward | finite merged output |
| Online expert selection | `dynamic_threshold` six-argument CUDA binding | `test_dynamic_threshold_kernel.py` and GPU smoke test | all configured `r` branches pass |
| Preconfigured UMIA Runtime | `rapidmoe_ae/profile.py`, `rapidmoe_deployment.py`, frozen V3 JSON | metadata check and dynamic endpoint | profile applies to layers 3–60 |
| Full UMIA parameter generation | not distributed | none | deliberately out of scope |
| DeepSeek-V3 dynamic API | frozen dynamic profile, same endpoint | `run_deepseek_v3.sh`, then `smoke_api.py --model deepseek-v3` | HTTP 200 and readable completion |

The scoped server registers only `/v1/models` for discovery and `/v1/chat/completions` for inference. The public Experiment 4 launcher accepts only model ID `deepseek-v3` and selects dynamic routing from the frozen profile.
