# AE configuration matrix

| Experiment type | Entry | AE status |
|---|---|---|
| Component/kernel smoke | `smoke.yaml` + `scripts/ae/run_smoke_test.sh` | Functional acceptance |
| One-layer real RESplit | `functional.yaml` + `scripts/ae/run_functional_test.sh` | Functional acceptance |
| Layer-38 KExpertsCPU comparison | `cpu_baseline_benchmark.yaml` + `scripts/ae/run_cpu_baseline_benchmark.sh` | CUDA Graph speed observation using two compact GGUFs |
| Memory footprint | resource fields emitted by both tests | Functional observation; no paper curve |
| V3 static split routing | `deepseek_v3_static.json` | Functional acceptance; fixed `r=2` deployment contract |
| V3 UMIA runtime selection | `deepseek_v3_deployment_profile.json` | Kernel smoke acceptance; read-only deployment profile |
| UMIA calibration/frontier search | none | Explicitly excluded |
| Serving TTFT/TPOT/SLO | same paper-scale entry | Informational only; no claimed numbers |
| Accuracy/quality suites | none | Explicitly excluded (MMLU/AIME/HumanEval/PPL) |
| DeepSeek-V3 end-to-end generation | `run_deepseek_v3.sh --mode dynamic|static-r2` | Experiment 4 modes; hardware-verified on 2×A800-80GB; API PASS |
| Qwen3 | none | Explicitly excluded |

The `.yaml` files intentionally contain JSON syntax. JSON is a strict subset of YAML, so both standard YAML readers and the dependency-free AE JSON loader accept the same bytes deterministically.
