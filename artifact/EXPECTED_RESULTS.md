# Expected results

Stable acceptance conditions are the `[PASS]` assertions shown in `expected_outputs/`. The smoke JSON must report `status=PASS`, two different V3 UMIA `r` values, and finite resource measurements. The V3 functional JSON must report `status=PASS`, `routing_mode=static`, observations for `r=1` and `r=2`, nonzero CPU/GPU branch norms, and finite deterministic outputs.

The CPU comparison JSON must identify CUDA Graph replay timing, five trials of
1000 replays, arithmetic-mean latency and standard deviation for the layer-38
`KExpertsCPU + Q4_K_M` and `KExpertsHybrid + RESplit` paths. Latency,
throughput and speedup are observations, not fixed pass thresholds; this
microbenchmark does not make a numerical-equivalence claim across the two
quantization formats. Each Experiment 4 API JSON must report `status=PASS`,
endpoint `/v1/chat/completions`, a nonempty `choices` list, and response text
equal to `RapidMoE AE OK` after surrounding whitespace is removed. The server
startup record must report either the frozen-profile `dynamic` mode or static
mode with `r=2`, `dynamic_topk=false`, and `threshold_enabled=false`.
