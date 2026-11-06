# Benchmark latency diagnosis — 2026-10-19

## Outcome
The accuracy coordinator runs four arms, both Python versions, and all files sequentially: 15,972 full-file evaluations. The completed control/3.14 pass took 2.997 hours inside file evaluation; its average was 5.633 seconds/file, median 3.378, p95 20.939. Extrapolating that one pass gives 24.99 hours of file evaluation alone; this is not a guaranteed ETA for other arms.

## Evidence and interpretation
- Raw recorded workload and telemetry: `benchmark-slowness-20261019.json`. Existing per-file monotonic timings were analyzed without rerunning GPU inference.
- Workload: dev_scripts.run_full_benchmarks -> dev_scripts.benchmark_full_ablation on pylingual-train, RTX 4090, FP32. Commands/revision fingerprints and family/runtime identities are recorded with remote benchmark provenance artifacts under /home/henryh/pylingual-shared-full-20261016/benchmark-50k.
- Coordinator code uses nested arm/version loops and blocking subprocess.run. No parallel file jobs are scheduled. Translation does batch statements within each file (normally 32); segmentation windows are batched at 8. This is not wholly unbatched inference.
- This is end-to-end decompilation, not a single model forward: segmentation, autoregressive statement generation (max_length=512), control-flow reconstruction, compilation/equivalence verification, and correction attempts. The first control traversed 17887 code objects and recorded 20740 verification checks.
- 54 timeouts consumed 1625.01 seconds (~15% of that pass). CompileError files accounted for 8550.03 seconds; failure labels do not isolate phase cost.
- Tiny valid compilation RPC probes: n=8, median=1.251 ms. These do not reproduce payload size or correction complexity; they do show that network round-trip latency alone is not multi-second for a tiny request.
- GPU utilization/memory/power sampled 8 times during existing work; raw samples preserved. These are activity observations, not kernel/phase attribution. CPU process telemetry is a snapshot/lifetime average, not a CPU stack profile. Exact CPU-vs-generation phase breakdown and allocator/live-memory breakdown are unavailable.

## Next options (not implemented)
Prioritize profiling the decode/correction path and improving scheduling/batching while preserving output fidelity, adapter state isolation, timeouts, and isolated throughput/VRAM measurements. Additional workers on the same 24GB GPU need measured peak-memory bounds; low utilization does not prove sufficient VRAM. Do not silently shorten output length, skip verification, alter precision, or shrink benchmark coverage, since those change the comparison.

No benchmark execution or model changes made during this diagnosis.
