# DeepSeek V4 BF16 GEMM performance benchmark on gfx1250

Benchmark dates:

- Original benchmark: 2026-08-13
- PR #875 follow-up: 2026-08-14
- Full perf-set rerun: 2026-08-16
- Consolidated production A16W16 and A8W8 update: 2026-08-17

## Scope and data

Results covered:

1. Aiter generic Triton, tuned per shape over a compact candidate set
2. Aiter Opus from [PR #4246](https://github.com/ROCm/aiter/pull/4246)
3. Local FlyDSL gfx1250 kernel, tuned per shape over a compact candidate set
4. Aiter Gluon as an additional gfx1250-native reference
5. FlyDSL [PR #875](https://github.com/ROCm/FlyDSL/pull/875)
6. Consolidated production A16W16 routes and the A8W8 tuned reference

Raw results:

- [`dsv4_gemm_benchmark_results_perfset_20260816.json`](./dsv4_gemm_benchmark_results_perfset_20260816.json)
- [`dsv4_flydsl_pr875_results_perfset_20260816.json`](./dsv4_flydsl_pr875_results_perfset_20260816.json)
- [`a16w16_hybrid_gfx1250_results_20260816.json`](./a16w16_hybrid_gfx1250_results_20260816.json)
- [`a16w16_hybrid_directb_results_20260817.json`](./a16w16_hybrid_directb_results_20260817.json)
- [`a8w8_tuned_reproduction_20260817.json`](./a8w8_tuned_reproduction_20260817.json)

Complete optimization history, reproduction commands, and debug notes:
[`DSV4_GFX1250_GEMM_OPTIMIZATION_NOTES.md`](./DSV4_GFX1250_GEMM_OPTIMIZATION_NOTES.md).

## Environment and timing protocol

| Component | Version / identity |
|---|---|
| GPU | AMD gfx1250, 256 CUs, physical GPU3 |
| Host perf setup | `455_perf_set.sh` applied; SDMA/SVM/XNACK enabled; NUMA balancing off |
| Docker image | `rocm/fw-bringup:gfx1250-atom-dev-20260729` |
| Docker image ID | `sha256:eaef911cf1cb0a3629ae62bf99649799824818d010c7b50ccbc9fba9c90887f9` |
| Python / PyTorch | 3.12.3 / `2.11.0+rocm7.15.0a20260712` |
| HIP runtime / hipcc | 7.15.0 / HIP 7.15.0, AMD clang 23.0.0git `aa451e1f...` |
| Triton | 3.8.0 |
| Installed FlyDSL | 0.2.4 |
| PR #875 FlyDSL runtime | 0.3.0, commit `e3025ff57366011bca44607736d381beffeb7f50` |
| Aiter | branch `pr-4246`, commit `0f144e5e4449a8b0a125715997c2386cc190f6f6` |

- 20 warmup calls and 50 final samples
- Tuning: 3 warmups and 8 samples
- PR #875 candidates: 5 consecutive correctness checks before timing
- BF16 inputs, FP32 accumulation, BF16 output, no bias
- K padding and unsupported-layout materialization occur outside the timed region
- `kernel_us`: median sum of profiler GPU kernel durations per call
- `e2e_us`: median HIP-event interval over ten calls, divided by ten
- `TFLOPS = 2*M*N*K / time_us / 1e6`
- `effective bandwidth = 2*(M*K + N*K + M*N) / time_us / 1e3 GB/s`

## Representative DSv4 shapes

`AI = 2*M*N*K / (2*(M*K + N*K + M*N)) FLOP/byte`

| Name | M | N | K | AI | Class | DSv4 CSV winner |
|---|---:|---:|---:|---:|---|---|
| decode_projection | 1 | 1024 | 4096 | 1.0 | memory/launch | Opus |
| decode_skinny_n | 32 | 64 | 7168 | 21.3 | memory/launch | Opus |
| decode_wide_n | 32 | 32320 | 7168 | 31.8 | memory | Triton |
| prefill_opus | 128 | 2048 | 4096 | 117.0 | mixed | Opus |
| transition_opus | 512 | 2048 | 7168 | 387.5 | mixed | Opus |
| compute_opus | 2048 | 1024 | 7168 | 623.3 | compute | Opus |
| compute_crossover | 4096 | 2048 | 4096 | 1024.0 | compute | Triton |
| compute_large | 16384 | 2048 | 4096 | 1260.3 | compute | Triton |

## Original four-backend results

### Kernel-only latency

Lower is better.

| Shape | Triton us | Gluon us | Opus PR #4246 us | Tuned FlyDSL us | Winner |
|---|---:|---:|---:|---:|---|
| (1,1024,4096) | 13.14 | 5.49 | **4.83** | 9.65 | Opus |
| (32,64,7168) | 20.81 | 17.62 | **5.13** | 15.38 | Opus |
| (32,32320,7168) | **68.13** | 68.49 | 443.76* | 69.03 | Triton |
| (128,2048,4096) | 14.30 | 15.96 | **11.48** | 13.02 | Opus |
| (512,2048,7168) | 42.38 | 49.89 | 49.26 | **31.79** | FlyDSL |
| (2048,1024,7168) | 51.66 | 93.20 | 63.34 | **49.23** | FlyDSL |
| (4096,2048,4096) | 108.25 | 119.51 | 426.49* | **84.69** | FlyDSL |
| (16384,2048,4096) | **335.35** | 482.12† | 1532.16* | 336.64 | Triton |

\* Opus heuristic because the merged DSv4 row selects Triton.
\† Gluon failed correctness and is excluded from winner selection.

### Kernel-only throughput

| Shape | Triton TFLOPS | Gluon TFLOPS | Opus TFLOPS | FlyDSL TFLOPS |
|---|---:|---:|---:|---:|
| (1,1024,4096) | 0.64 | 1.53 | **1.74** | 0.87 |
| (32,64,7168) | 1.41 | 1.67 | **5.73** | 1.91 |
| (32,32320,7168) | **217.64** | 216.49 | 33.41* | 214.80 |
| (128,2048,4096) | 150.17 | 134.53 | **187.14** | 164.96 |
| (512,2048,7168) | 354.68 | 301.29 | 305.15 | **472.92** |
| (2048,1024,7168) | 582.02 | 322.57 | 474.67 | **610.67** |
| (4096,2048,4096) | 634.85 | 575.00 | 161.13* | **811.46** |
| (16384,2048,4096) | **819.68** | 570.14† | 179.41* | 816.53 |

### Kernel-only effective memory bandwidth

This is an algorithmic lower-bound traffic rate, not a hardware-counter
measurement of HBM traffic. It assumes each BF16 A, B, and C element is moved
once. It excludes split-K workspace traffic and cache effects, so values can
exceed physical HBM bandwidth.

| Shape | Triton GB/s | Gluon GB/s | Opus GB/s | FlyDSL GB/s |
|---|---:|---:|---:|---:|
| (1,1024,4096) | 639.3 | 1530.7 | **1740.7** | 870.1 |
| (32,64,7168) | 66.3 | 78.3 | **269.3** | 89.7 |
| (32,32320,7168) | **6838.3** | 6802.3 | 1049.8* | 6749.1 |
| (128,2048,4096) | 1283.2 | 1149.5 | **1599.1** | 1409.6 |
| (512,2048,7168) | 915.4 | 777.6 | 787.6 | **1220.6** |
| (2048,1024,7168) | 933.8 | 517.5 | 761.5 | **979.7** |
| (4096,2048,4096) | 620.0 | 561.5 | 157.4* | **792.4** |
| (16384,2048,4096) | **650.4** | 452.4† | 142.4* | 647.9 |

### End-to-end GPU interval

| Shape | Triton us | Gluon us | Opus PR #4246 us | Tuned FlyDSL us | Winner |
|---|---:|---:|---:|---:|---|
| (1,1024,4096) | 37.54 | 89.73 | 32.41 | **10.44** | FlyDSL |
| (32,64,7168) | 37.77 | 86.51 | 32.67 | **16.21** | FlyDSL |
| (32,32320,7168) | **69.60** | 88.44 | 574.95* | 71.00 | Triton |
| (128,2048,4096) | 37.86 | 92.94 | 33.59 | **14.12** | FlyDSL |
| (512,2048,7168) | 43.96 | 88.44 | 49.62 | **33.19** | FlyDSL |
| (2048,1024,7168) | 53.84 | 95.03 | 64.35 | **50.58** | FlyDSL |
| (4096,2048,4096) | 92.63 | 120.51 | 528.94* | **88.30** | FlyDSL |
| (16384,2048,4096) | **336.31** | 483.33† | 1656.75* | 338.82 | Triton |

### Selected local FlyDSL configurations

| Shape | reg M | reg N | reg K | waves MxN | stages | overlap | swizzle | scheduler |
|---|---:|---:|---:|---|---:|---|---:|---|
| (1,1024,4096) | 1 | 1 | 4 | 1x2 | 3 | cross | 32 | default |
| (32,64,7168) | 1 | 1 | 4 | 1x2 | 3 | cross | 32 | default |
| (32,32320,7168) | 1 | 8 | 4 | 2x1 | 3 | cross | 32 | default |
| (128,2048,4096) | 1 | 4 | 4 | 2x1 | 3 | cross | 32 | default |
| (512,2048,7168) | 4 | 4 | 4 | 2x1 | 3 | cross | 32 | default |
| (2048,1024,7168) | 4 | 8 | 4 | 2x1 | 3 | cross | 32 | default |
| (4096,2048,4096) | 4 | 8 | 4 | 2x1 | 3 | cross | 32 | max-memory-clause |
| (16384,2048,4096) | 4 | 8 | 4 | 2x1 | 3 | cross | 32 | max-memory-clause |

## FlyDSL PR #875 results

Exact PR head: `1273aa33d5eb8c6f5b724f890fdf159439f8bcc4`.
Positive delta means PR #875 is slower than local FlyDSL. Differences around
5% should be treated as noise.

| Shape | PR #875 kernel us | e2e us | TFLOPS | GB/s | Local FlyDSL us | Delta | Overall winner |
|---|---:|---:|---:|---:|---:|---:|---|
| (1,1024,4096) | 12.12 | 13.00 | 0.69 | 693.1 | 9.65 | +25.5% | Opus |
| (32,64,7168) | 21.77 | 22.50 | 1.35 | 63.4 | 15.38 | +41.5% | Opus |
| (32,32320,7168) | 68.79 | 70.40 | 215.53 | 6772.1 | 69.03 | -0.3% | Triton |
| (128,2048,4096) | 20.81 | 22.67 | 103.20 | 881.8 | 13.02 | +59.8% | Opus |
| (512,2048,7168) | 39.54 | 40.89 | 380.19 | 981.2 | 31.79 | +24.4% | Local FlyDSL |
| (2048,1024,7168) | **38.78** | **42.34** | **775.32** | **1243.9** | 49.23 | **-21.2%** | PR #875 |
| (4096,2048,4096) | 101.08 | 93.59 | 679.85 | 663.9 | 84.69 | +19.4% | Local FlyDSL |
| (16384,2048,4096) | 358.72 | 360.74 | 766.27 | 608.0 | 336.64 | +6.6% | Triton |

Across all eight shapes, PR #875 is 17.1% slower by kernel-time geomean and
15.7% slower by end-to-end geomean versus local FlyDSL.

### Selected PR #875 configurations

| Shape | tile MxNxK | waves MxN | buffers | variant | split-K | waves/EU |
|---|---|---|---:|---|---:|---:|
| (1,1024,4096) | 16x64x128 | 1x4 | 3 | bandwidth-bound | 1 | default |
| (32,64,7168) | 32x64x128 | 2x2 | 3 | bandwidth-bound | 1 | default |
| (32,32320,7168) | 32x64x128 | 2x2 | 3 | bandwidth-bound | 1 | default |
| (128,2048,4096) | 128x128x128 | 2x4 | 2 | bandwidth-bound | 8 | default |
| (512,2048,7168) | 128x128x128 | 2x2 | 3 | bandwidth-bound | 1 | default |
| (2048,1024,7168) | 128x128x128 | 2x2 | 3 | bandwidth-bound | 1 | default |
| (4096,2048,4096) | 128x128x128 | 2x2 | 2 | bandwidth-bound | 1 | 2 |
| (16384,2048,4096) | 128x128x128 | 2x2 | 2 | compute-bound | 1 | default |

## Consolidated production A16W16

GPU3 results from 2026-08-17/18:

| Shape | Production kernel us | Production TFLOPS | Comparison baseline | Baseline us | Baseline TFLOPS | Correctness |
|---|---:|---:|---|---:|---:|---|
| (128,2048,4096) | **16.4430** | **130.602** | same-session producer/consumer | 27.0000 | 79.536 | passed |
| (512,2048,7168) | **29.6630** | **506.772** | same-session producer/consumer | 45.6270 | 329.463 | passed |
| (2048,1024,7168) | **37.0145** | **812.243** | same-process baseline | 48.91 | 614.66 | passed bit-exact |
| (4096,2048,4096) | **77.8600** | **882.603** | same-session producer/consumer | 84.5455 | 812.811 | passed bit-exact |
| (16384,2048,4096) | **285.1405** | **964.009** | same-session producer/consumer | 330.6135 | 831.418 | passed bit-exact |

The selected routes live in the single
[`kernels/gemm_a16w16_gfx1250.py`](./kernels/gemm_a16w16_gfx1250.py)
production file.

| Route | Path | reg MxNxK | waves MxN | buffers | swizzle | XCD remap |
|---|---|---|---|---:|---:|---|
| exact M=128/512/2048 production shapes | all-compute | 2x4x4 | 4x2 | 3 | 32 | — |
| aligned `M >= 4096` | A-LDS/direct-B | 8x4x2 | 1x4 | 3 | 8 | enabled |

### Current small-shape kernel comparison

| Shape | Production FlyDSL us | Opus PR #4246 us | Winner |
|---|---:|---:|---|
| (1,1024,4096) | 13.219 | **4.83** | Opus |
| (32,64,7168) | 18.507 | **5.13** | Opus |
| (128,2048,4096) | 16.443 | **11.48** | Opus |
| (512,2048,7168) | **29.663** | 49.26 | FlyDSL |

The production FlyDSL timings are the latest random-BF16 measurements; Opus
values are from the full perf-set. Differences below approximately 5% remain
within normal run-to-run noise.

## A8W8 tuned reference and reproduction

GPU3 measurements from 2026-08-17:

| Shape | CSV us | CSV PFLOPS | Reproduced us | Reproduced PFLOPS |
|---|---:|---:|---:|---:|
| (8192,6144,7168) | 130.27 | 5.539 | **118.36** | **6.096** |
| (16384,7168,4096) | 172.83 | 5.567 | **169.95** | **5.661** |
| (32768,7168,4096) | 340.64 | 5.649 | **382.20** | **5.034** |

The three-shape geometric-mean latency ratio is 1.0008. The selected A8W8
configuration is `256x256x128`, four waves, and four buffers.

## Measurement limitations

- Triton and FlyDSL use same-session compact candidate tuners, not exhaustive
  offline searches over every legal configuration.
- Opus is explicitly tuned only where the merged DSv4 CSV selects Opus; `*`
  marks heuristic values.
- Gluon is incorrect on `(16384,2048,4096)` and is excluded there; `†` marks
  that result.
- PR #875 M/N padding is materialized outside the timed region.
- Profiler overhead can make `kernel_us` slightly larger than batched `e2e_us`
  for long kernels.
- A8W8 reproduction values use profiler self-device time, matching the tuner.
- Differences below approximately 5% should be treated as timing noise.
