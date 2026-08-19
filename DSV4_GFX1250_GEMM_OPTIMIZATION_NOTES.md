# DeepSeek V4 BF16 GEMM optimization notes and debug log on gfx1250

Date: 2026-08-13
PR #875 follow-up: 2026-08-14
Perf-set full rerun: 2026-08-16

Concise performance tables: [`DSV4_GFX1250_GEMM_TUNED_BACKEND_BENCHMARK.md`](./DSV4_GFX1250_GEMM_TUNED_BACKEND_BENCHMARK.md).
Historical `hybrid` names below describe experiments; the final routes are
consolidated in [`kernels/gemm_a16w16_gfx1250.py`](./kernels/gemm_a16w16_gfx1250.py).

DeepSeek V4 was selected because PR #4246 contains a complete gfx1250
retune for its 180 BF16 GEMM shapes, while the Kimi-K3 block was not retuned
against the final cluster-grid round-up.

Compared backends:

1. Aiter generic Triton, tuned per shape over a compact candidate set
2. Aiter Opus from [PR #4246](https://github.com/ROCm/aiter/pull/4246)
3. Local FlyDSL gfx1250 kernel, tuned per shape over a compact candidate set
4. Aiter Gluon as an additional gfx1250-native reference
5. FlyDSL [PR #875](https://github.com/ROCm/FlyDSL/pull/875), measured as a
   follow-up with its own compact per-shape tuner

Raw results:
[`dsv4_gemm_benchmark_results_perfset_20260816.json`](./dsv4_gemm_benchmark_results_perfset_20260816.json).
PR #875 results:
[`dsv4_flydsl_pr875_results_perfset_20260816.json`](./dsv4_flydsl_pr875_results_perfset_20260816.json).
Hybrid experiment:
[`a16w16_hybrid_gfx1250_results_20260816.json`](./a16w16_hybrid_gfx1250_results_20260816.json).

Benchmark driver:
[`benchmark_model_gemm_backends.py`](./benchmark_model_gemm_backends.py)
with `--suite dsv4`.

## Executive summary

- 31 of 32 original-backend/shape measurements passed correctness. Gluon was
  incorrect on `(16384,2048,4096)` and is excluded there.
- Tuned Opus is the strongest small-M kernel:
  - `(1,1024,4096)`: 4.83 us vs FlyDSL 9.65 us
  - `(32,64,7168)`: 5.13 us vs FlyDSL 15.38 us
  - `(128,2048,4096)`: 11.48 us vs FlyDSL 13.02 us
- Tuned FlyDSL is strongest through the middle/large M regime:
  - `(512,2048,7168)`: **31.79 us**
  - `(2048,1024,7168)`: **49.23 us**
  - `(4096,2048,4096)`: **84.69 us**
- Tuned Triton wins the largest shape:
  - `(16384,2048,4096)`: **335.35 us**
  - FlyDSL: 336.64 us, only 0.4% slower
- Local FlyDSL wins end-to-end GPU interval on six of eight shapes among the
  original four backends because it launches one kernel with low dispatch overhead.
- The main remaining FlyDSL gap is small-M split-K. The current dense kernel is
  already competitive or best for M >= 512.
- PR #875 follow-up:
  - 17.1% slower than local FlyDSL by kernel-time geomean over the eight shapes
  - wins `(2048,1024,7168)` at **38.78 us**, 21.2% faster than local FlyDSL
  - reaches 358.72 us on the largest shape, 6.6% slower than local FlyDSL
  - split-K 28/56 is unstable on `(32,64,7168)` and fails 97/96 of 100 stress runs
- A8W8-inspired hybrid:
  - improves `(2048,1024,7168)` from 48.91 to **37.35 us** (+30.9% TFLOPS)
  - is 7.6% slower at M=4096 and effectively tied at M=16384
  - confirms that role-fused all-compute waves help only when the spatial grid
    under-fills the 256 CUs
- LDS reads and WMMA can overlap, but realistic lane-distinct reads remain
  below the 60% target: the best fixed-register 4x4 pipeline reaches
  **2.237 PFLOPS (50.2% of 4.454 PFLOPS)**. A 3.58-PFLOPS diagnostic result
  uses same-address LDS broadcasts and is not a production GEMM roof.

## Environment

| Component | Version / identity |
|---|---|
| GPU | AMD gfx1250, 256 CUs, physical GPU3 |
| Host perf setup | `455_perf_set.sh` applied; SDMA/SVM/XNACK enabled; NUMA balancing off |
| Docker image | `rocm/fw-bringup:gfx1250-atom-dev-20260729` |
| Docker image ID | `sha256:eaef911cf1cb0a3629ae62bf99649799824818d010c7b50ccbc9fba9c90887f9` |
| Python | 3.12.3 |
| PyTorch | `2.11.0+rocm7.15.0a20260712` |
| HIP runtime | 7.15.0 |
| hipcc | HIP 7.15.0, AMD clang 23.0.0git `aa451e1f...` |
| Triton | 3.8.0 |
| Installed FlyDSL | 0.2.4 |
| PR #875 FlyDSL runtime | 0.3.0, commit `e3025ff57366011bca44607736d381beffeb7f50` |
| Aiter branch | `pr-4246` |
| Aiter commit | `0f144e5e4449a8b0a125715997c2386cc190f6f6` |
| FlyDSL examples base | `82a86966f72eed9e3e201991556fb10dd0fac647` |

Source hashes:

| File | SHA-256 |
|---|---|
| benchmark-time `kernels/gemm_a16w16_gfx1250.py` | `0dc638c71165998814a3c8cd0154428aaf9e92aa30e36433f0767acb8b9cb172` |
| current `kernels/gemm_a16w16_gfx1250.py` | `ec9140a161520fa1698c00de29c15003843e9f42cdccdf281cfa522d0def91ce` |
| benchmark-time `benchmark_model_gemm_backends.py` | `77d8bf98e4da8b71c7dfc631a52ecdb4b3bea5bed5056695babd9db19649b8cb` |
| current `benchmark_model_gemm_backends.py` | `dc367c2d20f5cc679364852a2fce11f27528505a36ba4e39531342d185dd07ca` |
| `augment_gemm_benchmark_metrics.py` | `f6740204934ed41a8dc04cc236cf1e002f250d393d28ec1388997d61e9858aeb` |
| `455_perf_set.sh` | `6416051685c67dc77a9a525b3b9abe8bdb93d93f673d25f7eefa61d15dce4671` |
| `dsv4_gemm_benchmark_results_perfset_20260816.json` | `8035755d853c5d50efebe82796bf898e8386310839fccc4453a81c4670a8c68e` |
| PR #875 `kernels/gemm/gemm_a16w16_gfx1250.py` | `ca41c4f8a8eed6acad2bd2fc4f860f443fee4d66eaf674af79c67cb540fea1ac` |
| `benchmark_flydsl_pr875_dsv4.py` | `ccab3e9371c30d35f60a58b01fe0abd6ddc3c975632b0ddd1f1d684822f91e11` |
| `dsv4_flydsl_pr875_results_perfset_20260816.json` | `d5abd497d418f3abbf15a9fefc1743905044574548d7012d6a1bae829bd1a971` |
| `kernels/gemm_a16w16_gfx1250_hybrid.py` | `4909163eb464c8f52749b13d16904e57aa34c15f6d2ef411a604eebb7c50081d` |
| `benchmark_a16w16_hybrid_gfx1250.py` | `854cb7815b51a27fb50c0e26deb9bc6bcc7c48404efcc7d10e13c4f9bae5d374` |
| `test_gemm_a16w16_gfx1250.py` | `51a9f4000cb9e3b4a0d10d4e1584bf4abf870de4dfaa0252247c7fe7b69013ed` |
| `profile_a16w16_gfx1250.py` | `53ffc7aab3483d84205eb6be498a54f786ecacb519f6d698d2ee0068bbc2147c` |
| `a16w16_hybrid_gfx1250_results_20260816.json` | `d4a3e078c0dabaee257a166b55a879baf8dae6cb33aa9e1b719ba168d199d8fb` |
| `a16w16_hybrid_directb_results_20260817.json` | `9c2e36529c449ccfa7b0fcf86b37029786dd1bb41b232886aec0128432c9a693` |
| `a8w8_tuned_reproduction_20260817.json` | `13fc63a88391a25fdfd82a901057e862436fafd1105d2c7507c21160967a9f4d` |
| `ubench/03_matrix_core/matrix_core_bench.py` (roof extensions) | `f8e5f311c0f86469fc428b96209eef4d55666f91143109e3670a9916569803af` |
| `repro_flydsl_vgpr_only_gfx1250.py` | `be032e05c6951a740137964b8c378a898c5b6afc6522809b5a6890e952e808b9` |

## Representative DSv4 shapes

Arithmetic intensity:

```text
AI = 2*M*N*K / (2*(M*K + N*K + M*N)) FLOP/byte
```

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

## Fair tuning methodology

### Triton

For every shape the benchmark compiles and measures a shape-dependent candidate
set, then reruns the fastest correct candidate with the final settings.

Candidate axes include:

- block M: 16, 32, 64, 128, 256
- block N: 32, 64, 128, 256
- block K: 64, 128, 256
- 4/8 waves
- 2/3/4 stages where applicable
- waves-per-EU 1/2/4/6/8
- split-K 1/2/4/8 for small M

This is a same-session compact tuner, not the full offline Triton search.

### FlyDSL

FlyDSL is also tuned per shape. Candidate axes include:

- per-wave register tile M/N
- K tile 64/128
- 2/3 pipeline stages
- TileM/TileN wave decomposition
- sync/cross-stage overlap
- grouped-M swizzle 32/64
- waves-per-EU, kernarg preload, LLVM scheduling, and loop unroll

K padding and unsupported-layout materialization occur outside the timed
region. Runtime TDM bounds handle M/N edge tiles.

### Opus

When the DSv4 merged CSV selects Opus, the exact `kernelId` and `splitK` are
launched. When the merged CSV selects Triton, the displayed Opus value is its
shape heuristic and is marked with `*`; it is not an exhaustive Opus retune.

### Timing

- 20 warmup calls
- 50 final samples
- tuning: 3 warmups, 8 samples
- PR #875 candidates: 5 consecutive correctness checks before timing
- BF16 inputs, FP32 accumulation, BF16 output, no bias
- `kernel_us`: median sum of profiler GPU kernel durations per call
- `e2e_us`: median HIP-event interval over ten calls, divided by ten
- `TFLOPS = 2*M*N*K / time_us / 1e6`
- `effective bandwidth = 2*(M*K + N*K + M*N) / time_us / 1e3 GB/s`

## Kernel-only latency

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

## Kernel-only throughput

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

## Kernel-only effective memory bandwidth

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

## End-to-end GPU interval

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

Profiler overhead can make `kernel_us` slightly larger than batched `e2e_us`
for long kernels. Differences below approximately 5% should be treated as
noise.

## Selected FlyDSL configurations

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

## FlyDSL PR #875 follow-up

This follow-up benchmarks exact PR head
`1273aa33d5eb8c6f5b724f890fdf159439f8bcc4`. The kernel source matches that
commit byte-for-byte. The Docker image, PyTorch/HIP versions, physical GPU3,
input seed, timing method, warmups, and sample counts match the original
backends, which were rerun in the same perf-set session.

PR #875 requires the FlyDSL 0.3 API/ABI, so runtime commit
`e3025ff57366011bca44607736d381beffeb7f50` was mounted read-only into the
original benchmark image. The raw JSON records both the installed distribution
metadata and the imported runtime.

The compact tuner measured 14-16 candidates per shape over:

- tile M/N/K and wave decomposition
- two/three buffers
- bandwidth-bound and compute-bound pipelines
- waves-per-EU, kernarg preload, scheduler strategy, and loop unroll
- selected split-K factors for M <= 128

### Boundary handling

An initial unpadded run faulted on `(32,64,7168)`: PR #875 creates TDM atoms
with extents `[None, None]`, while its wrapper does not pad M/N inputs.
Candidate-dependent A/B padding was therefore materialized once outside the
timed region, matching the existing report's padding policy. Without that
harness safeguard, irregular M/N shapes are not stable.

Each candidate now must pass five consecutive correctness checks before
tuning. Of 118 candidate/shape trials, 116 passed. Split-K 28 and 56 on
`(32,64,7168)` were excluded. A separate 100-run stress test failed 97 times
for split 28 and 96 times for split 56, confirming an intermittent race rather
than normal floating-point noise.

### Performance

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
15.7% slower by end-to-end geomean versus local FlyDSL. It adds one overall
kernel-only win at `(2048,1024,7168)`.

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

Three buffers consistently win the decode and middle-M dense cases.
Compute-bound fragment rotation wins only at M=16384. A waves-per-EU value of
2 wins M=4096; kernarg-preload, scheduler-strategy, and loop-unroll variants do
not win a final shape.

## A8W8-inspired hybrid all-compute experiment

The A8W8 gfx1250 kernel suggested a hybrid role model: wave 0/1 issue A/B TDM
loads, but every wave also owns output accumulators and executes WMMA. A focused
A16W16 prototype tested four/eight compute waves, K tiles 64/128, two/three
buffers, and a quadrant schedule that issues the next TDM between WMMA groups.

The table below uses a same-process local baseline, so small differences from
the full-suite table are normal.

| Shape | Local kernel us | Hybrid kernel us | Local TFLOPS | Hybrid TFLOPS | Hybrid latency delta |
|---|---:|---:|---:|---:|---:|
| (2048,1024,7168) | 48.91 | **37.35** | 614.66 | **804.85** | **-23.6%** |
| (4096,2048,4096) | **84.99** | 91.43 | **808.59** | 751.60 | +7.6% |
| (16384,2048,4096) | **331.42** | 337.50 | **829.41** | 814.44 | +1.8% |

All ten candidates passed five consecutive correctness checks. The winning
M=2048 configuration is `128x128x128`, eight compute waves (`4x2`), and three
buffers. The quadrant/mid-WMMA TDM schedule did not win that shape; most of the
gain comes from eliminating dedicated producer waves and reducing per-thread
register pressure.

### Resource and bottleneck evidence

For `(2048,1024,7168)`, both the local and hybrid 128x128x128 three-buffer
kernels use approximately 204 KiB LDS and launch only 128 workgroups, so at
most half of the 256 CUs receive a workgroup.

- final ISA VGPR count: local 456, four-wave hybrid 200
- rocprof dispatch VGPR field: local 232, four-wave hybrid 104, eight-wave
  hybrid 64
- median SQ cycles: local 32.66M, eight-wave hybrid 22.74M (-30%)
- median GL2 read requests: 220.3K vs 221.0K, effectively unchanged

This confirms register/wave under-utilization at M=2048 rather than an HBM
bandwidth bottleneck. Eight-wave role fusion gives more useful WMMA waves on
each active CU and raises throughput by 30.9%.

At M=4096/16384 the grid already has enough workgroups to fill all CUs. The
local 2+2 kernel then wins or ties because each consumer owns a deeper 4x8
accumulator tile, reuses each LDS fragment across more WMMAs, and allows TDM
producers to run ahead without full-workgroup barriers. The hybrid's lower VGPR
count no longer compensates for its weaker operand reuse and barrier overhead.

The remaining gap to the 4.454 PFLOPS pure-WMMA microbenchmark is therefore not
only register allocation. Full GEMM must issue BF16 LDS reads at roughly the
same cadence as K=32 WMMAs, wait on TDM/LDS visibility, and store output; the
microbenchmark keeps operands in registers and performs none of that work.

### Feasibility of 60% theoretical peak

The requested 60% target is 2.6724 PFLOPS. A fixed-register assembly ladder
isolated each level of the data path:

| Roof | Waves | PFLOPS | % of 4.454 PFLOPS |
|---|---:|---:|---:|
| Pure WMMA, Cartesian 4x4 sources | 4 | 4.322 | 97.0% |
| Constant-source FlyDSL `fx.gemm`, compact runtime loop | 4 | 3.996 | 89.7% |
| Constant-source FlyDSL grouped WMMA, compact runtime loop | 4 | **4.294** | **96.4%** |
| Dynamic per-lane VGPR operands, raw fixed-register asm | 4 | 1.595 | 35.8% |
| Dynamic per-lane VGPR operands, compact FlyDSL loop | 4 | 1.574 | 35.3% |
| Same-address partial-wait pipeline (broadcast diagnostic) | 4 | 2.265 | 50.8% |
| Lane-distinct streaming LDS, 4x4 tile | 4 | 1.528 | 34.3% |
| Lane-distinct streaming LDS, 4x5 tile | 5 | 1.685 | 37.8% |
| Extended-VGPR streaming LDS, 4x8 tile | 4 | 1.777 | 39.9% |
| Extended-VGPR streaming LDS, 5x8 tile | 4 | **1.853** | **41.6%** |
| Old hybrid `reuse_rmem` diagnostic (not a roof) | — | 1.559 | 35.0% |
| Compact FlyDSL 4x8 LDS+WMMA | 4 | 1.285 | 28.9% |
| Best full GEMM | — | 0.829 | 18.6% |

The earlier 2.237-PFLOPS lane-distinct result was not reproducible in fresh
runs and is discarded. Same-address LDS requests still reach about 2.265
PFLOPS, but that exercises the broadcast path and is not representative of GEMM
operand traffic.

With all 32 lanes reading distinct locations, carrying DS reads across groups
does provide substantial latency overlap. For example, the extended-VGPR 4x8
tile improves from about 1.15 PFLOPS with load-all/wait-all to 1.777 PFLOPS
with staggered partial waits. Increasing reuse to 5x8 raises the stable roof to
1.853 PFLOPS. Larger 6x8/7x7 tiles exceed the efficient register residency
point and regress.

The earlier 1.559-PFLOPS “FlyDSL VGPR-only roof” was also a benchmark artifact.
It reused the full hybrid GEMM kernel, so it retained a TDM/LDS preload and
epilogue. Its `reuse_repeat` loop was compile-time expanded; attempts to
amortize fixed costs generated tens of thousands of straight-line WMMA
instructions and eventually became instruction-fetch limited. A minimal
four-wave runtime-loop reproduction reaches 3.996 PFLOPS with ordinary
`fx.gemm` and 4.294 PFLOPS with grouped inline WMMA, so FlyDSL's register-only
path is not below the 60% target. Full commands and ISA comparison are in
[`FLYDSL_VGPR_ONLY_REPRO.md`](./FLYDSL_VGPR_ONLY_REPRO.md).

gfx1250 can encode the 4x8 tile through `s_set_vgpr_msb`; it uses roughly 353
logical VGPRs. A compact FlyDSL 4x8 register-only loop reaches 4.307 PFLOPS, so
extended registers do not limit WMMA throughput by themselves. Once real LDS
loads are added, however, both the fixed-register assembly and FlyDSL paths
remain below 1.9 PFLOPS.

Therefore overlap is real, but an exact BF16 design that reloads both operands
from LDS every K=32 round does not reach the 2.6724-PFLOPS target. The next
architectural experiment is asymmetric staging: keep one operand on the
LDS/TDM path while loading the other directly from global/L2 into VGPRs. ROCm's
gfx1250 CK pipeline family includes this AGmem/BGmem/CReg organization; it
avoids requiring the LDS read path to feed both WMMA operands. A first
wait-all synthetic version reaches only 1.629 PFLOPS; a useful test still needs
the CK-style asynchronous global prefetch and hot-loop overlap.

The hybrid should be retained only as a shape-specific M=2048 path, not as a
general replacement for the 2+2 pipeline.

### A8W8 tuned reference and asynchronous direct-B prototype

The gfx1250 entries in
`dsv4_a8w8_blockscale_bpreshuffle_tuned_gemm.csv` show that large mxfp8_128
shapes sustain 5+ PFLOPS:

- peak **5.649 PFLOPS** at `(32768,7168,4096)`
- **5.567 PFLOPS** at `(16384,7168,4096)`
- **5.539 PFLOPS** at `(8192,6144,7168)`
- **5.147 / 5.114 PFLOPS** at `(8192/16384,4096,8192)`

GPU3 reproduction using Aiter's exact `run_perftest` metric:

- `(8192,6144,7168)`: CSV 130.27 us / 5.539 PFLOPS; reproduced
  **118.36 us / 6.096 PFLOPS**
- `(16384,7168,4096)`: CSV 172.83 us / 5.567 PFLOPS; reproduced
  **169.95 us / 5.661 PFLOPS**
- `(32768,7168,4096)`: CSV 340.64 us / 5.649 PFLOPS; reproduced
  **382.20 us / 5.034 PFLOPS**

The three-shape geometric-mean latency ratio is 1.0008, so the aggregate CSV
performance reproduces almost exactly, although the largest individual shape
is 12.2% slower. These numbers use profiler self-device time, matching the
tuner. A single-call CUDA event includes Python dispatch idle time and is not
comparable.

Those winning kernels use a `256x256x128`, four-wave, four-buffer FlyDSL
configuration. Both operands still use TDM and LDS; the major advantage over
BF16 is the K=128 FP8 WMMA and much greater compute per operand load.

The A16 hybrid now also has a product-correct asymmetric path:

- A uses TDM to LDS and then VGPR
- B loads directly from global/L2 into two alternating VGPR fragments
- B(K+1) is issued before WMMA(K), then waited after the current compute group
- the automatic dispatcher enables it only for `M>=4096`, `N%256==0`,
  `K%64==0`

Same-session tuned results:

- `(4096,2048,4096)`: 813.58 -> **854.94 TFLOPS** kernel (+5.1%);
  761.31 -> **816.39 TFLOPS** end-to-end (+7.2%)
- `(16384,2048,4096)`: 831.52 -> **918.20 TFLOPS** kernel (+10.4%);
  816.35 -> **904.03 TFLOPS** end-to-end (+10.7%)
- M=2048 under-fills this `128x256` spatial tile, so dispatch retains the
  existing all-compute hybrid instead

That cross-block-K prefetch is now implemented, and only the A-producing wave
executes `tensor_wait`; the best `(16384,2048,4096)` result is approximately
**0.96 PFLOPS** after selecting grouped-M swizzle 8 plus XCD remapping. Deeper
5x6/6x6/8x6 register tiles all regress; the original 8x4 tile remains optimal.
Two additional synchronization experiments were correct but
slower:

- stage DATA/FREE named barriers: 0.623 PFLOPS
- barrier-free per-wave A-TDM with disjoint LDS rows: best 0.793 PFLOPS

The named-barrier control cost and the less favorable spatial tile offset the
saved workgroup barrier. The automatic dispatcher therefore retains the
cross-K direct-B path without named/distributed staging.

A full A/B-global double-buffer experiment was also implemented. With even
`reg_k` it is correct and carries prefetch across block-K boundaries, but its
two A plus two B fragment banks raise live VGPR pressure enough to reduce
`(16384,2048,4096)` to about 0.740 PFLOPS. The asymmetric A-LDS/B-global path
therefore remains the production candidate; reaching 2 PFLOPS needs a lower-
register CK-style distribution rather than simply duplicating both operands.

The production file was consequently reduced from 1727 to 639 lines. It keeps
only two measured paths:

- `(2048,1024,7168)`: 8-wave all-compute, **36.974 us / 0.813 PFLOPS**
- aligned `M>=4096`: cross-K direct-B, including
  `(16384,2048,4096)` at **287.875 us / 0.955 PFLOPS**

Both measurements passed bit-exact comparison with the BF16 PyTorch reference.
The same 8-wave route was later enabled for the exact M=128 and M=512
production shapes:

- `(128,2048,4096)`: 27.000 -> **16.443 us**
- `(512,2048,7168)`: 45.627 -> **29.663 us**

Direct-B still reserves the otherwise-unused B LDS region. Removing it reduces
LDS from about 165 KiB to 58 KiB, admits more resident work, and regresses the
large shape from about 0.99 to 0.74 PFLOPS under the power-limited operating
point. The reservation therefore remains as an intentional occupancy throttle.

The direct-B path now keeps four single-column B fragments instead of two full
4-column fragments. After a column's eight WMMAs, its slot is refilled with the
same column from the next K step while the other three columns compute. This
preserves global traffic and prefetch distance while reducing final ISA VGPRs
from 427 to 376 and `s_wait_loadcnt` instructions from 27 to 21. Alternating
same-session measurements improve:

- `(4096,2048,4096)`: 107.245 -> **99.243 us** (+8.1% throughput)
- `(16384,2048,4096)`: median 339.239 -> **311.149 us** (+9.0% throughput)

Two follow-up A-path experiments did not survive sustained testing. Explicit
front/back A fragments were correct but regressed about 2%, confirming that
LLVM's existing partial LDS waits were already better scheduled. Raising
block-K from 64 to 128 briefly reached 0.92 PFLOPS, then power-throttled to
about 0.71 PFLOPS in repeated runs. Neither change is retained.

Quadrant, direct-AB, named-barrier, distributed-TDM, operand-reuse, diagonal,
row-window, clustering, and scheduling-search branches were removed from the
production implementation after failing to beat these paths.

An exact-config roof ladder (`8x4` per wave, four compute waves) refines the
bottleneck:

- low-activity constant standalone VGPR-only: **4.08 PFLOPS**
- high-entropy random standalone VGPR-only: **1.57 PFLOPS**
- full-kernel framework with operands reused in VGPR: **1.55 PFLOPS**
- full-kernel LDS-only diagnostic: **1.27 PFLOPS**
- normal TDM+LDS pipeline: **0.73 PFLOPS**
- asymmetric direct-B pipeline: **0.93 PFLOPS**

The original interpretation of 1.57 PFLOPS as a dynamic-VGPR operand ceiling
was wrong. A sustained raw-assembly run with low-activity operands holds
2364 MHz at about 1145 W and reaches 4.252 PFLOPS. With random lane-distinct
BF16 operands, the same hot loop drops to 770 MHz at about 1896 W and reaches
1.573 PFLOPS. A high-entropy operand pattern generated entirely with VALU also
drops to 1.683 PFLOPS, ruling out VMEM dependency tracking.

The CDNA5 ISA additionally documents `WAVE_SCHED_MODE[2]` for disabling the
16-cycle XDL arbitration stall; the old FlyDSL helper selects bit 4. The
isolated loop changes by less than 0.1% with bit 2, but enabling it in the
current production schedule violates WMMA hazard spacing and produces incorrect
output. It is therefore not usable until the required dependency delays are
emitted. Correcting the direction of `reuseA/reuseB` hints also changes the
isolated benchmark by less than 0.1%.

Consequently, 4.3 PFLOPS is a nominal-clock, low-switching roof, while
1.55--1.60 PFLOPS is the sustained high-activity roof under the current
automatic power/clock policy. The measured LDS and TDM deltas remain useful
end-to-end comparisons, but they cannot be interpreted as pure operand-path
costs without also normalizing effective SCLK and power.

## Why fused split-K is disabled

PR #4246 originally added 1378 fused single-kernel split-K kids in the
`[21000,30000)` range. The design keeps non-last partials dirty-resident in
GL2, synchronizes split workgroups with a cluster barrier, and lets the last
workgroup stage/reduce partials from LDS.

Commit:

```text
1dea7996a766cd901b59002aa0206bdfc6e3237b
[opus][gfx1250] unregister the fused split-K family until its pipeline is fixed
```

states that the pipeline still “misbehaves.” This is a functional/stability
issue, not a measured performance rejection. The commit does not document a
single narrowed root cause. To prevent a wrong kernel from being selected and
to avoid compiling 1378 unusable candidates, it sets:

```python
GFX1250_SPLITK_FUSE_ENABLED = False
```

Consequences:

- the kid list is empty
- tuner selection excludes fused kids
- codegen emits no fused instances
- the `[21000,30000)` kid range is unclaimed
- factory and device pipeline source remain in-tree

The PR performance sweep explicitly excluded fused kids; this report also does
not benchmark them. Re-enabling is mechanically a one-line change, but should
not be done until the pipeline passes broad correctness/stability testing.

## Optimization guidance

### Small M

Opus remains the strongest kernel-only backend for skinny M:

```text
(32,64,7168)
Opus:   5.13 us
FlyDSL: 15.38 us
gap:    3.00x
```

FlyDSL needs a dedicated small-M split-K family rather than more tuning of the
current dense kernel. The likely design space is:

- M tile 16/32
- K tile 256/512
- split-K sufficient to fill 256 CUs
- 1xN A multicast
- a fused/single-pass reduction to avoid Opus's two-dispatch gap

### M >= 512

The tuned FlyDSL path is already competitive:

- wins kernel-only at M=512, 2048, and 4096
- is within 0.4% of Triton at M=16384

Large-M optimization should preserve the current 2+2 / 128x128x128 family and
focus on:

- shape dispatch for stages=2 vs 3
- swizzle=32 vs 64
- scheduler strategy for the large-M family
- reducing conservative LDS/WMMA waits
- retaining compiler scheduling knobs only when per-shape measurements win

## Reproduction

### Fetch PR #4246

```bash
cd /home/xiaobizh/aiter
git fetch origin pull/4246/head:pr-4246
git switch pr-4246
```

### Run DSv4 suite on physical GPU3

```bash
cd /home/xiaobizh/flydsl-examples
source ./455_perf_set.sh

docker run --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  -e ROCR_VISIBLE_DEVICES=3 \
  -e HSA_ENABLE_SDMA=1 \
  -e HSA_USE_SVM=1 \
  -e HSA_XNACK=1 \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONPATH=/home/xiaobizh/aiter \
  -e AITER_ROOT=/home/xiaobizh/aiter \
  -e FLYDSL_EXAMPLES_ROOT=/home/xiaobizh/flydsl-examples \
  -e BENCH_DOCKER_IMAGE=rocm/fw-bringup:gfx1250-atom-dev-20260729 \
  -v /home/xiaobizh/aiter:/home/xiaobizh/aiter \
  -v /home/xiaobizh/flydsl-examples:/home/xiaobizh/flydsl-examples \
  -w /home/xiaobizh/flydsl-examples \
  --entrypoint=/bin/bash \
  rocm/fw-bringup:gfx1250-atom-dev-20260729 \
  -lc '
    ln -s \
      /usr/local/lib/python3.12/dist-packages/_rocm_sdk_core/lib/libamdhip64.so.7 \
      /usr/local/lib/python3.12/dist-packages/_rocm_sdk_core/lib/libamdhip64.so \
      2>/dev/null || true
    ln -s \
      /usr/local/lib/python3.12/dist-packages/_rocm_sdk_core/lib/llvm/amdgcn \
      /usr/local/lib/python3.12/dist-packages/_rocm_sdk_core/amdgcn \
      2>/dev/null || true
    python3 benchmark_model_gemm_backends.py \
      --suite dsv4 \
      --device 0 \
      --warmup 20 \
      --iterations 50 \
      --tune-warmup 3 \
      --tune-iterations 8 \
      --batch-repeats 10 \
      --tune-batch-repeats 3 \
      --output dsv4_gemm_benchmark_results_perfset_20260816.json
  '
```

### Run the PR #875 follow-up

Prepare the exact PR source and the compatible FlyDSL 0.3 runtime:

```bash
cd /home/xiaobizh/FlyDSL
git fetch origin pull/875/head:pr-875
git worktree add --detach /home/xiaobizh/FlyDSL-pr875-full \
  1273aa33d5eb8c6f5b724f890fdf159439f8bcc4

mkdir -p /home/xiaobizh/.cache/flydsl-e3025ff
docker run --rm \
  -v /home/xiaobizh/.cache/flydsl-e3025ff:/out \
  --entrypoint=/bin/bash \
  rocm/fw-bringup:ahmed-mla-c06-2-latest \
  -lc 'cp -aL /root/FlyDSL/python/flydsl /out/'
```

Run on physical GPU3:

```bash
docker run --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  -e ROCR_VISIBLE_DEVICES=3 \
  -e HSA_ENABLE_SDMA=1 \
  -e HSA_USE_SVM=1 \
  -e HSA_XNACK=1 \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONPATH=/opt/flydsl-e3025ff \
  -e FLYDSL_RUNTIME_COMMIT=e3025ff57366011bca44607736d381beffeb7f50 \
  -e BENCH_DOCKER_IMAGE=rocm/fw-bringup:gfx1250-atom-dev-20260729 \
  -v /home/xiaobizh/.cache/flydsl-e3025ff:/opt/flydsl-e3025ff:ro \
  -v /home/xiaobizh/FlyDSL-pr875-full:/home/xiaobizh/FlyDSL-pr875:ro \
  -v /home/xiaobizh/flydsl-examples:/home/xiaobizh/flydsl-examples \
  -w /home/xiaobizh/flydsl-examples \
  --entrypoint=/bin/bash \
  rocm/fw-bringup:gfx1250-atom-dev-20260729 \
  -lc '
    ln -s \
      /usr/local/lib/python3.12/dist-packages/_rocm_sdk_core/lib/libamdhip64.so.7 \
      /usr/local/lib/python3.12/dist-packages/_rocm_sdk_core/lib/libamdhip64.so \
      2>/dev/null || true
    ln -s \
      /usr/local/lib/python3.12/dist-packages/_rocm_sdk_core/lib/llvm/amdgcn \
      /usr/local/lib/python3.12/dist-packages/_rocm_sdk_core/amdgcn \
      2>/dev/null || true
    python3 benchmark_flydsl_pr875_dsv4.py \
      --pr-root /home/xiaobizh/FlyDSL-pr875 \
      --device 0 \
      --warmup 20 \
      --iterations 50 \
      --tune-warmup 3 \
      --tune-iterations 8 \
      --batch-repeats 10 \
      --tune-batch-repeats 3 \
      --correctness-repeats 5 \
      --output dsv4_flydsl_pr875_results_perfset_20260816.json
  '
```

### Run the focused production-route benchmark

Using the original FlyDSL 0.2.4 benchmark container after applying
`455_perf_set.sh`:

```bash
python3 benchmark_a16w16_routes_gfx1250.py \
  --device 0 \
  --warmup 20 \
  --iterations 50 \
  --tune-warmup 3 \
  --tune-iterations 8 \
  --batch-repeats 10 \
  --tune-batch-repeats 3 \
  --correctness-repeats 5 \
  --output a16w16_hybrid_gfx1250_results_20260816.json
```

## Limitations

- Triton and FlyDSL use same-session compact candidate tuners, not exhaustive
  offline searches over every legal configuration.
- Opus is explicitly tuned only where the merged DSv4 CSV selects Opus.
- Gluon is included for context but is not part of the requested fair
  Triton/Opus/FlyDSL comparison.
- Gluon is incorrect on `(16384,2048,4096)` in two consecutive perf-set runs.
- Profiler overhead is unstable around `(4096,2048,4096)`; the final table uses
  a dedicated retest, while the batched e2e interval remains the more stable
  number.
- PR #875 required FlyDSL runtime commit `e3025ff`; its M/N inputs were padded
  outside timing to avoid the observed unbounded-TDM page fault.
- PR #875 candidates require five consecutive correctness passes; high split-K
  remains race-prone and is excluded.
- Fused split-K is disabled and unmeasured.
- PR #4246 documents approximately +/-5% per-shape timing noise.
