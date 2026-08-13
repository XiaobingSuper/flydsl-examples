# DeepSeek V4 BF16 GEMM tuned backend benchmark on gfx1250

Date: 2026-08-13

DeepSeek V4 was selected because PR #4246 contains a complete gfx1250
retune for its 180 BF16 GEMM shapes, while the Kimi-K3 block was not retuned
against the final cluster-grid round-up.

Compared backends:

1. Aiter generic Triton, tuned per shape over a compact candidate set
2. Aiter Opus from [PR #4246](https://github.com/ROCm/aiter/pull/4246)
3. Local FlyDSL gfx1250 kernel, tuned per shape over a compact candidate set
4. Aiter Gluon as an additional gfx1250-native reference

Raw results:
[`dsv4_gemm_benchmark_results_final.json`](./dsv4_gemm_benchmark_results_final.json).

Benchmark driver:
[`benchmark_model_gemm_backends.py`](./benchmark_model_gemm_backends.py)
with `--suite dsv4`.

## Executive summary

- All 32 backend/shape measurements passed correctness.
- Tuned Opus is the strongest small-M kernel:
  - `(32,64,7168)`: 5.15 us vs FlyDSL 17.55 us
  - `(128,2048,4096)`: 11.46 us vs FlyDSL 12.08 us
- Tuned FlyDSL is strongest through the middle/large M regime:
  - `(512,2048,7168)`: **31.13 us**
  - `(2048,1024,7168)`: **46.67 us**
  - `(4096,2048,4096)`: **87.54 us**, effectively tied with Triton
- Tuned Triton wins the largest shape:
  - `(16384,2048,4096)`: **346.48 us**
  - FlyDSL: 354.42 us, only 2.3% slower
- FlyDSL wins end-to-end GPU interval on six of eight shapes because it launches
  one kernel and has low dispatch overhead.
- The main remaining FlyDSL gap is small-M split-K. The current dense kernel is
  already competitive or best for M >= 512.

## Environment

| Component | Version / identity |
|---|---|
| GPU | AMD gfx1250, 256 CUs, physical GPU3 |
| Docker image | `rocm/fw-bringup:gfx1250-atom-dev-20260729` |
| Docker image ID | `sha256:eaef911cf1cb0a3629ae62bf99649799824818d010c7b50ccbc9fba9c90887f9` |
| Python | 3.12.3 |
| PyTorch | `2.11.0+rocm7.15.0a20260712` |
| HIP runtime | 7.15.0 |
| hipcc | HIP 7.15.0, AMD clang 23.0.0git `aa451e1f...` |
| Triton | 3.8.0 |
| Installed FlyDSL | 0.2.4 |
| Aiter branch | `pr-4246` |
| Aiter commit | `0f144e5e4449a8b0a125715997c2386cc190f6f6` |
| FlyDSL examples base | `82a86966f72eed9e3e201991556fb10dd0fac647` |

Source hashes:

| File | SHA-256 |
|---|---|
| `kernels/gemm_a16w16_gfx1250.py` | `0dc638c71165998814a3c8cd0154428aaf9e92aa30e36433f0767acb8b9cb172` |
| `kernels/gemm_a16w16_gfx1250_all_compute.py` | `5c28d64148f73733ed728741625db285ff3d581b7709e8f4f1d8f20cfd118ea6` |
| benchmark-time `benchmark_model_gemm_backends.py` | `77d8bf98e4da8b71c7dfc631a52ecdb4b3bea5bed5056695babd9db19649b8cb` |
| current `benchmark_model_gemm_backends.py` | `30334d3022f177147207bda17a49fd35231ac76222d45d172955fd3e972700aa` |
| `augment_gemm_benchmark_metrics.py` | `f6740204934ed41a8dc04cc236cf1e002f250d393d28ec1388997d61e9858aeb` |

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
- grouped-M swizzle 16/32/64
- selected cluster multicast variants for small M

Input/output padding is materialized once outside the timed region.

### Opus

When the DSv4 merged CSV selects Opus, the exact `kernelId` and `splitK` are
launched. When the merged CSV selects Triton, the displayed Opus value is its
shape heuristic and is marked with `*`; it is not an exhaustive Opus retune.

### Timing

- 20 warmup calls
- 50 final samples
- tuning: 3 warmups, 8 samples
- BF16 inputs, FP32 accumulation, BF16 output, no bias
- `kernel_us`: median sum of profiler GPU kernel durations per call
- `e2e_us`: median HIP-event interval over ten calls, divided by ten
- `TFLOPS = 2*M*N*K / time_us / 1e6`
- `effective bandwidth = 2*(M*K + N*K + M*N) / time_us / 1e3 GB/s`

## Kernel-only latency

Lower is better.

| Shape | Triton us | Gluon us | Opus PR #4246 us | Tuned FlyDSL us | Winner |
|---|---:|---:|---:|---:|---|
| (1,1024,4096) | 13.86 | **4.41** | 5.19 | 9.55 | Gluon |
| (32,64,7168) | 20.37 | 17.18 | **5.15** | 17.55 | Opus |
| (32,32320,7168) | **68.21** | 70.27 | 465.27* | 68.86 | Triton |
| (128,2048,4096) | 14.12 | 12.66 | **11.46** | 12.08 | Opus |
| (512,2048,7168) | 42.48 | 50.37 | 50.29 | **31.13** | FlyDSL |
| (2048,1024,7168) | 56.70 | 99.24 | 60.72 | **46.67** | FlyDSL |
| (4096,2048,4096) | 88.63 | 118.19 | 404.30* | **87.54** | FlyDSL |
| (16384,2048,4096) | **346.48** | 456.58 | 1601.91* | 354.42 | Triton |

\* Opus heuristic because the merged DSv4 row selects Triton.

## Kernel-only throughput

| Shape | Triton TFLOPS | Gluon TFLOPS | Opus TFLOPS | FlyDSL TFLOPS |
|---|---:|---:|---:|---:|
| (1,1024,4096) | 0.61 | **1.90** | 1.62 | 0.88 |
| (32,64,7168) | 1.44 | 1.71 | **5.71** | 1.67 |
| (32,32320,7168) | **217.38** | 210.99 | 31.87* | 215.31 |
| (128,2048,4096) | 152.08 | 169.65 | **187.47** | 177.82 |
| (512,2048,7168) | 353.84 | 298.44 | 298.90 | **482.95** |
| (2048,1024,7168) | 530.21 | 302.94 | 495.11 | **644.21** |
| (4096,2048,4096) | 775.38 | 581.45 | 169.97* | **784.96** |
| (16384,2048,4096) | **793.35** | 602.04 | 171.59* | 775.56 |

## Kernel-only effective memory bandwidth

This is an algorithmic lower-bound traffic rate, not a hardware-counter
measurement of HBM traffic. It assumes each BF16 A, B, and C element is moved
once. It excludes split-K workspace traffic and cache effects, so values can
exceed physical HBM bandwidth.

| Shape | Triton GB/s | Gluon GB/s | Opus GB/s | FlyDSL GB/s |
|---|---:|---:|---:|---:|
| (1,1024,4096) | 606.0 | **1906.4** | 1619.5 | 879.2 |
| (32,64,7168) | 67.8 | 80.3 | **268.2** | 78.7 |
| (32,32320,7168) | **6830.2** | 6629.3 | 1001.3* | 6765.2 |
| (128,2048,4096) | 1299.5 | 1449.7 | **1601.9** | 1519.4 |
| (512,2048,7168) | 913.2 | 770.3 | 771.4 | **1246.5** |
| (2048,1024,7168) | 850.6 | 486.0 | 794.3 | **1033.5** |
| (4096,2048,4096) | 757.2 | 567.8 | 166.0* | **766.6** |
| (16384,2048,4096) | **629.5** | 477.7 | 136.2* | 615.4 |

## End-to-end GPU interval

| Shape | Triton us | Gluon us | Opus PR #4246 us | Tuned FlyDSL us | Winner |
|---|---:|---:|---:|---:|---|
| (1,1024,4096) | 38.20 | 87.83 | 32.39 | **10.30** | FlyDSL |
| (32,64,7168) | 37.97 | 86.18 | 32.44 | **18.23** | FlyDSL |
| (32,32320,7168) | **69.66** | 87.74 | 573.56* | 70.62 | Triton |
| (128,2048,4096) | 38.48 | 93.88 | 33.18 | **13.30** | FlyDSL |
| (512,2048,7168) | 44.22 | 86.24 | 52.21 | **33.10** | FlyDSL |
| (2048,1024,7168) | 58.75 | 99.89 | 64.45 | **48.10** | FlyDSL |
| (4096,2048,4096) | **89.13** | 119.46 | 531.89* | 89.16 | tie |
| (16384,2048,4096) | 339.37 | 457.65 | 1712.21* | **332.24** | FlyDSL |

Profiler overhead can make `kernel_us` slightly larger than batched `e2e_us`
for long kernels. Differences below approximately 5% should be treated as
noise.

## Selected FlyDSL configurations

| Shape | reg M | reg N | reg K | waves MxN | stages | overlap | swizzle |
|---|---:|---:|---:|---|---:|---|---:|
| (1,1024,4096) | 1 | 1 | 4 | 1x2 | 3 | cross | 32 |
| (32,64,7168) | 1 | 4 | 4 | 2x1 | 3 | cross | 32 |
| (32,32320,7168) | 1 | 8 | 4 | 2x1 | 3 | cross | 32 |
| (128,2048,4096) | 1 | 4 | 4 | 2x1 | 3 | cross | 32 |
| (512,2048,7168) | 4 | 4 | 4 | 2x1 | 3 | cross | 32 |
| (2048,1024,7168) | 4 | 8 | 4 | 2x1 | 3 | sync | 32 |
| (4096,2048,4096) | 4 | 8 | 4 | 2x1 | 3 | cross | 64 |
| (16384,2048,4096) | 4 | 8 | 4 | 2x1 | 3 | sync | 32 |

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
Opus:   5.15 us
FlyDSL: 17.55 us
gap:    3.41x
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
- is within 2.3% of Triton at M=16384

Large-M optimization should preserve the current 2+2 / 128x128x128 family and
focus on:

- shape dispatch for stages=2 vs 3
- swizzle=32 vs 64
- reducing conservative LDS/WMMA waits
- avoiding regressions from all-compute variants

## Reproduction

### Fetch PR #4246

```bash
cd /home/xiaobizh/aiter
git fetch origin pull/4246/head:pr-4246
git switch pr-4246
```

### Run DSv4 suite on physical GPU3

```bash
docker run --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  -e ROCR_VISIBLE_DEVICES=3 \
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
      --output dsv4_gemm_benchmark_results_final.json
  '
```

## Limitations

- Triton and FlyDSL use same-session compact candidate tuners, not exhaustive
  offline searches over every legal configuration.
- Opus is explicitly tuned only where the merged DSv4 CSV selects Opus.
- Gluon is included for context but is not part of the requested fair
  Triton/Opus/FlyDSL comparison.
- Fused split-K is disabled and unmeasured.
- PR #4246 documents approximately +/-5% per-shape timing noise.
