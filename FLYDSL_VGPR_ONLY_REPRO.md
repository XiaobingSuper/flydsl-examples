# FlyDSL gfx1250 BF16 VGPR-only WMMA reproduction

This reproduction isolates four wave32 compute waves. Each wave keeps four A
fragments, four B fragments, and sixteen independent C accumulators in
registers. The timed loop contains no LDS or global operand loads.

Source:

[`repro_flydsl_vgpr_only_gfx1250.py`](./repro_flydsl_vgpr_only_gfx1250.py)

## Run

Apply the normal performance settings first, then run on physical GPU3:

```bash
docker run --rm --device=/dev/kfd --device=/dev/dri \
  -e ROCR_VISIBLE_DEVICES=3 \
  -e HSA_ENABLE_SDMA=1 -e HSA_USE_SVM=1 -e HSA_XNACK=1 \
  -e PYTHONPATH=/workspace \
  -v /home/xiaobizh/flydsl-examples:/workspace \
  -w /workspace --entrypoint=/bin/bash \
  rocm/fw-bringup:gfx1250-atom-dev-20260729 -lc '
    ln -s /usr/local/lib/python3.12/dist-packages/_rocm_sdk_core/lib/libamdhip64.so.7 \
      /usr/local/lib/python3.12/dist-packages/_rocm_sdk_core/lib/libamdhip64.so \
      2>/dev/null || true
    python3 repro_flydsl_vgpr_only_gfx1250.py \
      --mode grouped --source constants \
      --iterations 6144 --unroll 4
  '
```

`intrinsic` emits ordinary `fx.gemm` operations. `grouped` emits the same
4x4 Cartesian group as one inline-assembly block. `--unroll 4` puts 64 WMMA
instructions in the runtime loop body, matching the raw assembly benchmark.

To dump every compilation stage and the final ISA, add:

```text
-e FLYDSL_DUMP_IR=1
-e FLYDSL_DUMP_DIR=/dump
-e FLYDSL_RUNTIME_ENABLE_CACHE=0
-v /path/to/dump:/dump
```

The final code is written to:

```text
/path/to/dump/vgpr_only_kernel_0/21_final_isa.s
```

## Measurements

Same GPU3 session, four waves/workgroup, 256 workgroups. The first ladder uses
compile-time constant A/B values:

| Implementation | Loop-body WMMAs | PFLOPS |
|---|---:|---:|
| Raw fixed-register assembly | 64 | 4.3222 |
| FlyDSL `fx.gemm`, unroll 1 | 16 | 3.1565 |
| FlyDSL grouped inline, unroll 1 | 16 | 3.5494 |
| FlyDSL `fx.gemm`, unroll 4 | 64 | 3.9961 |
| FlyDSL grouped inline, unroll 4 | 64 | 4.1267 |
| FlyDSL grouped inline, unroll 16 | 256 | 4.2939 |
| FlyDSL `fx.gemm` 4x8 tile, unroll 16 | 512 | 4.3072 |

The compact grouped FlyDSL loop reaches 99.3% of the raw result when the small
per-loop setup is amortized. The 4x8 result also confirms that gfx1250 extended
VGPR addressing does not reduce WMMA issue throughput when operands can be
rematerialized and aliased.

Real per-lane source values loaded once from global memory produce a different
roof, even though the timed loop contains no further memory operations:

| Dynamic-source register tile | PFLOPS |
|---|---:|
| Raw fixed-register assembly, 4x4 | 1.595 |
| 2x4 | 1.539 |
| 4x4 | 1.562 |
| 4x5 | 1.574 |
| 4x8 | 1.574 |
| 8x4 | 1.571 |

The result is insensitive to tile depth and WMMA unrolling. The corrected raw
host-buffer benchmark independently gives 1.5942--1.5949 PFLOPS. This plateau
was initially interpreted as a dynamic-VGPR operand ceiling, but controlled
data-pattern and clock measurements disprove that interpretation.

## Root cause: data-dependent power throttling

The CDNA5 ISA specifies a 16-cycle BF16 WMMA and a software-controlled
multicycle arbitration stall. Neither that stall, WMMA data hazards, global
load scoreboarding, nor operand reuse explains the 1.6-PFLOPS result:

- the isolated throughput loop changes by less than 0.1% when selecting the
  documented `WAVE_SCHED_MODE[2]`, legacy bit 4, both bits, or neither
- correcting `reuseA`/`reuseB` so the hint predicts reuse by the *next*
  instruction changes throughput by less than 0.1%
- lane-varying operands generated entirely with VALU, with no memory loads,
  reach the same low range when their element pattern has high entropy

Sustained raw-assembly runs expose the actual cause:

| Operand pattern | Initialization | PFLOPS | sampled SCLK | sampled power |
|---|---|---:|---:|---:|
| low-entropy / lane-uniform | scalar/VALU | 4.252 | 2364 MHz | 1145 W |
| random BF16, lane-distinct | global buffer | 1.573 | 770 MHz | 1896 W |

The instruction stream, 192-register A/B/C allocation, workgroup geometry, and
hot loop are otherwise identical. A generated high-entropy lane/element hash
also falls to 1.683 PFLOPS without any global load in the kernel loop. Thus the
large gap is caused by switching activity driving the device power governor to
reduce SCLK, not by a slower ISA path for VMEM-produced VGPRs.

The documented bit 2 must not be enabled in the production GEMM without adding
the ISA-required WMMA hazard spacing: doing so with the current expert schedule
produces incorrect output. The existing bit-4 helper is therefore left
unchanged until its callers emit the required dependency delays.

The accurate interpretation is:

- **about 4.3 PFLOPS** is the nominal-clock WMMA issue roof for low-activity
  operands
- **about 1.55--1.60 PFLOPS** is the sustained, power-limited roof for fully
  random BF16 data under the current automatic clock/power policy
- neither number alone is an architecture-independent WMMA throughput limit

## ISA comparison

Raw hot loop:

```text
s_sub_co_u32 s13, s13, 1
v_wmma ... v[0:7],   v[128:135], v[160:167], v[0:7]
v_wmma ... v[8:15],  v[128:135], v[168:175], v[8:15]
...
v_wmma ... v[120:127], v[152:159], v[184:191], v[120:127]
# The 16-WMMA Cartesian group is repeated four times.
s_cmp_gt_u32 s13, 0
s_cbranch_scc1 L_kernel_start
```

Its physical register assignment is fixed and contiguous:

```text
C: v0-v127
A: v128-v159
B: v160-v191
VGPR count: 192
```

The grouped FlyDSL hot loop also contains 64 consecutive WMMAs, but LLVM
materializes several constant source lanes before each loop iteration and adds
one destination-dependency wait:

```text
.LBB0_2:
s_add_nc_u64 s[2:3], s[2:3], -1
v_dual_mov_b32 ...
...
s_wait_alu depctr_va_vdst(0)
;;#ASMSTART
v_wmma ...                         # 64 consecutive WMMAs
...
;;#ASMEND
s_cbranch_scc1 .LBB0_2
```

It uses 194 VGPRs with no spills. These extra loop instructions explain most
of the remaining 4.1267 versus 4.3222 PFLOPS gap.

The intrinsic `fx.gemm` lowering is less regular. LLVM temporarily keeps some
splat sources in SGPRs and reconstructs them with `v_mov_b64`; it also inserts
`v_nop` and `s_delay_alu` between WMMA groups. Four-way unrolling amortizes
much of that cost and raises it from 3.1565 to 3.9961 PFLOPS.

## Why the earlier “FlyDSL VGPR-only = 1.559 PFLOPS” result was low

That result came from the hybrid GEMM diagnostic, not a minimal register-only
kernel. It still paid one TDM/LDS preload and the GEMM epilogue. Increasing
`reuse_repeat` attempted to amortize those costs, but the DSL loop was
compile-time expanded into a very large straight-line WMMA body.

The minimal reproduction exposes the instruction-fetch cliff directly:

| Compile-time groups per runtime loop | Static WMMAs | PFLOPS |
|---:|---:|---:|
| 128 | 2,048 | 4.3360 |
| 512 | 8,192 | 3.8781 |
| 1,024 | 16,384 | 2.1861 |
| 2,048 | 32,768 | 1.6540 |
| 6,144 | 98,304 | 0.7795 |

The original 1.559-PFLOPS hybrid diagnostic mixed fixed kernel costs and static
ISA expansion, so it was not a clean roof measurement. The compact
dynamic-source reproduction converges to the same 1.54--1.57-PFLOPS plateau,
but the clock/power controls above show this is a high-activity DVFS limit, not
dynamic WMMA operand-delivery latency. Constant-source 4.3-PFLOPS remains an
unsuitable target for random-data GEMM because it runs at a radically different
SCLK and power operating point.
