#!/usr/bin/env python3
"""Minimal gfx1250 BF16 VGPR-only WMMA reproduction.

The timed loop performs no global/LDS operand loads.  Four waves per workgroup
reuse four A fragments, four B fragments, and sixteen C accumulators already
resident in VGPRs.  Two emission modes make LLVM code-generation differences
easy to inspect:

  intrinsic: sixteen independent fx.gemm calls
  grouped:   the same sixteen WMMAs in one inline-assembly block

Example:
  python3 repro_flydsl_vgpr_only_gfx1250.py --mode intrinsic
  python3 repro_flydsl_vgpr_only_gfx1250.py --mode grouped
"""

from __future__ import annotations

import argparse
import statistics

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as llvm_dialect
from flydsl._mlir.dialects import rocdl as raw_rocdl
from flydsl._mlir.dialects import vector as vector_dialect
from flydsl.expr import arith, as_ir_value, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from kernels.gemm_a16w16_gfx1250 import _run_compiled


WAVE_SIZE = 32
WAVES_PER_WORKGROUP = 4
THREADS_PER_WORKGROUP = WAVE_SIZE * WAVES_PER_WORKGROUP
WORKGROUPS = 256
WMMA_FLOPS = 2 * 16 * 16 * 32


def create_module(
    mode: str,
    source_mode: str,
    arb_mode: str,
    unroll: int,
    reg_m: int,
    reg_n: int,
):
    if mode not in ("intrinsic", "grouped", "reuse_a", "reuse_b"):
        raise ValueError(f"unsupported mode: {mode}")
    if source_mode not in (
        "constants",
        "lane",
        "lane_elements",
        "lane_hash",
        "global",
        "global_uniform",
    ):
        raise ValueError(f"unsupported source mode: {source_mode}")
    if arb_mode not in ("none", "legacy", "cdna5", "both"):
        raise ValueError(f"unsupported arb mode: {arb_mode}")
    if unroll < 1:
        raise ValueError("unroll must be positive")
    if reg_m < 1 or reg_n < 1:
        raise ValueError("register tile dimensions must be positive")

    pairs = [(m, n) for m in range(reg_m) for n in range(reg_n)]
    wmmas_per_group = reg_m * reg_n
    grouped_asm = "\n".join(
        "v_wmma_f32_16x16x32_bf16 "
        f"${m * reg_n + n}, "
        f"${wmmas_per_group + m}, "
        f"${wmmas_per_group + reg_m + n}, "
        f"${m * reg_n + n}"
        for m, n in pairs
    )
    grouped_constraints = ",".join(
        ["=&v"] * wmmas_per_group
        + ["v"] * (reg_m + reg_n)
        + [str(i) for i in range(wmmas_per_group)]
    )

    @flyc.kernel(known_block_size=[THREADS_PER_WORKGROUP, 1, 1])
    def vgpr_only_kernel(
        arg_out: fx.Tensor,
        arg_source: fx.Tensor,
        i32_iterations: fx.Int32,
    ):
        if arb_mode in ("legacy", "both"):
            fx.rocdl.disable_xdl_arb_stall()
        if arb_mode in ("cdna5", "both"):
            # CDNA5 ISA: WAVE_SCHED_MODE bit 2 disables the 16-cycle
            # post-WMMA arbitration stall.  LLVM hwreg encoding:
            # ID=26, offset=2, size=1 -> 154.
            llvm_dialect.call_intrinsic(
                None,
                "llvm.amdgcn.s.setreg",
                [
                    arith.unwrap(arith.constant(154, type=T.i32)),
                    arith.unwrap(arith.constant(1, type=T.i32)),
                ],
                [],
                [],
            )

        a_atoms = [
            fx.make_rmem_tensor(16, fx.BFloat16)
            for _ in range_constexpr(reg_m)
        ]
        b_atoms = [
            fx.make_rmem_tensor(16, fx.BFloat16)
            for _ in range_constexpr(reg_n)
        ]
        c_atoms = [
            fx.make_rmem_tensor(8, fx.Float32)
            for _ in range_constexpr(wmmas_per_group)
        ]
        if source_mode in ("global", "global_uniform"):
            source_base = (
                fx.block_idx.x * THREADS_PER_WORKGROUP
                + fx.thread_idx.x
            ) * ((reg_m + reg_n) * 16)
            source_iter = fx.get_iter(arg_source)
            for idx in range_constexpr(reg_m):
                a_atoms[idx].store(
                    Vec(
                        fx.ptr_load(
                            fx.add_offset(
                                source_iter,
                                source_base + idx * 16,
                            ),
                            result_type=fx.Vector.make_type(
                                16,
                                fx.BFloat16,
                            ),
                        )
                    )
                )
            for idx in range_constexpr(reg_n):
                b_atoms[idx].store(
                    Vec(
                        fx.ptr_load(
                            fx.add_offset(
                                source_iter,
                                source_base + (reg_m + idx) * 16,
                            ),
                            result_type=fx.Vector.make_type(
                                16,
                                fx.BFloat16,
                            ),
                        )
                    )
                )
            llvm_dialect.inline_asm(
                None,
                [],
                "s_wait_loadcnt 0",
                "",
                has_side_effects=True,
            )

            def _detach_load_dependency(atom):
                raw = arith._to_raw(
                    vector_dialect.bitcast(
                        T.vec(8, T.i32),
                        as_ir_value(atom.load()),
                    )
                )
                detached = llvm_dialect.inline_asm(
                    raw.type,
                    [raw],
                    "",
                    "=v,0",
                    has_side_effects=True,
                )
                atom.store(
                    Vec(
                        vector_dialect.bitcast(
                            T.vec(16, T.bf16),
                            detached,
                        )
                    )
                )

            for idx in range_constexpr(reg_m):
                _detach_load_dependency(a_atoms[idx])
            for idx in range_constexpr(reg_n):
                _detach_load_dependency(b_atoms[idx])
        elif source_mode == "lane":
            lane_value = (
                fx.thread_idx.x % WAVE_SIZE
            ).to(fx.BFloat16)
            for idx in range_constexpr(reg_m):
                a_atoms[idx].store(
                    Vec.filled(16, lane_value, fx.BFloat16)
                )
            for idx in range_constexpr(reg_n):
                b_atoms[idx].store(
                    Vec.filled(16, lane_value, fx.BFloat16)
                )
        elif source_mode == "lane_elements":
            lane = fx.thread_idx.x % WAVE_SIZE

            def _lane_fragment(seed):
                values = [
                    (lane + seed + element + 1).to(fx.BFloat16)
                    for element in range_constexpr(16)
                ]
                return Vec(
                    vector_dialect.from_elements(
                        T.vec(16, T.bf16),
                        [as_ir_value(value) for value in values],
                    )
                )

            for idx in range_constexpr(reg_m):
                a_atoms[idx].store(_lane_fragment(idx * 16))
            for idx in range_constexpr(reg_n):
                b_atoms[idx].store(
                    _lane_fragment((reg_m + idx) * 16)
                )
        elif source_mode == "lane_hash":
            lane = fx.thread_idx.x % WAVE_SIZE

            def _lane_hash_fragment(seed):
                values = [
                    (
                        (
                            (lane + 1) * 17
                            + (seed + element + 1) * 29
                        )
                        % 251
                        - 125
                    ).to(fx.BFloat16)
                    for element in range_constexpr(16)
                ]
                return Vec(
                    vector_dialect.from_elements(
                        T.vec(16, T.bf16),
                        [as_ir_value(value) for value in values],
                    )
                )

            for idx in range_constexpr(reg_m):
                a_atoms[idx].store(
                    _lane_hash_fragment(idx * 16)
                )
            for idx in range_constexpr(reg_n):
                b_atoms[idx].store(
                    _lane_hash_fragment((reg_m + idx) * 16)
                )
        else:
            for idx in range_constexpr(reg_m):
                a_atoms[idx].store(
                    Vec.filled(16, float(idx + 1), fx.BFloat16)
                )
            for idx in range_constexpr(reg_n):
                b_atoms[idx].store(
                    Vec.filled(16, float(idx + 5), fx.BFloat16)
                )
        for idx in range_constexpr(wmmas_per_group):
            c_atoms[idx].store(
                Vec.filled(8, float(idx), fx.Float32)
            )

        mma_atom = fx.make_mma_atom(
            fx.rocdl.WMMA(
                16,
                16,
                32,
                fx.BFloat16,
                fx.Float32,
            )
        )

        initial = [atom.load() for atom in c_atoms]
        results = initial
        for _, state in range(
            0,
            i32_iterations,
            1,
            init=initial,
        ):
            for idx in range_constexpr(wmmas_per_group):
                c_atoms[idx].store(state[idx])

            if mode in ("intrinsic", "reuse_a", "reuse_b"):
                for _ in range_constexpr(unroll):
                    ordered_pairs = (
                        pairs
                        if mode != "reuse_b"
                        else [(m, n) for n in range(reg_n) for m in range(reg_m)]
                    )
                    for m, n in ordered_pairs:
                        idx = m * reg_n + n
                        if mode == "intrinsic":
                            fx.gemm(
                                mma_atom,
                                c_atoms[idx],
                                a_atoms[m],
                                b_atoms[n],
                                c_atoms[idx],
                            )
                        else:
                            a_raw = arith._to_raw(
                                as_ir_value(a_atoms[m].load())
                            )
                            b_raw = arith._to_raw(
                                as_ir_value(b_atoms[n].load())
                            )
                            c_raw = arith._to_raw(
                                c_atoms[idx].load()
                            )
                            # The hint belongs to the current instruction and
                            # predicts reuse by the following instruction.
                            reused = (
                                mode == "reuse_a" and n + 1 < reg_n
                            ) or (
                                mode == "reuse_b" and m + 1 < reg_m
                            )
                            result = raw_rocdl.wmma_f32_16x16x32_bf16_(
                                c_raw.type,
                                a_raw,
                                b_raw,
                                c_raw,
                                reuse_a=ir.BoolAttr.get(
                                    reused if mode == "reuse_a" else False
                                ),
                                reuse_b=ir.BoolAttr.get(
                                    reused if mode == "reuse_b" else False
                                ),
                            )
                            c_atoms[idx].store(Vec(result))
            else:
                a_raw = [
                    arith._to_raw(
                        vector_dialect.bitcast(
                            T.vec(8, T.i32),
                            as_ir_value(a_atoms[idx].load()),
                        )
                    )
                    for idx in range_constexpr(reg_m)
                ]
                b_raw = [
                    arith._to_raw(
                        vector_dialect.bitcast(
                            T.vec(8, T.i32),
                            as_ir_value(b_atoms[idx].load()),
                        )
                    )
                    for idx in range_constexpr(reg_n)
                ]
                c_raw = [
                    arith._to_raw(c_atoms[idx].load())
                    for idx in range_constexpr(wmmas_per_group)
                ]
                result_type = ir.Type.parse(
                    "!llvm.struct<("
                    + ", ".join(str(value.type) for value in c_raw)
                    + ")>"
                )
                grouped_results = llvm_dialect.inline_asm(
                    result_type,
                    a_raw + b_raw + c_raw,
                    "\n".join(
                        grouped_asm for _ in range(unroll)
                    ),
                    grouped_constraints,
                    has_side_effects=True,
                )
                for idx in range_constexpr(wmmas_per_group):
                    c_atoms[idx].store(
                        Vec(
                            llvm_dialect.extractvalue(
                                c_raw[idx].type,
                                grouped_results,
                                [idx],
                            )
                        )
                    )

            results = yield [atom.load() for atom in c_atoms]

        for idx in range_constexpr(wmmas_per_group):
            c_atoms[idx].store(results[idx])

        # Keep every accumulator chain live without affecting the timed loop.
        thread_offset = (
            fx.block_idx.x * THREADS_PER_WORKGROUP
            + fx.thread_idx.x
        ) * wmmas_per_group
        for idx in range_constexpr(wmmas_per_group):
            value = fx.get_scalar(c_atoms[idx].load()[0])
            fx.ptr_store(
                value,
                fx.add_offset(
                    fx.get_iter(arg_out),
                    thread_offset + idx,
                ),
            )

    @flyc.jit
    def launch(
        out: fx.Tensor,
        source: fx.Tensor,
        i32_iterations: fx.Int32,
        stream: fx.Stream,
    ):
        out_view = fx.make_view(
            fx.get_iter(out),
            fx.make_layout(
                (
                    WORKGROUPS
                    * THREADS_PER_WORKGROUP
                    * wmmas_per_group,
                ),
                (1,),
            ),
        )
        vgpr_only_kernel(
            out_view,
            source,
            i32_iterations,
            value_attrs={
                "rocdl.waves_per_eu": 1,
                "rocdl.flat_work_group_size": (
                    f"{THREADS_PER_WORKGROUP},"
                    f"{THREADS_PER_WORKGROUP}"
                ),
            },
        ).launch(
            grid=(WORKGROUPS, 1, 1),
            block=(THREADS_PER_WORKGROUP, 1, 1),
            stream=stream,
        )

    launch.compile_hints["llvm_options"] = {
        "amdgpu-expert-scheduling-mode": True,
        "unroll-threshold": 0,
    }
    return launch


def benchmark(
    mode: str,
    source_mode: str,
    arb_mode: str,
    iterations: int,
    unroll: int,
    reg_m: int,
    reg_n: int,
    warmup: int,
    repeats: int,
):
    if iterations % unroll:
        raise ValueError("iterations must be divisible by unroll")
    loop_iterations = iterations // unroll
    wmmas_per_group = reg_m * reg_n
    launch = create_module(
        mode,
        source_mode,
        arb_mode,
        unroll,
        reg_m,
        reg_n,
    )
    out = torch.empty(
        WORKGROUPS * THREADS_PER_WORKGROUP * wmmas_per_group,
        dtype=torch.float32,
        device="cuda",
    )
    source_elements = (reg_m + reg_n) * 16
    if source_mode == "global_uniform":
        source = torch.randn(
            source_elements,
            dtype=torch.bfloat16,
            device="cuda",
        ).repeat(WORKGROUPS * THREADS_PER_WORKGROUP)
    else:
        source = torch.randn(
            WORKGROUPS * THREADS_PER_WORKGROUP * source_elements,
            dtype=torch.bfloat16,
            device="cuda",
        )
    stream = torch.cuda.current_stream()
    device_index = out.device.index or 0
    _run_compiled(
        launch,
        out,
        source,
        loop_iterations,
        stream,
        device_index=device_index,
    )
    compiled = launch._compiled_dispatch_cache[device_index]

    for _ in range(warmup):
        compiled(out, source, loop_iterations, stream)
    torch.cuda.synchronize()

    samples_us = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        compiled(out, source, loop_iterations, stream)
        end.record()
        end.synchronize()
        samples_us.append(start.elapsed_time(end) * 1_000.0)

    median_us = statistics.median(samples_us)
    total_flops = (
        WORKGROUPS
        * WAVES_PER_WORKGROUP
        * iterations
        * wmmas_per_group
        * WMMA_FLOPS
    )
    pflops = total_flops / median_us / 1.0e9
    print(
        f"mode={mode} source={source_mode} arb={arb_mode} "
        f"tile={reg_m}x{reg_n} "
        f"groups={iterations} unroll={unroll} "
        f"median_us={median_us:.3f} PFLOPS={pflops:.4f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("intrinsic", "grouped", "reuse_a", "reuse_b"),
        default="intrinsic",
    )
    parser.add_argument(
        "--source",
        choices=(
            "constants",
            "lane",
            "lane_elements",
            "lane_hash",
            "global",
            "global_uniform",
        ),
        default="constants",
    )
    parser.add_argument(
        "--arb",
        choices=("none", "legacy", "cdna5", "both"),
        default="legacy",
        help="CDNA5 bit 2 requires explicit WMMA hazard spacing",
    )
    parser.add_argument("--iterations", type=int, default=6144)
    parser.add_argument("--unroll", type=int, default=4)
    parser.add_argument("--reg-m", type=int, default=4)
    parser.add_argument("--reg-n", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.unroll <= 0:
        parser.error("--unroll must be positive")
    if args.reg_m <= 0 or args.reg_n <= 0:
        parser.error("--reg-m and --reg-n must be positive")
    if args.iterations % args.unroll:
        parser.error("--iterations must be divisible by --unroll")
    benchmark(
        args.mode,
        args.source,
        args.arb,
        args.iterations,
        args.unroll,
        args.reg_m,
        args.reg_n,
        args.warmup,
        args.repeats,
    )


if __name__ == "__main__":
    main()
