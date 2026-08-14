#!/usr/bin/env python3
"""Wave-specialized tile-layout FP16/BF16 GEMM for gfx1250.

The kernel computes ``C[M, N] = A[M, K] @ B[N, K].T``.  Thread/value
ownership is derived from the gfx1250 16x16x32 WMMA atom through FlyDSL's
``TiledMma`` and tiled-copy APIs; no gfx1250 lane-to-fragment mapping is
spelled out by hand.

Wave 0 is the A TDM producer, wave 1 is the B TDM producer, and the remaining
two waves are WMMA consumers. A configurable circular LDS pipeline is
coordinated by per-slot DATA/FREE_A/FREE_B named barriers. Runtime M/N bounds
avoid whole-matrix tile padding; K and unsupported physical strides are
materialized once by the public wrapper.
"""

from functools import lru_cache
from threading import Lock

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as llvm_dialect
from flydsl._mlir.dialects import vector
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import as_ir_value, const_expr, gpu, range_constexpr
from flydsl.expr.rocdl import tdm_ops
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import check_smem_capacity

WMMA_M = 16
WMMA_N = 16
WMMA_K = 32
WAVE_SIZE = 32
LDS_PAD = 8
NUM_PRODUCER_WAVES = 2
SUPPORTED_CONSUMER_WAVES = (2,)
DEFAULT_PIPELINE_STAGES = 3
MAX_PIPELINE_STAGES = 3
KERNARG_PRELOAD_COUNT = 8
SCHED_STRATEGIES = (None, "max-ilp", "max-memory-clause")
_compiled_dispatch_lock = Lock()


class _NamedBarrier:
    """One gfx1250 named-barrier target-extension global."""

    def __init__(self, symbol: str, member_count: int):
        self.symbol = symbol
        self.member_count = member_count

    def addrof(self):
        return llvm_dialect.AddressOfOp(
            ir.Type.parse("!llvm.ptr<3>"),
            self.symbol,
        ).result

    def init(self):
        fx.rocdl.s_barrier_init(self.addrof(), self.member_count)

    def join(self):
        fx.rocdl.s_barrier_join(self.addrof())

    def signal(self):
        # Pointer form avoids depending on backend-assigned immediate IDs.
        # member_count=0 reuses the count programmed by s_barrier_init.
        fx.rocdl.s_barrier_signal_var(self.addrof(), 0)

    def wait(self):
        # Positive BAR# selects the named barrier most recently joined.
        fx.rocdl.s_barrier_wait(1)


class _PipelineBarriers:
    """CUTLASS-style circular DATA/FREE_A/FREE_B barrier set."""

    def __init__(self, prefix: str, num_stages: int, consumer_waves: int):
        self.data = [
            _NamedBarrier(
                f"{prefix}_{1 + s:02d}_data{s}",
                NUM_PRODUCER_WAVES + consumer_waves,
            )
            for s in range(num_stages)
        ]
        self.free_a = [
            _NamedBarrier(
                f"{prefix}_{1 + num_stages + s:02d}_free_a{s}",
                1 + consumer_waves,
            )
            for s in range(num_stages)
        ]
        self.free_b = [
            _NamedBarrier(
                f"{prefix}_{1 + 2 * num_stages + s:02d}_free_b{s}",
                1 + consumer_waves,
            )
            for s in range(num_stages)
        ]
        self.all = self.data + self.free_a + self.free_b
        self.finalized = False

    def finalize(self):
        if self.finalized:
            return
        barrier_type = ir.Type.parse('!llvm.target<"amdgcn.named.barrier", 0>')
        i32 = ir.IntegerType.get_signless(32)
        linkage = ir.Attribute.parse("#llvm.linkage<internal>")
        for barrier in self.all:
            ir.Operation.create(
                "llvm.mlir.global",
                attributes={
                    "sym_name": ir.StringAttr.get(barrier.symbol),
                    "global_type": ir.TypeAttr.get(barrier_type),
                    "linkage": linkage,
                    "addr_space": ir.IntegerAttr.get(i32, 3),
                },
                regions=1,
            )
        self.finalized = True


def create_gemm_a16w16_module(
    K: int,
    in_dtype: str = "bf16",
    out_dtype: str = "bf16",
    *,
    reg_m: int = 2,
    reg_n: int = 4,
    reg_k: int = 4,
    waves_m: int = 2,
    waves_n: int = 1,
    num_stages: int = DEFAULT_PIPELINE_STAGES,
    overlap_mode: str = "cross",
    swizzle_m: int = 32,
    waves_per_eu: int | None = None,
    kernarg_preload: bool = False,
    sched_strategy: str | None = None,
    main_loop_unroll: bool = False,
):
    """Create a specialized gfx1250 tile-layout GEMM launcher.

    ``K`` is the only compile-time problem extent. Runtime M/N bounds are
    applied by TDM for inputs and bounded buffer stores for output.
    """

    arch = str(get_rocm_arch() or "")
    if not arch.startswith("gfx1250"):
        raise RuntimeError(f"gemm_a16w16_gfx1250 requires gfx1250, got {arch!r}")
    if in_dtype not in ("f16", "bf16"):
        raise ValueError(f"in_dtype must be 'f16' or 'bf16', got {in_dtype!r}")
    if out_dtype not in ("f16", "bf16", "f32"):
        raise ValueError(f"out_dtype must be 'f16', 'bf16' or 'f32', got {out_dtype!r}")
    if waves_per_eu is not None and waves_per_eu < 1:
        raise ValueError(f"waves_per_eu must be positive, got {waves_per_eu}")
    if sched_strategy not in SCHED_STRATEGIES:
        raise ValueError(f"unsupported sched_strategy: {sched_strategy!r}")

    consumer_waves = waves_m * waves_n
    if consumer_waves not in SUPPORTED_CONSUMER_WAVES:
        raise ValueError(
            f"consumer waves must be one of {SUPPORTED_CONSUMER_WAVES}; "
            f"got waves_m={waves_m}, waves_n={waves_n}"
        )
    if reg_k < 2 or reg_k % 2:
        raise ValueError(f"reg_k must be a positive multiple of 2, got {reg_k}")
    if not 2 <= num_stages <= MAX_PIPELINE_STAGES:
        raise ValueError(
            f"num_stages must be in [2, {MAX_PIPELINE_STAGES}], got {num_stages}"
        )
    if overlap_mode not in ("sync", "cross"):
        raise ValueError(f"unsupported overlap_mode: {overlap_mode!r}")
    use_intra_overlap = overlap_mode != "sync"
    use_cross_overlap = overlap_mode == "cross"
    if swizzle_m < 1:
        raise ValueError(f"swizzle_m must be positive, got {swizzle_m}")

    block_m = WMMA_M * reg_m * waves_m
    block_n = WMMA_N * reg_n * waves_n
    block_k = WMMA_K * reg_k
    if block_k & (block_k - 1):
        raise ValueError(f"block_k must be a power of two, got {block_k}")
    threads_per_block = (NUM_PRODUCER_WAVES + consumer_waves) * WAVE_SIZE

    if K % block_k:
        raise ValueError(
            f"padded K ({K}) must be divisible by block K ({block_k})"
        )
    num_k_tiles = K // block_k
    # Do not reserve inactive circular-buffer slots for short K.
    num_stages = min(num_stages, num_k_tiles)

    elem_cls = fx.Float16 if in_dtype == "f16" else fx.BFloat16
    out_elem_cls = {
        "f16": fx.Float16,
        "bf16": fx.BFloat16,
        "f32": fx.Float32,
    }[out_dtype]
    out_elem_bytes = out_elem_cls.width // 8

    elem_bytes = elem_cls.width // 8
    a_tdm_shape = (block_m, block_k)
    b_tdm_shape = (block_n, block_k)
    a_lds_stride = block_k + LDS_PAD
    b_lds_stride = block_k + LDS_PAD
    a_logical_stride = (a_lds_stride, 1)
    b_logical_stride = (b_lds_stride, 1)
    stage_a_bytes = block_m * a_lds_stride * elem_bytes
    stage_b_offset = (stage_a_bytes + 15) // 16 * 16
    stage_b_bytes = block_n * b_lds_stride * elem_bytes
    stage_pitch_bytes = (
        (stage_b_offset + stage_b_bytes + 1023) // 1024 * 1024
    )
    real_lds_bytes = num_stages * stage_pitch_bytes

    # One direct-copy request covers 256 bytes.  A/B use separate SIMD pairs.
    req_a = block_m * block_k * elem_bytes // 256
    req_b = block_n * block_k * elem_bytes // 256
    if req_a >= 256 or req_b >= 256:
        raise ValueError(
            "TDM request budget exceeded: "
            f"A={req_a}, B={req_b}; each operand must stay below 256"
        )
    # Two resident workgroups share a 256-request pair.  Force one WG/CU via
    # LDS occupancy when either operand is outside the safe <128/WG budget.
    force_one_wg = req_a >= 128 or req_b >= 128
    arena_bytes = max(
        real_lds_bytes,
        160 * 1024 + 1024 if force_one_wg else 0,
    )
    check_smem_capacity(arena_bytes, arch)

    k_iters = block_k // WMMA_K
    # Each gfx1250 WMMA operand fragment contains 16 FP16/BF16 values per
    # atom and therefore lowers to two 128-bit LDS reads.  Keep these counts
    # next to the layout geometry so the expert scheduler mirrors the
    # instructions emitted by the tiled copies below.
    ds_reads_a_per_k = 2 * reg_m
    ds_reads_b_per_k = 2 * reg_n
    acc_size = block_m * block_n // (consumer_waves * WAVE_SIZE)
    full_groups = num_k_tiles // num_stages
    tail_steps = num_k_tiles % num_stages

    barriers = _PipelineBarriers(
        f"gemm_a16w16_{in_dtype}_{out_dtype}_"
        f"{block_m}x{block_n}x{block_k}_c{consumer_waves}_s{num_stages}_nbar",
        num_stages,
        consumer_waves,
    )

    @flyc.kernel(known_block_size=[threads_per_block, 1, 1])
    def gemm_kernel(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_bt: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        i32_lda: fx.Int32,
        i32_ldb: fx.Int32,
        i32_ldc: fx.Int32,
        tiled_mma: fx.TiledMma,
    ):
        fx.rocdl.disable_xdl_arb_stall()
        tid = fx.thread_idx.x
        wave = fx.rocdl.readfirstlane(T.i32, tid // WAVE_SIZE)
        grid_m = (i32_m + (block_m - 1)) // block_m
        grid_n = (i32_n + (block_n - 1)) // block_n
        linear_bid = fx.block_idx.x
        blocks_per_group = grid_n * swizzle_m
        group_id = linear_bid // blocks_per_group
        first_bid_m = group_id * swizzle_m
        remaining_m = grid_m - first_bid_m
        actual_group_m = (remaining_m < swizzle_m).select(
            remaining_m,
            swizzle_m,
        )
        in_group = linear_bid - group_id * blocks_per_group
        bid_m = first_bid_m + in_group % actual_group_m
        bid_n = in_group // actual_group_m

        blk_m = bid_m * block_m
        blk_n = bid_n * block_n
        m_oob = i32_m - fx.Int32(blk_m)
        n_oob = i32_n - fx.Int32(blk_n)

        arena = fx.SharedAllocator(static=False)
        arena.allocate(arena_bytes)
        base_ptr = arena.base_ptr

        def _stage_ptr(stage, byte_offset):
            return fx.recast_iter(
                elem_cls,
                fx.add_offset(
                    base_ptr,
                    stage * stage_pitch_bytes + byte_offset,
                ),
            )

        s_a_tdm_stages = [
            fx.make_view(
                _stage_ptr(stage, 0),
                fx.make_layout(a_tdm_shape, (a_lds_stride, 1)),
            )
            for stage in range(num_stages)
        ]
        s_b_tdm_stages = [
            fx.make_view(
                _stage_ptr(stage, stage_b_offset),
                fx.make_layout(b_tdm_shape, (b_lds_stride, 1)),
            )
            for stage in range(num_stages)
        ]
        s_a_stages = [
            fx.make_view(
                _stage_ptr(stage, 0),
                fx.make_layout((block_m, block_k), a_logical_stride),
            )
            for stage in range(num_stages)
        ]
        s_b_stages = [
            fx.make_view(
                _stage_ptr(stage, stage_b_offset),
                fx.make_layout((block_n, block_k), b_logical_stride),
            )
            for stage in range(num_stages)
        ]
        # The target-extension barriers are initialized by one consumer wave.
        if wave == NUM_PRODUCER_WAVES:
            for barrier in barriers.all:
                barrier.init()
        gpu.barrier()

        if wave < NUM_PRODUCER_WAVES:
            lda64 = fx.Int64(i32_lda)
            ldb64 = fx.Int64(i32_ldb)
            a_off = fx.Int64(blk_m) * lda64
            b_off = fx.Int64(blk_n) * ldb64
            a_imm_step = fx.Int64(block_k * elem_bytes)
            b_imm_step = fx.Int64(block_k * elem_bytes)
            g_a_tile = fx.Tensor(
                fx.make_view(
                    fx.add_offset(fx.get_iter(arg_a), a_off),
                    fx.make_layout(a_tdm_shape, (lda64, 1)),
                )
            )
            g_b_tile = fx.Tensor(
                fx.make_view(
                    fx.add_offset(fx.get_iter(arg_bt), b_off),
                    fx.make_layout(b_tdm_shape, (ldb64, 1)),
                )
            )

            atom_a = fx.rocdl.make_tdm_atom(
                g_a_tile,
                [m_oob, None],
                strides=[lda64, None],
                num_warps=1,
                pad_interval=block_k,
                pad_amount=LDS_PAD,
                early_timeout=True,
            )
            atom_b = fx.rocdl.make_tdm_atom(
                g_b_tile,
                [n_oob, None],
                strides=[ldb64, None],
                num_warps=1,
                pad_interval=block_k,
                pad_amount=LDS_PAD,
                early_timeout=True,
            )
            def _publish_data(stage, outstanding):
                tdm_ops.tensor_wait(outstanding)
                barriers.data[stage].signal()

            def _wait_free(barrier):
                barrier.join()
                barrier.signal()
                barrier.wait()

            def _produce(
                atom,
                global_tile,
                stages,
                free_barriers,
                imm_step,
            ):
                def _issue(stage, logical_step):
                    fx.copy(
                        atom,
                        global_tile,
                        stages[stage],
                        # TDM immediate offsets are bytes, not elements.
                        imm_offset=(
                            fx.Int64(logical_step)
                            * imm_step
                        ),
                    )

                if const_expr(num_k_tiles >= num_stages):
                    for stage in range_constexpr(num_stages):
                        _issue(stage, stage)

                    # Slot 0 has landed; the remaining stages stay in flight.
                    _publish_data(0, num_stages - 1)

                    steady_steps = num_k_tiles - num_stages
                    steady_groups = steady_steps // num_stages
                    steady_tail = steady_steps % num_stages
                    for group in range(steady_groups):
                        for stage in range_constexpr(num_stages):
                            logical_step = (
                                num_stages
                                + group * num_stages
                                + stage
                            )
                            _wait_free(free_barriers[stage])
                            _issue(stage, logical_step)
                            _publish_data(
                                (stage + 1) % num_stages,
                                num_stages - 1,
                            )
                    for stage in range_constexpr(steady_tail):
                        logical_step = (
                            num_stages
                            + steady_groups * num_stages
                            + stage
                        )
                        _wait_free(free_barriers[stage])
                        _issue(stage, logical_step)
                        _publish_data(
                            (stage + 1) % num_stages,
                            num_stages - 1,
                        )

                    # The final num_stages-1 loads have no later issue to
                    # publish them, so retire and commit them together.
                    tdm_ops.tensor_wait(0)
                    for pending in range_constexpr(num_stages - 1):
                        logical_step = (
                            num_k_tiles - (num_stages - 1) + pending
                        )
                        barriers.data[logical_step % num_stages].signal()
                else:
                    for stage in range_constexpr(num_k_tiles):
                        _issue(stage, stage)
                    tdm_ops.tensor_wait(0)
                    for stage in range_constexpr(num_k_tiles):
                        barriers.data[stage].signal()

            if wave == 0:
                _produce(
                    atom_a,
                    g_a_tile,
                    s_a_tdm_stages,
                    barriers.free_a,
                    a_imm_step,
                )
            else:
                _produce(
                    atom_b,
                    g_b_tile,
                    s_b_tdm_stages,
                    barriers.free_b,
                    b_imm_step,
                )

            # Match the consumer epilogue before producer waves exit.
            gpu.barrier()
        else:
            consumer_tid = fx.Int32(tid) - fx.Int32(
                NUM_PRODUCER_WAVES * WAVE_SIZE
            )
            universal_copy = fx.make_copy_atom(
                fx.UniversalCopy128b(),
                elem_cls,
            )
            thr_mma = tiled_mma.thr_slice(consumer_tid)
            thr_s2r_a = fx.make_tiled_copy_A(
                universal_copy,
                tiled_mma,
            ).get_slice(consumer_tid)
            thr_s2r_b = fx.make_tiled_copy_B(
                universal_copy,
                tiled_mma,
            ).get_slice(consumer_tid)
            p_a_s2r = [
                thr_s2r_a.partition_S(stage) for stage in s_a_stages
            ]
            p_b_s2r = [
                thr_s2r_b.partition_S(stage) for stage in s_b_stages
            ]
            c_base = (
                fx.Int64(blk_m) * fx.Int64(i32_ldc)
                + fx.Int64(blk_n)
            )
            c_total = fx.Int64(i32_m) * fx.Int64(i32_ldc)
            c_remaining = (c_base < c_total).select(
                c_total - c_base,
                fx.Int64(0),
            )
            c_tile = fx.rocdl.make_buffer_tensor(
                fx.Tensor(
                    fx.make_view(
                        fx.add_offset(fx.get_iter(arg_c), c_base),
                        fx.make_layout(
                            (block_m, block_n),
                            (fx.Int64(i32_ldc), 1),
                        ),
                    )
                ),
                max_size=False,
                num_records_bytes=(
                    c_remaining * fx.Int64(out_elem_bytes)
                ),
            )
            frag_c = thr_mma.make_fragment_C(c_tile)
            frag_a = thr_mma.make_fragment_A(s_a_stages[0])
            frag_b = thr_mma.make_fragment_B(s_b_stages[0])
            frag_a_retile = thr_s2r_a.retile(frag_a)
            frag_b_retile = thr_s2r_b.retile(frag_b)
            frag_c.fill(0)

            def _wait_data(stage):
                barrier = barriers.data[stage]
                barrier.join()
                barrier.signal()
                barrier.wait()

            def _consume(
                stage,
                preloaded=False,
                next_stage=None,
            ):
                def _load_k_fragment(source_stage, ki):
                    # Preserve the TiledMma ownership mapping.  Different K
                    # slices occupy disjoint RMEM locations, so the next
                    # fragment can be loaded while WMMA consumes the current
                    # one without creating sliced RMEM tensors.
                    fx.copy(
                        universal_copy,
                        p_b_s2r[source_stage][None, None, ki],
                        frag_b_retile[None, None, ki],
                    )
                    fx.copy(
                        universal_copy,
                        p_a_s2r[source_stage][None, None, ki],
                        frag_a_retile[None, None, ki],
                    )

                # Prime K=0 before the scheduled region.  The steady-state
                # loop issues K+1 LDS reads ahead of the current K WMMA.
                if const_expr(not preloaded):
                    _wait_data(stage)
                    _load_k_fragment(stage, 0)
                    fx.rocdl.s_wait_dscnt(0)
                fx.rocdl.sched_barrier(0)
                for ki in range_constexpr(k_iters):
                    if const_expr(use_intra_overlap and ki + 1 < k_iters):
                        _load_k_fragment(stage, ki + 1)
                    else:
                        if const_expr(ki + 1 == k_iters):
                            # CUTLASS consumer_release equivalent: all LDS
                            # reads for this stage are complete, so producers
                            # may reuse the slot while final WMMA consumes
                            # registers.
                            barriers.free_a[stage].signal()
                            barriers.free_b[stage].signal()
                        if const_expr(
                            use_cross_overlap
                            and ki + 1 == k_iters
                            and next_stage is not None
                        ):
                            # Aiter's gfx1250 compute-bound schedule carries
                            # the next TDM tile's first layout fragment across
                            # the stage boundary.  Waiting here is normally
                            # free because the producer runs num_stages ahead.
                            _wait_data(next_stage)
                            _load_k_fragment(next_stage, 0)
                    fx.gemm(
                        tiled_mma,
                        frag_c,
                        frag_a[None, None, ki],
                        frag_b[None, None, ki],
                        frag_c,
                        traversal_order=fx.GemmTraversalOrder.KMN,
                    )
                    if const_expr(use_intra_overlap and ki + 1 < k_iters):
                        # The independent WMMA chain above gives the LDS
                        # reads time to complete before their first use.
                        fx.rocdl.s_wait_dscnt(0)
                    elif const_expr(
                        use_cross_overlap
                        and ki + 1 == k_iters
                        and next_stage is not None
                    ):
                        # Carry a ready K=0 fragment into the next _consume.
                        fx.rocdl.s_wait_dscnt(0)
                    elif const_expr(not use_intra_overlap and ki + 1 < k_iters):
                        _load_k_fragment(stage, ki + 1)
                        fx.rocdl.s_wait_dscnt(0)

                # Describe the same DS-read/WMMA sequence to LLVM's expert
                # scheduler. This controls temporal ordering only.
                for ki in range_constexpr(k_iters):
                    if const_expr(use_intra_overlap and ki + 1 < k_iters):
                        fx.rocdl.sched_dsrd(ds_reads_b_per_k)
                        fx.rocdl.sched_dsrd(ds_reads_a_per_k)
                    elif const_expr(
                        use_cross_overlap
                        and ki + 1 == k_iters
                        and next_stage is not None
                    ):
                        fx.rocdl.sched_dsrd(ds_reads_b_per_k)
                        fx.rocdl.sched_dsrd(ds_reads_a_per_k)
                    for _ in range_constexpr(reg_m):
                        fx.rocdl.sched_mfma(reg_n)
                    if const_expr(not use_intra_overlap and ki + 1 < k_iters):
                        fx.rocdl.sched_dsrd(ds_reads_b_per_k)
                        fx.rocdl.sched_dsrd(ds_reads_a_per_k)
                fx.rocdl.sched_barrier(0)

            if const_expr(full_groups > 0):
                for _, state in range(
                    0,
                    full_groups,
                    1,
                    init=[frag_c.load()],
                ):
                    frag_c.store(state[0])
                    for stage in range_constexpr(num_stages):
                        _consume(
                            stage,
                            preloaded=use_cross_overlap and stage > 0,
                            next_stage=(
                                stage + 1
                                if const_expr(
                                    use_cross_overlap
                                    and stage + 1 < num_stages
                                )
                                else None
                            ),
                        )
                    results = yield [frag_c.load()]
                frag_c.store(results)
            for stage in range_constexpr(tail_steps):
                _consume(
                    stage,
                    preloaded=use_cross_overlap and stage > 0,
                    next_stage=(
                        stage + 1
                        if const_expr(
                            use_cross_overlap
                            and stage + 1 < tail_steps
                        )
                        else None
                    ),
                )

            # This barrier pairs with the producer-side epilogue.
            gpu.barrier()

            if fx.Int32(blk_m) < i32_m:
                if fx.Int32(blk_n) < i32_n:
                    copy_out = fx.make_copy_atom(
                        fx.rocdl.BufferCopy(out_elem_cls.width),
                        out_elem_cls,
                    )
                    thr_r2g_c = fx.make_tiled_copy_C(
                        copy_out,
                        tiled_mma,
                    ).get_slice(consumer_tid)
                    p_c_g = thr_r2g_c.partition_S(c_tile)
                    if const_expr(out_elem_cls is fx.Float32):
                        frag_c_out = frag_c
                    else:
                        frag_c_out = fx.make_fragment_like(
                            frag_c,
                            out_elem_cls.ir_type,
                        )
                        acc_vec = Vec(frag_c.load())
                        out_values = [
                            acc_vec[i].to(out_elem_cls)
                            for i in range_constexpr(acc_size)
                        ]
                        frag_c_out.store(
                            vector.from_elements(
                                T.vec(
                                    acc_size,
                                    out_elem_cls.ir_type,
                                ),
                                [
                                    as_ir_value(value)
                                    for value in out_values
                                ],
                            )
                        )
                    frag_c_retile = thr_r2g_c.retile(frag_c_out)
                    fx.copy(copy_out, frag_c_retile, p_c_g)

    @flyc.jit
    def launch_gemm(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_bt: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        i32_lda: fx.Int32,
        i32_ldb: fx.Int32,
        i32_ldc: fx.Int32,
        stream: fx.Stream,
    ):
        mma_atom = fx.make_mma_atom(
            fx.rocdl.WMMA(
                WMMA_M,
                WMMA_N,
                WMMA_K,
                elem_cls,
                fx.Float32,
            )
        )
        tiled_mma = fx.make_tiled_mma(
            mma_atom,
            fx.make_layout(
                (waves_m, waves_n, 1),
                (waves_n, 1, 0),
            ),
        )

        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            barriers.finalized = False
            barriers.finalize()

        grid_m = (i32_m + (block_m - 1)) // block_m
        grid_n = (i32_n + (block_n - 1)) // block_n
        gemm_kernel(
            arg_c,
            arg_a,
            arg_bt,
            i32_m,
            i32_n,
            i32_lda,
            i32_ldb,
            i32_ldc,
            tiled_mma,
        ).launch(
            grid=(grid_m * grid_n, 1, 1),
            block=(threads_per_block, 1, 1),
            stream=stream,
            value_attrs={"rocdl.waves_per_eu": waves_per_eu},
        )

    launch_gemm.compile_hints["llvm_options"] = {
        "unroll-threshold": 300 if main_loop_unroll else 0,
        "amdgpu-kernarg-preload": kernarg_preload,
        "amdgpu-kernarg-preload-count": KERNARG_PRELOAD_COUNT,
        "amdgpu-expert-scheduling-mode": True,
    }
    if sched_strategy is not None:
        launch_gemm.compile_hints["llvm_options"][
            "amdgpu-sched-strategy"
        ] = sched_strategy

    return launch_gemm, block_m, block_n, block_k


@lru_cache(maxsize=1024)
def _cached_module(
    K: int,
    in_dtype: str,
    out_dtype: str,
    reg_m: int,
    reg_n: int,
    reg_k: int,
    waves_m: int,
    waves_n: int,
    num_stages: int,
    overlap_mode: str,
    swizzle_m: int,
    waves_per_eu: int | None,
    kernarg_preload: bool,
    sched_strategy: str | None,
    main_loop_unroll: bool,
):
    return create_gemm_a16w16_module(
        K,
        in_dtype,
        out_dtype,
        reg_m=reg_m,
        reg_n=reg_n,
        reg_k=reg_k,
        waves_m=waves_m,
        waves_n=waves_n,
        num_stages=num_stages,
        overlap_mode=overlap_mode,
        swizzle_m=swizzle_m,
        waves_per_eu=waves_per_eu,
        kernarg_preload=kernarg_preload,
        sched_strategy=sched_strategy,
        main_loop_unroll=main_loop_unroll,
    )


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _prepare_matrix_layout(
    tensor: torch.Tensor,
    padded_k: int,
) -> tuple[torch.Tensor, int]:
    """Return a supported physical layout and its outer leading dimension."""
    rows, k = tensor.shape
    if tensor.stride(1) != 1:
        # gfx1250's transpose LDS load path currently causes pathological
        # compile times when combined with this wave-specialized pipeline.
        # Preserve layout correctness with one materialization while still
        # avoiding the previous M/N tile padding.
        tensor = tensor.contiguous()

    if k != padded_k:
        padded = torch.zeros(
            (rows, padded_k),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        padded[:, :k].copy_(tensor)
        tensor = padded

    return tensor, tensor.stride(0)


def _run_compiled(launch, *args, device_index: int):
    cache = getattr(launch, "_compiled_dispatch_cache", None)
    if cache is not None and device_index in cache:
        cache[device_index](*args)
        return

    dispatch_after_wait = False
    with _compiled_dispatch_lock:
        cache = getattr(launch, "_compiled_dispatch_cache", None)
        if cache is None:
            cache = {}
            launch._compiled_dispatch_cache = cache
        compiled = cache.get(device_index)
        if compiled is None:
            # flyc.compile performs the first dispatch while materializing the
            # fast C-ABI callable, so do not launch it a second time here.
            compiled = flyc.compile(launch, *args)
            cache[device_index] = compiled
        else:
            dispatch_after_wait = True
    if dispatch_after_wait:
        compiled(*args)


def _select_config(
    m: int,
    n: int,
    k: int,
    reg_m: int | None,
    reg_n: int | None,
    reg_k: int | None,
    waves_m: int | None,
    waves_n: int | None,
) -> tuple[int, int, int, int, int]:
    if all(value is None for value in (reg_m, reg_n, reg_k, waves_m, waves_n)):
        # Keep the measured two-consumer geometry; four-consumer variants
        # consistently lost to the deeper per-wave accumulator tile.
        k_rep = 4 if k >= 128 else 2
        if m <= 16:
            n_rep = 1 if n <= 32 else 2 if n <= 64 else 4
            return 1, n_rep, k_rep, 1, 2
        m_rep = 1 if m <= 32 else 2 if m <= 64 else 4
        n_rep = 2 if n <= 32 else 4 if n <= 64 else 8
        return m_rep, n_rep, k_rep, 2, 1

    if waves_m is None and waves_n is None:
        waves_m, waves_n = 2, 1
    elif waves_m is None:
        if waves_n not in (1, 2):
            raise ValueError(f"waves_n must divide two, got {waves_n}")
        waves_m = 2 // waves_n
    elif waves_n is None:
        if waves_m not in (1, 2):
            raise ValueError(f"waves_m must divide two, got {waves_m}")
        waves_n = 2 // waves_m

    return (
        (1 if waves_m == 1 else 2) if reg_m is None else reg_m,
        4 if reg_n is None else reg_n,
        4 if reg_k is None else reg_k,
        waves_m,
        waves_n,
    )


def gemm_a16w16(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    out_dtype: torch.dtype | None = None,
    reg_m: int | None = None,
    reg_n: int | None = None,
    reg_k: int | None = None,
    waves_m: int | None = None,
    waves_n: int | None = None,
    num_stages: int = DEFAULT_PIPELINE_STAGES,
    overlap_mode: str = "cross",
    swizzle_m: int = 32,
    waves_per_eu: int | None = None,
    kernarg_preload: bool = False,
    sched_strategy: str | None = None,
    main_loop_unroll: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``a @ b.T`` with the gfx1250 tile-layout kernel.

    A and B must be two-dimensional CUDA tensors of the same FP16/BF16 dtype.
    K-contiguous inputs are consumed directly; other physical layouts are
    materialized once. Runtime TDM bounds avoid M and input-N tile padding.
    """

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"a and b must be rank-2, got {a.ndim=} and {b.ndim=}")
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"K dimensions differ: {a.shape[1]} != {b.shape[1]}")
    if a.dtype != b.dtype or a.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("a and b must have the same float16 or bfloat16 dtype")
    if not a.is_cuda or not b.is_cuda or a.device != b.device:
        raise ValueError("a and b must be CUDA tensors on the same device")

    if out_dtype is None:
        out_dtype = a.dtype
    if out_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"unsupported output dtype: {out_dtype}")

    m, k = a.shape
    n = b.shape[0]
    if min(m, n, k) <= 0:
        raise ValueError(f"M, N and K must be positive, got {(m, n, k)}")
    reg_m, reg_n, reg_k, waves_m, waves_n = _select_config(
        m,
        n,
        k,
        reg_m,
        reg_n,
        reg_k,
        waves_m,
        waves_n,
    )
    block_n = WMMA_N * reg_n * waves_n
    block_k = WMMA_K * reg_k
    padded_k = max(_round_up(k, block_k), 2 * block_k)
    a_kernel, lda = _prepare_matrix_layout(
        a,
        padded_k,
    )
    b_kernel, ldb = _prepare_matrix_layout(
        b,
        padded_k,
    )

    dtype_name = "f16" if a.dtype == torch.float16 else "bf16"
    out_dtype_name = {
        torch.float16: "f16",
        torch.bfloat16: "bf16",
        torch.float32: "f32",
    }[out_dtype]
    launch, _, _, _ = _cached_module(
        padded_k,
        dtype_name,
        out_dtype_name,
        reg_m,
        reg_n,
        reg_k,
        waves_m,
        waves_n,
        num_stages,
        overlap_mode,
        swizzle_m,
        waves_per_eu,
        kernarg_preload,
        sched_strategy,
        main_loop_unroll,
    )
    if out is not None:
        if out.shape != (m, n) or out.dtype != out_dtype or out.device != a.device:
            raise ValueError(
                f"out must have shape {(m, n)}, dtype {out_dtype}, and device {a.device}"
            )
    output_stride = _round_up(n, block_n)
    if (
        out is not None
        and output_stride == n
        and out.is_contiguous()
    ):
        c_kernel = out
    else:
        c_kernel = torch.empty(
            (m, output_stride),
            dtype=out_dtype,
            device=a.device,
        )
    stream = torch.cuda.current_stream()
    _run_compiled(
        launch,
        c_kernel,
        a_kernel,
        b_kernel,
        m,
        n,
        lda,
        ldb,
        output_stride,
        stream,
        device_index=a.device.index or 0,
    )
    result = c_kernel[:, :n]
    if out is not None and c_kernel is out:
        return out
    if out is not None:
        out.copy_(result)
        return out
    return result


__all__ = [
    "create_gemm_a16w16_module",
    "gemm_a16w16",
]
