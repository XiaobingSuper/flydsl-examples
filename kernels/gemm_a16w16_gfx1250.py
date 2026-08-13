#!/usr/bin/env python3
"""Wave-specialized tile-layout FP16/BF16 GEMM for gfx1250.

The kernel computes ``C[M, N] = A[M, K] @ B[N, K].T``.  Thread/value
ownership is derived from the gfx1250 16x16x32 WMMA atom through FlyDSL's
``TiledMma`` and tiled-copy APIs; no gfx1250 lane-to-fragment mapping is
spelled out by hand.

The workgroup extends the Opus gfx1250 model: wave 0 is the A TDM producer,
wave 1 is the B TDM producer, and the remaining two or four waves are WMMA
consumers.  A configurable circular LDS pipeline is coordinated by per-slot
DATA/FREE_A/FREE_B named barriers.  The public wrapper pads arbitrary matrix
sizes to the kernel tile and removes the padding from the result.
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
from flydsl.expr import arith, as_ir_value, const_expr, gpu, range_constexpr
from flydsl.expr.rocdl import cluster, tdm_ops
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import check_smem_capacity

WMMA_M = 16
WMMA_N = 16
WMMA_K = 32
WAVE_SIZE = 32
NUM_PRODUCER_WAVES = 2
SUPPORTED_CONSUMER_WAVES = (2, 4)
DEFAULT_PIPELINE_STAGES = 3
MAX_PIPELINE_STAGES = 5
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


def _fence_release():
    llvm_dialect.FenceOp(
        llvm_dialect.AtomicOrdering.release,
        syncscope="workgroup",
    )


def _fence_acquire():
    llvm_dialect.FenceOp(
        llvm_dialect.AtomicOrdering.acquire,
        syncscope="workgroup",
    )


def create_gemm_a16w16_module(
    M: int,
    N: int,
    K: int,
    in_dtype: str = "bf16",
    out_dtype: str = "bf16",
    *,
    reg_m: int = 2,
    reg_n: int = 4,
    reg_k: int = 4,
    waves_m: int = 2,
    waves_n: int = 1,
    a_k_pad: int = 8,
    b_k_pad: int = 8,
    num_stages: int = DEFAULT_PIPELINE_STAGES,
    traversal_order: str = "KMN",
    overlap_mode: str = "cross",
    swizzle_m: int = 32,
    expert_schedule: bool = True,
    barrier_fences: bool = False,
    cluster_m: int = 1,
    cluster_n: int = 1,
    grouped_inline: bool = False,
):
    """Create a specialized gfx1250 tile-layout GEMM launcher.

    ``M``, ``N`` and ``K`` are compile-time padded extents.  A and B must be
    contiguous row-major tensors with shapes ``[M, K]`` and ``[N, K]``.
    """

    arch = str(get_rocm_arch() or "")
    if not arch.startswith("gfx1250"):
        raise RuntimeError(f"gemm_a16w16_gfx1250 requires gfx1250, got {arch!r}")
    if in_dtype not in ("f16", "bf16"):
        raise ValueError(f"in_dtype must be 'f16' or 'bf16', got {in_dtype!r}")
    if out_dtype not in ("f16", "bf16", "f32"):
        raise ValueError(f"out_dtype must be 'f16', 'bf16' or 'f32', got {out_dtype!r}")

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
    if traversal_order not in ("KMN", "KNM", "MKN", "MNK", "NKM", "NMK"):
        raise ValueError(f"unsupported traversal_order: {traversal_order!r}")
    gemm_traversal_order = getattr(fx.GemmTraversalOrder, traversal_order)
    if overlap_mode not in ("sync", "intra", "cross", "ring"):
        raise ValueError(f"unsupported overlap_mode: {overlap_mode!r}")
    use_intra_overlap = overlap_mode != "sync"
    use_cross_overlap = overlap_mode == "cross"
    use_ring_overlap = overlap_mode == "ring"
    if grouped_inline and use_ring_overlap:
        raise ValueError("grouped_inline does not support overlap_mode='ring'")
    if swizzle_m < 1:
        raise ValueError(f"swizzle_m must be positive, got {swizzle_m}")
    if cluster_m < 1 or cluster_n < 1 or cluster_m * cluster_n > 16:
        raise ValueError(
            f"cluster shape {(cluster_m, cluster_n)} must contain 1..16 WGs"
        )
    use_cluster = cluster_m > 1 or cluster_n > 1
    cluster_a_pattern = sum(
        1 << (ly * cluster_m)
        for ly in range(cluster_n)
    )
    cluster_b_pattern = (1 << cluster_m) - 1

    block_m = WMMA_M * reg_m * waves_m
    block_n = WMMA_N * reg_n * waves_n
    block_k = WMMA_K * reg_k
    if block_k & (block_k - 1):
        raise ValueError(f"block_k must be a power of two, got {block_k}")
    threads_per_block = (NUM_PRODUCER_WAVES + consumer_waves) * WAVE_SIZE

    if M % block_m or N % block_n or K % block_k:
        raise ValueError(
            f"padded shape {(M, N, K)} must be divisible by block shape "
            f"{(block_m, block_n, block_k)}"
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

    block_k_pad_a = block_k + a_k_pad
    block_k_pad_b = block_k + b_k_pad
    elem_bytes = elem_cls.width // 8
    stage_a_bytes = block_m * block_k_pad_a * elem_bytes
    stage_b_offset = (stage_a_bytes + 15) // 16 * 16
    stage_b_bytes = block_n * block_k_pad_b * elem_bytes
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
    # One WMMA-K round issues 24 ds_read_b128 instructions for the default
    # 128x128 consumer tile.  A 3-deep source ring can therefore retain the
    # next round in flight without exceeding the 6-bit DScnt budget:
    # 2 * 24 <= 56.  Grouping two K rounds would require 96 outstanding reads.
    wmma_iters_per_chunk = 1
    chunk_k = wmma_iters_per_chunk * WMMA_K
    k_chunks = k_iters // wmma_iters_per_chunk
    # Each gfx1250 WMMA operand fragment contains 16 FP16/BF16 values per
    # atom and therefore lowers to two 128-bit LDS reads.  Keep these counts
    # next to the layout geometry so the expert scheduler mirrors the
    # instructions emitted by the tiled copies below.
    ds_reads_a_per_k = 2 * reg_m
    ds_reads_b_per_k = 2 * reg_n
    ds_reads_per_chunk = (
        ds_reads_a_per_k + ds_reads_b_per_k
    ) * wmma_iters_per_chunk
    acc_size = block_m * block_n // (consumer_waves * WAVE_SIZE)
    mma_atoms_per_k = reg_m * reg_n
    if grouped_inline and mma_atoms_per_k > 16:
        raise ValueError(
            "grouped_inline is limited to 16 atoms so accumulators and "
            "operands fit without scratch spilling"
        )
    if grouped_inline and (
        reg_n % 4 != 0
        or reg_m * 4 > 16
    ):
        raise ValueError(
            "grouped_inline requires reg_n divisible by 4 and at most "
            "16 outputs per inline block"
        )
    grouped_n_per_block = 4
    grouped_outputs_per_block = reg_m * grouped_n_per_block
    grouped_num_blocks = reg_n // grouped_n_per_block
    grouped_pairs = [
        (m_iter, n_iter)
        for m_iter in range(reg_m)
        for n_iter in range(grouped_n_per_block)
    ]
    grouped_mnemonic = (
        "v_wmma_f32_16x16x32_f16"
        if in_dtype == "f16"
        else "v_wmma_f32_16x16x32_bf16"
    )
    grouped_asm = "\n".join(
        f"{grouped_mnemonic} "
        f"${m_iter * grouped_n_per_block + n_iter}, "
        f"${grouped_outputs_per_block + m_iter}, "
        f"${grouped_outputs_per_block + reg_m + n_iter}, "
        f"${m_iter * grouped_n_per_block + n_iter}"
        for m_iter, n_iter in grouped_pairs
    )
    grouped_constraints = ",".join(
        ["=v"] * grouped_outputs_per_block
        + ["v"] * (reg_m + grouped_n_per_block)
        + [
            str(idx)
            for idx in range(grouped_outputs_per_block)
        ]
    )
    grid_m = M // block_m
    grid_n = N // block_n
    if grid_m % cluster_m or grid_n % cluster_n:
        raise ValueError(
            f"grid {(grid_m, grid_n)} must be divisible by cluster "
            f"{(cluster_m, cluster_n)}"
        )
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
        tiled_mma: fx.TiledMma,
    ):
        fx.rocdl.disable_xdl_arb_stall()
        tid = fx.thread_idx.x
        wave = fx.rocdl.readfirstlane(T.i32, tid // WAVE_SIZE)
        if const_expr(use_cluster):
            bid_m = fx.block_idx.x
            bid_n = fx.block_idx.y
            local_x, local_y = cluster.compute_cluster_position()
            local_x_i32 = arith.index_cast(T.i32, local_x)
            local_y_i32 = arith.index_cast(T.i32, local_y)
            a_pattern = arith.constant(cluster_a_pattern, type=T.i32)
            a_mask = arith.shli(a_pattern, local_x_i32)
            b_pattern = arith.constant(cluster_b_pattern, type=T.i32)
            cluster_m_i32 = arith.constant(cluster_m, type=T.i32)
            b_mask = arith.shli(
                b_pattern,
                arith.muli(local_y_i32, cluster_m_i32),
            )
        else:
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
            a_mask = b_mask = 0

        g_c = fx.rocdl.make_buffer_tensor(arg_c)
        t_c = fx.flat_divide(g_c, fx.make_tile(block_m, block_n))[None, None, bid_m, bid_n]

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

        s_a_stages = [
            fx.make_view(
                _stage_ptr(stage, 0),
                fx.make_layout((block_m, block_k), (block_k_pad_a, 1)),
            )
            for stage in range(num_stages)
        ]
        s_b_stages = [
            fx.make_view(
                _stage_ptr(stage, stage_b_offset),
                fx.make_layout((block_n, block_k), (block_k_pad_b, 1)),
            )
            for stage in range(num_stages)
        ]
        s_a_chunks = [
            [
                fx.make_view(
                    fx.add_offset(_stage_ptr(stage, 0), chunk * chunk_k),
                    fx.make_layout(
                        (block_m, chunk_k),
                        (block_k_pad_a, 1),
                    ),
                )
                for chunk in range(k_chunks)
            ]
            for stage in range(num_stages)
        ]
        s_b_chunks = [
            [
                fx.make_view(
                    fx.add_offset(
                        _stage_ptr(stage, stage_b_offset),
                        chunk * chunk_k,
                    ),
                    fx.make_layout(
                        (block_n, chunk_k),
                        (block_k_pad_b, 1),
                    ),
                )
                for chunk in range(k_chunks)
            ]
            for stage in range(num_stages)
        ]

        # The target-extension barriers are initialized by one consumer wave.
        if wave == NUM_PRODUCER_WAVES:
            for barrier in barriers.all:
                barrier.init()
        gpu.barrier()
        if const_expr(use_cluster):
            cluster.cluster_barrier()

        if wave < NUM_PRODUCER_WAVES:
            a_off = fx.Int64(bid_m * block_m) * fx.Int64(K)
            b_off = fx.Int64(bid_n * block_n) * fx.Int64(K)
            g_a_tile = fx.Tensor(
                fx.make_view(
                    fx.add_offset(fx.get_iter(arg_a), a_off),
                    fx.make_layout((block_m, block_k), (K, 1)),
                )
            )
            g_b_tile = fx.Tensor(
                fx.make_view(
                    fx.add_offset(fx.get_iter(arg_bt), b_off),
                    fx.make_layout((block_n, block_k), (K, 1)),
                )
            )

            atom_a = fx.rocdl.make_tdm_atom(
                g_a_tile,
                [None, None],
                strides=[K, None],
                num_warps=1,
                pad_interval=block_k,
                pad_amount=a_k_pad,
                early_timeout=True,
            )
            atom_b = fx.rocdl.make_tdm_atom(
                g_b_tile,
                [None, None],
                strides=[K, None],
                num_warps=1,
                pad_interval=block_k,
                pad_amount=b_k_pad,
                early_timeout=True,
            )
            if const_expr(use_cluster):
                atom_a = fx.atom_set_value(
                    atom_a,
                    "workgroup_mask",
                    a_mask,
                )
                atom_b = fx.atom_set_value(
                    atom_b,
                    "workgroup_mask",
                    b_mask,
                )

            def _publish_data(stage, outstanding):
                tdm_ops.tensor_wait(outstanding)
                if const_expr(barrier_fences):
                    _fence_release()
                barriers.data[stage].signal()

            def _wait_free(barrier):
                barrier.join()
                if const_expr(barrier_fences):
                    _fence_release()
                barrier.signal()
                barrier.wait()
                if const_expr(barrier_fences):
                    _fence_acquire()

            def _produce(atom, global_tile, stages, free_barriers):
                def _issue(stage, logical_step):
                    fx.copy(
                        atom,
                        global_tile,
                        stages[stage],
                        # TDM immediate offsets are bytes, not elements.
                        imm_offset=(
                            fx.Int64(logical_step)
                            * fx.Int64(block_k * elem_bytes)
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
                    if const_expr(barrier_fences):
                        _fence_release()
                    for pending in range_constexpr(num_stages - 1):
                        logical_step = (
                            num_k_tiles - (num_stages - 1) + pending
                        )
                        barriers.data[logical_step % num_stages].signal()
                else:
                    for stage in range_constexpr(num_k_tiles):
                        _issue(stage, stage)
                    tdm_ops.tensor_wait(0)
                    if const_expr(barrier_fences):
                        _fence_release()
                    for stage in range_constexpr(num_k_tiles):
                        barriers.data[stage].signal()

            if wave == 0:
                _produce(atom_a, g_a_tile, s_a_stages, barriers.free_a)
            else:
                _produce(atom_b, g_b_tile, s_b_stages, barriers.free_b)

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

            if const_expr(grouped_inline):
                grouped_zero = arith.constant_vector(
                    0.0,
                    T.vec(8, T.f32),
                )
                grouped_zero_raw = arith._to_raw(grouped_zero)
                grouped_state_type = ir.Type.parse(
                    "!llvm.struct<("
                    + ", ".join(
                        str(grouped_zero_raw.type)
                        for _ in range(grouped_outputs_per_block)
                    )
                    + ")>"
                )
                grouped_init = [
                    llvm_dialect.inline_asm(
                        grouped_state_type,
                        [grouped_zero_raw] * grouped_outputs_per_block,
                        "",
                        ",".join(
                            ["=v"] * grouped_outputs_per_block
                            + [
                                str(idx)
                                for idx in range(
                                    grouped_outputs_per_block
                                )
                            ]
                        ),
                        has_side_effects=True,
                    )
                    for _ in range(grouped_num_blocks)
                ]
                row_coords = fx.make_view(
                    0,
                    fx.make_layout((block_m, block_n), (1, 0)),
                )
                col_coords = fx.make_view(
                    0,
                    fx.make_layout((block_m, block_n), (0, 1)),
                )
                grouped_rows = thr_mma.partition_C(row_coords)
                grouped_cols = thr_mma.partition_C(col_coords)
            else:
                frag_c = thr_mma.make_fragment_C(t_c)
            if const_expr(use_ring_overlap):
                p_a_chunks = [
                    [
                        thr_s2r_a.partition_S(chunk)
                        for chunk in stage_chunks
                    ]
                    for stage_chunks in s_a_chunks
                ]
                p_b_chunks = [
                    [
                        thr_s2r_b.partition_S(chunk)
                        for chunk in stage_chunks
                    ]
                    for stage_chunks in s_b_chunks
                ]
                frag_a_ring = [
                    thr_mma.make_fragment_A(s_a_chunks[0][0])
                    for _ in range(3)
                ]
                frag_b_ring = [
                    thr_mma.make_fragment_B(s_b_chunks[0][0])
                    for _ in range(3)
                ]
                frag_a_ring_retile = [
                    thr_s2r_a.retile(fragment)
                    for fragment in frag_a_ring
                ]
                frag_b_ring_retile = [
                    thr_s2r_b.retile(fragment)
                    for fragment in frag_b_ring
                ]
            else:
                frag_a = thr_mma.make_fragment_A(s_a_stages[0])
                frag_b = thr_mma.make_fragment_B(s_b_stages[0])
                frag_a_retile = thr_s2r_a.retile(frag_a)
                frag_b_retile = thr_s2r_b.retile(frag_b)
            if const_expr(not grouped_inline):
                frag_c.fill(0)

            def _wait_data(stage):
                barrier = barriers.data[stage]
                barrier.join()
                if const_expr(barrier_fences):
                    _fence_release()
                barrier.signal()
                barrier.wait()
                if const_expr(barrier_fences):
                    _fence_acquire()

            def _grouped_mma(a_fragment, b_fragment, c_state):
                a_atoms = [
                    fx.make_rmem_tensor(16, elem_cls)
                    for _ in range(reg_m)
                ]
                b_atoms = [
                    fx.make_rmem_tensor(16, elem_cls)
                    for _ in range(reg_n)
                ]
                for atom_idx in range_constexpr(reg_m):
                    a_atoms[atom_idx].store(
                        a_fragment[None, atom_idx].load()
                    )
                for atom_idx in range_constexpr(reg_n):
                    b_atoms[atom_idx].store(
                        b_fragment[None, atom_idx].load()
                    )
                a_raw = [
                    arith._to_raw(
                        vector.bitcast(
                            T.vec(8, T.i32),
                            as_ir_value(a_atoms[idx].load()),
                        )
                    )
                    for idx in range_constexpr(reg_m)
                ]
                b_raw = [
                    arith._to_raw(
                        vector.bitcast(
                            T.vec(8, T.i32),
                            as_ir_value(b_atoms[idx].load()),
                        )
                    )
                    for idx in range_constexpr(reg_n)
                ]
                next_states = []
                for group_idx in range_constexpr(grouped_num_blocks):
                    c_raw = [
                        llvm_dialect.extractvalue(
                            grouped_zero_raw.type,
                            c_state[group_idx],
                            [idx],
                        )
                        for idx in range_constexpr(
                            grouped_outputs_per_block
                        )
                    ]
                    b_begin = group_idx * grouped_n_per_block
                    next_states.append(
                        llvm_dialect.inline_asm(
                            grouped_state_type,
                            a_raw
                            + b_raw[
                                b_begin : b_begin
                                + grouped_n_per_block
                            ]
                            + c_raw,
                            grouped_asm,
                            grouped_constraints,
                            has_side_effects=True,
                        )
                    )
                return next_states

            def _consume(
                stage,
                preloaded=False,
                next_stage=None,
                c_state=None,
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
                if const_expr(expert_schedule):
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
                            if const_expr(barrier_fences):
                                _fence_release()
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
                    if const_expr(grouped_inline):
                        c_state = _grouped_mma(
                            frag_a[None, None, ki],
                            frag_b[None, None, ki],
                            c_state,
                        )
                    else:
                        fx.gemm(
                            tiled_mma,
                            frag_c,
                            frag_a[None, None, ki],
                            frag_b[None, None, ki],
                            frag_c,
                            traversal_order=gemm_traversal_order,
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

                if const_expr(expert_schedule):
                    # Describe the same DS-read/WMMA sequence to LLVM's expert
                    # scheduler.  This controls temporal ordering only; all
                    # spatial ownership still comes from the tile layouts.
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
                return c_state

            def _consume_ring(stage, preloaded=False, next_stage=None):
                base_buf = (stage * k_chunks) % 3

                def _load_chunk(source_stage, chunk, buf):
                    fx.copy(
                        universal_copy,
                        p_b_chunks[source_stage][chunk],
                        frag_b_ring_retile[buf],
                    )
                    fx.copy(
                        universal_copy,
                        p_a_chunks[source_stage][chunk],
                        frag_a_ring_retile[buf],
                    )
                    if const_expr(expert_schedule):
                        fx.rocdl.sched_barrier(0)

                if const_expr(not preloaded):
                    _wait_data(stage)
                    _load_chunk(stage, 0, base_buf)

                if const_expr(expert_schedule):
                    fx.rocdl.sched_barrier(0)
                for chunk in range_constexpr(k_chunks):
                    cur = (base_buf + chunk) % 3
                    has_same_stage_next = chunk + 1 < k_chunks
                    has_next_stage = (
                        chunk + 1 == k_chunks
                        and next_stage is not None
                    )
                    if const_expr(has_same_stage_next):
                        _load_chunk(
                            stage,
                            chunk + 1,
                            (cur + 1) % 3,
                        )
                    elif const_expr(has_next_stage):
                        _wait_data(next_stage)
                        _load_chunk(next_stage, 0, (cur + 1) % 3)

                    # Drain only the current register round.  The next round's
                    # ds_reads remain in flight and overlap this WMMA group.
                    remaining_ds = (
                        ds_reads_per_chunk
                        if has_same_stage_next or has_next_stage
                        else 0
                    )
                    fx.rocdl.s_wait_dscnt(remaining_ds)

                    if const_expr(chunk + 1 == k_chunks):
                        if const_expr(barrier_fences):
                            _fence_release()
                        barriers.free_a[stage].signal()
                        barriers.free_b[stage].signal()

                    fx.gemm(
                        tiled_mma,
                        frag_c,
                        frag_a_ring[cur],
                        frag_b_ring[cur],
                        frag_c,
                        traversal_order=gemm_traversal_order,
                    )

                if const_expr(expert_schedule):
                    for chunk in range_constexpr(k_chunks):
                        has_same_stage_next = chunk + 1 < k_chunks
                        has_next_stage = (
                            chunk + 1 == k_chunks
                            and next_stage is not None
                        )
                        if const_expr(
                            has_same_stage_next or has_next_stage
                        ):
                            fx.rocdl.sched_dsrd(ds_reads_per_chunk)
                        for _ in range_constexpr(reg_m):
                            fx.rocdl.sched_mfma(
                                reg_n * wmma_iters_per_chunk
                            )
                    fx.rocdl.sched_barrier(0)

            if const_expr(grouped_inline):
                grouped_state = list(grouped_init)
            if const_expr(full_groups > 0):
                if const_expr(grouped_inline):
                    for _, state in range(
                        0,
                        full_groups,
                        1,
                        init=grouped_init,
                    ):
                        grouped_state = list(state)
                        for stage in range_constexpr(num_stages):
                            grouped_state = _consume(
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
                                c_state=grouped_state,
                            )
                        grouped_results = yield grouped_state
                    if const_expr(grouped_num_blocks == 1):
                        grouped_state = [grouped_results]
                    else:
                        grouped_state = list(grouped_results)
                else:
                    for _, state in range(
                        0,
                        full_groups,
                        1,
                        init=[frag_c.load()],
                    ):
                        frag_c.store(state[0])
                        for stage in range_constexpr(num_stages):
                            if const_expr(use_ring_overlap):
                                _consume_ring(
                                    stage,
                                    preloaded=False,
                                    next_stage=None,
                                )
                            else:
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
                if const_expr(use_ring_overlap):
                    _consume_ring(
                        stage,
                        preloaded=False,
                        next_stage=None,
                    )
                else:
                    if const_expr(grouped_inline):
                        grouped_state = _consume(
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
                            c_state=grouped_state,
                        )
                    else:
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

            if const_expr(grouped_inline):
                c_iter = fx.get_iter(arg_c)
                for m_iter in range_constexpr(reg_m):
                    for n_iter in range_constexpr(reg_n):
                        atom_idx = m_iter * reg_n + n_iter
                        group_idx = n_iter // grouped_n_per_block
                        local_atom_idx = (
                            m_iter * grouped_n_per_block
                            + n_iter % grouped_n_per_block
                        )
                        values = Vec(
                            llvm_dialect.extractvalue(
                                grouped_zero_raw.type,
                                grouped_state[group_idx],
                                [local_atom_idx],
                            )
                        ).to(out_elem_cls)
                        row_atom = grouped_rows[None, m_iter, n_iter]
                        col_atom = grouped_cols[None, m_iter, n_iter]
                        for elem in range_constexpr(8):
                            row = fx.get_scalar(row_atom[elem])
                            col = fx.get_scalar(col_atom[elem])
                            global_row = bid_m * block_m + row
                            global_col = bid_n * block_n + col
                            fx.ptr_store(
                                values[elem],
                                fx.add_offset(
                                    c_iter,
                                    global_row * N + global_col,
                                ),
                            )
            else:
                copy_out = fx.make_copy_atom(
                    fx.rocdl.BufferCopy(out_elem_cls.width),
                    out_elem_cls,
                )
                thr_r2g_c = fx.make_tiled_copy_C(
                    copy_out,
                    tiled_mma,
                ).get_slice(consumer_tid)
                p_c_g = thr_r2g_c.partition_S(t_c)
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
                            T.vec(acc_size, out_elem_cls.ir_type),
                            [as_ir_value(value) for value in out_values],
                        )
                    )
                frag_c_retile = thr_r2g_c.retile(frag_c_out)
                fx.copy(copy_out, frag_c_retile, p_c_g)

    @flyc.jit
    def launch_gemm(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_bt: fx.Tensor,
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

        a_view = fx.make_view(
            fx.get_iter(arg_a),
            fx.make_layout((M, K), (K, 1)),
        )
        bt_view = fx.make_view(
            fx.get_iter(arg_bt),
            fx.make_layout((N, K), (K, 1)),
        )
        c_view = fx.make_view(
            fx.get_iter(arg_c),
            fx.make_layout((M, N), (N, 1)),
        )

        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            barriers.finalized = False
            barriers.finalize()

        kernel_grid = (
            (grid_m, grid_n, 1)
            if use_cluster
            else (grid_m * grid_n, 1, 1)
        )
        cluster_arg = (
            (cluster_m, cluster_n, 1)
            if use_cluster
            else None
        )
        gemm_kernel(c_view, a_view, bt_view, tiled_mma).launch(
            grid=kernel_grid,
            block=(threads_per_block, 1, 1),
            stream=stream,
            cluster=cluster_arg,
            value_attrs={
                "rocdl.cluster_dims": (
                    f"{cluster_m},{cluster_n},1"
                    if use_cluster
                    else None
                )
            },
        )

    launch_gemm.compile_hints["llvm_options"] = {"unroll-threshold": 0}
    if expert_schedule:
        launch_gemm.compile_hints["llvm_options"][
            "amdgpu-expert-scheduling-mode"
        ] = True

    return launch_gemm, block_m, block_n, block_k


@lru_cache(maxsize=128)
def _cached_module(
    M: int,
    N: int,
    K: int,
    in_dtype: str,
    out_dtype: str,
    reg_m: int,
    reg_n: int,
    reg_k: int,
    waves_m: int,
    waves_n: int,
    num_stages: int,
    traversal_order: str,
    overlap_mode: str,
    swizzle_m: int,
    expert_schedule: bool,
    barrier_fences: bool,
    cluster_m: int,
    cluster_n: int,
    grouped_inline: bool,
):
    return create_gemm_a16w16_module(
        M,
        N,
        K,
        in_dtype,
        out_dtype,
        reg_m=reg_m,
        reg_n=reg_n,
        reg_k=reg_k,
        waves_m=waves_m,
        waves_n=waves_n,
        num_stages=num_stages,
        traversal_order=traversal_order,
        overlap_mode=overlap_mode,
        swizzle_m=swizzle_m,
        expert_schedule=expert_schedule,
        barrier_fences=barrier_fences,
        cluster_m=cluster_m,
        cluster_n=cluster_n,
        grouped_inline=grouped_inline,
    )


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


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
        # Keep Opus's validated two-consumer geometry as the default.  Four
        # consumers remain available through explicit waves_m/waves_n tuning,
        # but currently lose to the deeper per-wave accumulator tile.
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
    traversal_order: str = "KMN",
    overlap_mode: str = "cross",
    swizzle_m: int = 32,
    expert_schedule: bool = True,
    barrier_fences: bool = False,
    cluster_m: int = 1,
    cluster_n: int = 1,
    grouped_inline: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``a @ b.T`` with the gfx1250 tile-layout kernel.

    A and B must be two-dimensional CUDA tensors of the same FP16/BF16 dtype.
    Non-contiguous inputs and arbitrary M/N/K sizes are handled by creating
    contiguous, zero-padded operands.
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
    block_m = WMMA_M * reg_m * waves_m
    block_n = WMMA_N * reg_n * waves_n
    block_k = WMMA_K * reg_k
    padded_m = _round_up(m, block_m * cluster_m)
    padded_n = _round_up(n, block_n * cluster_n)
    padded_k = max(_round_up(k, block_k), 2 * block_k)

    if (m, k) == (padded_m, padded_k):
        a_padded = a.contiguous()
    else:
        a_padded = torch.zeros(
            (padded_m, padded_k),
            dtype=a.dtype,
            device=a.device,
        )
        a_padded[:m, :k].copy_(a)
    if (n, k) == (padded_n, padded_k):
        b_padded = b.contiguous()
    else:
        b_padded = torch.zeros(
            (padded_n, padded_k),
            dtype=b.dtype,
            device=b.device,
        )
        b_padded[:n, :k].copy_(b)

    dtype_name = "f16" if a.dtype == torch.float16 else "bf16"
    out_dtype_name = {
        torch.float16: "f16",
        torch.bfloat16: "bf16",
        torch.float32: "f32",
    }[out_dtype]
    launch, _, _, _ = _cached_module(
        padded_m,
        padded_n,
        padded_k,
        dtype_name,
        out_dtype_name,
        reg_m,
        reg_n,
        reg_k,
        waves_m,
        waves_n,
        num_stages,
        traversal_order,
        overlap_mode,
        swizzle_m,
        expert_schedule,
        barrier_fences,
        cluster_m,
        cluster_n,
        grouped_inline,
    )
    if out is not None:
        if out.shape != (m, n) or out.dtype != out_dtype or out.device != a.device:
            raise ValueError(
                f"out must have shape {(m, n)}, dtype {out_dtype}, and device {a.device}"
            )
        if (m, n) == (padded_m, padded_n) and out.is_contiguous():
            c_padded = out
        else:
            c_padded = torch.empty(
                (padded_m, padded_n),
                dtype=out_dtype,
                device=a.device,
            )
    else:
        c_padded = torch.empty(
            (padded_m, padded_n),
            dtype=out_dtype,
            device=a.device,
        )
    stream = torch.cuda.current_stream()
    _run_compiled(
        launch,
        c_padded,
        a_padded,
        b_padded,
        stream,
        device_index=a.device.index or 0,
    )
    result = c_padded[:m, :n]
    if out is not None and c_padded is out:
        return out
    if out is not None:
        out.copy_(result)
        return out
    return result


__all__ = [
    "create_gemm_a16w16_module",
    "gemm_a16w16",
]
