#!/usr/bin/env python3
"""Four-wave all-compute BF16 GEMM prototype for gfx1250.

All four waves own WMMA output fragments.  TDM loads for the next LDS slot are
issued cooperatively by the same waves after the first WMMA-K round, so global
loads can progress while the remaining matrix instructions execute.  Thread
and value ownership remains defined by FlyDSL TiledMma/tiled-copy layouts.
"""

from functools import lru_cache

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as llvm_dialect
from flydsl._mlir.dialects import vector as vector_dialect
from flydsl.expr import arith, as_ir_value, const_expr, gpu, range_constexpr
from flydsl.expr.rocdl import cluster, tdm_ops
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import check_smem_capacity

from kernels.gemm_a16w16_gfx1250 import _round_up, _run_compiled


WMMA_M = WMMA_N = 16
WMMA_K = 32
WAVE_SIZE = 32


def create_gemm_a16w16_all_compute_module(
    M: int,
    N: int,
    K: int,
    *,
    reg_m: int = 4,
    reg_n: int = 4,
    reg_k: int = 4,
    waves_m: int = 2,
    waves_n: int = 2,
    num_buffers: int = 2,
    swizzle_m: int = 32,
    reuse_lds: bool = False,
    source_ring: bool = False,
    reuse_rmem: bool = False,
    traversal_order: str = "KMN",
    diagonal_traversal: bool = False,
    reuse_repeat: int = 1,
    use_xcd_remap: bool = False,
    pre_group_schedule: bool = False,
    cluster_m: int = 1,
    cluster_n: int = 1,
):
    arch = str(get_rocm_arch() or "")
    if not arch.startswith("gfx1250"):
        raise RuntimeError(f"requires gfx1250, got {arch!r}")
    num_waves = waves_m * waves_n
    if num_waves not in (4, 8):
        raise ValueError("all-compute kernel requires four or eight waves")
    if not 2 <= num_buffers <= 5:
        raise ValueError("num_buffers must be in [2, 5]")

    block_m = WMMA_M * reg_m * waves_m
    block_n = WMMA_N * reg_n * waves_n
    block_k = WMMA_K * reg_k
    if block_k & (block_k - 1):
        raise ValueError("block_k must be a power of two")
    if M % block_m or N % block_n or K % block_k:
        raise ValueError("padded problem must divide the block shape")

    num_k_tiles = K // block_k
    num_buffers = min(num_buffers, num_k_tiles)
    k_iters = block_k // WMMA_K
    block_threads = num_waves * WAVE_SIZE
    grid_m, grid_n = M // block_m, N // block_n
    if cluster_m < 1 or cluster_n < 1 or cluster_m * cluster_n > 16:
        raise ValueError("cluster must contain 1..16 workgroups")
    if grid_m % cluster_m or grid_n % cluster_n:
        raise ValueError("grid must be divisible by cluster dimensions")
    use_cluster = cluster_m > 1 or cluster_n > 1
    a_cluster_pattern = sum(
        1 << (ly * cluster_m)
        for ly in range(cluster_n)
    )
    b_cluster_pattern = (1 << cluster_m) - 1
    if traversal_order not in ("KMN", "KNM", "MKN", "MNK", "NKM", "NMK"):
        raise ValueError(f"unsupported traversal order: {traversal_order}")
    if diagonal_traversal and (reg_m != 4 or reg_n != 4):
        raise ValueError("diagonal traversal currently requires reg_m=reg_n=4")
    if reuse_repeat < 1:
        raise ValueError("reuse_repeat must be positive")
    gemm_traversal = getattr(fx.GemmTraversalOrder, traversal_order)
    diagonal_pairs = [
        (m_iter, n_iter)
        for m_iter in range(4)
        for n_iter in range(4)
    ]
    diagonal_asm = "\n".join(
        "v_wmma_f32_16x16x32_bf16 "
        f"${m_iter * 4 + n_iter}, "
        f"${16 + m_iter}, ${20 + n_iter}, "
        f"${m_iter * 4 + n_iter}"
        for m_iter, n_iter in diagonal_pairs
    )
    diagonal_constraints = ",".join(
        ["=v"] * 16
        + ["v"] * 8
        + [str(idx) for idx in range(16)]
    )

    k_pad = 8
    lds_stride = block_k + k_pad
    elem_bytes = 2
    stage_a_bytes = block_m * lds_stride * elem_bytes
    stage_b_offset = (stage_a_bytes + 15) // 16 * 16
    stage_b_bytes = block_n * lds_stride * elem_bytes
    stage_pitch = (
        (stage_b_offset + stage_b_bytes + 1023) // 1024 * 1024
    )
    arena_bytes = num_buffers * stage_pitch
    check_smem_capacity(arena_bytes, arch)

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def kernel(
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
            local_x_i32 = fx.Int32(local_x)
            local_y_i32 = fx.Int32(local_y)
            a_mask = fx.Int32(a_cluster_pattern) << local_x_i32
            b_mask = fx.Int32(b_cluster_pattern) << (
                local_y_i32 * fx.Int32(cluster_m)
            )
        else:
            linear_bid = fx.block_idx.x
            num_workgroups = grid_m * grid_n
            if const_expr(
                use_xcd_remap
                and num_workgroups >= 4 * 256
                and num_workgroups % 8 == 0
            ):
                intra_xcd = linear_bid // 8
                xcd = linear_bid % 8
                linear_bid = xcd * (num_workgroups // 8) + intra_xcd
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
        t_c = fx.flat_divide(
            g_c,
            fx.make_tile(block_m, block_n),
        )[None, None, bid_m, bid_n]

        arena = fx.SharedAllocator(static=False)
        arena.allocate(arena_bytes)
        base_ptr = arena.base_ptr
        base_bf16 = fx.recast_iter(fx.BFloat16, base_ptr)

        def _stage_views(stage):
            elem_base = stage * (stage_pitch // elem_bytes)
            a_ptr = fx.add_offset(base_bf16, elem_base)
            b_ptr = fx.add_offset(
                base_bf16,
                elem_base + stage_b_offset // elem_bytes,
            )
            return (
                fx.make_view(
                    a_ptr,
                    fx.make_layout(
                        (block_m, block_k),
                        (lds_stride, 1),
                    ),
                ),
                fx.make_view(
                    b_ptr,
                    fx.make_layout(
                        (block_n, block_k),
                        (lds_stride, 1),
                    ),
                ),
            )

        def _chunk_views(stage, ki):
            s_a, s_b = _stage_views(stage)
            return (
                fx.make_view(
                    fx.add_offset(fx.get_iter(s_a), ki * WMMA_K),
                    fx.make_layout(
                        (block_m, WMMA_K),
                        (lds_stride, 1),
                    ),
                ),
                fx.make_view(
                    fx.add_offset(fx.get_iter(s_b), ki * WMMA_K),
                    fx.make_layout(
                        (block_n, WMMA_K),
                        (lds_stride, 1),
                    ),
                ),
            )

        a_off = fx.Int64(bid_m * block_m) * fx.Int64(K)
        b_off = fx.Int64(bid_n * block_n) * fx.Int64(K)
        g_a = fx.Tensor(
            fx.make_view(
                fx.add_offset(fx.get_iter(arg_a), a_off),
                fx.make_layout((block_m, block_k), (K, 1)),
            )
        )
        g_b = fx.Tensor(
            fx.make_view(
                fx.add_offset(fx.get_iter(arg_bt), b_off),
                fx.make_layout((block_n, block_k), (K, 1)),
            )
        )
        atom_a = fx.rocdl.make_tdm_atom(
            g_a,
            [None, None],
            strides=[K, None],
            num_warps=1,
            pad_interval=block_k,
            pad_amount=k_pad,
            early_timeout=True,
        )
        atom_b = fx.rocdl.make_tdm_atom(
            g_b,
            [None, None],
            strides=[K, None],
            num_warps=1,
            pad_interval=block_k,
            pad_amount=k_pad,
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

        def _issue(stage, logical_k):
            s_a, s_b = _stage_views(stage)
            byte_offset = fx.Int64(logical_k) * fx.Int64(
                block_k * elem_bytes
            )
            if wave == 0:
                fx.copy(
                    atom_a,
                    g_a,
                    s_a,
                    imm_offset=byte_offset,
                )
            elif wave == 1:
                fx.copy(
                    atom_b,
                    g_b,
                    s_b,
                    imm_offset=byte_offset,
                )

        copy_atom = fx.make_copy_atom(
            fx.UniversalCopy128b(),
            fx.BFloat16,
        )
        thr_mma = tiled_mma.thr_slice(tid)
        thr_copy_a = fx.make_tiled_copy_A(
            copy_atom,
            tiled_mma,
        ).get_slice(tid)
        thr_copy_b = fx.make_tiled_copy_B(
            copy_atom,
            tiled_mma,
        ).get_slice(tid)
        frag_c = thr_mma.make_fragment_C(t_c)
        row_coords = fx.make_view(
            0,
            fx.make_layout((block_m, block_n), (1, 0)),
        )
        col_coords = fx.make_view(
            0,
            fx.make_layout((block_m, block_n), (0, 1)),
        )
        c_rows = thr_mma.partition_C(row_coords)
        c_cols = thr_mma.partition_C(col_coords)
        if const_expr(diagonal_traversal):
            diagonal_c = [
                fx.make_rmem_tensor(8, fx.Float32)
                for _ in range(16)
            ]
            for fragment in diagonal_c:
                fragment.store(Vec.filled(8, 0.0, fx.Float32))
        def _gemm(a_fragment, b_fragment, repeats=1):
            if const_expr(pre_group_schedule and not diagonal_traversal):
                fx.rocdl.sched_mfma(reg_m * reg_n)
            if const_expr(diagonal_traversal):
                a_atoms = [
                    fx.make_rmem_tensor(16, fx.BFloat16)
                    for _ in range(4)
                ]
                b_atoms = [
                    fx.make_rmem_tensor(16, fx.BFloat16)
                    for _ in range(4)
                ]
                for atom_idx in range_constexpr(4):
                    if const_expr(reuse_rmem or source_ring):
                        a_atoms[atom_idx].store(
                            a_fragment[None, atom_idx, 0].load()
                        )
                        b_atoms[atom_idx].store(
                            b_fragment[None, atom_idx, 0].load()
                        )
                    else:
                        a_atoms[atom_idx].store(
                            a_fragment[None, atom_idx].load()
                        )
                        b_atoms[atom_idx].store(
                            b_fragment[None, atom_idx].load()
                        )
                a_raw = [
                    arith._to_raw(
                        vector_dialect.bitcast(
                            T.vec(8, T.i32),
                            as_ir_value(a_atoms[idx].load()),
                        )
                    )
                    for idx in range_constexpr(4)
                ]
                b_raw = [
                    arith._to_raw(
                        vector_dialect.bitcast(
                            T.vec(8, T.i32),
                            as_ir_value(b_atoms[idx].load()),
                        )
                    )
                    for idx in range_constexpr(4)
                ]
                c_raw = [
                    arith._to_raw(diagonal_c[idx].load())
                    for idx in range_constexpr(16)
                ]
                result_type = ir.Type.parse(
                    "!llvm.struct<("
                    + ", ".join(str(value.type) for value in c_raw)
                    + ")>"
                )
                results = llvm_dialect.inline_asm(
                    result_type,
                    a_raw + b_raw + c_raw,
                    "\n".join(
                        diagonal_asm
                        for _ in range(repeats)
                    ),
                    diagonal_constraints,
                    has_side_effects=True,
                )
                for idx in range_constexpr(16):
                    diagonal_c[idx].store(
                        Vec(
                            llvm_dialect.extractvalue(
                                c_raw[idx].type,
                                results,
                                [idx],
                            )
                        )
                    )
            else:
                fx.gemm(
                    tiled_mma,
                    frag_c,
                    a_fragment,
                    b_fragment,
                    frag_c,
                    traversal_order=gemm_traversal,
                )
        if const_expr(use_cluster):
            cluster.cluster_barrier()
        if const_expr(reuse_rmem):
            reuse_a = thr_mma.make_fragment_A(_chunk_views(0, 0)[0])
            reuse_b = thr_mma.make_fragment_B(_chunk_views(0, 0)[1])
            reuse_a_retile = thr_copy_a.retile(reuse_a)
            reuse_b_retile = thr_copy_b.retile(reuse_b)
        elif const_expr(source_ring):
            frag_a_ring = [
                thr_mma.make_fragment_A(_chunk_views(0, 0)[0])
                for _ in range(3)
            ]
            frag_b_ring = [
                thr_mma.make_fragment_B(_chunk_views(0, 0)[1])
                for _ in range(3)
            ]
            frag_a_ring_retile = [
                thr_copy_a.retile(fragment)
                for fragment in frag_a_ring
            ]
            frag_b_ring_retile = [
                thr_copy_b.retile(fragment)
                for fragment in frag_b_ring
            ]
        else:
            frag_a = thr_mma.make_fragment_A(_stage_views(0)[0])
            frag_b = thr_mma.make_fragment_B(_stage_views(0)[1])
            frag_a_retile = thr_copy_a.retile(frag_a)
            frag_b_retile = thr_copy_b.retile(frag_b)
        frag_c.fill(0)

        def _pipeline_fence(outstanding):
            tdm_ops.tensor_wait(outstanding)
            if const_expr(use_cluster):
                cluster.cluster_barrier()
            else:
                gpu.barrier()

        def _compute(stage, prefetch_k):
            s_a, s_b = _stage_views(stage)
            if const_expr(prefetch_k is not None):
                _issue(prefetch_k % num_buffers, prefetch_k)
            if const_expr(source_ring):
                def _load_chunk(ki, buf):
                    c_a, c_b = _chunk_views(stage, ki)
                    fx.copy(
                        copy_atom,
                        thr_copy_b.partition_S(c_b),
                        frag_b_ring_retile[buf],
                    )
                    fx.copy(
                        copy_atom,
                        thr_copy_a.partition_S(c_a),
                        frag_a_ring_retile[buf],
                    )
                    fx.rocdl.sched_barrier(0)

                _load_chunk(0, 0)
                fx.rocdl.sched_barrier(0)
                for ki in range_constexpr(k_iters):
                    cur = ki % 3
                    if const_expr(ki + 1 < k_iters):
                        _load_chunk(ki + 1, (cur + 1) % 3)
                    fx.rocdl.s_wait_dscnt(
                        (2 * reg_m + 2 * reg_n)
                        if const_expr(ki + 1 < k_iters)
                        else 0
                    )
                    _gemm(
                        frag_a_ring[cur],
                        frag_b_ring[cur],
                    )
                for ki in range_constexpr(k_iters):
                    if const_expr(ki + 1 < k_iters):
                        fx.rocdl.sched_dsrd(2 * reg_m + 2 * reg_n)
                    if const_expr(not pre_group_schedule):
                        for _ in range_constexpr(reg_m):
                            fx.rocdl.sched_mfma(reg_n)
                fx.rocdl.sched_barrier(0)
            else:
                p_a = thr_copy_a.partition_S(s_a)
                p_b = thr_copy_b.partition_S(s_b)
                fx.rocdl.sched_barrier(0)
                for ki in range_constexpr(k_iters):
                    fx.copy(
                        copy_atom,
                        p_b[None, None, ki],
                        frag_b_retile[None, None, ki],
                    )
                    fx.copy(
                        copy_atom,
                        p_a[None, None, ki],
                        frag_a_retile[None, None, ki],
                    )
                    fx.rocdl.s_wait_dscnt(0)
                    _gemm(
                        frag_a[None, None, ki],
                        frag_b[None, None, ki],
                    )
                for _ in range_constexpr(k_iters):
                    fx.rocdl.sched_dsrd(2 * reg_n)
                    fx.rocdl.sched_dsrd(2 * reg_m)
                    if const_expr(not pre_group_schedule):
                        for _ in range_constexpr(reg_m):
                            fx.rocdl.sched_mfma(reg_n)
                fx.rocdl.sched_barrier(0)

        if const_expr(reuse_rmem):
            _issue(0, 0)
            _pipeline_fence(0)
            s_a, s_b = _stage_views(0)
            c_a, c_b = _chunk_views(0, 0)
            fx.copy(
                copy_atom,
                thr_copy_b.partition_S(c_b),
                reuse_b_retile,
            )
            fx.copy(
                copy_atom,
                thr_copy_a.partition_S(c_a),
                reuse_a_retile,
            )
            fx.rocdl.s_wait_dscnt(0)
            if const_expr(diagonal_traversal):
                _gemm(
                    reuse_a,
                    reuse_b,
                    repeats=num_k_tiles * k_iters * reuse_repeat,
                )
            else:
                init = [frag_c.load()]
                results = init
                for _, state in range(
                    0,
                    num_k_tiles * reuse_repeat,
                    1,
                    init=init,
                ):
                    frag_c.store(state[0])
                    for ki in range_constexpr(k_iters):
                        _gemm(
                            reuse_a,
                            reuse_b,
                        )
                    if const_expr(not pre_group_schedule):
                        for _ in range_constexpr(k_iters):
                            for _ in range_constexpr(reg_m):
                                fx.rocdl.sched_mfma(reg_n)
                    fx.rocdl.sched_barrier(0)
                    results = yield [frag_c.load()]
                frag_c.store(results)
        elif const_expr(reuse_lds):
            _issue(0, 0)
            _pipeline_fence(0)
            init = [frag_c.load()]
            results = init
            for _, state in range(0, num_k_tiles, 1, init=init):
                frag_c.store(state[0])
                _compute(0, None)
                results = yield [frag_c.load()]
            frag_c.store(results)
        else:
            for stage in range_constexpr(num_buffers - 1):
                _issue(stage, stage)

            steady = num_k_tiles - (num_buffers - 1)
            init = [frag_c.load()]
            results = init
            if const_expr(steady > 0):
                for kt, state in range(0, steady, 1, init=init):
                    frag_c.store(state[0])
                    _pipeline_fence(num_buffers - 2)
                    stage = kt % num_buffers
                    _compute(stage, kt + (num_buffers - 1))
                    results = yield [frag_c.load()]
                frag_c.store(results)

            for j in range_constexpr(num_buffers - 1):
                logical_k = steady + j
                _pipeline_fence(num_buffers - 2 - j)
                _compute(logical_k % num_buffers, None)

        if const_expr(diagonal_traversal):
            c_iter = fx.get_iter(arg_c)
            for m_iter in range_constexpr(4):
                for n_iter in range_constexpr(4):
                    atom_idx = m_iter * 4 + n_iter
                    values = Vec(
                        diagonal_c[atom_idx].load()
                    ).to(fx.BFloat16)
                    row_atom = c_rows[None, m_iter, n_iter]
                    col_atom = c_cols[None, m_iter, n_iter]
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
                fx.rocdl.BufferCopy(16),
                fx.BFloat16,
            )
            thr_out = fx.make_tiled_copy_C(
                copy_out,
                tiled_mma,
            ).get_slice(tid)
            frag_out = fx.make_fragment_like(
                frag_c,
                fx.BFloat16.ir_type,
            )
            frag_out.store(frag_c.load().to(fx.BFloat16))
            fx.copy(
                copy_out,
                thr_out.retile(frag_out),
                thr_out.partition_S(t_c),
            )

    @flyc.jit
    def launch(
        c: fx.Tensor,
        a: fx.Tensor,
        bt: fx.Tensor,
        stream: fx.Stream,
    ):
        mma_atom = fx.make_mma_atom(
            fx.rocdl.WMMA(
                WMMA_M,
                WMMA_N,
                WMMA_K,
                fx.BFloat16,
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
            fx.get_iter(a),
            fx.make_layout((M, K), (K, 1)),
        )
        b_view = fx.make_view(
            fx.get_iter(bt),
            fx.make_layout((N, K), (K, 1)),
        )
        c_view = fx.make_view(
            fx.get_iter(c),
            fx.make_layout((M, N), (N, 1)),
        )
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
        kernel(
            c_view,
            a_view,
            b_view,
            tiled_mma,
            value_attrs={
                "rocdl.waves_per_eu": 2 if num_waves == 8 else 1,
                "rocdl.flat_work_group_size": (
                    f"{block_threads},{block_threads}"
                ),
                "rocdl.cluster_dims": (
                    f"{cluster_m},{cluster_n},1"
                    if use_cluster
                    else None
                ),
            },
        ).launch(
            grid=kernel_grid,
            block=(block_threads, 1, 1),
            stream=stream,
            cluster=cluster_arg,
        )

    launch.compile_hints["llvm_options"] = {
        "amdgpu-expert-scheduling-mode": True,
        "unroll-threshold": 0,
    }
    return launch, block_m, block_n, block_k


@lru_cache(maxsize=64)
def _cached(
    M,
    N,
    K,
    reg_m,
    reg_n,
    reg_k,
    waves_m,
    waves_n,
    num_buffers,
    reuse_lds,
    source_ring,
    reuse_rmem,
    traversal_order,
    diagonal_traversal,
    reuse_repeat,
    use_xcd_remap,
    pre_group_schedule,
    cluster_m,
    cluster_n,
):
    return create_gemm_a16w16_all_compute_module(
        M,
        N,
        K,
        reg_m=reg_m,
        reg_n=reg_n,
        reg_k=reg_k,
        waves_m=waves_m,
        waves_n=waves_n,
        num_buffers=num_buffers,
        reuse_lds=reuse_lds,
        source_ring=source_ring,
        reuse_rmem=reuse_rmem,
        traversal_order=traversal_order,
        diagonal_traversal=diagonal_traversal,
        reuse_repeat=reuse_repeat,
        use_xcd_remap=use_xcd_remap,
        pre_group_schedule=pre_group_schedule,
        cluster_m=cluster_m,
        cluster_n=cluster_n,
    )


def gemm_a16w16_all_compute(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    reg_m: int = 4,
    reg_n: int = 4,
    reg_k: int = 4,
    waves_m: int = 2,
    waves_n: int = 2,
    num_buffers: int = 2,
    reuse_lds: bool = False,
    source_ring: bool = False,
    reuse_rmem: bool = False,
    traversal_order: str = "KMN",
    diagonal_traversal: bool = False,
    reuse_repeat: int = 1,
    use_xcd_remap: bool = False,
    pre_group_schedule: bool = False,
    cluster_m: int = 1,
    cluster_n: int = 1,
):
    if (
        a.ndim != 2
        or b.ndim != 2
        or a.shape[1] != b.shape[1]
        or a.dtype != torch.bfloat16
        or b.dtype != torch.bfloat16
        or not a.is_cuda
        or not b.is_cuda
    ):
        raise ValueError("expected compatible CUDA BF16 matrices")
    m, k = a.shape
    n = b.shape[0]
    block_m = WMMA_M * reg_m * waves_m
    block_n = WMMA_N * reg_n * waves_n
    block_k = WMMA_K * reg_k
    pm = _round_up(m, block_m * cluster_m)
    pn = _round_up(n, block_n * cluster_n)
    pk = max(_round_up(k, block_k), num_buffers * block_k)
    if (m, k) == (pm, pk):
        ap = a.contiguous()
    else:
        ap = torch.zeros((pm, pk), dtype=a.dtype, device=a.device)
        ap[:m, :k].copy_(a)
    if (n, k) == (pn, pk):
        bp = b.contiguous()
    else:
        bp = torch.zeros((pn, pk), dtype=b.dtype, device=b.device)
        bp[:n, :k].copy_(b)
    if (
        out is not None
        and out.shape == (m, n)
        and (m, n) == (pm, pn)
        and out.is_contiguous()
    ):
        cp = out
    else:
        cp = torch.empty((pm, pn), dtype=torch.bfloat16, device=a.device)
    launch, _, _, _ = _cached(
        pm,
        pn,
        pk,
        reg_m,
        reg_n,
        reg_k,
        waves_m,
        waves_n,
        num_buffers,
        reuse_lds,
        source_ring,
        reuse_rmem,
        traversal_order,
        diagonal_traversal,
        reuse_repeat,
        use_xcd_remap,
        pre_group_schedule,
        cluster_m,
        cluster_n,
    )
    _run_compiled(
        launch,
        cp,
        ap,
        bp,
        torch.cuda.current_stream(),
        device_index=a.device.index or 0,
    )
    result = cp[:m, :n]
    if out is cp:
        return out
    if out is not None:
        out.copy_(result)
        return out
    return result


__all__ = [
    "create_gemm_a16w16_all_compute_module",
    "gemm_a16w16_all_compute",
]
