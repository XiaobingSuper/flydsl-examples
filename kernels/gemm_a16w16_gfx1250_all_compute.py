#!/usr/bin/env python3
"""Role-fused FP16/BF16 all-compute GEMM family for gfx1250."""

from functools import lru_cache
from threading import Lock

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as llvm_dialect
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.rocdl import tdm_ops
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import check_smem_capacity

WMMA_M = 16
WMMA_N = 16
WMMA_K = 32
WAVE_SIZE = 32
_compiled_dispatch_lock = Lock()


def _create_all_compute_module(
    M: int,
    N: int,
    K: int,
    *,
    in_dtype: str = "bf16",
    reg_m: int = 4,
    reg_n: int = 4,
    reg_k: int = 4,
    waves_m: int = 2,
    waves_n: int = 2,
    num_buffers: int = 2,
    swizzle_m: int = 32,
    direct_b_global: bool = False,
    use_xcd_remap: bool = False,
    main_loop_unroll: bool = False,
):
    arch = str(get_rocm_arch() or "")
    if not arch.startswith("gfx1250"):
        raise RuntimeError(f"requires gfx1250, got {arch!r}")
    if in_dtype not in ("f16", "bf16"):
        raise ValueError(f"in_dtype must be 'f16' or 'bf16', got {in_dtype!r}")

    num_waves = waves_m * waves_n
    if num_waves not in (1, 2, 4, 8):
        raise ValueError("all-compute kernel requires 1, 2, 4, or 8 waves")
    if not 2 <= num_buffers <= 5:
        raise ValueError("num_buffers must be in [2, 5]")

    block_m = WMMA_M * reg_m * waves_m
    block_n = WMMA_N * reg_n * waves_n
    block_k = WMMA_K * reg_k
    if block_k & (block_k - 1):
        raise ValueError("block_k must be a power of two")
    if M % block_m or N % block_n or K % block_k:
        raise ValueError("padded problem must divide the block shape")
    if direct_b_global and (reg_k < 2 or reg_k % 2 != 0):
        raise ValueError("direct_b_global requires an even reg_k >= 2")

    num_k_tiles = K // block_k
    num_buffers = min(num_buffers, num_k_tiles)
    k_iters = block_k // WMMA_K
    block_threads = num_waves * WAVE_SIZE
    waves_per_eu = 2 if num_waves == 8 else 1 if num_waves == 4 else None
    grid_m, grid_n = M // block_m, N // block_n

    elem_cls = fx.Float16 if in_dtype == "f16" else fx.BFloat16
    k_pad = 8
    lds_stride = block_k + k_pad
    elem_bytes = 2
    stage_a_bytes = block_m * lds_stride * elem_bytes
    stage_b_offset = (stage_a_bytes + 15) // 16 * 16
    # Direct-B keeps this reservation as an occupancy throttle; removing the
    # unused B region raises power pressure and regresses the large shape.
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

        g_c = fx.rocdl.make_buffer_tensor(arg_c)
        t_c = fx.flat_divide(
            g_c,
            fx.make_tile(block_m, block_n),
        )[None, None, bid_m, bid_n]

        arena = fx.SharedAllocator(static=False)
        arena.allocate(arena_bytes)
        base_elem = fx.recast_iter(elem_cls, arena.base_ptr)

        def _stage_views(stage):
            elem_base = stage * (stage_pitch // elem_bytes)
            a_ptr = fx.add_offset(base_elem, elem_base)
            b_ptr = fx.add_offset(
                base_elem,
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

        def _global_b_chunk(logical_k, ki):
            elem_offset = (
                b_off
                + fx.Int64(logical_k) * fx.Int64(block_k)
                + fx.Int64(ki * WMMA_K)
            )
            return fx.Tensor(
                fx.make_view(
                    fx.add_offset(fx.get_iter(arg_bt), elem_offset),
                    fx.make_layout(
                        (block_n, WMMA_K),
                        (K, 1),
                    ),
                )
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
                if const_expr(num_waves == 1 and not direct_b_global):
                    fx.copy(
                        atom_b,
                        g_b,
                        s_b,
                        imm_offset=byte_offset,
                    )
            elif wave == 1:
                if const_expr(not direct_b_global):
                    fx.copy(
                        atom_b,
                        g_b,
                        s_b,
                        imm_offset=byte_offset,
                    )

        copy_atom = fx.make_copy_atom(
            fx.UniversalCopy128b(),
            elem_cls,
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

        if const_expr(direct_b_global):
            direct_a_frag = thr_mma.make_fragment_A(
                _chunk_views(0, 0)[0]
            )
            direct_a_retile = thr_copy_a.retile(direct_a_frag)
            # Keep one B slot per N fragment. Each slot is refilled for the
            # next K step after its current eight WMMAs have consumed it.
            b_column_layout = fx.make_layout(
                ((8, 2), 1, 1),
                ((1, 8), 16, 0),
            )
            direct_b_columns = [
                fx.make_fragment_like(
                    b_column_layout,
                    elem_cls.ir_type,
                )
                for _ in range(reg_n)
            ]
            direct_b_column_retile = [
                thr_copy_b.retile(fragment)
                for fragment in direct_b_columns
            ]
            c_column_layout = fx.make_layout(
                (8, reg_m, 1),
                (1, 8, 0),
            )
            direct_c_columns = [
                fx.make_fragment_like(
                    c_column_layout,
                    fx.Float32.ir_type,
                )
                for _ in range(reg_n)
            ]
        else:
            frag_a = thr_mma.make_fragment_A(_stage_views(0)[0])
            frag_b = thr_mma.make_fragment_B(_stage_views(0)[1])
            frag_a_retile = thr_copy_a.retile(frag_a)
            frag_b_retile = thr_copy_b.retile(frag_b)
        if const_expr(direct_b_global):
            for column in direct_c_columns:
                column.fill(0)
        else:
            frag_c.fill(0)

        def _pipeline_fence(outstanding):
            if const_expr(direct_b_global):
                if wave == 0:
                    tdm_ops.tensor_wait(outstanding)
            elif const_expr(num_waves == 1):
                tdm_ops.tensor_wait(outstanding * 2)
            else:
                tdm_ops.tensor_wait(outstanding)
            gpu.barrier()

        def _gemm(a_fragment, b_fragment):
            fx.gemm(
                tiled_mma,
                frag_c,
                a_fragment,
                b_fragment,
                frag_c,
                traversal_order=fx.GemmTraversalOrder.KMN,
            )

        def _wait_global_loads():
            llvm_dialect.inline_asm(
                None,
                [],
                "s_wait_loadcnt 0",
                "",
                has_side_effects=True,
            )

        def _compute(stage, logical_k, prefetch_k):
            s_a, s_b = _stage_views(stage)
            if const_expr(prefetch_k is not None):
                _issue(prefetch_k % num_buffers, prefetch_k)
            p_a = thr_copy_a.partition_S(s_a)

            if const_expr(direct_b_global):
                def _load_direct_b_column(
                    source_tile,
                    source_ki,
                    source_n,
                ):
                    p_b = thr_copy_b.partition_S(
                        _global_b_chunk(source_tile, source_ki)
                    )
                    fx.copy(
                        copy_atom,
                        p_b[None, source_n, None],
                        direct_b_column_retile[source_n][
                            None, 0, None
                        ],
                    )

                if logical_k == 0:
                    for ni in range_constexpr(reg_n):
                        _load_direct_b_column(0, 0, ni)
                _wait_global_loads()
                fx.rocdl.sched_barrier(0)
                for ki in range_constexpr(k_iters):
                    c_a, _ = _chunk_views(stage, ki)
                    fx.copy(
                        copy_atom,
                        thr_copy_a.partition_S(c_a),
                        direct_a_retile,
                    )
                    fx.rocdl.s_wait_dscnt(0)
                    for ni in range_constexpr(reg_n):
                        fx.gemm(
                            tiled_mma,
                            direct_c_columns[ni],
                            direct_a_frag,
                            direct_b_columns[ni],
                            direct_c_columns[ni],
                            traversal_order=(
                                fx.GemmTraversalOrder.KMN
                            ),
                        )
                        if const_expr(ki + 1 < k_iters):
                            _load_direct_b_column(
                                logical_k,
                                ki + 1,
                                ni,
                            )
                        else:
                            if logical_k + 1 < num_k_tiles:
                                _load_direct_b_column(
                                    logical_k + 1,
                                    0,
                                    ni,
                                )
            else:
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
                if const_expr(direct_b_global):
                    fx.rocdl.sched_dsrd(2 * reg_m)
                    for _ in range_constexpr(reg_n):
                        fx.rocdl.sched_mfma(reg_m)
                        fx.rocdl.sched_vmem(2)
                else:
                    fx.rocdl.sched_dsrd(2 * reg_n)
                    fx.rocdl.sched_dsrd(2 * reg_m)
                    for _ in range_constexpr(reg_m):
                        fx.rocdl.sched_mfma(reg_n)
            fx.rocdl.sched_barrier(0)

        for stage in range_constexpr(num_buffers - 1):
            _issue(stage, stage)

        steady = num_k_tiles - (num_buffers - 1)
        if const_expr(direct_b_global):
            init = [column.load() for column in direct_c_columns]
            results = init
            if const_expr(steady > 0):
                for kt, state in range(0, steady, 1, init=init):
                    for ni in range_constexpr(reg_n):
                        direct_c_columns[ni].store(state[ni])
                    stage = kt % num_buffers
                    _pipeline_fence(num_buffers - 2)
                    _compute(
                        stage,
                        kt,
                        kt + (num_buffers - 1),
                    )
                    results = yield [
                        column.load() for column in direct_c_columns
                    ]
                for ni in range_constexpr(reg_n):
                    direct_c_columns[ni].store(results[ni])

            for j in range_constexpr(num_buffers - 1):
                logical_k = steady + j
                stage = logical_k % num_buffers
                _pipeline_fence(num_buffers - 2 - j)
                _compute(stage, logical_k, None)

            for ni in range_constexpr(reg_n):
                frag_c[None, None, ni].store(
                    direct_c_columns[ni].load()
                )
        else:
            init = [frag_c.load()]
            results = init
            if const_expr(steady > 0):
                for kt, state in range(0, steady, 1, init=init):
                    frag_c.store(state[0])
                    stage = kt % num_buffers
                    _pipeline_fence(num_buffers - 2)
                    _compute(
                        stage,
                        kt,
                        kt + (num_buffers - 1),
                    )
                    results = yield [frag_c.load()]
                frag_c.store(results)

            for j in range_constexpr(num_buffers - 1):
                logical_k = steady + j
                stage = logical_k % num_buffers
                _pipeline_fence(num_buffers - 2 - j)
                _compute(stage, logical_k, None)

        copy_out = fx.make_copy_atom(
            fx.rocdl.BufferCopy(16),
            elem_cls,
        )
        thr_out = fx.make_tiled_copy_C(
            copy_out,
            tiled_mma,
        ).get_slice(tid)
        frag_out = fx.make_fragment_like(
            frag_c,
            elem_cls.ir_type,
        )
        frag_out.store(frag_c.load().to(elem_cls))
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
        kernel(
            c_view,
            a_view,
            b_view,
            tiled_mma,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu,
                "rocdl.flat_work_group_size": (
                    f"{block_threads},{block_threads}"
                ),
            },
        ).launch(
            grid=(grid_m * grid_n, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    launch.compile_hints["llvm_options"] = {
        "unroll-threshold": 300 if main_loop_unroll else 0,
        "amdgpu-expert-scheduling-mode": True,
    }
    return launch, block_m, block_n, block_k


@lru_cache(maxsize=64)
def _cached_all_compute_module(
    M,
    N,
    K,
    in_dtype,
    reg_m,
    reg_n,
    reg_k,
    waves_m,
    waves_n,
    num_buffers,
    swizzle_m,
    direct_b_global,
    use_xcd_remap,
    main_loop_unroll,
):
    return _create_all_compute_module(
        M,
        N,
        K,
        in_dtype=in_dtype,
        reg_m=reg_m,
        reg_n=reg_n,
        reg_k=reg_k,
        waves_m=waves_m,
        waves_n=waves_n,
        num_buffers=num_buffers,
        swizzle_m=swizzle_m,
        direct_b_global=direct_b_global,
        use_xcd_remap=use_xcd_remap,
        main_loop_unroll=main_loop_unroll,
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


def _validate_gemm_inputs(
    a: torch.Tensor,
    b: torch.Tensor,
    out_dtype: torch.dtype | None,
    out: torch.Tensor | None,
) -> tuple[int, int, int, torch.dtype]:
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
    if out_dtype != a.dtype:
        raise ValueError(
            "all-compute GEMM supports matching FP16->FP16 or BF16->BF16 only"
        )

    m, k = a.shape
    n = b.shape[0]
    if min(m, n, k) <= 0:
        raise ValueError(f"M, N and K must be positive, got {(m, n, k)}")
    if out is not None and (
        out.shape != (m, n)
        or out.dtype != out_dtype
        or out.device != a.device
    ):
        raise ValueError(
            f"out must have shape {(m, n)}, dtype {out_dtype}, and device {a.device}"
        )
    return m, n, k, out_dtype


def _gemm_all_compute(
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
    swizzle_m: int = 32,
    direct_b_global: bool = False,
    use_xcd_remap: bool = False,
    main_loop_unroll: bool = False,
    _skip_validation: bool = False,
) -> torch.Tensor:
    """Run one FP16/BF16 role-fused all-compute candidate."""

    if not _skip_validation:
        m, n, k, _ = _validate_gemm_inputs(a, b, a.dtype, out)
    else:
        m, k = a.shape
        n = b.shape[0]

    block_m = WMMA_M * reg_m * waves_m
    block_n = WMMA_N * reg_n * waves_n
    block_k = WMMA_K * reg_k
    padded_m = _round_up(m, block_m)
    padded_n = _round_up(n, block_n)
    padded_k = max(_round_up(k, block_k), num_buffers * block_k)

    if (m, k) == (padded_m, padded_k):
        a_kernel = a.contiguous()
    else:
        a_kernel = torch.zeros(
            (padded_m, padded_k),
            dtype=a.dtype,
            device=a.device,
        )
        a_kernel[:m, :k].copy_(a)
    if (n, k) == (padded_n, padded_k):
        b_kernel = b.contiguous()
    else:
        b_kernel = torch.zeros(
            (padded_n, padded_k),
            dtype=b.dtype,
            device=b.device,
        )
        b_kernel[:n, :k].copy_(b)
    if (
        out is not None
        and (m, n) == (padded_m, padded_n)
        and out.is_contiguous()
    ):
        c_kernel = out
    else:
        c_kernel = torch.empty(
            (padded_m, padded_n),
            dtype=a.dtype,
            device=a.device,
        )

    dtype_name = "f16" if a.dtype == torch.float16 else "bf16"
    launch, _, _, _ = _cached_all_compute_module(
        padded_m,
        padded_n,
        padded_k,
        dtype_name,
        reg_m,
        reg_n,
        reg_k,
        waves_m,
        waves_n,
        num_buffers,
        swizzle_m,
        direct_b_global,
        use_xcd_remap,
        main_loop_unroll,
    )
    _run_compiled(
        launch,
        c_kernel,
        a_kernel,
        b_kernel,
        torch.cuda.current_stream(),
        device_index=a.device.index or 0,
    )
    result = c_kernel[:m, :n]
    if out is c_kernel:
        return out
    if out is not None:
        out.copy_(result)
        return out
    return result


_BF16_EXACT_CONFIGS = {
    (128, 2048, 4096): dict(
        reg_m=2,
        reg_n=1,
        reg_k=4,
        waves_m=4,
        waves_n=2,
        num_buffers=4,
    ),
    (512, 2048, 7168): dict(
        reg_m=2,
        reg_n=2,
        reg_k=4,
        waves_m=4,
        waves_n=2,
        num_buffers=3,
    ),
    (2048, 1024, 7168): dict(
        reg_m=2,
        reg_n=4,
        reg_k=4,
        waves_m=4,
        waves_n=2,
        num_buffers=3,
    ),
}

_DIRECT_B_CONFIG = dict(
    reg_m=8,
    reg_n=4,
    reg_k=2,
    waves_m=1,
    waves_n=4,
    num_buffers=2,
    swizzle_m=32,
    direct_b_global=True,
    use_xcd_remap=True,
)


def _select_all_compute_config(
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
) -> tuple[str, dict]:
    if dtype == torch.bfloat16:
        exact = _BF16_EXACT_CONFIGS.get((m, n, k))
        if exact is not None:
            return "all_compute_8w", dict(exact)
        if m >= 4096 and n % 256 == 0 and k % 64 == 0:
            return "direct_b", dict(_DIRECT_B_CONFIG)

    if m <= 64:
        return (
            "all_compute_1w",
            {
                "reg_m": 1,
                "reg_n": 1 if n <= 64 else 2,
                "reg_k": 8 if k >= 256 else 2,
                "waves_m": 1,
                "waves_n": 1,
                "num_buffers": 3,
                "swizzle_m": 32,
                "main_loop_unroll": k >= 1024,
            },
        )
    if m <= 256:
        return (
            "all_compute_4w",
            {
                "reg_m": 2,
                "reg_n": 4,
                "reg_k": 4 if k >= 128 else 2,
                "waves_m": 2,
                "waves_n": 2,
                "num_buffers": 3,
                "swizzle_m": 32,
            },
        )
    return (
        "all_compute_8w",
        {
            "reg_m": 2,
            "reg_n": 4,
            "reg_k": 4 if k >= 128 else 2,
            "waves_m": 4,
            "waves_n": 2,
            "num_buffers": 3,
            "swizzle_m": 32,
        },
    )


def gemm_a16w16(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    out_dtype: torch.dtype | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``a @ b.T`` using only role-fused all-compute kernels."""

    m, n, k, out_dtype = _validate_gemm_inputs(a, b, out_dtype, out)
    _, config = _select_all_compute_config(m, n, k, a.dtype)
    return _gemm_all_compute(
        a,
        b,
        out=out,
        _skip_validation=True,
        **config,
    )


__all__ = [
    "gemm_a16w16",
    "_gemm_all_compute",
    "_select_all_compute_config",
]


