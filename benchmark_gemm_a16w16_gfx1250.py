#!/usr/bin/env python3
"""Measure gfx1250 A16W16 device-kernel time with torch.profiler."""

import argparse
import statistics

import flydsl  # noqa: F401 -- load COMGR before torch loads HIP LLVM
import torch
from torch.profiler import ProfilerActivity, profile

from kernels.gemm_a16w16_gfx1250 import (
    _gemm_producer_consumer,
    gemm_a16w16,
)


DEFAULT_SHAPES = ((2048, 2048, 2048), (4096, 4096, 4096), (8192, 8192, 8192))


def _parse_shape(value: str) -> tuple[int, int, int]:
    parts = value.lower().replace("x", " ").split()
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected MxNxK, got {value!r}")
    return tuple(map(int, parts))


def _device_kernel_times_us(prof) -> list[float]:
    times = []
    for event in prof.events():
        if event.device_time <= 0:
            continue
        name = event.name.lower()
        if "gemm_kernel" in name or "gemm_a16w16" in name:
            times.append(float(event.device_time))
    return times


def benchmark_shape(
    m: int,
    n: int,
    k: int,
    *,
    dtype: torch.dtype,
    warmup: int,
    iterations: int,
    peak_pflops: float,
    kernel_kwargs: dict | None = None,
) -> dict:
    torch.manual_seed(0)
    a = torch.randn((m, k), device="cuda", dtype=dtype)
    b = torch.randn((n, k), device="cuda", dtype=dtype)
    out = torch.empty((m, n), device="cuda", dtype=dtype)
    kernel_kwargs = kernel_kwargs or {}
    kernel = _gemm_producer_consumer if kernel_kwargs else gemm_a16w16

    for _ in range(warmup):
        kernel(a, b, out=out, **kernel_kwargs)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iterations):
            kernel(a, b, out=out, **kernel_kwargs)
        torch.cuda.synchronize()

    times_us = _device_kernel_times_us(prof)
    if len(times_us) != iterations:
        device_events = sorted(
            {
                event.name
                for event in prof.events()
                if event.device_time > 0
            }
        )
        raise RuntimeError(
            f"expected {iterations} GEMM kernel events, found {len(times_us)}; "
            f"device events={device_events}"
        )

    median_us = statistics.median(times_us)
    min_us = min(times_us)
    max_us = max(times_us)
    flops = 2 * m * n * k
    tflops = flops / (median_us * 1.0e6)
    pflops = tflops / 1000.0
    return {
        "shape": f"{m}x{n}x{k}",
        "median_us": median_us,
        "min_us": min_us,
        "max_us": max_us,
        "pflops": pflops,
        "peak_percent": pflops / peak_pflops * 100.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shape",
        type=_parse_shape,
        action="append",
        dest="shapes",
        help="shape to benchmark, for example 4096x4096x4096; repeatable",
    )
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--peak-pflops",
        type=float,
        default=4.454,
        help="reference BF16 peak; defaults to the local gfx1250 WMMA microbenchmark",
    )
    parser.add_argument(
        "--stages",
        type=int,
        default=3,
        choices=(2, 3),
        help="number of circular TDM/LDS pipeline stages",
    )
    parser.add_argument("--reg-m", type=int)
    parser.add_argument("--reg-n", type=int)
    parser.add_argument("--reg-k", type=int)
    parser.add_argument("--waves-m", type=int)
    parser.add_argument("--waves-n", type=int)
    parser.add_argument(
        "--overlap",
        choices=("sync", "cross"),
        default="cross",
    )
    parser.add_argument("--swizzle-m", type=int, default=32)
    parser.add_argument("--waves-per-eu", type=int)
    parser.add_argument("--kernarg-preload", action="store_true")
    parser.add_argument(
        "--sched-strategy",
        choices=("max-ilp", "max-memory-clause"),
    )
    parser.add_argument("--main-loop-unroll", action="store_true")
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    shapes = args.shapes or DEFAULT_SHAPES
    kernel_kwargs = {
        "num_stages": args.stages,
        "overlap_mode": args.overlap,
        "swizzle_m": args.swizzle_m,
        "waves_per_eu": args.waves_per_eu,
        "kernarg_preload": args.kernarg_preload,
        "sched_strategy": args.sched_strategy,
        "main_loop_unroll": args.main_loop_unroll,
        **{
            name: value
            for name, value in (
                ("reg_m", args.reg_m),
                ("reg_n", args.reg_n),
                ("reg_k", args.reg_k),
                ("waves_m", args.waves_m),
                ("waves_n", args.waves_n),
            )
            if value is not None
        },
    }

    print(
        f"GPU={torch.cuda.get_device_name(args.device)} dtype={args.dtype} "
        f"config={kernel_kwargs} theoretical_peak={args.peak_pflops:.3f} PFLOPS"
    )
    for shape in shapes:
        result = benchmark_shape(
            *shape,
            dtype=dtype,
            warmup=args.warmup,
            iterations=args.iterations,
            peak_pflops=args.peak_pflops,
            kernel_kwargs=kernel_kwargs,
        )
        print(
            f"{result['shape']}: median={result['median_us']:.3f} us "
            f"range=[{result['min_us']:.3f}, {result['max_us']:.3f}] us "
            f"throughput={result['pflops']:.3f} PFLOPS "
            f"peak={result['peak_percent']:.2f}%"
        )


if __name__ == "__main__":
    main()
