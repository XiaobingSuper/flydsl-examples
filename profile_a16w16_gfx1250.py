#!/usr/bin/env python3
"""Small repeatable workload for rocprofv3 A16W16 counter collection."""

import argparse

import flydsl  # noqa: F401 -- initialize COMGR before torch
import torch

from kernels.gemm_a16w16_gfx1250 import (
    _gemm_all_compute,
    _gemm_producer_consumer,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("local", "allcompute8", "directb"),
        required=True,
    )
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=7168)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    torch.manual_seed(1234)
    a = torch.randn((args.m, args.k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((args.n, args.k), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((args.m, args.n), device="cuda", dtype=torch.bfloat16)

    if args.mode == "local":
        fn = lambda: _gemm_producer_consumer(
            a,
            b,
            out=out,
            reg_m=4,
            reg_n=8,
            reg_k=4,
            waves_m=2,
            waves_n=1,
            num_stages=3,
            overlap_mode="cross",
        )
    elif args.mode == "allcompute8":
        fn = lambda: _gemm_all_compute(
            a,
            b,
            out=out,
            reg_m=2,
            reg_n=4,
            reg_k=4,
            waves_m=4,
            waves_n=2,
            num_buffers=3,
        )
    elif args.mode == "directb":
        fn = lambda: _gemm_all_compute(
            a,
            b,
            out=out,
            reg_m=8,
            reg_n=4,
            reg_k=2,
            waves_m=1,
            waves_n=4,
            num_buffers=3,
            swizzle_m=8,
            direct_b_global=True,
            use_xcd_remap=True,
        )

    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    for _ in range(args.iterations):
        fn()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
