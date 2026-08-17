#!/usr/bin/env python3
"""Tune focused all-compute A16W16 paths on compute-heavy DSv4 shapes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import flydsl  # noqa: F401 -- initialize COMGR before torch
import torch

from benchmark_model_gemm_backends import (
    _effective_bandwidth_gbps,
    _error_metrics,
    _estimated_bytes_moved,
    _kernel_median_us,
    _median_us,
)
from kernels.gemm_a16w16_gfx1250 import (
    _gemm_all_compute,
    _gemm_producer_consumer,
)

SHAPES = [
    {"name": "compute_opus", "M": 2048, "N": 1024, "K": 7168},
    {"name": "compute_crossover", "M": 4096, "N": 2048, "K": 4096},
    {"name": "compute_large", "M": 16384, "N": 2048, "K": 4096},
]

LOCAL_CONFIGS = {
    "compute_opus": {
        "reg_m": 4,
        "reg_n": 8,
        "reg_k": 4,
        "waves_m": 2,
        "waves_n": 1,
        "num_stages": 3,
        "overlap_mode": "cross",
    },
    "compute_crossover": {
        "reg_m": 4,
        "reg_n": 8,
        "reg_k": 4,
        "waves_m": 2,
        "waves_n": 1,
        "num_stages": 3,
        "overlap_mode": "cross",
        "sched_strategy": "max-memory-clause",
    },
    "compute_large": {
        "reg_m": 4,
        "reg_n": 8,
        "reg_k": 4,
        "waves_m": 2,
        "waves_n": 1,
        "num_stages": 3,
        "overlap_mode": "cross",
        "sched_strategy": "max-memory-clause",
    },
}


def _candidate(
    *,
    reg_m=4,
    reg_n=4,
    reg_k=4,
    waves_m=2,
    waves_n=2,
    num_buffers=2,
    swizzle_m=32,
    direct_b_global=False,
    use_xcd_remap=False,
):
    return {
        "reg_m": reg_m,
        "reg_n": reg_n,
        "reg_k": reg_k,
        "waves_m": waves_m,
        "waves_n": waves_n,
        "num_buffers": num_buffers,
        "swizzle_m": swizzle_m,
        "direct_b_global": direct_b_global,
        "use_xcd_remap": use_xcd_remap,
    }


CANDIDATES = [
    _candidate(
        reg_m=2,
        reg_n=4,
        reg_k=4,
        waves_m=4,
        waves_n=2,
        num_buffers=3,
    ),
    _candidate(
        reg_m=8,
        reg_n=4,
        reg_k=2,
        waves_m=1,
        waves_n=4,
        num_buffers=3,
        swizzle_m=8,
        direct_b_global=True,
        use_xcd_remap=True,
    ),
]


def _measure(
    fn,
    m,
    n,
    k,
    error,
    *,
    warmup,
    iterations,
    batch_repeats,
):
    e2e_us, e2e_min_us, e2e_max_us = _median_us(
        fn,
        warmup,
        iterations,
        batch_repeats,
    )
    kernel_us, kernels_per_call = _kernel_median_us(
        fn,
        warmup,
        iterations,
    )
    flops = 2.0 * m * n * k
    return {
        "status": "ok",
        "kernel_us": kernel_us,
        "e2e_us": e2e_us,
        "e2e_min_us": e2e_min_us,
        "e2e_max_us": e2e_max_us,
        "kernels_per_call": kernels_per_call,
        "tflops": flops / kernel_us / 1.0e6,
        "e2e_tflops": flops / e2e_us / 1.0e6,
        "effective_bandwidth_gbps": _effective_bandwidth_gbps(
            m,
            n,
            k,
            kernel_us,
        ),
        "estimated_bytes_moved": _estimated_bytes_moved(m, n, k),
        "error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--tune-warmup", type=int, default=3)
    parser.add_argument("--tune-iterations", type=int, default=8)
    parser.add_argument("--batch-repeats", type=int, default=10)
    parser.add_argument("--tune-batch-repeats", type=int, default=3)
    parser.add_argument("--correctness-repeats", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("a16w16_all_compute_gfx1250_results.json"),
    )
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    results = []

    for shape in SHAPES:
        m, n, k = shape["M"], shape["N"], shape["K"]
        print(f"\n=== {shape['name']} ({m}, {n}, {k}) ===", flush=True)
        torch.manual_seed(1234)
        a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
        out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        reference = (a.float() @ b.float().T).to(torch.bfloat16)

        local_fn = lambda: _gemm_producer_consumer(
            a,
            b,
            out=out,
            **LOCAL_CONFIGS[shape["name"]],
        )
        local_error = _error_metrics(local_fn(), reference)
        local = _measure(
            local_fn,
            m,
            n,
            k,
            local_error,
            warmup=args.warmup,
            iterations=args.iterations,
            batch_repeats=args.batch_repeats,
        )
        local["config"] = LOCAL_CONFIGS[shape["name"]]

        trials = []
        best = None
        for config in CANDIDATES:
            fn = lambda config=config: _gemm_all_compute(
                a,
                b,
                out=out,
                **config,
            )
            errors = [
                _error_metrics(fn(), reference)
                for _ in range(args.correctness_repeats)
            ]
            error = {
                "max_abs": max(item["max_abs"] for item in errors),
                "rel_l2": max(item["rel_l2"] for item in errors),
                "max_ref": max(item["max_ref"] for item in errors),
                "passed": all(item["passed"] for item in errors),
            }
            trial = {"config": config, "error": error}
            if error["passed"]:
                tune_us, _, _ = _median_us(
                    fn,
                    args.tune_warmup,
                    args.tune_iterations,
                    args.tune_batch_repeats,
                )
                trial["tune_e2e_us"] = tune_us
                if best is None or tune_us < best[0]:
                    best = (tune_us, config, fn, error)
            trials.append(trial)
            print(config, trial.get("tune_e2e_us"), error["passed"])

        if best is None:
            raise RuntimeError(f"no correct all-compute candidate for {shape}")
        _, config, fn, error = best
        all_compute = _measure(
            fn,
            m,
            n,
            k,
            error,
            warmup=args.warmup,
            iterations=args.iterations,
            batch_repeats=args.batch_repeats,
        )
        all_compute["config"] = config
        results.append(
            {
                **shape,
                "local": local,
                "all_compute": all_compute,
                "candidate_trials": trials,
            }
        )
        print(json.dumps(results[-1], indent=2), flush=True)

    payload = {
        "gpu": "physical GPU3 / gfx1250",
        "settings": vars(args) | {"output": str(args.output)},
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
