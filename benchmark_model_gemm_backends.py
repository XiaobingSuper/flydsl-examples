#!/usr/bin/env python3
"""Reproducible Kimi-K3 BF16 GEMM comparison on gfx1250.

Backends:
  * Aiter generic Triton (forced backend, small per-shape config sweep)
  * Aiter Gluon (gfx1250 native auto-config path)
  * Aiter Opus (PR #4246 tuned kid when present, heuristic otherwise)
  * Local FlyDSL tile-layout kernel (K/layout prep outside timed region)
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile


AITer_ROOT = Path(os.environ.get("AITER_ROOT", "/home/xiaobizh/aiter"))
FLYDSL_EXAMPLES_ROOT = Path(
    os.environ.get("FLYDSL_EXAMPLES_ROOT", "/home/xiaobizh/flydsl-examples")
)
MODEL_CONFIGS = {
    "kimik3": "kimik3_bf16_tuned_gemm.csv",
    "dsv4": "dsv4_bf16_tuned_gemm.csv",
}

KIMIK3_SHAPES = [
    {
        "name": "decode_opus",
        "category": "memory/launch-bound",
        "M": 1,
        "N": 1536,
        "K": 7168,
        "reason": "single-token decode; PR-tuned Opus winner",
    },
    {
        "name": "decode_short_k",
        "category": "memory/launch-bound",
        "M": 32,
        "N": 3072,
        "K": 128,
        "reason": "small-K projection; PR config keeps Triton",
    },
    {
        "name": "decode_wide",
        "category": "memory-bound",
        "M": 32,
        "N": 7168,
        "K": 4224,
        "reason": "wide decode projection; Triton-favored family",
    },
    {
        "name": "prefill_opus",
        "category": "mixed",
        "M": 128,
        "N": 896,
        "K": 7168,
        "reason": "small prefill/router projection; tuned Opus split-K",
    },
    {
        "name": "transition_opus",
        "category": "mixed",
        "M": 512,
        "N": 3072,
        "K": 7168,
        "reason": "largest M where Kimi config broadly favors Opus",
    },
    {
        "name": "compute_transition",
        "category": "compute-bound",
        "M": 1024,
        "N": 3072,
        "K": 7168,
        "reason": "backend crossover; Kimi config selects Triton",
    },
    {
        "name": "compute_large",
        "category": "compute-bound",
        "M": 4096,
        "N": 7168,
        "K": 8448,
        "reason": "large production GEMM; strongly compute-bound",
    },
]

DSV4_SHAPES = [
    {
        "name": "decode_projection",
        "category": "memory/launch-bound",
        "M": 1,
        "N": 1024,
        "K": 4096,
        "reason": "single-token DSv4 projection; tuned Opus winner",
    },
    {
        "name": "decode_skinny_n",
        "category": "memory/launch-bound",
        "M": 32,
        "N": 64,
        "K": 7168,
        "reason": "very skinny N projection with large K",
    },
    {
        "name": "decode_wide_n",
        "category": "memory-bound",
        "M": 32,
        "N": 32320,
        "K": 7168,
        "reason": "wide DSv4 projection; tuned config selects Triton",
    },
    {
        "name": "prefill_opus",
        "category": "mixed",
        "M": 128,
        "N": 2048,
        "K": 4096,
        "reason": "small prefill projection; tuned Opus cluster split-K",
    },
    {
        "name": "transition_opus",
        "category": "mixed",
        "M": 512,
        "N": 2048,
        "K": 7168,
        "reason": "mid-M transition still selected as Opus",
    },
    {
        "name": "compute_opus",
        "category": "compute-bound",
        "M": 2048,
        "N": 1024,
        "K": 7168,
        "reason": "compute-heavy shape whose tuned winner remains Opus",
    },
    {
        "name": "compute_crossover",
        "category": "compute-bound",
        "M": 4096,
        "N": 2048,
        "K": 4096,
        "reason": "large-M backend crossover; tuned config selects Triton",
    },
    {
        "name": "compute_large",
        "category": "compute-bound",
        "M": 16384,
        "N": 2048,
        "K": 4096,
        "reason": "large DSv4 production GEMM",
    },
]


def _git_rev(path: Path) -> dict[str, str]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            [
                "git",
                "-c",
                f"safe.directory={path}",
                "-C",
                str(path),
                *args,
            ],
            text=True,
        ).strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "status": run("status", "--short"),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed-as-package"


def _load_local_flydsl_module():
    path = FLYDSL_EXAMPLES_ROOT / "kernels/gemm_a16w16_gfx1250.py"
    spec = importlib.util.spec_from_file_location("local_gemm_a16w16_gfx1250", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_tuned_rows(path: Path) -> dict[tuple[int, int, int], dict[str, str]]:
    with path.open(newline="") as handle:
        rows = csv.DictReader(handle)
        return {
            (int(row["M"]), int(row["N"]), int(row["K"])): row
            for row in rows
            if row["gfx"] == "gfx1250"
        }


def _estimated_bytes_moved(m: int, n: int, k: int) -> float:
    return 2.0 * (m * k + n * k + m * n)


def _arithmetic_intensity(m: int, n: int, k: int) -> float:
    return 2.0 * m * n * k / _estimated_bytes_moved(m, n, k)


def _effective_bandwidth_gbps(
    m: int,
    n: int,
    k: int,
    time_us: float,
) -> float:
    return _estimated_bytes_moved(m, n, k) / time_us / 1.0e3


def _median_us(
    fn,
    warmup: int,
    iterations: int,
    batch_repeats: int,
) -> tuple[float, float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends):
        start.record()
        for _ in range(batch_repeats):
            fn()
        end.record()
    torch.cuda.synchronize()
    samples = [
        start.elapsed_time(end) * 1.0e3 / batch_repeats
        for start, end in zip(starts, ends)
    ]
    return statistics.median(samples), min(samples), max(samples)


def _kernel_median_us(fn, warmup: int, iterations: int) -> tuple[float, int]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iterations):
            fn()
        torch.cuda.synchronize()
    durations = [
        float(event.device_time)
        for event in prof.events()
        if event.device_time > 0
    ]
    if not durations:
        raise RuntimeError("profiler returned no GPU kernel events")
    kernels_per_call = len(durations) // iterations
    if kernels_per_call < 1 or kernels_per_call * iterations != len(durations):
        return sum(durations) / iterations, -1
    per_call = [
        sum(
            durations[
                idx * kernels_per_call : (idx + 1) * kernels_per_call
            ]
        )
        for idx in range(iterations)
    ]
    return statistics.median(per_call), kernels_per_call


def _error_metrics(output: torch.Tensor, reference: torch.Tensor) -> dict[str, float | bool]:
    out = output.float()
    ref = reference.float()
    max_abs = float((out - ref).abs().max())
    rel_l2 = float((out - ref).norm() / ref.norm().clamp_min(1.0e-6))
    max_ref = float(ref.abs().max())
    passed = max_abs <= max(0.1 * max_ref, 1.0)
    return {
        "max_abs": max_abs,
        "rel_l2": rel_l2,
        "max_ref": max_ref,
        "passed": passed,
    }


def _base_triton_config(
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
    waves_per_eu: int,
    split_k: int,
    k: int,
) -> dict:
    split_block = math.ceil(k / split_k / block_k) * block_k
    return {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": block_k,
        "GROUP_SIZE_M": 8 if block_m >= 64 else 1,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "waves_per_eu": waves_per_eu,
        "matrix_instr_nonkdim": 16,
        "cache_modifier": ".cg" if block_m <= 32 else None,
        "NUM_KSPLIT": split_k,
        "SPLITK_BLOCK_SIZE": split_block,
    }


def _triton_candidates(m: int, k: int) -> list[dict]:
    if m <= 32:
        tiles = [
            (16, 32, 256, 4, 3, 8),
            (16, 64, 128, 4, 3, 6),
            (32, 64, 128, 4, 3, 4),
            (32, 128, 64, 4, 3, 4),
        ]
        splits = [1, 2, 4, 8]
    elif m <= 256:
        tiles = [
            (32, 64, 128, 4, 3, 4),
            (64, 64, 128, 4, 3, 4),
            (64, 128, 64, 4, 3, 2),
            (128, 128, 64, 8, 2, 2),
        ]
        splits = [1, 2, 4]
    else:
        tiles = [
            (64, 128, 64, 4, 3, 2),
            (128, 128, 64, 8, 2, 2),
            (128, 256, 64, 8, 2, 1),
            (256, 128, 64, 8, 2, 1),
            (128, 256, 64, 8, 3, 2),
            (256, 128, 64, 8, 3, 2),
            (64, 256, 64, 8, 3, 2),
            (256, 64, 64, 8, 3, 2),
            (256, 256, 64, 8, 2, 1),
            (256, 256, 64, 8, 3, 1),
            (128, 128, 128, 8, 2, 1),
        ]
        splits = [1, 2] if m <= 1024 else [1]
    configs = []
    for tile in tiles:
        for split_k in splits:
            if math.ceil(k / split_k / tile[2]) >= 1:
                configs.append(_base_triton_config(*tile, split_k, k))
    return configs


def _flydsl_candidates(m: int, k: int) -> list[dict]:
    def cfg(
        reg_m,
        reg_n,
        reg_k,
        waves_m,
        waves_n,
        *,
        stages=3,
        overlap="cross",
        swizzle=32,
        waves_per_eu=None,
        kernarg_preload=False,
        sched_strategy=None,
        main_loop_unroll=False,
    ):
        return {
            "reg_m": reg_m,
            "reg_n": reg_n,
            "reg_k": reg_k,
            "waves_m": waves_m,
            "waves_n": waves_n,
            "num_stages": stages,
            "overlap_mode": overlap,
            "swizzle_m": swizzle,
            "waves_per_eu": waves_per_eu,
            "kernarg_preload": kernarg_preload,
            "sched_strategy": sched_strategy,
            "main_loop_unroll": main_loop_unroll,
        }

    candidates = [{"name": "auto"}]
    if m <= 32:
        candidates += [
            cfg(1, 1, 4, 1, 2),
            cfg(1, 2, 4, 1, 2),
            cfg(1, 4, 4, 1, 2),
            cfg(1, 2, 2, 1, 2),
            cfg(1, 4, 2, 1, 2),
        ]
        if m > 16:
            candidates += [
                cfg(1, 4, 4, 2, 1),
                cfg(1, 8, 4, 2, 1),
            ]
        candidates += [
            cfg(1, 4, 4, 1, 2, waves_per_eu=2),
            cfg(1, 4, 4, 1, 2, waves_per_eu=4),
            cfg(1, 4, 4, 1, 2, kernarg_preload=True),
            cfg(1, 4, 4, 1, 2, sched_strategy="max-ilp"),
        ]
    elif m <= 256:
        candidates += [
            cfg(1, 4, 4, 2, 1),
            cfg(2, 4, 4, 2, 1),
            cfg(2, 8, 4, 2, 1),
            cfg(4, 4, 4, 2, 1),
            cfg(3, 6, 4, 2, 1),
            cfg(2, 8, 2, 2, 1),
            cfg(2, 8, 4, 2, 1, waves_per_eu=2),
            cfg(2, 8, 4, 2, 1, kernarg_preload=True),
            cfg(2, 8, 4, 2, 1, sched_strategy="max-ilp"),
        ]
    else:
        candidates += [
            cfg(2, 4, 4, 2, 1),
            cfg(2, 8, 4, 2, 1),
            cfg(3, 6, 4, 2, 1),
            cfg(4, 4, 4, 2, 1),
            cfg(4, 8, 4, 2, 1),
            cfg(4, 8, 4, 2, 1, stages=2),
            cfg(4, 8, 2, 2, 1),
            cfg(4, 8, 4, 2, 1, overlap="sync"),
            cfg(4, 8, 4, 2, 1, swizzle=64),
            cfg(4, 8, 4, 2, 1, waves_per_eu=1),
            cfg(4, 8, 4, 2, 1, waves_per_eu=2),
            cfg(4, 8, 4, 2, 1, kernarg_preload=True),
            cfg(4, 8, 4, 2, 1, sched_strategy="max-ilp"),
            cfg(4, 8, 4, 2, 1, sched_strategy="max-memory-clause"),
        ]
    if k >= 1024:
        if m <= 32:
            candidates.append(
                cfg(1, 4, 4, 1, 2, main_loop_unroll=True)
            )
        elif m <= 256:
            candidates.append(
                cfg(2, 8, 4, 2, 1, main_loop_unroll=True)
            )
        else:
            candidates.append(
                cfg(4, 8, 4, 2, 1, main_loop_unroll=True)
            )
    return candidates


class PreparedFlyDSL:
    def __init__(self, module, a: torch.Tensor, b: torch.Tensor, config: dict):
        self.module = module
        m, k = a.shape
        n = b.shape[0]
        if config.get("name") == "auto":
            reg_m, reg_n, reg_k, waves_m, waves_n = module._select_config(
                m,
                n,
                k,
                None,
                None,
                None,
                None,
                None,
            )
            config = {
                "reg_m": reg_m,
                "reg_n": reg_n,
                "reg_k": reg_k,
                "waves_m": waves_m,
                "waves_n": waves_n,
                "num_stages": module.DEFAULT_PIPELINE_STAGES,
                "overlap_mode": "cross",
                "swizzle_m": 32,
                "waves_per_eu": None,
                "kernarg_preload": False,
                "sched_strategy": None,
                "main_loop_unroll": False,
            }
        else:
            config = dict(config)
            reg_m = config["reg_m"]
            reg_n = config["reg_n"]
            reg_k = config["reg_k"]
            waves_m = config["waves_m"]
            waves_n = config["waves_n"]
        self.config = config
        block_n = module.WMMA_N * reg_n * waves_n
        block_k = module.WMMA_K * reg_k
        pk = max(module._round_up(k, block_k), 2 * block_k)
        self.a, self.lda = module._prepare_matrix_layout(a, pk)
        self.b, self.ldb = module._prepare_matrix_layout(b, pk)
        self.ldc = module._round_up(n, block_n)
        self.c = torch.empty(
            (m, self.ldc),
            dtype=torch.bfloat16,
            device=a.device,
        )
        self.m = m
        self.n = n
        self.view = self.c[:, :n]
        self.launch, _, _, _ = module._cached_module(
            pk,
            "bf16",
            "bf16",
            reg_m,
            reg_n,
            reg_k,
            waves_m,
            waves_n,
            config["num_stages"],
            config["overlap_mode"],
            config["swizzle_m"],
            config["waves_per_eu"],
            config["kernarg_preload"],
            config["sched_strategy"],
            config["main_loop_unroll"],
        )

    def __call__(self):
        self.module._run_compiled(
            self.launch,
            self.c,
            self.a,
            self.b,
            self.m,
            self.n,
            self.lda,
            self.ldb,
            self.ldc,
            torch.cuda.current_stream(),
            device_index=self.a.device.index or 0,
        )
        return self.view


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=sorted(MODEL_CONFIGS), default="kimik3")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--tune-warmup", type=int, default=3)
    parser.add_argument("--tune-iterations", type=int, default=8)
    parser.add_argument("--batch-repeats", type=int, default=10)
    parser.add_argument("--tune-batch-repeats", type=int, default=3)
    parser.add_argument("--output", default="kimik3_gemm_benchmark_results.json")
    parser.add_argument("--shape", action="append", help="run only named shape")
    args = parser.parse_args()

    sys.path.insert(0, str(AITer_ROOT))
    torch.cuda.set_device(args.device)

    import aiter
    import flydsl
    import triton
    from aiter.ops.opus import gemm_a16w16_opus
    from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16

    local_flydsl = _load_local_flydsl_module()
    tuned_path = (
        AITer_ROOT
        / "aiter/configs/model_configs"
        / MODEL_CONFIGS[args.suite]
    )
    tuned_rows = _load_tuned_rows(tuned_path)
    suite_shapes = DSV4_SHAPES if args.suite == "dsv4" else KIMIK3_SHAPES
    selected = [
        shape
        for shape in suite_shapes
        if not args.shape or shape["name"] in set(args.shape)
    ]

    environment = {
        "docker_image": os.environ.get("BENCH_DOCKER_IMAGE", "unspecified"),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "triton": triton.__version__,
        "flydsl_package": _package_version("flydsl"),
        "flydsl_module": str(Path(flydsl.__file__).resolve()),
        "aiter_module": str(Path(aiter.__file__).resolve()),
        "aiter_git": _git_rev(AITer_ROOT),
        "flydsl_examples_git": _git_rev(FLYDSL_EXAMPLES_ROOT),
        "gpu_name": torch.cuda.get_device_name(args.device),
        "gpu_properties": str(torch.cuda.get_device_properties(args.device)),
        "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES", ""),
    }

    results = []
    for shape in selected:
        m, n, k = shape["M"], shape["N"], shape["K"]
        print(f"\n=== {shape['name']} ({m}, {n}, {k}) ===", flush=True)
        torch.manual_seed(1234)
        a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
        out_triton = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        out_gluon = torch.empty_like(out_triton)
        out_opus = torch.empty((1, m, n), device="cuda", dtype=torch.bfloat16)
        reference = (a.float() @ b.float().T).to(torch.bfloat16)

        # Generic Triton: tune a compact, documented candidate set.
        best_triton = None
        for config in _triton_candidates(m, k):
            fn = lambda cfg=config: gemm_a16w16(
                a,
                b,
                None,
                torch.bfloat16,
                out_triton,
                config=cfg,
                backend="triton",
            )
            try:
                fn()
                torch.cuda.synchronize()
                err = _error_metrics(out_triton, reference)
                if not err["passed"]:
                    continue
                tune_us, _, _ = _median_us(
                    fn,
                    args.tune_warmup,
                    args.tune_iterations,
                    args.tune_batch_repeats,
                )
                if best_triton is None or tune_us < best_triton[0]:
                    best_triton = (tune_us, config, fn, err)
            except Exception as exc:  # noqa: BLE001
                print(f"triton candidate failed: {exc}", flush=True)
        if best_triton is None:
            triton_result = {"status": "failed", "error": "no valid candidate"}
        else:
            _, triton_cfg, triton_fn, triton_err = best_triton
            us, min_us, max_us = _median_us(
                triton_fn,
                args.warmup,
                args.iterations,
                args.batch_repeats,
            )
            kernel_us, kernels_per_call = _kernel_median_us(
                triton_fn,
                args.warmup,
                args.iterations,
            )
            triton_result = {
                "status": "ok",
                "kernel_us": kernel_us,
                "e2e_us": us,
                "e2e_min_us": min_us,
                "e2e_max_us": max_us,
                "kernels_per_call": kernels_per_call,
                "tflops": 2.0 * m * n * k / kernel_us / 1.0e6,
                "e2e_tflops": 2.0 * m * n * k / us / 1.0e6,
                "effective_bandwidth_gbps": _effective_bandwidth_gbps(
                    m, n, k, kernel_us
                ),
                "e2e_effective_bandwidth_gbps": _effective_bandwidth_gbps(
                    m, n, k, us
                ),
                "config": triton_cfg,
                "error": triton_err,
            }

        gluon_fn = lambda: gemm_a16w16(
            a,
            b,
            None,
            torch.bfloat16,
            out_gluon,
            backend="gluon",
        )
        try:
            gluon_fn()
            torch.cuda.synchronize()
            gluon_err = _error_metrics(out_gluon, reference)
            us, min_us, max_us = _median_us(
                gluon_fn,
                args.warmup,
                args.iterations,
                args.batch_repeats,
            )
            kernel_us, kernels_per_call = _kernel_median_us(
                gluon_fn,
                args.warmup,
                args.iterations,
            )
            gluon_result = {
                "status": "ok" if gluon_err["passed"] else "incorrect",
                "kernel_us": kernel_us,
                "e2e_us": us,
                "e2e_min_us": min_us,
                "e2e_max_us": max_us,
                "kernels_per_call": kernels_per_call,
                "tflops": 2.0 * m * n * k / kernel_us / 1.0e6,
                "e2e_tflops": 2.0 * m * n * k / us / 1.0e6,
                "effective_bandwidth_gbps": _effective_bandwidth_gbps(
                    m, n, k, kernel_us
                ),
                "e2e_effective_bandwidth_gbps": _effective_bandwidth_gbps(
                    m, n, k, us
                ),
                "error": gluon_err,
            }
        except Exception as exc:  # noqa: BLE001
            gluon_result = {"status": "failed", "error": str(exc)}

        csv_row = tuned_rows.get((m, n, k))
        explicit_opus = csv_row is not None and csv_row["libtype"] == "opus"
        opus_kwargs = {}
        if explicit_opus:
            opus_kwargs = {
                "kernelId": int(csv_row["solidx"]),
                "splitK": int(csv_row["splitK"]),
            }
        opus_fn = lambda: gemm_a16w16_opus(
            a,
            b,
            None,
            torch.bfloat16,
            out=out_opus,
            **opus_kwargs,
        )
        try:
            opus_out = opus_fn()
            torch.cuda.synchronize()
            opus_err = _error_metrics(opus_out, reference)
            us, min_us, max_us = _median_us(
                opus_fn,
                args.warmup,
                args.iterations,
                args.batch_repeats,
            )
            kernel_us, kernels_per_call = _kernel_median_us(
                opus_fn,
                args.warmup,
                args.iterations,
            )
            opus_result = {
                "status": "ok" if opus_err["passed"] else "incorrect",
                "kernel_us": kernel_us,
                "e2e_us": us,
                "e2e_min_us": min_us,
                "e2e_max_us": max_us,
                "kernels_per_call": kernels_per_call,
                "tflops": 2.0 * m * n * k / kernel_us / 1.0e6,
                "e2e_tflops": 2.0 * m * n * k / us / 1.0e6,
                "effective_bandwidth_gbps": _effective_bandwidth_gbps(
                    m, n, k, kernel_us
                ),
                "e2e_effective_bandwidth_gbps": _effective_bandwidth_gbps(
                    m, n, k, us
                ),
                "mode": "csv-explicit" if explicit_opus else "heuristic",
                "kernelId": opus_kwargs.get("kernelId"),
                "splitK": opus_kwargs.get("splitK"),
                "error": opus_err,
            }
        except Exception as exc:  # noqa: BLE001
            opus_result = {"status": "failed", "error": str(exc)}

        best_flydsl = None
        for flydsl_config in _flydsl_candidates(m, k):
            try:
                prepared = PreparedFlyDSL(
                    local_flydsl,
                    a,
                    b,
                    flydsl_config,
                )
                flydsl_out = prepared()
                torch.cuda.synchronize()
                candidate_err = _error_metrics(flydsl_out, reference)
                if not candidate_err["passed"]:
                    continue
                tune_us, _, _ = _median_us(
                    prepared,
                    args.tune_warmup,
                    args.tune_iterations,
                    args.tune_batch_repeats,
                )
                if best_flydsl is None or tune_us < best_flydsl[0]:
                    best_flydsl = (
                        tune_us,
                        prepared,
                        candidate_err,
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"flydsl candidate failed: {exc}", flush=True)
        if best_flydsl is None:
            flydsl_result = {
                "status": "failed",
                "error": "no valid candidate",
            }
        else:
            _, prepared_flydsl, flydsl_err = best_flydsl
            us, min_us, max_us = _median_us(
                prepared_flydsl,
                args.warmup,
                args.iterations,
                args.batch_repeats,
            )
            kernel_us, kernels_per_call = _kernel_median_us(
                prepared_flydsl,
                args.warmup,
                args.iterations,
            )
            flydsl_result = {
                "status": "ok" if flydsl_err["passed"] else "incorrect",
                "kernel_us": kernel_us,
                "e2e_us": us,
                "e2e_min_us": min_us,
                "e2e_max_us": max_us,
                "kernels_per_call": kernels_per_call,
                "tflops": 2.0 * m * n * k / kernel_us / 1.0e6,
                "e2e_tflops": 2.0 * m * n * k / us / 1.0e6,
                "effective_bandwidth_gbps": _effective_bandwidth_gbps(
                    m, n, k, kernel_us
                ),
                "e2e_effective_bandwidth_gbps": _effective_bandwidth_gbps(
                    m, n, k, us
                ),
                "timing_scope": "kernel-only; input/output padding outside timing",
                "config": prepared_flydsl.config,
                "error": flydsl_err,
            }

        row = {
            **shape,
            "arithmetic_intensity_flop_per_byte": _arithmetic_intensity(m, n, k),
            "estimated_bytes_moved": _estimated_bytes_moved(m, n, k),
            "csv_selected_backend": csv_row["libtype"] if csv_row else None,
            "csv_us": float(csv_row["us"]) if csv_row else None,
            "backends": {
                "aiter_triton": triton_result,
                "aiter_gluon": gluon_result,
                "aiter_opus_pr4246": opus_result,
                "local_flydsl": flydsl_result,
            },
        }
        results.append(row)
        print(json.dumps(row["backends"], indent=2), flush=True)
        torch.cuda.empty_cache()

    payload = {
        "suite": args.suite,
        "tuned_config": str(tuned_path),
        "environment": environment,
        "settings": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "tune_warmup": args.tune_warmup,
            "tune_iterations": args.tune_iterations,
            "batch_repeats": args.batch_repeats,
            "tune_batch_repeats": args.tune_batch_repeats,
            "timing": (
                "kernel_us = median sum of torch.profiler GPU kernel durations "
                "per call; e2e_us = median batched HIP-event elapsed time"
            ),
            "dtype": "bf16 inputs, fp32 accumulation, bf16 output",
            "bias": False,
        },
        "derived_metrics": {
            "flops": "2*M*N*K",
            "estimated_bytes_moved": "2*(M*K + N*K + M*N)",
            "tflops": "flops / time_us / 1e6",
            "effective_bandwidth_gbps": (
                "estimated_bytes_moved / time_us / 1e3"
            ),
            "bandwidth_note": (
                "Algorithmic lower-bound effective bandwidth, not measured "
                "HBM traffic; excludes cache and split-K workspace effects."
            ),
        },
        "results": results,
    }
    output = Path(args.output)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {output.resolve()}")


if __name__ == "__main__":
    main()
