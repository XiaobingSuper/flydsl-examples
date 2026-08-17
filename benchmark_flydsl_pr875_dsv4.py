#!/usr/bin/env python3
"""Tune FlyDSL PR #875 on the representative DeepSeek-V4 GEMM shapes."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import sys
from pathlib import Path

import torch

from benchmark_model_gemm_backends import (
    DSV4_SHAPES,
    _effective_bandwidth_gbps,
    _error_metrics,
    _estimated_bytes_moved,
    _kernel_median_us,
    _median_us,
    _package_version,
)

PR_URL = "https://github.com/ROCm/FlyDSL/pull/875"
PR_HEAD = "1273aa33d5eb8c6f5b724f890fdf159439f8bcc4"
DEFAULT_PR_ROOT = Path("/home/xiaobizh/FlyDSL-pr875")


def _config(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    m_warp: int,
    n_warp: int,
    *,
    num_buffers: int = 2,
    waves_per_eu: int | None = None,
    kernarg_preload: bool = False,
    split_k: int = 1,
    sched_strategy: str | None = None,
    main_loop_unroll: bool = False,
    variant: str = "bandwidth_bound",
) -> dict:
    return {
        "tile_m": tile_m,
        "tile_n": tile_n,
        "tile_k": tile_k,
        "m_warp": m_warp,
        "n_warp": n_warp,
        "num_buffers": num_buffers,
        "waves_per_eu": waves_per_eu,
        "kernarg_preload": kernarg_preload,
        "split_k": split_k,
        "sched_strategy": sched_strategy,
        "main_loop_unroll": main_loop_unroll,
        "variant": variant,
    }


def _valid_config(config: dict, k: int) -> bool:
    tile_m = config["tile_m"]
    tile_n = config["tile_n"]
    tile_k = config["tile_k"]
    m_warp = config["m_warp"]
    n_warp = config["n_warp"]
    split_k = config["split_k"]
    num_buffers = config["num_buffers"]
    if not 1 <= m_warp * n_warp <= 8:
        return False
    if tile_m % (16 * m_warp) or tile_n % (16 * n_warp):
        return False
    if tile_k < 32 or tile_k & (tile_k - 1):
        return False
    if k % split_k or (k // split_k) % tile_k:
        return False
    return k // split_k // tile_k >= num_buffers - 1


def _split_candidates(
    m: int,
    n: int,
    k: int,
    anchor: dict,
) -> list[dict]:
    if m > 128:
        return []
    grid = math.ceil(m / anchor["tile_m"]) * math.ceil(
        n / anchor["tile_n"]
    )
    target = max(1.0, 256.0 / grid)
    k_tiles = k // anchor["tile_k"]
    divisors = [
        value
        for value in range(2, min(k_tiles, 128) + 1)
        if k_tiles % value == 0
    ]
    divisors.sort(key=lambda value: abs(math.log2(value / target)))
    return [
        {**anchor, "split_k": value, "num_buffers": 2}
        for value in divisors[:2]
        if target > 1.5
    ]


def _candidates(m: int, n: int, k: int) -> list[dict]:
    default = _config(128, 128, 32, 2, 4)
    if m <= 32:
        geometries = [
            _config(16, 64, 128, 1, 4),
            _config(16, 128, 128, 1, 4),
            _config(32, 64, 128, 2, 2),
            _config(32, 128, 128, 2, 4),
            _config(64, 128, 64, 2, 4, num_buffers=3),
        ]
        anchor = geometries[0] if m <= 16 else geometries[2]
        wpe_values = (2, 4)
    elif m <= 256:
        geometries = [
            _config(64, 64, 128, 2, 2),
            _config(64, 128, 128, 2, 4),
            _config(128, 128, 64, 2, 4),
            _config(128, 128, 128, 2, 4),
            _config(64, 256, 64, 2, 4, num_buffers=3),
        ]
        anchor = geometries[3]
        wpe_values = (2, 4)
    else:
        geometries = [
            _config(128, 128, 128, 2, 2),
            _config(128, 256, 64, 2, 4),
            _config(256, 128, 64, 4, 2),
            _config(128, 128, 64, 2, 2),
            _config(64, 256, 128, 2, 4),
        ]
        anchor = geometries[0]
        wpe_values = (1, 2)

    candidates = [default, *geometries]
    candidates += [
        {**anchor, "num_buffers": 3},
        {**anchor, "variant": "compute_bound"},
        {**anchor, "kernarg_preload": True},
        {**anchor, "sched_strategy": "max-ilp"},
        {**anchor, "sched_strategy": "max-memory-clause"},
        {**anchor, "main_loop_unroll": True},
        *[{**anchor, "waves_per_eu": value} for value in wpe_values],
        *_split_candidates(m, n, k, anchor),
    ]

    unique = []
    seen = set()
    for candidate in candidates:
        key = json.dumps(candidate, sort_keys=True)
        if key not in seen and _valid_config(candidate, k):
            seen.add(key)
            unique.append(candidate)
    return unique


def _load_pr875(pr_root: Path, use_pr_python: bool):
    import flydsl

    if use_pr_python:
        source_package = str(pr_root / "python/flydsl")
        if source_package not in flydsl.__path__:
            flydsl.__path__.insert(0, source_package)
        for name in list(sys.modules):
            if (
                name.startswith("flydsl.")
                and not name.startswith("flydsl._mlir")
            ):
                del sys.modules[name]
    sys.path.insert(0, str(pr_root))
    for name in list(sys.modules):
        if name == "kernels" or name.startswith("kernels."):
            del sys.modules[name]
    kernel = importlib.import_module(
        "kernels.gemm.gemm_a16w16_gfx1250"
    )
    tensor_shim = importlib.import_module("kernels.common.tensor_shim")
    return kernel, tensor_shim._run_compiled


class PreparedPR875:
    def __init__(
        self,
        kernel,
        run_compiled,
        launcher_cache: dict,
        a: torch.Tensor,
        b: torch.Tensor,
        config: dict,
    ):
        self.run_compiled = run_compiled
        self.m, self.k = a.shape
        self.n = b.shape[0]
        self.config = dict(config)
        m_padded = math.ceil(
            self.m / config["tile_m"]
        ) * config["tile_m"]
        self.n_stride = math.ceil(
            self.n / config["tile_n"]
        ) * config["tile_n"]
        if m_padded == self.m:
            self.a = a
        else:
            self.a = torch.zeros(
                (m_padded, self.k),
                device=a.device,
                dtype=a.dtype,
            )
            self.a[: self.m].copy_(a)
        if self.n_stride == self.n:
            self.b = b
        else:
            self.b = torch.zeros(
                (self.n_stride, self.k),
                device=b.device,
                dtype=b.dtype,
            )
            self.b[: self.n].copy_(b)
        self.bias = torch.empty(
            0,
            device=a.device,
            dtype=torch.bfloat16,
        )
        self.split_k = config["split_k"]
        accum_dtype = (
            torch.float32 if self.split_k > 1 else torch.bfloat16
        )
        self.accum = torch.empty(
            (self.m, self.n_stride),
            device=a.device,
            dtype=accum_dtype,
        )
        self.output = (
            torch.empty(
                (self.m, self.n),
                device=a.device,
                dtype=torch.bfloat16,
            )
            if self.split_k > 1
            else self.accum[:, : self.n]
        )
        cache_key = (
            self.k,
            "f32" if self.split_k > 1 else "bf16",
            json.dumps(config, sort_keys=True),
        )
        if cache_key not in launcher_cache:
            launcher_cache[cache_key] = kernel.compile_gemm_a16w16(
                M=0,
                N=0,
                K=self.k,
                tile_m=config["tile_m"],
                tile_n=config["tile_n"],
                tile_k=config["tile_k"],
                m_warp=config["m_warp"],
                n_warp=config["n_warp"],
                in_dtype="bf16",
                out_dtype=cache_key[1],
                num_buffers=config["num_buffers"],
                waves_per_eu=config["waves_per_eu"],
                kernarg_preload=config["kernarg_preload"],
                split_k=self.split_k,
                sched_strategy=config["sched_strategy"],
                main_loop_unroll=config["main_loop_unroll"],
                variant=config["variant"],
            )
        self.launcher = launcher_cache[cache_key]

    def __call__(self) -> torch.Tensor:
        if self.split_k > 1:
            self.accum.zero_()
        self.run_compiled(
            self.launcher,
            self.accum,
            self.a,
            self.b,
            self.bias,
            self.m,
            self.n_stride,
            torch.cuda.current_stream(),
        )
        if self.split_k > 1:
            self.output.copy_(self.accum[:, : self.n])
        return self.output


def _write_payload(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _exception_text(exc: Exception) -> str:
    lines = str(exc).splitlines()
    return "\n".join(lines[:8])[:2000]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr-root", type=Path, default=DEFAULT_PR_ROOT)
    parser.add_argument("--use-pr-python", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--tune-warmup", type=int, default=3)
    parser.add_argument("--tune-iterations", type=int, default=8)
    parser.add_argument("--batch-repeats", type=int, default=10)
    parser.add_argument("--tune-batch-repeats", type=int, default=3)
    parser.add_argument(
        "--correctness-repeats",
        type=int,
        default=5,
        help="consecutive correctness checks required per candidate",
    )
    parser.add_argument("--shape", action="append")
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="limit candidates per shape; 0 runs the full compact set",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dsv4_flydsl_pr875_results.json"),
    )
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    kernel, run_compiled = _load_pr875(
        args.pr_root.resolve(),
        args.use_pr_python,
    )
    import flydsl

    source = (
        args.pr_root / "kernels/gemm/gemm_a16w16_gfx1250.py"
    )
    selected = [
        shape
        for shape in DSV4_SHAPES
        if not args.shape or shape["name"] in set(args.shape)
    ]
    payload = {
        "suite": "dsv4",
        "backend": "flydsl_pr875",
        "pr_url": PR_URL,
        "pr_head": PR_HEAD,
        "source": str(source.resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "environment": {
            "docker_image": os.environ.get(
                "BENCH_DOCKER_IMAGE",
                "unspecified",
            ),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_hip": torch.version.hip,
            "flydsl_distribution": _package_version("flydsl"),
            "flydsl_runtime_version": getattr(
                flydsl,
                "__version__",
                "unknown",
            ),
            "flydsl_module": str(Path(flydsl.__file__).resolve()),
            "flydsl_runtime_commit": os.environ.get(
                "FLYDSL_RUNTIME_COMMIT",
                "unspecified",
            ),
            "gpu_name": torch.cuda.get_device_name(args.device),
            "gpu_properties": str(
                torch.cuda.get_device_properties(args.device)
            ),
            "rocr_visible_devices": os.environ.get(
                "ROCR_VISIBLE_DEVICES",
                "",
            ),
        },
        "settings": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "tune_warmup": args.tune_warmup,
            "tune_iterations": args.tune_iterations,
            "batch_repeats": args.batch_repeats,
            "tune_batch_repeats": args.tune_batch_repeats,
            "correctness_repeats": args.correctness_repeats,
            "selection_metric": "median batched HIP-event interval",
            "dtype": "bf16 inputs, fp32 accumulation, bf16 output",
            "input_preparation": (
                "A/B padded to each candidate's M/N tile outside timing"
            ),
            "padding_reason": (
                "PR875 TDM atoms use unbounded extents; an unpadded "
                "(32,64,7168) run caused a GPU page fault"
            ),
        },
        "results": [],
    }
    launcher_cache: dict = {}

    for shape in selected:
        m, n, k = shape["M"], shape["N"], shape["K"]
        print(f"\n=== {shape['name']} ({m}, {n}, {k}) ===", flush=True)
        torch.manual_seed(1234)
        a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
        reference = (a.float() @ b.float().T).to(torch.bfloat16)
        trials = []
        best = None

        candidates = _candidates(m, n, k)
        if args.max_candidates > 0:
            candidates = candidates[: args.max_candidates]
        for index, config in enumerate(candidates, 1):
            print(
                f"[{index}] {json.dumps(config, sort_keys=True)}",
                flush=True,
            )
            prepared = None
            try:
                prepared = PreparedPR875(
                    kernel,
                    run_compiled,
                    launcher_cache,
                    a,
                    b,
                    config,
                )
                errors = []
                for _ in range(args.correctness_repeats):
                    output = prepared()
                    torch.cuda.synchronize()
                    errors.append(_error_metrics(output, reference))
                error = {
                    "max_abs": max(item["max_abs"] for item in errors),
                    "rel_l2": max(item["rel_l2"] for item in errors),
                    "max_ref": max(item["max_ref"] for item in errors),
                    "passed": all(item["passed"] for item in errors),
                }
                if not error["passed"]:
                    trials.append(
                        {
                            "config": config,
                            "status": "incorrect",
                            "error": error,
                        }
                    )
                    continue
                tune_us, _, _ = _median_us(
                    prepared,
                    args.tune_warmup,
                    args.tune_iterations,
                    args.tune_batch_repeats,
                )
                trials.append(
                    {
                        "config": config,
                        "status": "ok",
                        "tune_e2e_us": tune_us,
                        "error": error,
                    }
                )
                print(f"    {tune_us:.3f} us", flush=True)
                if best is None or tune_us < best[0]:
                    best = (tune_us, prepared, error)
                    prepared = None
            except Exception as exc:  # noqa: BLE001
                error_text = _exception_text(exc)
                trials.append(
                    {
                        "config": config,
                        "status": "failed",
                        "error": error_text,
                    }
                )
                print(
                    f"    failed: {error_text.splitlines()[0]}",
                    flush=True,
                )
            finally:
                del prepared
                torch.cuda.empty_cache()

        row = {**shape, "candidate_trials": trials}
        if best is None:
            row["result"] = {
                "status": "failed",
                "error": "no correct candidate",
            }
        else:
            _, prepared, error = best
            e2e_us, e2e_min_us, e2e_max_us = _median_us(
                prepared,
                args.warmup,
                args.iterations,
                args.batch_repeats,
            )
            kernel_us, kernels_per_call = _kernel_median_us(
                prepared,
                args.warmup,
                args.iterations,
            )
            flops = 2.0 * m * n * k
            row["result"] = {
                "status": "ok",
                "kernel_us": kernel_us,
                "e2e_us": e2e_us,
                "e2e_min_us": e2e_min_us,
                "e2e_max_us": e2e_max_us,
                "kernels_per_call": kernels_per_call,
                "tflops": flops / kernel_us / 1.0e6,
                "e2e_tflops": flops / e2e_us / 1.0e6,
                "effective_bandwidth_gbps": (
                    _effective_bandwidth_gbps(m, n, k, kernel_us)
                ),
                "e2e_effective_bandwidth_gbps": (
                    _effective_bandwidth_gbps(m, n, k, e2e_us)
                ),
                "estimated_bytes_moved": _estimated_bytes_moved(
                    m,
                    n,
                    k,
                ),
                "config": prepared.config,
                "error": error,
            }
            print(json.dumps(row["result"], indent=2), flush=True)

        payload["results"].append(row)
        _write_payload(args.output, payload)
        del a, b, reference
        torch.cuda.empty_cache()

    print(f"\nwrote {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
