#!/usr/bin/env python3
"""Tune and compare local and Aiter PR #2725 A16W16 GEMMs on gfx1250.

The comparison intentionally imports the PR wrapper directly instead of using
``aiter.tuned_gemm``: PR #2725 currently has no ``flydsl_gfx1250`` solMap
registration. Both backends must run in the same process and FlyDSL runtime.

Example container environment:

    PYTHONPATH=/pr:/local python3 /local/benchmark_a16w16_implementations.py \
      --shapes '32,64,7168;512,2048,7168;4096,2048,4096' \
      --output /out/a16w16_implementation_tuning.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

import flydsl  # noqa: F401 -- initialize FlyDSL/COMGR before torch
import torch

import kernels.gemm_a16w16_gfx1250 as local_gemm


DEFAULT_SHAPES = [
    (32, 64, 7168),
    (512, 2048, 7168),
    (4096, 2048, 4096),
]

PR_GEOMETRIES = [
    # tile_m, tile_n, tile_k, m_warp, n_warp
    (16, 16, 256, 1, 1),
    (16, 64, 256, 1, 2),
    (32, 32, 128, 2, 2),
    (32, 64, 128, 2, 2),
    (64, 64, 128, 2, 2),
    (64, 64, 256, 4, 2),
    (64, 128, 128, 2, 4),
    (128, 128, 128, 2, 4),
    (128, 128, 128, 4, 2),
    (128, 256, 128, 4, 4),
]
pr2725_gemm = None


def _load_pr2725_wrapper(pr_root: Path):
    """Load the PR files directly while using the environment's tensor_shim."""
    kernel_name = "aiter.ops.flydsl.kernels.gemm_a16w16_kernel_gfx1250"
    kernel_path = (
        pr_root
        / "aiter"
        / "ops"
        / "flydsl"
        / "kernels"
        / "gemm_a16w16_kernel_gfx1250.py"
    )
    wrapper_path = (
        pr_root / "aiter" / "ops" / "flydsl" / "gemm_a16w16_gfx1250.py"
    )
    for path in (kernel_path, wrapper_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    kernel_spec = importlib.util.spec_from_file_location(kernel_name, kernel_path)
    kernel_module = importlib.util.module_from_spec(kernel_spec)
    sys.modules[kernel_name] = kernel_module
    kernel_spec.loader.exec_module(kernel_module)

    wrapper_spec = importlib.util.spec_from_file_location(
        "_pr2725_gemm_a16w16_wrapper", wrapper_path
    )
    wrapper_module = importlib.util.module_from_spec(wrapper_spec)
    wrapper_spec.loader.exec_module(wrapper_module)
    return wrapper_module.gemm_a16w16


def _parse_shapes(text: str) -> list[tuple[int, int, int]]:
    if not text:
        return list(DEFAULT_SHAPES)
    shapes = []
    for entry in text.split(";"):
        values = tuple(int(value) for value in entry.split(","))
        if len(values) != 3 or min(values) <= 0:
            raise ValueError(f"invalid shape {entry!r}; expected M,N,K")
        shapes.append(values)
    return shapes


def _median_us(fn, warmup: int, iterations: int, batch_repeats: int):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(batch_repeats):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / batch_repeats)
    return {
        "median_us": statistics.median(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "samples_us": samples,
    }


def _error_metrics(actual: torch.Tensor, reference: torch.Tensor):
    actual_f = actual.float()
    reference_f = reference.float()
    diff = (actual_f - reference_f).abs()
    max_abs = diff.max().item()
    denom = torch.linalg.vector_norm(reference_f).item()
    rel_l2 = torch.linalg.vector_norm(actual_f - reference_f).item() / max(
        denom, 1.0e-12
    )
    passed = True
    message = ""
    try:
        torch.testing.assert_close(actual, reference, atol=0.2, rtol=0.02)
    except AssertionError as exc:
        passed = False
        message = str(exc)
    return {
        "passed": passed,
        "max_abs": max_abs,
        "rel_l2": rel_l2,
        "message": message,
    }


def _dedupe(candidates: list[dict]) -> list[dict]:
    seen = set()
    result = []
    for candidate in candidates:
        key = json.dumps(candidate, sort_keys=True)
        if key not in seen:
            seen.add(key)
            result.append(candidate)
    return result


def _local_candidates(m: int, n: int, k: int) -> list[dict]:
    candidates = [{"route": "production"}]
    if m <= 64:
        geometries = [
            (1, 1, 4, 1, 1),
            (1, 1, 8, 1, 1),
            (1, 2, 4, 1, 1),
            (1, 1, 4, 1, 2),
            (1, 2, 4, 1, 2),
            (1, 2, 4, 2, 1),
        ]
    elif m <= 512:
        geometries = [
            (1, 2, 4, 2, 2),
            (2, 2, 4, 2, 2),
            (2, 4, 4, 2, 2),
            (2, 1, 4, 4, 2),
            (2, 2, 4, 4, 2),
            (2, 4, 4, 4, 2),
        ]
    else:
        geometries = [
            (2, 2, 4, 2, 2),
            (2, 4, 4, 2, 2),
            (2, 1, 4, 4, 2),
            (2, 2, 4, 4, 2),
            (2, 4, 4, 4, 2),
            (2, 4, 2, 4, 2),
        ]
    for reg_m, reg_n, reg_k, waves_m, waves_n in geometries:
        for buffers in (2, 3, 4):
            candidates.append(
                {
                    "route": "all_compute",
                    "reg_m": reg_m,
                    "reg_n": reg_n,
                    "reg_k": reg_k,
                    "waves_m": waves_m,
                    "waves_n": waves_n,
                    "num_buffers": buffers,
                    "swizzle_m": 32,
                    "direct_b_global": False,
                    "use_xcd_remap": False,
                }
            )
            if k >= 1024:
                candidates.append(
                    {
                        **candidates[-1],
                        "main_loop_unroll": True,
                    }
                )
    if n % 256 == 0 and k % 64 == 0:
        for buffers in (2, 3):
            for swizzle in (8, 16, 32):
                candidates.append(
                    {
                        "route": "all_compute",
                        "reg_m": 8,
                        "reg_n": 4,
                        "reg_k": 2,
                        "waves_m": 1,
                        "waves_n": 4,
                        "num_buffers": buffers,
                        "swizzle_m": swizzle,
                        "direct_b_global": True,
                        "use_xcd_remap": True,
                    }
                )
    return _dedupe(candidates)


def _legal_pr_split(k: int, tile_k: int, split_k: int, buffers: int) -> bool:
    padded_k = math.ceil(k / tile_k) * tile_k
    tiles = padded_k // tile_k
    return tiles % split_k == 0 and tiles // split_k >= buffers - 1


def _pr_phase1_candidates(m: int, n: int, k: int) -> list[dict]:
    candidates = []
    splits = (1, 2, 4, 6, 8, 16) if m <= 64 else (1, 2, 4) if m <= 256 else (1,)
    for tm, tn, tk, mw, nw in PR_GEOMETRIES:
        if tm % (16 * mw) or tn % (16 * nw):
            continue
        # Baseline both pipeline variants.
        candidates += [
            {
                "tile_m": tm,
                "tile_n": tn,
                "tile_k": tk,
                "m_warp": mw,
                "n_warp": nw,
                "num_buffers": 3,
                "split_k": 1,
                "variant": "bandwidth_bound",
            },
            {
                "tile_m": tm,
                "tile_n": tn,
                "tile_k": tk,
                "m_warp": mw,
                "n_warp": nw,
                "num_buffers": 2,
                "split_k": 1,
                "variant": "compute_bound",
            },
        ]
        for split in splits:
            if split == 1:
                continue
            for buffers in (2, 3):
                if _legal_pr_split(k, tk, split, buffers):
                    candidates.append(
                        {
                            "tile_m": tm,
                            "tile_n": tn,
                            "tile_k": tk,
                            "m_warp": mw,
                            "n_warp": nw,
                            "num_buffers": buffers,
                            "split_k": split,
                            "variant": "bandwidth_bound",
                        }
                    )
    return _dedupe(candidates)


def _expand_pr_candidates(top: list[dict], k: int) -> list[dict]:
    expanded = []
    for base in top:
        expanded.append(deepcopy(base))
        for waves_per_eu in (1, 2, 4):
            item = deepcopy(base)
            item["waves_per_eu"] = waves_per_eu
            expanded.append(item)
        for strategy in ("max-ilp", "max-memory-clause"):
            item = deepcopy(base)
            item["sched_strategy"] = strategy
            expanded.append(item)
        item = deepcopy(base)
        item["kernarg_preload"] = True
        expanded.append(item)
        if k >= 1024:
            item = deepcopy(base)
            item["main_loop_unroll"] = True
            expanded.append(item)
    return _dedupe(expanded)


def _make_local_fn(a, b, out, config):
    route = config["route"]
    kwargs = {key: value for key, value in config.items() if key != "route"}
    if route == "production":
        return lambda: local_gemm.gemm_a16w16(a, b, out=out)
    if route == "all_compute":
        return lambda: local_gemm._gemm_all_compute(a, b, out=out, **kwargs)
    raise ValueError(f"unknown local route: {route}")


def _make_pr_fn(a, b, out, config):
    if pr2725_gemm is None:
        raise RuntimeError("PR #2725 wrapper is not loaded")
    return lambda: pr2725_gemm(
        a,
        b,
        dtype=torch.bfloat16,
        y=out,
        **config,
    )


def _evaluate_candidates(
    backend,
    candidates,
    a,
    b,
    out,
    reference,
    args,
    *,
    phase,
):
    records = []
    make_fn = _make_local_fn if backend == "local" else _make_pr_fn
    for index, config in enumerate(candidates):
        started = time.time()
        record = {"config": config, "phase": phase}
        try:
            fn = make_fn(a, b, out, config)
            errors = []
            for _ in range(args.correctness_repeats):
                actual = fn()
                torch.cuda.synchronize()
                errors.append(_error_metrics(actual, reference))
            record["error"] = {
                "passed": all(item["passed"] for item in errors),
                "max_abs": max(item["max_abs"] for item in errors),
                "rel_l2": max(item["rel_l2"] for item in errors),
            }
            if record["error"]["passed"]:
                timing = _median_us(
                    fn,
                    args.tune_warmup,
                    args.tune_iterations,
                    args.tune_batch_repeats,
                )
                record["tune_e2e_us"] = timing["median_us"]
                record["tune_min_us"] = timing["min_us"]
                record["tune_max_us"] = timing["max_us"]
                record["status"] = "ok"
            else:
                record["status"] = "incorrect"
        except Exception as exc:  # noqa: BLE001 -- tuner records bad candidates
            record["status"] = "failed"
            record["error_message"] = f"{type(exc).__name__}: {exc}"
            try:
                torch.cuda.synchronize()
            except Exception:  # noqa: BLE001
                pass
            torch.cuda.empty_cache()
        record["wall_seconds"] = time.time() - started
        records.append(record)
        print(
            f"[{backend} {phase} {index + 1}/{len(candidates)}] "
            f"{record['status']} {record.get('tune_e2e_us')} {config}",
            flush=True,
        )
    return records


def _valid_sorted(records):
    return sorted(
        (record for record in records if record.get("status") == "ok"),
        key=lambda record: record["tune_e2e_us"],
    )


def _final_measure(backend, config, a, b, out, reference, args):
    fn = (
        _make_local_fn(a, b, out, config)
        if backend == "local"
        else _make_pr_fn(a, b, out, config)
    )
    actual = fn()
    torch.cuda.synchronize()
    error = _error_metrics(actual, reference)
    if not error["passed"]:
        raise RuntimeError(f"final candidate became incorrect: {error}")
    timing = _median_us(
        fn,
        args.final_warmup,
        args.final_iterations,
        args.final_batch_repeats,
    )
    flops = 2.0 * a.shape[0] * b.shape[0] * a.shape[1]
    return {
        "config": config,
        "error": error,
        "e2e_us": timing["median_us"],
        "min_us": timing["min_us"],
        "max_us": timing["max_us"],
        "tflops": flops / timing["median_us"] / 1.0e6,
    }


def main():
    global pr2725_gemm

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shapes",
        default=";".join(",".join(str(v) for v in shape) for shape in DEFAULT_SHAPES),
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--correctness-repeats", type=int, default=2)
    parser.add_argument("--tune-warmup", type=int, default=3)
    parser.add_argument("--tune-iterations", type=int, default=8)
    parser.add_argument("--tune-batch-repeats", type=int, default=5)
    parser.add_argument("--pr-finalists", type=int, default=3)
    parser.add_argument(
        "--max-local-screen",
        type=int,
        default=0,
        help="0 keeps all local screening candidates.",
    )
    parser.add_argument(
        "--max-pr-screen",
        type=int,
        default=0,
        help="0 keeps all PR geometry/split-K screening candidates.",
    )
    parser.add_argument("--final-warmup", type=int, default=20)
    parser.add_argument("--final-iterations", type=int, default=50)
    parser.add_argument("--final-batch-repeats", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("a16w16_implementation_tuning.json"),
    )
    parser.add_argument(
        "--replay-json",
        type=Path,
        help="Skip tuning and remeasure final configs from an earlier result.",
    )
    parser.add_argument(
        "--pr-root",
        type=Path,
        default=Path(os.environ["PR2725_ROOT"])
        if "PR2725_ROOT" in os.environ
        else None,
        help="PR #2725 source root; may also be set through PR2725_ROOT.",
    )
    parser.add_argument(
        "--expected-flydsl",
        default="0.3.0",
        help="Empty string disables the version check.",
    )
    args = parser.parse_args()

    if args.expected_flydsl and version("flydsl") != args.expected_flydsl:
        raise RuntimeError(
            f"expected flydsl {args.expected_flydsl}, got {version('flydsl')}"
        )
    torch.cuda.set_device(args.device)
    arch = getattr(torch.cuda.get_device_properties(args.device), "gcnArchName", "")
    if not arch.startswith("gfx1250"):
        raise RuntimeError(f"requires gfx1250, got {arch!r}")
    if args.pr_root is None:
        parser.error("--pr-root or PR2725_ROOT is required")
    pr2725_gemm = _load_pr2725_wrapper(args.pr_root)

    if args.replay_json is not None:
        source = json.loads(args.replay_json.read_text())
        replay = {
            "environment": {
                "torch": torch.__version__,
                "flydsl": version("flydsl"),
                "gpu": torch.cuda.get_device_name(args.device),
                "arch": arch,
            },
            "source": str(args.replay_json),
            "results": [],
        }
        for prior in source["results"]:
            m, n, k = prior["M"], prior["N"], prior["K"]
            torch.manual_seed(1234)
            a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
            b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
            reference = (a.float() @ b.float().T).to(torch.bfloat16)
            out_local = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
            out_pr = torch.empty_like(out_local)
            measured = {}
            for backend, out in (("local", out_local), ("pr2725", out_pr)):
                measured[backend] = _final_measure(
                    backend,
                    prior[backend]["final"]["config"],
                    a,
                    b,
                    out,
                    reference,
                    args,
                )
            measured.update({"M": m, "N": n, "K": k})
            measured["winner"] = (
                "local"
                if measured["local"]["e2e_us"] < measured["pr2725"]["e2e_us"]
                else "pr2725"
            )
            replay["results"].append(measured)
            print(
                f"REPLAY {(m, n, k)} {measured['winner']} "
                f"local={measured['local']['e2e_us']:.3f} "
                f"pr={measured['pr2725']['e2e_us']:.3f}",
                flush=True,
            )
        args.output.write_text(json.dumps(replay, indent=2) + "\n")
        print(f"wrote {args.output.resolve()}", flush=True)
        return

    payload = {
        "environment": {
            "torch": torch.__version__,
            "flydsl": version("flydsl"),
            "gpu": torch.cuda.get_device_name(args.device),
            "arch": arch,
            "timing": "HIP events around complete wrapper calls; compile excluded by warmup",
        },
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "results": [],
    }

    for m, n, k in _parse_shapes(args.shapes):
        print(f"\n===== shape {(m, n, k)} =====", flush=True)
        torch.manual_seed(1234)
        a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
        out_local = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        out_pr = torch.empty_like(out_local)
        reference = (a.float() @ b.float().T).to(torch.bfloat16)

        local_candidates = _local_candidates(m, n, k)
        if args.max_local_screen > 0:
            local_candidates = local_candidates[: args.max_local_screen]
        local_records = _evaluate_candidates(
            "local",
            local_candidates,
            a,
            b,
            out_local,
            reference,
            args,
            phase="screen",
        )
        local_valid = _valid_sorted(local_records)
        if not local_valid:
            raise RuntimeError(f"no valid local candidate for {(m, n, k)}")

        pr_phase1 = _pr_phase1_candidates(m, n, k)
        if args.max_pr_screen > 0:
            pr_phase1 = pr_phase1[: args.max_pr_screen]
        pr_records_1 = _evaluate_candidates(
            "pr2725",
            pr_phase1,
            a,
            b,
            out_pr,
            reference,
            args,
            phase="geometry",
        )
        pr_valid_1 = _valid_sorted(pr_records_1)
        if not pr_valid_1:
            raise RuntimeError(f"no valid PR candidate for {(m, n, k)}")
        pr_expanded = _expand_pr_candidates(
            [record["config"] for record in pr_valid_1[: args.pr_finalists]],
            k,
        )
        screened_keys = {
            json.dumps(record["config"], sort_keys=True) for record in pr_records_1
        }
        pr_expanded = [
            config
            for config in pr_expanded
            if json.dumps(config, sort_keys=True) not in screened_keys
        ]
        pr_records_2 = _evaluate_candidates(
            "pr2725",
            pr_expanded,
            a,
            b,
            out_pr,
            reference,
            args,
            phase="compiler",
        )
        pr_valid = _valid_sorted(pr_records_1 + pr_records_2)

        local_final = _final_measure(
            "local",
            local_valid[0]["config"],
            a,
            b,
            out_local,
            reference,
            args,
        )
        pr_final = _final_measure(
            "pr2725",
            pr_valid[0]["config"],
            a,
            b,
            out_pr,
            reference,
            args,
        )
        winner = "local" if local_final["e2e_us"] < pr_final["e2e_us"] else "pr2725"
        shape_result = {
            "M": m,
            "N": n,
            "K": k,
            "winner": winner,
            "local": {
                "final": local_final,
                "trials": local_records,
            },
            "pr2725": {
                "final": pr_final,
                "trials": pr_records_1 + pr_records_2,
            },
        }
        payload["results"].append(shape_result)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(
            f"WINNER {winner}: local={local_final['e2e_us']:.3f} us "
            f"PR={pr_final['e2e_us']:.3f} us",
            flush=True,
        )

    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
