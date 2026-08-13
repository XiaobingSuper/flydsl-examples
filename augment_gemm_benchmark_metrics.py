#!/usr/bin/env python3
"""Add derived TFLOPS and effective-bandwidth metrics to benchmark JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output")
    args = parser.parse_args()

    source = Path(args.input)
    destination = Path(args.output) if args.output else source
    payload = json.loads(source.read_text())
    payload["derived_metrics"] = {
        "flops": "2*M*N*K",
        "estimated_bytes_moved": "2*(M*K + N*K + M*N)",
        "tflops": "flops / time_us / 1e6",
        "effective_bandwidth_gbps": (
            "estimated_bytes_moved / time_us / 1e3"
        ),
        "bandwidth_note": (
            "Algorithmic lower-bound effective bandwidth, not measured HBM "
            "traffic; excludes cache and split-K workspace effects."
        ),
    }

    for row in payload["results"]:
        m, n, k = row["M"], row["N"], row["K"]
        flops = 2.0 * m * n * k
        bytes_moved = 2.0 * (m * k + n * k + m * n)
        row["estimated_bytes_moved"] = bytes_moved
        for backend in row["backends"].values():
            if backend.get("status") not in ("ok", "incorrect"):
                continue
            kernel_us = backend["kernel_us"]
            e2e_us = backend["e2e_us"]
            backend["tflops"] = flops / kernel_us / 1.0e6
            backend["e2e_tflops"] = flops / e2e_us / 1.0e6
            backend["effective_bandwidth_gbps"] = (
                bytes_moved / kernel_us / 1.0e3
            )
            backend["e2e_effective_bandwidth_gbps"] = (
                bytes_moved / e2e_us / 1.0e3
            )

    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(destination)


if __name__ == "__main__":
    main()
