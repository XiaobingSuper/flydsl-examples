#!/usr/bin/env python3
"""Production FP16/BF16 A16W16 GEMM entry for gfx1250.

Production dispatch is intentionally converged onto the role-fused all-compute
family. The former producer/consumer implementation is preserved in
``gemm_a16w16_gfx1250_producer_consumer_reference.py`` for study only.
"""

from .gemm_a16w16_gfx1250_all_compute import (
    _BF16_EXACT_CONFIGS,
    _DIRECT_B_CONFIG,
    _cached_all_compute_module,
    _create_all_compute_module,
    _gemm_all_compute,
    _run_compiled,
    _select_all_compute_config,
    gemm_a16w16,
)

__all__ = [
    "gemm_a16w16",
    "_gemm_all_compute",
    "_select_all_compute_config",
]
