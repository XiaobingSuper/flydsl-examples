#!/usr/bin/env python3

import pytest

import flydsl  # noqa: F401 -- load FlyDSL/COMGR before torch loads HIP LLVM
import torch
from flydsl.runtime.device import get_rocm_arch

from kernels import gemm_a16w16_gfx1250 as gemm_module

gemm_a16w16 = gemm_module.gemm_a16w16
_gemm_producer_consumer = gemm_module._gemm_producer_consumer


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not str(get_rocm_arch() or "").startswith("gfx1250"),
    reason="gfx1250 GPU is required",
)


def test_auto_direct_b_path():
    torch.manual_seed(7)
    a = torch.randn((4096, 64), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((256, 64), device="cuda", dtype=torch.bfloat16)

    actual = gemm_a16w16(a, b)
    expected = a @ b.T

    torch.testing.assert_close(actual, expected, atol=0.2, rtol=0.02)


@pytest.mark.parametrize(
    "in_dtype,out_dtype",
    [
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
        (torch.float16, torch.float32),
    ],
)
def test_aligned_tile(in_dtype, out_dtype):
    torch.manual_seed(0)
    a = torch.randn((128, 128), device="cuda", dtype=in_dtype)
    b = torch.randn((128, 128), device="cuda", dtype=in_dtype)

    actual = gemm_a16w16(a, b, out_dtype=out_dtype)
    expected = (a.float() @ b.float().T).to(out_dtype)

    torch.testing.assert_close(actual, expected, atol=0.2, rtol=0.02)


@pytest.mark.parametrize("in_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "a_transposed,b_transposed",
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_padding_and_noncontiguous_inputs(
    in_dtype,
    a_transposed,
    b_transposed,
):
    torch.manual_seed(1)
    m, n, k = 130, 193, 70
    a = torch.randn(
        (k, m) if a_transposed else (m, k),
        device="cuda",
        dtype=in_dtype,
    )
    b = torch.randn(
        (k, n) if b_transposed else (n, k),
        device="cuda",
        dtype=in_dtype,
    )
    if a_transposed:
        a = a.T
        assert not a.is_contiguous()
    if b_transposed:
        b = b.T
        assert not b.is_contiguous()

    actual = gemm_a16w16(a, b)
    expected = a @ b.T

    assert actual.shape == (m, n)
    torch.testing.assert_close(actual, expected, atol=0.2, rtol=0.02)


def test_runtime_m_output_guard():
    torch.manual_seed(8)
    m, n, k = 65, 128, 256
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    sentinel = 777.0
    parent = torch.full(
        (m + 64, n),
        sentinel,
        device="cuda",
        dtype=torch.bfloat16,
    )
    out = parent[:m]

    actual = _gemm_producer_consumer(
        a,
        b,
        out=out,
        reg_m=2,
        reg_n=4,
        reg_k=4,
        waves_m=2,
        waves_n=1,
    )

    torch.testing.assert_close(actual, a @ b.T, atol=0.2, rtol=0.02)
    assert torch.all(parent[m:] == sentinel)


def test_runtime_m_reuses_compiled_module():
    gemm_module._cached_module.cache_clear()
    common = {
        "reg_m": 2,
        "reg_n": 4,
        "reg_k": 4,
        "waves_m": 2,
        "waves_n": 1,
    }
    b = torch.randn((128, 256), device="cuda", dtype=torch.bfloat16)

    a0 = torch.randn((65, 256), device="cuda", dtype=torch.bfloat16)
    out0 = _gemm_producer_consumer(a0, b, **common)
    info0 = gemm_module._cached_module.cache_info()
    a1 = torch.randn((97, 256), device="cuda", dtype=torch.bfloat16)
    out1 = _gemm_producer_consumer(a1, b, **common)
    info1 = gemm_module._cached_module.cache_info()

    torch.testing.assert_close(out0, a0 @ b.T, atol=0.2, rtol=0.02)
    torch.testing.assert_close(out1, a1 @ b.T, atol=0.2, rtol=0.02)
    assert info1.misses == info0.misses
    assert info1.hits == info0.hits + 1


def test_compiler_tuning_options():
    torch.manual_seed(9)
    a = torch.randn((128, 512), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((128, 512), device="cuda", dtype=torch.bfloat16)

    actual = _gemm_producer_consumer(
        a,
        b,
        waves_per_eu=2,
        kernarg_preload=True,
        sched_strategy="max-ilp",
        main_loop_unroll=True,
    )

    torch.testing.assert_close(actual, a @ b.T, atol=0.2, rtol=0.02)


def test_preallocated_output():
    torch.manual_seed(2)
    a = torch.randn((128, 128), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((128, 128), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((128, 128), device="cuda", dtype=torch.bfloat16)

    actual = gemm_a16w16(a, b, out=out)

    assert actual.data_ptr() == out.data_ptr()
    torch.testing.assert_close(actual, a @ b.T, atol=0.2, rtol=0.02)


@pytest.mark.parametrize("num_stages", [2, 3])
def test_pipeline_stages(num_stages):
    torch.manual_seed(4)
    a = torch.randn((128, 512), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((128, 512), device="cuda", dtype=torch.bfloat16)

    actual = _gemm_producer_consumer(a, b, num_stages=num_stages)

    torch.testing.assert_close(actual, a @ b.T, atol=0.2, rtol=0.02)


@pytest.mark.parametrize("m,n", [(16, 257), (257, 16), (32, 32)])
def test_small_dimension_policy(m, n):
    torch.manual_seed(3)
    a = torch.randn((m, 320), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, 320), device="cuda", dtype=torch.bfloat16)

    actual = gemm_a16w16(a, b)

    torch.testing.assert_close(actual, a @ b.T, atol=0.2, rtol=0.02)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
