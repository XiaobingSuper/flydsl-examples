#!/usr/bin/env python3

import pytest

import flydsl  # noqa: F401 -- initialize FlyDSL/COMGR before torch
import torch
from flydsl.runtime.device import get_rocm_arch

from kernels import gemm_a16w16_gfx1250 as gemm_module


gemm_a16w16 = gemm_module.gemm_a16w16
_gemm_all_compute = gemm_module._gemm_all_compute

_requires_gfx1250 = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not str(get_rocm_arch() or "").startswith("gfx1250"),
    reason="gfx1250 GPU is required",
)


def test_route_policy():
    cases = [
        ((32, 64, 7168), torch.bfloat16, "all_compute_1w"),
        ((32, 64, 7168), torch.float16, "all_compute_1w"),
        ((128, 2048, 4096), torch.bfloat16, "all_compute_8w"),
        ((4096, 256, 64), torch.bfloat16, "direct_b"),
        ((4096, 256, 64), torch.float16, "all_compute_8w"),
        ((128, 257, 320), torch.bfloat16, "all_compute_4w"),
        ((512, 257, 320), torch.bfloat16, "all_compute_8w"),
    ]
    for shape, dtype, expected in cases:
        route, _ = gemm_module._select_all_compute_config(*shape, dtype)
        assert route == expected


def test_only_matching_half_output_supported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP device required for tensor validation")
    a = torch.empty((16, 32), dtype=torch.float16, device="cuda")
    b = torch.empty((16, 32), dtype=torch.float16, device="cuda")
    with pytest.raises(ValueError, match="matching FP16"):
        gemm_a16w16(a, b, out_dtype=torch.float32)


@_requires_gfx1250
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_aligned_fp16_bf16(dtype):
    torch.manual_seed(0)
    a = torch.randn((128, 128), device="cuda", dtype=dtype)
    b = torch.randn((128, 128), device="cuda", dtype=dtype)
    actual = gemm_a16w16(a, b)
    expected = (a.float() @ b.float().T).to(dtype)
    torch.testing.assert_close(actual, expected, atol=0.2, rtol=0.02)


@_requires_gfx1250
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_one_wave_small_m(dtype):
    torch.manual_seed(1)
    a = torch.randn((32, 7168), device="cuda", dtype=dtype)
    b = torch.randn((64, 7168), device="cuda", dtype=dtype)
    actual = gemm_a16w16(a, b)
    expected = (a.float() @ b.float().T).to(dtype)
    torch.testing.assert_close(actual, expected, atol=0.2, rtol=0.02)


@_requires_gfx1250
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "a_transposed,b_transposed",
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_padding_and_noncontiguous_inputs(dtype, a_transposed, b_transposed):
    torch.manual_seed(2)
    m, n, k = 130, 193, 70
    a = torch.randn(
        (k, m) if a_transposed else (m, k), device="cuda", dtype=dtype
    )
    b = torch.randn(
        (k, n) if b_transposed else (n, k), device="cuda", dtype=dtype
    )
    if a_transposed:
        a = a.T
    if b_transposed:
        b = b.T
    actual = gemm_a16w16(a, b)
    expected = (a.float() @ b.float().T).to(dtype)
    assert actual.shape == (m, n)
    torch.testing.assert_close(actual, expected, atol=0.2, rtol=0.02)


@_requires_gfx1250
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_preallocated_output(dtype):
    torch.manual_seed(3)
    a = torch.randn((128, 128), device="cuda", dtype=dtype)
    b = torch.randn((128, 128), device="cuda", dtype=dtype)
    out = torch.empty((128, 128), device="cuda", dtype=dtype)
    actual = gemm_a16w16(a, b, out=out)
    assert actual.data_ptr() == out.data_ptr()
    torch.testing.assert_close(
        actual, (a.float() @ b.float().T).to(dtype), atol=0.2, rtol=0.02
    )


@_requires_gfx1250
def test_bf16_direct_b_path():
    torch.manual_seed(4)
    a = torch.randn((4096, 64), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((256, 64), device="cuda", dtype=torch.bfloat16)
    actual = gemm_a16w16(a, b)
    expected = (a.float() @ b.float().T).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected, atol=0.2, rtol=0.02)


@_requires_gfx1250
def test_output_guard_with_padded_all_compute():
    torch.manual_seed(5)
    m, n, k = 65, 128, 256
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    parent = torch.full(
        (m + 64, n), 777.0, device="cuda", dtype=torch.bfloat16
    )
    out = parent[:m]
    actual = gemm_a16w16(a, b, out=out)
    torch.testing.assert_close(
        actual, (a.float() @ b.float().T).to(torch.bfloat16), atol=0.2, rtol=0.02
    )
    assert torch.all(parent[m:] == 777.0)


@_requires_gfx1250
@pytest.mark.parametrize("waves_m,waves_n", [(1, 1), (1, 2), (2, 2), (4, 2)])
def test_all_supported_wave_counts(waves_m, waves_n):
    torch.manual_seed(6)
    a = torch.randn((64, 128), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((64, 128), device="cuda", dtype=torch.bfloat16)
    actual = _gemm_all_compute(
        a,
        b,
        reg_m=1,
        reg_n=1,
        reg_k=2,
        waves_m=waves_m,
        waves_n=waves_n,
        num_buffers=2,
    )
    expected = (a.float() @ b.float().T).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected, atol=0.2, rtol=0.02)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
