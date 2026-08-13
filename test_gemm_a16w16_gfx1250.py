#!/usr/bin/env python3

import pytest

import flydsl  # noqa: F401 -- load FlyDSL/COMGR before torch loads HIP LLVM
import torch
from flydsl.runtime.device import get_rocm_arch

from kernels.gemm_a16w16_gfx1250 import gemm_a16w16
from kernels.gemm_a16w16_gfx1250_all_compute import (
    gemm_a16w16_all_compute,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not str(get_rocm_arch() or "").startswith("gfx1250"),
    reason="gfx1250 GPU is required",
)


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
def test_padding_and_noncontiguous_inputs(in_dtype):
    torch.manual_seed(1)
    m, n, k = 130, 193, 70
    a = torch.randn((k, m), device="cuda", dtype=in_dtype).T
    b = torch.randn((k, n), device="cuda", dtype=in_dtype).T
    assert not a.is_contiguous()
    assert not b.is_contiguous()

    actual = gemm_a16w16(a, b)
    expected = a @ b.T

    assert actual.shape == (m, n)
    torch.testing.assert_close(actual, expected, atol=0.2, rtol=0.02)


def test_preallocated_output():
    torch.manual_seed(2)
    a = torch.randn((128, 128), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((128, 128), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((128, 128), device="cuda", dtype=torch.bfloat16)

    actual = gemm_a16w16(a, b, out=out)

    assert actual.data_ptr() == out.data_ptr()
    torch.testing.assert_close(actual, a @ b.T, atol=0.2, rtol=0.02)


@pytest.mark.parametrize("num_stages", [2, 3, 4])
def test_pipeline_stages(num_stages):
    torch.manual_seed(4)
    a = torch.randn((128, 512), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((128, 512), device="cuda", dtype=torch.bfloat16)

    actual = gemm_a16w16(a, b, num_stages=num_stages)

    torch.testing.assert_close(actual, a @ b.T, atol=0.2, rtol=0.02)


@pytest.mark.parametrize("cluster_m,cluster_n", [(1, 2), (2, 1), (2, 2)])
def test_cluster_multicast(cluster_m, cluster_n):
    torch.manual_seed(5)
    a = torch.randn((512, 512), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((512, 512), device="cuda", dtype=torch.bfloat16)

    actual = gemm_a16w16(
        a,
        b,
        cluster_m=cluster_m,
        cluster_n=cluster_n,
    )

    torch.testing.assert_close(actual, a @ b.T, atol=0.2, rtol=0.02)


@pytest.mark.parametrize("grouped_inline", [False, True])
def test_all_compute_prototype(grouped_inline):
    torch.manual_seed(6)
    a = torch.randn((512, 512), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((512, 512), device="cuda", dtype=torch.bfloat16)

    actual = gemm_a16w16_all_compute(
        a,
        b,
        diagonal_traversal=grouped_inline,
    )

    torch.testing.assert_close(actual, a @ b.T, atol=0.2, rtol=0.02)


def test_grouped_inline_16_atom():
    torch.manual_seed(7)
    a = torch.randn((512, 512), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((512, 512), device="cuda", dtype=torch.bfloat16)

    actual = gemm_a16w16(
        a,
        b,
        reg_m=2,
        reg_n=4,
        reg_k=4,
        waves_m=2,
        waves_n=1,
        grouped_inline=True,
    )

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
