"""
Triton implementation of radix-7 FFT codelet.

This module provides forward and inverse radix-7 FFT butterfly operations.
Ported from CUDA radix_7.h

Optimized: vectorized BLOCK_SIZE groups per block,
zero-copy wrapper via view_as_real/view_as_complex, autotuned.
"""

import math

import torch
import triton
import triton.language as tl


# ======================== Kernels ========================
# Each block processes BLOCK_SIZE groups of 7 complex numbers.
# Data layout: interleaved float32 [r0,i0,r1,i1,...,r6,i6] per group (14 floats).

_CONFIGS = [
    triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
    triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
    triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
    triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
]


@triton.autotune(configs=_CONFIGS, key=['n_groups'])
@triton.jit
def _batch_rad7_fwd_kernel(
    x_ptr,
    out_ptr,
    n_groups,
    BLOCK_SIZE: tl.constexpr,
):
    C7Q1 = -1.1666666666666665
    C7Q2 = 0.7901564685254002
    C7Q3 = 0.05585426728964774
    C7Q4 = 0.7343022012357524
    C7Q5 = 0.4409585518440984
    C7Q6 = 0.3408729306239314
    C7Q7 = -0.5339693603377252
    C7Q8 = 0.8748422909616567

    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_groups
    base = offs * 14

    R0r = tl.load(x_ptr + base + 0,  mask=mask, other=0.0)
    R0i = tl.load(x_ptr + base + 1,  mask=mask, other=0.0)
    R1r = tl.load(x_ptr + base + 2,  mask=mask, other=0.0)
    R1i = tl.load(x_ptr + base + 3,  mask=mask, other=0.0)
    R2r = tl.load(x_ptr + base + 4,  mask=mask, other=0.0)
    R2i = tl.load(x_ptr + base + 5,  mask=mask, other=0.0)
    R3r = tl.load(x_ptr + base + 6,  mask=mask, other=0.0)
    R3i = tl.load(x_ptr + base + 7,  mask=mask, other=0.0)
    R4r = tl.load(x_ptr + base + 8,  mask=mask, other=0.0)
    R4i = tl.load(x_ptr + base + 9,  mask=mask, other=0.0)
    R5r = tl.load(x_ptr + base + 10, mask=mask, other=0.0)
    R5i = tl.load(x_ptr + base + 11, mask=mask, other=0.0)
    R6r = tl.load(x_ptr + base + 12, mask=mask, other=0.0)
    R6i = tl.load(x_ptr + base + 13, mask=mask, other=0.0)

    # Forward butterfly
    p0r = R1r + R6r;  p0i = R1i + R6i
    p1r = R1r - R6r;  p1i = R1i - R6i
    p2r = R2r + R5r;  p2i = R2i + R5i
    p3r = R2r - R5r;  p3i = R2i - R5i
    p4r = R4r + R3r;  p4i = R4i + R3i
    p5r = R4r - R3r;  p5i = R4i - R3i

    p6r = p2r + p0r;  p6i = p2i + p0i
    q4r = p2r - p0r;  q4i = p2i - p0i
    q2r = p0r - p4r;  q2i = p0i - p4i
    q3r = p4r - p2r;  q3i = p4i - p2i
    p7r = p5r + p3r;  p7i = p5i + p3i
    q7r = p5r - p3r;  q7i = p5i - p3i
    q6r = p1r - p5r;  q6i = p1i - p5i
    q8r = p3r - p1r;  q8i = p3i - p1i
    q1r = p6r + p4r;  q1i = p6i + p4i
    q5r = p7r + p1r;  q5i = p7i + p1i
    q0r = R0r + q1r;  q0i = R0i + q1i

    q1r = q1r * C7Q1;  q1i = q1i * C7Q1
    q2r = q2r * C7Q2;  q2i = q2i * C7Q2
    q3r = q3r * C7Q3;  q3i = q3i * C7Q3
    q4r = q4r * C7Q4;  q4i = q4i * C7Q4
    q5r = q5r * C7Q5;  q5i = q5i * C7Q5
    q6r = q6r * C7Q6;  q6i = q6i * C7Q6
    q7r = q7r * C7Q7;  q7i = q7i * C7Q7
    q8r = q8r * C7Q8;  q8i = q8i * C7Q8

    a0r = q0r + q1r;  a0i = q0i + q1i
    a1r = q2r + q3r;  a1i = q2i + q3i
    a2r = q4r - q3r;  a2i = q4i - q3i
    a3r = -q2r - q4r; a3i = -q2i - q4i
    a4r = q6r + q7r;  a4i = q6i + q7i
    a5r = q8r - q7r;  a5i = q8i - q7i
    a6r = -q8r - q6r; a6i = -q8i - q6i

    b0r = a0r + a1r;  b0i = a0i + a1i
    b1r = a0r + a2r;  b1i = a0i + a2i
    b2r = a0r + a3r;  b2i = a0i + a3i
    c0r = a4r + q5r;  c0i = a4i + q5i
    c1r = a5r + q5r;  c1i = a5i + q5i
    c2r = a6r + q5r;  c2i = a6i + q5i

    tl.store(out_ptr + base + 0,  q0r,        mask=mask)
    tl.store(out_ptr + base + 1,  q0i,        mask=mask)
    tl.store(out_ptr + base + 2,  b0r + c0i,  mask=mask)
    tl.store(out_ptr + base + 3,  b0i - c0r,  mask=mask)
    tl.store(out_ptr + base + 4,  b2r + c2i,  mask=mask)
    tl.store(out_ptr + base + 5,  b2i - c2r,  mask=mask)
    tl.store(out_ptr + base + 6,  b1r - c1i,  mask=mask)
    tl.store(out_ptr + base + 7,  b1i + c1r,  mask=mask)
    tl.store(out_ptr + base + 8,  b1r + c1i,  mask=mask)
    tl.store(out_ptr + base + 9,  b1i - c1r,  mask=mask)
    tl.store(out_ptr + base + 10, b2r - c2i,  mask=mask)
    tl.store(out_ptr + base + 11, b2i + c2r,  mask=mask)
    tl.store(out_ptr + base + 12, b0r - c0i,  mask=mask)
    tl.store(out_ptr + base + 13, b0i + c0r,  mask=mask)


@triton.autotune(configs=_CONFIGS, key=['n_groups'])
@triton.jit
def _batch_rad7_inv_kernel(
    x_ptr,
    out_ptr,
    n_groups,
    BLOCK_SIZE: tl.constexpr,
):
    C7Q1 = -1.1666666666666665
    C7Q2 = 0.7901564685254002
    C7Q3 = 0.05585426728964774
    C7Q4 = 0.7343022012357524
    C7Q5 = 0.4409585518440984
    C7Q6 = 0.3408729306239314
    C7Q7 = -0.5339693603377252
    C7Q8 = 0.8748422909616567

    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_groups
    base = offs * 14

    R0r = tl.load(x_ptr + base + 0,  mask=mask, other=0.0)
    R0i = tl.load(x_ptr + base + 1,  mask=mask, other=0.0)
    R1r = tl.load(x_ptr + base + 2,  mask=mask, other=0.0)
    R1i = tl.load(x_ptr + base + 3,  mask=mask, other=0.0)
    R2r = tl.load(x_ptr + base + 4,  mask=mask, other=0.0)
    R2i = tl.load(x_ptr + base + 5,  mask=mask, other=0.0)
    R3r = tl.load(x_ptr + base + 6,  mask=mask, other=0.0)
    R3i = tl.load(x_ptr + base + 7,  mask=mask, other=0.0)
    R4r = tl.load(x_ptr + base + 8,  mask=mask, other=0.0)
    R4i = tl.load(x_ptr + base + 9,  mask=mask, other=0.0)
    R5r = tl.load(x_ptr + base + 10, mask=mask, other=0.0)
    R5i = tl.load(x_ptr + base + 11, mask=mask, other=0.0)
    R6r = tl.load(x_ptr + base + 12, mask=mask, other=0.0)
    R6i = tl.load(x_ptr + base + 13, mask=mask, other=0.0)

    # Inverse butterfly (negated C7Q5-C7Q8)
    p0r = R1r + R6r;  p0i = R1i + R6i
    p1r = R1r - R6r;  p1i = R1i - R6i
    p2r = R2r + R5r;  p2i = R2i + R5i
    p3r = R2r - R5r;  p3i = R2i - R5i
    p4r = R4r + R3r;  p4i = R4i + R3i
    p5r = R4r - R3r;  p5i = R4i - R3i

    p6r = p2r + p0r;  p6i = p2i + p0i
    q4r = p2r - p0r;  q4i = p2i - p0i
    q2r = p0r - p4r;  q2i = p0i - p4i
    q3r = p4r - p2r;  q3i = p4i - p2i
    p7r = p5r + p3r;  p7i = p5i + p3i
    q7r = p5r - p3r;  q7i = p5i - p3i
    q6r = p1r - p5r;  q6i = p1i - p5i
    q8r = p3r - p1r;  q8i = p3i - p1i
    q1r = p6r + p4r;  q1i = p6i + p4i
    q5r = p7r + p1r;  q5i = p7i + p1i
    q0r = R0r + q1r;  q0i = R0i + q1i

    q1r = q1r * C7Q1;  q1i = q1i * C7Q1
    q2r = q2r * C7Q2;  q2i = q2i * C7Q2
    q3r = q3r * C7Q3;  q3i = q3i * C7Q3
    q4r = q4r * C7Q4;  q4i = q4i * C7Q4
    q5r = -q5r * C7Q5;  q5i = -q5i * C7Q5
    q6r = -q6r * C7Q6;  q6i = -q6i * C7Q6
    q7r = -q7r * C7Q7;  q7i = -q7i * C7Q7
    q8r = -q8r * C7Q8;  q8i = -q8i * C7Q8

    a0r = q0r + q1r;  a0i = q0i + q1i
    a1r = q2r + q3r;  a1i = q2i + q3i
    a2r = q4r - q3r;  a2i = q4i - q3i
    a3r = -q2r - q4r; a3i = -q2i - q4i
    a4r = q6r + q7r;  a4i = q6i + q7i
    a5r = q8r - q7r;  a5i = q8i - q7i
    a6r = -q8r - q6r; a6i = -q8i - q6i

    b0r = a0r + a1r;  b0i = a0i + a1i
    b1r = a0r + a2r;  b1i = a0i + a2i
    b2r = a0r + a3r;  b2i = a0i + a3i
    c0r = a4r + q5r;  c0i = a4i + q5i
    c1r = a5r + q5r;  c1i = a5i + q5i
    c2r = a6r + q5r;  c2i = a6i + q5i

    tl.store(out_ptr + base + 0,  q0r,        mask=mask)
    tl.store(out_ptr + base + 1,  q0i,        mask=mask)
    tl.store(out_ptr + base + 2,  b0r + c0i,  mask=mask)
    tl.store(out_ptr + base + 3,  b0i - c0r,  mask=mask)
    tl.store(out_ptr + base + 4,  b2r + c2i,  mask=mask)
    tl.store(out_ptr + base + 5,  b2i - c2r,  mask=mask)
    tl.store(out_ptr + base + 6,  b1r - c1i,  mask=mask)
    tl.store(out_ptr + base + 7,  b1i + c1r,  mask=mask)
    tl.store(out_ptr + base + 8,  b1r + c1i,  mask=mask)
    tl.store(out_ptr + base + 9,  b1i - c1r,  mask=mask)
    tl.store(out_ptr + base + 10, b2r - c2i,  mask=mask)
    tl.store(out_ptr + base + 11, b2i + c2r,  mask=mask)
    tl.store(out_ptr + base + 12, b0r - c0i,  mask=mask)
    tl.store(out_ptr + base + 13, b0i + c0r,  mask=mask)


# ======================== Python wrappers ========================
# Zero-copy: complex64 ↔ float32 via view_as_real / view_as_complex

def _complex_to_interleaved(x: torch.Tensor):
    """complex64 (..., 7) → float32 (n_groups, 14), zero-copy."""
    if not x.is_contiguous():
        x = x.contiguous()
    x_flat = x.reshape(-1, 7)
    n_groups = x_flat.shape[0]
    x_float = torch.view_as_real(x_flat).reshape(n_groups, 14)
    return x_float, n_groups, x.shape


def _interleaved_to_complex(out_float: torch.Tensor, n_groups: int, original_shape):
    """float32 (n_groups, 14) → complex64 with original_shape, zero-copy."""
    return torch.view_as_complex(out_float.reshape(n_groups, 7, 2)).reshape(original_shape)


def batch_fwd_rad7_b1(x: torch.Tensor) -> torch.Tensor:
    """Batch forward radix-7 FFT using Triton kernels."""
    if not torch.is_complex(x):
        raise ValueError("Radix-7 FFT requires complex input")
    x_float, n_groups, original_shape = _complex_to_interleaved(x)
    out_float = torch.empty_like(x_float)
    grid = lambda meta: (triton.cdiv(n_groups, meta['BLOCK_SIZE']),)
    _batch_rad7_fwd_kernel[grid](x_float, out_float, n_groups)
    return _interleaved_to_complex(out_float, n_groups, original_shape)


def batch_inv_rad7_b1(x: torch.Tensor) -> torch.Tensor:
    """Batch inverse radix-7 FFT using Triton kernels."""
    if not torch.is_complex(x):
        raise ValueError("Radix-7 FFT requires complex input")
    x_float, n_groups, original_shape = _complex_to_interleaved(x)
    out_float = torch.empty_like(x_float)
    grid = lambda meta: (triton.cdiv(n_groups, meta['BLOCK_SIZE']),)
    _batch_rad7_inv_kernel[grid](x_float, out_float, n_groups)
    return _interleaved_to_complex(out_float, n_groups, original_shape)


# Aliases for public API
triton_fwd_rad7_b1 = batch_fwd_rad7_b1
triton_inv_rad7_b1 = batch_inv_rad7_b1


# ======================== SoA (Structure-of-Arrays) layout kernels ========================
# Memory layout: all R0r values, then all R0i values, ..., then all R6i values.
# Stride between adjacent threads = 1 float → fully coalesced access.
# Trade-off: requires a transpose in the wrapper (permute + contiguous).

@triton.autotune(configs=_CONFIGS, key=['n_groups'])
@triton.jit
def _batch_rad7_fwd_soa_kernel(
    x_ptr,
    out_ptr,
    n_groups,
    BLOCK_SIZE: tl.constexpr,
):
    C7Q1 = -1.1666666666666665
    C7Q2 = 0.7901564685254002
    C7Q3 = 0.05585426728964774
    C7Q4 = 0.7343022012357524
    C7Q5 = 0.4409585518440984
    C7Q6 = 0.3408729306239314
    C7Q7 = -0.5339693603377252
    C7Q8 = 0.8748422909616567

    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_groups

    # SoA: component j stored at x_ptr + j * n_groups + offs
    # j: 0=R0r, 1=R0i, 2=R1r, 3=R1i, ..., 12=R6r, 13=R6i
    R0r = tl.load(x_ptr +  0 * n_groups + offs, mask=mask, other=0.0)
    R0i = tl.load(x_ptr +  1 * n_groups + offs, mask=mask, other=0.0)
    R1r = tl.load(x_ptr +  2 * n_groups + offs, mask=mask, other=0.0)
    R1i = tl.load(x_ptr +  3 * n_groups + offs, mask=mask, other=0.0)
    R2r = tl.load(x_ptr +  4 * n_groups + offs, mask=mask, other=0.0)
    R2i = tl.load(x_ptr +  5 * n_groups + offs, mask=mask, other=0.0)
    R3r = tl.load(x_ptr +  6 * n_groups + offs, mask=mask, other=0.0)
    R3i = tl.load(x_ptr +  7 * n_groups + offs, mask=mask, other=0.0)
    R4r = tl.load(x_ptr +  8 * n_groups + offs, mask=mask, other=0.0)
    R4i = tl.load(x_ptr +  9 * n_groups + offs, mask=mask, other=0.0)
    R5r = tl.load(x_ptr + 10 * n_groups + offs, mask=mask, other=0.0)
    R5i = tl.load(x_ptr + 11 * n_groups + offs, mask=mask, other=0.0)
    R6r = tl.load(x_ptr + 12 * n_groups + offs, mask=mask, other=0.0)
    R6i = tl.load(x_ptr + 13 * n_groups + offs, mask=mask, other=0.0)

    p0r = R1r + R6r;  p0i = R1i + R6i
    p1r = R1r - R6r;  p1i = R1i - R6i
    p2r = R2r + R5r;  p2i = R2i + R5i
    p3r = R2r - R5r;  p3i = R2i - R5i
    p4r = R4r + R3r;  p4i = R4i + R3i
    p5r = R4r - R3r;  p5i = R4i - R3i

    p6r = p2r + p0r;  p6i = p2i + p0i
    q4r = p2r - p0r;  q4i = p2i - p0i
    q2r = p0r - p4r;  q2i = p0i - p4i
    q3r = p4r - p2r;  q3i = p4i - p2i
    p7r = p5r + p3r;  p7i = p5i + p3i
    q7r = p5r - p3r;  q7i = p5i - p3i
    q6r = p1r - p5r;  q6i = p1i - p5i
    q8r = p3r - p1r;  q8i = p3i - p1i
    q1r = p6r + p4r;  q1i = p6i + p4i
    q5r = p7r + p1r;  q5i = p7i + p1i
    q0r = R0r + q1r;  q0i = R0i + q1i

    q1r = q1r * C7Q1;  q1i = q1i * C7Q1
    q2r = q2r * C7Q2;  q2i = q2i * C7Q2
    q3r = q3r * C7Q3;  q3i = q3i * C7Q3
    q4r = q4r * C7Q4;  q4i = q4i * C7Q4
    q5r = q5r * C7Q5;  q5i = q5i * C7Q5
    q6r = q6r * C7Q6;  q6i = q6i * C7Q6
    q7r = q7r * C7Q7;  q7i = q7i * C7Q7
    q8r = q8r * C7Q8;  q8i = q8i * C7Q8

    a0r = q0r + q1r;  a0i = q0i + q1i
    a1r = q2r + q3r;  a1i = q2i + q3i
    a2r = q4r - q3r;  a2i = q4i - q3i
    a3r = -q2r - q4r; a3i = -q2i - q4i
    a4r = q6r + q7r;  a4i = q6i + q7i
    a5r = q8r - q7r;  a5i = q8i - q7i
    a6r = -q8r - q6r; a6i = -q8i - q6i

    b0r = a0r + a1r;  b0i = a0i + a1i
    b1r = a0r + a2r;  b1i = a0i + a2i
    b2r = a0r + a3r;  b2i = a0i + a3i
    c0r = a4r + q5r;  c0i = a4i + q5i
    c1r = a5r + q5r;  c1i = a5i + q5i
    c2r = a6r + q5r;  c2i = a6i + q5i

    tl.store(out_ptr +  0 * n_groups + offs, q0r,        mask=mask)
    tl.store(out_ptr +  1 * n_groups + offs, q0i,        mask=mask)
    tl.store(out_ptr +  2 * n_groups + offs, b0r + c0i,  mask=mask)
    tl.store(out_ptr +  3 * n_groups + offs, b0i - c0r,  mask=mask)
    tl.store(out_ptr +  4 * n_groups + offs, b2r + c2i,  mask=mask)
    tl.store(out_ptr +  5 * n_groups + offs, b2i - c2r,  mask=mask)
    tl.store(out_ptr +  6 * n_groups + offs, b1r - c1i,  mask=mask)
    tl.store(out_ptr +  7 * n_groups + offs, b1i + c1r,  mask=mask)
    tl.store(out_ptr +  8 * n_groups + offs, b1r + c1i,  mask=mask)
    tl.store(out_ptr +  9 * n_groups + offs, b1i - c1r,  mask=mask)
    tl.store(out_ptr + 10 * n_groups + offs, b2r - c2i,  mask=mask)
    tl.store(out_ptr + 11 * n_groups + offs, b2i + c2r,  mask=mask)
    tl.store(out_ptr + 12 * n_groups + offs, b0r - c0i,  mask=mask)
    tl.store(out_ptr + 13 * n_groups + offs, b0i + c0r,  mask=mask)


@triton.autotune(configs=_CONFIGS, key=['n_groups'])
@triton.jit
def _batch_rad7_inv_soa_kernel(
    x_ptr,
    out_ptr,
    n_groups,
    BLOCK_SIZE: tl.constexpr,
):
    C7Q1 = -1.1666666666666665
    C7Q2 = 0.7901564685254002
    C7Q3 = 0.05585426728964774
    C7Q4 = 0.7343022012357524
    C7Q5 = 0.4409585518440984
    C7Q6 = 0.3408729306239314
    C7Q7 = -0.5339693603377252
    C7Q8 = 0.8748422909616567

    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_groups

    R0r = tl.load(x_ptr +  0 * n_groups + offs, mask=mask, other=0.0)
    R0i = tl.load(x_ptr +  1 * n_groups + offs, mask=mask, other=0.0)
    R1r = tl.load(x_ptr +  2 * n_groups + offs, mask=mask, other=0.0)
    R1i = tl.load(x_ptr +  3 * n_groups + offs, mask=mask, other=0.0)
    R2r = tl.load(x_ptr +  4 * n_groups + offs, mask=mask, other=0.0)
    R2i = tl.load(x_ptr +  5 * n_groups + offs, mask=mask, other=0.0)
    R3r = tl.load(x_ptr +  6 * n_groups + offs, mask=mask, other=0.0)
    R3i = tl.load(x_ptr +  7 * n_groups + offs, mask=mask, other=0.0)
    R4r = tl.load(x_ptr +  8 * n_groups + offs, mask=mask, other=0.0)
    R4i = tl.load(x_ptr +  9 * n_groups + offs, mask=mask, other=0.0)
    R5r = tl.load(x_ptr + 10 * n_groups + offs, mask=mask, other=0.0)
    R5i = tl.load(x_ptr + 11 * n_groups + offs, mask=mask, other=0.0)
    R6r = tl.load(x_ptr + 12 * n_groups + offs, mask=mask, other=0.0)
    R6i = tl.load(x_ptr + 13 * n_groups + offs, mask=mask, other=0.0)

    p0r = R1r + R6r;  p0i = R1i + R6i
    p1r = R1r - R6r;  p1i = R1i - R6i
    p2r = R2r + R5r;  p2i = R2i + R5i
    p3r = R2r - R5r;  p3i = R2i - R5i
    p4r = R4r + R3r;  p4i = R4i + R3i
    p5r = R4r - R3r;  p5i = R4i - R3i

    p6r = p2r + p0r;  p6i = p2i + p0i
    q4r = p2r - p0r;  q4i = p2i - p0i
    q2r = p0r - p4r;  q2i = p0i - p4i
    q3r = p4r - p2r;  q3i = p4i - p2i
    p7r = p5r + p3r;  p7i = p5i + p3i
    q7r = p5r - p3r;  q7i = p5i - p3i
    q6r = p1r - p5r;  q6i = p1i - p5i
    q8r = p3r - p1r;  q8i = p3i - p1i
    q1r = p6r + p4r;  q1i = p6i + p4i
    q5r = p7r + p1r;  q5i = p7i + p1i
    q0r = R0r + q1r;  q0i = R0i + q1i

    q1r = q1r * C7Q1;  q1i = q1i * C7Q1
    q2r = q2r * C7Q2;  q2i = q2i * C7Q2
    q3r = q3r * C7Q3;  q3i = q3i * C7Q3
    q4r = q4r * C7Q4;  q4i = q4i * C7Q4
    q5r = -q5r * C7Q5;  q5i = -q5i * C7Q5
    q6r = -q6r * C7Q6;  q6i = -q6i * C7Q6
    q7r = -q7r * C7Q7;  q7i = -q7i * C7Q7
    q8r = -q8r * C7Q8;  q8i = -q8i * C7Q8

    a0r = q0r + q1r;  a0i = q0i + q1i
    a1r = q2r + q3r;  a1i = q2i + q3i
    a2r = q4r - q3r;  a2i = q4i - q3i
    a3r = -q2r - q4r; a3i = -q2i - q4i
    a4r = q6r + q7r;  a4i = q6i + q7i
    a5r = q8r - q7r;  a5i = q8i - q7i
    a6r = -q8r - q6r; a6i = -q8i - q6i

    b0r = a0r + a1r;  b0i = a0i + a1i
    b1r = a0r + a2r;  b1i = a0i + a2i
    b2r = a0r + a3r;  b2i = a0i + a3i
    c0r = a4r + q5r;  c0i = a4i + q5i
    c1r = a5r + q5r;  c1i = a5i + q5i
    c2r = a6r + q5r;  c2i = a6i + q5i

    tl.store(out_ptr +  0 * n_groups + offs, q0r,        mask=mask)
    tl.store(out_ptr +  1 * n_groups + offs, q0i,        mask=mask)
    tl.store(out_ptr +  2 * n_groups + offs, b0r + c0i,  mask=mask)
    tl.store(out_ptr +  3 * n_groups + offs, b0i - c0r,  mask=mask)
    tl.store(out_ptr +  4 * n_groups + offs, b2r + c2i,  mask=mask)
    tl.store(out_ptr +  5 * n_groups + offs, b2i - c2r,  mask=mask)
    tl.store(out_ptr +  6 * n_groups + offs, b1r - c1i,  mask=mask)
    tl.store(out_ptr +  7 * n_groups + offs, b1i + c1r,  mask=mask)
    tl.store(out_ptr +  8 * n_groups + offs, b1r + c1i,  mask=mask)
    tl.store(out_ptr +  9 * n_groups + offs, b1i - c1r,  mask=mask)
    tl.store(out_ptr + 10 * n_groups + offs, b2r - c2i,  mask=mask)
    tl.store(out_ptr + 11 * n_groups + offs, b2i + c2r,  mask=mask)
    tl.store(out_ptr + 12 * n_groups + offs, b0r - c0i,  mask=mask)
    tl.store(out_ptr + 13 * n_groups + offs, b0i + c0r,  mask=mask)


def _to_soa(x: torch.Tensor):
    """complex64 (..., 7) → float32 (14 * n_groups,) in SoA order.

    Output memory: [r0_g0..r0_gN, i0_g0..i0_gN, r1_g0..r1_gN, ..., i6_g0..i6_gN]
    Requires a contiguous copy (permute), but enables fully coalesced kernel access.
    """
    if not x.is_contiguous():
        x = x.contiguous()
    original_shape = x.shape
    n_groups = x.numel() // 7
    # (n_groups, 7, 2) → permute → (7, 2, n_groups) → contiguous → (14 * n_groups,)
    x_soa = (torch.view_as_real(x.reshape(n_groups, 7))
             .permute(1, 2, 0)
             .contiguous()
             .reshape(-1))
    return x_soa, n_groups, original_shape


def _from_soa(out_soa: torch.Tensor, n_groups: int, original_shape):
    """float32 (14 * n_groups,) SoA → complex64 with original_shape."""
    # (14, n_groups) = (7, 2, n_groups) → permute → (n_groups, 7, 2) → view_as_complex
    out = (out_soa.reshape(7, 2, n_groups)
           .permute(2, 0, 1)
           .contiguous())
    return torch.view_as_complex(out).reshape(original_shape)


def batch_fwd_rad7_soa(x: torch.Tensor) -> torch.Tensor:
    """Batch forward radix-7 FFT — SoA layout for coalesced memory access."""
    if not torch.is_complex(x):
        raise ValueError("Radix-7 FFT requires complex input")
    x_soa, n_groups, original_shape = _to_soa(x)
    out_soa = torch.empty(14 * n_groups, dtype=torch.float32, device=x.device)
    grid = lambda meta: (triton.cdiv(n_groups, meta['BLOCK_SIZE']),)
    _batch_rad7_fwd_soa_kernel[grid](x_soa, out_soa, n_groups)
    return _from_soa(out_soa, n_groups, original_shape)


def batch_inv_rad7_soa(x: torch.Tensor) -> torch.Tensor:
    """Batch inverse radix-7 FFT — SoA layout for coalesced memory access."""
    if not torch.is_complex(x):
        raise ValueError("Radix-7 FFT requires complex input")
    x_soa, n_groups, original_shape = _to_soa(x)
    out_soa = torch.empty(14 * n_groups, dtype=torch.float32, device=x.device)
    grid = lambda meta: (triton.cdiv(n_groups, meta['BLOCK_SIZE']),)
    _batch_rad7_inv_soa_kernel[grid](x_soa, out_soa, n_groups)
    return _from_soa(out_soa, n_groups, original_shape)


# ======================== DFT Matrix (tl.dot) approach ========================
# Express 7-point FFT as matrix multiply: Y = X @ W  (14×14 real, padded to 16×16).
# tl.dot internally uses shared memory tiling → coalesced access without external transpose.
# ~2.5x more FLOPS than butterfly, but compute is hidden behind memory bandwidth.

def _make_dft7_matrix(forward=True):
    """Create 16×16 padded real DFT matrix for 7-point FFT."""
    N = 7
    W = torch.zeros(16, 16, dtype=torch.float32)
    for k in range(N):
        for n in range(N):
            angle = 2 * math.pi * n * k / N
            c, s = math.cos(angle), math.sin(angle)
            W[2*n, 2*k] = c
            W[2*n+1, 2*k] = s if forward else -s
            W[2*n, 2*k+1] = -s if forward else s
            W[2*n+1, 2*k+1] = c
    return W


_DFT7_FWD = _make_dft7_matrix(forward=True)
_DFT7_INV = _make_dft7_matrix(forward=False)
_dft7_gpu = {}


def _get_dft7(device, forward=True):
    key = (str(device), forward)
    if key not in _dft7_gpu:
        _dft7_gpu[key] = (_DFT7_FWD if forward else _DFT7_INV).to(device)
    return _dft7_gpu[key]


@triton.autotune(configs=_CONFIGS, key=['n_groups'])
@triton.jit
def _rad7_matmul_kernel(x_ptr, out_ptr, W_ptr, n_groups, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    group_offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    comp_offs = tl.arange(0, 16)

    # 2D AoS load: (BLOCK_SIZE, 16), columns 14-15 masked to 0
    base = group_offs[:, None] * 14 + comp_offs[None, :]
    mask = (group_offs[:, None] < n_groups) & (comp_offs[None, :] < 14)
    data = tl.load(x_ptr + base, mask=mask, other=0.0)

    # DFT matrix (16, 16) — small, L2-cached after first block
    W = tl.load(W_ptr + tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :])

    # (BLOCK_SIZE, 16) @ (16, 16) → (BLOCK_SIZE, 16)
    out = tl.dot(data, W, allow_tf32=False)

    tl.store(out_ptr + base, out, mask=mask)


def batch_fwd_rad7_matmul(x: torch.Tensor) -> torch.Tensor:
    """Batch forward radix-7 FFT — DFT matrix via tl.dot."""
    if not torch.is_complex(x):
        raise ValueError("Radix-7 FFT requires complex input")
    x_float, n_groups, orig_shape = _complex_to_interleaved(x)
    out_float = torch.empty_like(x_float)
    W = _get_dft7(x.device, forward=True)
    grid = lambda meta: (triton.cdiv(n_groups, meta['BLOCK_SIZE']),)
    _rad7_matmul_kernel[grid](x_float, out_float, W, n_groups)
    return _interleaved_to_complex(out_float, n_groups, orig_shape)


def batch_inv_rad7_matmul(x: torch.Tensor) -> torch.Tensor:
    """Batch inverse radix-7 FFT — DFT matrix via tl.dot."""
    if not torch.is_complex(x):
        raise ValueError("Radix-7 FFT requires complex input")
    x_float, n_groups, orig_shape = _complex_to_interleaved(x)
    out_float = torch.empty_like(x_float)
    W = _get_dft7(x.device, forward=False)
    grid = lambda meta: (triton.cdiv(n_groups, meta['BLOCK_SIZE']),)
    _rad7_matmul_kernel[grid](x_float, out_float, W, n_groups)
    return _interleaved_to_complex(out_float, n_groups, orig_shape)
