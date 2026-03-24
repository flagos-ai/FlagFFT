"""
Triton implementation of radix-9 FFT codelet.

This module provides forward and inverse radix-9 FFT butterfly operations.
Ported from CUDA radix_9.h
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _batch_rad9_kernel_complex_fwd(
    x_real_ptr,
    x_imag_ptr,
    out_real_ptr,
    out_imag_ptr,
    n_groups: tl.constexpr,
):
    """
    Batch forward radix-9 FFT kernel for complex-valued data.
    """
    # Constants defined inside kernel for Triton compatibility
    C9QA = 0.766044443118978
    C9QB = 0.6427876096865393
    C9QC = 0.1736481776669304
    C9QD = 0.984807753012208
    C9QE = 0.5
    C9QF = 0.8660254037844387
    C9QG = 0.9396926207859083
    C9QH = 0.3420201433256689

    pid = tl.program_id(0)

    if pid < n_groups:
        # Manual index computation
        idx0 = 9 * pid
        idx1 = 9 * pid + 1
        idx2 = 9 * pid + 2
        idx3 = 9 * pid + 3
        idx4 = 9 * pid + 4
        idx5 = 9 * pid + 5
        idx6 = 9 * pid + 6
        idx7 = 9 * pid + 7
        idx8 = 9 * pid + 8

        # Load all 9 inputs - manual unrolling
        R0 = tl.load(x_real_ptr + idx0)
        R1 = tl.load(x_real_ptr + idx1)
        R2 = tl.load(x_real_ptr + idx2)
        R3 = tl.load(x_real_ptr + idx3)
        R4 = tl.load(x_real_ptr + idx4)
        R5 = tl.load(x_real_ptr + idx5)
        R6 = tl.load(x_real_ptr + idx6)
        R7 = tl.load(x_real_ptr + idx7)
        R8 = tl.load(x_real_ptr + idx8)

        I0 = tl.load(x_imag_ptr + idx0)
        I1 = tl.load(x_imag_ptr + idx1)
        I2 = tl.load(x_imag_ptr + idx2)
        I3 = tl.load(x_imag_ptr + idx3)
        I4 = tl.load(x_imag_ptr + idx4)
        I5 = tl.load(x_imag_ptr + idx5)
        I6 = tl.load(x_imag_ptr + idx6)
        I7 = tl.load(x_imag_ptr + idx7)
        I8 = tl.load(x_imag_ptr + idx8)

        # Pre-compute sums and differences
        v0_real = R1 + R8
        v0_imag = I1 + I8
        v1_real = R2 + R7
        v1_imag = I2 + I7
        v2_real = R3 + R6
        v2_imag = I3 + I6

        p0_real = R1 - R8
        p0_imag = I1 - I8
        p1_real = R2 - R7
        p1_imag = I2 - I7
        p2_real = (R3 - R6) * C9QF
        p2_imag = (I3 - I6) * C9QF

        R4_plus_R5_real = R4 + R5
        R4_plus_R5_imag = I4 + I5
        R4_minus_R5_real = R4 - R5
        R4_minus_R5_imag = I4 - I5

        # Compute temporary values
        t8_real = (C9QB * p0_real + C9QD * p1_real + p2_real + C9QH * R4_minus_R5_real)
        t8_imag = (C9QB * p0_imag + C9QD * p1_imag + p2_imag + C9QH * R4_minus_R5_imag)

        R1_out_real = (R0 + C9QA * v0_real + C9QC * v1_real - C9QE * v2_real - C9QG * R4_plus_R5_real) + t8_imag
        R1_out_imag = (I0 + C9QA * v0_imag + C9QC * v1_imag - C9QE * v2_imag - C9QG * R4_plus_R5_imag) - t8_real
        R8_out_real = R1_out_real - 2.0 * t8_imag
        R8_out_imag = R1_out_imag + 2.0 * t8_real

        t7_real = (-C9QB * R4_minus_R5_real + C9QD * p0_real - p2_real + C9QH * p1_real)
        t7_imag = (-C9QB * R4_minus_R5_imag + C9QD * p0_imag - p2_imag + C9QH * p1_imag)

        R2_out_real = (R0 + C9QA * R4_plus_R5_real + C9QC * v0_real - C9QE * v2_real - C9QG * v1_real) + t7_imag
        R2_out_imag = (I0 + C9QA * R4_plus_R5_imag + C9QC * v0_imag - C9QE * v2_imag - C9QG * v1_imag) - t7_real
        R7_out_real = R2_out_real - 2.0 * t7_imag
        R7_out_imag = R2_out_imag + 2.0 * t7_real

        t6_real = C9QF * (p0_real + R4_minus_R5_real - p1_real)
        t6_imag = C9QF * (p0_imag + R4_minus_R5_imag - p1_imag)

        R3_out_real = (R0 + v2_real - C9QE * (v0_real + v1_real + R4_plus_R5_real)) + t6_imag
        R3_out_imag = (I0 + v2_imag - C9QE * (v0_imag + v1_imag + R4_plus_R5_imag)) - t6_real
        R6_out_real = R3_out_real - 2.0 * t6_imag
        R6_out_imag = R3_out_imag + 2.0 * t6_real

        t0_real = (-C9QB * p1_real - C9QD * R4_minus_R5_real + p2_real + C9QH * p0_real)
        t0_imag = (-C9QB * p1_imag - C9QD * R4_minus_R5_imag + p2_imag + C9QH * p0_imag)

        R0_out_real = R0 + v0_real + v1_real + v2_real + R4_plus_R5_real
        R0_out_imag = I0 + v0_imag + v1_imag + v2_imag + R4_plus_R5_imag

        R4_out_real = (R0 + C9QA * v1_real + C9QC * R4_plus_R5_real - C9QE * v2_real - C9QG * v0_real) + t0_imag
        R4_out_imag = (I0 + C9QA * v1_imag + C9QC * R4_plus_R5_imag - C9QE * v2_imag - C9QG * v0_imag) - t0_real
        R5_out_real = R4_out_real - 2.0 * t0_imag
        R5_out_imag = R4_out_imag + 2.0 * t0_real

        # Store results
        tl.store(out_real_ptr + idx0, R0_out_real)
        tl.store(out_imag_ptr + idx0, R0_out_imag)
        tl.store(out_real_ptr + idx1, R1_out_real)
        tl.store(out_imag_ptr + idx1, R1_out_imag)
        tl.store(out_real_ptr + idx2, R2_out_real)
        tl.store(out_imag_ptr + idx2, R2_out_imag)
        tl.store(out_real_ptr + idx3, R3_out_real)
        tl.store(out_imag_ptr + idx3, R3_out_imag)
        tl.store(out_real_ptr + idx4, R4_out_real)
        tl.store(out_imag_ptr + idx4, R4_out_imag)
        tl.store(out_real_ptr + idx5, R5_out_real)
        tl.store(out_imag_ptr + idx5, R5_out_imag)
        tl.store(out_real_ptr + idx6, R6_out_real)
        tl.store(out_imag_ptr + idx6, R6_out_imag)
        tl.store(out_real_ptr + idx7, R7_out_real)
        tl.store(out_imag_ptr + idx7, R7_out_imag)
        tl.store(out_real_ptr + idx8, R8_out_real)
        tl.store(out_imag_ptr + idx8, R8_out_imag)


@triton.jit
def _batch_rad9_kernel_complex_inv(
    x_real_ptr,
    x_imag_ptr,
    out_real_ptr,
    out_imag_ptr,
    n_groups: tl.constexpr,
):
    """
    Batch inverse radix-9 FFT kernel for complex-valued data.
    """
    # Constants defined inside kernel for Triton compatibility
    C9QA = 0.766044443118978
    C9QB = 0.6427876096865393
    C9QC = 0.1736481776669304
    C9QD = 0.984807753012208
    C9QE = 0.5
    C9QF = 0.8660254037844387
    C9QG = 0.9396926207859083
    C9QH = 0.3420201433256689

    pid = tl.program_id(0)

    if pid < n_groups:
        # Manual index computation
        idx0 = 9 * pid
        idx1 = 9 * pid + 1
        idx2 = 9 * pid + 2
        idx3 = 9 * pid + 3
        idx4 = 9 * pid + 4
        idx5 = 9 * pid + 5
        idx6 = 9 * pid + 6
        idx7 = 9 * pid + 7
        idx8 = 9 * pid + 8

        # Load all 9 inputs - manual unrolling
        R0 = tl.load(x_real_ptr + idx0)
        R1 = tl.load(x_real_ptr + idx1)
        R2 = tl.load(x_real_ptr + idx2)
        R3 = tl.load(x_real_ptr + idx3)
        R4 = tl.load(x_real_ptr + idx4)
        R5 = tl.load(x_real_ptr + idx5)
        R6 = tl.load(x_real_ptr + idx6)
        R7 = tl.load(x_real_ptr + idx7)
        R8 = tl.load(x_real_ptr + idx8)

        I0 = tl.load(x_imag_ptr + idx0)
        I1 = tl.load(x_imag_ptr + idx1)
        I2 = tl.load(x_imag_ptr + idx2)
        I3 = tl.load(x_imag_ptr + idx3)
        I4 = tl.load(x_imag_ptr + idx4)
        I5 = tl.load(x_imag_ptr + idx5)
        I6 = tl.load(x_imag_ptr + idx6)
        I7 = tl.load(x_imag_ptr + idx7)
        I8 = tl.load(x_imag_ptr + idx8)

        # Pre-compute sums and differences
        v0_real = R1 + R8
        v0_imag = I1 + I8
        v1_real = R2 + R7
        v1_imag = I2 + I7
        v2_real = R3 + R6
        v2_imag = I3 + I6

        p0_real = R1 - R8
        p0_imag = I1 - I8
        p1_real = R2 - R7
        p1_imag = I2 - I7
        p2_real = (R3 - R6) * C9QF
        p2_imag = (I3 - I6) * C9QF

        R4_plus_R5_real = R4 + R5
        R4_plus_R5_imag = I4 + I5
        R4_minus_R5_real = R4 - R5
        R4_minus_R5_imag = I4 - I5

        # Compute temporary values (inverse has opposite sign on imaginary part)
        t8_real = (C9QB * p0_real + C9QD * p1_real + p2_real + C9QH * R4_minus_R5_real)
        t8_imag = (C9QB * p0_imag + C9QD * p1_imag + p2_imag + C9QH * R4_minus_R5_imag)

        R1_out_real = (R0 + C9QA * v0_real + C9QC * v1_real - C9QE * v2_real - C9QG * R4_plus_R5_real) - t8_imag
        R1_out_imag = (I0 + C9QA * v0_imag + C9QC * v1_imag - C9QE * v2_imag - C9QG * R4_plus_R5_imag) + t8_real
        R8_out_real = R1_out_real + 2.0 * t8_imag
        R8_out_imag = R1_out_imag - 2.0 * t8_real

        t7_real = (-C9QB * R4_minus_R5_real + C9QD * p0_real - p2_real + C9QH * p1_real)
        t7_imag = (-C9QB * R4_minus_R5_imag + C9QD * p0_imag - p2_imag + C9QH * p1_imag)

        R2_out_real = (R0 + C9QA * R4_plus_R5_real + C9QC * v0_real - C9QE * v2_real - C9QG * v1_real) - t7_imag
        R2_out_imag = (I0 + C9QA * R4_plus_R5_imag + C9QC * v0_imag - C9QE * v2_imag - C9QG * v1_imag) + t7_real
        R7_out_real = R2_out_real + 2.0 * t7_imag
        R7_out_imag = R2_out_imag - 2.0 * t7_real

        t6_real = C9QF * (p0_real + R4_minus_R5_real - p1_real)
        t6_imag = C9QF * (p0_imag + R4_minus_R5_imag - p1_imag)

        R3_out_real = (R0 + v2_real - C9QE * (v0_real + v1_real + R4_plus_R5_real)) - t6_imag
        R3_out_imag = (I0 + v2_imag - C9QE * (v0_imag + v1_imag + R4_plus_R5_imag)) + t6_real
        R6_out_real = R3_out_real + 2.0 * t6_imag
        R6_out_imag = R3_out_imag - 2.0 * t6_real

        t0_real = (-C9QB * p1_real - C9QD * R4_minus_R5_real + p2_real + C9QH * p0_real)
        t0_imag = (-C9QB * p1_imag - C9QD * R4_minus_R5_imag + p2_imag + C9QH * p0_imag)

        R0_out_real = R0 + v0_real + v1_real + v2_real + R4_plus_R5_real
        R0_out_imag = I0 + v0_imag + v1_imag + v2_imag + R4_plus_R5_imag

        R4_out_real = (R0 + C9QA * v1_real + C9QC * R4_plus_R5_real - C9QE * v2_real - C9QG * v0_real) - t0_imag
        R4_out_imag = (I0 + C9QA * v1_imag + C9QC * R4_plus_R5_imag - C9QE * v2_imag - C9QG * v0_imag) + t0_real
        R5_out_real = R4_out_real + 2.0 * t0_imag
        R5_out_imag = R4_out_imag - 2.0 * t0_real

        # Store results
        tl.store(out_real_ptr + idx0, R0_out_real)
        tl.store(out_imag_ptr + idx0, R0_out_imag)
        tl.store(out_real_ptr + idx1, R1_out_real)
        tl.store(out_imag_ptr + idx1, R1_out_imag)
        tl.store(out_real_ptr + idx2, R2_out_real)
        tl.store(out_imag_ptr + idx2, R2_out_imag)
        tl.store(out_real_ptr + idx3, R3_out_real)
        tl.store(out_imag_ptr + idx3, R3_out_imag)
        tl.store(out_real_ptr + idx4, R4_out_real)
        tl.store(out_imag_ptr + idx4, R4_out_imag)
        tl.store(out_real_ptr + idx5, R5_out_real)
        tl.store(out_imag_ptr + idx5, R5_out_imag)
        tl.store(out_real_ptr + idx6, R6_out_real)
        tl.store(out_imag_ptr + idx6, R6_out_imag)
        tl.store(out_real_ptr + idx7, R7_out_real)
        tl.store(out_imag_ptr + idx7, R7_out_imag)
        tl.store(out_real_ptr + idx8, R8_out_real)
        tl.store(out_imag_ptr + idx8, R8_out_imag)


def batch_fwd_rad9_b1(x: torch.Tensor) -> torch.Tensor:
    """
    Batch forward radix-9 FFT using Triton kernels.
    """
    x = x.contiguous()
    original_shape = x.shape

    if torch.is_complex(x):
        x_flat = x.view(-1)
        n_groups = x_flat.shape[0] // 9

        x_real = x_flat.real.contiguous()
        x_imag = x_flat.imag.contiguous()

        out_real = torch.empty_like(x_real)
        out_imag = torch.empty_like(x_imag)

        grid = lambda meta: (n_groups,)
        _batch_rad9_kernel_complex_fwd[grid](x_real, x_imag, out_real, out_imag, n_groups)

        out = torch.complex(out_real, out_imag)
        return out.view(original_shape)
    else:
        raise ValueError("Radix-9 FFT requires complex input")


def batch_inv_rad9_b1(x: torch.Tensor) -> torch.Tensor:
    """
    Batch inverse radix-9 FFT using Triton kernels.
    """
    x = x.contiguous()
    original_shape = x.shape

    if torch.is_complex(x):
        x_flat = x.view(-1)
        n_groups = x_flat.shape[0] // 9

        x_real = x_flat.real.contiguous()
        x_imag = x_flat.imag.contiguous()

        out_real = torch.empty_like(x_real)
        out_imag = torch.empty_like(x_imag)

        grid = lambda meta: (n_groups,)
        _batch_rad9_kernel_complex_inv[grid](x_real, x_imag, out_real, out_imag, n_groups)

        out = torch.complex(out_real, out_imag)
        return out.view(original_shape)
    else:
        raise ValueError("Radix-9 FFT requires complex input")


# Aliases for public API
triton_fwd_rad9_b1 = batch_fwd_rad9_b1
triton_inv_rad9_b1 = batch_inv_rad9_b1
