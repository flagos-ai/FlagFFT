"""
Triton implementation of radix-5 FFT codelet.

This module provides forward and inverse radix-5 FFT butterfly operations.
Ported from CUDA radix_5.h
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _batch_rad5_kernel_complex_fwd(
    x_real_ptr,
    x_imag_ptr,
    out_real_ptr,
    out_imag_ptr,
    n_groups: tl.constexpr,
):
    """
    Batch forward radix-5 FFT kernel for complex-valued data.
    """
    # Constants defined inside kernel for Triton compatibility
    C5QA = 0.3090169943749474  # cos(2*pi/5) - 0.5
    C5QB = 0.9510565162951536
    C5QC = 0.5
    C5QD = 0.5877852522924731

    pid = tl.program_id(0)

    if pid < n_groups:
        idx0 = 5 * pid
        idx1 = 5 * pid + 1
        idx2 = 5 * pid + 2
        idx3 = 5 * pid + 3
        idx4 = 5 * pid + 4

        # Load real parts
        R0_real = tl.load(x_real_ptr + idx0)
        R1_real = tl.load(x_real_ptr + idx1)
        R2_real = tl.load(x_real_ptr + idx2)
        R3_real = tl.load(x_real_ptr + idx3)
        R4_real = tl.load(x_real_ptr + idx4)

        # Load imaginary parts
        R0_imag = tl.load(x_imag_ptr + idx0)
        R1_imag = tl.load(x_imag_ptr + idx1)
        R2_imag = tl.load(x_imag_ptr + idx2)
        R3_imag = tl.load(x_imag_ptr + idx3)
        R4_imag = tl.load(x_imag_ptr + idx4)

        # Forward FFT computation
        TR0 = R0_real + R1_real + R2_real + R3_real + R4_real
        TR1 = (R0_real - C5QC * (R2_real + R3_real)) + C5QB * (R1_imag - R4_imag) + C5QD * (R2_imag - R3_imag) + C5QA * ((R1_real - R2_real) + (R4_real - R3_real))
        TR4 = (R0_real - C5QC * (R2_real + R3_real)) - C5QB * (R1_imag - R4_imag) - C5QD * (R2_imag - R3_imag) + C5QA * ((R1_real - R2_real) + (R4_real - R3_real))
        TR2 = (R0_real - C5QC * (R1_real + R4_real)) - C5QB * (R2_imag - R3_imag) + C5QD * (R1_imag - R4_imag) + C5QA * ((R2_real - R1_real) + (R3_real - R4_real))
        TR3 = (R0_real - C5QC * (R1_real + R4_real)) + C5QB * (R2_imag - R3_imag) - C5QD * (R1_imag - R4_imag) + C5QA * ((R2_real - R1_real) + (R3_real - R4_real))

        TI0 = R0_imag + R1_imag + R2_imag + R3_imag + R4_imag
        TI1 = (R0_imag - C5QC * (R2_imag + R3_imag)) - C5QB * (R1_real - R4_real) - C5QD * (R2_real - R3_real) + C5QA * ((R1_imag - R2_imag) + (R4_imag - R3_imag))
        TI4 = (R0_imag - C5QC * (R2_imag + R3_imag)) + C5QB * (R1_real - R4_real) + C5QD * (R2_real - R3_real) + C5QA * ((R1_imag - R2_imag) + (R4_imag - R3_imag))
        TI2 = (R0_imag - C5QC * (R1_imag + R4_imag)) + C5QB * (R2_real - R3_real) - C5QD * (R1_real - R4_real) + C5QA * ((R2_imag - R1_imag) + (R3_imag - R4_imag))
        TI3 = (R0_imag - C5QC * (R1_imag + R4_imag)) - C5QB * (R2_real - R3_real) + C5QD * (R1_real - R4_real) + C5QA * ((R2_imag - R1_imag) + (R3_imag - R4_imag))

        # Store results
        tl.store(out_real_ptr + idx0, TR0)
        tl.store(out_real_ptr + idx1, TR1)
        tl.store(out_real_ptr + idx2, TR2)
        tl.store(out_real_ptr + idx3, TR3)
        tl.store(out_real_ptr + idx4, TR4)
        tl.store(out_imag_ptr + idx0, TI0)
        tl.store(out_imag_ptr + idx1, TI1)
        tl.store(out_imag_ptr + idx2, TI2)
        tl.store(out_imag_ptr + idx3, TI3)
        tl.store(out_imag_ptr + idx4, TI4)


@triton.jit
def _batch_rad5_kernel_complex_inv(
    x_real_ptr,
    x_imag_ptr,
    out_real_ptr,
    out_imag_ptr,
    n_groups: tl.constexpr,
):
    """
    Batch inverse radix-5 FFT kernel for complex-valued data.
    """
    # Constants defined inside kernel for Triton compatibility
    C5QA = 0.3090169943749474
    C5QB = 0.9510565162951536
    C5QC = 0.5
    C5QD = 0.5877852522924731

    pid = tl.program_id(0)

    if pid < n_groups:
        idx0 = 5 * pid
        idx1 = 5 * pid + 1
        idx2 = 5 * pid + 2
        idx3 = 5 * pid + 3
        idx4 = 5 * pid + 4

        # Load real parts
        R0_real = tl.load(x_real_ptr + idx0)
        R1_real = tl.load(x_real_ptr + idx1)
        R2_real = tl.load(x_real_ptr + idx2)
        R3_real = tl.load(x_real_ptr + idx3)
        R4_real = tl.load(x_real_ptr + idx4)

        # Load imaginary parts
        R0_imag = tl.load(x_imag_ptr + idx0)
        R1_imag = tl.load(x_imag_ptr + idx1)
        R2_imag = tl.load(x_imag_ptr + idx2)
        R3_imag = tl.load(x_imag_ptr + idx3)
        R4_imag = tl.load(x_imag_ptr + idx4)

        # Inverse FFT computation (sign changes)
        TR0 = R0_real + R1_real + R2_real + R3_real + R4_real
        TR1 = (R0_real - C5QC * (R2_real + R3_real)) - C5QB * (R1_imag - R4_imag) - C5QD * (R2_imag - R3_imag) + C5QA * ((R1_real - R2_real) + (R4_real - R3_real))
        TR4 = (R0_real - C5QC * (R2_real + R3_real)) + C5QB * (R1_imag - R4_imag) + C5QD * (R2_imag - R3_imag) + C5QA * ((R1_real - R2_real) + (R4_real - R3_real))
        TR2 = (R0_real - C5QC * (R1_real + R4_real)) + C5QB * (R2_imag - R3_imag) - C5QD * (R1_imag - R4_imag) + C5QA * ((R2_real - R1_real) + (R3_real - R4_real))
        TR3 = (R0_real - C5QC * (R1_real + R4_real)) - C5QB * (R2_imag - R3_imag) + C5QD * (R1_imag - R4_imag) + C5QA * ((R2_real - R1_real) + (R3_real - R4_real))

        TI0 = R0_imag + R1_imag + R2_imag + R3_imag + R4_imag
        TI1 = (R0_imag - C5QC * (R2_imag + R3_imag)) + C5QB * (R1_real - R4_real) + C5QD * (R2_real - R3_real) + C5QA * ((R1_imag - R2_imag) + (R4_imag - R3_imag))
        TI4 = (R0_imag - C5QC * (R2_imag + R3_imag)) - C5QB * (R1_real - R4_real) - C5QD * (R2_real - R3_real) + C5QA * ((R1_imag - R2_imag) + (R4_imag - R3_imag))
        TI2 = (R0_imag - C5QC * (R1_imag + R4_imag)) - C5QB * (R2_real - R3_real) + C5QD * (R1_real - R4_real) + C5QA * ((R2_imag - R1_imag) + (R3_imag - R4_imag))
        TI3 = (R0_imag - C5QC * (R1_imag + R4_imag)) + C5QB * (R2_real - R3_real) - C5QD * (R1_real - R4_real) + C5QA * ((R2_imag - R1_imag) + (R3_imag - R4_imag))

        # Store results
        tl.store(out_real_ptr + idx0, TR0)
        tl.store(out_real_ptr + idx1, TR1)
        tl.store(out_real_ptr + idx2, TR2)
        tl.store(out_real_ptr + idx3, TR3)
        tl.store(out_real_ptr + idx4, TR4)
        tl.store(out_imag_ptr + idx0, TI0)
        tl.store(out_imag_ptr + idx1, TI1)
        tl.store(out_imag_ptr + idx2, TI2)
        tl.store(out_imag_ptr + idx3, TI3)
        tl.store(out_imag_ptr + idx4, TI4)


def batch_fwd_rad5_b1(x: torch.Tensor) -> torch.Tensor:
    """
    Batch forward radix-5 FFT using Triton kernels.

    Args:
        x: Input tensor of shape (..., 5) for groups of 5 complex values to transform.

    Returns:
        Output tensor with the same shape as input.
    """
    x = x.contiguous()
    original_shape = x.shape

    if torch.is_complex(x):
        x_flat = x.view(-1)
        n_groups = x_flat.shape[0] // 5

        x_real = x_flat.real.contiguous()
        x_imag = x_flat.imag.contiguous()

        out_real = torch.empty_like(x_real)
        out_imag = torch.empty_like(x_imag)

        grid = lambda meta: (n_groups,)
        _batch_rad5_kernel_complex_fwd[grid](x_real, x_imag, out_real, out_imag, n_groups)

        out = torch.complex(out_real, out_imag)
        return out.view(original_shape)
    else:
        raise ValueError("Radix-5 FFT requires complex input")


def batch_inv_rad5_b1(x: torch.Tensor) -> torch.Tensor:
    """
    Batch inverse radix-5 FFT using Triton kernels.

    Args:
        x: Input tensor of shape (..., 5) for groups of 5 complex values to transform.

    Returns:
        Output tensor with the same shape as input.
    """
    x = x.contiguous()
    original_shape = x.shape

    if torch.is_complex(x):
        x_flat = x.view(-1)
        n_groups = x_flat.shape[0] // 5

        x_real = x_flat.real.contiguous()
        x_imag = x_flat.imag.contiguous()

        out_real = torch.empty_like(x_real)
        out_imag = torch.empty_like(x_imag)

        grid = lambda meta: (n_groups,)
        _batch_rad5_kernel_complex_inv[grid](x_real, x_imag, out_real, out_imag, n_groups)

        out = torch.complex(out_real, out_imag)
        return out.view(original_shape)
    else:
        raise ValueError("Radix-5 FFT requires complex input")


# Aliases for public API
triton_fwd_rad5_b1 = batch_fwd_rad5_b1
triton_inv_rad5_b1 = batch_inv_rad5_b1
