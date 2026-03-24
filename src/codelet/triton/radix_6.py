"""
Triton implementation of radix-6 FFT codelet.

This module provides forward and inverse radix-6 FFT butterfly operations.
Ported from CUDA radix_6.h

Radix-6 is implemented as a combination of radix-2 and radix-3.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _batch_rad6_kernel_complex_fwd(
    x_real_ptr,
    x_imag_ptr,
    out_real_ptr,
    out_imag_ptr,
    n_groups: tl.constexpr,
):
    """
    Batch forward radix-6 FFT kernel for complex-valued data.
    """
    # Constants defined inside kernel for Triton compatibility
    C3QA = 0.5  # cos(2*pi/3) - 0.5
    C3QB = 0.8660254037844386  # sin(2*pi/3)

    pid = tl.program_id(0)

    if pid < n_groups:
        idx0 = 6 * pid
        idx1 = 6 * pid + 1
        idx2 = 6 * pid + 2
        idx3 = 6 * pid + 3
        idx4 = 6 * pid + 4
        idx5 = 6 * pid + 5

        # Load real parts
        R0_real = tl.load(x_real_ptr + idx0)
        R1_real = tl.load(x_real_ptr + idx1)
        R2_real = tl.load(x_real_ptr + idx2)
        R3_real = tl.load(x_real_ptr + idx3)
        R4_real = tl.load(x_real_ptr + idx4)
        R5_real = tl.load(x_real_ptr + idx5)

        # Load imaginary parts
        R0_imag = tl.load(x_imag_ptr + idx0)
        R1_imag = tl.load(x_imag_ptr + idx1)
        R2_imag = tl.load(x_imag_ptr + idx2)
        R3_imag = tl.load(x_imag_ptr + idx3)
        R4_imag = tl.load(x_imag_ptr + idx4)
        R5_imag = tl.load(x_imag_ptr + idx5)

        # First radix-3 on even indices (0, 2, 4)
        TR0 = R0_real + R2_real + R4_real
        TR2 = (R0_real - C3QA * (R2_real + R4_real)) + C3QB * (R2_imag - R4_imag)
        TR4 = (R0_real - C3QA * (R2_real + R4_real)) - C3QB * (R2_imag - R4_imag)

        TI0 = R0_imag + R2_imag + R4_imag
        TI2 = (R0_imag - C3QA * (R2_imag + R4_imag)) - C3QB * (R2_real - R4_real)
        TI4 = (R0_imag - C3QA * (R2_imag + R4_imag)) + C3QB * (R2_real - R4_real)

        # First radix-3 on odd indices (1, 3, 5)
        TR1 = R1_real + R3_real + R5_real
        TR3 = (R1_real - C3QA * (R3_real + R5_real)) + C3QB * (R3_imag - R5_imag)
        TR5 = (R1_real - C3QA * (R3_real + R5_real)) - C3QB * (R3_imag - R5_imag)

        TI1 = R1_imag + R3_imag + R5_imag
        TI3 = (R1_imag - C3QA * (R3_imag + R5_imag)) - C3QB * (R3_real - R5_real)
        TI5 = (R1_imag - C3QA * (R3_imag + R5_imag)) + C3QB * (R3_real - R5_real)

        # Final radix-2 combination with twiddle factors
        # R0 = TR0 + TR1
        # R1 = TR2 + (C3QA * TR3 + C3QB * TI3)
        # R2 = TR4 + (-C3QA * TR5 + C3QB * TI5)
        # R3 = TR0 - TR1
        # R4 = TR2 - (C3QA * TR3 + C3QB * TI3)
        # R5 = TR4 - (-C3QA * TR5 + C3QB * TI5)

        out_R0_real = TR0 + TR1
        out_R0_imag = TI0 + TI1

        out_R1_real = TR2 + (C3QA * TR3 + C3QB * TI3)
        out_R1_imag = TI2 + (-C3QB * TR3 + C3QA * TI3)

        out_R2_real = TR4 + (-C3QA * TR5 + C3QB * TI5)
        out_R2_imag = TI4 + (-C3QB * TR5 - C3QA * TI5)

        out_R3_real = TR0 - TR1
        out_R3_imag = TI0 - TI1

        out_R4_real = TR2 - (C3QA * TR3 + C3QB * TI3)
        out_R4_imag = TI2 - (-C3QB * TR3 + C3QA * TI3)

        out_R5_real = TR4 - (-C3QA * TR5 + C3QB * TI5)
        out_R5_imag = TI4 - (-C3QB * TR5 - C3QA * TI5)

        # Store results
        tl.store(out_real_ptr + idx0, out_R0_real)
        tl.store(out_real_ptr + idx1, out_R1_real)
        tl.store(out_real_ptr + idx2, out_R2_real)
        tl.store(out_real_ptr + idx3, out_R3_real)
        tl.store(out_real_ptr + idx4, out_R4_real)
        tl.store(out_real_ptr + idx5, out_R5_real)
        tl.store(out_imag_ptr + idx0, out_R0_imag)
        tl.store(out_imag_ptr + idx1, out_R1_imag)
        tl.store(out_imag_ptr + idx2, out_R2_imag)
        tl.store(out_imag_ptr + idx3, out_R3_imag)
        tl.store(out_imag_ptr + idx4, out_R4_imag)
        tl.store(out_imag_ptr + idx5, out_R5_imag)


@triton.jit
def _batch_rad6_kernel_complex_inv(
    x_real_ptr,
    x_imag_ptr,
    out_real_ptr,
    out_imag_ptr,
    n_groups: tl.constexpr,
):
    """
    Batch inverse radix-6 FFT kernel for complex-valued data.
    """
    # Constants defined inside kernel for Triton compatibility
    C3QA = 0.5  # cos(2*pi/3) - 0.5
    C3QB = 0.8660254037844386  # sin(2*pi/3)

    pid = tl.program_id(0)

    if pid < n_groups:
        idx0 = 6 * pid
        idx1 = 6 * pid + 1
        idx2 = 6 * pid + 2
        idx3 = 6 * pid + 3
        idx4 = 6 * pid + 4
        idx5 = 6 * pid + 5

        # Load real parts
        R0_real = tl.load(x_real_ptr + idx0)
        R1_real = tl.load(x_real_ptr + idx1)
        R2_real = tl.load(x_real_ptr + idx2)
        R3_real = tl.load(x_real_ptr + idx3)
        R4_real = tl.load(x_real_ptr + idx4)
        R5_real = tl.load(x_real_ptr + idx5)

        # Load imaginary parts
        R0_imag = tl.load(x_imag_ptr + idx0)
        R1_imag = tl.load(x_imag_ptr + idx1)
        R2_imag = tl.load(x_imag_ptr + idx2)
        R3_imag = tl.load(x_imag_ptr + idx3)
        R4_imag = tl.load(x_imag_ptr + idx4)
        R5_imag = tl.load(x_imag_ptr + idx5)

        # First radix-3 on even indices (0, 2, 4) - inverse
        TR0 = R0_real + R2_real + R4_real
        TR2 = (R0_real - C3QA * (R2_real + R4_real)) - C3QB * (R2_imag - R4_imag)
        TR4 = (R0_real - C3QA * (R2_real + R4_real)) + C3QB * (R2_imag - R4_imag)

        TI0 = R0_imag + R2_imag + R4_imag
        TI2 = (R0_imag - C3QA * (R2_imag + R4_imag)) + C3QB * (R2_real - R4_real)
        TI4 = (R0_imag - C3QA * (R2_imag + R4_imag)) - C3QB * (R2_real - R4_real)

        # First radix-3 on odd indices (1, 3, 5) - inverse
        TR1 = R1_real + R3_real + R5_real
        TR3 = (R1_real - C3QA * (R3_real + R5_real)) - C3QB * (R3_imag - R5_imag)
        TR5 = (R1_real - C3QA * (R3_real + R5_real)) + C3QB * (R3_imag - R5_imag)

        TI1 = R1_imag + R3_imag + R5_imag
        TI3 = (R1_imag - C3QA * (R3_imag + R5_imag)) + C3QB * (R3_real - R5_real)
        TI5 = (R1_imag - C3QA * (R3_imag + R5_imag)) - C3QB * (R3_real - R5_real)

        # Final radix-2 combination with twiddle factors - inverse
        out_R0_real = TR0 + TR1
        out_R0_imag = TI0 + TI1

        out_R1_real = TR2 + (C3QA * TR3 - C3QB * TI3)
        out_R1_imag = TI2 + (C3QB * TR3 + C3QA * TI3)

        out_R2_real = TR4 + (-C3QA * TR5 - C3QB * TI5)
        out_R2_imag = TI4 + (C3QB * TR5 - C3QA * TI5)

        out_R3_real = TR0 - TR1
        out_R3_imag = TI0 - TI1

        out_R4_real = TR2 - (C3QA * TR3 - C3QB * TI3)
        out_R4_imag = TI2 - (C3QB * TR3 + C3QA * TI3)

        out_R5_real = TR4 - (-C3QA * TR5 - C3QB * TI5)
        out_R5_imag = TI4 - (C3QB * TR5 - C3QA * TI5)

        # Store results
        tl.store(out_real_ptr + idx0, out_R0_real)
        tl.store(out_real_ptr + idx1, out_R1_real)
        tl.store(out_real_ptr + idx2, out_R2_real)
        tl.store(out_real_ptr + idx3, out_R3_real)
        tl.store(out_real_ptr + idx4, out_R4_real)
        tl.store(out_real_ptr + idx5, out_R5_real)
        tl.store(out_imag_ptr + idx0, out_R0_imag)
        tl.store(out_imag_ptr + idx1, out_R1_imag)
        tl.store(out_imag_ptr + idx2, out_R2_imag)
        tl.store(out_imag_ptr + idx3, out_R3_imag)
        tl.store(out_imag_ptr + idx4, out_R4_imag)
        tl.store(out_imag_ptr + idx5, out_R5_imag)


def batch_fwd_rad6_b1(x: torch.Tensor) -> torch.Tensor:
    """
    Batch forward radix-6 FFT using Triton kernels.

    Args:
        x: Input tensor of shape (..., 6) for groups of 6 complex values to transform.

    Returns:
        Output tensor with the same shape as input.
    """
    x = x.contiguous()
    original_shape = x.shape

    if torch.is_complex(x):
        x_flat = x.view(-1)
        n_groups = x_flat.shape[0] // 6

        x_real = x_flat.real.contiguous()
        x_imag = x_flat.imag.contiguous()

        out_real = torch.empty_like(x_real)
        out_imag = torch.empty_like(x_imag)

        grid = lambda meta: (n_groups,)
        _batch_rad6_kernel_complex_fwd[grid](x_real, x_imag, out_real, out_imag, n_groups)

        out = torch.complex(out_real, out_imag)
        return out.view(original_shape)
    else:
        raise ValueError("Radix-6 FFT requires complex input")


def batch_inv_rad6_b1(x: torch.Tensor) -> torch.Tensor:
    """
    Batch inverse radix-6 FFT using Triton kernels.

    Args:
        x: Input tensor of shape (..., 6) for groups of 6 complex values to transform.

    Returns:
        Output tensor with the same shape as input.
    """
    x = x.contiguous()
    original_shape = x.shape

    if torch.is_complex(x):
        x_flat = x.view(-1)
        n_groups = x_flat.shape[0] // 6

        x_real = x_flat.real.contiguous()
        x_imag = x_flat.imag.contiguous()

        out_real = torch.empty_like(x_real)
        out_imag = torch.empty_like(x_imag)

        grid = lambda meta: (n_groups,)
        _batch_rad6_kernel_complex_inv[grid](x_real, x_imag, out_real, out_imag, n_groups)

        out = torch.complex(out_real, out_imag)
        return out.view(original_shape)
    else:
        raise ValueError("Radix-6 FFT requires complex input")


# Aliases for public API
triton_fwd_rad6_b1 = batch_fwd_rad6_b1
triton_inv_rad6_b1 = batch_inv_rad6_b1
