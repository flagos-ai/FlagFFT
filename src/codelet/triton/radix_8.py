"""
Triton implementation of radix-8 FFT codelet.

This module provides forward and inverse radix-8 FFT butterfly operations.
Ported from CUDA radix_8.h
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _batch_rad8_kernel_complex_fwd(
    x_real_ptr,
    x_imag_ptr,
    out_real_ptr,
    out_imag_ptr,
    n_groups: tl.constexpr,
):
    """
    Batch forward radix-8 FFT kernel for complex-valued data.
    """
    # Constants defined inside kernel for Triton compatibility
    C8Q = 0.7071067811865475  # 1/sqrt(2)

    pid = tl.program_id(0)

    if pid < n_groups:
        # Load indices
        idx0 = 8 * pid
        idx1 = 8 * pid + 1
        idx2 = 8 * pid + 2
        idx3 = 8 * pid + 3
        idx4 = 8 * pid + 4
        idx5 = 8 * pid + 5
        idx6 = 8 * pid + 6
        idx7 = 8 * pid + 7

        # Load real parts
        R0_real = tl.load(x_real_ptr + idx0)
        R1_real = tl.load(x_real_ptr + idx1)
        R2_real = tl.load(x_real_ptr + idx2)
        R3_real = tl.load(x_real_ptr + idx3)
        R4_real = tl.load(x_real_ptr + idx4)
        R5_real = tl.load(x_real_ptr + idx5)
        R6_real = tl.load(x_real_ptr + idx6)
        R7_real = tl.load(x_real_ptr + idx7)

        # Load imaginary parts
        R0_imag = tl.load(x_imag_ptr + idx0)
        R1_imag = tl.load(x_imag_ptr + idx1)
        R2_imag = tl.load(x_imag_ptr + idx2)
        R3_imag = tl.load(x_imag_ptr + idx3)
        R4_imag = tl.load(x_imag_ptr + idx4)
        R5_imag = tl.load(x_imag_ptr + idx5)
        R6_imag = tl.load(x_imag_ptr + idx6)
        R7_imag = tl.load(x_imag_ptr + idx7)

        # Stage 1: First set of butterflies (radix-2 style)
        # R1 = R0 - R1, R0 = R0 + R1
        new_R1_real = R0_real - R1_real
        new_R0_real = 2.0 * R0_real - new_R1_real
        new_R1_imag = R0_imag - R1_imag
        new_R0_imag = 2.0 * R0_imag - new_R1_imag

        new_R3_real = R2_real - R3_real
        new_R2_real = 2.0 * R2_real - new_R3_real
        new_R3_imag = R2_imag - R3_imag
        new_R2_imag = 2.0 * R2_imag - new_R3_imag

        new_R5_real = R4_real - R5_real
        new_R4_real = 2.0 * R4_real - new_R5_real
        new_R5_imag = R4_imag - R5_imag
        new_R4_imag = 2.0 * R4_imag - new_R5_imag

        new_R7_real = R6_real - R7_real
        new_R6_real = 2.0 * R6_real - new_R7_real
        new_R7_imag = R6_imag - R7_imag
        new_R6_imag = 2.0 * R6_imag - new_R7_imag

        # Stage 2: Second set with rotations
        s2_R2_real = new_R0_real - new_R2_real
        s2_R0_real = 2.0 * new_R0_real - s2_R2_real
        s2_R2_imag = new_R0_imag - new_R2_imag
        s2_R0_imag = 2.0 * new_R0_imag - s2_R2_imag

        # R3 = R1 + (-R3_imag, R3_real) for forward
        rot_R3_real = -new_R3_imag
        rot_R3_imag = new_R3_real
        s2_R3_real = new_R1_real + rot_R3_real
        s2_R3_imag = new_R1_imag + rot_R3_imag
        s2_R1_real = 2.0 * new_R1_real - s2_R3_real
        s2_R1_imag = 2.0 * new_R1_imag - s2_R3_imag

        s2_R6_real = new_R4_real - new_R6_real
        s2_R4_real = 2.0 * new_R4_real - s2_R6_real
        s2_R6_imag = new_R4_imag - new_R6_imag
        s2_R4_imag = 2.0 * new_R4_imag - s2_R6_imag

        rot_R7_real = -new_R7_imag
        rot_R7_imag = new_R7_real
        s2_R7_real = new_R5_real + rot_R7_real
        s2_R7_imag = new_R5_imag + rot_R7_imag
        s2_R5_real = 2.0 * new_R5_real - s2_R7_real
        s2_R5_imag = 2.0 * new_R5_imag - s2_R7_imag

        # Stage 3: Third set with C8Q twiddle factors
        s3_R4_real = s2_R0_real - s2_R4_real
        s3_R0_real = 2.0 * s2_R0_real - s3_R4_real
        s3_R4_imag = s2_R0_imag - s2_R4_imag
        s3_R0_imag = 2.0 * s2_R0_imag - s3_R4_imag

        # R5 = (R1 - C8Q * R5) - C8Q * (R5_imag, -R5_real)
        s3_R5_real = (s2_R1_real - C8Q * s2_R5_real) - C8Q * s2_R5_imag
        s3_R5_imag = (s2_R1_imag - C8Q * s2_R5_imag) + C8Q * s2_R5_real
        s3_R1_real = 2.0 * s2_R1_real - s3_R5_real
        s3_R1_imag = 2.0 * s2_R1_imag - s3_R5_imag

        # R6 = R2 + (-R6_imag, R6_real)
        rot_R6_real = -s2_R6_imag
        rot_R6_imag = s2_R6_real
        s3_R6_real = s2_R2_real + rot_R6_real
        s3_R6_imag = s2_R2_imag + rot_R6_imag
        s3_R2_real = 2.0 * s2_R2_real - s3_R6_real
        s3_R2_imag = 2.0 * s2_R2_imag - s3_R6_imag

        # R7 = (R3 + C8Q * R7) - C8Q * (R7_imag, -R7_real)
        s3_R7_real = (s2_R3_real + C8Q * s2_R7_real) - C8Q * s2_R7_imag
        s3_R7_imag = (s2_R3_imag + C8Q * s2_R7_imag) + C8Q * s2_R7_real
        s3_R3_real = 2.0 * s2_R3_real - s3_R7_real
        s3_R3_imag = 2.0 * s2_R3_imag - s3_R7_imag

        # Final output with swapping for bit-reversed order
        tl.store(out_real_ptr + idx0, s3_R0_real)
        tl.store(out_real_ptr + idx1, s3_R4_real)
        tl.store(out_real_ptr + idx2, s3_R2_real)
        tl.store(out_real_ptr + idx3, s3_R6_real)
        tl.store(out_real_ptr + idx4, s3_R1_real)
        tl.store(out_real_ptr + idx5, s3_R5_real)
        tl.store(out_real_ptr + idx6, s3_R3_real)
        tl.store(out_real_ptr + idx7, s3_R7_real)
        tl.store(out_imag_ptr + idx0, s3_R0_imag)
        tl.store(out_imag_ptr + idx1, s3_R4_imag)
        tl.store(out_imag_ptr + idx2, s3_R2_imag)
        tl.store(out_imag_ptr + idx3, s3_R6_imag)
        tl.store(out_imag_ptr + idx4, s3_R1_imag)
        tl.store(out_imag_ptr + idx5, s3_R5_imag)
        tl.store(out_imag_ptr + idx6, s3_R3_imag)
        tl.store(out_imag_ptr + idx7, s3_R7_imag)


@triton.jit
def _batch_rad8_kernel_complex_inv(
    x_real_ptr,
    x_imag_ptr,
    out_real_ptr,
    out_imag_ptr,
    n_groups: tl.constexpr,
):
    """
    Batch inverse radix-8 FFT kernel for complex-valued data.
    """
    # Constants defined inside kernel for Triton compatibility
    C8Q = 0.7071067811865475  # 1/sqrt(2)

    pid = tl.program_id(0)

    if pid < n_groups:
        idx0 = 8 * pid
        idx1 = 8 * pid + 1
        idx2 = 8 * pid + 2
        idx3 = 8 * pid + 3
        idx4 = 8 * pid + 4
        idx5 = 8 * pid + 5
        idx6 = 8 * pid + 6
        idx7 = 8 * pid + 7

        # Load real parts
        R0_real = tl.load(x_real_ptr + idx0)
        R1_real = tl.load(x_real_ptr + idx1)
        R2_real = tl.load(x_real_ptr + idx2)
        R3_real = tl.load(x_real_ptr + idx3)
        R4_real = tl.load(x_real_ptr + idx4)
        R5_real = tl.load(x_real_ptr + idx5)
        R6_real = tl.load(x_real_ptr + idx6)
        R7_real = tl.load(x_real_ptr + idx7)

        # Load imaginary parts
        R0_imag = tl.load(x_imag_ptr + idx0)
        R1_imag = tl.load(x_imag_ptr + idx1)
        R2_imag = tl.load(x_imag_ptr + idx2)
        R3_imag = tl.load(x_imag_ptr + idx3)
        R4_imag = tl.load(x_imag_ptr + idx4)
        R5_imag = tl.load(x_imag_ptr + idx5)
        R6_imag = tl.load(x_imag_ptr + idx6)
        R7_imag = tl.load(x_imag_ptr + idx7)

        # Stage 1: First set of butterflies
        new_R1_real = R0_real - R1_real
        new_R0_real = 2.0 * R0_real - new_R1_real
        new_R1_imag = R0_imag - R1_imag
        new_R0_imag = 2.0 * R0_imag - new_R1_imag

        new_R3_real = R2_real - R3_real
        new_R2_real = 2.0 * R2_real - new_R3_real
        new_R3_imag = R2_imag - R3_imag
        new_R2_imag = 2.0 * R2_imag - new_R3_imag

        new_R5_real = R4_real - R5_real
        new_R4_real = 2.0 * R4_real - new_R5_real
        new_R5_imag = R4_imag - R5_imag
        new_R4_imag = 2.0 * R4_imag - new_R5_imag

        new_R7_real = R6_real - R7_real
        new_R6_real = 2.0 * R6_real - new_R7_real
        new_R7_imag = R6_imag - R7_imag
        new_R6_imag = 2.0 * R6_imag - new_R7_imag

        # Stage 2: Second set with inverse rotations
        s2_R2_real = new_R0_real - new_R2_real
        s2_R0_real = 2.0 * new_R0_real - s2_R2_real
        s2_R2_imag = new_R0_imag - new_R2_imag
        s2_R0_imag = 2.0 * new_R0_imag - s2_R2_imag

        # Inverse: rotation is (R3_imag, -R3_real)
        rot_R3_real = new_R3_imag
        rot_R3_imag = -new_R3_real
        s2_R3_real = new_R1_real + rot_R3_real
        s2_R3_imag = new_R1_imag + rot_R3_imag
        s2_R1_real = 2.0 * new_R1_real - s2_R3_real
        s2_R1_imag = 2.0 * new_R1_imag - s2_R3_imag

        s2_R6_real = new_R4_real - new_R6_real
        s2_R4_real = 2.0 * new_R4_real - s2_R6_real
        s2_R6_imag = new_R4_imag - new_R6_imag
        s2_R4_imag = 2.0 * new_R4_imag - s2_R6_imag

        rot_R7_real = new_R7_imag
        rot_R7_imag = -new_R7_real
        s2_R7_real = new_R5_real + rot_R7_real
        s2_R7_imag = new_R5_imag + rot_R7_imag
        s2_R5_real = 2.0 * new_R5_real - s2_R7_real
        s2_R5_imag = 2.0 * new_R5_imag - s2_R7_imag

        # Stage 3: Third set with inverse twiddle factors
        s3_R4_real = s2_R0_real - s2_R4_real
        s3_R0_real = 2.0 * s2_R0_real - s3_R4_real
        s3_R4_imag = s2_R0_imag - s2_R4_imag
        s3_R0_imag = 2.0 * s2_R0_imag - s3_R4_imag

        # Inverse: different sign on C8Q terms
        s3_R5_real = (s2_R1_real - C8Q * s2_R5_real) + C8Q * s2_R5_imag
        s3_R5_imag = (s2_R1_imag - C8Q * s2_R5_imag) - C8Q * s2_R5_real
        s3_R1_real = 2.0 * s2_R1_real - s3_R5_real
        s3_R1_imag = 2.0 * s2_R1_imag - s3_R5_imag

        rot_R6_real = s2_R6_imag
        rot_R6_imag = -s2_R6_real
        s3_R6_real = s2_R2_real + rot_R6_real
        s3_R6_imag = s2_R2_imag + rot_R6_imag
        s3_R2_real = 2.0 * s2_R2_real - s3_R6_real
        s3_R2_imag = 2.0 * s2_R2_imag - s3_R6_imag

        s3_R7_real = (s2_R3_real + C8Q * s2_R7_real) + C8Q * s2_R7_imag
        s3_R7_imag = (s2_R3_imag + C8Q * s2_R7_imag) - C8Q * s2_R7_real
        s3_R3_real = 2.0 * s2_R3_real - s3_R7_real
        s3_R3_imag = 2.0 * s2_R3_imag - s3_R7_imag

        # Final output with swapping
        tl.store(out_real_ptr + idx0, s3_R0_real)
        tl.store(out_real_ptr + idx1, s3_R4_real)
        tl.store(out_real_ptr + idx2, s3_R2_real)
        tl.store(out_real_ptr + idx3, s3_R6_real)
        tl.store(out_real_ptr + idx4, s3_R1_real)
        tl.store(out_real_ptr + idx5, s3_R5_real)
        tl.store(out_real_ptr + idx6, s3_R3_real)
        tl.store(out_real_ptr + idx7, s3_R7_real)
        tl.store(out_imag_ptr + idx0, s3_R0_imag)
        tl.store(out_imag_ptr + idx1, s3_R4_imag)
        tl.store(out_imag_ptr + idx2, s3_R2_imag)
        tl.store(out_imag_ptr + idx3, s3_R6_imag)
        tl.store(out_imag_ptr + idx4, s3_R1_imag)
        tl.store(out_imag_ptr + idx5, s3_R5_imag)
        tl.store(out_imag_ptr + idx6, s3_R3_imag)
        tl.store(out_imag_ptr + idx7, s3_R7_imag)


def batch_fwd_rad8_b1(x: torch.Tensor) -> torch.Tensor:
    """
    Batch forward radix-8 FFT using Triton kernels.

    Args:
        x: Input tensor of shape (..., 8) for groups of 8 complex values to transform.

    Returns:
        Output tensor with the same shape as input.
    """
    x = x.contiguous()
    original_shape = x.shape

    if torch.is_complex(x):
        x_flat = x.view(-1)
        n_groups = x_flat.shape[0] // 8

        x_real = x_flat.real.contiguous()
        x_imag = x_flat.imag.contiguous()

        out_real = torch.empty_like(x_real)
        out_imag = torch.empty_like(x_imag)

        grid = lambda meta: (n_groups,)
        _batch_rad8_kernel_complex_fwd[grid](x_real, x_imag, out_real, out_imag, n_groups)

        out = torch.complex(out_real, out_imag)
        return out.view(original_shape)
    else:
        raise ValueError("Radix-8 FFT requires complex input")


def batch_inv_rad8_b1(x: torch.Tensor) -> torch.Tensor:
    """
    Batch inverse radix-8 FFT using Triton kernels.

    Args:
        x: Input tensor of shape (..., 8) for groups of 8 complex values to transform.

    Returns:
        Output tensor with the same shape as input.
    """
    x = x.contiguous()
    original_shape = x.shape

    if torch.is_complex(x):
        x_flat = x.view(-1)
        n_groups = x_flat.shape[0] // 8

        x_real = x_flat.real.contiguous()
        x_imag = x_flat.imag.contiguous()

        out_real = torch.empty_like(x_real)
        out_imag = torch.empty_like(x_imag)

        grid = lambda meta: (n_groups,)
        _batch_rad8_kernel_complex_inv[grid](x_real, x_imag, out_real, out_imag, n_groups)

        out = torch.complex(out_real, out_imag)
        return out.view(original_shape)
    else:
        raise ValueError("Radix-8 FFT requires complex input")


# Aliases for public API
triton_fwd_rad8_b1 = batch_fwd_rad8_b1
triton_inv_rad8_b1 = batch_inv_rad8_b1
