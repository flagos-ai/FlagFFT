"""
Triton implementation of radix-4 FFT codelet.

This module provides forward and inverse radix-4 FFT butterfly operations.
Ported from CUDA radix_4.h
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _batch_rad4_kernel_complex_fwd(
    x_real_ptr,
    x_imag_ptr,
    out_real_ptr,
    out_imag_ptr,
    n_groups: tl.constexpr,
):
    """
    Batch forward radix-4 FFT kernel for complex-valued data.

    The 4-point FFT butterfly operation.
    """
    pid = tl.program_id(0)

    if pid < n_groups:
        idx0 = 4 * pid
        idx1 = 4 * pid + 1
        idx2 = 4 * pid + 2
        idx3 = 4 * pid + 3

        # Load real parts
        R0_real = tl.load(x_real_ptr + idx0)
        R1_real = tl.load(x_real_ptr + idx1)
        R2_real = tl.load(x_real_ptr + idx2)
        R3_real = tl.load(x_real_ptr + idx3)

        # Load imaginary parts
        R0_imag = tl.load(x_imag_ptr + idx0)
        R1_imag = tl.load(x_imag_ptr + idx1)
        R2_imag = tl.load(x_imag_ptr + idx2)
        R3_imag = tl.load(x_imag_ptr + idx3)

        # Stage 1: First set of butterflies
        # R1 = R0 - R1, R0 = 2*R0 - R1 = R0 + R1
        new_R1_real = R0_real - R1_real
        new_R0_real = 2.0 * R0_real - new_R1_real
        new_R1_imag = R0_imag - R1_imag
        new_R0_imag = 2.0 * R0_imag - new_R1_imag

        # R3 = R2 - R3, R2 = 2*R2 - R3 = R2 + R3
        new_R3_real = R2_real - R3_real
        new_R2_real = 2.0 * R2_real - new_R3_real
        new_R3_imag = R2_imag - R3_imag
        new_R2_imag = 2.0 * R2_imag - new_R3_imag

        # Stage 2: Second set of butterflies
        # R2 = R0 - R2, R0 = 2*R0 - R2 = R0 + R2
        out_R2_real = new_R0_real - new_R2_real
        out_R0_real = 2.0 * new_R0_real - out_R2_real
        out_R2_imag = new_R0_imag - new_R2_imag
        out_R0_imag = 2.0 * new_R0_imag - out_R2_imag

        # R3 = R1 + (-R3_imag, R3_real) [90 degree rotation], R1 = 2*R1 - R3
        # Forward: rotation is (-y, x)
        rot_R3_real = -new_R3_imag
        rot_R3_imag = new_R3_real
        temp_R3_real = new_R1_real + rot_R3_real
        temp_R3_imag = new_R1_imag + rot_R3_imag
        out_R1_real = 2.0 * new_R1_real - temp_R3_real
        out_R1_imag = 2.0 * new_R1_imag - temp_R3_imag
        out_R3_real = temp_R3_real
        out_R3_imag = temp_R3_imag

        # Swap R1 and R2 for output order
        tl.store(out_real_ptr + idx0, out_R0_real)
        tl.store(out_real_ptr + idx1, out_R2_real)
        tl.store(out_real_ptr + idx2, out_R1_real)
        tl.store(out_real_ptr + idx3, out_R3_real)
        tl.store(out_imag_ptr + idx0, out_R0_imag)
        tl.store(out_imag_ptr + idx1, out_R2_imag)
        tl.store(out_imag_ptr + idx2, out_R1_imag)
        tl.store(out_imag_ptr + idx3, out_R3_imag)


@triton.jit
def _batch_rad4_kernel_complex_inv(
    x_real_ptr,
    x_imag_ptr,
    out_real_ptr,
    out_imag_ptr,
    n_groups: tl.constexpr,
):
    """
    Batch inverse radix-4 FFT kernel for complex-valued data.
    """
    pid = tl.program_id(0)

    if pid < n_groups:
        idx0 = 4 * pid
        idx1 = 4 * pid + 1
        idx2 = 4 * pid + 2
        idx3 = 4 * pid + 3

        # Load real parts
        R0_real = tl.load(x_real_ptr + idx0)
        R1_real = tl.load(x_real_ptr + idx1)
        R2_real = tl.load(x_real_ptr + idx2)
        R3_real = tl.load(x_real_ptr + idx3)

        # Load imaginary parts
        R0_imag = tl.load(x_imag_ptr + idx0)
        R1_imag = tl.load(x_imag_ptr + idx1)
        R2_imag = tl.load(x_imag_ptr + idx2)
        R3_imag = tl.load(x_imag_ptr + idx3)

        # Stage 1: First set of butterflies
        new_R1_real = R0_real - R1_real
        new_R0_real = 2.0 * R0_real - new_R1_real
        new_R1_imag = R0_imag - R1_imag
        new_R0_imag = 2.0 * R0_imag - new_R1_imag

        new_R3_real = R2_real - R3_real
        new_R2_real = 2.0 * R2_real - new_R3_real
        new_R3_imag = R2_imag - R3_imag
        new_R2_imag = 2.0 * R2_imag - new_R3_imag

        # Stage 2: Second set of butterflies
        out_R2_real = new_R0_real - new_R2_real
        out_R0_real = 2.0 * new_R0_real - out_R2_real
        out_R2_imag = new_R0_imag - new_R2_imag
        out_R0_imag = 2.0 * new_R0_imag - out_R2_imag

        # Inverse: rotation is (y, -x)
        rot_R3_real = new_R3_imag
        rot_R3_imag = -new_R3_real
        temp_R3_real = new_R1_real + rot_R3_real
        temp_R3_imag = new_R1_imag + rot_R3_imag
        out_R1_real = 2.0 * new_R1_real - temp_R3_real
        out_R1_imag = 2.0 * new_R1_imag - temp_R3_imag
        out_R3_real = temp_R3_real
        out_R3_imag = temp_R3_imag

        # Swap R1 and R2 for output order
        tl.store(out_real_ptr + idx0, out_R0_real)
        tl.store(out_real_ptr + idx1, out_R2_real)
        tl.store(out_real_ptr + idx2, out_R1_real)
        tl.store(out_real_ptr + idx3, out_R3_real)
        tl.store(out_imag_ptr + idx0, out_R0_imag)
        tl.store(out_imag_ptr + idx1, out_R2_imag)
        tl.store(out_imag_ptr + idx2, out_R1_imag)
        tl.store(out_imag_ptr + idx3, out_R3_imag)


def batch_fwd_rad4_b1(x: torch.Tensor) -> torch.Tensor:
    """
    Batch forward radix-4 FFT using Triton kernels.

    Args:
        x: Input tensor of shape (..., 4) for groups of 4 complex values to transform.

    Returns:
        Output tensor with the same shape as input.
    """
    x = x.contiguous()
    original_shape = x.shape

    if torch.is_complex(x):
        x_flat = x.view(-1)
        n_groups = x_flat.shape[0] // 4

        x_real = x_flat.real.contiguous()
        x_imag = x_flat.imag.contiguous()

        out_real = torch.empty_like(x_real)
        out_imag = torch.empty_like(x_imag)

        grid = lambda meta: (n_groups,)
        _batch_rad4_kernel_complex_fwd[grid](x_real, x_imag, out_real, out_imag, n_groups)

        out = torch.complex(out_real, out_imag)
        return out.view(original_shape)
    else:
        raise ValueError("Radix-4 FFT requires complex input")


def batch_inv_rad4_b1(x: torch.Tensor) -> torch.Tensor:
    """
    Batch inverse radix-4 FFT using Triton kernels.

    Args:
        x: Input tensor of shape (..., 4) for groups of 4 complex values to transform.

    Returns:
        Output tensor with the same shape as input.
    """
    x = x.contiguous()
    original_shape = x.shape

    if torch.is_complex(x):
        x_flat = x.view(-1)
        n_groups = x_flat.shape[0] // 4

        x_real = x_flat.real.contiguous()
        x_imag = x_flat.imag.contiguous()

        out_real = torch.empty_like(x_real)
        out_imag = torch.empty_like(x_imag)

        grid = lambda meta: (n_groups,)
        _batch_rad4_kernel_complex_inv[grid](x_real, x_imag, out_real, out_imag, n_groups)

        out = torch.complex(out_real, out_imag)
        return out.view(original_shape)
    else:
        raise ValueError("Radix-4 FFT requires complex input")


# Aliases for public API
triton_fwd_rad4_b1 = batch_fwd_rad4_b1
triton_inv_rad4_b1 = batch_inv_rad4_b1
