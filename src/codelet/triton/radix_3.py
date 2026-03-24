"""
Triton implementation of radix-3 FFT codelet.

This module provides forward and inverse radix-3 FFT butterfly operations.
Ported from CUDA radix_3.h
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _batch_rad3_kernel_complex(
    x_real_ptr,
    x_imag_ptr,
    out_real_ptr,
    out_imag_ptr,
    n_groups: tl.constexpr,
):
    """
    Batch radix-3 FFT kernel for complex-valued data.

    The 3-point FFT butterfly operation.
    """
    # Constants defined inside kernel for Triton compatibility
    C3QA = 0.5  # cos(2*pi/3) - 0.5
    C3QB = 0.8660254037844386  # sin(2*pi/3)

    pid = tl.program_id(0)

    if pid < n_groups:
        idx0 = 3 * pid
        idx1 = 3 * pid + 1
        idx2 = 3 * pid + 2

        # Load real parts
        R0_real = tl.load(x_real_ptr + idx0)
        R1_real = tl.load(x_real_ptr + idx1)
        R2_real = tl.load(x_real_ptr + idx2)

        # Load imaginary parts
        R0_imag = tl.load(x_imag_ptr + idx0)
        R1_imag = tl.load(x_imag_ptr + idx1)
        R2_imag = tl.load(x_imag_ptr + idx2)

        # Forward FFT computation
        TR0 = R0_real + R1_real + R2_real
        TR1 = (R0_real - C3QA * (R1_real + R2_real)) + C3QB * (R1_imag - R2_imag)
        TR2 = (R0_real - C3QA * (R1_real + R2_real)) - C3QB * (R1_imag - R2_imag)

        TI0 = R0_imag + R1_imag + R2_imag
        TI1 = (R0_imag - C3QA * (R1_imag + R2_imag)) - C3QB * (R1_real - R2_real)
        TI2 = (R0_imag - C3QA * (R1_imag + R2_imag)) + C3QB * (R1_real - R2_real)

        # Store results
        tl.store(out_real_ptr + idx0, TR0)
        tl.store(out_real_ptr + idx1, TR1)
        tl.store(out_real_ptr + idx2, TR2)
        tl.store(out_imag_ptr + idx0, TI0)
        tl.store(out_imag_ptr + idx1, TI1)
        tl.store(out_imag_ptr + idx2, TI2)


@triton.jit
def _batch_rad3_kernel_complex_inv(
    x_real_ptr,
    x_imag_ptr,
    out_real_ptr,
    out_imag_ptr,
    n_groups: tl.constexpr,
):
    """
    Batch inverse radix-3 FFT kernel for complex-valued data.
    """
    # Constants defined inside kernel for Triton compatibility
    C3QA = 0.5  # cos(2*pi/3) - 0.5
    C3QB = 0.8660254037844386  # sin(2*pi/3)

    pid = tl.program_id(0)

    if pid < n_groups:
        idx0 = 3 * pid
        idx1 = 3 * pid + 1
        idx2 = 3 * pid + 2

        # Load real parts
        R0_real = tl.load(x_real_ptr + idx0)
        R1_real = tl.load(x_real_ptr + idx1)
        R2_real = tl.load(x_real_ptr + idx2)

        # Load imaginary parts
        R0_imag = tl.load(x_imag_ptr + idx0)
        R1_imag = tl.load(x_imag_ptr + idx1)
        R2_imag = tl.load(x_imag_ptr + idx2)

        # Inverse FFT computation (sign change on C3QB terms)
        TR0 = R0_real + R1_real + R2_real
        TR1 = (R0_real - C3QA * (R1_real + R2_real)) - C3QB * (R1_imag - R2_imag)
        TR2 = (R0_real - C3QA * (R1_real + R2_real)) + C3QB * (R1_imag - R2_imag)

        TI0 = R0_imag + R1_imag + R2_imag
        TI1 = (R0_imag - C3QA * (R1_imag + R2_imag)) + C3QB * (R1_real - R2_real)
        TI2 = (R0_imag - C3QA * (R1_imag + R2_imag)) - C3QB * (R1_real - R2_real)

        # Store results
        tl.store(out_real_ptr + idx0, TR0)
        tl.store(out_real_ptr + idx1, TR1)
        tl.store(out_real_ptr + idx2, TR2)
        tl.store(out_imag_ptr + idx0, TI0)
        tl.store(out_imag_ptr + idx1, TI1)
        tl.store(out_imag_ptr + idx2, TI2)


def batch_fwd_rad3_b1(x: torch.Tensor) -> torch.Tensor:
    """
    Batch forward radix-3 FFT using Triton kernels.

    Args:
        x: Input tensor of shape (..., 3) for groups of 3 complex values to transform.

    Returns:
        Output tensor with the same shape as input.
    """
    # 只在非连续时才拷贝
    if not x.is_contiguous():
        x = x.contiguous()
    original_shape = x.shape

    if torch.is_complex(x):
        x_flat = x.reshape(-1)
        n_groups = x_flat.shape[0] // 3

        # 使用 view_as_real 避免分别拷贝 real/imag
        # 返回形状 (n, 2)，其中 [..., 0] 是实部，[..., 1] 是虚部
        x_real_view = x_flat.real
        x_imag_view = x_flat.imag

        # 只在需要时才 contiguous
        if not x_real_view.is_contiguous():
            x_real = x_real_view.contiguous()
            x_imag = x_imag_view.contiguous()
        else:
            x_real = x_real_view
            x_imag = x_imag_view

        out_real = torch.empty_like(x_real)
        out_imag = torch.empty_like(x_imag)

        grid = lambda meta: (n_groups,)
        _batch_rad3_kernel_complex[grid](x_real, x_imag, out_real, out_imag, n_groups)

        out = torch.complex(out_real, out_imag)
        return out.reshape(original_shape)
    else:
        raise ValueError("Radix-3 FFT requires complex input")


def batch_inv_rad3_b1(x: torch.Tensor) -> torch.Tensor:
    """
    Batch inverse radix-3 FFT using Triton kernels.

    Args:
        x: Input tensor of shape (..., 3) for groups of 3 complex values to transform.

    Returns:
        Output tensor with the same shape as input.
    """
    # 只在非连续时才拷贝
    if not x.is_contiguous():
        x = x.contiguous()
    original_shape = x.shape

    if torch.is_complex(x):
        x_flat = x.reshape(-1)
        n_groups = x_flat.shape[0] // 3

        x_real_view = x_flat.real
        x_imag_view = x_flat.imag

        if not x_real_view.is_contiguous():
            x_real = x_real_view.contiguous()
            x_imag = x_imag_view.contiguous()
        else:
            x_real = x_real_view
            x_imag = x_imag_view

        out_real = torch.empty_like(x_real)
        out_imag = torch.empty_like(x_imag)

        grid = lambda meta: (n_groups,)
        _batch_rad3_kernel_complex_inv[grid](x_real, x_imag, out_real, out_imag, n_groups)

        out = torch.complex(out_real, out_imag)
        return out.reshape(original_shape)
    else:
        raise ValueError("Radix-3 FFT requires complex input")


# Aliases for public API
triton_fwd_rad3_b1 = batch_fwd_rad3_b1
triton_inv_rad3_b1 = batch_inv_rad3_b1
