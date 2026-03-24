"""
Triton implementation of radix-2 FFT codelet.

This module provides forward and inverse radix-2 FFT butterfly operations.
Triton does not natively support complex types, so complex numbers are handled
by separating real and imaginary parts.

The correct 2-point FFT butterfly operation is:
    [out0, out1] = [in0 + in1, in0 - in1]
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# Real-valued radix-2 kernel
# ============================================================================

@triton.jit
def _batch_rad2_kernel_real(
    x_ptr,
    out_ptr,
    n_pairs: tl.constexpr,
):
    """
    Batch radix-2 FFT kernel for real-valued data.

    Computes the correct butterfly operation:
        out0 = R0 + R1
        out1 = R0 - R1
    """
    pid = tl.program_id(0)

    if pid < n_pairs:
        idx0 = 2 * pid
        idx1 = 2 * pid + 1

        R0 = tl.load(x_ptr + idx0)
        R1 = tl.load(x_ptr + idx1)

        # Correct butterfly operation
        out0 = R0 + R1
        out1 = R0 - R1

        tl.store(out_ptr + idx0, out0)
        tl.store(out_ptr + idx1, out1)


# ============================================================================
# Complex-valued radix-2 kernel (real and imaginary parts separate)
# ============================================================================

@triton.jit
def _batch_rad2_kernel_complex(
    x_real_ptr,
    x_imag_ptr,
    out_real_ptr,
    out_imag_ptr,
    n_pairs: tl.constexpr,
):
    """
    Batch radix-2 FFT kernel for complex-valued data.

    The real and imaginary parts are stored in separate arrays.

    For complex numbers z0 = (a + bi) and z1 = (c + di):
        out0 = z0 + z1 = (a+c) + (b+d)i
        out1 = z0 - z1 = (a-c) + (b-d)i

    This is the same butterfly operation applied independently to real and imag parts.
    """
    pid = tl.program_id(0)

    if pid < n_pairs:
        idx0 = 2 * pid
        idx1 = 2 * pid + 1

        # Load real parts
        R0_real = tl.load(x_real_ptr + idx0)
        R1_real = tl.load(x_real_ptr + idx1)

        # Load imaginary parts
        R0_imag = tl.load(x_imag_ptr + idx0)
        R1_imag = tl.load(x_imag_ptr + idx1)

        # Butterfly on real parts: out0 = R0 + R1, out1 = R0 - R1
        out0_real = R0_real + R1_real
        out1_real = R0_real - R1_real

        # Butterfly on imaginary parts: same operation
        out0_imag = R0_imag + R1_imag
        out1_imag = R0_imag - R1_imag

        # Store results
        tl.store(out_real_ptr + idx0, out0_real)
        tl.store(out_real_ptr + idx1, out1_real)
        tl.store(out_imag_ptr + idx0, out0_imag)
        tl.store(out_imag_ptr + idx1, out1_imag)


# ============================================================================
# Python API functions
# ============================================================================

def batch_fwd_rad2_b1(x: torch.Tensor) -> torch.Tensor:
    """
    Batch forward radix-2 FFT using Triton kernels.

    Supports both real and complex tensors.

    The 2-point FFT computes:
        [out0, out1] = [in0 + in1, in0 - in1]

    Args:
        x: Input tensor of shape (N, 2) for N pairs of values to transform.
           For complex tensors, the last dimension should be 2 (pairs of complex numbers).

    Returns:
        Output tensor with the same shape as input.
    """
    x = x.contiguous()
    original_shape = x.shape

    if torch.is_complex(x):
        # Handle complex tensors by separating real and imaginary parts
        x_flat = x.view(-1)
        n_pairs = x_flat.shape[0] // 2

        # Separate real and imaginary parts (must be contiguous for Triton)
        x_real = x_flat.real.contiguous()
        x_imag = x_flat.imag.contiguous()

        out_real = torch.empty_like(x_real)
        out_imag = torch.empty_like(x_imag)

        grid = lambda meta: (n_pairs,)
        _batch_rad2_kernel_complex[grid](
            x_real, x_imag, out_real, out_imag, n_pairs
        )

        # Combine real and imaginary parts
        out = torch.complex(out_real, out_imag)
        return out.view(original_shape)
    else:
        # Real-valued tensors
        x_flat = x.view(-1)
        n_pairs = x_flat.shape[0] // 2

        out = torch.empty_like(x_flat)

        grid = lambda meta: (n_pairs,)
        _batch_rad2_kernel_real[grid](x_flat, out, n_pairs)

        return out.view(original_shape)


def batch_inv_rad2_b1(x: torch.Tensor) -> torch.Tensor:
    """
    Batch inverse radix-2 FFT using Triton kernels.

    Note: For radix-2, forward and inverse operations are identical
    (the butterfly operation is its own inverse up to scaling).
    Applying the same operation twice: FFT(FFT(x)) = 2*x

    To properly invert, you need to divide by 2 after the second application.

    Args:
        x: Input tensor of shape (N, 2) or flattened to (N*2,).

    Returns:
        Output tensor with the same shape as input.
    """
    # For radix-2, forward and inverse are the same operation
    return batch_fwd_rad2_b1(x)


# ============================================================================
# Original CUDA-compatible API (using the original incorrect formula)
# These are kept for reference and compatibility with the CUDA code
# ============================================================================

@triton.jit
def fwd_rad2_b1_cuda_style(R0_ptr, R1_ptr, stride: tl.constexpr):
    """
    Forward radix-2 FFT butterfly operation matching the CUDA implementation.

    NOTE: This matches the CUDA code exactly but contains a bug!
    The CUDA implementation has:
        (*R1) = (*R0) - (*R1);     // new_R1 = R0 - R1
        (*R0) = 2.0 * (*R0) - (*R1); // new_R0 = 2*R0 - new_R1 = R0 + R1 (if computed correctly)

    But since R1 is overwritten first, the second line uses the NEW R1 value,
    giving: new_R0 = 2*R0 - (R0 - R1) = R0 + R1 (correct!)

    Actually, let me re-analyze:
        R1_new = R0_old - R1_old
        R0_new = 2 * R0_old - R1_new = 2 * R0_old - (R0_old - R1_old) = R0_old + R1_old

    So the CUDA code IS correct! The issue in my previous implementation
    was that I was computing sequentially and using wrong values.
    """
    R0 = tl.load(R0_ptr)
    R1 = tl.load(R1_ptr)

    # Store old values
    R0_old = R0
    R1_old = R1

    new_R1 = R0_old - R1_old
    new_R0 = 2.0 * R0_old - new_R1  # = R0_old + R1_old

    tl.store(R0_ptr, new_R0)
    tl.store(R1_ptr, new_R1)


@triton.jit
def inv_rad2_b1_cuda_style(R0_ptr, R1_ptr, stride: tl.constexpr):
    """
    Inverse radix-2 FFT butterfly operation matching the CUDA implementation.
    """
    # Same as forward for radix-2
    fwd_rad2_b1_cuda_style(R0_ptr, R1_ptr, stride)


# Aliases for the public API
fwd_rad2_b1 = fwd_rad2_b1_cuda_style
inv_rad2_b1 = inv_rad2_b1_cuda_style
triton_fwd_rad2_b1 = batch_fwd_rad2_b1
triton_inv_rad2_b1 = batch_inv_rad2_b1
