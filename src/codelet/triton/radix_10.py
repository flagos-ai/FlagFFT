"""
Triton implementation of radix-10 FFT codelet.

This module provides forward and inverse radix-10 FFT butterfly operations.
Ported from CUDA radix_10.h

Radix-10 is implemented as radix-5 combined with radix-2.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _batch_rad10_kernel_complex_fwd(
    x_real_ptr,
    x_imag_ptr,
    out_real_ptr,
    out_imag_ptr,
    n_groups: tl.constexpr,
):
    """
    Batch forward radix-10 FFT kernel for complex-valued data.
    """
    # Constants defined inside kernel for Triton compatibility
    C5QA = 0.3090169943749474
    C5QB = 0.9510565162951536
    C5QC = 0.5
    C5QD = 0.5877852522924731
    C5QE = 0.8090169943749474

    pid = tl.program_id(0)

    if pid < n_groups:
        # Manual index computation
        idx0 = 10 * pid
        idx1 = 10 * pid + 1
        idx2 = 10 * pid + 2
        idx3 = 10 * pid + 3
        idx4 = 10 * pid + 4
        idx5 = 10 * pid + 5
        idx6 = 10 * pid + 6
        idx7 = 10 * pid + 7
        idx8 = 10 * pid + 8
        idx9 = 10 * pid + 9

        # Load all 10 inputs - manual unrolling
        R0 = tl.load(x_real_ptr + idx0)
        R1 = tl.load(x_real_ptr + idx1)
        R2 = tl.load(x_real_ptr + idx2)
        R3 = tl.load(x_real_ptr + idx3)
        R4 = tl.load(x_real_ptr + idx4)
        R5 = tl.load(x_real_ptr + idx5)
        R6 = tl.load(x_real_ptr + idx6)
        R7 = tl.load(x_real_ptr + idx7)
        R8 = tl.load(x_real_ptr + idx8)
        R9 = tl.load(x_real_ptr + idx9)

        I0 = tl.load(x_imag_ptr + idx0)
        I1 = tl.load(x_imag_ptr + idx1)
        I2 = tl.load(x_imag_ptr + idx2)
        I3 = tl.load(x_imag_ptr + idx3)
        I4 = tl.load(x_imag_ptr + idx4)
        I5 = tl.load(x_imag_ptr + idx5)
        I6 = tl.load(x_imag_ptr + idx6)
        I7 = tl.load(x_imag_ptr + idx7)
        I8 = tl.load(x_imag_ptr + idx8)
        I9 = tl.load(x_imag_ptr + idx9)

        # Radix-5 on even indices (0, 2, 4, 6, 8)
        TR0 = R0 + R2 + R4 + R6 + R8
        TR2 = (R0 - C5QC * (R4 + R6)) + C5QB * (I2 - I8) + C5QD * (I4 - I6) + C5QA * ((R2 - R4) + (R8 - R6))
        TR8 = (R0 - C5QC * (R4 + R6)) - C5QB * (I2 - I8) - C5QD * (I4 - I6) + C5QA * ((R2 - R4) + (R8 - R6))
        TR4 = (R0 - C5QC * (R2 + R8)) - C5QB * (I4 - I6) + C5QD * (I2 - I8) + C5QA * ((R4 - R2) + (R6 - R8))
        TR6 = (R0 - C5QC * (R2 + R8)) + C5QB * (I4 - I6) - C5QD * (I2 - I8) + C5QA * ((R4 - R2) + (R6 - R8))

        TI0 = I0 + I2 + I4 + I6 + I8
        TI2 = (I0 - C5QC * (I4 + I6)) - C5QB * (R2 - R8) - C5QD * (R4 - R6) + C5QA * ((I2 - I4) + (I8 - I6))
        TI8 = (I0 - C5QC * (I4 + I6)) + C5QB * (R2 - R8) + C5QD * (R4 - R6) + C5QA * ((I2 - I4) + (I8 - I6))
        TI4 = (I0 - C5QC * (I2 + I8)) + C5QB * (R4 - R6) - C5QD * (R2 - R8) + C5QA * ((I4 - I2) + (I6 - I8))
        TI6 = (I0 - C5QC * (I2 + I8)) - C5QB * (R4 - R6) + C5QD * (R2 - R8) + C5QA * ((I4 - I2) + (I6 - I8))

        # Radix-5 on odd indices (1, 3, 5, 7, 9)
        TR1 = R1 + R3 + R5 + R7 + R9
        TR3 = (R1 - C5QC * (R5 + R7)) + C5QB * (I3 - I9) + C5QD * (I5 - I7) + C5QA * ((R3 - R5) + (R9 - R7))
        TR9 = (R1 - C5QC * (R5 + R7)) - C5QB * (I3 - I9) - C5QD * (I5 - I7) + C5QA * ((R3 - R5) + (R9 - R7))
        TR5 = (R1 - C5QC * (R3 + R9)) - C5QB * (I5 - I7) + C5QD * (I3 - I9) + C5QA * ((R5 - R3) + (R7 - R9))
        TR7 = (R1 - C5QC * (R3 + R9)) + C5QB * (I5 - I7) - C5QD * (I3 - I9) + C5QA * ((R5 - R3) + (R7 - R9))

        TI1 = I1 + I3 + I5 + I7 + I9
        TI3 = (I1 - C5QC * (I5 + I7)) - C5QB * (R3 - R9) - C5QD * (R5 - R7) + C5QA * ((I3 - I5) + (I9 - I7))
        TI9 = (I1 - C5QC * (I5 + I7)) + C5QB * (R3 - R9) + C5QD * (R5 - R7) + C5QA * ((I3 - I5) + (I9 - I7))
        TI5 = (I1 - C5QC * (I3 + I9)) + C5QB * (R5 - R7) - C5QD * (R3 - R9) + C5QA * ((I5 - I3) + (I7 - I9))
        TI7 = (I1 - C5QC * (I3 + I9)) - C5QB * (R5 - R7) + C5QD * (R3 - R9) + C5QA * ((I5 - I3) + (I7 - I9))

        # Final radix-2 combination with twiddle factors
        out_R0_real = TR0 + TR1
        out_R0_imag = TI0 + TI1
        out_R1_real = TR2 + (C5QE * TR3 + C5QD * TI3)
        out_R1_imag = TI2 + (-C5QD * TR3 + C5QE * TI3)
        out_R2_real = TR4 + (C5QA * TR5 + C5QB * TI5)
        out_R2_imag = TI4 + (-C5QB * TR5 + C5QA * TI5)
        out_R3_real = TR6 + (-C5QA * TR7 + C5QB * TI7)
        out_R3_imag = TI6 + (-C5QB * TR7 - C5QA * TI7)
        out_R4_real = TR8 + (-C5QE * TR9 + C5QD * TI9)
        out_R4_imag = TI8 + (-C5QD * TR9 - C5QE * TI9)
        out_R5_real = TR0 - TR1
        out_R5_imag = TI0 - TI1
        out_R6_real = TR2 - (C5QE * TR3 + C5QD * TI3)
        out_R6_imag = TI2 - (-C5QD * TR3 + C5QE * TI3)
        out_R7_real = TR4 - (C5QA * TR5 + C5QB * TI5)
        out_R7_imag = TI4 - (-C5QB * TR5 + C5QA * TI5)
        out_R8_real = TR6 - (-C5QA * TR7 + C5QB * TI7)
        out_R8_imag = TI6 - (-C5QB * TR7 - C5QA * TI7)
        out_R9_real = TR8 - (-C5QE * TR9 + C5QD * TI9)
        out_R9_imag = TI8 - (-C5QD * TR9 - C5QE * TI9)

        # Store results
        tl.store(out_real_ptr + idx0, out_R0_real)
        tl.store(out_imag_ptr + idx0, out_R0_imag)
        tl.store(out_real_ptr + idx1, out_R1_real)
        tl.store(out_imag_ptr + idx1, out_R1_imag)
        tl.store(out_real_ptr + idx2, out_R2_real)
        tl.store(out_imag_ptr + idx2, out_R2_imag)
        tl.store(out_real_ptr + idx3, out_R3_real)
        tl.store(out_imag_ptr + idx3, out_R3_imag)
        tl.store(out_real_ptr + idx4, out_R4_real)
        tl.store(out_imag_ptr + idx4, out_R4_imag)
        tl.store(out_real_ptr + idx5, out_R5_real)
        tl.store(out_imag_ptr + idx5, out_R5_imag)
        tl.store(out_real_ptr + idx6, out_R6_real)
        tl.store(out_imag_ptr + idx6, out_R6_imag)
        tl.store(out_real_ptr + idx7, out_R7_real)
        tl.store(out_imag_ptr + idx7, out_R7_imag)
        tl.store(out_real_ptr + idx8, out_R8_real)
        tl.store(out_imag_ptr + idx8, out_R8_imag)
        tl.store(out_real_ptr + idx9, out_R9_real)
        tl.store(out_imag_ptr + idx9, out_R9_imag)


@triton.jit
def _batch_rad10_kernel_complex_inv(
    x_real_ptr,
    x_imag_ptr,
    out_real_ptr,
    out_imag_ptr,
    n_groups: tl.constexpr,
):
    """
    Batch inverse radix-10 FFT kernel for complex-valued data.
    """
    # Constants defined inside kernel for Triton compatibility
    C5QA = 0.3090169943749474
    C5QB = 0.9510565162951536
    C5QC = 0.5
    C5QD = 0.5877852522924731
    C5QE = 0.8090169943749474

    pid = tl.program_id(0)

    if pid < n_groups:
        # Manual index computation
        idx0 = 10 * pid
        idx1 = 10 * pid + 1
        idx2 = 10 * pid + 2
        idx3 = 10 * pid + 3
        idx4 = 10 * pid + 4
        idx5 = 10 * pid + 5
        idx6 = 10 * pid + 6
        idx7 = 10 * pid + 7
        idx8 = 10 * pid + 8
        idx9 = 10 * pid + 9

        # Load all 10 inputs - manual unrolling
        R0 = tl.load(x_real_ptr + idx0)
        R1 = tl.load(x_real_ptr + idx1)
        R2 = tl.load(x_real_ptr + idx2)
        R3 = tl.load(x_real_ptr + idx3)
        R4 = tl.load(x_real_ptr + idx4)
        R5 = tl.load(x_real_ptr + idx5)
        R6 = tl.load(x_real_ptr + idx6)
        R7 = tl.load(x_real_ptr + idx7)
        R8 = tl.load(x_real_ptr + idx8)
        R9 = tl.load(x_real_ptr + idx9)

        I0 = tl.load(x_imag_ptr + idx0)
        I1 = tl.load(x_imag_ptr + idx1)
        I2 = tl.load(x_imag_ptr + idx2)
        I3 = tl.load(x_imag_ptr + idx3)
        I4 = tl.load(x_imag_ptr + idx4)
        I5 = tl.load(x_imag_ptr + idx5)
        I6 = tl.load(x_imag_ptr + idx6)
        I7 = tl.load(x_imag_ptr + idx7)
        I8 = tl.load(x_imag_ptr + idx8)
        I9 = tl.load(x_imag_ptr + idx9)

        # Radix-5 on even indices (inverse signs)
        TR0 = R0 + R2 + R4 + R6 + R8
        TR2 = (R0 - C5QC * (R4 + R6)) - C5QB * (I2 - I8) - C5QD * (I4 - I6) + C5QA * ((R2 - R4) + (R8 - R6))
        TR8 = (R0 - C5QC * (R4 + R6)) + C5QB * (I2 - I8) + C5QD * (I4 - I6) + C5QA * ((R2 - R4) + (R8 - R6))
        TR4 = (R0 - C5QC * (R2 + R8)) + C5QB * (I4 - I6) - C5QD * (I2 - I8) + C5QA * ((R4 - R2) + (R6 - R8))
        TR6 = (R0 - C5QC * (R2 + R8)) - C5QB * (I4 - I6) + C5QD * (I2 - I8) + C5QA * ((R4 - R2) + (R6 - R8))

        TI0 = I0 + I2 + I4 + I6 + I8
        TI2 = (I0 - C5QC * (I4 + I6)) + C5QB * (R2 - R8) + C5QD * (R4 - R6) + C5QA * ((I2 - I4) + (I8 - I6))
        TI8 = (I0 - C5QC * (I4 + I6)) - C5QB * (R2 - R8) - C5QD * (R4 - R6) + C5QA * ((I2 - I4) + (I8 - I6))
        TI4 = (I0 - C5QC * (I2 + I8)) - C5QB * (R4 - R6) + C5QD * (R2 - R8) + C5QA * ((I4 - I2) + (I6 - I8))
        TI6 = (I0 - C5QC * (I2 + I8)) + C5QB * (R4 - R6) - C5QD * (R2 - R8) + C5QA * ((I4 - I2) + (I6 - I8))

        # Radix-5 on odd indices (inverse signs)
        TR1 = R1 + R3 + R5 + R7 + R9
        TR3 = (R1 - C5QC * (R5 + R7)) - C5QB * (I3 - I9) - C5QD * (I5 - I7) + C5QA * ((R3 - R5) + (R9 - R7))
        TR9 = (R1 - C5QC * (R5 + R7)) + C5QB * (I3 - I9) + C5QD * (I5 - I7) + C5QA * ((R3 - R5) + (R9 - R7))
        TR5 = (R1 - C5QC * (R3 + R9)) + C5QB * (I5 - I7) - C5QD * (I3 - I9) + C5QA * ((R5 - R3) + (R7 - R9))
        TR7 = (R1 - C5QC * (R3 + R9)) - C5QB * (I5 - I7) + C5QD * (I3 - I9) + C5QA * ((R5 - R3) + (R7 - R9))

        TI1 = I1 + I3 + I5 + I7 + I9
        TI3 = (I1 - C5QC * (I5 + I7)) + C5QB * (R3 - R9) + C5QD * (R5 - R7) + C5QA * ((I3 - I5) + (I9 - I7))
        TI9 = (I1 - C5QC * (I5 + I7)) - C5QB * (R3 - R9) - C5QD * (R5 - R7) + C5QA * ((I3 - I5) + (I9 - I7))
        TI5 = (I1 - C5QC * (I3 + I9)) - C5QB * (R5 - R7) + C5QD * (R3 - R9) + C5QA * ((I5 - I3) + (I7 - I9))
        TI7 = (I1 - C5QC * (I3 + I9)) + C5QB * (R5 - R7) - C5QD * (R3 - R9) + C5QA * ((I5 - I3) + (I7 - I9))

        # Final radix-2 combination (inverse signs)
        out_R0_real = TR0 + TR1
        out_R0_imag = TI0 + TI1
        out_R1_real = TR2 + (C5QE * TR3 - C5QD * TI3)
        out_R1_imag = TI2 + (C5QD * TR3 + C5QE * TI3)
        out_R2_real = TR4 + (C5QA * TR5 - C5QB * TI5)
        out_R2_imag = TI4 + (C5QB * TR5 + C5QA * TI5)
        out_R3_real = TR6 + (-C5QA * TR7 - C5QB * TI7)
        out_R3_imag = TI6 + (C5QB * TR7 - C5QA * TI7)
        out_R4_real = TR8 + (-C5QE * TR9 - C5QD * TI9)
        out_R4_imag = TI8 + (C5QD * TR9 - C5QE * TI9)
        out_R5_real = TR0 - TR1
        out_R5_imag = TI0 - TI1
        out_R6_real = TR2 - (C5QE * TR3 - C5QD * TI3)
        out_R6_imag = TI2 - (C5QD * TR3 + C5QE * TI3)
        out_R7_real = TR4 - (C5QA * TR5 - C5QB * TI5)
        out_R7_imag = TI4 - (C5QB * TR5 + C5QA * TI5)
        out_R8_real = TR6 - (-C5QA * TR7 - C5QB * TI7)
        out_R8_imag = TI6 - (C5QB * TR7 - C5QA * TI7)
        out_R9_real = TR8 - (-C5QE * TR9 - C5QD * TI9)
        out_R9_imag = TI8 - (C5QD * TR9 - C5QE * TI9)

        # Store results
        tl.store(out_real_ptr + idx0, out_R0_real)
        tl.store(out_imag_ptr + idx0, out_R0_imag)
        tl.store(out_real_ptr + idx1, out_R1_real)
        tl.store(out_imag_ptr + idx1, out_R1_imag)
        tl.store(out_real_ptr + idx2, out_R2_real)
        tl.store(out_imag_ptr + idx2, out_R2_imag)
        tl.store(out_real_ptr + idx3, out_R3_real)
        tl.store(out_imag_ptr + idx3, out_R3_imag)
        tl.store(out_real_ptr + idx4, out_R4_real)
        tl.store(out_imag_ptr + idx4, out_R4_imag)
        tl.store(out_real_ptr + idx5, out_R5_real)
        tl.store(out_imag_ptr + idx5, out_R5_imag)
        tl.store(out_real_ptr + idx6, out_R6_real)
        tl.store(out_imag_ptr + idx6, out_R6_imag)
        tl.store(out_real_ptr + idx7, out_R7_real)
        tl.store(out_imag_ptr + idx7, out_R7_imag)
        tl.store(out_real_ptr + idx8, out_R8_real)
        tl.store(out_imag_ptr + idx8, out_R8_imag)
        tl.store(out_real_ptr + idx9, out_R9_real)
        tl.store(out_imag_ptr + idx9, out_R9_imag)


def batch_fwd_rad10_b1(x: torch.Tensor) -> torch.Tensor:
    """Batch forward radix-10 FFT using Triton kernels."""
    x = x.contiguous()
    original_shape = x.shape

    if torch.is_complex(x):
        x_flat = x.view(-1)
        n_groups = x_flat.shape[0] // 10

        x_real = x_flat.real.contiguous()
        x_imag = x_flat.imag.contiguous()

        out_real = torch.empty_like(x_real)
        out_imag = torch.empty_like(x_imag)

        grid = lambda meta: (n_groups,)
        _batch_rad10_kernel_complex_fwd[grid](x_real, x_imag, out_real, out_imag, n_groups)

        out = torch.complex(out_real, out_imag)
        return out.view(original_shape)
    else:
        raise ValueError("Radix-10 FFT requires complex input")


def batch_inv_rad10_b1(x: torch.Tensor) -> torch.Tensor:
    """Batch inverse radix-10 FFT using Triton kernels."""
    x = x.contiguous()
    original_shape = x.shape

    if torch.is_complex(x):
        x_flat = x.view(-1)
        n_groups = x_flat.shape[0] // 10

        x_real = x_flat.real.contiguous()
        x_imag = x_flat.imag.contiguous()

        out_real = torch.empty_like(x_real)
        out_imag = torch.empty_like(x_imag)

        grid = lambda meta: (n_groups,)
        _batch_rad10_kernel_complex_inv[grid](x_real, x_imag, out_real, out_imag, n_groups)

        out = torch.complex(out_real, out_imag)
        return out.view(original_shape)
    else:
        raise ValueError("Radix-10 FFT requires complex input")


# Aliases for public API
triton_fwd_rad10_b1 = batch_fwd_rad10_b1
triton_inv_rad10_b1 = batch_inv_rad10_b1
