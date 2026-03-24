"""
Triton implementation of radix-11 FFT codelet.

This module provides forward and inverse radix-11 FFT butterfly operations.
Ported from CUDA radix_11.h
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _batch_rad11_kernel_complex_fwd(
    x_real_ptr,
    x_imag_ptr,
    out_real_ptr,
    out_imag_ptr,
    n_groups: tl.constexpr,
):
    """
    Batch forward radix-11 FFT kernel for complex-valued data.
    """
    # Constants defined inside kernel for Triton compatibility
    Q11i1j1R = 0.8412535328311812
    Q11i1j1I = -0.5406408174555976
    Q11i1j2R = 0.4154150130018864
    Q11i1j2I = -0.9096319953545184
    Q11i1j3R = -0.14231483827328514
    Q11i1j3I = -0.9898214418809327
    Q11i1j4R = -0.6548607339452851
    Q11i1j4I = -0.7557495743542583
    Q11i1j5R = -0.9594929736144974
    Q11i1j5I = -0.2817325568414297
    Q11i2j1R = 0.4154150130018864
    Q11i2j1I = -0.9096319953545184
    Q11i2j2R = -0.6548607339452851
    Q11i2j2I = -0.7557495743542583
    Q11i2j3R = -0.9594929736144974
    Q11i2j3I = 0.2817325568414297
    Q11i2j4R = -0.14231483827328514
    Q11i2j4I = 0.9898214418809327
    Q11i2j5R = 0.8412535328311812
    Q11i2j5I = 0.5406408174555976
    Q11i3j1R = -0.14231483827328514
    Q11i3j1I = -0.9898214418809327
    Q11i3j2R = -0.9594929736144974
    Q11i3j2I = 0.2817325568414297
    Q11i3j3R = 0.4154150130018864
    Q11i3j3I = 0.9096319953545184
    Q11i3j4R = 0.8412535328311812
    Q11i3j4I = -0.5406408174555976
    Q11i3j5R = -0.6548607339452851
    Q11i3j5I = -0.7557495743542583
    Q11i4j1R = -0.6548607339452851
    Q11i4j1I = -0.7557495743542583
    Q11i4j2R = -0.14231483827328514
    Q11i4j2I = 0.9898214418809327
    Q11i4j3R = 0.8412535328311812
    Q11i4j3I = -0.5406408174555976
    Q11i4j4R = -0.9594929736144974
    Q11i4j4I = -0.2817325568414297
    Q11i4j5R = 0.4154150130018864
    Q11i4j5I = 0.9096319953545184
    Q11i5j1R = -0.9594929736144974
    Q11i5j1I = -0.2817325568414297
    Q11i5j2R = 0.8412535328311812
    Q11i5j2I = 0.5406408174555976
    Q11i5j3R = -0.6548607339452851
    Q11i5j3I = -0.7557495743542583
    Q11i5j4R = 0.4154150130018864
    Q11i5j4I = 0.9096319953545184
    Q11i5j5R = -0.14231483827328514
    Q11i5j5I = -0.9898214418809327

    pid = tl.program_id(0)

    if pid < n_groups:
        # Manual index computation
        idx0 = 11 * pid
        idx1 = 11 * pid + 1
        idx2 = 11 * pid + 2
        idx3 = 11 * pid + 3
        idx4 = 11 * pid + 4
        idx5 = 11 * pid + 5
        idx6 = 11 * pid + 6
        idx7 = 11 * pid + 7
        idx8 = 11 * pid + 8
        idx9 = 11 * pid + 9
        idx10 = 11 * pid + 10

        # Load all 11 inputs - manual unrolling
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
        R10 = tl.load(x_real_ptr + idx10)

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
        I10 = tl.load(x_imag_ptr + idx10)

        x0_real = R0 + R1 + R2 + R3 + R4 + R5 + R6 + R7 + R8 + R9 + R10
        x0_imag = I0 + I1 + I2 + I3 + I4 + I5 + I6 + I7 + I8 + I9 + I10

        x1_real = R0
        x1_imag = I0
        x2_real = R0
        x2_imag = I0
        x3_real = R0
        x3_imag = I0
        x4_real = R0
        x4_imag = I0
        x5_real = R0
        x5_imag = I0
        x6_real = R0
        x6_imag = I0
        x7_real = R0
        x7_imag = I0
        x8_real = R0
        x8_imag = I0
        x9_real = R0
        x9_imag = I0
        x10_real = R0
        x10_imag = I0

        # Process pairs (R1, R10)
        dp_real = R1 + R10
        dp_imag = I1 + I10
        dm_real = R1 - R10
        dm_imag = I1 - I10
        x1_real += Q11i1j1R * dp_real - Q11i1j1I * dm_imag
        x1_imag += Q11i1j1R * dp_imag + Q11i1j1I * dm_real
        x10_real += Q11i1j1R * dp_real + Q11i1j1I * dm_imag
        x10_imag += Q11i1j1R * dp_imag - Q11i1j1I * dm_real
        x2_real += Q11i2j1R * dp_real - Q11i2j1I * dm_imag
        x2_imag += Q11i2j1R * dp_imag + Q11i2j1I * dm_real
        x9_real += Q11i2j1R * dp_real + Q11i2j1I * dm_imag
        x9_imag += Q11i2j1R * dp_imag - Q11i2j1I * dm_real
        x3_real += Q11i3j1R * dp_real - Q11i3j1I * dm_imag
        x3_imag += Q11i3j1R * dp_imag + Q11i3j1I * dm_real
        x8_real += Q11i3j1R * dp_real + Q11i3j1I * dm_imag
        x8_imag += Q11i3j1R * dp_imag - Q11i3j1I * dm_real
        x4_real += Q11i4j1R * dp_real - Q11i4j1I * dm_imag
        x4_imag += Q11i4j1R * dp_imag + Q11i4j1I * dm_real
        x7_real += Q11i4j1R * dp_real + Q11i4j1I * dm_imag
        x7_imag += Q11i4j1R * dp_imag - Q11i4j1I * dm_real
        x5_real += Q11i5j1R * dp_real - Q11i5j1I * dm_imag
        x5_imag += Q11i5j1R * dp_imag + Q11i5j1I * dm_real
        x6_real += Q11i5j1R * dp_real + Q11i5j1I * dm_imag
        x6_imag += Q11i5j1R * dp_imag - Q11i5j1I * dm_real

        # Process pairs (R2, R9)
        dp_real = R2 + R9
        dp_imag = I2 + I9
        dm_real = R2 - R9
        dm_imag = I2 - I9
        x1_real += Q11i1j2R * dp_real - Q11i1j2I * dm_imag
        x1_imag += Q11i1j2R * dp_imag + Q11i1j2I * dm_real
        x10_real += Q11i1j2R * dp_real + Q11i1j2I * dm_imag
        x10_imag += Q11i1j2R * dp_imag - Q11i1j2I * dm_real
        x2_real += Q11i2j2R * dp_real - Q11i2j2I * dm_imag
        x2_imag += Q11i2j2R * dp_imag + Q11i2j2I * dm_real
        x9_real += Q11i2j2R * dp_real + Q11i2j2I * dm_imag
        x9_imag += Q11i2j2R * dp_imag - Q11i2j2I * dm_real
        x3_real += Q11i3j2R * dp_real - Q11i3j2I * dm_imag
        x3_imag += Q11i3j2R * dp_imag + Q11i3j2I * dm_real
        x8_real += Q11i3j2R * dp_real + Q11i3j2I * dm_imag
        x8_imag += Q11i3j2R * dp_imag - Q11i3j2I * dm_real
        x4_real += Q11i4j2R * dp_real - Q11i4j2I * dm_imag
        x4_imag += Q11i4j2R * dp_imag + Q11i4j2I * dm_real
        x7_real += Q11i4j2R * dp_real + Q11i4j2I * dm_imag
        x7_imag += Q11i4j2R * dp_imag - Q11i4j2I * dm_real
        x5_real += Q11i5j2R * dp_real - Q11i5j2I * dm_imag
        x5_imag += Q11i5j2R * dp_imag + Q11i5j2I * dm_real
        x6_real += Q11i5j2R * dp_real + Q11i5j2I * dm_imag
        x6_imag += Q11i5j2R * dp_imag - Q11i5j2I * dm_real

        # Process pairs (R3, R8)
        dp_real = R3 + R8
        dp_imag = I3 + I8
        dm_real = R3 - R8
        dm_imag = I3 - I8
        x1_real += Q11i1j3R * dp_real - Q11i1j3I * dm_imag
        x1_imag += Q11i1j3R * dp_imag + Q11i1j3I * dm_real
        x10_real += Q11i1j3R * dp_real + Q11i1j3I * dm_imag
        x10_imag += Q11i1j3R * dp_imag - Q11i1j3I * dm_real
        x2_real += Q11i2j3R * dp_real - Q11i2j3I * dm_imag
        x2_imag += Q11i2j3R * dp_imag + Q11i2j3I * dm_real
        x9_real += Q11i2j3R * dp_real + Q11i2j3I * dm_imag
        x9_imag += Q11i2j3R * dp_imag - Q11i2j3I * dm_real
        x3_real += Q11i3j3R * dp_real - Q11i3j3I * dm_imag
        x3_imag += Q11i3j3R * dp_imag + Q11i3j3I * dm_real
        x8_real += Q11i3j3R * dp_real + Q11i3j3I * dm_imag
        x8_imag += Q11i3j3R * dp_imag - Q11i3j3I * dm_real
        x4_real += Q11i4j3R * dp_real - Q11i4j3I * dm_imag
        x4_imag += Q11i4j3R * dp_imag + Q11i4j3I * dm_real
        x7_real += Q11i4j3R * dp_real + Q11i4j3I * dm_imag
        x7_imag += Q11i4j3R * dp_imag - Q11i4j3I * dm_real
        x5_real += Q11i5j3R * dp_real - Q11i5j3I * dm_imag
        x5_imag += Q11i5j3R * dp_imag + Q11i5j3I * dm_real
        x6_real += Q11i5j3R * dp_real + Q11i5j3I * dm_imag
        x6_imag += Q11i5j3R * dp_imag - Q11i5j3I * dm_real

        # Process pairs (R4, R7)
        dp_real = R4 + R7
        dp_imag = I4 + I7
        dm_real = R4 - R7
        dm_imag = I4 - I7
        x1_real += Q11i1j4R * dp_real - Q11i1j4I * dm_imag
        x1_imag += Q11i1j4R * dp_imag + Q11i1j4I * dm_real
        x10_real += Q11i1j4R * dp_real + Q11i1j4I * dm_imag
        x10_imag += Q11i1j4R * dp_imag - Q11i1j4I * dm_real
        x2_real += Q11i2j4R * dp_real - Q11i2j4I * dm_imag
        x2_imag += Q11i2j4R * dp_imag + Q11i2j4I * dm_real
        x9_real += Q11i2j4R * dp_real + Q11i2j4I * dm_imag
        x9_imag += Q11i2j4R * dp_imag - Q11i2j4I * dm_real
        x3_real += Q11i3j4R * dp_real - Q11i3j4I * dm_imag
        x3_imag += Q11i3j4R * dp_imag + Q11i3j4I * dm_real
        x8_real += Q11i3j4R * dp_real + Q11i3j4I * dm_imag
        x8_imag += Q11i3j4R * dp_imag - Q11i3j4I * dm_real
        x4_real += Q11i4j4R * dp_real - Q11i4j4I * dm_imag
        x4_imag += Q11i4j4R * dp_imag + Q11i4j4I * dm_real
        x7_real += Q11i4j4R * dp_real + Q11i4j4I * dm_imag
        x7_imag += Q11i4j4R * dp_imag - Q11i4j4I * dm_real
        x5_real += Q11i5j4R * dp_real - Q11i5j4I * dm_imag
        x5_imag += Q11i5j4R * dp_imag + Q11i5j4I * dm_real
        x6_real += Q11i5j4R * dp_real + Q11i5j4I * dm_imag
        x6_imag += Q11i5j4R * dp_imag - Q11i5j4I * dm_real

        # Process pairs (R5, R6)
        dp_real = R5 + R6
        dp_imag = I5 + I6
        dm_real = R5 - R6
        dm_imag = I5 - I6
        x1_real += Q11i1j5R * dp_real - Q11i1j5I * dm_imag
        x1_imag += Q11i1j5R * dp_imag + Q11i1j5I * dm_real
        x10_real += Q11i1j5R * dp_real + Q11i1j5I * dm_imag
        x10_imag += Q11i1j5R * dp_imag - Q11i1j5I * dm_real
        x2_real += Q11i2j5R * dp_real - Q11i2j5I * dm_imag
        x2_imag += Q11i2j5R * dp_imag + Q11i2j5I * dm_real
        x9_real += Q11i2j5R * dp_real + Q11i2j5I * dm_imag
        x9_imag += Q11i2j5R * dp_imag - Q11i2j5I * dm_real
        x3_real += Q11i3j5R * dp_real - Q11i3j5I * dm_imag
        x3_imag += Q11i3j5R * dp_imag + Q11i3j5I * dm_real
        x8_real += Q11i3j5R * dp_real + Q11i3j5I * dm_imag
        x8_imag += Q11i3j5R * dp_imag - Q11i3j5I * dm_real
        x4_real += Q11i4j5R * dp_real - Q11i4j5I * dm_imag
        x4_imag += Q11i4j5R * dp_imag + Q11i4j5I * dm_real
        x7_real += Q11i4j5R * dp_real + Q11i4j5I * dm_imag
        x7_imag += Q11i4j5R * dp_imag - Q11i4j5I * dm_real
        x5_real += Q11i5j5R * dp_real - Q11i5j5I * dm_imag
        x5_imag += Q11i5j5R * dp_imag + Q11i5j5I * dm_real
        x6_real += Q11i5j5R * dp_real + Q11i5j5I * dm_imag
        x6_imag += Q11i5j5R * dp_imag - Q11i5j5I * dm_real

        # Store results
        tl.store(out_real_ptr + idx0, x0_real)
        tl.store(out_imag_ptr + idx0, x0_imag)
        tl.store(out_real_ptr + idx1, x1_real)
        tl.store(out_imag_ptr + idx1, x1_imag)
        tl.store(out_real_ptr + idx2, x2_real)
        tl.store(out_imag_ptr + idx2, x2_imag)
        tl.store(out_real_ptr + idx3, x3_real)
        tl.store(out_imag_ptr + idx3, x3_imag)
        tl.store(out_real_ptr + idx4, x4_real)
        tl.store(out_imag_ptr + idx4, x4_imag)
        tl.store(out_real_ptr + idx5, x5_real)
        tl.store(out_imag_ptr + idx5, x5_imag)
        tl.store(out_real_ptr + idx6, x6_real)
        tl.store(out_imag_ptr + idx6, x6_imag)
        tl.store(out_real_ptr + idx7, x7_real)
        tl.store(out_imag_ptr + idx7, x7_imag)
        tl.store(out_real_ptr + idx8, x8_real)
        tl.store(out_imag_ptr + idx8, x8_imag)
        tl.store(out_real_ptr + idx9, x9_real)
        tl.store(out_imag_ptr + idx9, x9_imag)
        tl.store(out_real_ptr + idx10, x10_real)
        tl.store(out_imag_ptr + idx10, x10_imag)


@triton.jit
def _batch_rad11_kernel_complex_inv(
    x_real_ptr,
    x_imag_ptr,
    out_real_ptr,
    out_imag_ptr,
    n_groups: tl.constexpr,
):
    """
    Batch inverse radix-11 FFT kernel for complex-valued data.
    """
    # Constants defined inside kernel for Triton compatibility
    Q11i1j1R = 0.8412535328311812
    Q11i1j1I = -0.5406408174555976
    Q11i1j2R = 0.4154150130018864
    Q11i1j2I = -0.9096319953545184
    Q11i1j3R = -0.14231483827328514
    Q11i1j3I = -0.9898214418809327
    Q11i1j4R = -0.6548607339452851
    Q11i1j4I = -0.7557495743542583
    Q11i1j5R = -0.9594929736144974
    Q11i1j5I = -0.2817325568414297
    Q11i2j1R = 0.4154150130018864
    Q11i2j1I = -0.9096319953545184
    Q11i2j2R = -0.6548607339452851
    Q11i2j2I = -0.7557495743542583
    Q11i2j3R = -0.9594929736144974
    Q11i2j3I = 0.2817325568414297
    Q11i2j4R = -0.14231483827328514
    Q11i2j4I = 0.9898214418809327
    Q11i2j5R = 0.8412535328311812
    Q11i2j5I = 0.5406408174555976
    Q11i3j1R = -0.14231483827328514
    Q11i3j1I = -0.9898214418809327
    Q11i3j2R = -0.9594929736144974
    Q11i3j2I = 0.2817325568414297
    Q11i3j3R = 0.4154150130018864
    Q11i3j3I = 0.9096319953545184
    Q11i3j4R = 0.8412535328311812
    Q11i3j4I = -0.5406408174555976
    Q11i3j5R = -0.6548607339452851
    Q11i3j5I = -0.7557495743542583
    Q11i4j1R = -0.6548607339452851
    Q11i4j1I = -0.7557495743542583
    Q11i4j2R = -0.14231483827328514
    Q11i4j2I = 0.9898214418809327
    Q11i4j3R = 0.8412535328311812
    Q11i4j3I = -0.5406408174555976
    Q11i4j4R = -0.9594929736144974
    Q11i4j4I = -0.2817325568414297
    Q11i4j5R = 0.4154150130018864
    Q11i4j5I = 0.9096319953545184
    Q11i5j1R = -0.9594929736144974
    Q11i5j1I = -0.2817325568414297
    Q11i5j2R = 0.8412535328311812
    Q11i5j2I = 0.5406408174555976
    Q11i5j3R = -0.6548607339452851
    Q11i5j3I = -0.7557495743542583
    Q11i5j4R = 0.4154150130018864
    Q11i5j4I = 0.9096319953545184
    Q11i5j5R = -0.14231483827328514
    Q11i5j5I = -0.9898214418809327

    pid = tl.program_id(0)

    if pid < n_groups:
        # Manual index computation
        idx0 = 11 * pid
        idx1 = 11 * pid + 1
        idx2 = 11 * pid + 2
        idx3 = 11 * pid + 3
        idx4 = 11 * pid + 4
        idx5 = 11 * pid + 5
        idx6 = 11 * pid + 6
        idx7 = 11 * pid + 7
        idx8 = 11 * pid + 8
        idx9 = 11 * pid + 9
        idx10 = 11 * pid + 10

        # Load all 11 inputs - manual unrolling
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
        R10 = tl.load(x_real_ptr + idx10)

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
        I10 = tl.load(x_imag_ptr + idx10)

        x0_real = R0 + R1 + R2 + R3 + R4 + R5 + R6 + R7 + R8 + R9 + R10
        x0_imag = I0 + I1 + I2 + I3 + I4 + I5 + I6 + I7 + I8 + I9 + I10

        x1_real = R0
        x1_imag = I0
        x2_real = R0
        x2_imag = I0
        x3_real = R0
        x3_imag = I0
        x4_real = R0
        x4_imag = I0
        x5_real = R0
        x5_imag = I0
        x6_real = R0
        x6_imag = I0
        x7_real = R0
        x7_imag = I0
        x8_real = R0
        x8_imag = I0
        x9_real = R0
        x9_imag = I0
        x10_real = R0
        x10_imag = I0

        # Process pairs - inverse has opposite sign on imaginary coefficient
        dp_real = R1 + R10
        dp_imag = I1 + I10
        dm_real = R1 - R10
        dm_imag = I1 - I10
        x1_real += Q11i1j1R * dp_real + Q11i1j1I * dm_imag
        x1_imag += Q11i1j1R * dp_imag - Q11i1j1I * dm_real
        x10_real += Q11i1j1R * dp_real - Q11i1j1I * dm_imag
        x10_imag += Q11i1j1R * dp_imag + Q11i1j1I * dm_real
        x2_real += Q11i2j1R * dp_real + Q11i2j1I * dm_imag
        x2_imag += Q11i2j1R * dp_imag - Q11i2j1I * dm_real
        x9_real += Q11i2j1R * dp_real - Q11i2j1I * dm_imag
        x9_imag += Q11i2j1R * dp_imag + Q11i2j1I * dm_real
        x3_real += Q11i3j1R * dp_real + Q11i3j1I * dm_imag
        x3_imag += Q11i3j1R * dp_imag - Q11i3j1I * dm_real
        x8_real += Q11i3j1R * dp_real - Q11i3j1I * dm_imag
        x8_imag += Q11i3j1R * dp_imag + Q11i3j1I * dm_real
        x4_real += Q11i4j1R * dp_real + Q11i4j1I * dm_imag
        x4_imag += Q11i4j1R * dp_imag - Q11i4j1I * dm_real
        x7_real += Q11i4j1R * dp_real - Q11i4j1I * dm_imag
        x7_imag += Q11i4j1R * dp_imag + Q11i4j1I * dm_real
        x5_real += Q11i5j1R * dp_real + Q11i5j1I * dm_imag
        x5_imag += Q11i5j1R * dp_imag - Q11i5j1I * dm_real
        x6_real += Q11i5j1R * dp_real - Q11i5j1I * dm_imag
        x6_imag += Q11i5j1R * dp_imag + Q11i5j1I * dm_real

        dp_real = R2 + R9
        dp_imag = I2 + I9
        dm_real = R2 - R9
        dm_imag = I2 - I9
        x1_real += Q11i1j2R * dp_real + Q11i1j2I * dm_imag
        x1_imag += Q11i1j2R * dp_imag - Q11i1j2I * dm_real
        x10_real += Q11i1j2R * dp_real - Q11i1j2I * dm_imag
        x10_imag += Q11i1j2R * dp_imag + Q11i1j2I * dm_real
        x2_real += Q11i2j2R * dp_real + Q11i2j2I * dm_imag
        x2_imag += Q11i2j2R * dp_imag - Q11i2j2I * dm_real
        x9_real += Q11i2j2R * dp_real - Q11i2j2I * dm_imag
        x9_imag += Q11i2j2R * dp_imag + Q11i2j2I * dm_real
        x3_real += Q11i3j2R * dp_real + Q11i3j2I * dm_imag
        x3_imag += Q11i3j2R * dp_imag - Q11i3j2I * dm_real
        x8_real += Q11i3j2R * dp_real - Q11i3j2I * dm_imag
        x8_imag += Q11i3j2R * dp_imag + Q11i3j2I * dm_real
        x4_real += Q11i4j2R * dp_real + Q11i4j2I * dm_imag
        x4_imag += Q11i4j2R * dp_imag - Q11i4j2I * dm_real
        x7_real += Q11i4j2R * dp_real - Q11i4j2I * dm_imag
        x7_imag += Q11i4j2R * dp_imag + Q11i4j2I * dm_real
        x5_real += Q11i5j2R * dp_real + Q11i5j2I * dm_imag
        x5_imag += Q11i5j2R * dp_imag - Q11i5j2I * dm_real
        x6_real += Q11i5j2R * dp_real - Q11i5j2I * dm_imag
        x6_imag += Q11i5j2R * dp_imag + Q11i5j2I * dm_real

        dp_real = R3 + R8
        dp_imag = I3 + I8
        dm_real = R3 - R8
        dm_imag = I3 - I8
        x1_real += Q11i1j3R * dp_real + Q11i1j3I * dm_imag
        x1_imag += Q11i1j3R * dp_imag - Q11i1j3I * dm_real
        x10_real += Q11i1j3R * dp_real - Q11i1j3I * dm_imag
        x10_imag += Q11i1j3R * dp_imag + Q11i1j3I * dm_real
        x2_real += Q11i2j3R * dp_real + Q11i2j3I * dm_imag
        x2_imag += Q11i2j3R * dp_imag - Q11i2j3I * dm_real
        x9_real += Q11i2j3R * dp_real - Q11i2j3I * dm_imag
        x9_imag += Q11i2j3R * dp_imag + Q11i2j3I * dm_real
        x3_real += Q11i3j3R * dp_real + Q11i3j3I * dm_imag
        x3_imag += Q11i3j3R * dp_imag - Q11i3j3I * dm_real
        x8_real += Q11i3j3R * dp_real - Q11i3j3I * dm_imag
        x8_imag += Q11i3j3R * dp_imag + Q11i3j3I * dm_real
        x4_real += Q11i4j3R * dp_real + Q11i4j3I * dm_imag
        x4_imag += Q11i4j3R * dp_imag - Q11i4j3I * dm_real
        x7_real += Q11i4j3R * dp_real - Q11i4j3I * dm_imag
        x7_imag += Q11i4j3R * dp_imag + Q11i4j3I * dm_real
        x5_real += Q11i5j3R * dp_real + Q11i5j3I * dm_imag
        x5_imag += Q11i5j3R * dp_imag - Q11i5j3I * dm_real
        x6_real += Q11i5j3R * dp_real - Q11i5j3I * dm_imag
        x6_imag += Q11i5j3R * dp_imag + Q11i5j3I * dm_real

        dp_real = R4 + R7
        dp_imag = I4 + I7
        dm_real = R4 - R7
        dm_imag = I4 - I7
        x1_real += Q11i1j4R * dp_real + Q11i1j4I * dm_imag
        x1_imag += Q11i1j4R * dp_imag - Q11i1j4I * dm_real
        x10_real += Q11i1j4R * dp_real - Q11i1j4I * dm_imag
        x10_imag += Q11i1j4R * dp_imag + Q11i1j4I * dm_real
        x2_real += Q11i2j4R * dp_real + Q11i2j4I * dm_imag
        x2_imag += Q11i2j4R * dp_imag - Q11i2j4I * dm_real
        x9_real += Q11i2j4R * dp_real - Q11i2j4I * dm_imag
        x9_imag += Q11i2j4R * dp_imag + Q11i2j4I * dm_real
        x3_real += Q11i3j4R * dp_real + Q11i3j4I * dm_imag
        x3_imag += Q11i3j4R * dp_imag - Q11i3j4I * dm_real
        x8_real += Q11i3j4R * dp_real - Q11i3j4I * dm_imag
        x8_imag += Q11i3j4R * dp_imag + Q11i3j4I * dm_real
        x4_real += Q11i4j4R * dp_real + Q11i4j4I * dm_imag
        x4_imag += Q11i4j4R * dp_imag - Q11i4j4I * dm_real
        x7_real += Q11i4j4R * dp_real - Q11i4j4I * dm_imag
        x7_imag += Q11i4j4R * dp_imag + Q11i4j4I * dm_real
        x5_real += Q11i5j4R * dp_real + Q11i5j4I * dm_imag
        x5_imag += Q11i5j4R * dp_imag - Q11i5j4I * dm_real
        x6_real += Q11i5j4R * dp_real - Q11i5j4I * dm_imag
        x6_imag += Q11i5j4R * dp_imag + Q11i5j4I * dm_real

        dp_real = R5 + R6
        dp_imag = I5 + I6
        dm_real = R5 - R6
        dm_imag = I5 - I6
        x1_real += Q11i1j5R * dp_real + Q11i1j5I * dm_imag
        x1_imag += Q11i1j5R * dp_imag - Q11i1j5I * dm_real
        x10_real += Q11i1j5R * dp_real - Q11i1j5I * dm_imag
        x10_imag += Q11i1j5R * dp_imag + Q11i1j5I * dm_real
        x2_real += Q11i2j5R * dp_real + Q11i2j5I * dm_imag
        x2_imag += Q11i2j5R * dp_imag - Q11i2j5I * dm_real
        x9_real += Q11i2j5R * dp_real - Q11i2j5I * dm_imag
        x9_imag += Q11i2j5R * dp_imag + Q11i2j5I * dm_real
        x3_real += Q11i3j5R * dp_real + Q11i3j5I * dm_imag
        x3_imag += Q11i3j5R * dp_imag - Q11i3j5I * dm_real
        x8_real += Q11i3j5R * dp_real - Q11i3j5I * dm_imag
        x8_imag += Q11i3j5R * dp_imag + Q11i3j5I * dm_real
        x4_real += Q11i4j5R * dp_real + Q11i4j5I * dm_imag
        x4_imag += Q11i4j5R * dp_imag - Q11i4j5I * dm_real
        x7_real += Q11i4j5R * dp_real - Q11i4j5I * dm_imag
        x7_imag += Q11i4j5R * dp_imag + Q11i4j5I * dm_real
        x5_real += Q11i5j5R * dp_real + Q11i5j5I * dm_imag
        x5_imag += Q11i5j5R * dp_imag - Q11i5j5I * dm_real
        x6_real += Q11i5j5R * dp_real - Q11i5j5I * dm_imag
        x6_imag += Q11i5j5R * dp_imag + Q11i5j5I * dm_real

        # Store results
        tl.store(out_real_ptr + idx0, x0_real)
        tl.store(out_imag_ptr + idx0, x0_imag)
        tl.store(out_real_ptr + idx1, x1_real)
        tl.store(out_imag_ptr + idx1, x1_imag)
        tl.store(out_real_ptr + idx2, x2_real)
        tl.store(out_imag_ptr + idx2, x2_imag)
        tl.store(out_real_ptr + idx3, x3_real)
        tl.store(out_imag_ptr + idx3, x3_imag)
        tl.store(out_real_ptr + idx4, x4_real)
        tl.store(out_imag_ptr + idx4, x4_imag)
        tl.store(out_real_ptr + idx5, x5_real)
        tl.store(out_imag_ptr + idx5, x5_imag)
        tl.store(out_real_ptr + idx6, x6_real)
        tl.store(out_imag_ptr + idx6, x6_imag)
        tl.store(out_real_ptr + idx7, x7_real)
        tl.store(out_imag_ptr + idx7, x7_imag)
        tl.store(out_real_ptr + idx8, x8_real)
        tl.store(out_imag_ptr + idx8, x8_imag)
        tl.store(out_real_ptr + idx9, x9_real)
        tl.store(out_imag_ptr + idx9, x9_imag)
        tl.store(out_real_ptr + idx10, x10_real)
        tl.store(out_imag_ptr + idx10, x10_imag)


def batch_fwd_rad11_b1(x: torch.Tensor) -> torch.Tensor:
    """Batch forward radix-11 FFT using Triton kernels."""
    x = x.contiguous()
    original_shape = x.shape

    if torch.is_complex(x):
        x_flat = x.view(-1)
        n_groups = x_flat.shape[0] // 11

        x_real = x_flat.real.contiguous()
        x_imag = x_flat.imag.contiguous()

        out_real = torch.empty_like(x_real)
        out_imag = torch.empty_like(x_imag)

        grid = lambda meta: (n_groups,)
        _batch_rad11_kernel_complex_fwd[grid](x_real, x_imag, out_real, out_imag, n_groups)

        out = torch.complex(out_real, out_imag)
        return out.view(original_shape)
    else:
        raise ValueError("Radix-11 FFT requires complex input")


def batch_inv_rad11_b1(x: torch.Tensor) -> torch.Tensor:
    """Batch inverse radix-11 FFT using Triton kernels."""
    x = x.contiguous()
    original_shape = x.shape

    if torch.is_complex(x):
        x_flat = x.view(-1)
        n_groups = x_flat.shape[0] // 11

        x_real = x_flat.real.contiguous()
        x_imag = x_flat.imag.contiguous()

        out_real = torch.empty_like(x_real)
        out_imag = torch.empty_like(x_imag)

        grid = lambda meta: (n_groups,)
        _batch_rad11_kernel_complex_inv[grid](x_real, x_imag, out_real, out_imag, n_groups)

        out = torch.complex(out_real, out_imag)
        return out.view(original_shape)
    else:
        raise ValueError("Radix-11 FFT requires complex input")


# Aliases for public API
triton_fwd_rad11_b1 = batch_fwd_rad11_b1
triton_inv_rad11_b1 = batch_inv_rad11_b1
