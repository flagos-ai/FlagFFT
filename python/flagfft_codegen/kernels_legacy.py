# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

"""Legacy static JIT kernels and duplicate prime-algorithm builders (scheduled for removal)."""

from textwrap import dedent

import triton
import triton.language as tl

from .kernels_common import _dtype_suffix, _zero_other

@triton.jit
def _cmul(ar, ai, br, bi):
    return ar * br - ai * bi, ai * br + ar * bi


@triton.jit
def _transpose_complex_kernel(
    src_ptr,
    dst_ptr,
    src_batch_stride,
    src_row_stride,
    src_col_stride,
    dst_batch_stride,
    dst_row_stride,
    dst_col_stride,
    rows,
    cols,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    pid_col = tl.program_id(0)
    pid_row = tl.program_id(1)
    pid_batch = tl.program_id(2)

    row_offsets = pid_row * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col_offsets = pid_col * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    mask = (row_offsets[:, None] < rows) & (col_offsets[None, :] < cols)

    src_base = src_ptr + pid_batch * src_batch_stride
    src_offsets = (
        src_base
        + row_offsets[:, None] * src_row_stride
        + col_offsets[None, :] * src_col_stride
    )
    src_real = tl.load(src_offsets, mask=mask, other=0.0)
    src_imag = tl.load(src_offsets + 1, mask=mask, other=0.0)

    dst_base = dst_ptr + pid_batch * dst_batch_stride
    dst_offsets = (
        dst_base
        + col_offsets[None, :] * dst_row_stride
        + row_offsets[:, None] * dst_col_stride
    )
    tl.store(dst_offsets, src_real, mask=mask)
    tl.store(dst_offsets + 1, src_imag, mask=mask)


@triton.jit
def _twiddle_transpose_complex_kernel(
    src_ptr,
    twiddle_ptr,
    dst_ptr,
    src_batch_stride,
    src_row_stride,
    src_col_stride,
    twiddle_row_stride,
    twiddle_col_stride,
    dst_batch_stride,
    dst_row_stride,
    dst_col_stride,
    rows,
    cols,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    pid_col = tl.program_id(0)
    pid_row = tl.program_id(1)
    pid_batch = tl.program_id(2)

    row_offsets = pid_row * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col_offsets = pid_col * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    mask = (row_offsets[:, None] < rows) & (col_offsets[None, :] < cols)

    src_base = src_ptr + pid_batch * src_batch_stride
    src_offsets = (
        src_base
        + row_offsets[:, None] * src_row_stride
        + col_offsets[None, :] * src_col_stride
    )
    src_real = tl.load(src_offsets, mask=mask, other=0.0)
    src_imag = tl.load(src_offsets + 1, mask=mask, other=0.0)

    tw_offsets = (
        twiddle_ptr
        + row_offsets[:, None] * twiddle_row_stride
        + col_offsets[None, :] * twiddle_col_stride
    )
    tw_real = tl.load(tw_offsets, mask=mask, other=0.0)
    tw_imag = tl.load(tw_offsets + 1, mask=mask, other=0.0)
    out_real, out_imag = _cmul(src_real, src_imag, tw_real, tw_imag)

    dst_base = dst_ptr + pid_batch * dst_batch_stride
    dst_offsets = (
        dst_base
        + col_offsets[None, :] * dst_row_stride
        + row_offsets[:, None] * dst_col_stride
    )
    tl.store(dst_offsets, out_real, mask=mask)
    tl.store(dst_offsets + 1, out_imag, mask=mask)


@triton.jit
def _bluestein_prepare_kernel(
    in_ptr,
    chirp_ptr,
    out_ptr,
    n,
    m,
    nbatch,
    BLOCK: tl.constexpr,
):
    pid_block = tl.program_id(0)
    pid_batch = tl.program_id(1)
    offsets = pid_block * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < m
    in_mask = mask & (offsets < n)

    src = in_ptr + (pid_batch * n + offsets) * 2
    xr = tl.load(src, mask=in_mask, other=0.0)
    xi = tl.load(src + 1, mask=in_mask, other=0.0)
    cr = tl.load(chirp_ptr + offsets * 2, mask=in_mask, other=0.0)
    ci = tl.load(chirp_ptr + offsets * 2 + 1, mask=in_mask, other=0.0)
    yr, yi = _cmul(xr, xi, cr, ci)

    dst = out_ptr + (pid_batch * m + offsets) * 2
    tl.store(dst, yr, mask=mask)
    tl.store(dst + 1, yi, mask=mask)


@triton.jit
def _bluestein_pointwise_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    m,
    nbatch,
    BLOCK: tl.constexpr,
):
    pid_block = tl.program_id(0)
    pid_batch = tl.program_id(1)
    offsets = pid_block * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < m

    a = a_ptr + (pid_batch * m + offsets) * 2
    b = b_ptr + offsets * 2
    ar = tl.load(a, mask=mask, other=0.0)
    ai = tl.load(a + 1, mask=mask, other=0.0)
    br = tl.load(b, mask=mask, other=0.0)
    bi = tl.load(b + 1, mask=mask, other=0.0)
    pr, pi = _cmul(ar, ai, br, bi)

    dst = out_ptr + (pid_batch * m + offsets) * 2
    tl.store(dst, pr, mask=mask)
    tl.store(dst + 1, -pi, mask=mask)


@triton.jit
def _bluestein_finalize_kernel(
    in_ptr,
    chirp_ptr,
    out_ptr,
    n,
    m,
    nbatch,
    BLOCK: tl.constexpr,
):
    pid_block = tl.program_id(0)
    pid_batch = tl.program_id(1)
    offsets = pid_block * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n

    src = in_ptr + (pid_batch * m + offsets) * 2
    xr = tl.load(src, mask=mask, other=0.0) / m
    xi = -tl.load(src + 1, mask=mask, other=0.0) / m
    cr = tl.load(chirp_ptr + offsets * 2, mask=mask, other=0.0)
    ci = tl.load(chirp_ptr + offsets * 2 + 1, mask=mask, other=0.0)
    yr, yi = _cmul(xr, xi, cr, ci)

    dst = out_ptr + (pid_batch * n + offsets) * 2
    tl.store(dst, yr, mask=mask)
    tl.store(dst + 1, yi, mask=mask)


def _build_rader_prepare_kernel_source(dtype: str) -> tuple[str, str, list[str]]:
    zero = _zero_other(dtype)
    return (
        "_rader_prepare_kernel",
        dedent(
            f"""
            @triton.jit
            def _rader_prepare_kernel(
                in_ptr,
                idx_ptr,
                out_ptr,
                n,
                m,
                nbatch,
            ):
                pid_block = tl.program_id(0)
                pid_batch = tl.program_id(1)
                offsets = pid_block * 256 + tl.arange(0, 256)
                mask = offsets < m
                inv_offsets = tl.where(offsets == 0, 0, m - offsets)
                src_index = tl.load(idx_ptr + inv_offsets, mask=mask, other=0)

                src = in_ptr + (pid_batch * n + src_index) * 2
                xr = tl.load(src, mask=mask, other={zero})
                xi = tl.load(src + 1, mask=mask, other={zero})

                dst = out_ptr + (pid_batch * m + offsets) * 2
                tl.store(dst, xr, mask=mask)
                tl.store(dst + 1, xi, mask=mask)
            """
        ),
        ["in_ptr", "idx_ptr", "out_ptr", "n", "m", "nbatch"],
    )


def _build_rader_pointwise_kernel_source(dtype: str) -> tuple[str, str, list[str]]:
    zero = _zero_other(dtype)
    return (
        "_rader_pointwise_kernel",
        dedent(
            f"""
            @triton.jit
            def _rader_pointwise_kernel(
                a_ptr,
                b_ptr,
                out_ptr,
                m,
                nbatch,
            ):
                pid_block = tl.program_id(0)
                pid_batch = tl.program_id(1)
                offsets = pid_block * 256 + tl.arange(0, 256)
                mask = offsets < m

                a = a_ptr + (pid_batch * m + offsets) * 2
                b = b_ptr + offsets * 2
                ar = tl.load(a, mask=mask, other={zero})
                ai = tl.load(a + 1, mask=mask, other={zero})
                br = tl.load(b, mask=mask, other={zero})
                bi = tl.load(b + 1, mask=mask, other={zero})
                pr, pi = _cmul(ar, ai, br, bi)

                dst = out_ptr + (pid_batch * m + offsets) * 2
                tl.store(dst, pr, mask=mask)
                tl.store(dst + 1, -pi, mask=mask)
            """
        ),
        ["a_ptr", "b_ptr", "out_ptr", "m", "nbatch"],
    )


def _build_rader_finalize_kernel_source(
    n: int, dtype: str
) -> tuple[str, str, list[str]]:
    zero = _zero_other(dtype)
    div_cast = "tl.cast(m, tl.float64)" if dtype == "complex128" else "m"
    sum_block = 1
    while sum_block < n:
        sum_block <<= 1
    return (
        "_rader_finalize_kernel",
        dedent(
            f"""
            @triton.jit
            def _rader_finalize_kernel(
                input_ptr,
                conv_ptr,
                idx_ptr,
                out_ptr,
                n,
                m,
                nbatch,
            ):
                pid_block = tl.program_id(0)
                pid_batch = tl.program_id(1)
                offsets = pid_block * 256 + tl.arange(0, 256)
                mask = offsets < m

                src = conv_ptr + (pid_batch * m + offsets) * 2
                cr = tl.load(src, mask=mask, other={zero}) / {div_cast}
                ci = -tl.load(src + 1, mask=mask, other={zero}) / {div_cast}
                x0 = input_ptr + pid_batch * n * 2
                x0r = tl.load(x0)
                x0i = tl.load(x0 + 1)

                dst_index = tl.load(idx_ptr + offsets, mask=mask, other=0)
                dst = out_ptr + (pid_batch * n + dst_index) * 2
                tl.store(dst, x0r + cr, mask=mask)
                tl.store(dst + 1, x0i + ci, mask=mask)

                sum_offsets = tl.arange(0, {sum_block})
                sum_mask = sum_offsets < n
                sum_src = input_ptr + (pid_batch * n + sum_offsets) * 2
                sr = tl.load(sum_src, mask=sum_mask, other={zero})
                si = tl.load(sum_src + 1, mask=sum_mask, other={zero})
                out0 = out_ptr + pid_batch * n * 2
                block0 = pid_block == 0
                tl.store(out0, tl.sum(sr, axis=0), mask=block0)
                tl.store(out0 + 1, tl.sum(si, axis=0), mask=block0)
            """
        ),
        ["input_ptr", "conv_ptr", "idx_ptr", "out_ptr", "n", "m", "nbatch"],
    )


__all__ = [
    "_bluestein_finalize_kernel",
    "_bluestein_pointwise_kernel",
    "_bluestein_prepare_kernel",
    "_build_rader_finalize_kernel_source",
    "_build_rader_pointwise_kernel_source",
    "_build_rader_prepare_kernel_source",
    "_cmul",
    "_transpose_complex_kernel",
    "_twiddle_transpose_complex_kernel",
]
