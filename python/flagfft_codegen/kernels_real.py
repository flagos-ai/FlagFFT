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

"""Real-transform pointwise kernels, including packed even-length transforms."""

from textwrap import dedent

from .kernels_common import _dtype_suffix, _next_power_of_two, _zero_other

def _packed_layout(n_cols: int, block: int = 256) -> tuple[int, int]:
    """Choose a (columns, rows-per-block) tile for tiny row-wise kernels.

    Rows shorter than one block are packed so a full 256-lane tile stays
    busy; rows longer than the block keep one row per block with the column
    axis spread across grid.x (the historical behavior).
    """
    block_cols = min(block, _next_power_of_two(n_cols))
    rows_per_block = max(1, block // block_cols)
    return block_cols, rows_per_block



def _build_real_to_complex_kernel_source(
    n: int, dtype: str
) -> tuple[str, list[str], list[str]]:
    block = 256
    block_cols, rows_per_block = _packed_layout(n, block)
    zero = _zero_other(dtype)
    suffix = _dtype_suffix(dtype)
    kernel_name = f"_real_to_complex_kernel_n{n}_{suffix}"
    source = dedent(
        f"""
        @triton.jit
        def {kernel_name}(
            in_ptr,
            out_ptr,
            input_distance,
            nbatch,
        ):
            pid_block = tl.program_id(0)
            pid_batch = tl.program_id(1)
            row_offsets = pid_batch * {rows_per_block} + tl.arange(0, {rows_per_block})[:, None]
            col_offsets = pid_block * {block_cols} + tl.arange(0, {block_cols})[None, :]
            mask = (row_offsets < nbatch) & (col_offsets < {n})
            safe_rows = tl.minimum(row_offsets, nbatch - 1)
            safe_cols = tl.minimum(col_offsets, {n - 1})
            xr = tl.load(
                in_ptr + safe_rows * input_distance + safe_cols, mask=mask, other={zero}
            )
            dst = out_ptr + (safe_rows * {n} + safe_cols) * 2
            tl.store(dst, xr, mask=mask)
            tl.store(dst + 1, 0.0, mask=mask)
        """
    )
    return (
        kernel_name,
        source,
        ["in_ptr", "out_ptr", "input_distance", "nbatch"],
        rows_per_block,
    )


def _build_r2c_half_pack_kernel_source(
    n: int, dtype: str
) -> tuple[str, list[str], list[str]]:
    half = n // 2 + 1
    block = 256
    block_cols, rows_per_block = _packed_layout(half, block)
    zero = _zero_other(dtype)
    suffix = _dtype_suffix(dtype)
    kernel_name = f"_r2c_half_pack_kernel_n{n}_{suffix}"
    source = dedent(
        f"""
        @triton.jit
        def {kernel_name}(
            in_ptr,
            out_ptr,
            output_distance,
            nbatch,
        ):
            pid_block = tl.program_id(0)
            pid_batch = tl.program_id(1)
            row_offsets = pid_batch * {rows_per_block} + tl.arange(0, {rows_per_block})[:, None]
            col_offsets = pid_block * {block_cols} + tl.arange(0, {block_cols})[None, :]
            mask = (row_offsets < nbatch) & (col_offsets < {half})
            safe_rows = tl.minimum(row_offsets, nbatch - 1)
            safe_cols = tl.minimum(col_offsets, {half - 1})
            src = in_ptr + (safe_rows * {n} + safe_cols) * 2
            xr = tl.load(src, mask=mask, other={zero})
            xi = tl.load(src + 1, mask=mask, other={zero})
            dst = out_ptr + (safe_rows * output_distance + safe_cols) * 2
            tl.store(dst, xr, mask=mask)
            tl.store(dst + 1, xi, mask=mask)
        """
    )
    return (
        kernel_name,
        source,
        ["in_ptr", "out_ptr", "output_distance", "nbatch"],
        rows_per_block,
    )


def _build_r2c_packed_postprocess_kernel_source(
    n: int, dtype: str
) -> tuple[str, list[str], list[str]]:
    if n % 2 != 0:
        raise ValueError("packed real FFT requires an even length")
    packed = n // 2
    half = packed + 1
    block = 256
    block_cols, rows_per_block = _packed_layout(half, block)
    zero = _zero_other(dtype)
    suffix = _dtype_suffix(dtype)
    kernel_name = f"_r2c_packed_postprocess_kernel_n{n}_{suffix}"
    source = dedent(
        f"""
        @triton.jit
        def {kernel_name}(
            in_ptr,
            twiddle_ptr,
            out_ptr,
            output_distance,
            nbatch,
        ):
            pid_block = tl.program_id(0)
            pid_batch = tl.program_id(1)
            row_offsets = pid_batch * {rows_per_block} + tl.arange(0, {rows_per_block})[:, None]
            col_offsets = pid_block * {block_cols} + tl.arange(0, {block_cols})[None, :]
            mask = (row_offsets < nbatch) & (col_offsets < {half})
            safe_rows = tl.minimum(row_offsets, nbatch - 1)
            k = tl.minimum(col_offsets, {packed})
            a_k = k % {packed}
            b_k = ({packed} - k) % {packed}
            a_ptr = in_ptr + (safe_rows * {packed} + a_k) * 2
            b_ptr = in_ptr + (safe_rows * {packed} + b_k) * 2
            ar = tl.load(a_ptr, mask=mask, other={zero})
            ai = tl.load(a_ptr + 1, mask=mask, other={zero})
            br = tl.load(b_ptr, mask=mask, other={zero})
            bi = -tl.load(b_ptr + 1, mask=mask, other={zero})
            wr = tl.load(twiddle_ptr + k * 2, mask=mask, other={zero})
            wi = tl.load(twiddle_ptr + k * 2 + 1, mask=mask, other={zero})
            sum_r = ar + br
            sum_i = ai + bi
            diff_r = ar - br
            diff_i = ai - bi
            prod_r = diff_r * wr - diff_i * wi
            prod_i = diff_i * wr + diff_r * wi
            out_r = 0.5 * (sum_r + prod_i)
            out_i = 0.5 * (sum_i - prod_r)
            dst = out_ptr + (safe_rows * output_distance + k) * 2
            tl.store(dst, out_r, mask=mask)
            tl.store(dst + 1, out_i, mask=mask)
        """
    )
    return (
        kernel_name,
        source,
        ["in_ptr", "twiddle_ptr", "out_ptr", "output_distance", "nbatch"],
        rows_per_block,
    )


def _build_c2r_packed_preprocess_kernel_source(
    n: int, dtype: str
) -> tuple[str, list[str], list[str]]:
    if n % 2 != 0:
        raise ValueError("packed real FFT requires an even length")
    packed = n // 2
    block = 256
    block_cols, rows_per_block = _packed_layout(packed, block)
    zero = _zero_other(dtype)
    suffix = _dtype_suffix(dtype)
    kernel_name = f"_c2r_packed_preprocess_kernel_n{n}_{suffix}"
    source = dedent(
        f"""
        @triton.jit
        def {kernel_name}(
            in_ptr,
            twiddle_ptr,
            out_ptr,
            input_distance,
            nbatch,
        ):
            pid_block = tl.program_id(0)
            pid_batch = tl.program_id(1)
            row_offsets = pid_batch * {rows_per_block} + tl.arange(0, {rows_per_block})[:, None]
            col_offsets = pid_block * {block_cols} + tl.arange(0, {block_cols})[None, :]
            mask = (row_offsets < nbatch) & (col_offsets < {packed})
            safe_rows = tl.minimum(row_offsets, nbatch - 1)
            k = tl.minimum(col_offsets, {packed - 1})
            q = {packed} - k
            x_ptr = in_ptr + (safe_rows * input_distance + k) * 2
            q_ptr = in_ptr + (safe_rows * input_distance + q) * 2
            xr = tl.load(x_ptr, mask=mask, other={zero})
            xi = tl.load(x_ptr + 1, mask=mask, other={zero})
            qr = tl.load(q_ptr, mask=mask, other={zero})
            qi = -tl.load(q_ptr + 1, mask=mask, other={zero})
            wr = tl.load(twiddle_ptr + k * 2, mask=mask, other={zero})
            wi = tl.load(twiddle_ptr + k * 2 + 1, mask=mask, other={zero})
            sum_r = xr + qr
            sum_i = xi + qi
            diff_r = xr - qr
            diff_i = xi - qi
            # Multiply the difference by conj(W), then by +i.  Omitting the
            # conventional 0.5 here supplies the factor of two needed to turn
            # an unnormalised N/2 inverse FFT into an unnormalised N-point C2R.
            prod_r = diff_r * wr + diff_i * wi
            prod_i = diff_i * wr - diff_r * wi
            packed_r = sum_r - prod_i
            packed_i = sum_i + prod_r
            dst = out_ptr + (safe_rows * {packed} + k) * 2
            tl.store(dst, packed_r, mask=mask)
            tl.store(dst + 1, packed_i, mask=mask)
        """
    )
    return (
        kernel_name,
        source,
        ["in_ptr", "twiddle_ptr", "out_ptr", "input_distance", "nbatch"],
        rows_per_block,
    )


def _build_compact_to_hermitian_full_kernel_source(
    n: int, dtype: str
) -> tuple[str, list[str], list[str]]:
    half = n // 2 + 1
    block = 256
    block_cols, rows_per_block = _packed_layout(n, block)
    nyquist_guard = f" | (safe_cols == {n // 2})" if n % 2 == 0 else ""
    zero = _zero_other(dtype)
    suffix = _dtype_suffix(dtype)
    kernel_name = f"_compact_to_hermitian_full_kernel_n{n}_{suffix}"
    source = dedent(
        f"""
        @triton.jit
        def {kernel_name}(
            in_ptr,
            out_ptr,
            input_distance,
            nbatch,
        ):
            pid_block = tl.program_id(0)
            pid_batch = tl.program_id(1)
            row_offsets = pid_batch * {rows_per_block} + tl.arange(0, {rows_per_block})[:, None]
            col_offsets = pid_block * {block_cols} + tl.arange(0, {block_cols})[None, :]
            mask = (row_offsets < nbatch) & (col_offsets < {n})
            safe_rows = tl.minimum(row_offsets, nbatch - 1)
            safe_cols = tl.minimum(col_offsets, {n - 1})
            src_k = tl.where(safe_cols < {half}, safe_cols, {n} - safe_cols)
            src = in_ptr + (safe_rows * input_distance + src_k) * 2
            xr = tl.load(src, mask=mask, other={zero})
            xi = tl.load(src + 1, mask=mask, other={zero})
            xi = tl.where(safe_cols < {half}, xi, -xi)
            xi = tl.where((safe_cols == 0){nyquist_guard}, 0.0, xi)
            dst = out_ptr + (safe_rows * {n} + safe_cols) * 2
            tl.store(dst, xr, mask=mask)
            tl.store(dst + 1, xi, mask=mask)
        """
    )
    return (
        kernel_name,
        source,
        ["in_ptr", "out_ptr", "input_distance", "nbatch"],
        rows_per_block,
    )


def _build_complex_to_real_kernel_source(
    n: int, dtype: str
) -> tuple[str, list[str], list[str]]:
    block = 256
    block_cols, rows_per_block = _packed_layout(n, block)
    zero = _zero_other(dtype)
    suffix = _dtype_suffix(dtype)
    kernel_name = f"_complex_to_real_kernel_n{n}_{suffix}"
    source = dedent(
        f"""
        @triton.jit
        def {kernel_name}(
            in_ptr,
            out_ptr,
            output_distance,
            nbatch,
        ):
            pid_block = tl.program_id(0)
            pid_batch = tl.program_id(1)
            row_offsets = pid_batch * {rows_per_block} + tl.arange(0, {rows_per_block})[:, None]
            col_offsets = pid_block * {block_cols} + tl.arange(0, {block_cols})[None, :]
            mask = (row_offsets < nbatch) & (col_offsets < {n})
            safe_rows = tl.minimum(row_offsets, nbatch - 1)
            safe_cols = tl.minimum(col_offsets, {n - 1})
            src = in_ptr + (safe_rows * {n} + safe_cols) * 2
            xr = tl.load(src, mask=mask, other={zero})
            dst = out_ptr + safe_rows * output_distance + safe_cols
            tl.store(dst, xr, mask=mask)
        """
    )
    return (
        kernel_name,
        source,
        ["in_ptr", "out_ptr", "output_distance", "nbatch"],
        rows_per_block,
    )


__all__ = [
    "_build_c2r_packed_preprocess_kernel_source",
    "_build_compact_to_hermitian_full_kernel_source",
    "_build_complex_to_real_kernel_source",
    "_build_r2c_half_pack_kernel_source",
    "_build_r2c_packed_postprocess_kernel_source",
    "_build_real_to_complex_kernel_source",
    "_packed_layout",
]
