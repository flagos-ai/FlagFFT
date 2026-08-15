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

"""Real-transform pointwise kernels: expand/half-pack/Hermitian-full/real-pack."""

from textwrap import dedent

from .kernels_common import _dtype_suffix, _zero_other

def _packed_layout(n_cols: int, block: int = 256) -> tuple[int, int]:
    """Choose a (columns, rows-per-block) tile for tiny row-wise kernels.

    Rows shorter than one block are packed so a full 256-lane tile stays
    busy; rows longer than the block keep one row per block with the column
    axis spread across grid.x (the historical behavior).
    """
    block_cols = min(block, _next_pow2(n_cols))
    rows_per_block = max(1, block // block_cols)
    return block_cols, rows_per_block


def _next_pow2(value: int) -> int:
    power = 1
    while power < value:
        power *= 2
    return power


def _build_tiled_transpose3d_v2_kernel_source(
    s0: int,
    s1: int,
    s2: int,
    order: str,
    dtype: str,
    tile: int = 32,
) -> tuple[str, str, list[str], int]:
    """Vectorized (ld/st.global.v2) variant of the 3D axis permutation.

    Identical tile mapping to _build_tiled_transpose3d_kernel_source but
    loads/stores each complex element as one 8-byte transaction via inline
    asm.  Out-of-tile lanes clamp to valid addresses, so unconditional
    vectorized stores are idempotent (the duplicate value matches the value
    written by the corner lane).
    """
    if order not in {"021", "210", "201", "120"}:
        raise ValueError(f"unsupported 3D transpose order: {order}")
    suffix = _dtype_suffix(dtype)
    if dtype != "complex64":
        raise ValueError("v2 transpose requires complex64")
    total_complex = s0 * s1 * s2
    total_float = total_complex * 2
    transpose_descs = {
        "021": dict(
            num_slices=s0,
            rows=s2,
            cols=s1,
            src_slice_stride=s1 * s2,
            src_col_stride=s2,
            dst_slice_stride=s2 * s1,
            dst_row_stride=s1,
        ),
        "210": dict(
            num_slices=s1,
            rows=s2,
            cols=s0,
            src_slice_stride=s2,
            src_col_stride=s1 * s2,
            dst_slice_stride=s0,
            dst_row_stride=s1 * s0,
        ),
        "201": dict(
            num_slices=s0,
            rows=s2,
            cols=s1,
            src_slice_stride=s1 * s2,
            src_col_stride=s2,
            dst_slice_stride=s1,
            dst_row_stride=s0 * s1,
        ),
        "120": dict(
            num_slices=s1,
            rows=s2,
            cols=s0,
            src_slice_stride=s2,
            src_col_stride=s1 * s2,
            dst_slice_stride=s2 * s0,
            dst_row_stride=s0,
        ),
    }
    desc = transpose_descs[order]
    num_slices = desc["num_slices"]
    rows = desc["rows"]
    cols = desc["cols"]
    src_slice_stride = desc["src_slice_stride"]
    src_col_stride = desc["src_col_stride"]
    dst_slice_stride = desc["dst_slice_stride"]
    dst_row_stride = desc["dst_row_stride"]
    tile_cols = (cols + tile - 1) // tile
    tile_rows = (rows + tile - 1) // tile
    tiles_per_slice = tile_cols * tile_rows
    grid_x = num_slices * tiles_per_slice
    kernel_name = (
        f"_tiled_transpose3d_kernel_{order}_n{s0}_{s1}_{s2}_{suffix}_t{tile}_v2"
    )
    source = dedent(
        f"""
        @triton.jit
        def {kernel_name}(
            in_ptr,
            out_ptr,
            nbatch,
        ):
            pid_block = tl.program_id(0)
            pid_batch = tl.program_id(2)

            slice_idx = pid_block // {tiles_per_slice}
            tile_in_slice = pid_block % {tiles_per_slice}
            tile_row = tile_in_slice // {tile_cols}
            tile_col = tile_in_slice % {tile_cols}

            row_offsets = tile_row * {tile} + tl.arange(0, {tile})
            col_offsets = tile_col * {tile} + tl.arange(0, {tile})
            safe_rows = tl.minimum(row_offsets, {rows - 1})
            safe_cols = tl.minimum(col_offsets, {cols - 1})

            # Load transposed tile [ci][ri] with the source-contiguous axis
            # (rows) innermost; one v2 transaction per complex element.
            src_base = (
                pid_batch * {total_float}
                + slice_idx * {src_slice_stride} * 2
                + safe_cols[:, None] * {src_col_stride} * 2
                + safe_rows[None, :] * 2
            )
            src_r, src_i = tl.inline_asm_elementwise(
                "ld.global.v2.f32 {{$0, $1}}, [$2];",
                "=f,=f,l",
                [tl.cast(in_ptr + src_base, tl.uint64)],
                dtype=(tl.float32, tl.float32),
                is_pure=False,
                pack=1,
            )

            dst_r = tl.trans(src_r)
            dst_i = tl.trans(src_i)
            dst_base = (
                pid_batch * {total_float}
                + slice_idx * {dst_slice_stride} * 2
                + safe_rows[:, None] * {dst_row_stride} * 2
                + safe_cols[None, :] * 2
            )
            tl.inline_asm_elementwise(
                "st.global.v2.f32 [$1], {{$2, $3}}; mov.u32 $0, 0;",
                "=r,l,f,f",
                [tl.cast(out_ptr + dst_base, tl.uint64), dst_r, dst_i],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )
        """
    )
    return (
        kernel_name,
        source,
        ["in_ptr", "out_ptr", "nbatch"],
        grid_x,
    )


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
    "_build_compact_to_hermitian_full_kernel_source",
    "_build_complex_to_real_kernel_source",
    "_build_r2c_half_pack_kernel_source",
    "_build_real_to_complex_kernel_source",
    "_build_tiled_transpose3d_v2_kernel_source",
    "_next_pow2",
    "_packed_layout",
]
