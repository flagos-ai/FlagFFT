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

"""Layout kernels: reshape packs and tiled 2D/3D transposes."""

from textwrap import dedent

from .kernels_common import _dtype_suffix, _zero_other

def _build_reshape_pack_kernel_source(
    n1: int, n2: int, dtype: str
) -> tuple[str, list[str], list[str]]:
    """Emit a pointwise (batch, R=n1, C=n2) -> (batch, C, R) transpose pack kernel.

    Used by generic nested four-step both for the row pre-transpose (stage 1) and the
    final natural-order pack (stage 5). Complex elements are stored interleaved
    (real, imag) with stride 2.
    """
    R = n1
    C = n2
    total = R * C
    block = 256
    zero = _zero_other(dtype)
    suffix = _dtype_suffix(dtype)
    kernel_name = f"_reshape_pack_kernel_n{R}_{C}_{suffix}"
    source = dedent(
        f"""
        @triton.jit
        def {kernel_name}(
            in_ptr,
            out_ptr,
            nbatch,
        ):
            pid_block = tl.program_id(0)
            pid_batch = tl.program_id(1)
            offsets = pid_block * {block} + tl.arange(0, {block})
            mask = offsets < {total}
            r = offsets // {C}
            c = offsets - r * {C}
            src = in_ptr + (pid_batch * {total} + offsets) * 2
            xr = tl.load(src, mask=mask, other={zero})
            xi = tl.load(src + 1, mask=mask, other={zero})
            dst_off = c * {R} + r
            dst = out_ptr + (pid_batch * {total} + dst_off) * 2
            tl.store(dst, xr, mask=mask)
            tl.store(dst + 1, xi, mask=mask)
        """
    )
    return kernel_name, source, ["in_ptr", "out_ptr", "nbatch"]


def _build_twiddle_reshape_pack_kernel_source(
    n1: int, n2: int, dtype: str
) -> tuple[str, list[str], list[str]]:
    """Emit a pointwise twiddle-multiply + (batch, R=n1, C=n2) -> (batch, C, R) pack kernel.

    The twiddle table is laid out as (R, C) row-major complex: twiddle[r * C + c].
    Used by generic nested four-step as the stage 3 fused twiddle+transpose pass.
    Caller-side mapping is R=n2, C=n1; the four-step twiddle table built by
    build_raw_four_step_twiddle uses (row=j2, col=k1) → (row * n1 + col) which
    matches twiddle[r * C + c] when (R, C) = (n2, n1).
    """
    R = n1
    C = n2
    total = R * C
    block = 256
    zero = _zero_other(dtype)
    suffix = _dtype_suffix(dtype)
    kernel_name = f"_twiddle_reshape_pack_kernel_n{R}_{C}_{suffix}"
    source = dedent(
        f"""
        @triton.jit
        def {kernel_name}(
            in_ptr,
            twiddle_ptr,
            out_ptr,
            nbatch,
        ):
            pid_block = tl.program_id(0)
            pid_batch = tl.program_id(1)
            offsets = pid_block * {block} + tl.arange(0, {block})
            mask = offsets < {total}
            r = offsets // {C}
            c = offsets - r * {C}
            src = in_ptr + (pid_batch * {total} + offsets) * 2
            xr = tl.load(src, mask=mask, other={zero})
            xi = tl.load(src + 1, mask=mask, other={zero})
            tw = twiddle_ptr + offsets * 2
            tr = tl.load(tw, mask=mask, other={zero})
            ti = tl.load(tw + 1, mask=mask, other={zero})
            yr, yi = _cmul(xr, xi, tr, ti)
            dst_off = c * {R} + r
            dst = out_ptr + (pid_batch * {total} + dst_off) * 2
            tl.store(dst, yr, mask=mask)
            tl.store(dst + 1, yi, mask=mask)
        """
    )
    return kernel_name, source, ["in_ptr", "twiddle_ptr", "out_ptr", "nbatch"]

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

def _build_tiled_transpose_kernel_source(
    n0: int, n1: int, dtype: str, tile_size: int = 32
) -> tuple[str, list[str], list[str]]:
    """Emit a tiled (batch, M=n0, N=n1) -> (batch, N, M) transpose kernel.

    Decomposes the matrix into tile_size x tile_size blocks across the grid — each
    program loads one coalesced tile from global memory, transposes it in registers
    with tl.trans, and writes it back with coalesced stores.
    """
    zero = _zero_other(dtype)
    suffix = _dtype_suffix(dtype)
    total_complex = n0 * n1
    total_float = total_complex * 2  # interleaved complex: 2 floats per element
    kernel_name = f"_tiled_transpose_kernel_n{n0}_{n1}_{suffix}"
    use_register_transpose = total_complex <= 128 * 1024
    if use_register_transpose:
        transpose_lines = (
            "\n            # Transpose the register tile so the flattened store axis is the\n"
            "            # contiguous destination row.\n"
            "            src_real = tl.trans(src_real)\n"
            "            src_imag = tl.trans(src_imag)\n"
            f"            row_mask_t = row_offsets[None, :] < {n0}\n"
            f"            col_mask_t = col_offsets[:, None] < {n1}\n"
            "            mask = col_mask_t & row_mask_t\n"
        )
        dst_offsets_expr = f"(safe_col[:, None] * {n0} + safe_row[None, :]) * 2"
    else:
        transpose_lines = ""
        dst_offsets_expr = f"(safe_col[None, :] * {n0} + safe_row[:, None]) * 2"
    source = dedent(
        f"""
        @triton.jit
        def {kernel_name}(
            in_ptr,
            out_ptr,
            nbatch,
        ):
            # Program IDs
            pid_tile_col = tl.program_id(0)
            pid_tile_row = tl.program_id(1)
            pid_batch = tl.program_id(2)

            # Tile offsets
            tile_row_start = pid_tile_row * {tile_size}
            tile_col_start = pid_tile_col * {tile_size}

            # Row and column offsets within tile
            row_offsets = tile_row_start + tl.arange(0, {tile_size})
            col_offsets = tile_col_start + tl.arange(0, {tile_size})

            # Mask for valid elements
            row_mask = row_offsets < {n0}
            col_mask = col_offsets < {n1}
            mask = row_mask[:, None] & col_mask[None, :]

            # Clamp offsets to valid range to avoid out-of-bounds addresses
            safe_row = tl.minimum(row_offsets, {n0 - 1})
            safe_col = tl.minimum(col_offsets, {n1 - 1})

            # Source element offsets in floats (row-major: batch * n0 * n1 * 2 + (row * n1 + col) * 2)
            src_elem_offsets = pid_batch * {total_float} + (safe_row[:, None] * {n1} + safe_col[None, :]) * 2

            # Load from source (complex elements are interleaved: real, imag)
            src_real = tl.load(in_ptr + src_elem_offsets, mask=mask, other={zero})
            src_imag = tl.load(in_ptr + src_elem_offsets + 1, mask=mask, other={zero})

            {transpose_lines}

            # Destination element offsets in floats (transposed: batch * n0 * n1 * 2 + (col * n0 + row) * 2)
            dst_elem_offsets = pid_batch * {total_float} + {dst_offsets_expr}

            # Store to destination (transposed)
            tl.store(out_ptr + dst_elem_offsets, src_real, mask=mask)
            tl.store(out_ptr + dst_elem_offsets + 1, src_imag, mask=mask)
        """
    )
    return kernel_name, source, ["in_ptr", "out_ptr", "nbatch"]


def _build_tiled_transpose3d_kernel_source(
    s0: int, s1: int, s2: int, order: str, dtype: str, block: int = 1024
) -> tuple[str, list[str], list[str]]:
    """Emit a generic 3D axis-permutation kernel.

    The source cube has dimensions (s0, s1, s2); the destination cube dims are
    derived from the axis order:
      "021": dst (s0, s2, s1)   dst[x0][x1][x2] = src[x0][x2][x1]
      "210": dst (s2, s1, s0)   dst[x0][x1][x2] = src[x2][x1][x0]
      "201": dst (s2, s0, s1)   dst[x0][x1][x2] = src[x1][x2][x0]
      "120": dst (s1, s2, s0)   dst[x0][x1][x2] = src[x2][x0][x1]

    Each program copies a contiguous block of destination elements, computing
    the source offset with baked-in div/mod arithmetic.  Correctness-first:
    no shared memory, the tiling is purely for grid parallelism.
    """
    if order not in {"021", "210", "201", "120"}:
        raise ValueError(f"unsupported 3D transpose order: {order}")
    zero = _zero_other(dtype)
    suffix = _dtype_suffix(dtype)
    dims = {
        "021": (s0, s2, s1),
        "210": (s2, s1, s0),
        "201": (s2, s0, s1),
        "120": (s1, s2, s0),
    }
    d0, d1, d2 = dims[order]
    total_complex = s0 * s1 * s2
    total_float = total_complex * 2
    # src coords (y0, y1, y2) for a given dst coord (x0, x1, x2).
    src_coords = {
        "021": "y0 = x0; y1 = x2; y2 = x1;",
        "210": "y0 = x2; y1 = x1; y2 = x0;",
        "201": "y0 = x1; y1 = x2; y2 = x0;",
        "120": "y0 = x2; y1 = x0; y2 = x1;",
    }
    kernel_name = f"_tiled_transpose3d_kernel_{order}_n{s0}_{s1}_{s2}_{suffix}"
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

            offsets = pid_block * {block} + tl.arange(0, {block})
            mask = offsets < {total_complex}
            safe = tl.minimum(offsets, {total_complex - 1})

            # Decompose destination linear index into (x0, x1, x2).
            x0 = safe // {d1 * d2}
            rem = safe % {d1 * d2}
            x1 = rem // {d2}
            x2 = rem % {d2}

            # Map to source coordinates.
            {src_coords[order]}

            src_elem_offsets = pid_batch * {total_float} + (y0 * {s1 * s2} + y1 * {s2} + y2) * 2
            src_real = tl.load(in_ptr + src_elem_offsets, mask=mask, other={zero})
            src_imag = tl.load(in_ptr + src_elem_offsets + 1, mask=mask, other={zero})

            dst_elem_offsets = pid_batch * {total_float} + offsets * 2
            tl.store(out_ptr + dst_elem_offsets, src_real, mask=mask)
            tl.store(out_ptr + dst_elem_offsets + 1, src_imag, mask=mask)
        """
    )
    return kernel_name, source, ["in_ptr", "out_ptr", "nbatch"]


__all__ = [
    "_build_reshape_pack_kernel_source",
    "_build_tiled_transpose3d_kernel_source",
    "_build_tiled_transpose3d_v2_kernel_source",
    "_build_tiled_transpose_kernel_source",
    "_build_twiddle_reshape_pack_kernel_source",
]
