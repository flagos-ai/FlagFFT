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

from .kernels_common import _dtype_suffix, _npu_backend_active, _zero_other

NPU_RESHAPE_TILE = 32
NPU_RESHAPE_PROGRAMS = 40


def npu_tiled_reshape_pack(n1: int, n2: int, dtype: str) -> bool:
    """Whether the four-step reshape uses the table-driven NPU tile transpose."""
    return _npu_backend_active() and dtype == "complex64" and n1 > 1 and n2 > 1


def npu_reshape_pack_grid(n1: int, n2: int) -> int:
    tiles = ((n1 + NPU_RESHAPE_TILE - 1) // NPU_RESHAPE_TILE) * (
        (n2 + NPU_RESHAPE_TILE - 1) // NPU_RESHAPE_TILE
    )
    return min(NPU_RESHAPE_PROGRAMS, tiles)


def _npu_reshape_pack_source(n1: int, n2: int, *, twiddle: bool) -> tuple[str, list[str], list[str]]:
    """Tile-local transpose pack using the 2D transpose byte-offset table.

    The portable pack stores every element to a row-strided address, which the
    Ascend memory system turns into one transaction per element. Load a
    (tile x tile) complex block contiguously, permute it with the same table
    the 2D transpose uses, and write whole transposed rows instead.
    """
    R, C = n1, n2
    total = R * C
    tile = NPU_RESHAPE_TILE
    half = 2 * tile
    n_flat = half * tile
    row_tiles = (R + tile - 1) // tile
    col_tiles = (C + tile - 1) // tile
    tiles = row_tiles * col_tiles
    programs = min(NPU_RESHAPE_PROGRAMS, tiles)
    suffix = _dtype_suffix("complex64")
    kind = "_twiddle_reshape_pack_kernel" if twiddle else "_reshape_pack_kernel"
    name = f"{kind}_n{R}_{C}_{suffix}_t{tile}p{programs}"

    lines = ["@triton.jit", f"def {name}("]
    lines.append("    in_ptr,")
    if twiddle:
        lines.append("    twiddle_ptr,")
    lines.extend(["    idx_ptr,", "    out_ptr,", "    nbatch,", "):"])
    lines.extend([
        "    pid = tl.program_id(0)",
        "    batch = tl.program_id(1).to(tl.int64)",
        f"    src_base = in_ptr + batch * {2 * total}",
        f"    dst_base = out_ptr + batch * {2 * total}",
    ])
    if twiddle:
        # The twiddle table is shared by every batch row.
        lines.append("    tw_base = twiddle_ptr")
    lines.extend([
        f"    flat_lane = tl.arange(0, {n_flat})",
        "    perm = tl.load(idx_ptr + flat_lane) >> 2",
    ])
    if twiddle:
        lines.extend([
            "    swap = flat_lane ^ 1",
            "    even = (flat_lane % 2) == 0",
        ])
    lines.extend([
        f"    row_lane = tl.arange(0, {tile})",
        f"    word_lane = tl.arange(0, {half})",
        f"    for tile_id in range(pid, {tiles}, tl.num_programs(0)):",
        f"        tile_row = tile_id // {col_tiles}",
        f"        tile_col = tile_id % {col_tiles}",
        f"        row = tile_row * {tile} + row_lane",
        f"        word = tile_col * {half} + word_lane",
        f"        src_mask = ((row[:, None] < {R}) & (word[None, :] < {2 * C}) &",
        "                    (batch < nbatch))",
        f"        data = tl.load(src_base + row[:, None] * {2 * C} + word[None, :],",
        "                       mask=src_mask, other=0.0)",
        f"        flat_data = tl.reshape(data, ({n_flat},))",
    ])
    if twiddle:
        lines.extend([
            f"        tw = tl.load(tw_base + row[:, None] * {2 * C} + word[None, :],",
            "                      mask=src_mask, other=0.0)",
            f"        tw_flat = tl.reshape(tw, ({n_flat},))",
            "        x_partner = tl.gather(flat_data, swap, 0)",
            "        w_partner = tl.gather(tw_flat, swap, 0)",
            "        x_r = tl.where(even, flat_data, x_partner)",
            "        x_i = tl.where(even, x_partner, flat_data)",
            "        w_r = tl.where(even, tw_flat, w_partner)",
            "        w_i = tl.where(even, w_partner, tw_flat)",
            "        product = tl.where(even, x_r * w_r - x_i * w_i,",
            "                           x_r * w_i + x_i * w_r)",
            f"        data = tl.reshape(tl.gather(product, perm, 0), ({tile}, {half}))",
        ])
    else:
        lines.extend([
            f"        data = tl.reshape(tl.gather(flat_data, perm, 0), ({tile}, {half}))",
        ])
    lines.extend([
        f"        col = tile_col * {tile} + row_lane",
        f"        out_word = tile_row * {half} + word_lane",
        f"        dst_mask = ((col[:, None] < {C}) & (out_word[None, :] < {2 * R}) &",
        "                    (batch < nbatch))",
        f"        tl.store(dst_base + col[:, None] * {2 * R} + out_word[None, :],",
        "                 data, mask=dst_mask)",
    ])
    args = ["in_ptr"]
    if twiddle:
        args.append("twiddle_ptr")
    args += ["idx_ptr", "out_ptr", "nbatch"]
    return name, "\n".join(lines) + "\n", args


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
    if npu_tiled_reshape_pack(n1, n2, dtype):
        return _npu_reshape_pack_source(n1, n2, twiddle=False)
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
    if npu_tiled_reshape_pack(n1, n2, dtype):
        return _npu_reshape_pack_source(n1, n2, twiddle=True)
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


def _transpose3d_sliced_layout(s0: int, s1: int, s2: int, order: str) -> tuple[int, ...]:
    # Slices, rows, columns, source slice/column strides, destination slice/row strides.
    return {
        "021": (s0, s2, s1, s1 * s2, s2, s2 * s1, s1),
        "210": (s1, s2, s0, s2, s1 * s2, s0, s1 * s0),
        "201": (s0, s2, s1, s1 * s2, s2, s1, s0 * s1),
        "120": (s1, s2, s0, s2, s1 * s2, s2 * s0, s0),
    }[order]


def _build_maca_warp_transpose3d_kernel_source(
    s0: int, s1: int, s2: int, order: str, groups: int,
    *, tile: tuple[int, int] = (8, 8),
) -> tuple[str, str, list[str], int]:
    """Transpose each tile of complex FP32 pairs in registers."""
    slices, rows, cols, src_slice, src_col, dst_slice, dst_row = _transpose3d_sliced_layout(s0, s1, s2, order)
    tile_rows, tile_cols = tile
    tile_elements = tile_rows * tile_cols
    row_tiles = (rows + tile_rows - 1) // tile_rows
    col_tiles = (cols + tile_cols - 1) // tile_cols
    tiles_per_slice = row_tiles * col_tiles
    total_tiles = slices * tiles_per_slice
    if total_tiles % groups:
        raise ValueError("MACA warp transpose requires whole groups of tiles")
    tile_name = "8" if tile == (8, 8) else f"{tile_rows}x{tile_cols}"
    name = f"_maca_transpose3d_{order}_n{s0}_{s1}_{s2}_warp{tile_name}_g{groups}"
    partial_tile = rows % tile_rows or cols % tile_cols
    src_mask = f", (src_r < {rows}) & (src_c < {cols}), 0" if partial_tile else ""
    dst_mask = f", (dst_r < {rows}) & (dst_c < {cols})" if partial_tile else ""
    source = dedent(f"""
        @triton.jit
        def {name}(in_ptr, out_ptr, nbatch):
            tile = tl.program_id(0) * {groups} + tl.arange(0, {groups})
            batch = tl.program_id(2).to(tl.int64)
            slice_id = tile // {tiles_per_slice}
            row_tile = tile % {row_tiles}
            col_tile = (tile // {row_tiles}) % {col_tiles}
            lane = tl.arange(0, {tile_elements})
            src_r = row_tile[:, None] * {tile_rows} + (lane % {tile_rows})[None, :]
            src_c = col_tile[:, None] * {tile_cols} + (lane // {tile_rows})[None, :]
            base = batch * {s0 * s1 * s2}
            src = base + slice_id[:, None] * {src_slice} + src_c * {src_col} + src_r
            values = tl.load(tl.cast(in_ptr, tl.pointer_type(tl.int64)) + src{src_mask})
            tiles = tl.reshape(values, ({groups}, {tile_cols}, {tile_rows}))
            permuted = tl.reshape(tl.permute(tiles, (0, 2, 1)), ({groups}, {tile_elements}))
            dst_r = row_tile[:, None] * {tile_rows} + (lane // {tile_cols})[None, :]
            dst_c = col_tile[:, None] * {tile_cols} + (lane % {tile_cols})[None, :]
            dst = base + slice_id[:, None] * {dst_slice} + dst_r * {dst_row} + dst_c
            tl.store(tl.cast(out_ptr, tl.pointer_type(tl.int64)) + dst, permuted{dst_mask})
    """)
    return name, source, ["in_ptr", "out_ptr", "nbatch"], total_tiles // groups


def _build_tiled_transpose3d_tile_kernel_source(
    s0: int,
    s1: int,
    s2: int,
    order: str,
    dtype: str,
    tile: int | tuple[int, int] = 32,
    packed_complex: bool = False,
    pair_store: bool = False,
    pair: bool = False,
    vec_store: bool = False,
    tile_traversal: str = "col",
    component_pair: bool = False,
) -> tuple[str, str, list[str], int]:
    """Portable register-tile 3D axis permutation.

    Same slice/tile mapping as the vectorized v2 kernel, but the source tile is
    loaded with plain ``tl.load`` and flipped with ``tl.trans`` instead of
    ``ld/st.global.v2`` inline asm.  The source-contiguous axis is the tile
    inner dimension, so loads are coalesced; the register transpose makes the
    destination-contiguous axis the inner dimension of the store.  This keeps
    the coalescing benefit on toolchains that cannot lower the PTX inline asm.
    """
    if order not in {"021", "210", "201", "120"}:
        raise ValueError(f"unsupported 3D transpose order: {order}")
    suffix = _dtype_suffix(dtype)
    if dtype not in {"complex64", "complex128"}:
        raise ValueError("tiled 3D transpose requires a complex dtype")
    if packed_complex and dtype != "complex64":
        raise ValueError("packed 3D transpose requires complex64")
    if pair and dtype != "complex64":
        raise ValueError("paired tiled 3D transpose requires complex64")
    if vec_store and dtype != "complex128":
        raise ValueError("vector-store tiled 3D transpose requires complex128")
    if tile_traversal not in {"col", "row"}:
        raise ValueError("tile_traversal must be col or row")
    if isinstance(tile, tuple):
        tile_rows, tile_cols = tile
        if tile_rows <= 0 or tile_cols <= 0:
            raise ValueError("transpose tile dimensions must be positive")
        tile_name = f"{tile_rows}x{tile_cols}"
    else:
        tile_rows = tile_cols = tile
        tile_name = str(tile)
    zero = _zero_other(dtype)
    total_complex = s0 * s1 * s2
    total_float = total_complex * 2
    scalar_width = 1 if pair else 2
    total_scalar = total_complex * scalar_width
    (num_slices, rows, cols, src_slice_stride, src_col_stride,
     dst_slice_stride, dst_row_stride) = _transpose3d_sliced_layout(s0, s1, s2, order)
    num_tile_cols = (cols + tile_cols - 1) // tile_cols
    num_tile_rows = (rows + tile_rows - 1) // tile_rows
    tiles_per_slice = num_tile_cols * num_tile_rows
    if tile_traversal == "row":
        tile_row_expr = f"tile_in_slice % {num_tile_rows}"
        tile_col_expr = f"tile_in_slice // {num_tile_rows}"
    else:
        tile_row_expr = f"tile_in_slice // {num_tile_cols}"
        tile_col_expr = f"tile_in_slice % {num_tile_cols}"
    grid_x = num_slices * tiles_per_slice
    kernel_name = (
        f"_tiled_transpose3d_kernel_{order}_n{s0}_{s1}_{s2}_{suffix}_t{tile_name}_tile"
        + ("_u64" if packed_complex else "")
        + ("_pair" if pair or pair_store else "")
        + ("_vec" if vec_store else "")
        + ("_rmajor" if tile_traversal == "row" else "")
        + ("_components" if component_pair else "")
    )
    if pair:
        store_lines = (
            "tl.store(tl.cast(out_ptr, tl.pointer_type(tl.int64)) + dst_base, "
            "dst_pair, mask=store_mask)"
        )
    elif pair_store or vec_store:
        store_lines = (
            "dst_pair = tl.join(dst_r, dst_i)\n"
            "            pair_base = dst_base[:, :, None] + tl.arange(0, 2)[None, None, :]\n"
            "            tl.store(out_ptr + pair_base, dst_pair, mask=store_mask[:, :, None])"
        )
    else:
        store_lines = (
            "tl.store(out_ptr + dst_base, dst_r, mask=store_mask)\n"
            "            tl.store(out_ptr + dst_base + 1, dst_i, mask=store_mask)"
        )
    if component_pair:
        load_store = f"""
            component = tl.arange(0, 2)
            src_base = (
                pid_batch * {total_float}
                + slice_idx * {src_slice_stride} * 2
                + safe_cols[:, None] * {src_col_stride} * 2
                + safe_rows[None, :] * 2
            )
            values = tl.load(in_ptr + src_base[:, :, None] + component[None, None, :],
                             mask=load_mask[:, :, None], other={zero})
            dst_base = (
                pid_batch * {total_float}
                + slice_idx * {dst_slice_stride} * 2
                + safe_rows[:, None] * {dst_row_stride} * 2
                + safe_cols[None, :] * 2
            )
            store_mask = row_mask[:, None] & col_mask[None, :]
            tl.store(out_ptr + dst_base[:, :, None] + component[None, None, :],
                     tl.permute(values, (1, 0, 2)), mask=store_mask[:, :, None])
        """
    elif packed_complex:
        load_store = f"""
            src = in_ptr.to(tl.pointer_type(tl.uint64))
            dst = out_ptr.to(tl.pointer_type(tl.uint64))
            src_base = (
                pid_batch * {total_complex}
                + slice_idx * {src_slice_stride}
                + safe_cols[:, None] * {src_col_stride}
                + safe_rows[None, :]
            )
            values = tl.load(src + src_base, mask=load_mask, other=0)
            dst_base = (
                pid_batch * {total_complex}
                + slice_idx * {dst_slice_stride}
                + safe_rows[:, None] * {dst_row_stride}
                + safe_cols[None, :]
            )
            store_mask = row_mask[:, None] & col_mask[None, :]
            tl.store(dst + dst_base, tl.trans(values), mask=store_mask)
        """
    elif pair:
        load_store = f"""
            src_base = (
                pid_batch * {total_scalar}
                + slice_idx * {src_slice_stride} * {scalar_width}
                + safe_cols[:, None] * {src_col_stride} * {scalar_width}
                + safe_rows[None, :] * {scalar_width}
            )
            src_pair = tl.load(tl.cast(in_ptr, tl.pointer_type(tl.int64)) + src_base,
                               mask=load_mask, other=0)
            dst_pair = tl.trans(src_pair)
            dst_base = (
                pid_batch * {total_scalar}
                + slice_idx * {dst_slice_stride} * {scalar_width}
                + safe_rows[:, None] * {dst_row_stride} * {scalar_width}
                + safe_cols[None, :] * {scalar_width}
            )
            store_mask = row_mask[:, None] & col_mask[None, :]
            {store_lines}
        """
    else:
        load_store = f"""
            src_base = (
                pid_batch * {total_float}
                + slice_idx * {src_slice_stride} * 2
                + safe_cols[:, None] * {src_col_stride} * 2
                + safe_rows[None, :] * 2
            )
            src_r = tl.load(in_ptr + src_base, mask=load_mask, other={zero})
            src_i = tl.load(in_ptr + src_base + 1, mask=load_mask, other={zero})

            dst_r = tl.trans(src_r)
            dst_i = tl.trans(src_i)

            dst_base = (
                pid_batch * {total_float}
                + slice_idx * {dst_slice_stride} * 2
                + safe_rows[:, None] * {dst_row_stride} * 2
                + safe_cols[None, :] * 2
            )
            store_mask = row_mask[:, None] & col_mask[None, :]
            {store_lines}
        """
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
            tile_row = {tile_row_expr}
            tile_col = {tile_col_expr}

            row_offsets = tile_row * {tile_rows} + tl.arange(0, {tile_rows})
            col_offsets = tile_col * {tile_cols} + tl.arange(0, {tile_cols})
            row_mask = row_offsets < {rows}
            col_mask = col_offsets < {cols}
            load_mask = col_mask[:, None] & row_mask[None, :]
            safe_rows = tl.minimum(row_offsets, {rows - 1})
            safe_cols = tl.minimum(col_offsets, {cols - 1})

            # Load the tile with the source-contiguous axis (rows) innermost.
            {load_store}
        """
    )
    return kernel_name, source, ["in_ptr", "out_ptr", "nbatch"], grid_x


def _build_tiled_transpose3d_slice_group_kernel_source(
    s0: int,
    s1: int,
    s2: int,
    order: str,
    dtype: str,
    *,
    tile: int = 32,
    slice_group: int = 2,
) -> tuple[str, str, list[str], int]:
    """Emit a tiled transpose CTA that handles adjacent slices sequentially."""
    if order not in {"021", "210", "201", "120"}:
        raise ValueError(f"unsupported 3D transpose order: {order}")
    if dtype != "complex64":
        raise ValueError("sequential slice grouping currently supports complex64 only")
    if slice_group < 2 or slice_group & (slice_group - 1):
        raise ValueError("slice_group must be a power of two greater than one")

    total_complex = s0 * s1 * s2
    total_float = total_complex * 2
    transpose_descs = {
        "021": (s0, s2, s1, s1 * s2, s2, s2 * s1, s1),
        "210": (s1, s2, s0, s2, s1 * s2, s0, s1 * s0),
        "201": (s0, s2, s1, s1 * s2, s2, s1, s0 * s1),
        "120": (s1, s2, s0, s2, s1 * s2, s2 * s0, s0),
    }
    (
        num_slices,
        rows,
        cols,
        src_slice_stride,
        src_col_stride,
        dst_slice_stride,
        dst_row_stride,
    ) = transpose_descs[order]
    tile_cols = (cols + tile - 1) // tile
    tile_rows = (rows + tile - 1) // tile
    tiles_per_slice = tile_cols * tile_rows
    grid_x = ((num_slices + slice_group - 1) // slice_group) * tiles_per_slice
    kernel_name = (
        f"_tiled_transpose3d_kernel_{order}_n{s0}_{s1}_{s2}_f32_t{tile}"
        f"_tile_sliceg{slice_group}seq"
    )
    zero = _zero_other(dtype)
    source = dedent(
        f"""\
        @triton.jit
        def {kernel_name}(in_ptr, out_ptr, nbatch):
            pid_block = tl.program_id(0)
            pid_batch = tl.program_id(2)

            slice_group_idx = pid_block // {tiles_per_slice}
            tile_in_slice = pid_block % {tiles_per_slice}
            tile_row = tile_in_slice // {tile_cols}
            tile_col = tile_in_slice % {tile_cols}

            row_offsets = tile_row * {tile} + tl.arange(0, {tile})
            col_offsets = tile_col * {tile} + tl.arange(0, {tile})
            row_mask = row_offsets < {rows}
            col_mask = col_offsets < {cols}
            safe_rows = tl.minimum(row_offsets, {rows - 1})
            safe_cols = tl.minimum(col_offsets, {cols - 1})

            for group_offset in tl.static_range(0, {slice_group}):
                slice_idx = slice_group_idx * {slice_group} + group_offset
                slice_mask = slice_idx < {num_slices}
                src_base = (
                    pid_batch * {total_float}
                    + slice_idx * {src_slice_stride} * 2
                    + safe_cols[:, None] * {src_col_stride} * 2
                    + safe_rows[None, :] * 2
                )
                load_mask = slice_mask & col_mask[:, None] & row_mask[None, :]
                src_r = tl.load(in_ptr + src_base, mask=load_mask, other={zero})
                src_i = tl.load(in_ptr + src_base + 1, mask=load_mask, other={zero})
                dst_r = tl.trans(src_r)
                dst_i = tl.trans(src_i)

                safe_slice = tl.minimum(slice_idx, {num_slices - 1})
                dst_base = (
                    pid_batch * {total_float}
                    + safe_slice * {dst_slice_stride} * 2
                    + safe_rows[:, None] * {dst_row_stride} * 2
                    + safe_cols[None, :] * 2
                )
                store_mask = slice_mask & row_mask[:, None] & col_mask[None, :]
                tl.store(out_ptr + dst_base, dst_r, mask=store_mask)
                tl.store(out_ptr + dst_base + 1, dst_i, mask=store_mask)
        """
    )
    return kernel_name, source, ["in_ptr", "out_ptr", "nbatch"], grid_x


def _build_tiled_transpose_kernel_source(
    n0: int, n1: int, dtype: str, tile_size: int = 32,
    *, register_transpose: bool | None = None,
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
    use_register_transpose = (
        total_complex <= 128 * 1024
        if register_transpose is None else register_transpose
    )
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


def _build_packed_transpose_kernel_source(
    n0: int, n1: int, tile_size: int = 32
) -> tuple[str, str, list[str]]:
    """Transpose complex64 bitwise with one 64-bit load/store per element.

    No arithmetic is performed on the payload. Explicit masks preserve tails
    without duplicate stores, including an odd half-spectrum row pitch.
    """
    kernel_name = f"_packed_transpose_kernel_n{n0}_{n1}_t{tile_size}"
    source = dedent(
        f"""
        @triton.jit
        def {kernel_name}(in_ptr, out_ptr, nbatch):
            row = tl.program_id(1) * {tile_size} + tl.arange(0, {tile_size})
            col = tl.program_id(0) * {tile_size} + tl.arange(0, {tile_size})
            batch = tl.program_id(2)
            src = in_ptr.to(tl.pointer_type(tl.uint64))
            dst = out_ptr.to(tl.pointer_type(tl.uint64))
            value = tl.load(
                src + batch * {n0 * n1} + row[:, None] * {n1} + col[None, :],
                mask=(row[:, None] < {n0}) & (col[None, :] < {n1}) & (batch < nbatch),
                other=0,
            )
            value = tl.trans(value)
            tl.store(
                dst + batch * {n0 * n1} + col[:, None] * {n0} + row[None, :],
                value,
                mask=(col[:, None] < {n1}) & (row[None, :] < {n0}) & (batch < nbatch),
            )
        """
    )
    return kernel_name, source, ["in_ptr", "out_ptr", "nbatch"]

def _build_tiled_transpose3d_pair_slice_group_kernel_source(
    s0: int,
    s1: int,
    s2: int,
    order: str,
    *,
    tile: int = 16,
    slice_group: int = 2,
    tile_traversal: str = "row",
) -> tuple[str, str, list[str], int]:
    """Emit one CTA that sequentially transposes a group of paired-complex slices."""
    if order not in {"021", "210", "201", "120"}:
        raise ValueError(f"unsupported 3D transpose order: {order}")
    if slice_group < 2 or slice_group & (slice_group - 1):
        raise ValueError("slice_group must be a power of two greater than one")
    if tile_traversal not in {"col", "row"}:
        raise ValueError("tile_traversal must be col or row")

    transpose_descs = {
        "021": (s0, s2, s1, s1 * s2, s2, s2 * s1, s1),
        "210": (s1, s2, s0, s2, s1 * s2, s0, s1 * s0),
        "201": (s0, s2, s1, s1 * s2, s2, s1, s0 * s1),
        "120": (s1, s2, s0, s2, s1 * s2, s2 * s0, s0),
    }
    (
        num_slices,
        rows,
        cols,
        src_slice_stride,
        src_col_stride,
        dst_slice_stride,
        dst_row_stride,
    ) = transpose_descs[order]
    tile_cols = (cols + tile - 1) // tile
    tile_rows = (rows + tile - 1) // tile
    tiles_per_slice = tile_cols * tile_rows
    tile_row_expr = (
        f"tile_in_slice % {tile_rows}"
        if tile_traversal == "row"
        else f"tile_in_slice // {tile_cols}"
    )
    tile_col_expr = (
        f"tile_in_slice // {tile_rows}"
        if tile_traversal == "row"
        else f"tile_in_slice % {tile_cols}"
    )
    grid_x = ((num_slices + slice_group - 1) // slice_group) * tiles_per_slice
    total_complex = s0 * s1 * s2
    kernel_name = (
        f"_tiled_transpose3d_kernel_{order}_n{s0}_{s1}_{s2}_f32_t{tile}"
        f"_tile_pair_sliceg{slice_group}seq"
        + ("_rmajor" if tile_traversal == "row" else "")
    )
    source = dedent(
        f"""\
        @triton.jit
        def {kernel_name}(in_ptr, out_ptr, nbatch):
            pid_block = tl.program_id(0)
            pid_batch = tl.program_id(2)

            slice_group_idx = pid_block // {tiles_per_slice}
            tile_in_slice = pid_block % {tiles_per_slice}
            tile_row = {tile_row_expr}
            tile_col = {tile_col_expr}

            row_offsets = tile_row * {tile} + tl.arange(0, {tile})
            col_offsets = tile_col * {tile} + tl.arange(0, {tile})
            row_mask = row_offsets < {rows}
            col_mask = col_offsets < {cols}
            safe_rows = tl.minimum(row_offsets, {rows - 1})
            safe_cols = tl.minimum(col_offsets, {cols - 1})

            for group_offset in tl.static_range(0, {slice_group}):
                slice_idx = slice_group_idx * {slice_group} + group_offset
                slice_mask = slice_idx < {num_slices}
                src_base = (
                    pid_batch * {total_complex}
                    + slice_idx * {src_slice_stride}
                    + safe_cols[:, None] * {src_col_stride}
                    + safe_rows[None, :]
                )
                load_mask = slice_mask & col_mask[:, None] & row_mask[None, :]
                src_pair = tl.load(
                    tl.cast(in_ptr, tl.pointer_type(tl.int64)) + src_base,
                    mask=load_mask,
                    other=0,
                )
                dst_pair = tl.trans(src_pair)

                safe_slice = tl.minimum(slice_idx, {num_slices - 1})
                dst_base = (
                    pid_batch * {total_complex}
                    + safe_slice * {dst_slice_stride}
                    + safe_rows[:, None] * {dst_row_stride}
                    + safe_cols[None, :]
                )
                store_mask = slice_mask & row_mask[:, None] & col_mask[None, :]
                tl.store(
                    tl.cast(out_ptr, tl.pointer_type(tl.int64)) + dst_base,
                    dst_pair,
                    mask=store_mask,
                )
        """
    )
    return kernel_name, source, ["in_ptr", "out_ptr", "nbatch"], grid_x


def _build_tiled_transpose3d_pair_slice_row_group_kernel_source(
    s0: int,
    s1: int,
    s2: int,
    order: str,
    *,
    tile: int = 16,
    slice_group: int = 2,
    row_group: int = 2,
    tile_traversal: str = "row",
) -> tuple[str, str, list[str], int]:
    """Emit one CTA that serially transposes slice and row groups of 16x16 tiles."""
    if order not in {"021", "210", "201", "120"}:
        raise ValueError(f"unsupported 3D transpose order: {order}")
    for value, name in ((slice_group, "slice_group"), (row_group, "row_group")):
        if value < 1 or value & (value - 1):
            raise ValueError(f"{name} must be a positive power of two")
    if tile_traversal not in {"col", "row"}:
        raise ValueError("tile_traversal must be col or row")

    transpose_descs = {
        "021": (s0, s2, s1, s1 * s2, s2, s2 * s1, s1),
        "210": (s1, s2, s0, s2, s1 * s2, s0, s1 * s0),
        "201": (s0, s2, s1, s1 * s2, s2, s1, s0 * s1),
        "120": (s1, s2, s0, s2, s1 * s2, s2 * s0, s0),
    }
    (
        num_slices,
        rows,
        cols,
        src_slice_stride,
        src_col_stride,
        dst_slice_stride,
        dst_row_stride,
    ) = transpose_descs[order]
    tile_cols = (cols + tile - 1) // tile
    tile_rows = (rows + tile - 1) // tile
    row_groups = (tile_rows + row_group - 1) // row_group
    tiles_per_slice = tile_cols * row_groups
    if tile_traversal == "row":
        tile_row_group_expr = f"tile_in_slice % {row_groups}"
        tile_col_expr = f"tile_in_slice // {row_groups}"
    else:
        tile_row_group_expr = f"tile_in_slice // {tile_cols}"
        tile_col_expr = f"tile_in_slice % {tile_cols}"
    grid_x = (
        (num_slices + slice_group - 1) // slice_group
    ) * tiles_per_slice
    total_complex = s0 * s1 * s2
    kernel_name = (
        f"_tiled_transpose3d_kernel_{order}_n{s0}_{s1}_{s2}_f32_t{tile}"
        f"_tile_pair_sliceg{slice_group}seq_rowg{row_group}seq"
        + ("_rmajor" if tile_traversal == "row" else "")
    )
    source = dedent(
        f"""\
        @triton.jit
        def {kernel_name}(in_ptr, out_ptr, nbatch):
            pid_block = tl.program_id(0)
            pid_batch = tl.program_id(2)

            slice_group_idx = pid_block // {tiles_per_slice}
            tile_in_slice = pid_block % {tiles_per_slice}
            tile_row_group = {tile_row_group_expr}
            tile_col = {tile_col_expr}

            col_offsets = tile_col * {tile} + tl.arange(0, {tile})
            col_mask = col_offsets < {cols}
            safe_cols = tl.minimum(col_offsets, {cols - 1})

            for slice_group_offset in tl.static_range(0, {slice_group}):
                slice_idx = slice_group_idx * {slice_group} + slice_group_offset
                slice_mask = slice_idx < {num_slices}
                safe_slice = tl.minimum(slice_idx, {num_slices - 1})
                for row_group_offset in tl.static_range(0, {row_group}):
                    tile_row = tile_row_group * {row_group} + row_group_offset
                    row_offsets = tile_row * {tile} + tl.arange(0, {tile})
                    row_mask = row_offsets < {rows}
                    safe_rows = tl.minimum(row_offsets, {rows - 1})
                    src_base = (
                        pid_batch * {total_complex}
                        + slice_idx * {src_slice_stride}
                        + safe_cols[:, None] * {src_col_stride}
                        + safe_rows[None, :]
                    )
                    load_mask = slice_mask & col_mask[:, None] & row_mask[None, :]
                    src_pair = tl.load(
                        tl.cast(in_ptr, tl.pointer_type(tl.int64)) + src_base,
                        mask=load_mask,
                        other=0,
                    )
                    dst_pair = tl.trans(src_pair)

                    dst_base = (
                        pid_batch * {total_complex}
                        + safe_slice * {dst_slice_stride}
                        + safe_rows[:, None] * {dst_row_stride}
                        + safe_cols[None, :]
                    )
                    store_mask = slice_mask & row_mask[:, None] & col_mask[None, :]
                    tl.store(
                        tl.cast(out_ptr, tl.pointer_type(tl.int64)) + dst_base,
                        dst_pair,
                        mask=store_mask,
                    )
        """
    )
    return kernel_name, source, ["in_ptr", "out_ptr", "nbatch"], grid_x


def _build_tiled_transpose_kernel_source(
    n0: int, n1: int, dtype: str, tile_size: int = 32,
    *, register_transpose: bool | None = None,
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
    use_register_transpose = (
        total_complex <= 128 * 1024
        if register_transpose is None else register_transpose
    )
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


def _build_packed_transpose_kernel_source(
    n0: int, n1: int, tile_size: int = 32
) -> tuple[str, str, list[str]]:
    """Transpose complex64 bitwise with one 64-bit load/store per element.

    No arithmetic is performed on the payload. Explicit masks preserve tails
    without duplicate stores, including an odd half-spectrum row pitch.
    """
    kernel_name = f"_packed_transpose_kernel_n{n0}_{n1}_t{tile_size}"
    source = dedent(
        f"""
        @triton.jit
        def {kernel_name}(in_ptr, out_ptr, nbatch):
            row = tl.program_id(1) * {tile_size} + tl.arange(0, {tile_size})
            col = tl.program_id(0) * {tile_size} + tl.arange(0, {tile_size})
            batch = tl.program_id(2)
            src = in_ptr.to(tl.pointer_type(tl.uint64))
            dst = out_ptr.to(tl.pointer_type(tl.uint64))
            value = tl.load(
                src + batch * {n0 * n1} + row[:, None] * {n1} + col[None, :],
                mask=(row[:, None] < {n0}) & (col[None, :] < {n1}) & (batch < nbatch),
                other=0,
            )
            value = tl.trans(value)
            tl.store(
                dst + batch * {n0 * n1} + col[:, None] * {n0} + row[None, :],
                value,
                mask=(col[:, None] < {n1}) & (row[None, :] < {n0}) & (batch < nbatch),
            )
        """
    )
    return kernel_name, source, ["in_ptr", "out_ptr", "nbatch"]

def _build_tiled_transpose_kernel_source(
    n0: int, n1: int, dtype: str, tile_size: int = 32,
    *, register_transpose: bool | None = None,
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
    use_register_transpose = (
        total_complex <= 128 * 1024
        if register_transpose is None else register_transpose
    )
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


def _build_packed_transpose_kernel_source(
    n0: int, n1: int, tile_size: int = 32
) -> tuple[str, str, list[str]]:
    """Transpose complex64 bitwise with one 64-bit load/store per element.

    No arithmetic is performed on the payload. Explicit masks preserve tails
    without duplicate stores, including an odd half-spectrum row pitch.
    """
    kernel_name = f"_packed_transpose_kernel_n{n0}_{n1}_t{tile_size}"
    source = dedent(
        f"""
        @triton.jit
        def {kernel_name}(in_ptr, out_ptr, nbatch):
            row = tl.program_id(1) * {tile_size} + tl.arange(0, {tile_size})
            col = tl.program_id(0) * {tile_size} + tl.arange(0, {tile_size})
            batch = tl.program_id(2)
            src = in_ptr.to(tl.pointer_type(tl.uint64))
            dst = out_ptr.to(tl.pointer_type(tl.uint64))
            value = tl.load(
                src + batch * {n0 * n1} + row[:, None] * {n1} + col[None, :],
                mask=(row[:, None] < {n0}) & (col[None, :] < {n1}) & (batch < nbatch),
                other=0,
            )
            value = tl.trans(value)
            tl.store(
                dst + batch * {n0 * n1} + col[:, None] * {n0} + row[None, :],
                value,
                mask=(col[:, None] < {n1}) & (row[None, :] < {n0}) & (batch < nbatch),
            )
        """
    )
    return kernel_name, source, ["in_ptr", "out_ptr", "nbatch"]


def _build_paired_fp64_transpose_kernel_source(
    n0: int, n1: int, tile_size: int, *,
    flatten_pair: bool = False, column_first: bool = False,
) -> tuple[str, str, list[str]]:
    """Swap matrix axes while keeping both doubles of each value adjacent."""
    layout = "_flat" if flatten_pair else "_column" if column_first else ""
    kernel_name = f"_paired_transpose_kernel_n{n0}_{n1}_f64{layout}_t{tile_size}"
    row_axis, col_axis = (0, 1) if column_first else (1, 0)
    wide_index = ".to(tl.int64)" if 2 * n0 * n1 >= 2**31 else ""
    if flatten_pair:
        source = dedent(
            f"""
            @triton.jit
            def {kernel_name}(in_ptr, out_ptr, nbatch):
                tile_row = tl.program_id({row_axis}) * {tile_size}
                tile_col = tl.program_id({col_axis}) * {tile_size}
                batch = tl.program_id(2)
                batch_base = batch.to(tl.int64) * {2 * n0 * n1}
                in_ptr += batch_base
                out_ptr += batch_base
                row = tile_row + tl.arange(0, {tile_size})
                col = tile_col + tl.arange(0, {tile_size})
                src_pair = 2 * tile_col + tl.arange(0, {2 * tile_size})
                dst_pair = 2 * tile_row + tl.arange(0, {2 * tile_size})
                values = tl.load(
                    in_ptr + row{wide_index}[:, None] * {2 * n1} + src_pair[None, :],
                    (row[:, None] < {n0}) & (src_pair[None, :] < {2 * n1}) & (batch < nbatch),
                    other=0.0,
                )
                values = tl.reshape(values, ({tile_size}, {tile_size}, 2))
                values = tl.reshape(tl.permute(values, (1, 0, 2)), ({tile_size}, {2 * tile_size}))
                tl.store(
                    out_ptr + col{wide_index}[:, None] * {2 * n0} + dst_pair[None, :],
                    values,
                    (col[:, None] < {n1}) & (dst_pair[None, :] < {2 * n0}) & (batch < nbatch),
                )
            """
        )
        return kernel_name, source, ["in_ptr", "out_ptr", "nbatch"]

    input_base = "(in_ptr + batch_base)" if column_first else "in_ptr"
    output_base = "(out_ptr + batch_base)" if column_first else "out_ptr"
    batch_offset = "" if column_first else "batch_base + "
    source = dedent(
        f"""
        @triton.jit
        def {kernel_name}(in_ptr, out_ptr, nbatch):
            row = tl.program_id({row_axis}) * {tile_size} + tl.arange(0, {tile_size})
            col = tl.program_id({col_axis}) * {tile_size} + tl.arange(0, {tile_size})
            batch = tl.program_id(2)
            component = tl.arange(0, 2)[None, None, :]
            batch_base = batch.to(tl.int64) * {2 * n0 * n1}
            src = {batch_offset}(row{wide_index}[:, None, None] * {n1} + col[None, :, None]) * 2
            row_mask = row < {n0}
            col_mask = col < {n1}
            valid_batch = batch < nbatch
            values = tl.load(
                {input_base} + src + component,
                mask=row_mask[:, None, None] & col_mask[None, :, None] & valid_batch,
                other=0.0,
            )
            dst = {batch_offset}(col{wide_index}[:, None, None] * {n0} + row[None, :, None]) * 2
            tl.store(
                {output_base} + dst + component,
                tl.permute(values, (1, 0, 2)),
                mask=col_mask[:, None, None] & row_mask[None, :, None] & valid_batch,
            )
        """
    )
    return kernel_name, source, ["in_ptr", "out_ptr", "nbatch"]


def _build_flat_fp64_transpose3d_kernel_source(
    n0: int, n1: int, n2: int, order: str,
) -> tuple[str, str, list[str], int]:
    """Transpose interleaved FP64 pairs in contiguous rows of each plane."""
    if order == "201":
        rows, cols, layers = n0 * n1, n2, 1
    elif order == "120":
        rows, cols, layers = n0, n1 * n2, 1
    elif order == "021":
        rows, cols, layers = n1, n2, n0
    elif order == "210":
        rows, cols, layers = n0, n2, n1
    else:
        raise ValueError(f"unsupported flat FP64 3D transpose order: {order}")
    tile = 16
    row_tiles = (rows + tile - 1) // tile
    col_tiles = (cols + tile - 1) // tile
    plane_tiles = row_tiles * col_tiles
    batch_stride = 2 * rows * cols * layers
    if order == "021":
        src_layer_stride = dst_layer_stride = 2 * rows * cols
        src_row_stride, dst_row_stride = 2 * cols, 2 * rows
    else:
        src_layer_stride, dst_layer_stride = 2 * cols, 2 * rows
        src_row_stride, dst_row_stride = 2 * layers * cols, 2 * layers * rows
    wide_index = ".to(tl.int64)" if batch_stride >= 2**31 else ""
    name = f"_tiled_transpose3d_kernel_{order}_n{n0}_{n1}_{n2}_f64_flat_t{tile}"
    source = dedent(
        f"""
        @triton.jit
        def {name}(in_ptr, out_ptr, nbatch):
            block = tl.program_id(0)
            layer = block // {plane_tiles}
            tile_in_plane = block % {plane_tiles}
            tile_row = tile_in_plane // {col_tiles} * {tile}
            tile_col = tile_in_plane % {col_tiles} * {tile}
            batch = tl.program_id(2)
            batch_base = batch.to(tl.int64) * {batch_stride}
            src_base = in_ptr + batch_base + layer{wide_index} * {src_layer_stride}
            dst_base = out_ptr + batch_base + layer{wide_index} * {dst_layer_stride}
            row = tile_row + tl.arange(0, {tile})
            col = tile_col + tl.arange(0, {tile})
            src_pair = tile_col * 2 + tl.arange(0, {2 * tile})
            dst_pair = tile_row * 2 + tl.arange(0, {2 * tile})
            values = tl.load(
                src_base + row{wide_index}[:, None] * {src_row_stride} + src_pair[None, :],
                (row[:, None] < {rows}) & (src_pair[None, :] < {2 * cols}) & (batch < nbatch),
                other=0.0,
            )
            values = tl.reshape(values, ({tile}, {tile}, 2))
            values = tl.reshape(tl.permute(values, (1, 0, 2)), ({tile}, {2 * tile}))
            tl.store(
                dst_base + col{wide_index}[:, None] * {dst_row_stride} + dst_pair[None, :],
                values,
                (col[:, None] < {cols}) & (dst_pair[None, :] < {2 * rows}) & (batch < nbatch),
            )
        """
    )
    return name, source, ["in_ptr", "out_ptr", "nbatch"], plane_tiles * layers


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


def _build_complex_pair_transpose_kernel_source(
    n0: int, n1: int, tile_size: int = 32
) -> tuple[str, str, list[str]]:
    """Transpose interleaved complex64 with one on-chip vector permutation.

    This implements the existing 2D transpose IO contract. The caller retains
    the original FFT plan, scratch buffers and launch grid.
    """
    if n0 < 1 or n1 < 1 or tile_size < 1 or tile_size & (tile_size - 1):
        raise ValueError("transpose dimensions must be positive and tile_size a power of two")
    name = f"_complex_pair_transpose_n{n0}_{n1}_t{tile_size}"
    log_tile = tile_size.bit_length() - 1
    # Keep per-matrix coordinates narrow when even masked tile offsets fit.
    # The outer batch address remains 64-bit for every matrix size.
    index_type = "int32" if 2 * (n0 + tile_size) * (n1 + tile_size) < 2**31 else "int64"
    source = dedent(f"""
        @triton.jit
        def {name}(in_ptr, out_ptr, nbatch):
            row = tl.program_id(1).to(tl.{index_type}) * {tile_size} + tl.arange(0, {tile_size})
            batch = tl.program_id(2).to(tl.int64)
            word = tl.program_id(0).to(tl.{index_type}) * {2 * tile_size} + tl.arange(0, {2 * tile_size})
            input_batch = in_ptr + batch * {2 * n0 * n1}
            output_batch = out_ptr + batch * {2 * n0 * n1}
            source = row[:, None] * {2 * n1} + word[None, :]
            mask = (row[:, None] < {n0}) & (word[None, :] < {2 * n1}) & (batch < nbatch)
            point = tl.load(input_batch + source, mask, 0.0)
            point = tl.reshape(point, ({2 * tile_size * tile_size},))
            z = tl.arange(0, {2 * tile_size * tile_size})
            # All local coordinates are nonnegative powers-of-two ranges.
            # Signed // and % otherwise expand into division/conversion buffers
            # on the Ascend compiler, even for these constant denominators.
            index = ((z >> {log_tile + 1}) << 1) + (((z >> 1) & {tile_size - 1}) << {log_tile + 1}) + (z & 1)
            point = tl.gather(point, index, 0)
            point = tl.reshape(point, ({tile_size}, {2 * tile_size}))
            col = tl.program_id(0).to(tl.{index_type}) * {tile_size} + tl.arange(0, {tile_size})
            word = tl.program_id(1).to(tl.{index_type}) * {2 * tile_size} + tl.arange(0, {2 * tile_size})
            target = col[:, None] * {2 * n0} + word[None, :]
            mask = (col[:, None] < {n1}) & (word[None, :] < {2 * n0}) & (batch < nbatch)
            tl.store(output_batch + target, point, mask)
    """)
    return name, source, ["in_ptr", "out_ptr", "nbatch"]


__all__ = [
    "_build_complex_pair_transpose_kernel_source",
    "_build_reshape_pack_kernel_source",
    "_build_tiled_transpose3d_kernel_source",
    "_build_tiled_transpose3d_slice_group_kernel_source",
    "_build_tiled_transpose3d_pair_slice_group_kernel_source",
    "_build_tiled_transpose3d_pair_slice_row_group_kernel_source",
    "_build_tiled_transpose3d_tile_kernel_source",
    "_build_tiled_transpose3d_v2_kernel_source",
    "_build_tiled_transpose_kernel_source",
    "_build_twiddle_reshape_pack_kernel_source",
]
