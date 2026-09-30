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

"""Direct DFT kernel source generation."""

import os
from textwrap import dedent
from typing import Literal

from .kernels_common import _dtype_suffix, _maca_backend_active, lane_block_for
from .maca_tail_policy import resource_default


def _build_direct_dft_kernel_source(
    n: int,
    direction: Literal["forward", "inverse"],
    dtype: str,
    *,
    strided: bool = False,
    reduction_tile: int = 32,
    column_tile: int = 1,
) -> tuple[str, str, list[str]]:
    if reduction_tile not in (8, 16, 32):
        raise ValueError("DirectDFT reduction tile must be 8, 16 or 32")
    if column_tile not in (1, 32, 64) or (not strided and column_tile != 1):
        raise ValueError("DirectDFT column tile must be 1, 32 or 64 for a strided transform")
    block = lane_block_for(n)
    acc_dtype = "tl.float64" if dtype == "complex128" else "tl.float32"
    suffix = _dtype_suffix(dtype)
    prefix = "direct_idft" if direction == "inverse" else "direct_dft"
    kernel_name = (
        f"{prefix}_strided_kernel_n{n}_{suffix}_b{block}_c{column_tile}_r{reduction_tile}"
        if strided
        else f"{prefix}_kernel_n{n}_{suffix}_b{block}"
    )
    tree = (
        dtype == "complex128" and n <= 32 and _maca_backend_active()
        and os.environ.get("FLAGFFT_MACA_REAL_DFT_REDUCTION",
                           resource_default("REAL_DFT_REDUCTION") or "kahan") == "tree"
    )
    if tree:
        kernel_name += "_tree"
    in_placeholder = "base + j * outer_stride" if strided else f"pid_batch * {n} + j"
    out_placeholder = "base + k * outer_stride" if strided else f"pid_batch * {n} + k"
    param_extra = "            outer_stride,\n" if strided else ""
    base_init = (
        "            batch_index = pid_batch // outer_stride\n"
        f"            base = batch_index * ({n} * outer_stride) + "
        "(pid_batch - batch_index * outer_stride)\n"
        if strided
        else ""
    )
    compensation_init = ""
    accumulation = """
                acc_r += xr * wr - xi * wi
                acc_i += xr * wi + xi * wr
    """
    loop = f"""
            for j_base in tl.static_range(0, {n}, {reduction_tile}):
                j = j_base + tl.arange(0, {reduction_tile})[:, None]
                j_mask = j < {n}
                xr = tl.load(
                    in_ptr + ({in_placeholder}) * 2,
                    mask=j_mask,
                    other=0.0,
                )
                xi = tl.load(
                    in_ptr + ({in_placeholder}) * 2 + 1,
                    mask=j_mask,
                    other=0.0,
                )
                matrix_mask = j_mask & mask[None, :]
                matrix_offsets = j * {n} + k[None, :]
                wr = tl.load(dft_r_ptr + matrix_offsets, mask=matrix_mask, other=0.0)
                wi = tl.load(dft_i_ptr + matrix_offsets, mask=matrix_mask, other=0.0)
                acc_r += tl.sum(xr * wr - xi * wi, axis=0)
                acc_i += tl.sum(xr * wi + xi * wr, axis=0)
    """
    if dtype == "complex128" and not tree:
        compensation_init = f"""
            comp_r = tl.zeros(({block},), dtype={acc_dtype})
            comp_i = tl.zeros(({block},), dtype={acc_dtype})
        """
        accumulation = """
                term_r = xr * wr - xi * wi
                term_i = xr * wi + xi * wr
                corrected_r = term_r - comp_r
                corrected_i = term_i - comp_i
                next_r = acc_r + corrected_r
                next_i = acc_i + corrected_i
                comp_r = (next_r - acc_r) - corrected_r
                comp_i = (next_i - acc_i) - corrected_i
                acc_r = next_r
                acc_i = next_i
        """
        loop = f"""
            for j in tl.range(0, {n}):
                xr = tl.load(in_ptr + ({in_placeholder}) * 2)
                xi = tl.load(in_ptr + ({in_placeholder}) * 2 + 1)
                wr = tl.load(dft_r_ptr + k * {n} + j, mask=mask, other=0.0)
                wi = tl.load(dft_i_ptr + k * {n} + j, mask=mask, other=0.0)
                {accumulation}
        """
    if strided and column_tile > 1:
        coalesced_source = dedent(
            f"""
            @triton.jit
            def {kernel_name}(
                in_ptr,
                out_ptr,
                dft_r_ptr,
                dft_i_ptr,
                outer_stride,
                nbatch,
            ):
                pid = tl.program_id(0)
                columns_per_matrix = tl.cdiv(outer_stride, {column_tile})
                programs_per_matrix = {n} * columns_per_matrix
                matrix_index = pid // programs_per_matrix
                program_in_matrix = pid - matrix_index * programs_per_matrix
                k = program_in_matrix // columns_per_matrix
                column_tile_index = program_in_matrix - k * columns_per_matrix
                columns = column_tile_index * {column_tile} + tl.arange(0, {column_tile})
                base = matrix_index * ({n} * outer_stride)
                column_mask = columns < outer_stride
                acc_r = tl.zeros(({column_tile},), dtype={acc_dtype})
                acc_i = tl.zeros(({column_tile},), dtype={acc_dtype})
                for j in tl.static_range(0, {n}):
                    src = in_ptr + (base + j * outer_stride + columns) * 2
                    xr = tl.load(src, mask=column_mask, other=0.0)
                    xi = tl.load(src + 1, mask=column_mask, other=0.0)
                    twiddle_offset = j * {n} + k
                    wr = tl.load(dft_r_ptr + twiddle_offset)
                    wi = tl.load(dft_i_ptr + twiddle_offset)
                    acc_r += xr * wr - xi * wi
                    acc_i += xr * wi + xi * wr
                dst = out_ptr + (base + k * outer_stride + columns) * 2
                tl.store(dst, acc_r, mask=column_mask)
                tl.store(dst + 1, acc_i, mask=column_mask)
            """
        )
        return (
            kernel_name,
            coalesced_source,
            ["in_ptr", "out_ptr", "dft_r_ptr", "dft_i_ptr", "outer_stride", "nbatch"],
        )
    source = dedent(
        f"""
        @triton.jit
        def {kernel_name}(
            in_ptr,
            out_ptr,
            dft_r_ptr,
            dft_i_ptr,
{param_extra}            nbatch,
        ):
            pid_batch = tl.program_id(0)
            if pid_batch >= nbatch:
                return
{base_init}            k = tl.arange(0, {block})
            mask = k < {n}
            acc_r = tl.zeros(({block},), dtype={acc_dtype})
            acc_i = tl.zeros(({block},), dtype={acc_dtype})
            {compensation_init}
            {loop}
            dst = out_ptr + ({out_placeholder}) * 2
            tl.store(dst, acc_r, mask=mask)
            tl.store(dst + 1, acc_i, mask=mask)
        """
    )
    return (
        kernel_name,
        source,
        (
            ["in_ptr", "out_ptr", "dft_r_ptr", "dft_i_ptr", "outer_stride", "nbatch"]
            if strided
            else ["in_ptr", "out_ptr", "dft_r_ptr", "dft_i_ptr", "nbatch"]
        ),
    )


__all__ = [
    "_build_direct_dft_kernel_source",
]
