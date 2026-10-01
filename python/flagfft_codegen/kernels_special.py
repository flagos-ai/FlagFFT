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
    cube: bool = False,
    cube_transposed: bool = False,
) -> tuple[str, str, list[str]]:
    if cube or cube_transposed:
        if n != 64 or dtype != "complex64":
            raise ValueError("the Ascend Cube DFT kernel supports complex64 length 64 only")
        return _build_cube_dft_kernel_source(strided=strided, transpose_output=cube_transposed)
    block = lane_block_for(n)
    acc_dtype = "tl.float64" if dtype == "complex128" else "tl.float32"
    suffix = _dtype_suffix(dtype)
    prefix = "direct_idft" if direction == "inverse" else "direct_dft"
    kernel_name = (
        f"{prefix}_strided_kernel_n{n}_{suffix}_b{block}"
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
            for j_base in tl.static_range(0, {n}, 32):
                j = j_base + tl.arange(0, 32)[:, None]
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


def _build_cube_dft_kernel_source(
    *, strided: bool = False, transpose_output: bool = False
) -> tuple[str, str, list[str]]:
    """Build the measured 64-point FP32 Cube matmul kernel."""
    if transpose_output:
        if strided:
            raise ValueError("the transposed Cube DFT path requires contiguous input")
        kernel_name = "direct_dft_cube_transposed_kernel_n64_complex64_b16"
        source = dedent(
            f"""
            @triton.jit
            def {kernel_name}(in_ptr, out_ptr, dft_r_ptr, dft_i_ptr, nbatch):
                line_block = tl.program_id(0)
                line = line_block * 16 + tl.arange(0, 16)
                out = tl.arange(0, 64)
                k = tl.arange(0, 16)
                base = line * 64
                acc_r = tl.zeros((16, 64), dtype=tl.float32)
                acc_i = tl.zeros((16, 64), dtype=tl.float32)
                for kk in tl.static_range(4):
                    j = kk * 16 + k
                    line_mask = line < nbatch
                    xr = tl.load(in_ptr + (base[:, None] + j[None, :]) * 2,
                                 mask=line_mask[:, None], other=0.0)
                    xi = tl.load(in_ptr + (base[:, None] + j[None, :]) * 2 + 1,
                                 mask=line_mask[:, None], other=0.0)
                    wr = tl.load(dft_r_ptr + j[:, None] * 64 + out[None, :])
                    wi = tl.load(dft_i_ptr + j[:, None] * 64 + out[None, :])
                    acc_r += tl.dot(xr, wr) - tl.dot(xi, wi)
                    acc_i += tl.dot(xr, wi) + tl.dot(xi, wr)
                batch_index = line // 64
                line_offset = line % 64
                dst = out_ptr + (batch_index[None, :] * 4096 +
                                 out[:, None] * 64 + line_offset[None, :]) * 2
                store_mask = line_mask[None, :]
                tl.store(dst, tl.trans(acc_r), mask=store_mask)
                tl.store(dst + 1, tl.trans(acc_i), mask=store_mask)
            """
        )
        return kernel_name, source, ["in_ptr", "out_ptr", "dft_r_ptr", "dft_i_ptr", "nbatch"]

    kernel_name = (
        "direct_dft_cube_strided_kernel_n64_complex64_b16"
        if strided
        else "direct_dft_cube_kernel_n64_complex64_b16"
    )
    base = (
        "            batch_index = line // outer_stride\n"
        "            base = batch_index * (64 * outer_stride) + "
        "(line - batch_index * outer_stride)\n"
        if strided
        else "            base = line * 64\n"
    )
    input_offset = (
        "base[:, None] + j[None, :] * outer_stride"
        if strided
        else "base[:, None] + j[None, :]"
    )
    output_offset = (
        "base[:, None] + out[None, :] * outer_stride"
        if strided
        else "base[:, None] + out[None, :]"
    )
    extra = "            outer_stride,\n" if strided else ""
    source = dedent(
        f"""
        @triton.jit
        def {kernel_name}(in_ptr, out_ptr, dft_r_ptr, dft_i_ptr,
{extra}            nbatch):
            line_block = tl.program_id(0)
            line = line_block * 16 + tl.arange(0, 16)
            out = tl.arange(0, 64)
            k = tl.arange(0, 16)
{base}            acc_r = tl.zeros((16, 64), dtype=tl.float32)
            acc_i = tl.zeros((16, 64), dtype=tl.float32)
            for kk in tl.static_range(4):
                j = kk * 16 + k
                line_mask = line < nbatch
                xr = tl.load(in_ptr + ({input_offset}) * 2,
                             mask=line_mask[:, None], other=0.0)
                xi = tl.load(in_ptr + ({input_offset}) * 2 + 1,
                             mask=line_mask[:, None], other=0.0)
                wr = tl.load(dft_r_ptr + j[:, None] * 64 + out[None, :])
                wi = tl.load(dft_i_ptr + j[:, None] * 64 + out[None, :])
                acc_r += tl.dot(xr, wr) - tl.dot(xi, wi)
                acc_i += tl.dot(xr, wi) + tl.dot(xi, wr)
            dst = out_ptr + ({output_offset}) * 2
            tl.store(dst, acc_r, mask=line_mask[:, None])
            tl.store(dst + 1, acc_i, mask=line_mask[:, None])
        """
    )
    arg_names = (
        ["in_ptr", "out_ptr", "dft_r_ptr", "dft_i_ptr", "outer_stride", "nbatch"]
        if strided
        else ["in_ptr", "out_ptr", "dft_r_ptr", "dft_i_ptr", "nbatch"]
    )
    return kernel_name, source, arg_names


__all__ = [
    "_build_direct_dft_kernel_source",
]
