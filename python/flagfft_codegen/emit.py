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

"""Kernel source emission: per-family builders plus the registry-driven dispatch."""

import ast
import importlib.util
import json
import os
import sys
from pathlib import Path
from textwrap import dedent
from typing import Any
from functools import wraps
from .maca_tail_policy import maca_tail_kernel_scope, rader_vector_io_default, variant_suffix
from .artifacts import write_text_atomic

from .kernels_common import (
    LeafPlan,
    _dtype_suffix,
    _hcu_backend_active,
    _declared_backend,
    _ix_backend_active,
    _maca_backend_active,
    _maca_knob,
    _npu_backend_active,
    _zero_other,
    codelet_radices_for,
    direct_dft_batch_tile,
    emitted_leaf_factors,
    lane_block_for,
)
from .kernels_layout import (
    _build_complex_pair_transpose_kernel_source,
    _build_flat_fp64_transpose3d_kernel_source,
    _build_packed_transpose_kernel_source,
    _build_paired_fp64_transpose_kernel_source,
    _build_reshape_pack_kernel_source,
    _build_tiled_transpose3d_kernel_source,
    _build_tiled_transpose3d_slice_group_kernel_source,
    _build_tiled_transpose3d_pair_slice_group_kernel_source,
    _build_tiled_transpose3d_pair_slice_row_group_kernel_source,
    _build_tiled_transpose3d_tile_kernel_source,
    _build_tiled_transpose3d_v2_kernel_source,
    _build_tiled_transpose_kernel_source,
    _build_twiddle_reshape_pack_kernel_source,
    npu_reshape_pack_grid,
    npu_tiled_reshape_pack,
)
from .kernels_leaf import _build_leaf_kernel_source_for_io
from .kernels_real import (
    _npu_packed_boundary_groups,
    _npu_packed_boundary_direct_midpoint,
    _build_c2r_packed_preprocess_kernel_source,
    _build_real_pair_kernel_source,
    _build_compact_to_hermitian_full_kernel_source,
    _build_complex_to_real_kernel_source,
    _build_r2c_half_pack_kernel_source,
    _build_r2c_packed_postprocess_kernel_source,
    _build_real_to_complex_kernel_source,
)
from .kernels_special import _build_cube_dft_2d_kernel_source, _build_direct_dft_kernel_source
from .kernels_real import _build_real_direct_dft_kernel_source
from .kernels_stockham import build_stockham_stage
from .backend_profile import current_profile
from .metadata import _metadata, _module_source, _signature
from .registry import (
    DIRECT_DFT,
    FOUR_STEP_COL_NAMES,
    FOUR_STEP_ROW_NAMES,
    STOCKHAM,
    kernel_spec,
    module_name_for,
)

_BLUESTEIN_BLOCK = 256
_BLUESTEIN_NUM_WARPS = 4
_BLUESTEIN_NUM_STAGES = 4
_RADER_BLOCK = 256
_RADER_NUM_WARPS = 4
_RADER_NUM_STAGES = 4
_RESHAPE_NUM_WARPS = 4
_RESHAPE_NUM_STAGES = 1


def _bluestein_kernel_source(kind: str, dtype: str) -> tuple[str, str, list[str]]:
    zero = _zero_other(dtype)
    npu = current_profile().backend == "npu" and dtype == "complex64"
    div_cast = "tl.cast(m, tl.float64)" if dtype == "complex128" else "m"
    if kind == "bluestein_prepare":
        if npu:
            return (
                "_bluestein_prepare_kernel",
                dedent(
                    f"""
                    @triton.jit
                    def _bluestein_prepare_kernel(in_ptr, chirp_ptr, out_ptr, n, m, nbatch):
                        pid_block = tl.program_id(0)
                        pid_batch = tl.program_id(1)
                        base = pid_block * 512
                        lane = tl.arange(0, 512)
                        lane2 = tl.arange(0, 1024)
                        in_mask = lane2 < 2 * (n - base)
                        out_mask = lane2 < 2 * (m - base)
                        src = tl.load(in_ptr + (pid_batch * n + base) * 2 + lane2, mask=in_mask, other={zero})
                        xr = tl.gather(src, lane * 2, 0)
                        xi = tl.gather(src, lane * 2 + 1, 0)
                        chirp = tl.load(chirp_ptr + base * 2 + lane2, mask=in_mask, other={zero})
                        cr = tl.gather(chirp, lane * 2, 0)
                        ci = tl.gather(chirp, lane * 2 + 1, 0)
                        yr, yi = _cmul(xr, xi, cr, ci)
                        packed = tl.zeros((1024,), tl.float32)
                        packed = tle.dsa.insert_slice(packed, yr, [0], [512], [1])
                        packed = tle.dsa.insert_slice(packed, yi, [512], [512], [1])
                        index = lane2 // 2 + (lane2 % 2) * 512
                        result = tl.gather(packed, index, 0)
                        tl.store(out_ptr + (pid_batch * m + base) * 2 + lane2, result, mask=out_mask)
                    """
                ),
                ["in_ptr", "chirp_ptr", "out_ptr", "n", "m", "nbatch"],
            )
        return (
            "_bluestein_prepare_kernel",
            dedent(
                f"""
                @triton.jit
                def _bluestein_prepare_kernel(
                    in_ptr,
                    chirp_ptr,
                    out_ptr,
                    n,
                    m,
                    nbatch,
                ):
                    pid_block = tl.program_id(0)
                    pid_batch = tl.program_id(1)
                    offsets = pid_block * 256 + tl.arange(0, 256)
                    mask = offsets < m
                    in_mask = mask & (offsets < n)

                    src = in_ptr + (pid_batch * n + offsets) * 2
                    xr = tl.load(src, mask=in_mask, other={zero})
                    xi = tl.load(src + 1, mask=in_mask, other={zero})
                    cr = tl.load(chirp_ptr + offsets * 2, mask=in_mask, other={zero})
                    ci = tl.load(chirp_ptr + offsets * 2 + 1, mask=in_mask, other={zero})
                    yr, yi = _cmul(xr, xi, cr, ci)

                    dst = out_ptr + (pid_batch * m + offsets) * 2
                    tl.store(dst, yr, mask=mask)
                    tl.store(dst + 1, yi, mask=mask)
                """
            ),
            ["in_ptr", "chirp_ptr", "out_ptr", "n", "m", "nbatch"],
        )
    if kind == "bluestein_pointwise":
        if npu:
            return (
                "_bluestein_pointwise_kernel",
                dedent(
                    f"""
                    @triton.jit
                    def _bluestein_pointwise_kernel(a_ptr, b_ptr, out_ptr, m, nbatch):
                        pid_block = tl.program_id(0)
                        pid_batch = tl.program_id(1)
                        base = pid_block * 512
                        lane = tl.arange(0, 512)
                        lane2 = tl.arange(0, 1024)
                        mask = lane2 < 2 * (m - base)
                        a = tl.load(a_ptr + (pid_batch * m + base) * 2 + lane2, mask=mask, other={zero})
                        b = tl.load(b_ptr + base * 2 + lane2, mask=mask, other={zero})
                        ar = tl.gather(a, lane * 2, 0)
                        ai = tl.gather(a, lane * 2 + 1, 0)
                        br = tl.gather(b, lane * 2, 0)
                        bi = tl.gather(b, lane * 2 + 1, 0)
                        pr, pi = _cmul(ar, ai, br, bi)
                        packed = tl.zeros((1024,), tl.float32)
                        packed = tle.dsa.insert_slice(packed, pr, [0], [512], [1])
                        packed = tle.dsa.insert_slice(packed, -pi, [512], [512], [1])
                        index = lane2 // 2 + (lane2 % 2) * 512
                        result = tl.gather(packed, index, 0)
                        tl.store(out_ptr + (pid_batch * m + base) * 2 + lane2, result, mask=mask)
                    """
                ),
                ["a_ptr", "b_ptr", "out_ptr", "m", "nbatch"],
            )
        return (
            "_bluestein_pointwise_kernel",
            dedent(
                f"""
                @triton.jit
                def _bluestein_pointwise_kernel(
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
    if kind == "bluestein_finalize":
        if npu:
            return (
                "_bluestein_finalize_kernel",
                dedent(
                    f"""
                    @triton.jit
                    def _bluestein_finalize_kernel(in_ptr, chirp_ptr, out_ptr, n, m, nbatch):
                        pid_block = tl.program_id(0)
                        pid_batch = tl.program_id(1)
                        base = pid_block * 512
                        lane = tl.arange(0, 512)
                        lane2 = tl.arange(0, 1024)
                        mask = lane2 < 2 * (n - base)
                        src = tl.load(in_ptr + (pid_batch * m + base) * 2 + lane2, mask=mask, other={zero})
                        xr = tl.gather(src, lane * 2, 0) / {div_cast}
                        xi = -tl.gather(src, lane * 2 + 1, 0) / {div_cast}
                        chirp = tl.load(chirp_ptr + base * 2 + lane2, mask=mask, other={zero})
                        cr = tl.gather(chirp, lane * 2, 0)
                        ci = tl.gather(chirp, lane * 2 + 1, 0)
                        yr, yi = _cmul(xr, xi, cr, ci)
                        packed = tl.zeros((1024,), tl.float32)
                        packed = tle.dsa.insert_slice(packed, yr, [0], [512], [1])
                        packed = tle.dsa.insert_slice(packed, yi, [512], [512], [1])
                        index = lane2 // 2 + (lane2 % 2) * 512
                        result = tl.gather(packed, index, 0)
                        tl.store(out_ptr + (pid_batch * n + base) * 2 + lane2, result, mask=mask)
                    """
                ),
                ["in_ptr", "chirp_ptr", "out_ptr", "n", "m", "nbatch"],
            )
        return (
            "_bluestein_finalize_kernel",
            dedent(
                f"""
                @triton.jit
                def _bluestein_finalize_kernel(
                    in_ptr,
                    chirp_ptr,
                    out_ptr,
                    n,
                    m,
                    nbatch,
                ):
                    pid_block = tl.program_id(0)
                    pid_batch = tl.program_id(1)
                    offsets = pid_block * 256 + tl.arange(0, 256)
                    mask = offsets < n

                    src = in_ptr + (pid_batch * m + offsets) * 2
                    xr = tl.load(src, mask=mask, other={zero}) / {div_cast}
                    xi = -tl.load(src + 1, mask=mask, other={zero}) / {div_cast}
                    cr = tl.load(chirp_ptr + offsets * 2, mask=mask, other={zero})
                    ci = tl.load(chirp_ptr + offsets * 2 + 1, mask=mask, other={zero})
                    yr, yi = _cmul(xr, xi, cr, ci)

                    dst = out_ptr + (pid_batch * n + offsets) * 2
                    tl.store(dst, yr, mask=mask)
                    tl.store(dst + 1, yi, mask=mask)
                """
            ),
            ["in_ptr", "chirp_ptr", "out_ptr", "n", "m", "nbatch"],
        )
    raise ValueError(f"unsupported JIT kernel kind: {kind}")


def _emit_bluestein_jit_kernel(
    *,
    kernel: str,
    n: int,
    m: int,
    dtype: str = "complex64",
    out_dir: Path,
) -> dict[str, Any]:
    kernel_name, kernel_source, arg_names = _bluestein_kernel_source(kernel, dtype)
    suffix = _dtype_suffix(dtype)
    module_name = f"flagfft_jit_{kernel}_n{n}_m{m}_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    module_path = out_dir / f"{module_name}.py"
    write_text_atomic(module_path, _module_source(kernel_source))
    metadata = {
        "module_path": str(module_path),
        "kernel_name": kernel_name,
        "signature": _signature(arg_names, dtype),
        "num_warps": _BLUESTEIN_NUM_WARPS,
        "num_stages": _BLUESTEIN_NUM_STAGES,
        "batch_per_block": 1,
        "arg_names": arg_names,
        "kernel_type": kernel,
        "dtype": dtype,
        "bluestein_n": int(n),
        "bluestein_m": int(m),
        "block": 512 if current_profile().backend == "npu" and dtype == "complex64" else _BLUESTEIN_BLOCK,
    }
    write_text_atomic(out_dir / f"{module_name}.json", json.dumps(metadata, sort_keys=True))
    return metadata


def _rader_kernel_source(
    kind: str, n: int, m: int, dtype: str
) -> tuple[str, str, list[str]]:
    zero = _zero_other(dtype)
    div_cast = "tl.cast(m, tl.float64)" if dtype == "complex128" else "m"
    vector_io = (dtype in {"complex64", "complex128"} and _maca_backend_active() and
                 os.environ.get("FLAGFFT_MACA_RADER_VEC_IO",
                                "1" if rader_vector_io_default() else "0") == "1")
    prepare_load = (
        "pair = tl.load(src[:, None] + tl.arange(0, 2)[None, :], "
        "mask=mask[:, None], other=0.0)\n"
        "                    xr, xi = tl.split(pair)"
        if vector_io else
        f"xr = tl.load(src, mask=mask, other={zero})\n"
        f"                    xi = tl.load(src + 1, mask=mask, other={zero})"
    )
    finalize_store = (
        "tl.store(dst[:, None] + tl.arange(0, 2)[None, :], "
        "tl.join(yr, yi), mask=mask[:, None])"
        if vector_io else
        "tl.store(dst, yr, mask=mask)\n"
        "                    tl.store(dst + 1, yi, mask=mask)"
    )
    if kind == "rader_prepare":
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
                    {prepare_load}

                    dst = out_ptr + (pid_batch * m + offsets) * 2
                    tl.store(dst, xr, mask=mask)
                    tl.store(dst + 1, xi, mask=mask)
                """
            ),
            ["in_ptr", "idx_ptr", "out_ptr", "n", "m", "nbatch"],
        )
    if kind == "rader_pointwise":
        return (
            "_rader_pointwise_kernel",
            dedent(
                f"""
                @triton.jit
                def _rader_pointwise_kernel(
                    a_ptr,
                    b_ptr,
                    out_ptr,
                    input_ptr,
                    dc_ptr,
                    n,
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

                    # A[0] is the sum of the permuted nonzero input.
                    # Reuse it before the inverse convolution overwrites A.
                    if pid_block == 0:
                        a0 = a_ptr + pid_batch * m * 2
                        x0 = input_ptr + pid_batch * n * 2
                        dc = dc_ptr + pid_batch * n * 2
                        tl.store(dc, tl.load(a0) + tl.load(x0))
                        tl.store(dc + 1, tl.load(a0 + 1) + tl.load(x0 + 1))
                """
            ),
            ["a_ptr", "b_ptr", "out_ptr", "input_ptr", "dc_ptr", "n", "m", "nbatch"],
        )
    if kind == "rader_finalize":
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
                    yr = x0r + cr
                    yi = x0i + ci

                    dst_index = tl.load(idx_ptr + offsets, mask=mask, other=0)
                    dst = out_ptr + (pid_batch * n + dst_index) * 2
                    {finalize_store}

                """
            ),
            ["input_ptr", "conv_ptr", "idx_ptr", "out_ptr", "n", "m", "nbatch"],
        )
    raise ValueError(f"unsupported JIT kernel kind: {kind}")


def _emit_rader_jit_kernel(
    *,
    kernel: str,
    n: int,
    m: int,
    dtype: str = "complex64",
    out_dir: Path,
) -> dict[str, Any]:
    kernel_name, kernel_source, arg_names = _rader_kernel_source(kernel, n, m, dtype)
    suffix = _dtype_suffix(dtype)
    module_name = f"flagfft_jit_{kernel}_n{n}_m{m}_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    module_path = out_dir / f"{module_name}.py"
    write_text_atomic(module_path, _module_source(kernel_source))
    metadata = {
        "module_path": str(module_path),
        "kernel_name": kernel_name,
        "signature": _signature(arg_names, dtype),
        "num_warps": _RADER_NUM_WARPS,
        "num_stages": _RADER_NUM_STAGES,
        "batch_per_block": 1,
        "arg_names": arg_names,
        "kernel_type": kernel,
        "dtype": dtype,
        "rader_n": int(n),
        "rader_m": int(m),
        "block": _RADER_BLOCK,
    }
    write_text_atomic(out_dir / f"{module_name}.json", json.dumps(metadata, sort_keys=True))
    return metadata


def _emit_reshape_jit_kernel(
    *,
    kernel: str,
    n1: int,
    n2: int,
    dtype: str = "complex64",
    out_dir: Path,
) -> dict[str, Any]:
    if kernel == "reshape_pack":
        kernel_name, kernel_source, arg_names = _build_reshape_pack_kernel_source(
            n1, n2, dtype
        )
    elif kernel == "twiddle_reshape_pack":
        (
            kernel_name,
            kernel_source,
            arg_names,
        ) = _build_twiddle_reshape_pack_kernel_source(n1, n2, dtype)
    else:
        raise ValueError(f"unsupported reshape kernel kind: {kernel}")

    suffix = _dtype_suffix(dtype)
    module_name = f"flagfft_jit_{kernel}_n{n1}_{n2}_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    module_path = out_dir / f"{module_name}.py"
    write_text_atomic(module_path, _module_source(kernel_source))
    metadata = {
        "module_path": str(module_path),
        "kernel_name": kernel_name,
        "signature": _signature(arg_names, dtype),
        "num_warps": _RESHAPE_NUM_WARPS,
        "num_stages": _RESHAPE_NUM_STAGES,
        "batch_per_block": 1,
        "arg_names": arg_names,
        "kernel_type": kernel,
        "dtype": dtype,
        "reshape_n1": int(n1),
        "reshape_n2": int(n2),
        "block": 256,
    }
    if npu_tiled_reshape_pack(n1, n2, dtype):
        metadata["grid_x_override"] = npu_reshape_pack_grid(n1, n2)
    write_text_atomic(out_dir / f"{module_name}.json", json.dumps(metadata, sort_keys=True))
    return metadata


def _emit_r2c_pointwise_jit_kernel(
    *,
    kernel: str,
    n: int,
    dtype: str,
    out_dir: Path,
    target: str = "",
    perm_form: str = "outer",
) -> dict[str, Any]:
    elements_per_program = 256
    if target.startswith("npu:") and dtype == "complex64" and n >= 512:
        elements_per_program = int(os.getenv("FLAGFFT_NPU_REAL_BOUNDARY_TILE", "256"))
        if elements_per_program not in {256, 512, 1024, 2048}:
            raise ValueError("NPU real boundary tile must be 256, 512, 1024 or 2048")
    num_warps = _RESHAPE_NUM_WARPS
    if target.startswith("npu:") and n == 256:
        # AIV conversion kernels are sensitive to work per block. Keep the
        # profiled one-row/four-thread layout as the default; expose narrow
        # experiment knobs so rows and block threads can be measured without
        # changing other targets or kernel families.
        rows_text = os.getenv("FLAGFFT_NPU_3D_REAL_POINTWISE_ROWS", "1")
        warps_text = os.getenv("FLAGFFT_NPU_3D_REAL_POINTWISE_WARPS", "4")
        try:
            rows_per_program = int(rows_text)
            num_warps = int(warps_text)
        except ValueError as exc:
            raise ValueError("NPU 256-point real pointwise rows/warps must be integers") from exc
        if rows_per_program not in {1, 2, 4, 8}:
            raise ValueError("NPU 256-point real pointwise rows must be one of 1, 2, 4, 8")
        if num_warps not in {4, 8, 16, 32}:
            raise ValueError("NPU 256-point real pointwise warps must be one of 4, 8, 16, 32")
        elements_per_program *= rows_per_program
    maca_paired = perm_form in {"maca_paired_real_boundary_w2", "maca_paired_real_boundary_w4"}
    if maca_paired:
        if not (_maca_backend_active() and current_profile().device_arch == "102" and n == 663000
                and dtype in {"complex64", "complex128"}
                and kernel in {"r2c_packed_postprocess", "c2r_packed_preprocess"}):
            raise ValueError("the paired real boundary requires a packed 663000-point C550 FFT")
        from .kernels_real import _build_maca_paired_real_boundary

        kernel_name, kernel_source, arg_names, elements_per_program = _build_maca_paired_real_boundary(
            n, dtype, inverse=kernel == "c2r_packed_preprocess",
        )
        rows_per_block = 1
        num_warps = int(perm_form[-1])
    elif kernel in {"r2c_pair_pack", "r2c_pair_unpack", "c2r_pair_pack", "c2r_pair_unpack"}:
        kernel_name, kernel_source, arg_names, rows_per_block = _build_real_pair_kernel_source(n, dtype, kernel)
    elif kernel == "real_to_complex":
        (
            kernel_name,
            kernel_source,
            arg_names,
            rows_per_block,
        ) = _build_real_to_complex_kernel_source(n, dtype, elements_per_program)
    elif kernel == "r2c_half_pack":
        (
            kernel_name,
            kernel_source,
            arg_names,
            rows_per_block,
        ) = _build_r2c_half_pack_kernel_source(n, dtype, elements_per_program)
    elif kernel == "r2c_packed_postprocess":
        (
            kernel_name,
            kernel_source,
            arg_names,
            rows_per_block,
        ) = _build_r2c_packed_postprocess_kernel_source(n, dtype)
    elif kernel == "c2r_packed_preprocess":
        (
            kernel_name,
            kernel_source,
            arg_names,
            rows_per_block,
        ) = _build_c2r_packed_preprocess_kernel_source(n, dtype)
    elif kernel == "compact_to_hermitian_full":
        (
            kernel_name,
            kernel_source,
            arg_names,
            rows_per_block,
        ) = _build_compact_to_hermitian_full_kernel_source(n, dtype, elements_per_program)
    elif kernel == "complex_to_real":
        (
            kernel_name,
            kernel_source,
            arg_names,
            rows_per_block,
        ) = _build_complex_to_real_kernel_source(n, dtype, elements_per_program)
    else:
        raise ValueError(f"unsupported R2C pointwise kernel kind: {kernel}")

    dtype_tag = _dtype_suffix(dtype)
    module_name = f"flagfft_jit_{kernel}_n{n}_{dtype_tag}"
    if maca_paired:
        module_name += f"_paired_w{num_warps}"
    out_dir.mkdir(parents=True, exist_ok=True)
    module_path = out_dir / f"{module_name}.py"
    write_text_atomic(module_path, _module_source(kernel_source))

    sys.path.insert(0, str(module_path.parent))
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load generated kernel module {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    arg_names = list(getattr(module, kernel_name).arg_names)
    grid_columns = (n // 2 + 1 if kernel in {"r2c_half_pack", "r2c_packed_postprocess"}
                    else n // 2 if kernel == "c2r_packed_preprocess" else n)
    if (target.startswith("npu:") and dtype == "complex64" and n >= 512
            and kernel in {"r2c_packed_postprocess", "c2r_packed_preprocess"}
            and os.getenv("FLAGFFT_NPU_PAIRED_REAL_BOUNDARY", "0") == "1"):
        grid_columns = n // 4 + 1 - int(_npu_packed_boundary_direct_midpoint(n, True, elements_per_program))
        elements_per_program *= _npu_packed_boundary_groups(n, True)
    column_block = 256
    if maca_paired:
        grid_columns = n // 4 + 1
        column_block = elements_per_program
    elif kernel in {"r2c_packed_postprocess","c2r_packed_preprocess"}:
        from .kernels_real import packed_real_boundary_block, _packed_layout
        width=n//2 + (1 if kernel=="r2c_packed_postprocess" else 0)
        column_block,_=_packed_layout(width,packed_real_boundary_block(n,dtype))
    metadata = {
        "module_path": str(module_path),
        "kernel_name": kernel_name,
        "signature": _signature(arg_names, dtype),
        "num_warps": num_warps,
        "num_stages": _RESHAPE_NUM_STAGES,
        "batch_per_block": 1,
        "arg_names": arg_names,
        "kernel_type": kernel,
        "dtype": dtype,
        "length": int(n),
        # Keep the historical block ABI; the explicit column grid also
        # supports wider NPU tiles. Row grouping belongs to grid.y.
        "block": column_block,
        "grid_x_override": (
            (grid_columns + elements_per_program - 1) // elements_per_program
            if maca_paired or kernel in {"real_to_complex", "r2c_half_pack",
                          "compact_to_hermitian_full", "complex_to_real"}
            or (target.startswith("npu:") and dtype == "complex64" and n >= 512)
            else 0
        ),
        "rows_per_block": int(rows_per_block),
    }
    write_text_atomic(out_dir / f"{module_name}.json", json.dumps(metadata, sort_keys=True))
    return metadata


def _scoped_maca_tail(fn):
    @wraps(fn)
    def scoped(**kwargs):
        with maca_tail_kernel_scope(
            kwargs["kernel"], kwargs["length"], kwargs["dtype"],
            kwargs["four_step_n1"], kwargs["four_step_n2"],
        ):
            return fn(**kwargs)
    return scoped


@_scoped_maca_tail
def emit_jit_kernel(
    *,
    kernel: str,
    length: int,
    factors: tuple[int, ...],
    lanes: int,
    num_warps: int,
    generic_radices: tuple[int, ...],
    smem_size: int,
    direction: str,
    dtype: str,
    prime_n: int,
    four_step_n1: int,
    four_step_n2: int,
    perm_form: str = "outer",
    permutation_span: int = 0,
    permutation_batch: int = 0,
    hcu_full_smem: bool = False,
    out_dir: Path,
) -> dict[str, Any]:
    if (kernel == "leaf_packed_r2c" and _ix_backend_active()
            and os.getenv("FLAGFFT_IX_3D_PACKED_R2C") == "1"):
        # Each JIT source process emits one kernel.  Keep the portable
        # exchange required by this packed boundary local to that process.
        os.environ["FLAGFFT_IX_PORTABLE_LEAF"] = "1"
    plan = LeafPlan(
        length=length,
        factors=factors,
        remainder=1,
        lanes=lanes,
        num_warps=num_warps,
        generic_radices=generic_radices,
        smem_size=smem_size,
        direction=direction,
        dtype=dtype,
    )
    spec = kernel_spec(kernel)
    maca_dif_layout = {
        "maca_fp64_dif": (2048, 2, 1),
        "maca_fp64_dif_w4": (2048, 4, 1),
        "maca_fp64_packed_dif1024": (1024, 4, 1),
        "maca_fp64_dif_first64": (64, 8, 32),
        "maca_fp64_ct128_prepare": (128, 4, 4),
        "maca_fp64_ct128_finish": (128, 4, 8),
        "maca_fp64_ct128_prepare_dif": (128, 2, 8),
        "maca_fp64_ct128_finish_dif": (128, 2, 4),
        "maca_fp64_3d_r2c64": (64, 2, 32),
        "maca_fp64_3d_c2r64": (64, 2, 32),
        "maca_fp32_3d_c2r64": (64, 2, 32),
        "maca_3d_prime_r2c64": (64, 4, 32),
        "maca_3d_prime_c2r64": (64, 4, 32),
        "maca_3d_prime_outer16": (16, 4, 64),
    }.get(perm_form) if kernel == "leaf" else None
    maca_prime_dif = perm_form.startswith("maca_prime_dif_")
    maca_split_column = perm_form in {
        "maca_split_column1768_prepare", "maca_split_column1768_finish",
        "maca_split_column1768_prepare_pfa", "maca_split_column1768_finish_pfa",
    }
    if maca_split_column:
        pfa_column = perm_form.endswith("_pfa")
        fp64_column = dtype == "complex128"
        allowed_warps = (2,) if fp64_column else ((4,) if pfa_column else (2, 4))
        if not (_maca_backend_active() and current_profile().device_arch == "102"
                and kernel == "four_step_col" and dtype in {"complex64", "complex128"}
                and num_warps in allowed_warps
                and (length, four_step_n1, four_step_n2) == (1768, 375, 1768)):
            raise ValueError("the split column requires FP32 or FP64 FourStep 375 × 1768 on C550")
        from .kernels_maca_column import build_maca_split_column1768

        prepare_column = perm_form.startswith("maca_split_column1768_prepare")
        kernel_name, kernel_source, column_grid = build_maca_split_column1768(
            direction, prepare_column, num_warps, pfa=pfa_column, fp64=fp64_column,
        )
        n1, n2 = four_step_n1, four_step_n2
    elif maca_prime_dif:
        mode = perm_form.removeprefix("maca_prime_dif_")
        expected_kernel = "leaf_bluestein" if mode == "c2c" else f"leaf_bluestein_{mode}"
        if not (_maca_backend_active() and current_profile().device_arch == "102"
                and mode in {"c2c", "r2c", "c2r"} and kernel == expected_kernel
                and dtype in {"complex64", "complex128"} and length == 2048 and prime_n in {997, 1009}):
            raise ValueError("the MACA prime DIF kernel requires a C550 997- or 1009-point transform")
        from .kernels_maca_dif import build_maca_prime_dif

        kernel_name, kernel_source = build_maca_prime_dif(prime_n, mode)
        n1 = n2 = 0
    elif maca_dif_layout:
        required_length, maca_warps, maca_pack = maca_dif_layout
        prime_axes = perm_form.startswith("maca_3d_prime_")
        required_dtypes = (("complex64", "complex128") if prime_axes else
                           ("complex64",) if perm_form == "maca_fp32_3d_c2r64" else ("complex128",))
        if not (_maca_backend_active() and current_profile().device_arch == "102"
                and dtype in required_dtypes and length == required_length):
            raise ValueError(f"{perm_form} requires a {required_length}-point C550 transform of {required_dtypes}")
        if perm_form == "maca_3d_prime_outer16":
            from .kernels_maca_dif import build_maca_3d_prime_real_outer

            kernel_name, kernel_source = build_maca_3d_prime_real_outer(direction, fp64=dtype == "complex128")
        elif perm_form.startswith("maca_fp64_ct128"):
            from .kernels_maca_ct import build_maca_fp64_outer_ct, build_maca_fp64_outer_dif

            builder = (build_maca_fp64_outer_dif if perm_form.endswith("_dif")
                       else build_maca_fp64_outer_ct)
            kernel_name, kernel_source = builder(direction, prepare="_prepare" in perm_form)
        elif perm_form.endswith(("_r2c64", "_c2r64")):
            from .kernels_maca_dif import build_maca_3d_real_boundary

            expected_direction = "inverse" if perm_form.endswith("_c2r64") else "forward"
            if direction != expected_direction:
                raise ValueError(f"{perm_form} requires the {expected_direction} direction")
            kernel_name, kernel_source = build_maca_3d_real_boundary(
                direction, fp64=dtype == "complex128", middle_length=997 if prime_axes else 2048)
        elif perm_form == "maca_fp64_packed_dif1024":
            from .kernels_maca_dif import build_maca_fp64_packed_dif

            kernel_name, kernel_source = build_maca_fp64_packed_dif(direction)
        else:
            from .kernels_maca_dif import build_maca_fp64_dif_first_axis, build_maca_fp64_dif_leaf

            if required_length == 64:
                kernel_name, kernel_source = build_maca_fp64_dif_first_axis(direction)
            else:
                kernel_name, kernel_source = build_maca_fp64_dif_leaf(
                    direction, four_warps=perm_form == "maca_fp64_dif_w4",
                )
        n1 = n2 = 0
    elif spec.is_leaf_like:
        io_mode = (
            "permuted_r2c"
            if kernel == "leaf_r2c" and perm_form == "permuted"
            else spec.io_mode
        )
        if kernel in FOUR_STEP_ROW_NAMES and plan.length != four_step_n1:
            raise ValueError(
                f"four-step {kernel} kernel length must equal n1: "
                f"length={plan.length}, n1={four_step_n1}"
            )
        if kernel in FOUR_STEP_COL_NAMES and plan.length != four_step_n2:
            raise ValueError(
                f"four-step {kernel} kernel length must equal n2: "
                f"length={plan.length}, n2={four_step_n2}"
            )
        kernel_name, kernel_source = _build_leaf_kernel_source_for_io(
            plan,
            io_mode=io_mode,
            prime_n=prime_n,
            four_step_n1=four_step_n1,
            four_step_n2=four_step_n2,
            perm_form=perm_form,
            permutation_span=permutation_span,
            permutation_batch=permutation_batch,
            hcu_full_smem=hcu_full_smem,
        )
        n1 = four_step_n1 if spec.is_four_step else 0
        n2 = four_step_n2 if spec.is_four_step else 0
    elif spec.family == STOCKHAM:
        if len(factors) not in (1, 2, 3):
            raise ValueError("a Stockham stage needs a radix and optional span/block")
        stockham_block = factors[2] if len(factors) == 3 else 128
        kernel_name, kernel_source = build_stockham_stage(
            length, factors[0], direction, dtype, factors[1] if len(factors) >= 2 else 0,
            stockham_block
        )
        n1 = n2 = 0
    elif spec.family == DIRECT_DFT:
        if kernel in {"direct_dft_r2c", "direct_dft_c2r"}:
            expected_direction = "inverse" if kernel == "direct_dft_c2r" else "forward"
            if direction != expected_direction:
                raise ValueError(f"{kernel} requires direction={expected_direction}")
            kernel_name, kernel_source, _ = _build_real_direct_dft_kernel_source(
                length, dtype, inverse=(kernel == "direct_dft_c2r")
            )
        elif kernel in {"direct_dft_cube_2d_row", "direct_dft_cube_2d_col"}:
            kernel_name, kernel_source, _ = _build_cube_dft_2d_kernel_source(
                length, direction, dtype, column=(kernel == "direct_dft_cube_2d_col")
            )
        else:
            kernel_name, kernel_source, _ = _build_direct_dft_kernel_source(
                length,
                direction,
                dtype,
                strided=kernel in {"direct_dft_strided", "direct_dft_cube_strided"},
                cube=kernel in {"direct_dft_cube", "direct_dft_cube_strided"},
                cube_transposed=kernel == "direct_dft_cube_transposed",
            )
        n1 = 0
        n2 = 0
    else:
        raise ValueError(f"unsupported JIT kernel kind: {kernel}")

    lane_block = lane_block_for(lanes)
    direction_tag = "inv" if direction == "inverse" else "fwd"
    factor_tag = "_".join(str(x) for x in factors)
    dtype_tag = _dtype_suffix(dtype)
    module_name = module_name_for(
        spec,
        direction_tag=direction_tag,
        factor_tag=factor_tag,
        lanes=lanes,
        lane_block=lane_block,
        dtype_tag=dtype_tag,
        length=length,
        prime_n=prime_n,
        four_step_n1=n1,
        four_step_n2=n2,
        perm_form=perm_form,
    )
    if spec.family == STOCKHAM or maca_dif_layout or maca_prime_dif or maca_split_column:
        # Include the lowering and span in the generated-module identity.
        module_name = f"flagfft_jit_{kernel_name}"
    if kernel_name.startswith("ix_warp256_"):
        from .target import ix_warp_fft_pack
        module_name += f"_warp{ix_warp_fft_pack()}"
    if permutation_span:
        module_name += f"_s{permutation_span}"
    if permutation_batch:
        module_name += f"_nb{permutation_batch}"
    module_name += variant_suffix()

    out_dir.mkdir(parents=True, exist_ok=True)
    module_path = out_dir / f"{module_name}.py"
    if maca_split_column:
        radices = () if prepare_column else (8,)
    elif spec.is_leaf_like and not (maca_dif_layout or maca_prime_dif):
        radices = tuple(
            sorted(
                codelet_radices_for(factors)
                | codelet_radices_for(emitted_leaf_factors(plan, spec.io_mode))
            )
        )
    elif spec.family == STOCKHAM:
        radices = tuple(sorted(codelet_radices_for(factors[:1])))
    else:
        radices = ()
    write_text_atomic(module_path, _module_source(kernel_source, radices))

    # Metadata describes the emitted function, not the generator's active
    # device. Do not import backend-specific TLE helpers just to read arguments.
    function = next(
        node
        for node in ast.parse(kernel_source).body
        if isinstance(node, ast.FunctionDef) and node.name == kernel_name
    )
    arg_names = [arg.arg for arg in function.args.args]
    metadata = _metadata(
        module_path=module_path,
        kernel_name=kernel_name,
        arg_names=arg_names,
        plan=plan,
        kernel_type=kernel,
        n1=n1,
        n2=n2,
        dtype=dtype,
        hcu_full_smem=hcu_full_smem,
        permutation_span=permutation_span,
        permutation_batch=permutation_batch,
    )
    if maca_dif_layout:
        metadata.update(num_warps=maca_warps, batch_per_block=maca_pack)
    if maca_prime_dif:
        metadata.update(num_warps=4 if dtype == "complex128" else 2, batch_per_block=1)
    if maca_split_column:
        metadata.update(num_warps=num_warps, grid_x_override=column_grid, tle_fused_twiddle=False)
    if permutation_span:
        metadata["permutation_span"] = permutation_span
    if permutation_batch:
        metadata["permutation_batch"] = permutation_batch
    if spec.family == STOCKHAM:
        metadata["butterflies_per_block"] = stockham_block
    if kernel in {"direct_dft_cube", "direct_dft_cube_strided", "direct_dft_cube_transposed"}:
        metadata["batch_per_block"] = 16
    if kernel in {"direct_dft", "direct_dft_r2c", "direct_dft_c2r"}:
        metadata["batch_per_block"] = direct_dft_batch_tile(length, dtype=dtype)
    if kernel == "direct_dft_cube_transposed":
        metadata["grid_y_override"] = 1
    if kernel in {"direct_dft_cube_2d_row", "direct_dft_cube_2d_col"}:
        metadata["batch_per_block"] = 16
        metadata["grid_y_override"] = 4
    write_text_atomic(out_dir / f"{module_name}.json", json.dumps(metadata, sort_keys=True))
    return metadata


def _emit_tiled_transpose_jit_kernel(
    *,
    n0: int,
    n1: int,
    dtype: str = "complex64",
    tile_size: int = 32,
    out_dir: Path,
    npu_vector_transpose: bool = False,
) -> dict[str, Any]:
    cuda_mode = os.environ.get("FLAGFFT_CUDA_2D_TRANSPOSE", "auto") if _declared_backend() == "cuda" else "auto"
    hcu_mode = os.environ.get("FLAGFFT_HCU_2D_TRANSPOSE", "auto") if _hcu_backend_active() else "auto"
    maca_target = _maca_backend_active()
    maca_mode = _maca_knob("2D_TRANSPOSE") if maca_target or _ix_backend_active() else ""
    maca_tuned = maca_target and maca_mode in {"tuned", "vector"}
    c550 = maca_tuned and current_profile().device_arch == "102"
    fp64_vector = c550 and maca_mode == "vector" and dtype == "complex128"
    column_first = fp64_vector and (n0, n1) == (1025, 2048)
    flatten_pair = fp64_vector and (n0, n1) in {(2048, 1025), (2048, 2048)}
    num_warps = 4
    if c550 and min(n0, n1) >= 1024:
        if dtype == "complex128":
            tile_size = 16
        elif n0 % 2 and n1 % tile_size == 0:
            # A wider CTA improves stores to the odd half-spectrum pitch.
            num_warps = 8
    if hcu_mode not in {"auto", "register", "scalar", "coalesced"}:
        raise ValueError("FLAGFFT_HCU_2D_TRANSPOSE must be auto, register, scalar or coalesced")
    if cuda_mode not in {"auto", "register", "scalar", "packed"}:
        raise ValueError("FLAGFFT_CUDA_2D_TRANSPOSE must be auto, register, scalar or packed")
    if cuda_mode == "packed" and dtype != "complex64":
        raise ValueError("CUDA packed 2D transpose requires complex64")
    if npu_vector_transpose:
        if not _npu_backend_active() or dtype != "complex64":
            raise ValueError("NPU vector transpose requires NPU complex64")
        from .kernels_npu_transpose2d import build_npu_table_transpose2d_source
        kernel_name, kernel_source, arg_names, npu_programs = build_npu_table_transpose2d_source(
            n0, n1, tile_size
        )
    elif (
        cuda_mode == "packed" or (hcu_mode == "coalesced" and dtype == "complex64")
        or ((maca_mode == "packed" or maca_tuned) and dtype == "complex64")
    ):
        kernel_name, kernel_source, arg_names = _build_packed_transpose_kernel_source(
            n0, n1, tile_size
        )
    elif maca_target and (maca_mode == "packed" or maca_tuned) and dtype == "complex128":
        kernel_name, kernel_source, arg_names = _build_paired_fp64_transpose_kernel_source(
            n0, n1, tile_size, flatten_pair=flatten_pair, column_first=column_first,
        )
    else:
        kernel_name, kernel_source, arg_names = _build_tiled_transpose_kernel_source(
            n0, n1, dtype, tile_size,
            register_transpose={"register": True, "scalar": False, "coalesced": True}.get(
                hcu_mode if _hcu_backend_active() else cuda_mode),
        )
    suffix = _dtype_suffix(dtype)
    module_name = f"flagfft_jit_tiled_transpose_n{n0}_{n1}_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    module_path = out_dir / f"{module_name}.py"
    write_text_atomic(module_path, _module_source(kernel_source))
    metadata = {
        "module_path": str(module_path),
        "kernel_name": kernel_name,
        "signature": _signature(arg_names, dtype),
        "num_warps": num_warps,
        "num_stages": 1,
        "batch_per_block": 1,
        "arg_names": arg_names,
        "kernel_type": "tiled_transpose",
        "dtype": dtype,
        "reshape_n1": int(n0),
        "reshape_n2": int(n1),
        "tile_size": int(tile_size),
        "block": tile_size * tile_size,
    }
    if npu_vector_transpose:
        metadata["grid_x_override"] = npu_programs
        metadata["grid_y_override"] = 1
        metadata["npu_table_transpose"] = True
    elif column_first:
        metadata["grid_x_override"] = (n0 + tile_size - 1) // tile_size
        metadata["grid_y_override"] = (n1 + tile_size - 1) // tile_size
    write_text_atomic(out_dir / f"{module_name}.json", json.dumps(metadata, sort_keys=True))
    return metadata


def _transpose3d_v2_supported() -> bool:
    """Whether the vectorized (inline-asm) 3D transpose variant can be used.

    The v2 kernel relies on ld/st.global.v2 inline asm; the MThreads MTGPU
    LLVM backend (FlagTree mthreads) cannot allocate registers for it, and
    the PPU/Iluvatar toolchains do not support the PTX inline asm either.
    Those targets use the portable register-tile variant instead, which keeps
    both the load and the store side coalesced.
    """
    from .kernels_common import _non_nvidia_backend_active, _npu_backend_active

    if _npu_backend_active():
        return False
    return not _non_nvidia_backend_active()


# Backends whose 3D correctness and performance have been validated with the
# portable register-tile transpose.  Each backend is added here after its own
# validation; the remaining non-NVIDIA targets keep the previous v1 kernel.
_PORTABLE_TRANSPOSE3D_BACKENDS = frozenset({"ix", "maca", "musa", "hcu", "npu"})


def _portable_transpose3d_supported() -> bool:
    """Whether the portable register-tile 3D transpose is enabled for this backend."""
    from .kernels_common import _declared_backend

    return _declared_backend() in _PORTABLE_TRANSPOSE3D_BACKENDS


def _emit_tiled_transpose3d_jit_kernel(
    *,
    n0: int,
    n1: int,
    n2: int,
    order: str,
    dtype: str = "complex64",
    out_dir: Path,
) -> dict[str, Any]:
    transpose_warps = 4
    cuda_fp64_mode = (
        os.environ.get("FLAGFFT_CUDA_3D_FP64_TRANSPOSE", "v1")
        if _declared_backend() == "cuda" and dtype == "complex128"
        else "v1"
    )
    if cuda_fp64_mode not in {"v1", "tile16", "tile32", "tile64", "tile16vec", "tile32vec"}:
        raise ValueError(
            "FLAGFFT_CUDA_3D_FP64_TRANSPOSE must be v1, tile16, tile32, tile64, tile16vec or tile32vec"
        )
    if _declared_backend() == "npu":
        raw_warps = os.environ.get("FLAGFFT_NPU_3D_TRANSPOSE_WARPS")
        if raw_warps is not None:
            if raw_warps not in {"4", "8"}:
                raise ValueError("FLAGFFT_NPU_3D_TRANSPOSE_WARPS must be 4 or 8")
            transpose_warps = int(raw_warps)

    if _declared_backend() == "maca":
        maca_mode_default = (
            "pair16sg2rg2"
            if dtype == "complex64" and tuple(sorted((n0, n1, n2))) == (256, 256, 256)
            else "pair16"
        )
        maca_mode = os.environ.get("FLAGFFT_MACA_TRANSPOSE3D", maca_mode_default)
    else:
        maca_mode = "tile32"
    if maca_mode not in {
        "tile16", "tile32", "tile64", "pair16", "pair32", "pair64", "pair64x16",
        "pair16rg2", "pair16sg2rg2", "v1",
    }:
        raise ValueError(
            "FLAGFFT_MACA_TRANSPOSE3D must be tile16, tile32, tile64, pair16, "
            "pair32, pair64, pair64x16, pair16rg2, pair16sg2rg2 or v1"
        )
    if maca_mode == "pair16sg2rg2" and (
        dtype != "complex64"
        or tuple(sorted((n0, n1, n2)))
        not in {(64, 128, 2048), (256, 256, 256)}
    ):
        # The experimental row/slice-group kernel is only validated for
        # these FP32 shapes. An all-suite environment must leave every other
        # dtype and shape on the established pair16 path.
        maca_mode = "pair16"
    if maca_mode == "pair16rg2" and (
        dtype != "complex64" or tuple(sorted((n0, n1, n2))) != (256, 256, 256)
    ):
        # This screen only specializes the FP32 cube. Let an all-suite run
        # keep unrelated dtypes and shapes on their regular transpose policy.
        maca_mode = "pair16"
    pair64_cube_1024_candidate = (
        _declared_backend() == "maca"
        and maca_mode in {"pair64", "pair64x16"}
        and dtype == "complex64"
        and tuple(sorted((n0, n1, n2))) == (256, 256, 256)
        and os.environ.get("FLAGFFT_MACA_TRANSPOSE3D_WARPS") == "16"
    )
    maca_warps = (
        os.environ.get("FLAGFFT_MACA_TRANSPOSE3D_WARPS", "8" if dtype == "complex64" else "4")
        if _declared_backend() == "maca"
        else "4"
    )
    if maca_warps == "16" and not pair64_cube_1024_candidate:
        # Keep a broad transpose screen from leaking its 1024-thread CTA
        # setting into other dtypes and shapes. The vendor-matched tile is a
        # narrowly scoped FP32 cube experiment only.
        maca_mode = "pair16"
        maca_warps = "8" if dtype == "complex64" else "4"
    if maca_warps not in {"2", "4", "8", "16"}:
        raise ValueError("FLAGFFT_MACA_TRANSPOSE3D_WARPS must be 2, 4, 8 or the scoped pair64 cube value 16")
    traversal_default = "col"
    if (
        _declared_backend() == "maca"
        and dtype == "complex64"
        and tuple(sorted((n0, n1, n2)))
        in {(64, 128, 2048), (256, 256, 256)}
    ):
        traversal_default = "row"
    maca_traversal = (
        os.environ.get("FLAGFFT_MACA_TRANSPOSE3D_TRAVERSAL", traversal_default)
        if _declared_backend() == "maca"
        else "col"
    )
    if maca_traversal not in {"col", "row"}:
        raise ValueError("FLAGFFT_MACA_TRANSPOSE3D_TRAVERSAL must be col or row")
    maca_slice_group_default = (
        "2"
        if (
            _declared_backend() == "maca"
            and dtype == "complex64"
            and maca_mode == "pair16"
            and maca_warps == "8"
            and maca_traversal == "row"
            and tuple(sorted((n0, n1, n2))) in {
                (64, 128, 2048),
                (256, 256, 256),
            }
        )
        else "1"
    )
    maca_slice_group = os.environ.get(
        "FLAGFFT_MACA_TRANSPOSE3D_SLICE_GROUP", maca_slice_group_default
    )
    if maca_slice_group not in {"1", "2"}:
        raise ValueError("FLAGFFT_MACA_TRANSPOSE3D_SLICE_GROUP must be 1 or 2")
    maca_grouped_slice_tile = (
        _declared_backend() == "maca"
        and dtype == "complex64"
        and maca_mode == "pair16"
        and maca_warps == "8"
        and maca_traversal == "row"
        and maca_slice_group == "2"
    )
    fp64_mode = (
        os.environ.get("FLAGFFT_MACA_TRANSPOSE3D_FP64", "flat")
        if _declared_backend() == "maca"
        else "v1"
    )
    if fp64_mode not in {"tile16", "tile32", "tile64", "tile16vec", "tile32vec", "flat", "v1"}:
        raise ValueError(
            "FLAGFFT_MACA_TRANSPOSE3D_FP64 must be tile16, tile32, tile64, tile16vec, tile32vec, flat or v1"
        )
    if fp64_mode == "flat":
        layout = (n0, n1, n2, order)
        long_complex = layout in {(128, 64, 2048, "210"), (2048, 64, 128, "201")}
        long_real = os.environ.get("FLAGFFT_MACA_3D_FP64_COMPACT_TRANSPOSE", "1") == "1" and layout in {
            (128, 2048, 33, "021"), (128, 33, 2048, "021"), (128, 33, 2048, "210"),
            (2048, 33, 128, "210"), (2048, 33, 128, "201"), (128, 2048, 33, "120"),
        }
        if current_profile().device_arch != "102" or maca_warps != "4" or not (long_complex or long_real):
            fp64_mode = "tile16vec"
    if maca_mode == "pair64x16" and _portable_transpose3d_supported():
        if dtype != "complex64" or maca_warps != "16" or maca_traversal != "row":
            raise ValueError("pair64x16 requires complex64, 16 warps and row traversal")
        if tuple(sorted((n0, n1, n2))) != (256, 256, 256):
            raise ValueError("pair64x16 is only screened for the MACA 256^3 cube")
        (
            kernel_name,
            kernel_source,
            arg_names,
            grid_x,
        ) = _build_tiled_transpose3d_tile_kernel_source(
            n0, n1, n2, order, dtype, tile=(64, 16), pair=True,
            tile_traversal=maca_traversal,
        )
    elif maca_mode == "pair16rg2" and _portable_transpose3d_supported():
        if dtype != "complex64" or maca_warps != "8" or maca_traversal != "row":
            raise ValueError("pair16rg2 requires complex64, 8 warps and row traversal")
        if tuple(sorted((n0, n1, n2))) != (256, 256, 256):
            raise ValueError("pair16rg2 is only screened for the MACA 256^3 cube")
        (
            kernel_name,
            kernel_source,
            arg_names,
            grid_x,
        ) = _build_tiled_transpose3d_pair_slice_row_group_kernel_source(
            n0, n1, n2, order, tile=16, slice_group=1,
            row_group=2,
            tile_traversal=maca_traversal,
        )
    elif maca_mode == "pair16sg2rg2" and _portable_transpose3d_supported():
        if dtype != "complex64" or maca_warps != "8" or maca_traversal != "row":
            raise ValueError(
                "pair16sg2rg2 requires complex64, 8 warps and row traversal"
            )
        if tuple(sorted((n0, n1, n2))) not in {
            (64, 128, 2048),
            (256, 256, 256),
        }:
            raise ValueError("pair16sg2rg2 is only screened for the MACA cube and long shapes")
        (
            kernel_name,
            kernel_source,
            arg_names,
            grid_x,
        ) = _build_tiled_transpose3d_pair_slice_row_group_kernel_source(
            n0, n1, n2, order, tile=16, slice_group=2,
            row_group=2,
            tile_traversal=maca_traversal,
        )
    elif (_hcu_backend_active()
          and os.getenv("FLAGFFT_HCU_3D_TRANSPOSE_MODE", "auto") in {"flat", "resource"}
          and order in {"201", "120"}):
        # Rotating an outer axis is a single matrix transpose after merging
        # the two adjacent axes. Keep tiles contiguous across slice boundaries.
        rows, cols = ((n0 * n1, n2) if order == "201" else (n0, n1 * n2))
        (kernel_name, kernel_source, arg_names, grid_x) = _build_tiled_transpose3d_tile_kernel_source(
            1, rows, cols, "021", dtype,
            tile=32 if dtype == "complex64" else 16,
        )
    elif (_hcu_backend_active()
          and (os.getenv("FLAGFFT_HCU_3D_TRANSPOSE_MODE", "auto") == "packed"
               or (os.getenv("FLAGFFT_HCU_3D_TRANSPOSE_MODE", "auto") == "resource"
                   and os.getenv("FLAGFFT_HCU_TRANSPOSE_COMPACT", "0") == "1"))):
        tile_setting = os.getenv("FLAGFFT_HCU_3D_TRANSPOSE_TILE", "auto")
        if tile_setting == "resource":
            capacity = (current_profile().max_dynamic_shared_memory or 65536) // 2
            element_bytes = 16 if dtype == "complex128" else 8
            tile = 16
            while tile < 64 and (tile * 2) ** 2 * element_bytes <= capacity:
                tile *= 2
        else:
            tile = 32 if tile_setting == "auto" else int(tile_setting)
        (kernel_name, kernel_source, arg_names, grid_x) = _build_tiled_transpose3d_tile_kernel_source(
            n0, n1, n2, order, dtype, tile=tile,
            packed_complex=dtype == "complex64",
            component_pair=dtype == "complex128",
        )
    elif cuda_fp64_mode != "v1":
        # CUDA's FP64 fallback gathers strided source values. Screen the
        # portable tile that coalesces both sides, independently of FP32 PTX.
        (kernel_name, kernel_source, arg_names, grid_x) = _build_tiled_transpose3d_tile_kernel_source(
            n0, n1, n2, order, dtype, tile=int(cuda_fp64_mode[4:6]),
            vec_store=cuda_fp64_mode.endswith("vec"),
        )
    elif dtype == "complex64" and _transpose3d_v2_supported():
        (
            kernel_name,
            kernel_source,
            arg_names,
            grid_x,
        ) = _build_tiled_transpose3d_v2_kernel_source(n0, n1, n2, order, dtype, tile=16)
    elif maca_grouped_slice_tile and _portable_transpose3d_supported():
        (kernel_name, kernel_source, arg_names, grid_x) = _build_tiled_transpose3d_pair_slice_group_kernel_source(
            n0, n1, n2, order, tile=16, slice_group=2, tile_traversal=maca_traversal
        )
    elif dtype == "complex64" and _declared_backend() == "maca" and _portable_transpose3d_supported() and maca_mode != "v1":
        (kernel_name, kernel_source, arg_names, grid_x) = _build_tiled_transpose3d_tile_kernel_source(
            n0, n1, n2, order, dtype, tile=int(maca_mode[4:]), pair=maca_mode.startswith("pair"),
            tile_traversal=maca_traversal
        )
    elif dtype == "complex128" and fp64_mode == "flat":
        (kernel_name, kernel_source, arg_names, grid_x) = _build_flat_fp64_transpose3d_kernel_source(
            n0, n1, n2, order,
        )
    elif dtype == "complex128" and _declared_backend() == "maca" and fp64_mode != "v1":
        (kernel_name, kernel_source, arg_names, grid_x) = _build_tiled_transpose3d_tile_kernel_source(
            n0, n1, n2, order, dtype, tile=int(fp64_mode[4:6]),
            vec_store=fp64_mode.endswith("vec"), tile_traversal=maca_traversal
        )
    elif _declared_backend() != "maca" and _portable_transpose3d_supported() and (
        dtype == "complex64"
        or (
            dtype == "complex128"
            and _hcu_backend_active()
            and (
                os.getenv("FLAGFFT_HCU_3D_FP64_TILE", "auto") == "1"
                or (
                    os.getenv("FLAGFFT_HCU_3D_FP64_TILE", "auto") == "auto"
                    and (n0, n1, n2, order) in {
                        (256, 256, 129, "021"),
                        (16, 64, 997, "210"),
                        (16, 33, 997, "210"),
                    }
                )
            )
        )
    ):
        tile = 32 if dtype == "complex64" else 16
        packed_complex = False
        pair_store = False
        if _ix_backend_active():
            tile = 16
            ix_tile = os.getenv("FLAGFFT_IX_3D_TRANSPOSE_TILE")
            if ix_tile is not None:
                if ix_tile not in {"8", "16", "32"}:
                    raise ValueError("FLAGFFT_IX_3D_TRANSPOSE_TILE must be 8, 16 or 32")
                tile = int(ix_tile)
            packed_complex = os.getenv("FLAGFFT_IX_3D_PACKED_TRANSPOSE") == "1"
            (
                kernel_name,
                kernel_source,
                arg_names,
                grid_x,
            ) = _build_tiled_transpose3d_tile_kernel_source(
                n0,
                n1,
                n2,
                order,
                dtype,
                tile=tile,
                packed_complex=packed_complex,
                pair_store=pair_store,
            )
        elif _hcu_backend_active():
            transpose_slice_group = os.getenv(
                "FLAGFFT_HCU_3D_TRANSPOSE_SLICE_GROUP", "auto"
            )
            if transpose_slice_group not in {"auto", "1", "2"}:
                raise ValueError(
                    "FLAGFFT_HCU_3D_TRANSPOSE_SLICE_GROUP must be auto, 1 or 2"
                )
            if dtype == "complex64":
                transpose_tile = os.getenv("FLAGFFT_HCU_3D_TRANSPOSE_TILE", "auto")
                if transpose_tile in {"auto", "resource"}:
                    tile = (
                        16
                        if (n0, n1, n2, order)
                        in {
                            (16, 64, 997, "210"),
                            (16, 33, 997, "210"),
                            # The FP32 HCU C2R long-axis middle transpose has
                            # high L1 stalls with tile32; tile16 was 17.8%
                            # faster in an isolated 128x2048x64 A-B-B-A.
                            (128, 33, 2048, "210"),
                        }
                        else 32
                    )
                else:
                    tile = int(transpose_tile)
            pair_store = os.getenv("FLAGFFT_HCU_3D_TRANSPOSE_PAIR", "0") == "1"
            if (
                dtype == "complex64"
                and transpose_slice_group == "2"
                and (n0, n1, n2, order) == (128, 64, 2048, "210")
                and tile == 32
                and not pair_store
            ):
                (
                    kernel_name,
                    kernel_source,
                    arg_names,
                    grid_x,
                ) = _build_tiled_transpose3d_slice_group_kernel_source(
                    n0,
                    n1,
                    n2,
                    order,
                    dtype,
                    tile=tile,
                    slice_group=2,
                )
            else:
                (
                    kernel_name,
                    kernel_source,
                    arg_names,
                    grid_x,
                ) = _build_tiled_transpose3d_tile_kernel_source(
                    n0,
                    n1,
                    n2,
                    order,
                    dtype,
                    tile=tile,
                    packed_complex=packed_complex,
                    pair_store=pair_store,
                )
        else:
            (
                kernel_name,
                kernel_source,
                arg_names,
                grid_x,
            ) = _build_tiled_transpose3d_tile_kernel_source(
                n0, n1, n2, order, dtype, tile=tile,
                packed_complex=packed_complex, pair_store=pair_store,
            )
    else:
        kernel_name, kernel_source, arg_names = _build_tiled_transpose3d_kernel_source(
            n0, n1, n2, order, dtype
        )
        grid_x = 0
    suffix = _dtype_suffix(dtype)
    warp_suffix = f"_w{transpose_warps}" if transpose_warps != 4 else ""
    if warp_suffix:
        kernel_name_with_warps = f"{kernel_name}{warp_suffix}"
        kernel_source = kernel_source.replace(
            f"def {kernel_name}(", f"def {kernel_name_with_warps}(", 1
        )
        kernel_name = kernel_name_with_warps
    module_name = f"flagfft_jit_transpose3d_{order}_n{n0}_{n1}_{n2}_{suffix}{warp_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    module_path = out_dir / f"{module_name}.py"
    write_text_atomic(module_path, _module_source(kernel_source))
    metadata = {
        "module_path": str(module_path),
        "kernel_name": kernel_name,
        "signature": _signature(arg_names, dtype),
        "num_warps": int(maca_warps) if _declared_backend() == "maca" else transpose_warps,
        "num_stages": 1,
        "batch_per_block": 1,
        "arg_names": arg_names,
        "kernel_type": "transpose3d",
        "dtype": dtype,
        "transpose3d_n0": int(n0),
        "transpose3d_n1": int(n1),
        "transpose3d_n2": int(n2),
        "transpose3d_order": order,
        "grid_x_override": int(grid_x),
    }
    write_text_atomic(out_dir / f"{module_name}.json", json.dumps(metadata, sort_keys=True))
    return metadata


__all__ = [
    "_BLUESTEIN_BLOCK",
    "_BLUESTEIN_NUM_STAGES",
    "_BLUESTEIN_NUM_WARPS",
    "_RADER_BLOCK",
    "_RADER_NUM_STAGES",
    "_RADER_NUM_WARPS",
    "_RESHAPE_NUM_STAGES",
    "_RESHAPE_NUM_WARPS",
    "_bluestein_kernel_source",
    "_emit_bluestein_jit_kernel",
    "_emit_r2c_pointwise_jit_kernel",
    "_emit_rader_jit_kernel",
    "_emit_reshape_jit_kernel",
    "_emit_tiled_transpose3d_jit_kernel",
    "_emit_tiled_transpose_jit_kernel",
    "_rader_kernel_source",
    "_transpose3d_v2_supported",
    "emit_jit_kernel",
]
