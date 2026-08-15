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

import importlib.util
import json
import sys
from pathlib import Path
from textwrap import dedent
from typing import Any

from .kernels_common import (
    _dtype_suffix,
    _next_power_of_two,
    _zero_other,
    LeafPlan,
    codelet_radices_for,
    lane_block_for,
)
from .kernels_layout import (
    _build_reshape_pack_kernel_source,
    _build_tiled_transpose3d_kernel_source,
    _build_tiled_transpose3d_v2_kernel_source,
    _build_tiled_transpose_kernel_source,
    _build_twiddle_reshape_pack_kernel_source,
)
from .kernels_leaf import _build_leaf_kernel_source_for_io
from .kernels_real import (
    _build_compact_to_hermitian_full_kernel_source,
    _build_complex_to_real_kernel_source,
    _build_r2c_half_pack_kernel_source,
    _build_real_to_complex_kernel_source,
)
from .kernels_special import _build_direct_dft_kernel_source
from .metadata import _metadata, _module_source, _signature
from .registry import (
    DIRECT_DFT,
    FOUR_STEP_COL_NAMES,
    FOUR_STEP_ROW_NAMES,
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
    div_cast = "tl.cast(m, tl.float64)" if dtype == "complex128" else "m"
    if kind == "bluestein_prepare":
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
    module_path.write_text(_module_source(kernel_source))
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
        "block": _BLUESTEIN_BLOCK,
    }
    (out_dir / f"{module_name}.json").write_text(json.dumps(metadata, sort_keys=True))
    return metadata


def _rader_kernel_source(
    kind: str, n: int, m: int, dtype: str
) -> tuple[str, str, list[str]]:
    zero = _zero_other(dtype)
    div_cast = "tl.cast(m, tl.float64)" if dtype == "complex128" else "m"
    sum_block = _next_power_of_two(n)
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
                    xr = tl.load(src, mask=mask, other={zero})
                    xi = tl.load(src + 1, mask=mask, other={zero})

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
                    tl.store(dst, yr, mask=mask)
                    tl.store(dst + 1, yi, mask=mask)

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
    module_path.write_text(_module_source(kernel_source))
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
    (out_dir / f"{module_name}.json").write_text(json.dumps(metadata, sort_keys=True))
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
    module_path.write_text(_module_source(kernel_source))
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
    (out_dir / f"{module_name}.json").write_text(json.dumps(metadata, sort_keys=True))
    return metadata


def _emit_r2c_pointwise_jit_kernel(
    *,
    kernel: str,
    n: int,
    dtype: str,
    out_dir: Path,
) -> dict[str, Any]:
    if kernel == "real_to_complex":
        (
            kernel_name,
            kernel_source,
            arg_names,
            rows_per_block,
        ) = _build_real_to_complex_kernel_source(n, dtype)
    elif kernel == "r2c_half_pack":
        (
            kernel_name,
            kernel_source,
            arg_names,
            rows_per_block,
        ) = _build_r2c_half_pack_kernel_source(n, dtype)
    elif kernel == "compact_to_hermitian_full":
        (
            kernel_name,
            kernel_source,
            arg_names,
            rows_per_block,
        ) = _build_compact_to_hermitian_full_kernel_source(n, dtype)
    elif kernel == "complex_to_real":
        (
            kernel_name,
            kernel_source,
            arg_names,
            rows_per_block,
        ) = _build_complex_to_real_kernel_source(n, dtype)
    else:
        raise ValueError(f"unsupported R2C pointwise kernel kind: {kernel}")

    dtype_tag = _dtype_suffix(dtype)
    module_name = f"flagfft_jit_{kernel}_n{n}_{dtype_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    module_path = out_dir / f"{module_name}.py"
    module_path.write_text(_module_source(kernel_source))

    sys.path.insert(0, str(module_path.parent))
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load generated kernel module {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    arg_names = list(getattr(module, kernel_name).arg_names)
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
        "length": int(n),
        "block": 256,
        "rows_per_block": int(rows_per_block),
    }
    (out_dir / f"{module_name}.json").write_text(json.dumps(metadata, sort_keys=True))
    return metadata


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
    out_dir: Path,
) -> dict[str, Any]:
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
    if spec.is_leaf_like:
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
            io_mode=spec.io_mode,
            prime_n=prime_n,
            four_step_n1=four_step_n1,
            four_step_n2=four_step_n2,
        )
        n1 = four_step_n1 if spec.is_four_step else 0
        n2 = four_step_n2 if spec.is_four_step else 0
    elif spec.family == DIRECT_DFT:
        kernel_name, kernel_source, _ = _build_direct_dft_kernel_source(
            length,
            direction,
            dtype,
            strided=(kernel == "direct_dft_strided"),
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
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    module_path = out_dir / f"{module_name}.py"
    radices = tuple(sorted(codelet_radices_for(factors))) if spec.is_leaf_like else ()
    module_path.write_text(_module_source(kernel_source, radices))

    sys.path.insert(0, str(module_path.parent))
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load generated kernel module {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    arg_names = list(getattr(module, kernel_name).arg_names)
    metadata = _metadata(
        module_path=module_path,
        kernel_name=kernel_name,
        arg_names=arg_names,
        plan=plan,
        kernel_type=kernel,
        n1=n1,
        n2=n2,
        dtype=dtype,
    )
    (out_dir / f"{module_name}.json").write_text(json.dumps(metadata, sort_keys=True))
    return metadata


def _emit_tiled_transpose_jit_kernel(
    *,
    n0: int,
    n1: int,
    dtype: str = "complex64",
    tile_size: int = 32,  # must match constexpr tile_size in raw_nodes.cpp CompiledRaw2DNode::execute()
    out_dir: Path,
) -> dict[str, Any]:
    kernel_name, kernel_source, arg_names = _build_tiled_transpose_kernel_source(
        n0, n1, dtype, tile_size
    )
    suffix = _dtype_suffix(dtype)
    module_name = f"flagfft_jit_tiled_transpose_n{n0}_{n1}_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    module_path = out_dir / f"{module_name}.py"
    module_path.write_text(_module_source(kernel_source))
    metadata = {
        "module_path": str(module_path),
        "kernel_name": kernel_name,
        "signature": _signature(arg_names, dtype),
        "num_warps": 4,
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
    (out_dir / f"{module_name}.json").write_text(json.dumps(metadata, sort_keys=True))
    return metadata


def _transpose3d_v2_supported() -> bool:
    """Whether the vectorized (inline-asm) 3D transpose variant can be used.

    The v2 kernel relies on ld/st.global.v2 inline asm; the MThreads MTGPU
    LLVM backend (FlagTree mthreads) cannot allocate registers for it, and
    the PPU toolchain does not support the PTX inline asm either, so fall
    back to the plain tiled transpose on both.
    """
    try:
        from triton._C import libtriton

        return not (hasattr(libtriton, "mthreads") or hasattr(libtriton, "ppu"))
    except ImportError:
        return True


def _emit_tiled_transpose3d_jit_kernel(
    *,
    n0: int,
    n1: int,
    n2: int,
    order: str,
    dtype: str = "complex64",
    out_dir: Path,
) -> dict[str, Any]:
    if dtype == "complex64" and _transpose3d_v2_supported():
        (
            kernel_name,
            kernel_source,
            arg_names,
            grid_x,
        ) = _build_tiled_transpose3d_v2_kernel_source(n0, n1, n2, order, dtype, tile=16)
    else:
        kernel_name, kernel_source, arg_names = _build_tiled_transpose3d_kernel_source(
            n0, n1, n2, order, dtype
        )
        grid_x = 0
    suffix = _dtype_suffix(dtype)
    module_name = f"flagfft_jit_transpose3d_{order}_n{n0}_{n1}_{n2}_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    module_path = out_dir / f"{module_name}.py"
    module_path.write_text(_module_source(kernel_source))
    metadata = {
        "module_path": str(module_path),
        "kernel_name": kernel_name,
        "signature": _signature(arg_names, dtype),
        "num_warps": 4,
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
    (out_dir / f"{module_name}.json").write_text(json.dumps(metadata, sort_keys=True))
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
