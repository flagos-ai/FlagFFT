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

"""Small 3D complex FFT kernels that combine the two innermost axes."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .artifacts import write_text_atomic
from .kernels_common import _dtype_suffix
from .metadata import _module_source, _signature
from .target import backend_name


def emit_fused_16_plane_kernel(*, dtype: str, direction: str, out_dir: Path) -> dict:
    """Emit a 16x16 plane FFT; one block owns one complete plane.

    Both axes use decimation in time. The load reverses the bits on each axis,
    then ``tl.gather`` performs four radix-two butterfly stages per axis. The
    twiddle buffers carry the forward/inverse sign and precision.
    """
    source = """
@triton.jit
def fused_16_plane_fft_kernel(in_ptr, out_ptr, tw_r_ptr, tw_i_ptr):
    plane = tl.program_id(0)
    idx = tl.arange(0, 256)
    row = idx // 16
    col = idx % 16
    rev_row = tl.full((256,), 0, tl.int32)
    rev_col = tl.full((256,), 0, tl.int32)
    for bit in tl.static_range(4):
        rev_row = (rev_row << 1) | ((row >> bit) & 1)
        rev_col = (rev_col << 1) | ((col >> bit) & 1)
    src = (plane * 256 + rev_row * 16 + rev_col) * 2
    xr = tl.load(in_ptr + src)
    xi = tl.load(in_ptr + src + 1)

    for stage in tl.static_range(4):
        partner = idx ^ (1 << stage)
        pr = tl.gather(xr, partner, 0)
        pi = tl.gather(xi, partner, 0)
        upper = (col & (1 << stage)) != 0
        ar = tl.where(upper, pr, xr)
        ai = tl.where(upper, pi, xi)
        br = tl.where(upper, xr, pr)
        bi = tl.where(upper, xi, pi)
        tw = (col & ((1 << stage) - 1)) * (16 >> (stage + 1))
        wr = tl.load(tw_r_ptr + tw)
        wi = tl.load(tw_i_ptr + tw)
        tr = wr * br - wi * bi
        ti = wr * bi + wi * br
        xr = tl.where(upper, ar - tr, ar + tr)
        xi = tl.where(upper, ai - ti, ai + ti)

    for stage in tl.static_range(4):
        partner = idx ^ (16 << stage)
        pr = tl.gather(xr, partner, 0)
        pi = tl.gather(xi, partner, 0)
        upper = (row & (1 << stage)) != 0
        ar = tl.where(upper, pr, xr)
        ai = tl.where(upper, pi, xi)
        br = tl.where(upper, xr, pr)
        bi = tl.where(upper, xi, pi)
        tw = (row & ((1 << stage) - 1)) * (16 >> (stage + 1))
        wr = tl.load(tw_r_ptr + tw)
        wi = tl.load(tw_i_ptr + tw)
        tr = wr * br - wi * bi
        ti = wr * bi + wi * br
        xr = tl.where(upper, ar - tr, ar + tr)
        xi = tl.where(upper, ai - ti, ai + ti)

    dst = (plane * 256 + idx) * 2
    tl.store(out_ptr + dst, xr)
    tl.store(out_ptr + dst + 1, xi)
"""
    name = f"flagfft_jit_fused_16_plane_{direction}_{_dtype_suffix(dtype)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    module_path = out_dir / f"{name}.py"
    write_text_atomic(module_path, _module_source(source))
    args = ["in_ptr", "out_ptr", "tw_r_ptr", "tw_i_ptr"]
    metadata = {
        "module_path": str(module_path),
        "kernel_name": "fused_16_plane_fft_kernel",
        "signature": _signature(args, dtype),
        "num_warps": 8,
        "num_stages": 1,
        "batch_per_block": 1,
        "arg_names": args,
        "kernel_type": "fused_16_plane",
        "dtype": dtype,
        "direction": direction,
    }
    write_text_atomic(out_dir / f"{name}.json", json.dumps(metadata, sort_keys=True))
    return metadata


def emit_fused_32_plane_kernel(*, dtype: str, direction: str, out_dir: Path) -> dict:
    """Emit a 32x32 complex plane FFT for the MACA 3D fusion experiment."""
    num_warps_text = (
        os.environ.get("FLAGFFT_MACA_3D_C2C_FUSED32_WARPS", "8")
        if backend_name() == "maca"
        else "8"
    )
    if num_warps_text not in {"4", "8"}:
        raise ValueError("FLAGFFT_MACA_3D_C2C_FUSED32_WARPS must be 4 or 8")
    num_warps = int(num_warps_text)
    source = """
@triton.jit
def fused_32_plane_fft_kernel(in_ptr, out_ptr, tw_r_ptr, tw_i_ptr):
    plane = tl.program_id(0)
    idx = tl.arange(0, 1024)
    row = idx // 32
    col = idx % 32
    rev_row = tl.full((1024,), 0, tl.int32)
    rev_col = tl.full((1024,), 0, tl.int32)
    for bit in tl.static_range(5):
        rev_row = (rev_row << 1) | ((row >> bit) & 1)
        rev_col = (rev_col << 1) | ((col >> bit) & 1)
    src = (plane * 1024 + rev_row * 32 + rev_col) * 2
    xr = tl.load(in_ptr + src)
    xi = tl.load(in_ptr + src + 1)

    for stage in tl.static_range(5):
        partner = idx ^ (1 << stage)
        pr = tl.gather(xr, partner, 0)
        pi = tl.gather(xi, partner, 0)
        upper = (col & (1 << stage)) != 0
        ar = tl.where(upper, pr, xr)
        ai = tl.where(upper, pi, xi)
        br = tl.where(upper, xr, pr)
        bi = tl.where(upper, xi, pi)
        tw = (col & ((1 << stage) - 1)) * (32 >> (stage + 1))
        wr = tl.load(tw_r_ptr + tw)
        wi = tl.load(tw_i_ptr + tw)
        tr = wr * br - wi * bi
        ti = wr * bi + wi * br
        xr = tl.where(upper, ar - tr, ar + tr)
        xi = tl.where(upper, ai - ti, ai + ti)

    for stage in tl.static_range(5):
        partner = idx ^ (32 << stage)
        pr = tl.gather(xr, partner, 0)
        pi = tl.gather(xi, partner, 0)
        upper = (row & (1 << stage)) != 0
        ar = tl.where(upper, pr, xr)
        ai = tl.where(upper, pi, xi)
        br = tl.where(upper, xr, pr)
        bi = tl.where(upper, xi, pi)
        tw = (row & ((1 << stage) - 1)) * (32 >> (stage + 1))
        wr = tl.load(tw_r_ptr + tw)
        wi = tl.load(tw_i_ptr + tw)
        tr = wr * br - wi * bi
        ti = wr * bi + wi * br
        xr = tl.where(upper, ar - tr, ar + tr)
        xi = tl.where(upper, ai - ti, ai + ti)

    dst = (plane * 1024 + idx) * 2
    tl.store(out_ptr + dst, xr)
    tl.store(out_ptr + dst + 1, xi)
"""
    name = f"flagfft_jit_fused_32_plane_{direction}_{_dtype_suffix(dtype)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    module_path = out_dir / f"{name}.py"
    write_text_atomic(module_path, _module_source(source))
    args = ["in_ptr", "out_ptr", "tw_r_ptr", "tw_i_ptr"]
    metadata = {
        "module_path": str(module_path),
        "kernel_name": "fused_32_plane_fft_kernel",
        "signature": _signature(args, dtype),
        "num_warps": num_warps,
        "num_stages": 1,
        "batch_per_block": 1,
        "arg_names": args,
        "kernel_type": "fused_32_plane",
        "dtype": dtype,
        "direction": direction,
    }
    write_text_atomic(out_dir / f"{name}.json", json.dumps(metadata, sort_keys=True))
    return metadata


def _emit_fused_real_plane_kernel(
    *, n: int, dtype: str, direction: str, out_dir: Path
) -> dict:
    """Emit one fused square R2C/C2R plane for a small 3D transform."""
    if direction not in {"forward", "inverse"}:
        raise ValueError("fused real plane direction must be forward or inverse")
    if n not in {16, 32}:
        raise ValueError("fused real plane size must be 16 or 32")
    real_kind = "r2c" if direction == "forward" else "c2r"
    half = n // 2 + 1
    elements = n * n
    stages = n.bit_length() - 1
    if real_kind == "r2c":
        scalar_type = "tl.float64" if dtype == "complex128" else "tl.float32"
        load = f"""    src = plane * {elements} + rev_row * {n} + rev_col
    xr = tl.load(in_ptr + src)
    xi = tl.full(({elements},), 0.0, {scalar_type})
"""
        store = f"""    dst = (plane * {n * half} + row * {half} + col) * 2
    tl.store(out_ptr + dst, xr, mask=col < {half})
    tl.store(out_ptr + dst + 1, xi, mask=col < {half})
"""
    else:
        load = f"""    reflected = rev_col >= {half}
    src_row = tl.where(reflected, ({n} - rev_row) % {n}, rev_row)
    src_col = tl.where(reflected, {n} - rev_col, rev_col)
    src = (plane * {n * half} + src_row * {half} + src_col) * 2
    xr = tl.load(in_ptr + src)
    xi = tl.load(in_ptr + src + 1)
    xi = tl.where(reflected, -xi, xi)
"""
        store = f"""    dst = plane * {elements} + idx
    tl.store(out_ptr + dst, xr)
"""

    kernel_name = f"fused_{n}_real_plane_fft_kernel"
    num_warps = 4 if n == 16 else 8
    source = f"""
@triton.jit
def {kernel_name}(in_ptr, out_ptr, tw_r_ptr, tw_i_ptr):
    plane = tl.program_id(0)
    idx = tl.arange(0, {elements})
    row = idx // {n}
    col = idx % {n}
    rev_row = tl.full(({elements},), 0, tl.int32)
    rev_col = tl.full(({elements},), 0, tl.int32)
    for bit in tl.static_range({stages}):
        rev_row = (rev_row << 1) | ((row >> bit) & 1)
        rev_col = (rev_col << 1) | ((col >> bit) & 1)
{load}
    for stage in tl.static_range({stages}):
        partner = idx ^ (1 << stage)
        pr = tl.gather(xr, partner, 0)
        pi = tl.gather(xi, partner, 0)
        upper = (col & (1 << stage)) != 0
        ar = tl.where(upper, pr, xr)
        ai = tl.where(upper, pi, xi)
        br = tl.where(upper, xr, pr)
        bi = tl.where(upper, xi, pi)
        tw = (col & ((1 << stage) - 1)) * ({n} >> (stage + 1))
        wr = tl.load(tw_r_ptr + tw)
        wi = tl.load(tw_i_ptr + tw)
        tr = wr * br - wi * bi
        ti = wr * bi + wi * br
        xr = tl.where(upper, ar - tr, ar + tr)
        xi = tl.where(upper, ai - ti, ai + ti)
    for stage in tl.static_range({stages}):
        partner = idx ^ ({n} << stage)
        pr = tl.gather(xr, partner, 0)
        pi = tl.gather(xi, partner, 0)
        upper = (row & (1 << stage)) != 0
        ar = tl.where(upper, pr, xr)
        ai = tl.where(upper, pi, xi)
        br = tl.where(upper, xr, pr)
        bi = tl.where(upper, xi, pi)
        tw = (row & ((1 << stage) - 1)) * ({n} >> (stage + 1))
        wr = tl.load(tw_r_ptr + tw)
        wi = tl.load(tw_i_ptr + tw)
        tr = wr * br - wi * bi
        ti = wr * bi + wi * br
        xr = tl.where(upper, ar - tr, ar + tr)
        xi = tl.where(upper, ai - ti, ai + ti)
{store}
"""
    name = f"flagfft_jit_fused_{n}_real_plane_{direction}_{_dtype_suffix(dtype)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    module_path = out_dir / f"{name}.py"
    write_text_atomic(module_path, _module_source(source))
    args = ["in_ptr", "out_ptr", "tw_r_ptr", "tw_i_ptr"]
    metadata = {
        "module_path": str(module_path),
        "kernel_name": kernel_name,
        "signature": _signature(args, dtype),
        "num_warps": num_warps,
        "num_stages": 1,
        "batch_per_block": 1,
        "arg_names": args,
        "kernel_type": f"fused_{n}_real_plane",
        "dtype": dtype,
        "direction": direction,
    }
    write_text_atomic(out_dir / f"{name}.json", json.dumps(metadata, sort_keys=True))
    return metadata


def emit_fused_16_real_plane_kernel(*, dtype: str, direction: str, out_dir: Path) -> dict:
    """Emit one fused 16x16 R2C/C2R plane for the small-cube experiment."""
    return _emit_fused_real_plane_kernel(n=16, dtype=dtype, direction=direction, out_dir=out_dir)


def emit_fused_32_real_plane_kernel(*, dtype: str, direction: str, out_dir: Path) -> dict:
    """Emit one fused 32x32 R2C/C2R plane for the small-cube experiment."""
    return _emit_fused_real_plane_kernel(n=32, dtype=dtype, direction=direction, out_dir=out_dir)
