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
from pathlib import Path

from .artifacts import write_text_atomic
from .kernels_common import _dtype_suffix, _hcu_backend_active
from .metadata import _module_source, _signature


def emit_fused_16_plane_kernel(
    *, dtype: str, direction: str, out_dir: Path, plane_size: int = 16
) -> dict:
    """Emit a small square plane FFT; one block owns one complete plane.

    Both axes use decimation in time. The load reverses the bits on each axis,
    then ``tl.gather`` performs four radix-two butterfly stages per axis. The
    twiddle buffers carry the forward/inverse sign and precision.
    """
    if plane_size not in (16, 32):
        raise ValueError("fused plane size must be 16 or 32")
    plane_elements = plane_size * plane_size
    stages = plane_size.bit_length() - 1
    source = f"""
@triton.jit
def fused_{plane_size}_plane_fft_kernel(in_ptr, out_ptr, tw_r_ptr, tw_i_ptr):
    plane = tl.program_id(0)
    idx = tl.arange(0, {plane_elements})
    row = idx // {plane_size}
    col = idx % {plane_size}
    rev_row = tl.full(({plane_elements},), 0, tl.int32)
    rev_col = tl.full(({plane_elements},), 0, tl.int32)
    for bit in tl.static_range({stages}):
        rev_row = (rev_row << 1) | ((row >> bit) & 1)
        rev_col = (rev_col << 1) | ((col >> bit) & 1)
    src = (plane * {plane_elements} + rev_row * {plane_size} + rev_col) * 2
    xr = tl.load(in_ptr + src)
    xi = tl.load(in_ptr + src + 1)

    for stage in tl.static_range({stages}):
        partner = idx ^ (1 << stage)
        pr = tl.gather(xr, partner, 0)
        pi = tl.gather(xi, partner, 0)
        upper = (col & (1 << stage)) != 0
        ar = tl.where(upper, pr, xr)
        ai = tl.where(upper, pi, xi)
        br = tl.where(upper, xr, pr)
        bi = tl.where(upper, xi, pi)
        tw = (col & ((1 << stage) - 1)) * ({plane_size} >> (stage + 1))
        wr = tl.load(tw_r_ptr + tw)
        wi = tl.load(tw_i_ptr + tw)
        tr = wr * br - wi * bi
        ti = wr * bi + wi * br
        xr = tl.where(upper, ar - tr, ar + tr)
        xi = tl.where(upper, ai - ti, ai + ti)

    for stage in tl.static_range({stages}):
        partner = idx ^ ({plane_size} << stage)
        pr = tl.gather(xr, partner, 0)
        pi = tl.gather(xi, partner, 0)
        upper = (row & (1 << stage)) != 0
        ar = tl.where(upper, pr, xr)
        ai = tl.where(upper, pi, xi)
        br = tl.where(upper, xr, pr)
        bi = tl.where(upper, xi, pi)
        tw = (row & ((1 << stage) - 1)) * ({plane_size} >> (stage + 1))
        wr = tl.load(tw_r_ptr + tw)
        wi = tl.load(tw_i_ptr + tw)
        tr = wr * br - wi * bi
        ti = wr * bi + wi * br
        xr = tl.where(upper, ar - tr, ar + tr)
        xi = tl.where(upper, ai - ti, ai + ti)

    dst = (plane * {plane_elements} + idx) * 2
    tl.store(out_ptr + dst, xr)
    tl.store(out_ptr + dst + 1, xi)
"""
    name = f"flagfft_jit_fused_{plane_size}_plane_{direction}_{_dtype_suffix(dtype)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    module_path = out_dir / f"{name}.py"
    write_text_atomic(module_path, _module_source(source))
    args = ["in_ptr", "out_ptr", "tw_r_ptr", "tw_i_ptr"]
    metadata = {
        "module_path": str(module_path),
        "kernel_name": f"fused_{plane_size}_plane_fft_kernel",
        "signature": _signature(args, dtype),
        "num_warps": 4 if _hcu_backend_active() else 8,
        "num_stages": 1,
        "batch_per_block": 1,
        "arg_names": args,
        "kernel_type": f"fused_{plane_size}_plane",
        "dtype": dtype,
        "direction": direction,
    }
    write_text_atomic(out_dir / f"{name}.json", json.dumps(metadata, sort_keys=True))
    return metadata


def emit_fused_32_real_plane_kernel(
    *, dtype: str, direction: str, out_dir: Path
) -> dict:
    """Fuse the two inner axes of a 32x32 R2C or C2R plane."""
    if dtype not in {"complex64", "complex128"}:
        raise ValueError("fused real plane requires a complex precision")
    if direction not in {"forward", "inverse"}:
        raise ValueError("fused real plane direction must be forward or inverse")
    scalar = "tl.float64" if dtype == "complex128" else "tl.float32"
    inverse = direction == "inverse"
    if inverse:
        input_load = """
    logical_row = rev_row
    logical_col = rev_col
    mirrored = logical_col > 16
    source_row = tl.where(mirrored, (-logical_row) & 31, logical_row)
    source_col = tl.where(mirrored, 32 - logical_col, logical_col)
    src = (plane * 32 * 17 + source_row * 17 + source_col) * 2
    xr = tl.load(in_ptr + src)
    xi = tl.load(in_ptr + src + 1)
    xi = tl.where(mirrored, -xi, xi)
"""
        output_store = """
    dst = plane * 32 * 32 + idx
    tl.store(out_ptr + dst, xr)
"""
        variant = "fused_32_c2r"
    else:
        input_load = f"""
    src = plane * 32 * 32 + rev_row * 32 + rev_col
    xr = tl.load(in_ptr + src)
    xi = tl.full((1024,), 0.0, {scalar})
"""
        output_store = """
    keep = col <= 16
    dst = (plane * 32 * 17 + row * 17 + col) * 2
    tl.store(out_ptr + dst, xr, mask=keep)
    tl.store(out_ptr + dst + 1, xi, mask=keep)
"""
        variant = "fused_32_r2c"
    source = f"""
@triton.jit
def fused_32_real_plane_fft_kernel(in_ptr, out_ptr, tw_r_ptr, tw_i_ptr):
    plane = tl.program_id(0)
    idx = tl.arange(0, 1024)
    row = idx // 32
    col = idx % 32
    rev_row = tl.full((1024,), 0, tl.int32)
    rev_col = tl.full((1024,), 0, tl.int32)
    for bit in tl.static_range(5):
        rev_row = (rev_row << 1) | ((row >> bit) & 1)
        rev_col = (rev_col << 1) | ((col >> bit) & 1)
{input_load}

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

{output_store}
"""
    name = f"flagfft_jit_{variant}_{direction}_{_dtype_suffix(dtype)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    module_path = out_dir / f"{name}.py"
    write_text_atomic(module_path, _module_source(source))
    args = ["in_ptr", "out_ptr", "tw_r_ptr", "tw_i_ptr"]
    metadata = {
        "module_path": str(module_path),
        "kernel_name": "fused_32_real_plane_fft_kernel",
        "signature": _signature(args, dtype),
        "num_warps": 4 if _hcu_backend_active() else 8,
        "num_stages": 1,
        "batch_per_block": 1,
        "arg_names": args,
        "kernel_type": "fused_32_real_plane",
        "dtype": dtype,
        "direction": direction,
    }
    write_text_atomic(out_dir / f"{name}.json", json.dumps(metadata, sort_keys=True))
    return metadata


def emit_fused_rect_plane_kernel(
    *,
    dtype: str,
    direction: str,
    out_dir: Path,
    plane_n0: int,
    plane_n1: int,
    middle_size: int,
) -> dict:
    """Emit a rectangular two-axis complex FFT plane with permuted output.

    The HCU 16x64 variant handles both short axes of a 16x997x64 transform in
    one block per middle-axis coordinate. Its output is laid out as
    (n0,n2,n1), so the long middle-axis FFT can consume contiguous rows.
    """
    if plane_n0 != 16 or plane_n1 != 64 or middle_size != 997:
        raise ValueError("fused rectangular plane currently supports 16x64 with middle size 997")
    plane_elements = plane_n0 * plane_n1
    stages0 = plane_n0.bit_length() - 1
    stages1 = plane_n1.bit_length() - 1
    source = f"""
@triton.jit
def fused_16x64_plane_fft_kernel(in_ptr, out_ptr, tw_r_ptr, tw_i_ptr):
    plane = tl.program_id(0)
    batch = plane // {middle_size}
    middle = plane % {middle_size}
    idx = tl.arange(0, {plane_elements})
    row = idx // {plane_n1}
    col = idx % {plane_n1}
    rev_row = tl.full(({plane_elements},), 0, tl.int32)
    rev_col = tl.full(({plane_elements},), 0, tl.int32)
    for bit in tl.static_range({stages0}):
        rev_row = (rev_row << 1) | ((row >> bit) & 1)
    for bit in tl.static_range({stages1}):
        rev_col = (rev_col << 1) | ((col >> bit) & 1)
    src = (((batch * {plane_n0} + rev_row) * {middle_size} + middle) * {plane_n1} + rev_col) * 2
    xr = tl.load(in_ptr + src)
    xi = tl.load(in_ptr + src + 1)

    for stage in tl.static_range({stages1}):
        partner = idx ^ (1 << stage)
        pr = tl.gather(xr, partner, 0)
        pi = tl.gather(xi, partner, 0)
        upper = (col & (1 << stage)) != 0
        ar = tl.where(upper, pr, xr)
        ai = tl.where(upper, pi, xi)
        br = tl.where(upper, xr, pr)
        bi = tl.where(upper, xi, pi)
        tw = (col & ((1 << stage) - 1)) * ({plane_n1} >> (stage + 1))
        wr = tl.load(tw_r_ptr + tw)
        wi = tl.load(tw_i_ptr + tw)
        tr = wr * br - wi * bi
        ti = wr * bi + wi * br
        xr = tl.where(upper, ar - tr, ar + tr)
        xi = tl.where(upper, ai - ti, ai + ti)

    for stage in tl.static_range({stages0}):
        partner = idx ^ ({plane_n1} << stage)
        pr = tl.gather(xr, partner, 0)
        pi = tl.gather(xi, partner, 0)
        upper = (row & (1 << stage)) != 0
        ar = tl.where(upper, pr, xr)
        ai = tl.where(upper, pi, xi)
        br = tl.where(upper, xr, pr)
        bi = tl.where(upper, xi, pi)
        tw = (row & ((1 << stage) - 1)) * ({plane_n1} >> (stage + 1))
        wr = tl.load(tw_r_ptr + tw)
        wi = tl.load(tw_i_ptr + tw)
        tr = wr * br - wi * bi
        ti = wr * bi + wi * br
        xr = tl.where(upper, ar - tr, ar + tr)
        xi = tl.where(upper, ai - ti, ai + ti)

    dst = ((batch * {plane_n0} + row) * {plane_n1} + col) * {middle_size} + middle
    tl.store(out_ptr + dst * 2, xr)
    tl.store(out_ptr + dst * 2 + 1, xi)
"""
    name = f"flagfft_jit_fused_rect_plane_{plane_n0}x{plane_n1}x{middle_size}_{direction}_{_dtype_suffix(dtype)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    module_path = out_dir / f"{name}.py"
    write_text_atomic(module_path, _module_source(source))
    args = ["in_ptr", "out_ptr", "tw_r_ptr", "tw_i_ptr"]
    metadata = {
        "module_path": str(module_path),
        "kernel_name": "fused_16x64_plane_fft_kernel",
        "signature": _signature(args, dtype),
        "num_warps": 4,
        "num_stages": 1,
        "batch_per_block": 1,
        "arg_names": args,
        "kernel_type": "fused_rect_plane",
        "dtype": dtype,
        "direction": direction,
        "plane_n0": plane_n0,
        "plane_n1": plane_n1,
        "middle_size": middle_size,
    }
    write_text_atomic(out_dir / f"{name}.json", json.dumps(metadata, sort_keys=True))
    return metadata


def emit_fused_16_cube_kernel(*, dtype: str, direction: str, out_dir: Path) -> dict:
    """Compute one output plane per block, including the outer 16-point DFT.

    Each block rereads the small cube, so there is no global synchronization
    between axes. The outer DFT is evaluated directly before the two inner
    radix-two FFTs; this trades redundant cached reads for one launch.
    """
    scalar = "tl.float64" if dtype == "complex128" else "tl.float32"
    bits = 4
    n = 16
    quarter_r, quarter_i = ("bi", "-br") if direction == "forward" else ("-bi", "br")
    source = f"""
@triton.jit
def fused_16_cube_fft_kernel(in_ptr, out_ptr, tw_r_ptr, tw_i_ptr):
    output_plane = tl.program_id(0) % 16
    cube = tl.program_id(0) // 16
    idx = tl.arange(0, 256)
    row = idx // 16
    col = idx % 16
    rev_row = tl.full((256,), 0, tl.int32)
    rev_col = tl.full((256,), 0, tl.int32)
    for bit in tl.static_range(4):
        rev_row = (rev_row << 1) | ((row >> bit) & 1)
        rev_col = (rev_col << 1) | ((col >> bit) & 1)

    xr = tl.full((256,), 0.0, {scalar})
    xi = tl.full((256,), 0.0, {scalar})
    for input_plane in tl.static_range(16):
        src = ((cube * 16 + input_plane) * 256 + rev_row * 16 + rev_col) * 2
        ar = tl.load(in_ptr + src)
        ai = tl.load(in_ptr + src + 1)
        phase = (input_plane * output_plane) % 16
        wr = tl.load(tw_r_ptr + phase)
        wi = tl.load(tw_i_ptr + phase)
        xr = xr + wr * ar - wi * ai
        xi = xi + wr * ai + wi * ar

    for stage in tl.static_range({bits}):
        partner = idx ^ (1 << stage)
        pr = tl.gather(xr, partner, 0)
        pi = tl.gather(xi, partner, 0)
        upper = (col & (1 << stage)) != 0
        ar = tl.where(upper, pr, xr)
        ai = tl.where(upper, pi, xi)
        br = tl.where(upper, xr, pr)
        bi = tl.where(upper, xi, pi)
        if stage == 0:
            tr = br
            ti = bi
        elif stage == 1:
            tr = tl.where((col & 1) != 0, {quarter_r}, br)
            ti = tl.where((col & 1) != 0, {quarter_i}, bi)
        else:
            tw = (col & ((1 << stage) - 1)) * ({n} >> (stage + 1))
            wr = tl.load(tw_r_ptr + tw)
            wi = tl.load(tw_i_ptr + tw)
            tr = wr * br - wi * bi
            ti = wr * bi + wi * br
        xr = tl.where(upper, ar - tr, ar + tr)
        xi = tl.where(upper, ai - ti, ai + ti)

    for stage in tl.static_range({bits}):
        partner = idx ^ ({n} << stage)
        pr = tl.gather(xr, partner, 0)
        pi = tl.gather(xi, partner, 0)
        upper = (row & (1 << stage)) != 0
        ar = tl.where(upper, pr, xr)
        ai = tl.where(upper, pi, xi)
        br = tl.where(upper, xr, pr)
        bi = tl.where(upper, xi, pi)
        if stage == 0:
            tr = br
            ti = bi
        elif stage == 1:
            tr = tl.where((row & 1) != 0, {quarter_r}, br)
            ti = tl.where((row & 1) != 0, {quarter_i}, bi)
        else:
            tw = (row & ((1 << stage) - 1)) * ({n} >> (stage + 1))
            wr = tl.load(tw_r_ptr + tw)
            wi = tl.load(tw_i_ptr + tw)
            tr = wr * br - wi * bi
            ti = wr * bi + wi * br
        xr = tl.where(upper, ar - tr, ar + tr)
        xi = tl.where(upper, ai - ti, ai + ti)

    dst = ((cube * 16 + output_plane) * 256 + idx) * 2
    tl.store(out_ptr + dst, xr)
    tl.store(out_ptr + dst + 1, xi)
"""
    name = f"flagfft_jit_fused_16_cube_{direction}_{_dtype_suffix(dtype)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    module_path = out_dir / f"{name}.py"
    write_text_atomic(module_path, _module_source(source))
    args = ["in_ptr", "out_ptr", "tw_r_ptr", "tw_i_ptr"]
    metadata = {
        "module_path": str(module_path),
        "kernel_name": "fused_16_cube_fft_kernel",
        "signature": _signature(args, dtype),
        "num_warps": 4,
        "num_stages": 1,
        "batch_per_block": 1,
        "arg_names": args,
        "kernel_type": "fused_16_cube",
        "dtype": dtype,
        "direction": direction,
    }
    write_text_atomic(out_dir / f"{name}.json", json.dumps(metadata, sort_keys=True))
    return metadata


def emit_fused_16_real_cube_kernel(*, dtype: str, direction: str, out_dir: Path) -> dict:
    """Compute one compact real-to-complex frequency plane per block."""
    source = f"""
@triton.jit
def fused_16_cube_fft_kernel(in_ptr, out_ptr, tw_r_ptr, tw_i_ptr):
    pid = tl.program_id(0)
    k0 = pid % 16
    batch = pid // 16
    idx = tl.arange(0, 256)
    row = idx // 16
    col = idx % 16
    ar = tl.full((256,), 0.0, tl.float32)
    ai = tl.full((256,), 0.0, tl.float32)
    for i0 in tl.static_range(16):
        vr = tl.load(in_ptr + batch * 4096 + i0 * 256 + idx)
        vi = tl.full((256,), 0.0, tl.float32)
        tw = (k0 * i0) % 16
        wr = tl.load(tw_r_ptr + tw)
        wi = tl.load(tw_i_ptr + tw)
        ar += wr * vr - wi * vi
        ai += wr * vi + wi * vr

    rev_row = tl.full((256,), 0, tl.int32)
    rev_col = tl.full((256,), 0, tl.int32)
    for bit in tl.static_range(4):
        rev_row = (rev_row << 1) | ((row >> bit) & 1)
        rev_col = (rev_col << 1) | ((col >> bit) & 1)
    reverse = rev_row * 16 + rev_col
    xr = tl.gather(ar, reverse, 0)
    xi = tl.gather(ai, reverse, 0)

    for stage in tl.static_range(4):
        partner = idx ^ (1 << stage)
        pr = tl.gather(xr, partner, 0)
        pi = tl.gather(xi, partner, 0)
        upper = (col & (1 << stage)) != 0
        br = tl.where(upper, xr, pr)
        bi = tl.where(upper, xi, pi)
        cr = tl.where(upper, pr, xr)
        ci = tl.where(upper, pi, xi)
        tw = (col & ((1 << stage) - 1)) * (16 >> (stage + 1))
        wr = tl.load(tw_r_ptr + tw)
        wi = tl.load(tw_i_ptr + tw)
        tr = wr * br - wi * bi
        ti = wr * bi + wi * br
        xr = tl.where(upper, cr - tr, cr + tr)
        xi = tl.where(upper, ci - ti, ci + ti)

    for stage in tl.static_range(4):
        partner = idx ^ (16 << stage)
        pr = tl.gather(xr, partner, 0)
        pi = tl.gather(xi, partner, 0)
        upper = (row & (1 << stage)) != 0
        br = tl.where(upper, xr, pr)
        bi = tl.where(upper, xi, pi)
        cr = tl.where(upper, pr, xr)
        ci = tl.where(upper, pi, xi)
        tw = (row & ((1 << stage) - 1)) * (16 >> (stage + 1))
        wr = tl.load(tw_r_ptr + tw)
        wi = tl.load(tw_i_ptr + tw)
        tr = wr * br - wi * bi
        ti = wr * bi + wi * br
        xr = tl.where(upper, cr - tr, cr + tr)
        xi = tl.where(upper, ci - ti, ci + ti)

    dst = (batch * 16 * 16 * 9 + k0 * 16 * 9 + row * 9 + col) * 2
    keep = col <= 8
    tl.store(out_ptr + dst, xr, mask=keep)
    tl.store(out_ptr + dst + 1, xi, mask=keep)
"""
    name = f"flagfft_jit_fused_16_real_cube_{direction}_{_dtype_suffix(dtype)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    module_path = out_dir / f"{name}.py"
    write_text_atomic(module_path, _module_source(source))
    args = ["in_ptr", "out_ptr", "tw_r_ptr", "tw_i_ptr"]
    metadata = {
        "module_path": str(module_path),
        "kernel_name": "fused_16_cube_fft_kernel",
        "signature": _signature(args, dtype),
        "num_warps": 4,
        "num_stages": 1,
        "batch_per_block": 1,
        "arg_names": args,
        "kernel_type": "fused_16_real_cube",
        "dtype": dtype,
        "direction": direction,
    }
    write_text_atomic(out_dir / f"{name}.json", json.dumps(metadata, sort_keys=True))
    return metadata


def emit_fused_32_plane_kernel(*, dtype: str, direction: str, out_dir: Path) -> dict:
    """Emit a 32x32 complex plane FFT for the MACA 3D fusion experiment."""
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
        "num_warps": 8,
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


def emit_fused_32_column_kernel(*, dtype: str, direction: str, out_dir: Path) -> dict:
    """Transform 32 outer-axis points for sixteen adjacent output columns."""
    source = """
@triton.jit
def fused_32_column_fft_kernel(in_ptr, out_ptr, tw_r_ptr, tw_i_ptr, outer_stride):
    tile = tl.program_id(0)
    batch = tl.program_id(1)
    idx = tl.arange(0, 512)
    row = idx // 16
    col = tile * 16 + idx % 16
    reverse_row = tl.full((512,), 0, tl.int32)
    for bit in tl.static_range(5):
        reverse_row = (reverse_row << 1) | ((row >> bit) & 1)
    src = (batch * 32 * outer_stride + reverse_row * outer_stride + col) * 2
    valid = col < outer_stride
    xr = tl.load(in_ptr + src, mask=valid, other=0.0)
    xi = tl.load(in_ptr + src + 1, mask=valid, other=0.0)

    for stage in tl.static_range(5):
        partner = idx ^ (16 << stage)
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

    dst = (batch * 32 * outer_stride + row * outer_stride + col) * 2
    tl.store(out_ptr + dst, xr, mask=valid)
    tl.store(out_ptr + dst + 1, xi, mask=valid)
"""
    name = f"flagfft_jit_fused_32_column_{direction}_{_dtype_suffix(dtype)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    module_path = out_dir / f"{name}.py"
    write_text_atomic(module_path, _module_source(source))
    args = ["in_ptr", "out_ptr", "tw_r_ptr", "tw_i_ptr", "outer_stride"]
    metadata = {
        "module_path": str(module_path),
        "kernel_name": "fused_32_column_fft_kernel",
        "signature": _signature(args, dtype),
        "num_warps": 4,
        "num_stages": 1,
        "batch_per_block": 1,
        "arg_names": args,
        "kernel_type": "fused_32_column",
        "dtype": dtype,
        "direction": direction,
    }
    write_text_atomic(out_dir / f"{name}.json", json.dumps(metadata, sort_keys=True))
    return metadata
