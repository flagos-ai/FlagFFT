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
from .kernels_common import _dtype_suffix
from .metadata import _module_source, _signature


def emit_fused_plane_kernel(
    *, n: int, dtype: str, direction: str, out_dir: Path, real_input: bool = False,
    target: str = "",
) -> dict:
    """Emit a small square plane FFT; one block owns one complete plane.

    Both axes use decimation in time. The load reverses the bits on each axis,
    then ``tl.gather`` performs four radix-two butterfly stages per axis. The
    twiddle buffers carry the forward/inverse sign and precision.
    """
    if n not in (16, 32):
        raise ValueError("fused plane supports 16 or 32")
    bits = n.bit_length() - 1
    plane_size = n * n
    load_source = (
        f"xr = tl.load(in_ptr + plane * {plane_size} + rev_row * {n} + rev_col)\n"
        f"    xi = tl.full(({plane_size},), 0.0, tl.float32)"
        if real_input else
        "xr = tl.load(in_ptr + src)\n    xi = tl.load(in_ptr + src + 1)"
    )
    store_source = (
        f"dst = (plane * {n} * {n // 2 + 1} + row * {n // 2 + 1} + col) * 2\n"
        f"    keep = col <= {n // 2}\n"
        "    tl.store(out_ptr + dst, xr, mask=keep)\n"
        "    tl.store(out_ptr + dst + 1, xi, mask=keep)"
        if real_input else
        f"dst = (plane * {plane_size} + idx) * 2\n"
        "    tl.store(out_ptr + dst, xr)\n"
        "    tl.store(out_ptr + dst + 1, xi)"
    )
    source = f"""
@triton.jit
def fused_plane_fft_kernel(in_ptr, out_ptr, tw_r_ptr, tw_i_ptr):
    plane = tl.program_id(0)
    idx = tl.arange(0, {plane_size})
    row = idx // {n}
    col = idx % {n}
    rev_row = tl.full(({plane_size},), 0, tl.int32)
    rev_col = tl.full(({plane_size},), 0, tl.int32)
    for bit in tl.static_range({bits}):
        rev_row = (rev_row << 1) | ((row >> bit) & 1)
        rev_col = (rev_col << 1) | ((col >> bit) & 1)
    src = (plane * {plane_size} + rev_row * {n} + rev_col) * 2
    {load_source}

    for stage in tl.static_range({bits}):
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

    for stage in tl.static_range({bits}):
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

    {store_source}
"""
    kind = f"fused_{n}_{'real_' if real_input else ''}plane"
    name = f"flagfft_jit_{kind}_{direction}_{_dtype_suffix(dtype)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    module_path = out_dir / f"{name}.py"
    write_text_atomic(module_path, _module_source(source))
    args = ["in_ptr", "out_ptr", "tw_r_ptr", "tw_i_ptr"]
    metadata = {
        "module_path": str(module_path),
        "kernel_name": "fused_plane_fft_kernel",
        "signature": _signature(args, dtype),
        "num_warps": 16 if n == 32 and target.startswith("ix:") else 8,
        "num_stages": 1,
        "batch_per_block": 1,
        "arg_names": args,
        "kernel_type": kind,
        "dtype": dtype,
        "direction": direction,
    }
    write_text_atomic(out_dir / f"{name}.json", json.dumps(metadata, sort_keys=True))
    return metadata


def emit_fused_16_cube_kernel(
    *, dtype: str, direction: str, out_dir: Path, real_input: bool = False
) -> dict:
    """Compute one output frequency plane per block for a single 16^3 cube.

    The outer axis uses a direct 16-point sum across planes. The remaining
    16x16 transform stays in the block, avoiding a second kernel launch.
    """
    load_source = (
        "vr = tl.load(in_ptr + batch * 4096 + i0 * 256 + idx)\n"
        "        vi = tl.full((256,), 0.0, tl.float32)"
        if real_input else
        "src = (batch * 4096 + i0 * 256 + idx) * 2\n"
        "        vr = tl.load(in_ptr + src)\n"
        "        vi = tl.load(in_ptr + src + 1)"
    )
    store_source = (
        "dst = (batch * 16 * 16 * 9 + k0 * 16 * 9 + row * 9 + col) * 2\n"
        "    keep = col <= 8\n"
        "    tl.store(out_ptr + dst, xr, mask=keep)\n"
        "    tl.store(out_ptr + dst + 1, xi, mask=keep)"
        if real_input else
        "dst = (batch * 4096 + k0 * 256 + idx) * 2\n"
        "    tl.store(out_ptr + dst, xr)\n"
        "    tl.store(out_ptr + dst + 1, xi)"
    )
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
        {load_source}
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

    {store_source}
"""
    kind = "fused_16_real_cube" if real_input else "fused_16_cube"
    name = f"flagfft_jit_{kind}_{direction}_{_dtype_suffix(dtype)}"
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
        "kernel_type": kind,
        "dtype": dtype,
        "direction": direction,
    }
    write_text_atomic(out_dir / f"{name}.json", json.dumps(metadata, sort_keys=True))
    return metadata


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
