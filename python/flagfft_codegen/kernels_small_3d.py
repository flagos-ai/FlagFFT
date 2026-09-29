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
    *, dtype: str, direction: str, out_dir: Path, plane_size: int = 16,
    real_kind: str | None = None,
) -> dict:
    """Emit a small square plane FFT; one block owns one complete plane.

    Both axes use decimation in time. The load reverses the bits on each axis,
    then ``tl.gather`` performs four radix-two butterfly stages per axis. The
    twiddle buffers carry the forward/inverse sign and precision.
    """
    if plane_size not in (16, 32):
        raise ValueError("fused plane size must be 16 or 32")
    if real_kind not in (None, "r2c", "c2r"):
        raise ValueError("real_kind must be r2c, c2r or None")
    if real_kind == "r2c" and direction != "forward":
        raise ValueError("real forward plane requires forward direction")
    if real_kind == "c2r" and direction != "inverse":
        raise ValueError("real inverse plane requires inverse direction")
    plane_elements = plane_size * plane_size
    stages = plane_size.bit_length() - 1
    half = plane_size // 2 + 1
    if real_kind == "r2c":
        scalar_dtype = "tl.float64" if dtype == "complex128" else "tl.float32"
        load = f"""    src = plane * {plane_elements} + rev_row * {plane_size} + rev_col
    xr = tl.load(in_ptr + src)
    xi = tl.full(({plane_elements},), 0.0, {scalar_dtype})
"""
        store = f"""    dst = (plane * {plane_size * half} + row * {half} + col) * 2
    tl.store(out_ptr + dst, xr, mask=col < {half})
    tl.store(out_ptr + dst + 1, xi, mask=col < {half})
"""
    elif real_kind == "c2r":
        load = f"""    reflected = rev_col >= {half}
    src_row = tl.where(reflected, (-rev_row) % {plane_size}, rev_row)
    src_col = tl.where(reflected, {plane_size} - rev_col, rev_col)
    src = (plane * {plane_size * half} + src_row * {half} + src_col) * 2
    xr = tl.load(in_ptr + src)
    xi = tl.load(in_ptr + src + 1)
    xi = tl.where(reflected, -xi, xi)
"""
        store = f"""    dst = plane * {plane_elements} + idx
    tl.store(out_ptr + dst, xr)
"""
    else:
        load = f"""    src = (plane * {plane_elements} + rev_row * {plane_size} + rev_col) * 2
    xr = tl.load(in_ptr + src)
    xi = tl.load(in_ptr + src + 1)
"""
        store = f"""    dst = (plane * {plane_elements} + idx) * 2
    tl.store(out_ptr + dst, xr)
    tl.store(out_ptr + dst + 1, xi)
"""
    kernel_name = f"fused_{plane_size}_{'real_' if real_kind else ''}plane_fft_kernel"
    source = f"""
@triton.jit
def {kernel_name}(in_ptr, out_ptr, tw_r_ptr, tw_i_ptr):
    plane = tl.program_id(0)
    idx = tl.arange(0, {plane_elements})
    row = idx // {plane_size}
    col = idx % {plane_size}
    rev_row = tl.full(({plane_elements},), 0, tl.int32)
    rev_col = tl.full(({plane_elements},), 0, tl.int32)
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

{store}
"""
    name = f"flagfft_jit_fused_{plane_size}_{'real_' if real_kind else ''}plane_{direction}_{_dtype_suffix(dtype)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    module_path = out_dir / f"{name}.py"
    write_text_atomic(module_path, _module_source(source))
    args = ["in_ptr", "out_ptr", "tw_r_ptr", "tw_i_ptr"]
    metadata = {
        "module_path": str(module_path),
        "kernel_name": kernel_name,
        "signature": _signature(args, dtype),
        "num_warps": 4 if _hcu_backend_active() else 8,
        "num_stages": 1,
        "batch_per_block": 1,
        "arg_names": args,
        "kernel_type": f"fused_{plane_size}_{'real_' if real_kind else ''}plane",
        "dtype": dtype,
        "direction": direction,
    }
    write_text_atomic(out_dir / f"{name}.json", json.dumps(metadata, sort_keys=True))
    return metadata
