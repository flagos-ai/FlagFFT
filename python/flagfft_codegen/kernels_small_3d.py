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
from .kernels_common import _dtype_suffix, _ix_backend_active
from .metadata import _module_source, _signature


def emit_fused_plane_kernel(
    *, n: int, dtype: str, direction: str, out_dir: Path, real_input: bool = False
) -> dict:
    """Emit a small square plane FFT; one block owns one complete plane.

    Both axes use decimation in time. The load reverses the bits on each axis,
    then ``tl.gather`` performs four radix-two butterfly stages per axis. The
    twiddle buffers carry the forward/inverse sign and precision.
    """
    if n not in (16, 32):
        raise ValueError("fused plane supports 16 or 32")
    num_warps = 8
    if _ix_backend_active():
        value = os.getenv("FLAGFFT_IX_3D_PLANE_WARPS")
        if value is not None:
            if value not in {"2", "4", "8"}:
                raise ValueError("FLAGFFT_IX_3D_PLANE_WARPS must be 2, 4 or 8")
            num_warps = int(value)
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
        "num_warps": num_warps,
        "num_stages": 1,
        "batch_per_block": 1,
        "arg_names": args,
        "kernel_type": kind,
        "dtype": dtype,
        "direction": direction,
    }
    write_text_atomic(out_dir / f"{name}.json", json.dumps(metadata, sort_keys=True))
    return metadata
