# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Experimental fused square 2D FFT kernels."""

from __future__ import annotations

import json
from pathlib import Path

from .artifacts import write_text_atomic
from .kernels_common import _dtype_suffix
from .metadata import _module_source, _signature


def emit_fused_2d_kernel(*, n: int, dtype: str, direction: str, out_dir: Path) -> dict:
    """Emit a single-program radix-2 square C2C transform.

    One program owns a complete plane and performs both dimensions in
    registers. This first experiment deliberately targets only 64x64; larger
    planes need a different tiling strategy to bound the register footprint.
    """
    if n != 64:
        raise ValueError("experimental fused 2D kernel currently supports only 64x64")
    if direction not in ("forward", "inverse"):
        raise ValueError("direction must be forward or inverse")

    bits = n.bit_length() - 1
    quarter_r, quarter_i = ("bi", "-br") if direction == "forward" else ("-bi", "br")
    source = f"""
@triton.jit
def fused_2d_fft_kernel(in_ptr, out_ptr, tw_r_ptr, tw_i_ptr):
    batch = tl.program_id(0)
    idx = tl.arange(0, {n * n})
    row = idx // {n}
    col = idx % {n}
    rev_row = tl.full(({n * n},), 0, tl.int32)
    rev_col = tl.full(({n * n},), 0, tl.int32)
    for bit in tl.static_range({bits}):
        rev_row = (rev_row << 1) | ((row >> bit) & 1)
        rev_col = (rev_col << 1) | ((col >> bit) & 1)
    src = (batch * {n * n} + rev_row * {n} + rev_col) * 2
    xr = tl.load(in_ptr + src)
    xi = tl.load(in_ptr + src + 1)

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

    dst = (batch * {n * n} + idx) * 2
    tl.store(out_ptr + dst, xr)
    tl.store(out_ptr + dst + 1, xi)
"""
    name = f"flagfft_jit_fused_2d_{direction}_{_dtype_suffix(dtype)}_n{n}"
    out_dir.mkdir(parents=True, exist_ok=True)
    module_path = out_dir / f"{name}.py"
    write_text_atomic(module_path, _module_source(source))
    args = ["in_ptr", "out_ptr", "tw_r_ptr", "tw_i_ptr"]
    metadata = {
        "module_path": str(module_path),
        "kernel_name": "fused_2d_fft_kernel",
        "signature": _signature(args, dtype),
        "num_warps": 4,
        "num_stages": 1,
        "batch_per_block": 1,
        "arg_names": args,
        "kernel_type": "fused_2d",
        "dtype": dtype,
        "direction": direction,
        "length": n,
    }
    write_text_atomic(out_dir / f"{name}.json", json.dumps(metadata, sort_keys=True))
    return metadata
