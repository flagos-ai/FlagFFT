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


def emit_fused_2d_kernel(
    *, n: int, dtype: str, direction: str, transpose_output: bool, out_dir: Path
) -> dict:
    """Emit one radix-2 FFT per program and fold the axis permutation into stores.

    A program owns one row. The 2D executor launches this kernel twice; each
    launch transforms one axis and writes the result transposed for the next
    axis. This bounds the NPU UB use to a single 1D transform.
    """
    if n != 64:
        raise ValueError("experimental fused 2D kernel currently supports only 64x64")
    if direction not in ("forward", "inverse"):
        raise ValueError("direction must be forward or inverse")

    bits = n.bit_length() - 1
    quarter_r, quarter_i = ("bi", "-br") if direction == "forward" else ("-bi", "br")
    store_expr = (
        f"(batch * {n * n} + idx * {n} + row) * 2"
        if transpose_output else
        f"(batch * {n * n} + row * {n} + idx) * 2"
    )
    suffix = "transpose" if transpose_output else "rowmajor"
    source = f"""
@triton.jit
def fused_2d_fft_kernel(in_ptr, out_ptr, tw_r_ptr, tw_i_ptr):
    row_id = tl.program_id(0)
    batch = row_id // {n}
    row = row_id % {n}
    # Keep the gather vector at the NPU's safe exchange width.  The Ascend
    # lowering can address past UB for gathers over a bare 64-element vector;
    # the regular leaf emitter avoids this by padding its exchange to 128.
    idx = tl.arange(0, 128)
    logical_idx = idx & ({n} - 1)
    valid = idx < {n}
    rev = tl.full((128,), 0, tl.int32)
    for bit in tl.static_range({bits}):
        rev = (rev << 1) | ((logical_idx >> bit) & 1)
    src = (batch * {n * n} + row * {n} + rev) * 2
    xr = tl.load(in_ptr + src, mask=valid, other=0.0)
    xi = tl.load(in_ptr + src + 1, mask=valid, other=0.0)

    for stage in tl.static_range({bits}):
        partner = logical_idx ^ (1 << stage)
        pr = tl.gather(xr, partner, 0)
        pi = tl.gather(xi, partner, 0)
        upper = (logical_idx & (1 << stage)) != 0
        ar = tl.where(upper, pr, xr)
        ai = tl.where(upper, pi, xi)
        br = tl.where(upper, xr, pr)
        bi = tl.where(upper, xi, pi)
        if stage == 0:
            tr = br
            ti = bi
        elif stage == 1:
            tr = tl.where((logical_idx & 1) != 0, {quarter_r}, br)
            ti = tl.where((logical_idx & 1) != 0, {quarter_i}, bi)
        else:
            tw = (logical_idx & ((1 << stage) - 1)) * ({n} >> (stage + 1))
            wr = tl.load(tw_r_ptr + tw, mask=valid, other=0.0)
            wi = tl.load(tw_i_ptr + tw, mask=valid, other=0.0)
            tr = wr * br - wi * bi
            ti = wr * bi + wi * br
        xr = tl.where(upper, ar - tr, ar + tr)
        xi = tl.where(upper, ai - ti, ai + ti)

    dst = {store_expr}
    tl.store(out_ptr + dst, xr, mask=valid)
    tl.store(out_ptr + dst + 1, xi, mask=valid)
"""
    name = f"flagfft_jit_fused_2d_{direction}_{_dtype_suffix(dtype)}_n{n}_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    module_path = out_dir / f"{name}.py"
    write_text_atomic(module_path, _module_source(source))
    args = ["in_ptr", "out_ptr", "tw_r_ptr", "tw_i_ptr"]
    metadata = {
        "module_path": str(module_path),
        "kernel_name": "fused_2d_fft_kernel",
        "signature": _signature(args, dtype),
        # Keep each program to one row, while using two hardware threads for
        # the 128-wide exchange vector, matching the working portable leaf.
        "num_warps": 2,
        "num_stages": 1,
        "batch_per_block": 1,
        "arg_names": args,
        "kernel_type": "fused_2d",
        "dtype": dtype,
        "direction": direction,
        "length": n,
        "transpose_output": transpose_output,
    }
    write_text_atomic(out_dir / f"{name}.json", json.dumps(metadata, sort_keys=True))
    return metadata
