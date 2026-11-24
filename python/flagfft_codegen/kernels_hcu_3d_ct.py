# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Two-pass multidimensional Cooley-Tukey FFT, with bounded local tiles."""

import json
import math

from .artifacts import write_text_atomic
from .backend_profile import current_profile
from .metadata import _module_source, _signature


def ct_geometry(n0, n1, n2, budget=8192):
    if (min(n0, n1) < 2 or not 4 <= n2 <= 256
            or any(n & (n - 1) for n in (n0, n1, n2))):
        return None
    h0 = h1 = 1
    while h0 * h1 * n2 * 2 <= budget and (h0 < n0 or h1 < n1):
        if n0 // h0 >= n1 // h1 and h0 < n0:
            h0 *= 2
        else:
            h1 *= 2
    l0, l1 = n0 // h0, n1 // h1
    tile = min(n2, 32)
    while tile > 1 and l0 * l1 * tile > budget:
        tile //= 2
    return (h0, h1, tile) if tile >= 4 else None


def _fft_axis(lines, shape, axis, inverse):
    n = shape[axis]
    if n == 1:
        return
    bits = n.bit_length() - 1
    axis_shape = [1, 1, 1]
    axis_shape[axis] = n
    shape_text = str(tuple(shape))
    lines.extend([
        f"    lane = tl.reshape(tl.arange(0, {n}), {tuple(axis_shape)})",
        "    reverse = tl.zeros_like(lane)",
    ])
    for bit in range(bits):
        lines.append(f"    reverse = reverse | (((lane >> {bit}) & 1) << {bits - bit - 1})")
    lines.extend([
        f"    route = tl.broadcast_to(reverse, {shape_text})",
        f"    xr = tl.gather(xr, route, {axis})",
        f"    xi = tl.gather(xi, route, {axis})",
    ])
    sign = 1 if inverse else -1
    for stage in range(bits):
        half = 1 << stage
        lines.extend([
            f"    route = tl.broadcast_to(lane ^ {half}, {shape_text})",
            f"    pr = tl.gather(xr, route, {axis})",
            f"    pi = tl.gather(xi, route, {axis})",
            f"    high = (lane & {half}) != 0",
            "    ar = tl.where(high, pr, xr)",
            "    ai = tl.where(high, pi, xi)",
            "    br = tl.where(high, xr, pr)",
            "    bi = tl.where(high, xi, pi)",
        ])
        if half == 1:
            lines.extend(["    tr = br", "    ti = bi"])
        else:
            lines.extend([
                f"    angle = (lane & {half - 1}) * {sign * math.pi / half:.17g}",
                "    wr = tl.cos(angle)", "    wi = tl.sin(angle)",
                "    tr = br * wr - bi * wi", "    ti = br * wi + bi * wr",
            ])
        lines.extend([
            "    xr = tl.where(high, ar - tr, ar + tr)",
            "    xi = tl.where(high, ai - ti, ai + ti)",
        ])


def build_ct_source(n0, n1, n2, h0, h1, direction, first):
    if (any(n <= 0 or n & (n - 1) for n in (n0, n1, n2, h0, h1))
            or n0 % h0 or n1 % h1 or direction not in {"forward", "inverse"}):
        raise ValueError("HCU CT FFT requires divisible power-of-two dimensions")
    l0, l1 = n0 // h0, n1 // h1
    tile = min(n2, 32)
    while tile > 1 and l0 * l1 * tile > 8192:
        tile //= 2
    if h0 * h1 * n2 > 8192 or tile < 4:
        raise ValueError("HCU CT tile exceeds its local register/shared-memory budget")
    total = n0 * n1 * n2
    name = f"hcu_ct3d_{'first' if first else 'last'}_{n0}_{n1}_{n2}_h{h0}_{h1}_{direction}"
    lines = ["@triton.jit", f"def {name}(in_ptr, out_ptr, nbatch):",
             "    pid = tl.program_id(0)", "    batch = tl.program_id(2).to(tl.int64)"]
    if first:
        shape = (h0, h1, n2)
        grid = l0 * l1
        lines.extend([
            f"    low0 = pid // {l1}", f"    low1 = pid % {l1}",
            f"    a = tl.arange(0, {h0})[:, None, None]",
            f"    b = tl.arange(0, {h1})[None, :, None]",
            f"    c = tl.arange(0, {n2})[None, None, :]",
            f"    source = batch * {total} + (low0 + a * {l0}) * {n1 * n2} + (low1 + b * {l1}) * {n2} + c",
            "    xr = tl.load(in_ptr + source * 2)",
            "    xi = tl.load(in_ptr + source * 2 + 1)",
        ])
        for axis in (2, 1, 0):
            _fft_axis(lines, shape, axis, direction == "inverse")
        lines.append(f"    target = batch * {total} + ((a * {h1} + b) * {l0 * l1} + low0 * {l1} + low1) * {n2} + c")
    else:
        shape = (l0, l1, tile)
        freq_tiles = n2 // tile
        grid = h0 * h1 * freq_tiles
        sign = 1 if direction == "inverse" else -1
        lines.extend([
            f"    high0 = pid // {h1 * freq_tiles}",
            f"    high1 = (pid // {freq_tiles}) % {h1}",
            f"    freq_base = (pid % {freq_tiles}) * {tile}",
            f"    a = tl.arange(0, {l0})[:, None, None]",
            f"    b = tl.arange(0, {l1})[None, :, None]",
            f"    c = freq_base + tl.arange(0, {tile})[None, None, :]",
            f"    source = batch * {total} + ((high0 * {h1} + high1) * {l0 * l1} + a * {l1} + b) * {n2} + c",
            "    xr = tl.load(in_ptr + source * 2)",
            "    xi = tl.load(in_ptr + source * 2 + 1)",
            f"    angle = a * high0 * {sign * 2 * math.pi / n0:.17g} + b * high1 * {sign * 2 * math.pi / n1:.17g}",
            "    wr = tl.cos(angle)", "    wi = tl.sin(angle)",
            "    tr = xr * wr - xi * wi", "    xi = xr * wi + xi * wr", "    xr = tr",
        ])
        for axis in (1, 0):
            _fft_axis(lines, shape, axis, direction == "inverse")
        lines.append(f"    target = batch * {total} + (high0 + a * {h0}) * {n1 * n2} + (high1 + b * {h1}) * {n2} + c")
    lines.extend(["    tl.store(out_ptr + target * 2, xr)",
                  "    tl.store(out_ptr + target * 2 + 1, xi)"])
    return name, "\n".join(lines) + "\n", grid


def emit_ct_kernel(*, n0, n1, n2, h0, h1, direction, first, dtype, out_dir):
    if dtype != "complex64":
        raise ValueError("HCU CT prototype supports complex64")
    name, source, grid = build_ct_source(n0, n1, n2, h0, h1, direction, first)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (name + ".py")
    write_text_atomic(path, _module_source(source))
    args = ["in_ptr", "out_ptr", "nbatch"]
    metadata = dict(module_path=str(path), kernel_name=name,
                    signature=_signature(args, dtype), arg_names=args,
                    num_warps=8, num_stages=1, batch_per_block=1,
                    grid_x_override=grid, dtype=dtype,
                    kernel_type="hcu_3d_ct_first" if first else "hcu_3d_ct_last")
    current_profile().validate(metadata["num_warps"])
    metadata.update(warp_size=current_profile().warp_size,
                    profile_id=current_profile().fingerprint)
    write_text_atomic(out_dir / (name + ".json"), json.dumps(metadata, sort_keys=True))
    return metadata
