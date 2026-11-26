# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Good–Thomas spectral and scatter kernels for C550 Rader FFTs."""

import math


SPECTRAL_PACK = 8
SPECTRAL_WARPS = 2
PFA_SCATTER_CONFIGS = {
    8191: ((7, 10), 117, 8, 2),
    16381: ((5, 4, 9), 91, 4, 4),
}


def _radix_axis_source(tag: str, radix: int, shape: tuple[int, ...], axis: int) -> str:
    """Split one digit axis into codelet inputs and restore its padded layout."""
    size = math.prod(shape)
    padded = shape[axis]
    columns = size // padded
    bits = padded.bit_length() - 1
    permutation = tuple(index for index in range(len(shape)) if index != axis) + (axis,)
    view = tuple(shape[index] for index in permutation)
    natural = tuple(permutation.index(index) for index in range(len(shape)))
    lines = [
        "@triton.jit",
        f"def {tag}(real, imag):",
        f"    zero = tl.full(({columns},), 0, tl.float32)",
    ]
    branch_shape = (columns,) + (2,) * bits
    for part, data in (("r","real"),("i","imag")):
        lines.extend([
            f"    view_{part} = tl.permute(tl.reshape({data}, {shape}), {permutation})",
            f"    bank_{part} = tl.reshape(view_{part}, {branch_shape})",
        ])

        def unpack(name, digit, stride, remaining):
            if digit >= radix:
                return
            if remaining == 0:
                lines.append(f"    a{part}{digit} = {name}")
                return
            low = f"low_{part}_{remaining}_{digit}"
            high = f"high_{part}_{remaining}_{digit}"
            lines.append(f"    {low}, {high} = tl.split({name})")
            unpack(low, digit, stride * 2, remaining - 1)
            unpack(high, digit + stride, stride * 2, remaining - 1)

        unpack(f"bank_{part}", 0, 1, bits)
    inputs = ", ".join(f"a{part}{j}" for part in ("r", "i") for j in range(radix))
    outputs = ", ".join(f"x{part}{j}" for part in ("r", "i") for j in range(radix))
    lines.append(f"    {outputs} = _fwd_rad{radix}_b1({inputs})")

    def nested(names):
        if len(names) == 1:
            return names[0]
        return f"tl.join({nested(names[::2])}, {nested(names[1::2])})"

    for part in ("r", "i"):
        ordered = [f"x{part}{j}" if j < radix else "zero" for j in range(padded)]
        lines += [
            f"    packed_{part} = tl.reshape({nested(ordered)}, {view})",
            f"    result_{part} = tl.reshape(tl.permute(packed_{part}, {natural}), ({size},))",
        ]
    return "\n".join([*lines, "    return result_r, result_i", ""])


def build_maca_rader16381_spectral() -> tuple[str, str]:
    """Apply the convolution spectrum between two 91-point Good–Thomas FFTs."""
    prime, n1 = 16381, 180
    q, r = 7, 13
    tile = SPECTRAL_PACK
    qpad, rpad = 1 << (q - 1).bit_length(), 1 << (r - 1).bit_length()
    size = qpad * rpad * tile
    # The time and frequency residues use complementary Good–Thomas maps.
    q_inverse = r * pow(r, -1, q)
    r_inverse = q * pow(q, -1, r)
    n2 = q * r
    name = "maca_rader_four_step_spectral_16381"
    kernel = f"""
@triton.jit
def {name}(b_fft_ptr, user_ptr, dc_ptr, zero_ptr, in_ptr, out_ptr, nbatch):
    batch = tl.program_id(1)
    flat = tl.arange(0, {size})
    q_digit = flat // {tile * rpad}
    r_digit = flat // {tile} % {rpad}
    k1 = tl.program_id(0) * {tile} + flat % {tile}
    valid = (batch < nbatch) & (q_digit < {q}) & (r_digit < {r}) & (k1 < {n1})
    time = (q_digit * {r} + r_digit * {q}) % {n2}
    freq = (q_digit * {q_inverse} + r_digit * {r_inverse}) % {n2}
    base = batch * {prime - 1}
    pair = tl.arange(0, 2)[None, :]
    source = (base + time * {n1} + k1) * 2
    xr, xi = tl.split(tl.load(in_ptr + source[:, None] + pair, valid[:, None], 0.0))
    xr, xi = _fft_r(xr, xi)
    xr, xi = _fft_q(xr, xi)
    is_dc = valid & (q_digit == 0) & (r_digit == 0) & (k1 == 0)
    dc_offset = batch * {2 * prime} + tl.zeros_like(flat)
    zero_offset = batch * 2 + tl.zeros_like(flat)
    x0r = tl.load(user_ptr + dc_offset, is_dc, 0.0)
    x0i = tl.load(user_ptr + dc_offset + 1, is_dc, 0.0)
    tl.store(zero_ptr + zero_offset, x0r, is_dc)
    tl.store(zero_ptr + zero_offset + 1, x0i, is_dc)
    tl.store(dc_ptr + dc_offset, xr + x0r, is_dc)
    tl.store(dc_ptr + dc_offset + 1, xi + x0i, is_dc)
    weight = (freq * {n1} + k1) * 2
    wr, wi = tl.split(tl.load(b_fft_ptr + weight[:, None] + pair, valid[:, None], 0.0))
    pr = xr * wr - xi * wi
    pi = -(xr * wi + xi * wr)
    pr, pi = _fft_q(pr, pi)
    pr, pi = _fft_r(pr, pi)
    target = (base + k1 * {n2} + time) * 2
    tl.store(out_ptr + target[:, None] + pair, tl.join(pr, pi), valid[:, None])
"""
    shape = (qpad, rpad, tile)
    helpers = (_radix_axis_source("_fft_r", r, shape, 1), _radix_axis_source("_fft_q", q, shape, 0))
    return name, "\n".join((*helpers, kernel))


def _crt(radices: tuple[int, ...]) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    length = math.prod(radices)
    steps = tuple(length // radix for radix in radices)
    multipliers = tuple(step * pow(step, -1, radix) for step, radix in zip(steps, radices))
    return length, steps, multipliers


def build_maca_rader_pfa_scatter(prime: int, *, narrow_indices: bool = False) -> tuple[str, str]:
    """Map the final convolution FFT's residue digits directly to public output."""
    radices, n2, tile, _ = PFA_SCATTER_CONFIGS[prime]
    n1, steps, multipliers = _crt(radices)
    shape = tuple(1 << (radix - 1).bit_length() for radix in radices) + (tile,)
    size = math.prod(shape)
    index_ptr = "idx16_ptr" if narrow_indices else "idx_ptr"
    width_tag = "_i16" if narrow_indices else ""
    label = f"maca_rader_four_step_scatter_pfa{width_tag}_{prime}"
    lines = [
        _radix_axis_source(f"_boundary_axis{axis}", radix, shape, axis)
        for axis, radix in enumerate(radices)
    ]
    lines += ["@triton.jit", f"def {label}({index_ptr}, zero_ptr, in_ptr, out_ptr, nbatch):",
              "    batch = tl.program_id(1)", f"    flat = tl.arange(0, {size})",
              f"    col = tl.program_id(0) * {tile} + flat % {tile}"]
    digits = []
    valid = ["(batch < nbatch)", f"(col < {n2})"]
    for axis, radix in enumerate(radices):
        digit = f"digit{axis}"
        digits.append(digit)
        lines.append(f"    {digit} = flat // {math.prod(shape[axis + 1:])} % {shape[axis]}")
        valid.append(f"({digit} < {radix})")
    time = " + ".join(f"{digit} * {step}" for digit, step in zip(digits, steps))
    frequency = " + ".join(f"{digit} * {step}" for digit, step in zip(digits, multipliers))
    lines += [f"    mask = {' & '.join(valid)}",
              f"    input_row = ({time}) % {n1}",
              f"    output_row = ({frequency}) % {n1}",
              f"    scratch_base = batch * {prime - 1}",
              "    pair = tl.arange(0, 2)[None, :]"]
    lines += [f"    source = (scratch_base + input_row * {n2} + col) * 2",
              f"    index = tl.load({index_ptr} + output_row * {n2} + col, mask, 0)",
              f"    destination = (batch * {prime} + index) * 2"]
    lines.append("    real, imag = tl.split(tl.load(in_ptr + source[:, None] + pair, mask[:, None], 0.0))")
    for axis in range(len(radices)):
        lines.append(f"    real, imag = _boundary_axis{axis}(real, imag)")
    lines += ["    x0r = tl.load(zero_ptr + batch * 2, batch < nbatch, 0.0)",
              "    x0i = tl.load(zero_ptr + batch * 2 + 1, batch < nbatch, 0.0)",
              f"    real, imag = x0r + real / {prime - 1}, x0i - imag / {prime - 1}"]
    lines.append("    tl.store(out_ptr + destination[:, None] + pair, tl.join(real, imag), mask[:, None])")
    return label, "\n".join(lines) + "\n"
