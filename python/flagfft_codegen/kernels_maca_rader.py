# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""The fused frequency-domain stage of the C550 16381-point Rader FFT."""

SPECTRAL_PACK = 8
SPECTRAL_WARPS = 2


def _radix_axis_source(axis: str, radix: int, qpad: int, rpad: int, tile: int) -> str:
    """Split one digit axis into codelet inputs and restore its padded layout."""
    columns = rpad * tile if axis == "q" else qpad * tile
    padded = qpad if axis == "q" else rpad
    bits = padded.bit_length() - 1
    permutation = (1, 2, 0) if axis == "q" else (0, 2, 1)
    lines = [
        "@triton.jit",
        f"def _fft_{axis}(real, imag):",
        f"    zero = tl.full(({columns},), 0, tl.float32)",
    ]
    branch_shape = (columns,) + (2,) * bits
    for part, data in (("r","real"),("i","imag")):
        lines.extend([
            f"    view_{part} = tl.permute(tl.reshape({data}, ({qpad}, {rpad}, {tile})), {permutation})",
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

    if axis == "q":
        view = (rpad, tile, qpad)
        permutation = (2, 0, 1)
    else:
        view = (qpad, tile, rpad)
        permutation = (0, 2, 1)
    size = qpad * rpad * tile
    for part in ("r", "i"):
        ordered = [f"x{part}{j}" if j < radix else "zero" for j in range(padded)]
        lines += [
            f"    packed_{part} = tl.reshape({nested(ordered)}, {view})",
            f"    result_{part} = tl.reshape(tl.permute(packed_{part}, {permutation}), ({size},))",
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
    helpers = (_radix_axis_source("r", r, qpad, rpad, tile), _radix_axis_source("q", q, qpad, rpad, tile))
    return name, "\n".join((*helpers, kernel))
