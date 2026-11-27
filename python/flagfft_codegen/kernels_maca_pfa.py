# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""C550 Good–Thomas FFT kernels."""

from textwrap import dedent, indent


def build_maca_fp32_fft210_pfa(direction: str, *, real_inverse: bool = False) -> tuple[str, str]:
    """Use Good–Thomas with factors 5, 6 and 7 for two C550 FFTs per CTA."""
    if real_inverse and direction != "inverse":
        raise ValueError("the compact 210-point PFA requires an inverse transform")
    factors = (7, 5, 6) if real_inverse else (5, 6, 7)
    pack = 2
    kind = "c2r" if real_inverse else "fft"
    name = f"maca_fp32_{kind}210_pfa_{direction}"
    conjugate = -1 if direction == "inverse" else 1
    helpers = dedent(f"""
        @triton.jit
        def _pfa_split8(values):
            lo, hi = tl.split(tl.reshape(values, ({pack * 64}, 2, 2, 2)))
            e0, e1 = tl.split(lo)
            o0, o1 = tl.split(hi)
            a, e = tl.split(e0)
            c, g = tl.split(e1)
            b, f = tl.split(o0)
            d, h = tl.split(o1)
            return a, b, c, d, e, f, g, h

        @triton.jit
        def _pfa_join8(a, b, c, d, e, f, g, h):
            values = tl.join(tl.join(tl.join(a, e), tl.join(c, g)),
                             tl.join(tl.join(b, f), tl.join(d, h)))
            return tl.reshape(values, ({pack}, 8, 8, 8))
    """)
    for radix in factors:
        vars_r = ", ".join(f"r{k}" for k in range(8))
        vars_i = ", ".join(f"i{k}" for k in range(8))
        inputs = ", ".join([*(f"r{k}" for k in range(radix)), *(f"i{k} * ({conjugate})" for k in range(radix))])
        outputs = ", ".join([*(f"r{k}" for k in range(radix)), *(f"i{k}" for k in range(radix))])
        helpers += dedent(f"""
            @triton.jit
            def _pfa_radix{radix}(r, i):
                {vars_r} = _pfa_split8(r)
                {vars_i} = _pfa_split8(i)
                {outputs} = _fwd_rad{radix}_b1({inputs})
                return _pfa_join8({vars_r}), _pfa_join8({vars_i}) * ({conjugate})
        """)
    # Good–Thomas input strides and the CRT weights for natural frequency order.
    strides = [210 // radix for radix in factors]
    crt = [step * pow(step, -1, factor) for step, factor in zip(strides, factors, strict=True)]
    if real_inverse:
        distances = "input_distance, output_distance, "
        boundary_load = dedent("""\
            compact = tl.minimum(physical, 210 - physical)
            src = ((row[:, None, None, None] * input_distance + compact) * 2)[:, :, :, :, None] + pair
            r, i = tl.split(tl.load(in_ptr + src, valid[:, :, :, :, None], 0.0))
            i = tl.where(physical > 105, -i, i)
            i = tl.where((physical == 0) | (physical == 105), 0.0, i)""")
        boundary_store = dedent("""\
            dst = row[:, None, None, None] * output_distance + natural
            tl.store(out_ptr + dst, r, valid)""")
    else:
        distances = ""
        boundary_load = dedent("""\
            base = row[:, None, None, None] * 420
            src = (base + physical * 2)[:, :, :, :, None] + pair
            r, i = tl.split(tl.load(in_ptr + src, valid[:, :, :, :, None], 0.0))""")
        boundary_store = dedent("""\
            dst = (base + natural * 2)[:, :, :, :, None] + pair
            tl.store(out_ptr + dst, tl.join(r, i), valid[:, :, :, :, None])""")
    start = dedent(f"""
        @triton.jit
        def {name}(in_ptr, out_ptr, {distances}nbatch):
            row = tl.program_id(0) * {pack} + tl.arange(0, {pack})
            a = tl.arange(0, 8)[None, :, None, None]
            b = tl.arange(0, 8)[None, None, :, None]
            c = tl.arange(0, 8)[None, None, None, :]
            pair = tl.arange(0, 2)[None, None, None, None, :]
            physical = (a * {strides[0]} + b * {strides[1]} + c * {strides[2]}) % 210
            valid = ((row[:, None, None, None] < nbatch) &
                     (a < {factors[0]}) & (b < {factors[1]}) & (c < {factors[2]}))
{indent(boundary_load, "            ")}
            r, i = _pfa_radix{factors[2]}(r, i)
            r, i = _pfa_radix{factors[1]}(tl.permute(r, (0, 1, 3, 2)), tl.permute(i, (0, 1, 3, 2)))
            r, i = _pfa_radix{factors[0]}(tl.permute(r, (0, 3, 2, 1)), tl.permute(i, (0, 3, 2, 1)))
            r, i = tl.permute(r, (0, 3, 1, 2)), tl.permute(i, (0, 3, 1, 2))
            natural = (a * {crt[0]} + b * {crt[1]} + c * {crt[2]}) % 210
{indent(boundary_store, "            ")}
    """)
    return name, helpers + start
