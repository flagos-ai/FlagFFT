# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""C550 Good–Thomas FFT kernels."""

from textwrap import dedent


def build_maca_fp32_fft210_pfa(direction: str) -> tuple[str, str]:
    """Use Good–Thomas with factors 5, 6 and 7 for two C550 FFTs per CTA."""
    factors = (5, 6, 7)
    pack = 2
    name = f"maca_fp32_fft210_pfa_{direction}"
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
    start = dedent(f"""
        @triton.jit
        def {name}(in_ptr, out_ptr, nbatch):
            row = tl.program_id(0) * {pack} + tl.arange(0, {pack})
            a = tl.arange(0, 8)[None, :, None, None]
            b = tl.arange(0, 8)[None, None, :, None]
            c = tl.arange(0, 8)[None, None, None, :]
            pair = tl.arange(0, 2)[None, None, None, None, :]
            physical = (a * {strides[0]} + b * {strides[1]} + c * {strides[2]}) % 210
            valid = ((row[:, None, None, None] < nbatch) &
                     (a < {factors[0]}) & (b < {factors[1]}) & (c < {factors[2]}))
            base = row[:, None, None, None] * 420
            src = (base + physical * 2)[:, :, :, :, None] + pair
            r, i = tl.split(tl.load(in_ptr + src, valid[:, :, :, :, None], 0.0))
            r, i = _pfa_radix{factors[2]}(r, i)
            r, i = _pfa_radix{factors[1]}(tl.permute(r, (0, 1, 3, 2)), tl.permute(i, (0, 1, 3, 2)))
            r, i = _pfa_radix{factors[0]}(tl.permute(r, (0, 3, 2, 1)), tl.permute(i, (0, 3, 2, 1)))
            r, i = tl.permute(r, (0, 3, 1, 2)), tl.permute(i, (0, 3, 1, 2))
            natural = (a * {crt[0]} + b * {crt[1]} + c * {crt[2]}) % 210
            dst = (base + natural * 2)[:, :, :, :, None] + pair
            tl.store(out_ptr + dst, tl.join(r, i), valid[:, :, :, :, None])
    """)
    return name, helpers + start

