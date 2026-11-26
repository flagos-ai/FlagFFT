# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Split the C550 FourStep column for 663000 = 375 × 1768."""

from textwrap import dedent

from .ix_prime_codelets import emit_prime_codelet


def build_maca_split_column1768(direction: str, prepare: bool, warps: int) -> tuple[str, str, int]:
    phase = "prepare" if prepare else "finish"
    suffix = "_w4" if warps == 4 else ""
    name = f"maca_split_column1768_{phase}{suffix}_{direction}"
    tile = 128 if prepare else 32
    columns, radix13, radix17, radix8 = 375, 13, 17, 8
    inner = radix17 * radix8
    total = columns * radix13 * inner
    work = columns * (inner if prepare else radix13)
    helpers = ""
    if prepare:
        args = "in_ptr, twiddle_ptr, out_ptr, phase_r_ptr, phase_i_ptr, nbatch"
    else:
        args = "in_ptr, out_ptr, phase_r_ptr, phase_i_ptr, nbatch"
        sign = 1 if direction == "inverse" else -1
        helpers = dedent(f"""
            @triton.jit
            def _split8(value):
                even, odd = tl.split(tl.reshape(tl.permute(value, (1, 0)), ({tile}, 2, 2, 2)))
                e0, e1 = tl.split(even)
                o0, o1 = tl.split(odd)
                x0, x4 = tl.split(e0)
                x2, x6 = tl.split(e1)
                x1, x5 = tl.split(o0)
                x3, x7 = tl.split(o1)
                return x0, x1, x2, x3, x4, x5, x6, x7

            @triton.jit
            def _join8(a, b, c, d, e, f, g, h):
                even = tl.join(tl.join(a, e), tl.join(c, g))
                odd = tl.join(tl.join(b, f), tl.join(d, h))
                return tl.permute(tl.reshape(tl.join(even, odd), ({tile}, 8)), (1, 0))

            @triton.jit
            def _last8(real, imag):
                r0, r1, r2, r3, r4, r5, r6, r7 = _split8(real)
                i0, i1, i2, i3, i4, i5, i6, i7 = _split8(imag)
                a, b, c, d, e, f, g, h, u, v, w, x, y, z, s, t = _fwd_rad8_b1(
                    r0, r1, r2, r3, r4, r5, r6, r7,
                    i0 * {-sign}, i1 * {-sign}, i2 * {-sign}, i3 * {-sign},
                    i4 * {-sign}, i5 * {-sign}, i6 * {-sign}, i7 * {-sign},
                )
                return _join8(a, b, c, d, e, f, g, h), _join8(u, v, w, x, y, z, s, t) * {-sign}
        """)
    lines = [
        "@triton.jit",
        f"def {name}({args}):",
        "    batch = tl.program_id(1).to(tl.int64)",
        "    if batch >= nbatch:",
        "        return",
        f"    lane = tl.program_id(0) * {tile} + tl.arange(0, {tile})",
        f"    valid = lane < {work}",
        f"    base = batch * {2 * total}",
    ]
    if prepare:
        # Consume the ordinary FourStep row layout and apply its cross-axis phase.
        lines += [
            f"    b, col = lane // {columns}, lane % {columns}",
            "    pair = tl.arange(0, 2)[None, :]",
        ]
        for j in range(radix13):
            lines += [
                f"    src = (lane + {inner * columns * j})[:, None] * 2 + pair",
                "    r, i = tl.split(tl.load(in_ptr + base + src, valid[:, None], 0.0))",
                "    co, si = tl.split(tl.load(twiddle_ptr + src, valid[:, None], 0.0))",
                f"    r{j}, i{j} = _cmul(r, i, co, si)",
            ]
        lines += emit_prime_codelet("    ", radix13, direction)
        for k in range(radix13):
            lines += [
                f"    co = tl.load(phase_r_ptr + b * {k}, valid, 0.0)",
                f"    si = tl.load(phase_i_ptr + b * {k}, valid, 0.0)",
                f"    r, i = _cmul(r{k}, i{k}, co, si)",
                f"    dst = (b * {radix13 * columns} + col + {k * columns}) * 2",
                "    tl.store(out_ptr + base + dst[:, None] + pair, tl.join(r, i), valid[:, None])",
            ]
    else:
        # Keep the 17-point and eight-point butterflies in the same CTA.
        lines += [
            f"    b = tl.arange(0, {radix8})",
            f"    offset = b[:, None] * {work} + lane[None, :]",
            "    pair = tl.arange(0, 2)[None, None, :]",
        ]
        for j in range(radix17):
            lines += [
                f"    src = (offset + {j * radix8 * work}) * 2",
                f"    r{j}, i{j} = tl.split(tl.load(in_ptr + base + src[:, :, None] + pair, "
                "valid[None, :, None], 0.0))",
            ]
        lines += emit_prime_codelet("    ", radix17, direction)
        for k in range(radix17):
            lines += [
                f"    co = tl.load(phase_r_ptr + b * {k})[:, None]",
                f"    si = tl.load(phase_i_ptr + b * {k})[:, None]",
                f"    r, i = _cmul(r{k}, i{k}, co, si)",
                "    r, i = _last8(r, i)",
                f"    dst = (b[:, None] * {radix17 * work} + lane[None, :] + {k * work}) * 2",
                "    tl.store(out_ptr + base + dst[:, :, None] + pair, tl.join(r, i), valid[None, :, None])",
            ]
    return name, helpers + "\n".join(lines) + "\n", (work + tile - 1) // tile
