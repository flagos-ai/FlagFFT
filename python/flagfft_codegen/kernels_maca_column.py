# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Split C550 FourStep columns for the complex and packed-real 663000 plans."""

from textwrap import dedent

from .ix_prime_codelets import emit_prime_codelet


def build_maca_split_column1768(
    direction: str, prepare: bool, warps: int, *, pfa: bool = False, fp64: bool = False,
) -> tuple[str, str, int]:
    phase = "prepare" if prepare else "finish"
    suffix = "_w4" if warps == 4 else ""
    precision = "_fp64" if fp64 else ""
    name = f"maca_split_column1768_{phase}{suffix}{'_pfa' if pfa else ''}{precision}_{direction}"
    tile = 128 if prepare else (16 if fp64 else 32)
    columns, radix13, radix17, radix8 = 375, 13, 17, 8
    inner = radix17 * radix8
    total = columns * radix13 * inner
    work = columns * (inner if prepare else radix13)
    helpers = ""
    args = "in_ptr, twiddle_ptr, out_ptr" if prepare else "in_ptr, out_ptr"
    if not pfa:
        args += ", phase_r_ptr, phase_i_ptr"
    args += ", nbatch"
    if not prepare:
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
            if pfa:
                row = f"(b * {radix13} + {inner * j}) % {radix13 * inner}"
                index = f"({row}) * {columns} + col"
            else:
                index = f"lane + {inner * columns * j}"
            lines += [
                f"    src = ({index})[:, None] * 2 + pair",
                "    r, i = tl.split(tl.load(in_ptr + base + src, valid[:, None], 0.0))",
                "    co, si = tl.split(tl.load(twiddle_ptr + src, valid[:, None], 0.0))",
                f"    r{j}, i{j} = _cmul(r, i, co, si)",
            ]
        lines += emit_prime_codelet("    ", radix13, direction)
        for k in range(radix13):
            if pfa:
                lines.append(f"    r, i = r{k}, i{k}")
            else:
                lines += [
                    f"    co = tl.load(phase_r_ptr + b * {k}, valid, 0.0)",
                    f"    si = tl.load(phase_i_ptr + b * {k}, valid, 0.0)",
                    f"    r, i = _cmul(r{k}, i{k}, co, si)",
                ]
            lines += [
                f"    dst = (b * {radix13 * columns} + col + {k * columns}) * 2",
                "    tl.store(out_ptr + base + dst[:, None] + pair, tl.join(r, i), valid[:, None])",
            ]
    else:
        # Keep the 17-point and eight-point butterflies in the same CTA.
        lines += [
            f"    b = tl.arange(0, {radix8})",
            "    pair = tl.arange(0, 2)[None, None, :]",
        ]
        if pfa:
            # Restore natural frequencies via the CRT for 13, 17 and 8.
            axis = radix13 * inner
            weights = tuple((axis // r) * pow(axis // r, -1, r) for r in (radix13, radix17, radix8))
            lines += [
                f"    column = lane % {columns}",
                f"    k13 = lane // {columns}",
                f"    output_base = k13[None, :] * {weights[0]} + b[:, None] * {weights[2]}",
            ]
        else:
            lines.append(f"    offset = b[:, None] * {work} + lane[None, :]")
        for j in range(radix17):
            if pfa:
                row = f"(b * {radix17} + {radix8 * j}) % {inner}"
                index = f"({row})[:, None] * {work} + lane[None, :]"
            else:
                index = f"offset + {j * radix8 * work}"
            lines += [
                f"    src = ({index}) * 2",
                f"    r{j}, i{j} = tl.split(tl.load(in_ptr + base + src[:, :, None] + pair, "
                "valid[None, :, None], 0.0))",
            ]
        lines += emit_prime_codelet("    ", radix17, direction)
        for k in range(radix17):
            if pfa:
                lines += [
                    f"    r, i = _last8(r{k}, i{k})",
                    f"    freq = (output_base + {k * weights[1]}) % {axis}",
                    f"    dst = (freq * {columns} + column[None, :]) * 2",
                ]
            else:
                lines += [
                    f"    co = tl.load(phase_r_ptr + b * {k})[:, None]",
                    f"    si = tl.load(phase_i_ptr + b * {k})[:, None]",
                    f"    r, i = _cmul(r{k}, i{k}, co, si)",
                    "    r, i = _last8(r, i)",
                    f"    dst = (b[:, None] * {radix17 * work} + lane[None, :] + {k * work}) * 2",
                ]
            lines.append(
                "    tl.store(out_ptr + base + dst[:, :, None] + pair, tl.join(r, i), valid[None, :, None])",
            )
    return name, helpers + "\n".join(lines) + "\n", (work + tile - 1) // tile


def build_maca_split_column1105(
    direction: str, prepare: bool, finish_tile: int, *, fp64: bool,
) -> tuple[str, str, int]:
    """Use Good–Thomas for 1105 = 13 × 17 × 5 and return natural frequencies."""
    phase = "prepare" if prepare else "finish"
    precision = "_fp64" if fp64 else ""
    name = f"maca_split_column1105_{phase}_t{finish_tile}{precision}_{direction}"
    tile = 4 * finish_tile if prepare else finish_tile
    columns, radix13, radix17, radix5 = 300, 13, 17, 5
    inner = radix17 * radix5
    length = radix13 * inner
    work = columns * (inner if prepare else radix13)
    args = "in_ptr, twiddle_ptr, out_ptr, nbatch" if prepare else "in_ptr, out_ptr, nbatch"
    body = [
        "@triton.jit",
        f"def {name}({args}):",
        "    batch = tl.program_id(1).to(tl.int64)",
        "    if batch >= nbatch:",
        "        return",
        f"    lane = tl.program_id(0) * {tile} + tl.arange(0, {tile})",
        f"    valid = lane < {work}",
        f"    column = lane % {columns}",
        f"    base = batch * {2 * length * columns}",
    ]
    if prepare:
        body += [f"    b = lane // {columns}", "    pair = tl.arange(0, 2)[None, :]"]
        for j in range(radix13):
            body += [
                f"    row = (b * {radix13} + {inner * j}) % {length}",
                f"    src = (row * {columns} + column)[:, None] * 2 + pair",
                "    r, i = tl.split(tl.load(in_ptr + base + src, valid[:, None], 0.0))",
                "    co, si = tl.split(tl.load(twiddle_ptr + src, valid[:, None], 0.0))",
                f"    r{j}, i{j} = _cmul(r, i, co, si)",
            ]
        body += emit_prime_codelet("    ", radix13, direction)
        for k in range(radix13):
            body += [
                f"    dst = (b * {radix13 * columns} + column + {k * columns})[:, None] * 2 + pair",
                f"    tl.store(out_ptr + base + dst, tl.join(r{k}, i{k}), valid[:, None])",
            ]
        return name, "\n".join(body) + "\n", (work + tile - 1) // tile

    conjugate = -1 if direction == "inverse" else 1
    last_radix = dedent(f"""
        @triton.jit
        def _split5(value):
            e, o = tl.split(tl.reshape(tl.permute(value, (1, 0)), ({tile}, 2, 2, 2)))
            e0, e1 = tl.split(e)
            o0, o1 = tl.split(o)
            x0, x4 = tl.split(e0)
            x2, _ = tl.split(e1)
            x1, _ = tl.split(o0)
            x3, _ = tl.split(o1)
            return x0, x1, x2, x3, x4

        @triton.jit
        def _join5(a, b, c, d, e):
            even = tl.join(tl.join(a, e), tl.join(c, a))
            odd = tl.join(tl.join(b, a), tl.join(d, a))
            return tl.permute(tl.reshape(tl.join(even, odd), ({tile}, 8)), (1, 0))

        @triton.jit
        def _last5(real, imag):
            r0, r1, r2, r3, r4 = _split5(real)
            i0, i1, i2, i3, i4 = _split5(imag)
            a, b, c, d, e, u, v, w, x, y = _fwd_rad5_b1(
                r0, r1, r2, r3, r4, i0 * {conjugate}, i1 * {conjugate},
                i2 * {conjugate}, i3 * {conjugate}, i4 * {conjugate})
            return _join5(a, b, c, d, e), _join5(u, v, w, x, y) * {conjugate}
    """)
    weight13, weight17, weight5 = (
        (length // r) * pow(length // r, -1, r) for r in (radix13, radix17, radix5)
    )
    body += [
        "    b = tl.arange(0, 8)",
        f"    valid_rows = (b[:, None] < {radix5}) & valid[None, :]",
        "    pair = tl.arange(0, 2)[None, None, :]",
        f"    k13 = lane // {columns}",
        f"    output_base = k13[None, :] * {weight13} + b[:, None] * {weight5}",
    ]
    for j in range(radix17):
        body += [
            f"    row = (b * {radix17} + {j * radix5}) % {inner}",
            f"    src = (row[:, None] * {work} + lane[None, :])[:, :, None] * 2 + pair",
            f"    r{j}, i{j} = tl.split(tl.load(in_ptr + base + src, valid_rows[:, :, None], 0.0))",
        ]
    body += emit_prime_codelet("    ", radix17, direction)
    for k in range(radix17):
        body += [
            f"    r, i = _last5(r{k}, i{k})",
            f"    freq = (output_base + {k * weight17}) % {length}",
            f"    dst = (freq * {columns} + column[None, :])[:, :, None] * 2 + pair",
            "    tl.store(out_ptr + base + dst, tl.join(r, i), valid_rows[:, :, None])",
        ]
    return name, last_radix + "\n".join(body) + "\n", (work + tile - 1) // tile
