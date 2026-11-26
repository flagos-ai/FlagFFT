# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Register-tiled C550 FP64 FFT with radix-4 DIF stages."""

from textwrap import dedent


def _dif2048_helpers(direction: str | None, *, length: int = 2048, skip_unit_rotation: bool = False) -> str:
    sign = "SIGN" if direction is None else (1 if direction == "inverse" else -1)
    sign_arg = ", SIGN: tl.constexpr" if direction is None else ""
    unit_rotation = "            if indices.shape[0] == 1:\n                return real, imag\n" if skip_unit_rotation else ""
    return dedent(
        f"""
        @triton.jit
        def _dif_split_quarters(values, GROUPS: tl.constexpr, QUARTER: tl.constexpr):
            groups = tl.reshape(values, (GROUPS, 4, QUARTER))
            digits = tl.reshape(tl.permute(groups, (0, 2, 1)), (GROUPS, QUARTER, 2, 2))
            ac, bd = tl.split(digits)
            a, c = tl.split(ac)
            b, d = tl.split(bd)
            return a, b, c, d

        @triton.jit
        def _dif_join_quarters(a, b, c, d, GROUPS: tl.constexpr, QUARTER: tl.constexpr):
            digits = tl.join(tl.join(a, c), tl.join(b, d))
            tile = tl.reshape(digits, (GROUPS, QUARTER, 4))
            return tl.reshape(tl.permute(tile, (0, 2, 1)), ({length},))

        @triton.jit
        def _dif_rotate(real, imag, roots_r, roots_i, indices):
{unit_rotation}            co = tl.load(roots_r + indices)[None, :]
            si = tl.load(roots_i + indices)[None, :]
            return real * co - imag * si, real * si + imag * co

        @triton.jit
        def _dif_radix_four(real, imag, roots_r, roots_i, STAGE: tl.constexpr{sign_arg}):
            GROUPS: tl.constexpr = 1 << (2 * STAGE)
            QUARTER: tl.constexpr = {length} // (4 * GROUPS)
            ar, br, cr, dr = _dif_split_quarters(real, GROUPS, QUARTER)
            ai, bi, ci, di = _dif_split_quarters(imag, GROUPS, QUARTER)
            even_r, even_i = ar + cr, ai + ci
            odd_r, odd_i = br + dr, bi + di
            diff_even_r, diff_even_i = ar - cr, ai - ci
            diff_odd_r, diff_odd_i = br - dr, bi - di
            roots = tl.arange(0, QUARTER) * GROUPS
            out0_r, out0_i = even_r + odd_r, even_i + odd_i
            out1_r, out1_i = _dif_rotate(
                diff_even_r - ({sign}) * diff_odd_i,
                diff_even_i + ({sign}) * diff_odd_r, roots_r, roots_i, roots,
            )
            out2_r, out2_i = _dif_rotate(
                even_r - odd_r, even_i - odd_i, roots_r, roots_i, roots * 2,
            )
            out3_r, out3_i = _dif_rotate(
                diff_even_r + ({sign}) * diff_odd_i,
                diff_even_i - ({sign}) * diff_odd_r, roots_r, roots_i, roots * 3,
            )
            return (
                _dif_join_quarters(out0_r, out1_r, out2_r, out3_r, GROUPS, QUARTER),
                _dif_join_quarters(out0_i, out1_i, out2_i, out3_i, GROUPS, QUARTER),
            )

        """
    ) + dedent(
        """
        @triton.jit
        def _dif_natural_output(value):
            result = tl.reshape(value, (4, 4, 4, 4, 4))
            result = tl.permute(result, (4, 3, 2, 1, 0))
            return tl.reshape(result, (1024,))
        """ if length == 1024 else """
        @triton.jit
        def _dif_natural_output(value):
            low, high = tl.split(tl.reshape(value, (1024, 2)))
            result = tl.join(low + high, low - high)
            result = tl.reshape(result, (4, 4, 4, 4, 4, 2))
            result = tl.permute(result, (5, 4, 3, 2, 1, 0))
            return tl.reshape(result, (2048,))
        """
    )


def _dif_terminal8_helpers(direction: str) -> str:
    conjugate = 1 if direction == "forward" else -1
    return dedent(f"""
        @triton.jit
        def _split_terminal8(values):
            even, odd = tl.split(tl.reshape(values, (256, 2, 2, 2)))
            e0, e1 = tl.split(even)
            o0, o1 = tl.split(odd)
            a, e = tl.split(e0)
            c, g = tl.split(e1)
            b, f = tl.split(o0)
            d, h = tl.split(o1)
            return a, b, c, d, e, f, g, h

        @triton.jit
        def _join_terminal8(a, b, c, d, e, f, g, h):
            even = tl.join(tl.join(a, e), tl.join(c, g))
            odd = tl.join(tl.join(b, f), tl.join(d, h))
            values = tl.reshape(tl.join(even, odd), (4, 4, 4, 4, 8))
            return tl.reshape(tl.permute(values, (4, 3, 2, 1, 0)), (2048,))

        @triton.jit
        def _dif_terminal8(real, imag):
            r0, r1, r2, r3, r4, r5, r6, r7 = _split_terminal8(real)
            i0, i1, i2, i3, i4, i5, i6, i7 = _split_terminal8(({conjugate}) * imag)
            a, b, c, d, e, f, g, h, j, k, l, m, n, p, q, s = _fwd_rad8_b1(
                r0, r1, r2, r3, r4, r5, r6, r7, i0, i1, i2, i3, i4, i5, i6, i7)
            return (_join_terminal8(a, b, c, d, e, f, g, h),
                    ({conjugate}) * _join_terminal8(j, k, l, m, n, p, q, s))
    """)


def build_maca_fp64_dif_leaf(
    direction: str, *, four_warps: bool = False, terminal8: bool = False,
) -> tuple[str, str]:
    suffix = "_terminal8" if terminal8 else ("_w4" if four_warps else "")
    name = f"maca_fp64_dif2048{suffix}_{direction}"
    helpers = _dif2048_helpers(direction)
    if terminal8:
        helpers += _dif_terminal8_helpers(direction)
        stage_count = 4
        finish = "real, imag = _dif_terminal8(real, imag)\n            value = tl.join(real, imag)"
    else:
        stage_count = 5
        finish = "value = tl.join(_dif_natural_output(real), _dif_natural_output(imag))"
    source = helpers + dedent(
        f"""
        @triton.jit
        def {name}(in_ptr, out_ptr, roots_r_ptr, roots_i_ptr, nbatch):
            batch = tl.program_id(0)
            if batch >= nbatch:
                return
            base = batch.to(tl.int64) * 4096
            k = tl.arange(0, 2048)
            pair = tl.arange(0, 2)[None, :]
            offset = k[:, None] * 2 + pair
            real, imag = tl.split(tl.load((in_ptr + base) + offset))
            for stage in tl.static_range({stage_count}):
                real, imag = _dif_radix_four(real, imag, roots_r_ptr, roots_i_ptr, stage)
            {finish}
            tl.store((out_ptr + base) + offset, value)
        """
    )
    return name, source


def build_maca_fp64_packed_dif(direction: str, *, gather_reflection: bool = False) -> tuple[str, str]:
    inverse = direction == "inverse"
    if gather_reflection and direction != "forward":
        raise ValueError("gather reflection requires the forward packed DIF kernel")
    kind = "c2r" if inverse else "r2c"
    name = f"maca_fp64_packed_dif1024_{kind}"
    reflection = "br, bi = _reflect(real), _reflect(imag)"
    if gather_reflection:
        name += "_gather"
        reflection = "q = (-k) & 1023\n                br, bi = tl.gather(real, q, 0), tl.gather(imag, q, 0)"
    source = _dif2048_helpers(direction, length=1024, skip_unit_rotation=gather_reflection) + dedent(f"""
        @triton.jit
        def _reflect_bit(values, BIT: tl.constexpr):
            LOWER: tl.constexpr = 1 << BIT
            GROUP: tl.constexpr = 1024 // (LOWER * 2)
            digits = tl.reshape(values, (GROUP, 2, LOWER))
            a, b = tl.split(tl.permute(digits, (0, 2, 1)))
            swap = tl.arange(0, LOWER)[None, :] != 0
            result = tl.join(tl.where(swap, b, a), tl.where(swap, a, b))
            return tl.reshape(tl.permute(result, (0, 2, 1)), (1024,))

        @triton.jit
        def _reflect(values):
            for bit in tl.static_range(1, 10):
                values = _reflect_bit(values, bit)
            return values

        @triton.jit
        def {name}(in_ptr, out_ptr, roots_r_ptr, roots_i_ptr, phase_r_ptr, phase_i_ptr,
                   input_distance, output_distance, nbatch):
            row = tl.program_id(0).to(tl.int64)
            if row >= nbatch:
                return
            k = tl.arange(0, 1024)
            pair = tl.arange(0, 2)[None, :]
            index = k[:, None] * 2 + pair
            co, si = tl.load(phase_r_ptr + k), tl.load(phase_i_ptr + k)
            if {inverse}:
                src = in_ptr + row * input_distance * 2
                ar, ai = tl.split(tl.load(src + index))
                br, bi = tl.split(tl.load(src + (1024 - k[:, None]) * 2 + pair))
                ai = tl.where(k == 0, 0.0, ai)
                bi = tl.where(k == 0, 0.0, bi)
                dr, di = ar - br, ai + bi
                real = ar + br - (dr * si + di * co)
                imag = ai - bi + (dr * co - di * si)
            else:
                real, imag = tl.split(tl.load(in_ptr + row * input_distance + index))
            for stage in tl.static_range(5):
                real, imag = _dif_radix_four(real, imag, roots_r_ptr, roots_i_ptr, stage)
            real, imag = _dif_natural_output(real), _dif_natural_output(imag)
            if {inverse}:
                tl.store(out_ptr + row * output_distance + index, tl.join(real, imag))
            else:
                {reflection}
                dr, di = real - br, imag + bi
                out_r = (real + br + dr * si + di * co) * 0.5
                out_i = (imag - bi - dr * co + di * si) * 0.5
                dst = out_ptr + row * output_distance * 2
                tl.store(dst + index, tl.join(out_r, out_i))
                tl.store(dst + 2048 + k, real - imag, k == 0)
                tl.store(dst + 2049 + k, 0.0, k == 0)
    """)
    return name, source


def build_maca_prime_dif(n: int, mode: str) -> tuple[str, str]:
    name = f"maca_prime_dif_{mode}_n{n}"
    source = _dif2048_helpers(None) + dedent(
        f"""
        @triton.jit
        def _prime_input(in_ptr, chirp_ptr, row, k, N: tl.constexpr, MODE: tl.constexpr):
            HALF: tl.constexpr = N // 2 + 1
            pair = tl.arange(0, 2)[None, :]
            if MODE == "r2c":
                r = tl.load(in_ptr + row * N + k, k < N, 0.0)
                i = tl.zeros_like(r)
            elif MODE == "c2r":
                mirror = tl.where(k < HALF, k, N - k)
                offset = (row * HALF + mirror[:, None]) * 2 + pair
                r, i = tl.split(tl.load(in_ptr + offset, k[:, None] < N, 0.0))
                i = tl.where(k < HALF, i, -i)
            else:
                offset = (row * N + k[:, None]) * 2 + pair
                r, i = tl.split(tl.load(in_ptr + offset, k[:, None] < N, 0.0))
            cr, ci = tl.split(tl.load(chirp_ptr + k[:, None] * 2 + pair, k[:, None] < N, 0.0))
            return r * cr - i * ci, i * cr + r * ci

        @triton.jit
        def _prime_output(out_ptr, chirp_ptr, row, k, r, i, N: tl.constexpr, MODE: tl.constexpr):
            pair = tl.arange(0, 2)[None, :]
            cr, ci = tl.split(tl.load(chirp_ptr + k[:, None] * 2 + pair, k[:, None] < N, 0.0))
            re = r * cr - i * ci
            if MODE == "c2r":
                tl.store(out_ptr + row * N + k, re, k < N)
            else:
                ODIST: tl.constexpr = N // 2 + 1 if MODE == "r2c" else N
                im = i * cr + r * ci
                offset = (row * ODIST + k[:, None]) * 2 + pair
                tl.store(out_ptr + offset, tl.join(re, im), k[:, None] < ODIST)

        @triton.jit
        def {name}(in_ptr, b_fft_ptr, chirp_ptr, out_ptr,
                   roots_r_ptr, roots_fwd_ptr, roots_inv_ptr, nbatch):
            row = tl.program_id(0)
            if row >= nbatch:
                return
            k = tl.arange(0, 2048)
            r, i = _prime_input(in_ptr, chirp_ptr, row, k, {n}, "{mode}")
            for stage in tl.static_range(5):
                r, i = _dif_radix_four(r, i, roots_r_ptr, roots_fwd_ptr, stage, -1)
            r, i = _dif_natural_output(r), _dif_natural_output(i)
            pair = tl.arange(0, 2)[None, :]
            br, bi = tl.split(tl.load(b_fft_ptr + k[:, None] * 2 + pair))
            br, bi = br * (1.0 / 2048), bi * (1.0 / 2048)
            r, i = r * br - i * bi, i * br + r * bi
            for stage in tl.static_range(5):
                r, i = _dif_radix_four(r, i, roots_r_ptr, roots_inv_ptr, stage, 1)
            r, i = _dif_natural_output(r), _dif_natural_output(i)
            _prime_output(out_ptr, chirp_ptr, row, k, r, i, {n}, "{mode}")
        """
    )
    return name, source


def _dif_column_helpers(length: int, direction: str, *, tile: int = 32, root_stride: int = 1) -> str:
    sign = 1 if direction == "inverse" else -1
    phase_stride = f" * {root_stride}" if root_stride != 1 else ""
    return dedent(
        f"""
        @triton.jit
        def _first_axis_split(values, GROUPS: tl.constexpr, QUARTER: tl.constexpr):
            groups = tl.reshape(values, (GROUPS, 4, QUARTER, {tile}))
            digits = tl.reshape(tl.permute(groups, (0, 2, 3, 1)), (GROUPS, QUARTER, {tile}, 2, 2))
            ac, bd = tl.split(digits)
            a, c = tl.split(ac)
            b, d = tl.split(bd)
            return a, b, c, d

        @triton.jit
        def _first_axis_join(a, b, c, d, GROUPS: tl.constexpr, QUARTER: tl.constexpr):
            digits = tl.reshape(tl.join(tl.join(a, c), tl.join(b, d)), (GROUPS, QUARTER, {tile}, 4))
            return tl.reshape(tl.permute(digits, (0, 3, 1, 2)), ({length}, {tile}))

        @triton.jit
        def _first_axis_rotate(real, imag, roots_r, roots_i, phase):
            co = tl.load(roots_r + phase)[None, :, None]
            si = tl.load(roots_i + phase)[None, :, None]
            return real * co - imag * si, real * si + imag * co

        @triton.jit
        def _first_axis_radix_four(real, imag, roots_r, roots_i, STAGE: tl.constexpr):
            GROUPS: tl.constexpr = 1 << (2 * STAGE)
            QUARTER: tl.constexpr = {length} // (4 * GROUPS)
            ar, br, cr, dr = _first_axis_split(real, GROUPS, QUARTER)
            ai, bi, ci, di = _first_axis_split(imag, GROUPS, QUARTER)
            even_r, even_i = ar + cr, ai + ci
            odd_r, odd_i = br + dr, bi + di
            diff_even_r, diff_even_i = ar - cr, ai - ci
            diff_odd_r, diff_odd_i = br - dr, bi - di
            r1 = diff_even_r - ({sign}) * diff_odd_i
            i1 = diff_even_i + ({sign}) * diff_odd_r
            r2, i2 = even_r - odd_r, even_i - odd_i
            r3 = diff_even_r + ({sign}) * diff_odd_i
            i3 = diff_even_i - ({sign}) * diff_odd_r
            if QUARTER > 1:
                phase = tl.arange(0, QUARTER) * GROUPS{phase_stride}
                r1, i1 = _first_axis_rotate(r1, i1, roots_r, roots_i, phase)
                r2, i2 = _first_axis_rotate(r2, i2, roots_r, roots_i, phase * 2)
                r3, i3 = _first_axis_rotate(r3, i3, roots_r, roots_i, phase * 3)
            return (
                _first_axis_join(even_r + odd_r, r1, r2, r3, GROUPS, QUARTER),
                _first_axis_join(even_i + odd_i, i1, i2, i3, GROUPS, QUARTER),
            )
        """
    )


def build_maca_fp64_dif_first_axis(direction: str) -> tuple[str, str]:
    name = f"maca_fp64_dif_first64_{direction}"
    source = _dif_column_helpers(64, direction) + dedent(
        f"""
        @triton.jit
        def _first_axis_natural(values):
            digits = tl.reshape(values, (4, 4, 4, 32))
            return tl.reshape(tl.permute(digits, (2, 1, 0, 3)), (64, 32))

        @triton.jit
        def {name}(in_ptr, out_ptr, roots_r_ptr, roots_i_ptr, nbatch):
            block = tl.program_id(0)
            if block * 32 >= nbatch:
                return
            # One tile spans 32 complete rows of a dense 2048 by 64 plane.
            plane = block // 64
            middle = (block % 64) * 32 + tl.arange(0, 32)
            short = tl.arange(0, 64)
            pair = tl.arange(0, 2)[None, None, :]
            base = plane.to(tl.int64) * 262144
            src = (middle[None, :] * 64 + short[:, None]) * 2
            real, imag = tl.split(tl.load((in_ptr + base) + src[:, :, None] + pair))
            for stage in tl.static_range(3):
                real, imag = _first_axis_radix_four(real, imag, roots_r_ptr, roots_i_ptr, stage)
            values = tl.join(_first_axis_natural(real), _first_axis_natural(imag))
            dst = (short[:, None] * 2048 + middle[None, :]) * 2
            tl.store((out_ptr + base) + dst[:, :, None] + pair, values)
        """
    )
    return name, source


def _maca_real48_fft8_io(tile: int) -> str:
    return dedent(f"""
        @triton.jit
        def _split_fft8(values):
            even, odd = tl.split(tl.reshape(tl.trans(values), ({tile}, 2, 2, 2)))
            e0, e1 = tl.split(even)
            o0, o1 = tl.split(odd)
            x0, x4 = tl.split(e0)
            x2, x6 = tl.split(e1)
            x1, x5 = tl.split(o0)
            x3, x7 = tl.split(o1)
            return x0, x1, x2, x3, x4, x5, x6, x7

        @triton.jit
        def _join_fft8(a, b, c, d, e, f, g, h):
            even = tl.join(tl.join(a, e), tl.join(c, g))
            odd = tl.join(tl.join(b, f), tl.join(d, h))
            return tl.trans(tl.reshape(tl.join(even, odd), ({tile}, 8)))
    """)


def build_maca_fp64_r2c48_to_columns() -> tuple[str, str]:
    """Write the C550 46189 × 48 forward rows in compact column order."""
    name = "maca_fp64_r2c48_to_columns"
    width, tile = 46189, 16
    source = _maca_real48_fft8_io(tile) + dedent(f"""
        @triton.jit
        def _forward8(real, imag):
            r0, r1, r2, r3, r4, r5, r6, r7 = _split_fft8(real)
            i0, i1, i2, i3, i4, i5, i6, i7 = _split_fft8(imag)
            a, b, c, d, e, f, g, h, j, k, l, m, n, p, q, s = _fwd_rad8_b1(
                r0, r1, r2, r3, r4, r5, r6, r7, i0, i1, i2, i3, i4, i5, i6, i7)
            return _join_fft8(a, b, c, d, e, f, g, h), _join_fft8(j, k, l, m, n, p, q, s)

        @triton.jit
        def _reflect8(values):
            a, b, c, d, e, f, g, h = _split_fft8(values)
            return _join_fft8(a, h, g, f, e, d, c, b)

        @triton.jit
        def _store_columns(out_ptr, phase_r_ptr, phase_i_ptr, k8, k3: tl.constexpr,
                           dst, valid, ar, ai, peer_r, peer_i):
            k = (9 * k8 + 16 * k3) % 24
            br, bi = _reflect8(peer_r), _reflect8(peer_i)
            co = tl.load(phase_r_ptr + k)[:, None]
            si = tl.load(phase_i_ptr + k)[:, None]
            dr, di = ar - br, ai + bi
            re = (ar + br + dr * si + di * co) * 0.5
            im = (ai - bi - dr * co + di * si) * 0.5
            pair = tl.arange(0, 2)[None, None, :]
            offset = dst[None, :] + k[:, None] * {2 * width}
            tl.store(out_ptr + offset[:, :, None] + pair, tl.join(re, im), valid[None, :, None])
            if k3 == 0:
                # Packed bin zero also contains the real Nyquist mode.
                nyquist = dst[None, :] + {48 * width} + k8[:, None] * 2
                nyquist_valid = (k8 == 0)[:, None] & valid[None, :]
                tl.store(out_ptr + nyquist, ar - ai, nyquist_valid)
                tl.store(out_ptr + nyquist + 1, 0.0, nyquist_valid)

        @triton.jit
        def _load_pairs(in_ptr, row, valid, u, v: tl.constexpr):
            k = (3 * u + 8 * v) % 24
            pair = tl.arange(0, 2)[None, None, :]
            offset = row[None, :].to(tl.int64) * 48 + k[:, None] * 2
            return tl.split(tl.load(in_ptr + offset[:, :, None] + pair, valid[None, :, None], 0.0))

        @triton.jit
        def {name}(in_ptr, out_ptr, phase_r_ptr, phase_i_ptr, nbatch):
            # One tile can span the last rows of one matrix and the first of the next.
            physical = tl.program_id(0) * {tile} + tl.arange(0, {tile})
            matrix, row = physical // {width}, physical % {width}
            valid = physical < nbatch
            dst = matrix.to(tl.int64) * {50 * width} + row * 2
            u = tl.arange(0, 8)
            a, ai = _load_pairs(in_ptr, physical, valid, u, 0)
            b, bi = _load_pairs(in_ptr, physical, valid, u, 1)
            c, ci = _load_pairs(in_ptr, physical, valid, u, 2)
            r0, r1, r2, i0, i1, i2 = _fwd_rad3_b1(a, b, c, ai, bi, ci)
            r0, i0 = _forward8(r0, i0)
            r1, i1 = _forward8(r1, i1)
            r2, i2 = _forward8(r2, i2)
            _store_columns(out_ptr, phase_r_ptr, phase_i_ptr, u, 0, dst, valid, r0, i0, r0, i0)
            _store_columns(out_ptr, phase_r_ptr, phase_i_ptr, u, 1, dst, valid, r1, i1, r2, i2)
            _store_columns(out_ptr, phase_r_ptr, phase_i_ptr, u, 2, dst, valid, r2, i2, r1, i1)
    """)
    return name, source


def build_maca_fp64_c2r48_from_columns() -> tuple[str, str]:
    """Consume the column-major half spectrum of the C550 46189 × 48 inverse."""
    name = "maca_fp64_c2r48_from_columns"
    width, tile = 46189, 16
    source = _maca_real48_fft8_io(tile) + dedent(f"""
        @triton.jit
        def _inverse8(real, imag):
            r0, r1, r2, r3, r4, r5, r6, r7 = _split_fft8(real)
            i0, i1, i2, i3, i4, i5, i6, i7 = _split_fft8(-imag)
            a, b, c, d, e, f, g, h, t0, t1, t2, t3, t4, t5, t6, t7 = _fwd_rad8_b1(
                r0, r1, r2, r3, r4, r5, r6, r7, i0, i1, i2, i3, i4, i5, i6, i7)
            return _join_fft8(a, b, c, d, e, f, g, h), -_join_fft8(t0, t1, t2, t3, t4, t5, t6, t7)

        @triton.jit
        def _reconstruct(in_ptr, phase_r_ptr, phase_i_ptr, base, valid, u, v: tl.constexpr):
            k = (3 * u + 8 * v) % 24
            pair = tl.arange(0, 2)[None, None, :]
            src = base[None, :] + k[:, None] * {2 * width}
            mirror = base[None, :] + (24 - k[:, None]) * {2 * width}
            mask = valid[None, :, None]
            ar, ai = tl.split(tl.load(in_ptr + src[:, :, None] + pair, mask, 0.0))
            br, bi = tl.split(tl.load(in_ptr + mirror[:, :, None] + pair, mask, 0.0))
            ai = tl.where(k[:, None] == 0, 0.0, ai)
            bi = tl.where(k[:, None] == 0, 0.0, bi)
            co = tl.load(phase_r_ptr + k)[:, None]
            si = tl.load(phase_i_ptr + k)[:, None]
            dr, di = ar - br, ai + bi
            return ar + br - (di * co - dr * si), ai - bi + dr * co + di * si

        @triton.jit
        def _write_real(out_ptr, rows, valid, k8, k3: tl.constexpr, real, imag):
            index = (9 * k8 + 16 * k3) % 24
            target = rows[None, :].to(tl.int64) * 48 + index[:, None] * 2
            offsets = tl.permute(target[:, :, None] + tl.arange(0, 2)[None, None, :], (1, 0, 2))
            values = tl.permute(tl.join(real, imag), (1, 0, 2))
            tl.store(out_ptr + offsets, values, valid[:, None, None])

        @triton.jit
        def {name}(in_ptr, out_ptr, phase_r_ptr, phase_i_ptr, nbatch):
            # nbatch counts logical rows. One tile can span two input matrices.
            rows = tl.program_id(0) * {tile} + tl.arange(0, {tile})
            matrix, row = rows // {width}, rows % {width}
            base = matrix.to(tl.int64) * {50 * width} + row * 2
            valid = rows < nbatch
            u = tl.arange(0, 8)
            a, ai = _reconstruct(in_ptr, phase_r_ptr, phase_i_ptr, base, valid, u, 0)
            b, bi = _reconstruct(in_ptr, phase_r_ptr, phase_i_ptr, base, valid, u, 1)
            c, ci = _reconstruct(in_ptr, phase_r_ptr, phase_i_ptr, base, valid, u, 2)
            r0, r1, r2, i0, i1, i2 = _fwd_rad3_b1(a, b, c, -ai, -bi, -ci)
            r0, i0 = _inverse8(r0, -i0)
            r1, i1 = _inverse8(r1, -i1)
            r2, i2 = _inverse8(r2, -i2)
            _write_real(out_ptr, rows, valid, u, 0, r0, i0)
            _write_real(out_ptr, rows, valid, u, 1, r1, i1)
            _write_real(out_ptr, rows, valid, u, 2, r2, i2)
    """)
    return name, source


def build_maca_fp64_real32_plane(direction: str) -> tuple[str, str]:
    name = f"maca_real32_plane_dif_{direction}"
    inverse = direction == "inverse"
    source = _dif_column_helpers(32, direction) + dedent(f"""
        @triton.jit
        def _plane_fft32_finish(values):
            pair = tl.permute(tl.reshape(values, (16, 2, 32)), (0, 2, 1))
            a, b = tl.split(pair)
            digits = tl.reshape(tl.join(a + b, a - b), (4, 4, 32, 2))
            return tl.reshape(tl.permute(digits, (3, 1, 0, 2)), (32, 32))

        @triton.jit
        def _plane_fft32(real, imag, roots_r, roots_i):
            for stage in tl.static_range(2):
                real, imag = _first_axis_radix_four(real, imag, roots_r, roots_i, stage)
            return _plane_fft32_finish(real), _plane_fft32_finish(imag)

        @triton.jit
        def {name}(in_ptr, out_ptr, tw_r_ptr, tw_i_ptr):
            plane = tl.program_id(0)
            row = tl.arange(0, 32)[:, None]
            col = tl.arange(0, 32)[None, :]
            pair = tl.arange(0, 2)[None, None, :]
            if {inverse}:
                reflected = col > 16
                src_row = tl.where(reflected, (32 - row) % 32, row)
                src_col = tl.where(reflected, 32 - col, col)
                src = (plane * 544 + src_row * 17 + src_col) * 2
                real, imag = tl.split(tl.load(in_ptr + src[:, :, None] + pair))
                imag = tl.where(reflected, -imag, imag)
            else:
                real = tl.load(in_ptr + plane * 1024 + row * 32 + col)
                imag = tl.zeros((32, 32), tl.float64)
            real, imag = _plane_fft32(tl.trans(real), tl.trans(imag), tw_r_ptr, tw_i_ptr)
            real, imag = _plane_fft32(tl.trans(real), tl.trans(imag), tw_r_ptr, tw_i_ptr)
            if {inverse}:
                tl.store(out_ptr + plane * 1024 + row * 32 + col, real)
            else:
                dst = (plane * 544 + row * 17 + col) * 2
                tl.store(out_ptr + dst[:, :, None] + pair, tl.join(real, imag), (col < 17)[:, :, None])
    """)
    return name, source


def build_maca_3d_real_boundary(
    direction: str, *, fp64: bool, middle_length: int = 2048,
) -> tuple[str, str]:
    inverse = direction == "inverse"
    mode = "c2r" if inverse else "r2c"
    precision = "fp64" if fp64 else "fp32"
    masked = middle_length % 32 != 0
    tiles_per_plane = (middle_length + 31) // 32
    prime = "prime_" if masked else ""
    name = f"maca_{precision}_3d_{prime}{mode}64"
    mask_declaration = f"valid = middle < {middle_length}" if masked else ""
    load_mask = ", valid[None, :, None], 0.0" if masked else ""
    store_mask = ", valid[None, :, None]" if masked else ""
    nyquist_mask = " & valid[None, :]" if masked else ""
    zero_mask = ", valid" if masked else ""
    source = _dif_column_helpers(32, direction) + dedent(
        f"""
        @triton.jit
        def _packed_natural(values):
            pairs = tl.permute(tl.reshape(values, (16, 2, 32)), (0, 2, 1))
            a, b = tl.split(pairs)
            values = tl.permute(tl.join(a + b, a - b), (0, 2, 1))
            digits = tl.reshape(values, (4, 4, 2, 32))
            return tl.reshape(tl.permute(digits, (2, 1, 0, 3)), (32, 32))

        @triton.jit
        def _packed_mirror_bit(values, BIT: tl.constexpr):
            LOWER: tl.constexpr = 1 << BIT
            GROUPS: tl.constexpr = 32 // (LOWER * 2)
            digits = tl.reshape(values, (GROUPS, 2, LOWER, 32))
            low, high = tl.split(tl.permute(digits, (0, 2, 3, 1)))
            flip = tl.arange(0, LOWER)[None, :, None] != 0
            swapped = tl.join(tl.where(flip, high, low), tl.where(flip, low, high))
            return tl.reshape(tl.permute(swapped, (0, 3, 1, 2)), (32, 32))

        @triton.jit
        def _packed_mirror(values):
            # Negation modulo 32 complements the bits above the lowest set bit.
            for bit in tl.static_range(1, 5):
                values = _packed_mirror_bit(values, bit)
            return values

        @triton.jit
        def {name}(in_ptr, out_ptr, roots_r_ptr, roots_i_ptr, phase_r_ptr, phase_i_ptr, nbatch):
            block = tl.program_id(0)
            if block * 32 >= nbatch:
                return
            INVERSE: tl.constexpr = {inverse}
            plane = (block // {tiles_per_plane}).to(tl.int64)
            middle = (block % {tiles_per_plane}) * 32 + tl.arange(0, 32)
            {mask_declaration}
            k = tl.arange(0, 32)
            pair = tl.arange(0, 2)[None, None, :]
            real_base = plane * {middle_length * 64}
            compact_base = plane * {middle_length * 66}
            real_off = real_base + middle[None, :] * 64 + k[:, None] * 2
            compact_off = compact_base + (k[:, None] * {middle_length} + middle[None, :]) * 2
            co = tl.load(phase_r_ptr + k)[:, None]
            si = tl.load(phase_i_ptr + k)[:, None]
            if INVERSE:
                rev_off = compact_base + ((32 - k[:, None]) * {middle_length} + middle[None, :]) * 2
                ar, ai = tl.split(tl.load(in_ptr + compact_off[:, :, None] + pair{load_mask}))
                br, bi = tl.split(tl.load(in_ptr + rev_off[:, :, None] + pair{load_mask}))
                dr, di = ar - br, ai + bi
                real = ar + br - (dr * si + di * co)
                imag = ai - bi + (dr * co - di * si)
            else:
                real, imag = tl.split(tl.load(in_ptr + real_off[:, :, None] + pair{load_mask}))
            for stage in tl.static_range(2):
                real, imag = _first_axis_radix_four(real, imag, roots_r_ptr, roots_i_ptr, stage)
            real, imag = _packed_natural(real), _packed_natural(imag)
            if INVERSE:
                tl.store(out_ptr + real_off[:, :, None] + pair, tl.join(real, imag){store_mask})
            else:
                br, bi = _packed_mirror(real), _packed_mirror(imag)
                dr, di = real - br, imag + bi
                out_r = (real + br + dr * si + di * co) * 0.5
                out_i = (imag - bi - dr * co + di * si) * 0.5
                tl.store(out_ptr + compact_off[:, :, None] + pair, tl.join(out_r, out_i){store_mask})
                nyquist = compact_base + (32 * {middle_length} + middle) * 2
                nyquist_off = tl.broadcast_to(nyquist[None, :], (32, 32))
                tl.store(out_ptr + nyquist_off, real - imag, (k[:, None] == 0){nyquist_mask})
                tl.store(out_ptr + nyquist + 1, 0.0{zero_mask})
        """
    )
    return name, source


def build_maca_3d_prime_real_outer(direction: str, *, fp64: bool, wide: bool = False) -> tuple[str, str]:
    precision = "fp64" if fp64 else "fp32"
    inverse = direction == "inverse"
    if wide and (inverse or fp64):
        raise ValueError("the wide outer tile requires forward FP32")
    half_tile, middle_tile = (4, 16) if inverse else (32 if wide else 16, 4)
    tile = half_tile * middle_tile
    middle_tiles = (997 + middle_tile - 1) // middle_tile
    tiles_per_batch = ((33 + half_tile - 1) // half_tile) * middle_tiles
    suffix = "_4x16" if inverse else ("_32x4" if wide else "")
    name = f"maca_{precision}_3d_prime_outer16_{direction}{suffix}"
    source_layout, target_layout = ("middle_major", "half_major") if inverse else (
        "half_major", "middle_major"
    )
    source = _dif_column_helpers(16, direction, tile=tile) + dedent(f"""
        @triton.jit
        def {name}(in_ptr, out_ptr, roots_r_ptr, roots_i_ptr, nbatch):
            block = tl.program_id(0)
            if block * {tile} >= nbatch:
                return
            batch = block // {tiles_per_batch}
            local = block % {tiles_per_batch}
            lane = tl.arange(0, {tile})
            h = (local // {middle_tiles}) * {half_tile} + lane // {middle_tile}
            m = (local % {middle_tiles}) * {middle_tile} + lane % {middle_tile}
            valid = (h < 33) & (m < 997)
            k = tl.arange(0, 16)
            pair = tl.arange(0, 2)[None, None, :]
            base = (batch.to(tl.int64) * 16 + k) * 32901
            half_major = (base[:, None] + h[None, :] * 997 + m[None, :]) * 2
            middle_major = (base[:, None] + m[None, :] * 33 + h[None, :]) * 2
            r, i = tl.split(tl.load(in_ptr + {source_layout}[:, :, None] + pair, valid[None, :, None], 0.0))
            for step in tl.static_range(2):
                r, i = _first_axis_radix_four(r, i, roots_r_ptr, roots_i_ptr, step)
            r = tl.reshape(tl.permute(tl.reshape(r, (4, 4, {tile})), (1, 0, 2)), (16, {tile}))
            i = tl.reshape(tl.permute(tl.reshape(i, (4, 4, {tile})), (1, 0, 2)), (16, {tile}))
            tl.store(out_ptr + {target_layout}[:, :, None] + pair, tl.join(r, i), valid[None, :, None])
    """)
    return name, source
