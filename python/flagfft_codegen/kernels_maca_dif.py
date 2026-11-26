# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Register-tiled C550 FP64 FFT with radix-4 DIF stages."""

from textwrap import dedent


def _dif2048_helpers(direction: str | None) -> str:
    sign = "SIGN" if direction is None else (1 if direction == "inverse" else -1)
    sign_arg = ", SIGN: tl.constexpr" if direction is None else ""
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
            return tl.reshape(tl.permute(tile, (0, 2, 1)), (2048,))

        @triton.jit
        def _dif_rotate(real, imag, roots_r, roots_i, indices):
            co = tl.load(roots_r + indices)[None, :]
            si = tl.load(roots_i + indices)[None, :]
            return real * co - imag * si, real * si + imag * co

        @triton.jit
        def _dif_radix_four(real, imag, roots_r, roots_i, STAGE: tl.constexpr{sign_arg}):
            GROUPS: tl.constexpr = 1 << (2 * STAGE)
            QUARTER: tl.constexpr = 2048 // (4 * GROUPS)
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

        @triton.jit
        def _dif_natural_output(value):
            low, high = tl.split(tl.reshape(value, (1024, 2)))
            result = tl.join(low + high, low - high)
            result = tl.reshape(result, (4, 4, 4, 4, 4, 2))
            result = tl.permute(result, (5, 4, 3, 2, 1, 0))
            return tl.reshape(result, (2048,))
        """
    )


def build_maca_fp64_dif_leaf(direction: str) -> tuple[str, str]:
    name = f"maca_fp64_dif2048_{direction}"
    source = _dif2048_helpers(direction) + dedent(
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
            for stage in tl.static_range(5):
                real, imag = _dif_radix_four(real, imag, roots_r_ptr, roots_i_ptr, stage)
            value = tl.join(_dif_natural_output(real), _dif_natural_output(imag))
            tl.store((out_ptr + base) + offset, value)
        """
    )
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


def build_maca_fp64_dif_first_axis(direction: str) -> tuple[str, str]:
    name = f"maca_fp64_dif_first64_{direction}"
    sign = 1 if direction == "inverse" else -1
    source = dedent(
        f"""
        @triton.jit
        def _first_axis_split(values, GROUPS: tl.constexpr, QUARTER: tl.constexpr):
            groups = tl.reshape(values, (GROUPS, 4, QUARTER, 32))
            digits = tl.reshape(tl.permute(groups, (0, 2, 3, 1)), (GROUPS, QUARTER, 32, 2, 2))
            ac, bd = tl.split(digits)
            a, c = tl.split(ac)
            b, d = tl.split(bd)
            return a, b, c, d

        @triton.jit
        def _first_axis_join(a, b, c, d, GROUPS: tl.constexpr, QUARTER: tl.constexpr):
            digits = tl.reshape(tl.join(tl.join(a, c), tl.join(b, d)), (GROUPS, QUARTER, 32, 4))
            return tl.reshape(tl.permute(digits, (0, 3, 1, 2)), (64, 32))

        @triton.jit
        def _first_axis_rotate(real, imag, roots_r, roots_i, phase):
            co = tl.load(roots_r + phase)[None, :, None]
            si = tl.load(roots_i + phase)[None, :, None]
            return real * co - imag * si, real * si + imag * co

        @triton.jit
        def _first_axis_radix_four(real, imag, roots_r, roots_i, STAGE: tl.constexpr):
            GROUPS: tl.constexpr = 1 << (2 * STAGE)
            QUARTER: tl.constexpr = 64 // (4 * GROUPS)
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
                phase = tl.arange(0, QUARTER) * GROUPS
                r1, i1 = _first_axis_rotate(r1, i1, roots_r, roots_i, phase)
                r2, i2 = _first_axis_rotate(r2, i2, roots_r, roots_i, phase * 2)
                r3, i3 = _first_axis_rotate(r3, i3, roots_r, roots_i, phase * 3)
            return (
                _first_axis_join(even_r + odd_r, r1, r2, r3, GROUPS, QUARTER),
                _first_axis_join(even_i + odd_i, i1, i2, i3, GROUPS, QUARTER),
            )

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
