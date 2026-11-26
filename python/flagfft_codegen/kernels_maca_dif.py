# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Register-tiled C550 FP64 FFT with radix-4 DIF stages."""

from textwrap import dedent


def build_maca_fp64_dif_leaf(direction: str) -> tuple[str, str]:
    name = f"maca_fp64_dif2048_{direction}"
    sign = 1 if direction == "inverse" else -1
    source = dedent(
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
        def _dif_radix_four(real, imag, roots_r, roots_i, STAGE: tl.constexpr):
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
