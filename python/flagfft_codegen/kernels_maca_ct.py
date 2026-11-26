# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Two-pass outer-axis C550 FFT for the dense 128 by 2048 by 64 shape."""

from textwrap import dedent


def build_maca_fp64_outer_ct(direction: str, *, prepare: bool) -> tuple[str, str]:
    phase = "prepare" if prepare else "finish"
    name = f"maca_fp64_ct128_{phase}_{direction}"
    helpers = dedent(
        """
        @triton.jit
        def _outer_stage(real, imag, roots_r, roots_i, STEP: tl.constexpr, LOG: tl.constexpr):
            HALF: tl.constexpr = 1 << STEP
            GROUPS: tl.constexpr = 1 << (LOG - STEP - 1)
            paired_r = tl.reshape(real, (GROUPS, 2, HALF, 64))
            paired_i = tl.reshape(imag, (GROUPS, 2, HALF, 64))
            ar, br = tl.split(tl.permute(paired_r, (0, 2, 3, 1)))
            ai, bi = tl.split(tl.permute(paired_i, (0, 2, 3, 1)))
            phase = tl.arange(0, HALF) * (128 // (2 * HALF))
            co = tl.load(roots_r + phase)[None, :, None]
            si = tl.load(roots_i + phase)[None, :, None]
            tr, ti = br * co - bi * si, br * si + bi * co
            next_r = tl.permute(tl.join(ar + tr, ar - tr), (0, 3, 1, 2))
            next_i = tl.permute(tl.join(ai + ti, ai - ti), (0, 3, 1, 2))
            return tl.reshape(next_r, (1 << LOG, 64)), tl.reshape(next_i, (1 << LOG, 64))

        @triton.jit
        def _outer_bit_reverse(index, LOG: tl.constexpr):
            reverse = tl.zeros_like(index)
            for bit in tl.static_range(LOG):
                reverse = (reverse << 1) | ((index >> bit) & 1)
            return reverse
        """
    )
    if prepare:
        body = dedent(
            f"""
            @triton.jit
            def {name}(in_ptr, out_ptr, roots_r_ptr, roots_i_ptr, nbatch):
                block = tl.program_id(0)
                if block * 4 >= nbatch:
                    return
                cube = block // 32768
                part = (block // 2048) % 16
                group = block % 2048
                base = cube.to(tl.int64) * 33554432
                y_base = (group % 256) * 8
                z_base = (group // 256) * 8
                sample = tl.arange(0, 64)
                y = y_base + sample % 8
                z = z_base + sample // 8
                index = tl.arange(0, 8)
                input_row = _outer_bit_reverse(index, 3) * 16 + part
                pair = tl.arange(0, 2)[None, None, :]
                src = input_row[:, None] * 131072 + z[None, :] * 2048 + y[None, :]
                real, imag = tl.split(tl.load((in_ptr + base) + src[:, :, None] * 2 + pair))
                for step in tl.static_range(3):
                    real, imag = _outer_stage(real, imag, roots_r_ptr, roots_i_ptr, step, 3)
                phase = index * part
                co = tl.load(roots_r_ptr + phase)[:, None]
                si = tl.load(roots_i_ptr + phase)[:, None]
                real, imag = real * co - imag * si, real * si + imag * co
                values = tl.reshape(tl.join(real, imag), (8, 8, 8, 2))
                values = tl.reshape(tl.permute(values, (0, 2, 1, 3)), (8, 64, 2))
                output_row = index * 16 + part
                out_y = y_base + sample // 8
                out_z = z_base + sample % 8
                dst = output_row[:, None] * 131072 + out_y[None, :] * 64 + out_z[None, :]
                tl.store((out_ptr + base) + dst[:, :, None] * 2 + pair, values)
            """
        )
    else:
        body = dedent(
            f"""
            @triton.jit
            def {name}(in_ptr, out_ptr, roots_r_ptr, roots_i_ptr, nbatch):
                block = tl.program_id(0)
                if block * 8 >= nbatch:
                    return
                cube = block // 16384
                part = (block // 2048) % 8
                base = cube.to(tl.int64) * 33554432
                column = (block % 2048) * 64 + tl.arange(0, 64)
                index = tl.arange(0, 16)
                pair = tl.arange(0, 2)[None, None, :]
                input_row = part * 16 + _outer_bit_reverse(index, 4)
                src = input_row[:, None] * 131072 + column[None, :]
                real, imag = tl.split(tl.load((in_ptr + base) + src[:, :, None] * 2 + pair))
                for step in tl.static_range(4):
                    real, imag = _outer_stage(real, imag, roots_r_ptr, roots_i_ptr, step, 4)
                output_row = index * 8 + part
                dst = output_row[:, None] * 131072 + column[None, :]
                tl.store((out_ptr + base) + dst[:, :, None] * 2 + pair, tl.join(real, imag))
            """
        )
    return name, helpers + body
