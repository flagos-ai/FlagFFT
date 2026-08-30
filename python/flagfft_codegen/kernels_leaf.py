# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

"""Mixed-radix leaf generation: codelet emitters, stage/route emission, thread-local leaves and I/O builders."""

import math
from textwrap import dedent
from typing import Literal

from .kernels_common import (
    LeafIoMode,
    LeafPlan,
    _NATURAL_ORDER_CODELET_RADICES,
    _THREAD_LOCAL_MIXED_RADICES,
    _TLE_SMEM_SWIZZLE_SHIFT,
    _is_double_dtype,
    _non_nvidia_backend_active,
    _real_element_bytes,
    _tl_real_dtype,
    _use_single_smem_buffer,
    contiguous_batch_pack_for,
    cooperative_stage_lanes_for,
    four_step_col_inner_pack_for,
    four_step_row_inner_pack_for,
    lane_block_for,
    use_tle_fused_twiddle,
    use_four_step_row_fused_twiddle,
)

def _fmt_const(value: float) -> str:
    if abs(value) < 1e-8:
        value = 0.0
    elif abs(value - 1.0) < 1e-8:
        value = 1.0
    elif abs(value + 1.0) < 1e-8:
        value = -1.0
    return repr(float(value))


def _direction_sign(direction: Literal["forward", "inverse"]) -> float:
    return 1.0 if direction == "inverse" else -1.0


def _emit_table_codelet(
    indent: str, radix: int, lane_block: int, dtype: str = "complex64"
) -> list[str]:
    lines: list[str] = []
    for kout in range(radix):
        lines.append(f"{indent}acc_r_{kout} = tl.zeros_like(r0)")
        lines.append(f"{indent}acc_i_{kout} = tl.zeros_like(i0)")

    for kout in range(radix):
        for nin in range(radix):
            lines.append(
                f"{indent}wr = tl.load(dft{radix}_r_ptr + {kout * radix + nin})"
            )
            lines.append(
                f"{indent}wi = tl.load(dft{radix}_i_ptr + {kout * radix + nin})"
            )
            lines.append(f"{indent}pr, pi = _cmul(r{nin}, i{nin}, wr, wi)")
            lines.append(f"{indent}acc_r_{kout} += pr")
            lines.append(f"{indent}acc_i_{kout} += pi")

    for kout in range(radix):
        lines.append(f"{indent}r{kout} = acc_r_{kout}")
        lines.append(f"{indent}i{kout} = acc_i_{kout}")
    return lines


def _emit_natural_order_codelet_call(
    indent: str,
    radix: int,
    direction: Literal["forward", "inverse"],
    indices: list[int] | None = None,
) -> list[str]:
    if indices is None:
        indices = list(range(radix))
    if len(indices) != radix:
        raise ValueError(f"radix-{radix} codelet requires {radix} register indices")
    lines: list[str] = []
    if direction == "inverse":
        for idx in indices:
            lines.append(f"{indent}i{idx} = -i{idx}")
    lines.append(f"{indent}(")
    for idx in indices:
        lines.append(f"{indent}    r{idx},")
    for idx in indices:
        lines.append(f"{indent}    i{idx},")
    args = ", ".join([*(f"r{idx}" for idx in indices), *(f"i{idx}" for idx in indices)])
    lines.append(f"{indent}) = _fwd_rad{radix}_b1({args})")
    if direction == "inverse":
        for idx in indices:
            lines.append(f"{indent}i{idx} = -i{idx}")
    return lines


def _emit_radix16_codelet_call(
    indent: str,
    direction: Literal["forward", "inverse"],
    offset: int = 0,
) -> list[str]:
    lines: list[str] = []
    indices = [offset + idx for idx in range(16)]
    if direction == "inverse":
        for idx in indices:
            lines.append(f"{indent}i{idx} = -i{idx}")
    lines.append(f"{indent}(")
    for idx in indices:
        lines.append(f"{indent}    r{idx},")
    for idx in indices:
        lines.append(f"{indent}    i{idx},")
    radix16_order = (0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15)
    args = ", ".join(
        [
            *(f"r{offset + idx}" for idx in radix16_order),
            *(f"i{offset + idx}" for idx in radix16_order),
        ]
    )
    lines.append(f"{indent}) = _fwd_rad16_b1({args})")
    if direction == "inverse":
        for idx in indices:
            lines.append(f"{indent}i{idx} = -i{idx}")
    return lines


def _emit_local_radix32_codelet_call(
    indent: str, direction: Literal["forward", "inverse"]
) -> list[str]:
    """Emit a radix-32 FFT whose complete working set is owned by one thread."""
    lines = _emit_radix16_codelet_call(indent, direction)
    lines.extend(_emit_radix16_codelet_call(indent, direction, offset=16))
    sign = _direction_sign(direction)
    for idx in range(16):
        wr = _fmt_const(math.cos(sign * 2.0 * math.pi * idx / 32.0))
        wi = _fmt_const(math.sin(sign * 2.0 * math.pi * idx / 32.0))
        lines.extend(
            [
                (
                    f"{indent}odd_tw_r{idx}, odd_tw_i{idx} = "
                    f"_cmul(r{idx + 16}, i{idx + 16}, {wr}, {wi})"
                ),
                f"{indent}even_r{idx} = r{idx}",
                f"{indent}even_i{idx} = i{idx}",
                f"{indent}r{idx} = even_r{idx} + odd_tw_r{idx}",
                f"{indent}i{idx} = even_i{idx} + odd_tw_i{idx}",
                f"{indent}r{idx + 16} = even_r{idx} - odd_tw_r{idx}",
                f"{indent}i{idx + 16} = even_i{idx} - odd_tw_i{idx}",
            ]
        )
    return lines


def _emit_natural_order_radix32_codelet_call(
    indent: str, direction: Literal["forward", "inverse"]
) -> list[str]:
    """Emit radix-32 for naturally ordered inputs using the factorized codelet."""
    lines: list[str] = []
    for idx in range(32):
        lines.append(f"{indent}rad32_in_r{idx} = r{idx}")
        lines.append(f"{indent}rad32_in_i{idx} = i{idx}")
    for idx in range(32):
        source_idx = 2 * (idx % 16) + idx // 16
        lines.append(f"{indent}r{idx} = rad32_in_r{source_idx}")
        lines.append(f"{indent}i{idx} = rad32_in_i{source_idx}")
    lines.extend(_emit_local_radix32_codelet_call(indent, direction))
    return lines


def _emit_local_mixed_codelet_call(
    indent: str,
    radix: int,
    direction: Literal["forward", "inverse"],
) -> list[str]:
    """Emit a register-only FFT for supported composite large-1D leaf radices."""
    if radix == 32:
        return _emit_local_radix32_codelet_call(indent, direction)

    split = {
        18: (3, 6),
        20: (5, 4),
        24: (3, 8),
        25: (5, 5),
        27: (3, 9),
        28: (7, 4),
        30: (3, 10),
    }.get(radix)
    if split is None:
        raise ValueError(f"unsupported thread-local mixed radix {radix}")

    outer_radix, inner_radix = split
    lines: list[str] = []
    for outer_digit in range(outer_radix):
        indices = [
            outer_digit + outer_radix * inner_digit
            for inner_digit in range(inner_radix)
        ]
        lines.extend(
            _emit_natural_order_codelet_call(indent, inner_radix, direction, indices)
        )

    sign = _direction_sign(direction)
    for inner_freq in range(inner_radix):
        indices = [
            outer_digit + outer_radix * inner_freq for outer_digit in range(outer_radix)
        ]
        for outer_digit, register_idx in enumerate(indices[1:], start=1):
            angle = sign * 2.0 * math.pi * outer_digit * inner_freq / float(radix)
            wr = _fmt_const(math.cos(angle))
            wi = _fmt_const(math.sin(angle))
            lines.append(
                f"{indent}r{register_idx}, i{register_idx} = "
                f"_cmul(r{register_idx}, i{register_idx}, {wr}, {wi})"
            )
        lines.extend(
            _emit_natural_order_codelet_call(indent, outer_radix, direction, indices)
        )

    for outer_freq in range(outer_radix):
        for inner_freq in range(inner_radix):
            output_idx = inner_freq + inner_radix * outer_freq
            register_idx = outer_freq + outer_radix * inner_freq
            lines.append(f"{indent}mixed_out_r{output_idx} = r{register_idx}")
            lines.append(f"{indent}mixed_out_i{output_idx} = i{register_idx}")
    for output_idx in range(radix):
        lines.append(f"{indent}r{output_idx} = mixed_out_r{output_idx}")
        lines.append(f"{indent}i{output_idx} = mixed_out_i{output_idx}")
    return lines


def _time_major_stride(radices: tuple[int, ...], axis: int) -> int:
    return math.prod(radices[axis + 1 :])


def _emit_input_base(
    indent: str,
    factors: tuple[int, ...],
    lanes: int,
    group_var: str,
) -> list[str]:
    lines = [
        f"{indent}codelet_in = tl.where(lane_mask, lane + {lanes} * {group_var}, 0)",
        f"{indent}rem_in = codelet_in",
        f"{indent}input_base = lane * 0",
    ]
    for axis in range(len(factors) - 1, 0, -1):
        lines.append(f"{indent}digit_in_{axis} = rem_in % {factors[axis]}")
        lines.append(f"{indent}rem_in = rem_in // {factors[axis]}")
        stride = _time_major_stride(factors, axis)
        if stride == 1:
            lines.append(f"{indent}input_base += digit_in_{axis}")
        else:
            lines.append(f"{indent}input_base += digit_in_{axis} * {stride}")
    return lines


def _emit_input_index(
    indent: str, out_var: str, factors: tuple[int, ...], digit: int
) -> list[str]:
    offset = digit * _time_major_stride(factors, 0)
    if offset == 0:
        return [f"{indent}{out_var} = input_base"]
    return [f"{indent}{out_var} = input_base + {offset}"]


def _emit_output_base(
    indent: str,
    factors: tuple[int, ...],
    lanes: int,
    group_var: str,
) -> list[str]:
    last_stage = len(factors) - 1
    lines = [
        f"{indent}codelet_out = tl.where(lane_mask, lane + {lanes} * {group_var}, 0)",
        f"{indent}rem_out = codelet_out",
        f"{indent}output_base = lane * 0",
    ]
    stride = 1
    for axis in range(last_stage):
        lines.append(f"{indent}digit_out_{axis} = rem_out % {factors[axis]}")
        lines.append(f"{indent}rem_out = rem_out // {factors[axis]}")
        if stride == 1:
            lines.append(f"{indent}output_base += digit_out_{axis}")
        else:
            lines.append(f"{indent}output_base += digit_out_{axis} * {stride}")
        stride *= factors[axis]
    return lines


def _emit_output_index(
    indent: str, out_var: str, factors: tuple[int, ...], digit: int
) -> list[str]:
    last_stride = math.prod(factors[: len(factors) - 1])
    offset = digit * last_stride
    if offset == 0:
        return [f"{indent}{out_var} = output_base"]
    return [f"{indent}{out_var} = output_base + {offset}"]


def _emit_route_base(
    indent: str,
    stage: int,
    factors: tuple[int, ...],
    lanes: int,
    group_var: str,
) -> list[str]:
    lines = [
        f"{indent}codelet_route{stage} = tl.where(lane_mask, lane + {lanes} * {group_var}, 0)",
        f"{indent}rem_route{stage} = codelet_route{stage}",
        f"{indent}route_codelet_base{stage} = lane * 0",
    ]

    stride = 1
    for axis in range(stage):
        lines.append(
            f"{indent}digit_route{stage}_{axis} = rem_route{stage} % {factors[axis]}"
        )
        lines.append(f"{indent}rem_route{stage} = rem_route{stage} // {factors[axis]}")
        if stride == 1:
            lines.append(
                f"{indent}route_codelet_base{stage} += digit_route{stage}_{axis}"
            )
        else:
            lines.append(
                f"{indent}route_codelet_base{stage} += digit_route{stage}_{axis} * {stride}"
            )
        stride *= factors[axis]

    stride *= factors[stage]

    for axis in range(len(factors) - 1, stage, -1):
        lines.append(
            f"{indent}digit_route{stage}_{axis} = rem_route{stage} % {factors[axis]}"
        )
        lines.append(f"{indent}rem_route{stage} = rem_route{stage} // {factors[axis]}")
        if axis == stage + 1:
            lines.append(f"{indent}next_digit{stage} = digit_route{stage}_{axis}")
        else:
            if stride == 1:
                lines.append(
                    f"{indent}route_codelet_base{stage} += digit_route{stage}_{axis}"
                )
            else:
                lines.append(
                    f"{indent}route_codelet_base{stage} += digit_route{stage}_{axis} * {stride}"
                )
            stride *= factors[axis]

    return lines


def _emit_route_index(
    indent: str,
    out_var: str,
    stage: int,
    factors: tuple[int, ...],
    lanes: int,
    digit: int,
) -> list[str]:
    suffix = out_var.removeprefix("dst")
    radix_next = factors[stage + 1]
    current_stride = math.prod(factors[:stage])
    offset = digit * current_stride
    if offset == 0:
        lines = [f"{indent}next_codelet{stage}_{suffix} = route_codelet_base{stage}"]
    else:
        lines = [
            f"{indent}next_codelet{stage}_{suffix} = route_codelet_base{stage} + {offset}"
        ]
    lines.append(
        f"{indent}{out_var} = (next_codelet{stage}_{suffix} % {lanes}) + "
        f"{lanes} * ((next_codelet{stage}_{suffix} // {lanes}) * {radix_next} + next_digit{stage})"
    )
    return lines


def _emit_stage_block(
    stage: int,
    factors: tuple[int, ...],
    n: int,
    lanes: int,
    lane_block: int,
    *,
    io_mode: LeafIoMode = "contiguous",
    bluestein_pass: int = 0,
    bluestein_intermediate_buffer: str = "smem_a",
    prime_n: int = 0,
    four_step_n1: int = 0,
    four_step_n2: int = 0,
    smem_pack: int = 1,
    fuse_twiddle_into_row: bool = False,
    single_smem_buffer: bool = False,
    direction: Literal["forward", "inverse"] = "forward",
    dtype: str = "complex64",
    stage_lanes: tuple[int, ...] | None = None,
) -> list[str]:
    radix = factors[stage]
    current_lanes = stage_lanes[stage] if stage_lanes is not None else lanes
    next_lanes = (
        stage_lanes[stage + 1]
        if stage_lanes is not None and stage + 1 < len(stage_lanes)
        else current_lanes
    )
    groups = n // (current_lanes * radix)
    is_last = stage == len(factors) - 1
    source_buffer = (
        None
        if stage == 0
        else ("smem_b" if single_smem_buffer or stage % 2 == 1 else "smem_a")
    )
    dest_buffer = (
        None
        if is_last
        else ("smem_b" if single_smem_buffer or stage % 2 == 0 else "smem_a")
    )
    zero = "0.0"

    lines: list[str] = []
    if stage_lanes is not None:
        lines.append(f"    lane_mask = base_lane_mask & (lane < {current_lanes})")
    vectorized_complex_io = (
        io_mode in {"contiguous", "contiguous_c2r"} and not _non_nvidia_backend_active()
    )
    vector_suffix = "f64" if _is_double_dtype(dtype) else "f32"
    vector_reg = "d" if _is_double_dtype(dtype) else "f"
    vector_dtype = "tl.float64" if _is_double_dtype(dtype) else "tl.float32"
    lines.append(f"    for group_{stage} in tl.range(0, {groups}):")
    indent = "        "

    for j in range(radix):
        lines.append(
            f"{indent}logical_phys{j} = tl.where(lane_mask, lane + "
            f"{current_lanes} * (group_{stage} * {radix} + {j}), 0)"
        )
        if smem_pack > 1:
            lines.append(f"{indent}phys{j} = logical_phys{j} + smem_offset")
        else:
            lines.append(f"{indent}phys{j} = logical_phys{j}")
        if fuse_twiddle_into_row and stage > 0:
            lines.append(
                f"{indent}smem_phys{j} = logical_phys{j} ^ "
                f"(logical_phys{j} >> {_TLE_SMEM_SWIZZLE_SHIFT})"
            )
            if smem_pack > 1:
                lines.append(f"{indent}smem_phys{j} += smem_offset")
    if stage == 0:
        lines.extend(_emit_input_base(indent, factors, current_lanes, f"group_{stage}"))
    if is_last:
        lines.extend(
            _emit_output_base(indent, factors, current_lanes, f"group_{stage}")
        )
    else:
        lines.extend(
            _emit_route_base(indent, stage, factors, current_lanes, f"group_{stage}")
        )

    for j in range(radix):
        if stage == 0:
            lines.extend(_emit_input_index(indent, f"in{j}", factors, j))
            if io_mode == "contiguous":
                if vectorized_complex_io:
                    lines.append(
                        f"{indent}r{j}, i{j} = tl.inline_asm_elementwise("
                        "'{\\n"
                        ".reg .pred p;\\n"
                        "setp.ne.b32 p, $3, 0;\\n"
                        f"@p ld.global.v2.{vector_suffix} {{$0, $1}}, [$2];\\n"
                        f"@!p mov.{vector_suffix} $0, 0.0;\\n"
                        f"@!p mov.{vector_suffix} $1, 0.0;\\n"
                        "}', \"=" + vector_reg + ",=" + vector_reg + ",l,r\", ["
                        f"tl.cast(in_ptr + (batch_base + in{j}) * 2, tl.uint64), "
                        "tl.cast(lane_mask, tl.int32)], "
                        f"dtype=({vector_dtype}, {vector_dtype}), is_pure=False, pack=1)"
                    )
                else:
                    lines.append(
                        f"{indent}r{j} = tl.load(in_ptr + (batch_base + in{j}) * 2, "
                        f"mask=lane_mask, other={zero})"
                    )
                    lines.append(
                        f"{indent}i{j} = tl.load(in_ptr + (batch_base + in{j}) * 2 + 1, "
                        f"mask=lane_mask, other={zero})"
                    )
            elif io_mode == "strided":
                lines.append(
                    f"{indent}r{j} = tl.load(in_ptr + (batch_base + in{j} * outer_stride) * 2, "
                    f"mask=lane_mask, other={zero})"
                )
                lines.append(
                    f"{indent}i{j} = tl.load(in_ptr + (batch_base + in{j} * outer_stride) * 2 + 1, "
                    f"mask=lane_mask, other={zero})"
                )
            elif io_mode == "contiguous_r2c":
                lines.append(
                    f"{indent}r{j} = tl.load(in_ptr + input_batch_base + in{j}, mask=lane_mask, other={zero})"
                )
                lines.append(f"{indent}i{j} = r{j} * 0.0")
            elif io_mode == "contiguous_c2r":
                half_n = n // 2 + 1
                nyquist_guard = f" | (in{j} == {n // 2})" if n % 2 == 0 else ""
                lines.append(
                    f"{indent}compact_idx{j} = tl.where(in{j} < {half_n}, in{j}, {n} - in{j})"
                )
                lines.append(
                    f"{indent}src_ptr{j} = in_ptr + (input_batch_base + compact_idx{j}) * 2"
                )
                if vectorized_complex_io:
                    lines.append(
                        f"{indent}r{j}, i{j} = tl.inline_asm_elementwise("
                        "'{\\n"
                        ".reg .pred p;\\n"
                        "setp.ne.b32 p, $3, 0;\\n"
                        f"@p ld.global.v2.{vector_suffix} {{$0, $1}}, [$2];\\n"
                        f"@!p mov.{vector_suffix} $0, 0.0;\\n"
                        f"@!p mov.{vector_suffix} $1, 0.0;\\n"
                        "}', \"=" + vector_reg + ",=" + vector_reg + ",l,r\", ["
                        f"tl.cast(src_ptr{j}, tl.uint64), "
                        "tl.cast(lane_mask, tl.int32)], "
                        f"dtype=({vector_dtype}, {vector_dtype}), is_pure=False, pack=1)"
                    )
                else:
                    lines.append(
                        f"{indent}r{j} = tl.load(src_ptr{j}, mask=lane_mask, other={zero})"
                    )
                    lines.append(
                        f"{indent}i{j} = tl.load(src_ptr{j} + 1, mask=lane_mask, other={zero})"
                    )
                lines.append(f"{indent}i{j} = tl.where(in{j} < {half_n}, i{j}, -i{j})")
                lines.append(
                    f"{indent}i{j} = tl.where((in{j} == 0){nyquist_guard}, 0.0, i{j})"
                )
            elif io_mode == "bluestein_prepare_leaf":
                lines.append(f"{indent}prime_mask{j} = lane_mask & (in{j} < {prime_n})")
                lines.append(
                    f"{indent}src_ptr{j} = in_ptr + (current_batch * {prime_n} + in{j}) * 2"
                )
                lines.append(
                    f"{indent}r{j} = tl.load(src_ptr{j}, mask=prime_mask{j}, other={zero})"
                )
                lines.append(
                    f"{indent}i{j} = tl.load(src_ptr{j} + 1, mask=prime_mask{j}, other={zero})"
                )
                lines.append(
                    f"{indent}chirp_r{j} = tl.load(chirp_ptr + in{j} * 2, mask=prime_mask{j}, other={zero})"
                )
                lines.append(
                    f"{indent}chirp_i{j} = tl.load(chirp_ptr + in{j} * 2 + 1, mask=prime_mask{j}, other={zero})"
                )
                lines.append(
                    f"{indent}r{j}, i{j} = _cmul(r{j}, i{j}, chirp_r{j}, chirp_i{j})"
                )
            elif io_mode == "bluestein_finish_leaf":
                lines.append(
                    f"{indent}ar{j} = tl.load(in_ptr + (batch_base + in{j}) * 2, mask=lane_mask, other={zero})"
                )
                lines.append(
                    f"{indent}ai{j} = tl.load(in_ptr + (batch_base + in{j}) * 2 + 1, mask=lane_mask, other={zero})"
                )
                lines.append(
                    f"{indent}br{j} = tl.load(b_fft_ptr + in{j} * 2, mask=lane_mask, other={zero})"
                )
                lines.append(
                    f"{indent}bi{j} = tl.load(b_fft_ptr + in{j} * 2 + 1, mask=lane_mask, other={zero})"
                )
                lines.append(
                    f"{indent}r{j}, point_i{j} = _cmul(ar{j}, ai{j}, br{j}, bi{j})"
                )
                lines.append(f"{indent}i{j} = -point_i{j}")
            elif io_mode == "bluestein_full_leaf":
                if bluestein_pass == 0:
                    lines.append(
                        f"{indent}prime_mask{j} = lane_mask & (in{j} < {prime_n})"
                    )
                    lines.append(
                        f"{indent}src_ptr{j} = in_ptr + (current_batch * {prime_n} + in{j}) * 2"
                    )
                    lines.append(
                        f"{indent}r{j} = tl.load(src_ptr{j}, mask=prime_mask{j}, other={zero})"
                    )
                    lines.append(
                        f"{indent}i{j} = tl.load(src_ptr{j} + 1, mask=prime_mask{j}, other={zero})"
                    )
                    lines.append(
                        f"{indent}chirp_r{j} = tl.load(chirp_ptr + in{j} * 2, mask=prime_mask{j}, other={zero})"
                    )
                    lines.append(
                        f"{indent}chirp_i{j} = tl.load(chirp_ptr + in{j} * 2 + 1, mask=prime_mask{j}, other={zero})"
                    )
                    lines.append(
                        f"{indent}r{j}, i{j} = _cmul(r{j}, i{j}, chirp_r{j}, chirp_i{j})"
                    )
                else:
                    intermediate_index = f"in{j}"
                    if smem_pack > 1:
                        lines.append(
                            f"{indent}intermediate_in{j} = in{j} + smem_offset"
                        )
                        intermediate_index = f"intermediate_in{j}"
                    lines.append(
                        f"{indent}r{j} = tl.load(tle.gpu.local_ptr({bluestein_intermediate_buffer}_r, "
                        f"({intermediate_index},)), mask=lane_mask, other={zero})"
                    )
                    lines.append(
                        f"{indent}i{j} = tl.load(tle.gpu.local_ptr({bluestein_intermediate_buffer}_i, "
                        f"({intermediate_index},)), mask=lane_mask, other={zero})"
                    )
            elif io_mode == "bluestein_four_step_prepare_row":
                lines.append(
                    f"{indent}src_idx{j} = in{j} * {four_step_n2} + four_step_inner"
                )
                lines.append(
                    f"{indent}prime_mask{j} = lane_mask & (src_idx{j} < {prime_n})"
                )
                lines.append(
                    f"{indent}src_ptr{j} = in_ptr + (four_step_batch * {prime_n} + src_idx{j}) * 2"
                )
                lines.append(
                    f"{indent}r{j} = tl.load(src_ptr{j}, mask=prime_mask{j}, other={zero})"
                )
                lines.append(
                    f"{indent}i{j} = tl.load(src_ptr{j} + 1, mask=prime_mask{j}, other={zero})"
                )
                lines.append(
                    f"{indent}chirp_r{j} = tl.load(chirp_ptr + src_idx{j} * 2, mask=prime_mask{j}, other={zero})"
                )
                lines.append(
                    f"{indent}chirp_i{j} = tl.load(chirp_ptr + src_idx{j} * 2 + 1, mask=prime_mask{j}, other={zero})"
                )
                lines.append(
                    f"{indent}r{j}, i{j} = _cmul(r{j}, i{j}, chirp_r{j}, chirp_i{j})"
                )
            elif io_mode == "bluestein_four_step_pointwise_row":
                lines.append(
                    f"{indent}src_idx{j} = in{j} * {four_step_n2} + four_step_inner"
                )
                lines.append(
                    f"{indent}ar{j} = tl.load(in_ptr + (four_step_batch_base + src_idx{j}) * 2, "
                    f"mask=lane_mask, other={zero})"
                )
                lines.append(
                    f"{indent}ai{j} = tl.load(in_ptr + (four_step_batch_base + src_idx{j}) * 2 + 1, "
                    f"mask=lane_mask, other={zero})"
                )
                lines.append(
                    f"{indent}br{j} = tl.load(b_fft_ptr + src_idx{j} * 2, mask=lane_mask, other={zero})"
                )
                lines.append(
                    f"{indent}bi{j} = tl.load(b_fft_ptr + src_idx{j} * 2 + 1, mask=lane_mask, other={zero})"
                )
                lines.append(
                    f"{indent}r{j}, point_i{j} = _cmul(ar{j}, ai{j}, br{j}, bi{j})"
                )
                lines.append(f"{indent}i{j} = -point_i{j}")
            elif io_mode == "four_step_row":
                lines.append(
                    f"{indent}src_idx{j} = in{j} * {four_step_n2} + four_step_inner"
                )
                lines.append(
                    f"{indent}r{j} = tl.load(in_ptr + (four_step_batch_base + src_idx{j}) * 2, "
                    f"mask=lane_mask, other={zero})"
                )
                lines.append(
                    f"{indent}i{j} = tl.load(in_ptr + (four_step_batch_base + src_idx{j}) * 2 + 1, "
                    f"mask=lane_mask, other={zero})"
                )
            elif io_mode == "four_step_row_strided":
                lines.append(
                    f"{indent}src_idx{j} = in{j} * {four_step_n2} + four_step_inner"
                )
                lines.append(
                    f"{indent}r{j} = tl.load(in_ptr + "
                    f"(four_step_batch_base + src_idx{j} * outer_stride) * 2, "
                    f"mask=lane_mask, other={zero})"
                )
                lines.append(
                    f"{indent}i{j} = tl.load(in_ptr + "
                    f"(four_step_batch_base + src_idx{j} * outer_stride) * 2 + 1, "
                    f"mask=lane_mask, other={zero})"
                )
            elif io_mode == "four_step_real_row":
                lines.append(
                    f"{indent}src_idx{j} = in{j} * {four_step_n2} + four_step_inner"
                )
                lines.append(
                    f"{indent}r{j} = tl.load(in_ptr + four_step_batch * input_distance + src_idx{j}, "
                    f"mask=lane_mask, other={zero})"
                )
                lines.append(f"{indent}i{j} = r{j} * 0.0")
            elif io_mode == "four_step_hermitian_row":
                full_n = four_step_n1 * four_step_n2
                half_n = full_n // 2 + 1
                nyquist_guard = (
                    f" | (src_idx{j} == {full_n // 2})" if full_n % 2 == 0 else ""
                )
                lines.append(
                    f"{indent}src_idx{j} = in{j} * {four_step_n2} + four_step_inner"
                )
                lines.append(
                    f"{indent}compact_idx{j} = tl.where(src_idx{j} < {half_n}, src_idx{j}, {full_n} - src_idx{j})"
                )
                lines.append(
                    f"{indent}src_ptr{j} = in_ptr + (four_step_batch * input_distance + compact_idx{j}) * 2"
                )
                lines.append(
                    f"{indent}r{j} = tl.load(src_ptr{j}, mask=lane_mask, other={zero})"
                )
                lines.append(
                    f"{indent}i{j} = tl.load(src_ptr{j} + 1, mask=lane_mask, other={zero})"
                )
                lines.append(
                    f"{indent}i{j} = tl.where(src_idx{j} < {half_n}, i{j}, -i{j})"
                )
                lines.append(
                    f"{indent}i{j} = tl.where((src_idx{j} == 0){nyquist_guard}, 0.0, i{j})"
                )
            elif io_mode == "four_step_col_strided":
                lines.append(
                    f"{indent}src_idx{j} = in{j} * {four_step_n1} + four_step_inner"
                )
                lines.append(
                    f"{indent}r{j} = tl.load(in_ptr + "
                    f"(four_step_batch_base + src_idx{j} * outer_stride) * 2, "
                    f"mask=lane_mask, other={zero})"
                )
                lines.append(
                    f"{indent}i{j} = tl.load(in_ptr + "
                    f"(four_step_batch_base + src_idx{j} * outer_stride) * 2 + 1, "
                    f"mask=lane_mask, other={zero})"
                )
                lines.append(
                    f"{indent}tw_r{j} = tl.load(twiddle_ptr + src_idx{j} * 2, "
                    f"mask=lane_mask, other={zero})"
                )
                lines.append(
                    f"{indent}tw_i{j} = tl.load(twiddle_ptr + src_idx{j} * 2 + 1, "
                    f"mask=lane_mask, other={zero})"
                )
                lines.append(
                    f"{indent}r{j}, i{j} = _cmul(r{j}, i{j}, tw_r{j}, tw_i{j})"
                )
            else:
                lines.append(
                    f"{indent}src_idx{j} = in{j} * {four_step_n1} + four_step_inner"
                )
                if fuse_twiddle_into_row:
                    lines.append(
                        f"{indent}r{j} = tl.load(in_ptr + (four_step_batch_base + src_idx{j}) * 2, "
                        f"mask=lane_mask, other={zero})"
                    )
                    lines.append(
                        f"{indent}i{j} = tl.load(in_ptr + (four_step_batch_base + src_idx{j}) * 2 + 1, "
                        f"mask=lane_mask, other={zero})"
                    )
                else:
                    lines.append(
                        f"{indent}r{j} = tl.load(in_ptr + (four_step_batch_base + src_idx{j}) * 2, "
                        f"mask=lane_mask, other={zero})"
                    )
                    lines.append(
                        f"{indent}i{j} = tl.load(in_ptr + (four_step_batch_base + src_idx{j}) * 2 + 1, "
                        f"mask=lane_mask, other={zero})"
                    )
                    lines.append(
                        f"{indent}tw_r{j} = tl.load(twiddle_ptr + src_idx{j} * 2, mask=lane_mask, other={zero})"
                    )
                    lines.append(
                        f"{indent}tw_i{j} = tl.load(twiddle_ptr + src_idx{j} * 2 + 1, mask=lane_mask, other={zero})"
                    )
                    lines.append(
                        f"{indent}r{j}, i{j} = _cmul(r{j}, i{j}, tw_r{j}, tw_i{j})"
                    )
        else:
            load_index = f"smem_phys{j}" if fuse_twiddle_into_row else f"phys{j}"
            lines.append(
                f"{indent}r{j} = tl.load(tle.gpu.local_ptr({source_buffer}_r, ({load_index},)), "
                f"mask=lane_mask, other={zero})"
            )
            lines.append(
                f"{indent}i{j} = tl.load(tle.gpu.local_ptr({source_buffer}_i, ({load_index},)), "
                f"mask=lane_mask, other={zero})"
            )
            lines.append(
                f"{indent}twr = tl.load(tw{stage}_r_ptr + logical_phys{j}, mask=lane_mask, other={zero})"
            )
            lines.append(
                f"{indent}twi = tl.load(tw{stage}_i_ptr + logical_phys{j}, mask=lane_mask, other={zero})"
            )
            lines.append(f"{indent}r{j}, i{j} = _cmul(r{j}, i{j}, twr, twi)")

    if single_smem_buffer and stage > 0 and not is_last:
        lines.append(f"{indent}tl.debug_barrier()")

    if radix == 16:
        lines.extend(_emit_radix16_codelet_call(indent, direction))
    elif radix == 32:
        lines.extend(_emit_natural_order_radix32_codelet_call(indent, direction))
    elif radix in _THREAD_LOCAL_MIXED_RADICES:
        lines.extend(_emit_local_mixed_codelet_call(indent, radix, direction))
    elif radix in _NATURAL_ORDER_CODELET_RADICES:
        lines.extend(_emit_natural_order_codelet_call(indent, radix, direction))
    else:
        lines.extend(_emit_table_codelet(indent, radix, lane_block, dtype))

    for j in range(radix):
        if is_last:
            lines.extend(_emit_output_index(indent, f"out_idx{j}", factors, j))
            if io_mode in {"contiguous", "strided", "bluestein_prepare_leaf"}:
                if io_mode == "strided":
                    lines.append(
                        f"{indent}tl.store(out_ptr + (batch_base + out_idx{j} * outer_stride) * 2, "
                        f"r{j}, mask=lane_mask)"
                    )
                    lines.append(
                        f"{indent}tl.store(out_ptr + (batch_base + out_idx{j} * outer_stride) * 2 + 1, "
                        f"i{j}, mask=lane_mask)"
                    )
                else:
                    if vectorized_complex_io:
                        lines.append(
                            f"{indent}tl.inline_asm_elementwise("
                            "'{\\n"
                            ".reg .pred p;\\n"
                            "setp.ne.b32 p, $4, 0;\\n"
                            f"@p st.global.v2.{vector_suffix} [$1], {{$2, $3}};\\n"
                            "mov.u32 $0, 0;\\n"
                            "}', \"=r,l," + vector_reg + "," + vector_reg + ",r\", ["
                            f"tl.cast(out_ptr + (batch_base + out_idx{j}) * 2, tl.uint64), "
                            f"r{j}, i{j}, tl.cast(lane_mask, tl.int32)], "
                            "dtype=tl.int32, is_pure=False, pack=1)"
                        )
                    else:
                        lines.append(
                            f"{indent}tl.store(out_ptr + (batch_base + out_idx{j}) * 2, "
                            f"r{j}, mask=lane_mask)"
                        )
                        lines.append(
                            f"{indent}tl.store(out_ptr + (batch_base + out_idx{j}) * 2 + 1, "
                            f"i{j}, mask=lane_mask)"
                        )
            elif io_mode == "contiguous_r2c":
                lines.append(
                    f"{indent}compact_mask{j} = lane_mask & (out_idx{j} < {n // 2 + 1})"
                )
                lines.append(
                    f"{indent}dst_ptr{j} = out_ptr + (output_batch_base + out_idx{j}) * 2"
                )
                lines.append(
                    f"{indent}tl.store(dst_ptr{j}, r{j}, mask=compact_mask{j})"
                )
                lines.append(
                    f"{indent}tl.store(dst_ptr{j} + 1, i{j}, mask=compact_mask{j})"
                )
            elif io_mode == "contiguous_c2r":
                lines.append(
                    f"{indent}tl.store(out_ptr + output_batch_base + out_idx{j}, r{j}, mask=lane_mask)"
                )
            elif io_mode == "bluestein_finish_leaf":
                lines.append(
                    f"{indent}prime_mask{j} = lane_mask & (out_idx{j} < {prime_n})"
                )
                lines.append(f"{indent}scaled_r{j} = r{j} / {n}")
                lines.append(f"{indent}scaled_i{j} = -i{j} / {n}")
                lines.append(
                    f"{indent}chirp_r{j} = tl.load(chirp_ptr + out_idx{j} * 2, mask=prime_mask{j}, other={zero})"
                )
                lines.append(
                    f"{indent}chirp_i{j} = tl.load(chirp_ptr + out_idx{j} * 2 + 1, mask=prime_mask{j}, other={zero})"
                )
                lines.append(
                    f"{indent}final_r{j}, final_i{j} = _cmul(scaled_r{j}, scaled_i{j}, chirp_r{j}, chirp_i{j})"
                )
                lines.append(
                    f"{indent}dst_ptr{j} = out_ptr + (current_batch * {prime_n} + out_idx{j}) * 2"
                )
                lines.append(
                    f"{indent}tl.store(dst_ptr{j}, final_r{j}, mask=prime_mask{j})"
                )
                lines.append(
                    f"{indent}tl.store(dst_ptr{j} + 1, final_i{j}, mask=prime_mask{j})"
                )
            elif io_mode == "bluestein_full_leaf":
                if bluestein_pass == 0:
                    lines.append(
                        f"{indent}br{j} = tl.load(b_fft_ptr + out_idx{j} * 2, mask=lane_mask, other={zero})"
                    )
                    lines.append(
                        f"{indent}bi{j} = tl.load(b_fft_ptr + out_idx{j} * 2 + 1, mask=lane_mask, other={zero})"
                    )
                    lines.append(
                        f"{indent}point_r{j}, point_i{j} = _cmul(r{j}, i{j}, br{j}, bi{j})"
                    )
                    intermediate_index = f"out_idx{j}"
                    if smem_pack > 1:
                        lines.append(
                            f"{indent}intermediate_out{j} = out_idx{j} + smem_offset"
                        )
                        intermediate_index = f"intermediate_out{j}"
                    lines.append(
                        f"{indent}tl.store(tle.gpu.local_ptr({bluestein_intermediate_buffer}_r, "
                        f"({intermediate_index},)), point_r{j}, mask=lane_mask)"
                    )
                    lines.append(
                        f"{indent}tl.store(tle.gpu.local_ptr({bluestein_intermediate_buffer}_i, "
                        f"({intermediate_index},)), -point_i{j}, mask=lane_mask)"
                    )
                else:
                    lines.append(
                        f"{indent}prime_mask{j} = lane_mask & (out_idx{j} < {prime_n})"
                    )
                    lines.append(f"{indent}scaled_r{j} = r{j} / {n}")
                    lines.append(f"{indent}scaled_i{j} = -i{j} / {n}")
                    lines.append(
                        f"{indent}chirp_r{j} = tl.load(chirp_ptr + out_idx{j} * 2, mask=prime_mask{j}, other={zero})"
                    )
                    lines.append(
                        f"{indent}chirp_i{j} = tl.load(chirp_ptr + out_idx{j} * 2 +1, mask=prime_mask{j}, other={zero})"
                    )
                    lines.append(
                        f"{indent}final_r{j}, final_i{j} = _cmul(scaled_r{j}, scaled_i{j}, chirp_r{j}, chirp_i{j})"
                    )
                    lines.append(
                        f"{indent}dst_ptr{j} = out_ptr + (current_batch * {prime_n} + out_idx{j}) * 2"
                    )
                    lines.append(
                        f"{indent}tl.store(dst_ptr{j}, final_r{j}, mask=prime_mask{j})"
                    )
                    lines.append(
                        f"{indent}tl.store(dst_ptr{j} + 1, final_i{j}, mask=prime_mask{j})"
                    )
            elif io_mode == "bluestein_four_step_finish_col":
                lines.append(
                    f"{indent}dst_idx{j} = out_idx{j} * {four_step_n1} + four_step_inner"
                )
                lines.append(
                    f"{indent}prime_mask{j} = lane_mask & (dst_idx{j} < {prime_n})"
                )
                lines.append(
                    f"{indent}scaled_r{j} = r{j} / {four_step_n1 * four_step_n2}"
                )
                lines.append(
                    f"{indent}scaled_i{j} = -i{j} / {four_step_n1 * four_step_n2}"
                )
                lines.append(
                    f"{indent}chirp_r{j} = tl.load(chirp_ptr + dst_idx{j} * 2, mask=prime_mask{j}, other={zero})"
                )
                lines.append(
                    f"{indent}chirp_i{j} = tl.load(chirp_ptr + dst_idx{j} * 2 + 1, mask=prime_mask{j}, other={zero})"
                )
                lines.append(
                    f"{indent}final_r{j}, final_i{j} = _cmul(scaled_r{j}, scaled_i{j}, chirp_r{j}, chirp_i{j})"
                )
                lines.append(
                    f"{indent}dst_ptr{j} = out_ptr + (four_step_batch * {prime_n} + dst_idx{j}) * 2"
                )
                lines.append(
                    f"{indent}tl.store(dst_ptr{j}, final_r{j}, mask=prime_mask{j})"
                )
                lines.append(
                    f"{indent}tl.store(dst_ptr{j} + 1, final_i{j}, mask=prime_mask{j})"
                )
            elif io_mode == "four_step_row_strided":
                lines.append(
                    f"{indent}dst_idx{j} = four_step_inner * {four_step_n1} + out_idx{j}"
                )
                lines.append(
                    f"{indent}tl.store(out_ptr + "
                    f"(four_step_batch_base + dst_idx{j} * outer_stride) * 2, "
                    f"r{j}, mask=lane_mask)"
                )
                lines.append(
                    f"{indent}tl.store(out_ptr + "
                    f"(four_step_batch_base + dst_idx{j} * outer_stride) * 2 + 1, "
                    f"i{j}, mask=lane_mask)"
                )
            elif io_mode in {
                "four_step_row",
                "four_step_real_row",
                "four_step_hermitian_row",
                "bluestein_four_step_prepare_row",
                "bluestein_four_step_pointwise_row",
            }:
                lines.append(
                    f"{indent}dst_idx{j} = four_step_inner * {four_step_n1} + out_idx{j}"
                )
                if fuse_twiddle_into_row:
                    if _is_double_dtype(dtype):
                        lines.append(
                            f"{indent}tw_r{j} = tl.load(twiddle_ptr + dst_idx{j} * 2, "
                            f"mask=lane_mask, other={zero})"
                        )
                        lines.append(
                            f"{indent}tw_i{j} = tl.load(twiddle_ptr + dst_idx{j} * 2 + 1, "
                            f"mask=lane_mask, other={zero})"
                        )
                    else:
                        outer_twiddle_scale = (
                            _direction_sign(direction)
                            * 2.0
                            * math.pi
                            / (four_step_n1 * four_step_n2)
                        )
                        lines.append(
                            f"{indent}outer_angle{j} = four_step_inner * out_idx{j} * "
                            f"{outer_twiddle_scale:.17g}"
                        )
                        lines.append(
                            f"{indent}tw_i{j}, tw_r{j} = "
                            "tl.inline_asm_elementwise("
                            '"sin.approx.f32 $0, $2; cos.approx.f32 $1, $2;", '
                            f'"=f,=f,f", [outer_angle{j}], '
                            "dtype=(tl.float32, tl.float32), "
                            "is_pure=True, pack=1)"
                        )
                    lines.append(
                        f"{indent}r{j}, i{j} = _cmul(r{j}, i{j}, tw_r{j}, tw_i{j})"
                    )
                lines.append(
                    f"{indent}tl.store(out_ptr + (four_step_batch_base + dst_idx{j}) * 2, r{j}, mask=lane_mask)"
                )
                lines.append(
                    f"{indent}tl.store(out_ptr + (four_step_batch_base + dst_idx{j}) * 2 + 1, i{j}, mask=lane_mask)"
                )
            elif io_mode == "four_step_r2c_col":
                lines.append(
                    f"{indent}dst_idx{j} = out_idx{j} * {four_step_n1} + four_step_inner"
                )
                lines.append(
                    f"{indent}compact_mask{j} = lane_mask & (dst_idx{j} < {four_step_n1 * four_step_n2 // 2 + 1})"
                )
                lines.append(
                    f"{indent}dst_ptr{j} = out_ptr + (four_step_batch * output_distance + dst_idx{j}) * 2"
                )
                lines.append(
                    f"{indent}tl.store(dst_ptr{j}, r{j}, mask=compact_mask{j})"
                )
                lines.append(
                    f"{indent}tl.store(dst_ptr{j} + 1, i{j}, mask=compact_mask{j})"
                )
            elif io_mode == "four_step_c2r_col":
                lines.append(
                    f"{indent}dst_idx{j} = out_idx{j} * {four_step_n1} + four_step_inner"
                )
                lines.append(
                    f"{indent}dst_ptr{j} = out_ptr + four_step_batch * output_distance + dst_idx{j}"
                )
                lines.append(f"{indent}tl.store(dst_ptr{j}, r{j}, mask=lane_mask)")
            elif io_mode == "four_step_col_strided":
                lines.append(
                    f"{indent}dst_idx{j} = out_idx{j} * {four_step_n1} + four_step_inner"
                )
                lines.append(
                    f"{indent}tl.store(out_ptr + "
                    f"(four_step_batch_base + dst_idx{j} * outer_stride) * 2, "
                    f"r{j}, mask=lane_mask)"
                )
                lines.append(
                    f"{indent}tl.store(out_ptr + "
                    f"(four_step_batch_base + dst_idx{j} * outer_stride) * 2 + 1, "
                    f"i{j}, mask=lane_mask)"
                )
            else:
                lines.append(
                    f"{indent}dst_idx{j} = out_idx{j} * {four_step_n1} + four_step_inner"
                )
                lines.append(
                    f"{indent}tl.store(out_ptr + (four_step_batch_base + dst_idx{j}) * 2, r{j}, mask=lane_mask)"
                )
                lines.append(
                    f"{indent}tl.store(out_ptr + (four_step_batch_base + dst_idx{j}) * 2 + 1, i{j}, mask=lane_mask)"
                )
        else:
            lines.extend(
                _emit_route_index(indent, f"dst{j}", stage, factors, next_lanes, j)
            )
            store_index = f"dst{j}"
            if fuse_twiddle_into_row:
                lines.append(
                    f"{indent}smem_dst{j} = dst{j} ^ "
                    f"(dst{j} >> {_TLE_SMEM_SWIZZLE_SHIFT})"
                )
                if smem_pack > 1:
                    lines.append(f"{indent}smem_dst{j} += smem_offset")
                store_index = f"smem_dst{j}"
            elif smem_pack > 1:
                lines.append(f"{indent}smem_dst{j} = dst{j} + smem_offset")
                store_index = f"smem_dst{j}"
            lines.append(
                f"{indent}tl.store(tle.gpu.local_ptr({dest_buffer}_r, ({store_index},)), r{j}, mask=lane_mask)"
            )
            lines.append(
                f"{indent}tl.store(tle.gpu.local_ptr({dest_buffer}_i, ({store_index},)), i{j}, mask=lane_mask)"
            )

    if not is_last:
        lines.append("    tl.debug_barrier()")
    return lines


def _leaf_kernel_params(
    plan: LeafPlan, *, include_four_step_twiddle: bool = False
) -> list[str]:
    factors = plan.factors
    generic_radices = plan.generic_radices
    params = ["in_ptr"]
    if include_four_step_twiddle:
        params.append("twiddle_ptr")
    params.append("out_ptr")
    for stage in range(1, len(factors)):
        params.append(f"tw{stage}_r_ptr")
        params.append(f"tw{stage}_i_ptr")
    for radix in generic_radices:
        params.append(f"dft{radix}_r_ptr")
        params.append(f"dft{radix}_i_ptr")
    return params


def _leaf_kernel_params_for_io(
    plan: LeafPlan,
    *,
    io_mode: LeafIoMode,
    include_four_step_twiddle: bool = False,
) -> list[str]:
    params = _leaf_kernel_params(
        plan, include_four_step_twiddle=include_four_step_twiddle
    )
    if io_mode == "strided":
        params.append("outer_stride")
    if io_mode == "bluestein_prepare_leaf":
        params.insert(1, "chirp_ptr")
    elif io_mode == "bluestein_finish_leaf":
        params[1:1] = ["b_fft_ptr", "chirp_ptr"]
    elif io_mode == "bluestein_full_leaf":
        params[1:1] = ["b_fft_ptr", "chirp_ptr"]
    elif io_mode in {
        "bluestein_four_step_prepare_row",
        "bluestein_four_step_finish_col",
    }:
        params.insert(1, "chirp_ptr")
    elif io_mode == "bluestein_four_step_pointwise_row":
        params.insert(1, "b_fft_ptr")
    if io_mode in {
        "contiguous_r2c",
        "contiguous_c2r",
        "four_step_real_row",
        "four_step_hermitian_row",
    }:
        params.append("input_distance")
    if io_mode in {
        "contiguous_r2c",
        "contiguous_c2r",
        "four_step_r2c_col",
        "four_step_c2r_col",
    }:
        params.append("output_distance")
    if io_mode in {"four_step_row_strided", "four_step_col_strided"}:
        params.append("outer_stride")
    params.append("nbatch")
    return params

def _use_thread_local_mixed_leaf(
    plan: LeafPlan,
    *,
    io_mode: LeafIoMode,
    four_step_n1: int,
    four_step_n2: int,
) -> bool:
    if _non_nvidia_backend_active():
        return False
    if io_mode.endswith("_strided"):
        return False
    if io_mode.startswith("bluestein_four_step_"):
        return False
    if len(plan.factors) != 2:
        return False
    register_radix, cross_radix = plan.factors
    row_modes = {
        "four_step_row",
        "four_step_real_row",
        "four_step_hermitian_row",
    }
    col_modes = {
        "four_step_col",
        "four_step_r2c_col",
        "four_step_c2r_col",
    }
    expected_length = four_step_n1 if io_mode in row_modes else four_step_n2
    return (
        io_mode in row_modes | col_modes
        and plan.dtype in {"complex64", "complex128"}
        and register_radix in _THREAD_LOCAL_MIXED_RADICES
        and cross_radix == 32
        and plan.length == register_radix * cross_radix
        and plan.length == expected_length
        and (
            use_tle_fused_twiddle(four_step_n1, four_step_n2, plan.dtype)
            or use_four_step_row_fused_twiddle(four_step_n1, four_step_n2, plan.dtype)
        )
    )


def _distributed_join_tree(names: list[str]) -> str:
    if len(names) == 1:
        return names[0]
    if len(names) & (len(names) - 1):
        raise ValueError("distributed join tree requires a power-of-two input count")
    return (
        f"tl.join({_distributed_join_tree(names[0::2])}, "
        f"{_distributed_join_tree(names[1::2])})"
    )


def _emit_distributed_split_tree(
    indent: str,
    source: str,
    names: list[str],
    prefix: str,
) -> list[str]:
    if len(names) == 1:
        return [f"{indent}{names[0]} = {source}"]

    even_names = names[0::2]
    odd_names = names[1::2]
    even_source = (
        even_names[0] if len(even_names) == 1 else f"{prefix}_even{len(names)}"
    )
    odd_source = odd_names[0] if len(odd_names) == 1 else f"{prefix}_odd{len(names)}"
    lines = [f"{indent}{even_source}, {odd_source} = tl.split({source})"]
    if len(even_names) > 1:
        lines.extend(
            _emit_distributed_split_tree(indent, even_source, even_names, f"{prefix}_e")
        )
    if len(odd_names) > 1:
        lines.extend(
            _emit_distributed_split_tree(indent, odd_source, odd_names, f"{prefix}_o")
        )
    return lines


def _build_thread_local_mixed_four_step_kernel_source(
    plan: LeafPlan,
    *,
    io_mode: LeafIoMode,
    four_step_n1: int,
    four_step_n2: int,
) -> tuple[str, str]:
    # Each thread owns the first composite register FFT, followed by one
    # shared exchange and a full register-only radix-32 FFT.
    register_radix = plan.factors[0]
    inner_pack = 4
    physical_lanes = 32
    vector_block = physical_lanes * inner_pack
    smem_chunk = next(chunk for chunk in (8, 4, 2, 1) if register_radix % chunk == 0)
    smem_chunk_dims = int(math.log2(smem_chunk))
    smem_reshape_dims = ", ".join(["1"] * smem_chunk_dims)
    smem_block_dims = ", ".join(["2"] * smem_chunk_dims)
    smem_n = plan.smem_size * inner_pack
    vector_suffix = "f64" if _is_double_dtype(plan.dtype) else "f32"
    vector_reg = "d" if _is_double_dtype(plan.dtype) else "f"
    vector_dtype = "tl.float64" if _is_double_dtype(plan.dtype) else "tl.float32"
    asm_load_constraints = f'"={vector_reg},={vector_reg},l"'
    asm_store_constraints = f'"=r,l,{vector_reg},{vector_reg}"'
    row_modes = {
        "four_step_row",
        "four_step_real_row",
        "four_step_hermitian_row",
    }
    col_modes = {
        "four_step_col",
        "four_step_r2c_col",
        "four_step_c2r_col",
    }
    if io_mode not in row_modes | col_modes:
        raise ValueError(f"unsupported thread-local four-step I/O mode {io_mode}")
    include_outer_twiddle = io_mode in row_modes
    inner_count = four_step_n2 if io_mode in row_modes else four_step_n1
    source_stride = four_step_n2 if io_mode in row_modes else four_step_n1
    params = _leaf_kernel_params_for_io(
        plan,
        io_mode=io_mode,
        include_four_step_twiddle=include_outer_twiddle,
    )
    kernel_prefix = "ifft" if plan.direction == "inverse" else "fft"
    kernel_name = (
        f"{io_mode}_{kernel_prefix}_kernel_{register_radix}_32_thread_local"
        f"_n{four_step_n1}_{four_step_n2}_l{plan.lanes}_b32_t{smem_chunk}"
        "_v5g_itwsincos_otwrec_nw4"
    )

    body: list[str] = ["@triton.jit", f"def {kernel_name}("]
    for idx, param in enumerate(params):
        suffix = "," if idx < len(params) - 1 else ""
        body.append(f"    {param}{suffix}")
    body.extend(
        [
            "):",
            f"    four_step_inner_base = tl.program_id(0) * {inner_pack}",
            "    four_step_batch = tl.program_id(1)",
            "    if four_step_batch >= nbatch:",
            "        return",
            f"    lane_vec = tl.arange(0, {vector_block})",
            f"    inner_slot = lane_vec % {inner_pack}",
            f"    fft_thread = lane_vec // {inner_pack}",
            "    four_step_inner = four_step_inner_base + inner_slot",
            f"    lane_mask = four_step_inner < {inner_count}",
            f"    output_lane_mask = lane_mask & (fft_thread < {register_radix})",
            f"    smem_offset = inner_slot * {plan.smem_size}",
            (
                f"    four_step_batch_base = "
                f"four_step_batch * {four_step_n1 * four_step_n2}"
            ),
            (
                f"    smem_r = tle.gpu.alloc([{smem_n}], dtype={vector_dtype}, "
                "layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)"
            ),
            (
                f"    smem_i = tle.gpu.alloc([{smem_n}], dtype={vector_dtype}, "
                "layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)"
            ),
        ]
    )

    for idx in range(register_radix):
        input_digit = 2 * (idx % 16) + idx // 16 if register_radix == 32 else idx
        body.append(f"    input_idx{idx} = fft_thread + {32 * input_digit}")
        body.append(
            f"    src_idx{idx} = input_idx{idx} * {source_stride} + " "four_step_inner"
        )
        if io_mode == "four_step_real_row":
            body.append(
                f"    input_offset{idx} = "
                f"four_step_batch * input_distance + src_idx{idx}"
            )
            body.append(f"    r{idx} = tl.load(in_ptr + input_offset{idx})")
            body.append(f"    i{idx} = r{idx} * 0.0")
        elif io_mode == "four_step_hermitian_row":
            full_n = four_step_n1 * four_step_n2
            half_n = full_n // 2 + 1
            nyquist_guard = (
                f" | (src_idx{idx} == {full_n // 2})" if full_n % 2 == 0 else ""
            )
            body.append(
                f"    compact_idx{idx} = "
                f"tl.where(src_idx{idx} < {half_n}, src_idx{idx}, {full_n} - src_idx{idx})"
            )
            body.append(
                f"    input_offset{idx} = "
                f"(four_step_batch * input_distance + compact_idx{idx}) * 2"
            )
            body.append(
                f"    r{idx}, i{idx} = tl.inline_asm_elementwise("
                f'"ld.global.v2.{vector_suffix} {{$0, $1}}, [$2];", '
                f"{asm_load_constraints}, "
                f"[tl.cast(in_ptr + input_offset{idx}, tl.uint64)], "
                f"dtype=({vector_dtype}, {vector_dtype}), is_pure=False, pack=1)"
            )
            body.append(
                f"    i{idx} = tl.where(src_idx{idx} < {half_n}, i{idx}, -i{idx})"
            )
            body.append(
                f"    i{idx} = "
                f"tl.where((src_idx{idx} == 0){nyquist_guard}, 0.0, i{idx})"
            )
        else:
            body.append(
                f"    input_offset{idx} = " f"(four_step_batch_base + src_idx{idx}) * 2"
            )
            body.append(
                # The selected leaves and pack=4 cover every input lane exactly.
                f"    r{idx}, i{idx} = tl.inline_asm_elementwise("
                f'"ld.global.v2.{vector_suffix} {{$0, $1}}, [$2];", '
                f"{asm_load_constraints}, "
                f"[tl.cast(in_ptr + input_offset{idx}, tl.uint64)], "
                f"dtype=({vector_dtype}, {vector_dtype}), is_pure=False, pack=1)"
            )

    body.extend(_emit_local_mixed_codelet_call("    ", register_radix, plan.direction))

    body.append(
        f"    smem_store_mask = tl.broadcast_to("
        f"tl.reshape(lane_mask, {vector_block}, {smem_reshape_dims}), "
        f"{vector_block}, {smem_block_dims})"
    )
    body.append(
        f"    smem_load_mask = tl.broadcast_to("
        f"tl.reshape(output_lane_mask, {vector_block}, {smem_reshape_dims}), "
        f"{vector_block}, {smem_block_dims})"
    )
    for chunk_base in range(0, register_radix, smem_chunk):
        chunk_indices = range(chunk_base, chunk_base + smem_chunk)
        for idx in chunk_indices:
            if idx > 0:
                body.extend(
                    [
                        f"    inner_tw_idx{idx} = {idx} + {register_radix} * fft_thread",
                        (
                            f"    inner_tw_r{idx} = "
                            f"tl.load(tw1_r_ptr + inner_tw_idx{idx})"
                        ),
                        (
                            f"    inner_tw_i{idx} = "
                            f"tl.load(tw1_i_ptr + inner_tw_idx{idx})"
                        ),
                        (
                            f"    r{idx}, i{idx} = "
                            f"_cmul(r{idx}, i{idx}, "
                            f"inner_tw_r{idx}, inner_tw_i{idx})"
                        ),
                    ]
                )
            body.append(f"    smem_logical{idx} = {idx * 32} + fft_thread")
            body.append(
                f"    smem_phys{idx} = smem_logical{idx} ^ "
                f"(smem_logical{idx} >> {_TLE_SMEM_SWIZZLE_SHIFT})"
            )
            body.append(f"    smem_phys{idx} += smem_offset")
        body.extend(
            [
                (
                    f"    smem_store_index_{chunk_base} = "
                    f"{_distributed_join_tree([f'smem_phys{idx}' for idx in chunk_indices])}"
                ),
                (
                    f"    smem_store_r_{chunk_base} = "
                    f"{_distributed_join_tree([f'r{idx}' for idx in chunk_indices])}"
                ),
                (
                    f"    smem_store_i_{chunk_base} = "
                    f"{_distributed_join_tree([f'i{idx}' for idx in chunk_indices])}"
                ),
                (
                    "    tl.store(tle.gpu.local_ptr("
                    f"smem_r, (smem_store_index_{chunk_base},)), "
                    f"smem_store_r_{chunk_base}, mask=smem_store_mask)"
                ),
                (
                    "    tl.store(tle.gpu.local_ptr("
                    f"smem_i, (smem_store_index_{chunk_base},)), "
                    f"smem_store_i_{chunk_base}, mask=smem_store_mask)"
                ),
            ]
        )
    body.append("    tl.debug_barrier()")

    for chunk_base in range(0, 32, smem_chunk):
        chunk_indices = range(chunk_base, chunk_base + smem_chunk)
        for idx in chunk_indices:
            second_input = 2 * (idx % 16) + idx // 16
            body.append(f"    smem_logical{idx} = fft_thread * 32 + " f"{second_input}")
            body.append(
                f"    smem_phys{idx} = smem_logical{idx} ^ "
                f"(smem_logical{idx} >> {_TLE_SMEM_SWIZZLE_SHIFT})"
            )
            body.append(f"    smem_phys{idx} += smem_offset")
        body.extend(
            [
                (
                    f"    smem_load_index_{chunk_base} = "
                    f"{_distributed_join_tree([f'smem_phys{idx}' for idx in chunk_indices])}"
                ),
                (
                    f"    smem_load_r_{chunk_base} = tl.load("
                    "tle.gpu.local_ptr("
                    f"smem_r, (smem_load_index_{chunk_base},)), "
                    "mask=smem_load_mask, other=0.0)"
                ),
                (
                    f"    smem_load_i_{chunk_base} = tl.load("
                    "tle.gpu.local_ptr("
                    f"smem_i, (smem_load_index_{chunk_base},)), "
                    "mask=smem_load_mask, other=0.0)"
                ),
            ]
        )
        body.extend(
            _emit_distributed_split_tree(
                "    ",
                f"smem_load_r_{chunk_base}",
                [f"r{idx}" for idx in chunk_indices],
                f"smem_load_r_{chunk_base}",
            )
        )
        body.extend(
            _emit_distributed_split_tree(
                "    ",
                f"smem_load_i_{chunk_base}",
                [f"i{idx}" for idx in chunk_indices],
                f"smem_load_i_{chunk_base}",
            )
        )

    body.extend(_emit_local_radix32_codelet_call("    ", plan.direction))

    if include_outer_twiddle:
        body.extend(
            [
                ("    outer_base_idx = fft_thread"),
                (
                    f"    outer_base_offset = "
                    f"(four_step_inner * {four_step_n1} + outer_base_idx) * 2"
                ),
                (
                    f"    outer_step_offset = "
                    f"(four_step_inner * {four_step_n1} + {register_radix}) * 2"
                ),
                (
                    "    outer_tw_r = tl.load(twiddle_ptr + outer_base_offset, "
                    "mask=output_lane_mask, other=0.0)"
                ),
                (
                    "    outer_tw_i = tl.load(twiddle_ptr + outer_base_offset + 1, "
                    "mask=output_lane_mask, other=0.0)"
                ),
                (
                    "    outer_step_r = tl.load(twiddle_ptr + outer_step_offset, "
                    "mask=output_lane_mask, other=0.0)"
                ),
                (
                    "    outer_step_i = tl.load(twiddle_ptr + outer_step_offset + 1, "
                    "mask=output_lane_mask, other=0.0)"
                ),
            ]
        )

    for idx in range(32):
        body.append(f"    out_idx{idx} = fft_thread + {register_radix * idx}")
        if include_outer_twiddle:
            body.append(
                f"    dst_idx{idx} = "
                f"four_step_inner * {four_step_n1} + out_idx{idx}"
            )
            body.append(
                f"    r{idx}, i{idx} = "
                f"_cmul(r{idx}, i{idx}, outer_tw_r, outer_tw_i)"
            )
            if idx < 31:
                body.extend(
                    [
                        (
                            "    outer_next_r = "
                            "outer_tw_r * outer_step_r - outer_tw_i * outer_step_i"
                        ),
                        (
                            "    outer_next_i = "
                            "outer_tw_i * outer_step_r + outer_tw_r * outer_step_i"
                        ),
                        "    outer_tw_r = outer_next_r",
                        "    outer_tw_i = outer_next_i",
                    ]
                )
        else:
            body.append(
                f"    dst_idx{idx} = "
                f"out_idx{idx} * {four_step_n1} + four_step_inner"
            )
        if io_mode == "four_step_r2c_col":
            half_n = four_step_n1 * four_step_n2 // 2 + 1
            body.append(
                f"    compact_mask{idx} = "
                f"output_lane_mask & (dst_idx{idx} < {half_n})"
            )
            body.append(
                f"    output_offset{idx} = "
                f"(four_step_batch * output_distance + dst_idx{idx}) * 2"
            )
            body.append(
                f"    tl.store(out_ptr + output_offset{idx}, r{idx}, "
                f"mask=compact_mask{idx})"
            )
            body.append(
                f"    tl.store(out_ptr + output_offset{idx} + 1, i{idx}, "
                f"mask=compact_mask{idx})"
            )
        elif io_mode == "four_step_c2r_col":
            body.append(
                f"    output_offset{idx} = "
                f"four_step_batch * output_distance + dst_idx{idx}"
            )
            body.append(
                f"    tl.store(out_ptr + output_offset{idx}, r{idx}, "
                "mask=output_lane_mask)"
            )
        elif register_radix == 32:
            body.append(
                f"    output_offset{idx} = "
                f"(four_step_batch_base + dst_idx{idx}) * 2"
            )
            body.append(
                f"    output_dummy{idx} = tl.inline_asm_elementwise("
                f'"st.global.v2.{vector_suffix} [$1], {{$2, $3}}; mov.u32 $0, 0;", '
                f"{asm_store_constraints}, "
                f"[tl.cast(out_ptr + output_offset{idx}, tl.uint64), "
                f"r{idx}, i{idx}], "
                "dtype=tl.int32, is_pure=False, pack=1)"
            )
        else:
            body.append(
                f"    output_offset{idx} = "
                f"(four_step_batch_base + dst_idx{idx}) * 2"
            )
            body.append(
                f"    tl.store(out_ptr + output_offset{idx}, r{idx}, "
                "mask=output_lane_mask)"
            )
            body.append(
                f"    tl.store(out_ptr + output_offset{idx} + 1, i{idx}, "
                "mask=output_lane_mask)"
            )
    return kernel_name, "\n".join(body)


def _build_leaf_kernel_source_for_io(
    plan: LeafPlan,
    *,
    io_mode: LeafIoMode,
    prime_n: int = 0,
    four_step_n1: int = 0,
    four_step_n2: int = 0,
) -> tuple[str, str]:
    if _use_thread_local_mixed_leaf(
        plan,
        io_mode=io_mode,
        four_step_n1=four_step_n1,
        four_step_n2=four_step_n2,
    ):
        return _build_thread_local_mixed_four_step_kernel_source(
            plan,
            io_mode=io_mode,
            four_step_n1=four_step_n1,
            four_step_n2=four_step_n2,
        )

    factors = plan.factors
    n = plan.length
    smem_n = plan.smem_size
    stage_lanes = cooperative_stage_lanes_for(plan)
    uses_cooperative_stage_lanes = any(lanes != plan.lanes for lanes in stage_lanes)
    active_lanes = max(stage_lanes, default=plan.lanes)
    lane_block = lane_block_for(active_lanes)
    contiguous_modes = {
        "contiguous",
        "strided",
        "contiguous_r2c",
        "contiguous_c2r",
        "bluestein_prepare_leaf",
        "bluestein_finish_leaf",
        "bluestein_full_leaf",
    }
    batch_pack = contiguous_batch_pack_for(plan) if io_mode in contiguous_modes else 1
    row_modes = {
        "four_step_row",
        "four_step_row_strided",
        "four_step_real_row",
        "four_step_hermitian_row",
        "bluestein_four_step_prepare_row",
        "bluestein_four_step_pointwise_row",
    }
    col_modes = {
        "four_step_col",
        "four_step_col_strided",
        "four_step_r2c_col",
        "four_step_c2r_col",
        "bluestein_four_step_finish_col",
    }
    is_strided_four_step = io_mode in {"four_step_row_strided", "four_step_col_strided"}
    if io_mode in row_modes:
        inner_pack = (
            1
            if is_strided_four_step
            else four_step_row_inner_pack_for(
                four_step_n1, four_step_n2, plan.dtype, plan
            )
        )
    elif io_mode in col_modes:
        inner_pack = (
            1
            if is_strided_four_step
            else four_step_col_inner_pack_for(
                four_step_n1, four_step_n2, plan.dtype, plan
            )
        )
    else:
        inner_pack = 1
    fuse_twiddle_into_row = (
        False
        if is_strided_four_step
        else use_four_step_row_fused_twiddle(four_step_n1, four_step_n2, plan.dtype)
    )
    single_smem_buffer = _use_single_smem_buffer(
        plan,
        io_mode=io_mode,
        four_step_n1=four_step_n1,
        four_step_n2=four_step_n2,
    )
    smem_pack = max(batch_pack, inner_pack)
    vector_block = lane_block * smem_pack
    smem_slot_stride = plan.smem_size + 1 if batch_pack >= 4 else plan.smem_size
    smem_n = lane_block_for(smem_slot_stride * smem_pack)
    params = _leaf_kernel_params_for_io(
        plan,
        io_mode=io_mode,
        include_four_step_twiddle=(
            io_mode
            in {
                "four_step_row",
                "bluestein_four_step_prepare_row",
                "bluestein_four_step_pointwise_row",
                "four_step_real_row",
                "four_step_hermitian_row",
            }
            if fuse_twiddle_into_row
            else io_mode
            in {
                "four_step_col",
                "four_step_col_strided",
                "four_step_r2c_col",
                "four_step_c2r_col",
                "bluestein_four_step_finish_col",
            }
        ),
    )

    suffix = "_".join(str(x) for x in factors)
    if io_mode == "contiguous":
        kernel_prefix = "ifft" if plan.direction == "inverse" else "fft"
        kernel_name = f"{kernel_prefix}_kernel_{suffix}_l{plan.lanes}_b{lane_block}"
    elif io_mode == "strided":
        kernel_prefix = "ifft" if plan.direction == "inverse" else "fft"
        kernel_name = (
            f"{kernel_prefix}_strided_kernel_{suffix}_l{plan.lanes}_b{lane_block}"
        )
    elif io_mode == "contiguous_r2c":
        kernel_name = f"r2c_leaf_kernel_{suffix}_l{plan.lanes}_b{lane_block}"
    elif io_mode == "contiguous_c2r":
        kernel_name = f"c2r_leaf_kernel_{suffix}_l{plan.lanes}_b{lane_block}"
    elif io_mode == "bluestein_prepare_leaf":
        kernel_name = f"bluestein_prepare_leaf_kernel_{suffix}_n{prime_n}_m{n}_l{plan.lanes}_b{lane_block}"
    elif io_mode == "bluestein_finish_leaf":
        kernel_name = f"bluestein_finish_leaf_kernel_{suffix}_n{prime_n}_m{n}_l{plan.lanes}_b{lane_block}"
    elif io_mode == "bluestein_full_leaf":
        kernel_name = f"bluestein_leaf_kernel_{suffix}_n{prime_n}_m{n}_l{plan.lanes}_b{lane_block}"
    elif io_mode.startswith("bluestein_four_step_"):
        kernel_name = (
            f"{io_mode}_fft_kernel_{suffix}_p{prime_n}_n{four_step_n1}_{four_step_n2}"
            f"_l{plan.lanes}_b{lane_block}"
        )
    else:
        kernel_prefix = "ifft" if plan.direction == "inverse" else "fft"
        kernel_name = (
            f"{io_mode}_{kernel_prefix}_kernel_{suffix}_n{four_step_n1}_{four_step_n2}"
            f"_l{plan.lanes}_b{lane_block}"
        )
    body: list[str] = [
        "@triton.jit",
        f"def {kernel_name}(",
    ]
    for idx, param in enumerate(params):
        suffix = "," if idx < len(params) - 1 else ""
        body.append(f"    {param}{suffix}")
    body.append("):")
    if io_mode in contiguous_modes:
        body.append("    pid = tl.program_id(0)")
        body.append(f"    batch_id = pid * {batch_pack}")
        body.append("    if batch_id >= nbatch:")
        body.append("        return")
    else:
        if io_mode in row_modes | col_modes and inner_pack > 1:
            body.append(f"    four_step_inner_base = tl.program_id(0) * {inner_pack}")
        else:
            body.append("    four_step_inner = tl.program_id(0)")
        body.append("    four_step_batch = tl.program_id(1)")
        body.append("    if four_step_batch >= nbatch:")
        body.append("        return")
    body.append(f"    lane_vec = tl.arange(0, {vector_block})")
    if io_mode in contiguous_modes:
        if batch_pack == 1:
            body.append("    current_batch = batch_id")
            body.append("    lane = lane_vec")
            body.append(f"    lane_mask = lane < {active_lanes}")
            if io_mode == "strided":
                body.append("    batch_index = current_batch // outer_stride")
                body.append(
                    f"    batch_base = batch_index * ({n} * outer_stride) + "
                    "(current_batch - batch_index * outer_stride)"
                )
        else:
            body.append(f"    batch_slot = lane_vec // {lane_block}")
            body.append(f"    lane = lane_vec - batch_slot * {lane_block}")
            body.append("    current_batch = batch_id + batch_slot")
            body.append(
                f"    lane_mask = (lane < {active_lanes}) & (current_batch < nbatch)"
            )
            if io_mode == "strided":
                body.append("    batch_index = current_batch // outer_stride")
                body.append(
                    f"    batch_base = batch_index * ({n} * outer_stride) + "
                    "(current_batch - batch_index * outer_stride)"
                )
            else:
                body.append(f"    batch_base = current_batch * {n}")
            body.append(f"    smem_offset = batch_slot * {smem_slot_stride}")
        if batch_pack == 1:
            if io_mode != "strided":
                body.append(f"    batch_base = current_batch * {n}")
        if io_mode in {"contiguous_r2c", "contiguous_c2r"}:
            body.append("    input_batch_base = current_batch * input_distance")
            body.append("    output_batch_base = current_batch * output_distance")
    else:
        if io_mode in row_modes | col_modes and inner_pack > 1:
            four_step_inner_count = (
                four_step_n2 if io_mode in row_modes else four_step_n1
            )
            body.append(f"    inner_slot = lane_vec % {inner_pack}")
            body.append(f"    lane = lane_vec // {inner_pack}")
            body.append("    four_step_inner = four_step_inner_base + inner_slot")
            body.append(
                f"    lane_mask = (lane < {active_lanes}) & "
                f"(four_step_inner < {four_step_inner_count})"
            )
            body.append(f"    smem_offset = inner_slot * {smem_slot_stride}")
        else:
            body.append("    lane = lane_vec")
            body.append(f"    lane_mask = lane < {active_lanes}")
        if is_strided_four_step:
            body.append("    four_step_batch_index = four_step_batch // outer_stride")
            body.append(
                f"    four_step_batch_base = "
                f"four_step_batch_index * ({four_step_n1 * four_step_n2} * outer_stride) + "
                "(four_step_batch - four_step_batch_index * outer_stride)"
            )
        else:
            body.append(
                f"    four_step_batch_base = four_step_batch * {four_step_n1 * four_step_n2}"
            )

    if uses_cooperative_stage_lanes:
        body.append("    base_lane_mask = lane_mask")

    if len(factors) > 1:
        tl_dtype = _tl_real_dtype(plan.dtype)
        if not single_smem_buffer:
            body.append(
                f"    smem_a_r = tle.gpu.alloc([{smem_n}], dtype={tl_dtype}, layout=None, scope=tle.gpu.smem, "
                f"nv_mma_shared_layout=False)"
            )
            body.append(
                f"    smem_a_i = tle.gpu.alloc([{smem_n}], dtype={tl_dtype}, layout=None, scope=tle.gpu.smem, "
                f"nv_mma_shared_layout=False)"
            )
        body.append(
            f"    smem_b_r = tle.gpu.alloc([{smem_n}], dtype={tl_dtype}, layout=None, scope=tle.gpu.smem, "
            f"nv_mma_shared_layout=False)"
        )
        body.append(
            f"    smem_b_i = tle.gpu.alloc([{smem_n}], dtype={tl_dtype}, layout=None, scope=tle.gpu.smem, "
            f"nv_mma_shared_layout=False)"
        )

    if io_mode == "bluestein_full_leaf":
        if len(factors) < 2:
            raise ValueError("fused Bluestein leaf requires at least two FFT stages")
        last_source = "smem_b" if (len(factors) - 1) % 2 == 1 else "smem_a"
        intermediate_buffer = "smem_a" if last_source == "smem_b" else "smem_b"

    for stage in range(len(factors)):
        body.extend(
            _emit_stage_block(
                stage,
                factors,
                n,
                plan.lanes,
                lane_block,
                io_mode=io_mode,
                bluestein_pass=0,
                bluestein_intermediate_buffer=intermediate_buffer
                if io_mode == "bluestein_full_leaf"
                else "smem_a",
                prime_n=prime_n,
                four_step_n1=four_step_n1,
                four_step_n2=four_step_n2,
                smem_pack=smem_pack,
                fuse_twiddle_into_row=fuse_twiddle_into_row,
                single_smem_buffer=single_smem_buffer,
                direction=plan.direction,
                dtype=plan.dtype,
                stage_lanes=stage_lanes if uses_cooperative_stage_lanes else None,
            )
        )

    if io_mode == "bluestein_full_leaf":
        body.append("    tl.debug_barrier()")
        for stage in range(len(factors)):
            body.extend(
                _emit_stage_block(
                    stage,
                    factors,
                    n,
                    plan.lanes,
                    lane_block,
                    io_mode=io_mode,
                    bluestein_pass=1,
                    bluestein_intermediate_buffer=intermediate_buffer,
                    prime_n=prime_n,
                    four_step_n1=four_step_n1,
                    four_step_n2=four_step_n2,
                    smem_pack=smem_pack,
                    fuse_twiddle_into_row=fuse_twiddle_into_row,
                    single_smem_buffer=single_smem_buffer,
                    direction=plan.direction,
                    dtype=plan.dtype,
                    stage_lanes=stage_lanes if uses_cooperative_stage_lanes else None,
                )
            )

    return kernel_name, "\n".join(body)


def _build_leaf_kernel_source(plan: LeafPlan) -> tuple[str, str]:
    return _build_leaf_kernel_source_for_io(plan, io_mode="contiguous")


def _build_four_step_row_kernel_source(
    plan: LeafPlan, n1: int, n2: int
) -> tuple[str, str]:
    if plan.length != n1:
        raise ValueError(
            f"four-step row kernel length must equal n1: length={plan.length}, n1={n1}"
        )
    return _build_leaf_kernel_source_for_io(
        plan, io_mode="four_step_row", four_step_n1=n1, four_step_n2=n2
    )


def _build_four_step_col_kernel_source(
    plan: LeafPlan, n1: int, n2: int
) -> tuple[str, str]:
    if plan.length != n2:
        raise ValueError(
            f"four-step col kernel length must equal n2: length={plan.length}, n2={n2}"
        )
    return _build_leaf_kernel_source_for_io(
        plan, io_mode="four_step_col", four_step_n1=n1, four_step_n2=n2
    )


__all__ = [
    "_build_four_step_col_kernel_source",
    "_build_four_step_row_kernel_source",
    "_build_leaf_kernel_source",
    "_build_leaf_kernel_source_for_io",
    "_build_thread_local_mixed_four_step_kernel_source",
    "_direction_sign",
    "_distributed_join_tree",
    "_emit_distributed_split_tree",
    "_emit_input_base",
    "_emit_input_index",
    "_emit_local_mixed_codelet_call",
    "_emit_local_radix32_codelet_call",
    "_emit_natural_order_codelet_call",
    "_emit_natural_order_radix32_codelet_call",
    "_emit_output_base",
    "_emit_output_index",
    "_emit_radix16_codelet_call",
    "_emit_route_base",
    "_emit_route_index",
    "_emit_stage_block",
    "_emit_table_codelet",
    "_fmt_const",
    "_leaf_kernel_params",
    "_leaf_kernel_params_for_io",
    "_time_major_stride",
    "_use_thread_local_mixed_leaf",
]
