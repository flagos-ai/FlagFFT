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
from typing import Literal

from .kernels_common import (
    _NATURAL_ORDER_CODELET_RADICES,
    _PORTABLE_EXCHANGE_MIN_ELEMENTS,
    _THREAD_LOCAL_MIXED_RADICES,
    _TLE_SMEM_SWIZZLE_SHIFT,
    LeafIoMode,
    LeafPlan,
    _is_double_dtype,
    _maca_backend_active,
    _maca_knob,
    _non_nvidia_backend_active,
    _tl_real_dtype,
    _use_single_smem_buffer,
    contiguous_batch_pack_for,
    cooperative_stage_lanes_for,
    emitted_leaf_factors,
    four_step_col_inner_pack_for,
    four_step_row_inner_pack_for,
    lane_block_for,
    permuted_store_batch_pack_for,
    use_four_step_row_fused_twiddle,
    use_tle_fused_twiddle,
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


def _vector_asm_suffix(dtype: str) -> str:
    return "f64" if _is_double_dtype(dtype) else "f32"


def _vector_asm_reg(dtype: str) -> str:
    return "d" if _is_double_dtype(dtype) else "f"


def _vector_asm_dtype(dtype: str) -> str:
    return "tl.float64" if _is_double_dtype(dtype) else "tl.float32"


_COMPLEX_PAIR_OFFSETS = "_fft_pair_offsets"


def _portable_complex_vector_io() -> bool:
    """Whether to vectorize complex IO with a ``[..., 2]`` block.

    The scalar form issues two 4-byte accesses per complex element, which is
    limited by load/store throughput rather than DRAM.  A block whose
    innermost dimension has stride 1 lets Triton emit one wide access
    natively, without the ``ld.global.v2`` inline asm that the MetaX plugin
    cannot compile.
    """
    return _maca_backend_active() and _maca_knob("VEC_IO", "0") not in {"", "0"}


def _emit_vectorized_complex_load(
    indent: str,
    ptr: str,
    mask: str,
    dest: str,
    dtype: str,
) -> list[str]:
    if _portable_complex_vector_io():
        pair = "_pair_" + dest.split(",")[0].strip()
        return [
            f"{indent}{pair} = tl.load(({ptr})[:, None] + {_COMPLEX_PAIR_OFFSETS}, "
            f"mask={mask}[:, None], other=0.0)",
            f"{indent}{dest} = tl.split({pair})",
        ]
    suffix = _vector_asm_suffix(dtype)
    reg = _vector_asm_reg(dtype)
    tl_dtype = _vector_asm_dtype(dtype)
    return [
        f"{indent}{dest} = tl.inline_asm_elementwise(",
        "'{\\n"
        ".reg .pred p;\\n"
        "setp.ne.b32 p, $3, 0;\\n"
        f"@p ld.global.v2.{suffix} {{$0, $1}}, [$2];\\n"
        f"@!p mov.{suffix} $0, 0.0;\\n"
        f"@!p mov.{suffix} $1, 0.0;\\n"
        "}', "
        f'"={reg},={reg},l,r", '
        f"[tl.cast({ptr}, tl.uint64), tl.cast({mask}, tl.int32)], "
        f"dtype=({tl_dtype}, {tl_dtype}), is_pure=False, pack=1)",
    ]


def _emit_vectorized_complex_store(
    indent: str,
    ptr: str,
    r_name: str,
    i_name: str,
    mask: str,
    dtype: str,
) -> list[str]:
    if _portable_complex_vector_io():
        return [
            f"{indent}tl.store(({ptr})[:, None] + {_COMPLEX_PAIR_OFFSETS}, "
            f"tl.join({r_name}, {i_name}), mask={mask}[:, None])",
        ]
    suffix = _vector_asm_suffix(dtype)
    reg = _vector_asm_reg(dtype)
    return [
        f"{indent}tl.inline_asm_elementwise(",
        "'{\\n"
        ".reg .pred p;\\n"
        "setp.ne.b32 p, $4, 0;\\n"
        f"@p st.global.v2.{suffix} [$1], {{$2, $3}};\\n"
        "mov.u32 $0, 0;\\n"
        "}', "
        f'"=r,l,{reg},{reg},r", '
        f"[tl.cast({ptr}, tl.uint64), {r_name}, {i_name}, tl.cast({mask}, tl.int32)], "
        "dtype=tl.int32, is_pure=False, pack=1)",
    ]


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


def _emit_lane_output_base(
    indent: str,
    factors: tuple[int, ...],
    lane_var: str,
    out_var: str,
) -> list[str]:
    """`output_base` restricted to the lane axis.

    The permuted store reshapes the register tile to (batch, lane) and transposes
    it, so the output index has to be a lane-only vector of width `lane_block`
    rather than the full `[vector_block]` tensor.  Valid whenever the emitting
    stage has a single group, which makes the existing `output_base` a function
    of `lane` alone.
    """
    last_stage = len(factors) - 1
    # Keep the accumulator independent of the expression used to decode the
    # lane.  `lane_var` may include the stage group offset; interpolating it
    # into `lane_var * 0` without parentheses would leave a term such as
    # `lane_only + group * 0`, corrupting every group after the first.
    lines = [f"{indent}rem_lane = {lane_var}", f"{indent}{out_var} = 0"]
    stride = 1
    for axis in range(last_stage):
        lines.append(f"{indent}digit_lane_{axis} = rem_lane % {factors[axis]}")
        lines.append(f"{indent}rem_lane = rem_lane // {factors[axis]}")
        if stride == 1:
            lines.append(f"{indent}{out_var} += digit_lane_{axis}")
        else:
            lines.append(f"{indent}{out_var} += digit_lane_{axis} * {stride}")
        stride *= factors[axis]
    return lines


def _emit_permuted_store(
    indent: str,
    digit: int,
    factors: tuple[int, ...],
    pack: int,
    lane_block: int,
) -> list[str]:
    """Store one radix digit with the batch axis made contiguous.

    The register tile is reshaped (batch, lane) and transposed to (lane, batch),
    so its innermost dimension walks the batch slots and Triton vectorizes the
    store.  Addressing each element directly instead -- no `tl.trans` -- measured
    37% *slower* end to end on MUSA (2.25 against 1.64 ms at 256^3) even though a
    standalone store benchmark preferred it: without the transpose the tensor's
    fast axis is `lane`, whose elements sit `perm_k_stride` apart, so the store
    falls back to scalar 8-byte accesses.
    """
    offset = digit * math.prod(factors[: len(factors) - 1])
    base = "output_base_lane" if offset == 0 else f"(output_base_lane + {offset})"
    if pack == 1:
        # A one-slot tile has no real batch dimension.  Avoid materializing a
        # degenerate (lane, 1) transpose: MUSA's legacy lowering can associate
        # that singleton layout with the wrong pointer lanes.  The ordinary
        # one-dimensional tensor is both semantically exact and cheaper.
        address = f"{base} * perm_k_stride + perm_gbase_scalar"
        return [
            f"{indent}perm_addr{digit} = {address}",
            f"{indent}tl.store(out_ptr + perm_addr{digit} * 2, r{digit}, "
            f"mask=lane_mask)",
            f"{indent}tl.store(out_ptr + perm_addr{digit} * 2 + 1, i{digit}, "
            f"mask=lane_mask)",
        ]
    else:
        address = f"{base}[:, None] * perm_k_stride + perm_gbase[None, :]"
        mask = "perm_store_mask"
    return [
        f"{indent}zr{digit} = tl.trans(tl.reshape(r{digit}, ({pack}, {lane_block})))",
        f"{indent}zi{digit} = tl.trans(tl.reshape(i{digit}, ({pack}, {lane_block})))",
        f"{indent}perm_addr{digit} = {address}",
        f"{indent}tl.store(out_ptr + perm_addr{digit} * 2, zr{digit}, "
        f"mask={mask})",
        f"{indent}tl.store(out_ptr + perm_addr{digit} * 2 + 1, zi{digit}, "
        f"mask={mask})",
    ]


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


def _emit_exchange_load(
    indent: str, buffer: str, index: str, digit: int, portable: bool,
    direct: bool = False,
) -> list[str]:
    if direct:
        return [
            f"{indent}r{digit} = {buffer}_register_r{digit}",
            f"{indent}i{digit} = {buffer}_register_i{digit}",
        ]
    if portable:
        return [
            f"{indent}r{digit} = tl.where(lane_mask, tl.gather({buffer}_r, {index}, 0), 0.0)",
            f"{indent}i{digit} = tl.where(lane_mask, tl.gather({buffer}_i, {index}, 0), 0.0)",
        ]
    return [
        f"{indent}r{digit} = tl.load(tle.gpu.local_ptr({buffer}_r, ({index},)), mask=lane_mask, other=0.0)",
        f"{indent}i{digit} = tl.load(tle.gpu.local_ptr({buffer}_i, ({index},)), mask=lane_mask, other=0.0)",
    ]


def _emit_exchange_store(
    indent: str,
    buffer: str,
    index: str,
    digit: int,
    real: str,
    imag: str,
    portable: bool,
) -> list[str]:
    if portable:
        return [
            f"{indent}exchange_r{digit} = {real}",
            f"{indent}exchange_i{digit} = {imag}",
        ]
    return [
        f"{indent}tl.store(tle.gpu.local_ptr({buffer}_r, ({index},)), {real}, mask=lane_mask)",
        f"{indent}tl.store(tle.gpu.local_ptr({buffer}_i, ({index},)), {imag}, mask=lane_mask)",
    ]


def _portable_exchange_lane_floor(smem_pack: int) -> int:
    """Lane block needed for the exchange tensor to span more than one warp.

    ``vector_block = lane_block * smem_pack`` is the tensor the gather reads,
    so packing raises it without spending lanes.  ``auto`` derives the floor
    from the pack; a number pins it, and 128 reproduces the bring-up constant.
    """
    override = _maca_knob("LANE_MIN", "auto")
    if override == "auto":
        return max(1, _PORTABLE_EXCHANGE_MIN_ELEMENTS // max(smem_pack, 1))
    return int(override)


def _emit_structured_portable_exchange(
    buffer: str,
    stage: int,
    factors: tuple[int, ...],
    lane_block: int,
    pack: int,
    natural_order: bool,
    interleaved: bool,
) -> list[str]:
    """Express a power-of-two stage route as a static axis permutation."""
    n = math.prod(factors)
    radix = factors[stage]
    lanes = n // radix
    padding = lane_block // lanes
    # Codelet indices consume earlier digits low-to-high, then later digits
    # high-to-low.  Tensor axes list those digits in descending significance.
    source_axes = [*range(stage + 1, len(factors)), *range(stage - 1, -1, -1), stage]
    if natural_order:
        output_axes = [stage, *range(stage - 1, -1, -1)]
    else:
        output_axes = [*range(stage + 1, len(factors)), stage, *range(stage - 1, -1, -1)]
    shape = (pack, *(factors[axis] for axis in source_axes))
    permutation = (0, *(source_axes.index(axis) + 1 for axis in output_axes))
    lines = []
    for component in ("r", "i"):
        name = f"exchange_structured_{component}"
        joined = _distributed_join_tree(
            [f"exchange_{component}{digit}" for digit in range(radix)]
        )
        if interleaved:
            lines.append(f"    {name} = tl.reshape({joined}, ({lane_block}, {pack}, {radix}))")
            lines.append(f"    {name} = tl.trans({name}, (1, 0, 2))")
        else:
            lines.append(f"    {name} = tl.reshape({joined}, ({pack}, {lane_block}, {radix}))")
        if padding > 1:
            lines.append(f"    {name} = tl.reshape({name}, ({pack}, {padding}, {lanes}, {radix}))")
            lines.append(f"    {name} = tl.trans({name}, (0, 2, 3, 1))")
            trim_shape = (pack, lanes, radix, *((2,) * (padding.bit_length() - 1)))
            lines.append(f"    {name} = tl.reshape({name}, {trim_shape})")
            for _ in range(padding.bit_length() - 1):
                lines.append(f"    {name}, exchange_discard = tl.split({name})")
        lines.append(f"    {name} = tl.reshape({name}, {shape})")
        lines.append(f"    {name} = tl.trans({name}, {permutation})")
        lines.append(f"    {buffer}_{component} = tl.reshape({name}, ({n * pack},))")
    return lines


def _structured_exchange_supported(
    factors: tuple[int, ...], size: int, slot_stride: int, pack: int
) -> bool:
    n = math.prod(factors)
    return (
        all(factor & (factor - 1) == 0 for factor in factors)
        and slot_stride == n
        and size == n * pack
    )


def _emit_direct_exchange_registers(
    buffer: str,
    radix: int,
    n: int,
    lane_block: int,
    pack: int,
    interleaved: bool,
) -> list[str]:
    """Split the routed tensor directly into the next stage's registers."""
    lanes = n // radix
    split_shape = (pack, lane_block, *((2,) * (radix.bit_length() - 1)))
    highest_first = _maca_knob("SPLIT_ORDER", "lsb") == "msb"
    lines = []
    for component in ("r", "i"):
        prefix = f"{buffer}_register_{component}"
        lines.append(f"    {prefix} = tl.reshape({buffer}_{component}, ({pack}, {radix}, {lanes}))")
        lines.append(f"    {prefix} = tl.trans({prefix}, (0, 2, 1))")
        # Pad the whole register bank before splitting.  Padding each register
        # separately makes the MACA backend emit two barriers per component
        # and radix output (64 extra barriers on the 2048-point radix-16 stage).
        padded_lanes = lanes
        while padded_lanes < lane_block:
            lines.append(f"    {prefix} = tl.join({prefix}, tl.zeros_like({prefix}))")
            lines.append(f"    {prefix} = tl.trans({prefix}, (0, 3, 1, 2))")
            padded_lanes *= 2
            lines.append(f"    {prefix} = tl.reshape({prefix}, ({pack}, {padded_lanes}, {radix}))")
        lines.append(f"    {prefix} = tl.reshape({prefix}, {split_shape})")
        if highest_first:
            # A cross-lane radix bit can otherwise remain until the last
            # split, causing the compiler to convert each two-register pair
            # separately.  Split that high bit while the bank is still whole.
            permutation = (0, 1, *range(len(split_shape) - 1, 1, -1))
            lines.append(f"    {prefix} = tl.trans({prefix}, {permutation})")
        lines.extend(_emit_distributed_split_tree(
            "    ", prefix, [f"{prefix}{digit}" for digit in range(radix)], prefix,
            highest_first=highest_first,
        ))
        for digit in range(radix):
            name = f"{prefix}{digit}"
            if interleaved:
                lines.append(f"    {name} = tl.trans({name}, (1, 0))")
            lines.append(f"    {name} = tl.reshape({name}, ({pack * lane_block},))")
    return lines


def _emit_portable_exchange(
    buffer: str,
    stage: int,
    factors: tuple[int, ...],
    lane_block: int,
    size: int,
    slot_stride: int,
    pack: int,
    natural_order: bool = False,
    register_lane_stride: int = 1,
    register_slot_stride: int | None = None,
) -> list[str]:
    """Invert the codelet routing and gather from each register tensor.

    Each butterfly runs once; padded lanes/digits never become FFT state.
    The compiler supplies any shared-memory layout conversions, avoiding
    TLE local pointers in the MetaX plugin.

    ``register_lane_stride``/``register_slot_stride`` describe how the register
    tensors that hold the codelet outputs are laid out.  Contiguous batch
    packing strides the slot by the lane block, while four-step inner packing
    interleaves lane and slot, so the gather index has to follow whichever
    layout produced those tensors.
    """
    if register_slot_stride is None:
        register_slot_stride = lane_block
    n = math.prod(factors)
    radix = factors[stage]
    if (
        _maca_knob("EXCHANGE") in {"transpose", "direct", "direct_all"}
        and _structured_exchange_supported(factors, size, slot_stride, pack)
        and lane_block >= n // radix
    ):
        lines = _emit_structured_portable_exchange(
            buffer, stage, factors, lane_block, pack, natural_order,
            interleaved=register_lane_stride > 1,
        )
        if _maca_knob("EXCHANGE") in {"direct", "direct_all"} and not natural_order:
            lines.extend(_emit_direct_exchange_registers(
                buffer, factors[stage + 1], n, lane_block, pack,
                interleaved=register_lane_stride > 1,
            ))
        return lines
    lines = [
        f"    exchange_pos = tl.arange(0, {size})",
        f"    exchange_slot = exchange_pos // {slot_stride}",
        f"    exchange_local = exchange_pos % {slot_stride}",
        f"    exchange_valid = (exchange_local < {n}) & (exchange_slot < {pack})",
    ]
    if natural_order:
        lanes = n // radix
        lines += [
            f"    exchange_codelet = exchange_local % {lanes}",
            f"    exchange_digit = exchange_local // {lanes}",
        ]
    else:
        next_lanes = n // factors[stage + 1]
        lines += [
            f"    exchange_rem = exchange_local % {next_lanes}",
            f"    exchange_next_digit = exchange_local // {next_lanes}",
            "    exchange_codelet = exchange_rem * 0",
        ]
        stride = 1
        for axis in range(stage):
            lines += [
                f"    exchange_codelet += (exchange_rem % {factors[axis]}) * {stride}",
                f"    exchange_rem = exchange_rem // {factors[axis]}",
            ]
            stride *= factors[axis]
        lines += [
            f"    exchange_digit = exchange_rem % {radix}",
            f"    exchange_rem = exchange_rem // {radix}",
        ]
        for axis in range(len(factors) - 1, stage + 1, -1):
            lines += [
                f"    exchange_codelet += (exchange_rem % {factors[axis]}) * {stride}",
                f"    exchange_rem = exchange_rem // {factors[axis]}",
            ]
            stride *= factors[axis]
        lines.append(f"    exchange_codelet += exchange_next_digit * {stride}")
    lines.append(
        f"    exchange_src = exchange_codelet * {register_lane_stride} + "
        f"exchange_slot * {register_slot_stride}"
    )
    lines.append("    exchange_src = tl.where(exchange_valid, exchange_src, 0)")
    # Join the register dimension before routing.  The original expression
    # gathers every radix output into an entire leaf-sized vector and selects
    # one of them at each element.  A joined tensor permits one gather per
    # component, while retaining the same route and padded-lane semantics.
    # Keep this experimental until MACA compilation and timing are validated.
    method = _maca_knob("EXCHANGE")
    if method == "direct_all" or (
        method in {"join", "transpose", "direct"} and radix & (radix - 1) == 0
    ):
        vector_block = lane_block * pack
        joined_radix = 1 << (radix - 1).bit_length()
        lines.append(
            f"    exchange_joined_index = tl.where(exchange_valid, "
            f"exchange_src * {joined_radix} + exchange_digit, 0)"
        )
        for component in ("r", "i"):
            joined = _distributed_join_tree(
                [f"exchange_{component}{digit}" for digit in range(radix)]
                + [f"tl.zeros_like(exchange_{component}0)"] * (joined_radix - radix)
            )
            lines.append(
                f"    exchange_joined_{component} = tl.reshape({joined}, "
                f"({vector_block * joined_radix},))"
            )
            lines.append(
                f"    {buffer}_{component} = tl.where(exchange_valid, "
                f"tl.gather(exchange_joined_{component}, "
                "exchange_joined_index, 0), 0.0)"
            )
        return lines
    for component in ("r", "i"):
        lines.append(
            f"    {buffer}_{component} = tl.full(({size},), 0, exchange_{component}0.dtype)"
        )
        for digit in range(radix):
            lines.append(
                f"    {buffer}_{component} = tl.where(exchange_valid & (exchange_digit == {digit}), "
                f"tl.gather(exchange_{component}{digit}, exchange_src, 0), {buffer}_{component})"
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
    inner_pack: int = 1,
    fuse_twiddle_into_row: bool = False,
    single_smem_buffer: bool = False,
    direction: Literal["forward", "inverse"] = "forward",
    dtype: str = "complex64",
    stage_lanes: tuple[int, ...] | None = None,
    portable_exchange: bool = False,
    exchange_size: int = 0,
    exchange_slot_stride: int = 0,
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
    vector_io_allowed = not _non_nvidia_backend_active() or _portable_complex_vector_io()
    vectorized_four_step_complex_io = (
        io_mode
        in {
            "four_step_row",
            "four_step_col",
            "four_step_r2c_col",
            "four_step_c2r_col",
            "four_step_hermitian_row",
        }
        and vector_io_allowed
    )
    vectorized_complex_io = (
        io_mode in {"contiguous", "contiguous_c2r", "permuted_store"}
        or vectorized_four_step_complex_io
    ) and vector_io_allowed
    vector_suffix = "f64" if _is_double_dtype(dtype) else "f32"
    vector_reg = "d" if _is_double_dtype(dtype) else "f"
    vector_dtype = "tl.float64" if _is_double_dtype(dtype) else "tl.float32"
    if portable_exchange:
        assert groups == 1
        lines.append(f"    group_{stage} = 0")
        indent = "    "
    else:
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
        if io_mode == "permuted_store":
            active_lanes = max(stage_lanes) if stage_lanes is not None else lanes
            lines.append(f"{indent}lane_only = tl.arange(0, {lane_block})")
            lines.append(f"{indent}perm_lane_mask = lane_only < {active_lanes}")
            lines.append(
                f"{indent}lane_only = tl.where(perm_lane_mask, lane_only, 0)"
            )
            lines.extend(
                _emit_lane_output_base(
                    indent,
                    factors,
                    f"lane_only + {current_lanes} * group_{stage}",
                    "output_base_lane",
                )
            )
            if smem_pack == 1:
                lines.append(
                    f"{indent}perm_store_mask = perm_lane_mask[:, None]"
                )
            else:
                lines.append(
                    f"{indent}perm_store_mask = perm_lane_mask[:, None] & perm_mask[None, :]"
                )
    else:
        lines.extend(
            _emit_route_base(indent, stage, factors, current_lanes, f"group_{stage}")
        )

    for j in range(radix):
        if stage == 0:
            lines.extend(_emit_input_index(indent, f"in{j}", factors, j))
            if io_mode in {"contiguous", "permuted_store"}:
                if vectorized_complex_io:
                    lines.extend(
                        _emit_vectorized_complex_load(
                            indent,
                            f"in_ptr + (batch_base + in{j}) * 2",
                            "lane_mask",
                            f"r{j}, i{j}",
                            dtype,
                        )
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
                    lines.extend(
                        _emit_vectorized_complex_load(
                            indent, f"src_ptr{j}", "lane_mask", f"r{j}, i{j}", dtype
                        )
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
                    lines.extend(
                        _emit_exchange_load(
                            indent,
                            bluestein_intermediate_buffer,
                            intermediate_index,
                            j,
                            portable_exchange,
                        )
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
                if vectorized_complex_io:
                    lines.extend(
                        _emit_vectorized_complex_load(
                            indent,
                            f"in_ptr + (four_step_batch_base + src_idx{j}) * 2",
                            "lane_mask",
                            f"r{j}, i{j}",
                            dtype,
                        )
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
                if vectorized_complex_io:
                    lines.extend(
                        _emit_vectorized_complex_load(
                            indent, f"src_ptr{j}", "lane_mask", f"r{j}, i{j}", dtype
                        )
                    )
                else:
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
                    if vectorized_complex_io:
                        lines.extend(
                            _emit_vectorized_complex_load(
                                indent,
                                f"in_ptr + (four_step_batch_base + src_idx{j}) * 2",
                                "lane_mask",
                                f"r{j}, i{j}",
                                dtype,
                            )
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
                else:
                    if vectorized_complex_io:
                        lines.extend(
                            _emit_vectorized_complex_load(
                                indent,
                                f"in_ptr + (four_step_batch_base + src_idx{j}) * 2",
                                "lane_mask",
                                f"r{j}, i{j}",
                                dtype,
                            )
                        )
                        lines.extend(
                            _emit_vectorized_complex_load(
                                indent,
                                f"twiddle_ptr + src_idx{j} * 2",
                                "lane_mask",
                                f"tw_r{j}, tw_i{j}",
                                dtype,
                            )
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
            lines.extend(
                _emit_exchange_load(
                    indent, source_buffer, load_index, j, portable_exchange,
                    direct=(
                        portable_exchange
                        and _maca_knob("EXCHANGE") in {"direct", "direct_all"}
                        and _structured_exchange_supported(
                            factors, exchange_size, exchange_slot_stride, smem_pack
                        )
                    ),
                )
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
            if io_mode == "permuted_store":
                lines.extend(
                    _emit_permuted_store(indent, j, factors, smem_pack, lane_block)
                )
                continue
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
                        lines.extend(
                            _emit_vectorized_complex_store(
                                indent,
                                f"out_ptr + (batch_base + out_idx{j}) * 2",
                                f"r{j}",
                                f"i{j}",
                                "lane_mask",
                                dtype,
                            )
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
                    lines.extend(
                        _emit_exchange_store(
                            indent,
                            bluestein_intermediate_buffer,
                            intermediate_index,
                            j,
                            f"point_r{j}",
                            f"-point_i{j}",
                            portable_exchange,
                        )
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
                        if vectorized_complex_io:
                            lines.extend(
                                _emit_vectorized_complex_load(
                                    indent,
                                    f"twiddle_ptr + dst_idx{j} * 2",
                                    "lane_mask",
                                    f"tw_r{j}, tw_i{j}",
                                    dtype,
                                )
                            )
                        else:
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
                if vectorized_complex_io:
                    lines.extend(
                        _emit_vectorized_complex_store(
                            indent,
                            f"out_ptr + (four_step_batch_base + dst_idx{j}) * 2",
                            f"r{j}",
                            f"i{j}",
                            "lane_mask",
                            dtype,
                        )
                    )
                else:
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
                if vectorized_complex_io:
                    lines.extend(
                        _emit_vectorized_complex_store(
                            indent,
                            f"dst_ptr{j}",
                            f"r{j}",
                            f"i{j}",
                            f"compact_mask{j}",
                            dtype,
                        )
                    )
                else:
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
                if vectorized_complex_io:
                    lines.extend(
                        _emit_vectorized_complex_store(
                            indent,
                            f"out_ptr + (four_step_batch_base + dst_idx{j}) * 2",
                            f"r{j}",
                            f"i{j}",
                            "lane_mask",
                            dtype,
                        )
                    )
                else:
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
            lines.extend(
                _emit_exchange_store(
                    indent,
                    dest_buffer,
                    store_index,
                    j,
                    f"r{j}",
                    f"i{j}",
                    portable_exchange,
                )
            )

    if portable_exchange:
        buffer = (
            bluestein_intermediate_buffer
            if is_last and io_mode == "bluestein_full_leaf" and bluestein_pass == 0
            else dest_buffer
        )
        if buffer is not None:
            lines.extend(
                _emit_portable_exchange(
                    buffer,
                    stage,
                    factors,
                    lane_block,
                    exchange_size,
                    exchange_slot_stride,
                    smem_pack,
                    natural_order=is_last,
                    register_lane_stride=inner_pack if inner_pack > 1 else 1,
                    register_slot_stride=1 if inner_pack > 1 else lane_block,
                )
            )
    elif not is_last:
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
    if io_mode == "permuted_store":
        params.append("perm_span")
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
    *,
    highest_first: bool = False,
) -> list[str]:
    if len(names) == 1:
        return [f"{indent}{names[0]} = {source}"]

    even_names = names[:len(names) // 2] if highest_first else names[0::2]
    odd_names = names[len(names) // 2:] if highest_first else names[1::2]
    even_source = (
        even_names[0] if len(even_names) == 1 else f"{prefix}_even{len(names)}"
    )
    odd_source = odd_names[0] if len(odd_names) == 1 else f"{prefix}_odd{len(names)}"
    lines = [f"{indent}{even_source}, {odd_source} = tl.split({source})"]
    if len(even_names) > 1:
        lines.extend(
            _emit_distributed_split_tree(
                indent, even_source, even_names, f"{prefix}_e",
                highest_first=highest_first,
            )
        )
    if len(odd_names) > 1:
        lines.extend(
            _emit_distributed_split_tree(
                indent, odd_source, odd_names, f"{prefix}_o",
                highest_first=highest_first,
            )
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
    perm_form: str = "outer",
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

    portable_exchange = _maca_backend_active()
    factors = emitted_leaf_factors(plan, io_mode)
    n = plan.length
    smem_n = plan.smem_size
    stage_lanes = (
        tuple(n // radix for radix in factors)
        if portable_exchange
        else cooperative_stage_lanes_for(plan)
    )
    uses_cooperative_stage_lanes = portable_exchange or any(
        lanes != plan.lanes for lanes in stage_lanes
    )
    active_lanes = max(stage_lanes, default=plan.lanes)
    lane_block = lane_block_for(active_lanes)
    contiguous_modes = {
        "contiguous",
        "strided",
        "permuted_store",
        "contiguous_r2c",
        "contiguous_c2r",
        "bluestein_prepare_leaf",
        "bluestein_finish_leaf",
        "bluestein_full_leaf",
    }
    if io_mode == "permuted_store":
        batch_pack = permuted_store_batch_pack_for(plan)
    elif io_mode in contiguous_modes:
        batch_pack = contiguous_batch_pack_for(plan)
    else:
        batch_pack = 1
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
    if portable_exchange and len(factors) > 1:
        # The MetaX plugin cannot lower the warp-shuffle path, so the exchange
        # must gather from a tensor wider than one 64-thread warp.  Packing
        # widens that tensor for free, whereas raising the lane block idles
        # most lanes on a small leaf -- which is what starved the throughput
        # bound batch and 3D shapes.
        lane_block = max(lane_block, _portable_exchange_lane_floor(smem_pack))
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
    elif io_mode == "permuted_store":
        kernel_prefix = "ifft" if plan.direction == "inverse" else "fft"
        kernel_name = (
            f"permuted_store_{perm_form}_{kernel_prefix}_kernel_{suffix}"
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
    if _portable_complex_vector_io():
        body.append(f"    {_COMPLEX_PAIR_OFFSETS} = tl.arange(0, 2)[None, :]")
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
            if io_mode == "permuted_store" and perm_form == "inner":
                # The packed path derives perm_base from pid below.  With one
                # batch slot, pid already is the row index, but the common
                # inner-form address equations still consume this name.
                body.append("    perm_base = batch_id")
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
            if io_mode == "permuted_store" and perm_form == "inner":
                # This pass permutes an axis whose output position is scaled by
                # the *other* cube dimension, so a block has to span that
                # dimension rather than consecutive rows: its rows are strided
                # by perm_span in the row index.  That is exactly what makes the
                # store run contiguous.
                body.append(
                    f"    perm_base = (pid // perm_span) * {batch_pack} * perm_span "
                    "+ (pid % perm_span)"
                )
                body.append("    current_batch = perm_base + batch_slot * perm_span")
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
        if io_mode == "permuted_store":
            # `perm_gbase` is the output address of each batch slot's row start
            # and `perm_k_stride` the stride of the FFT output index; the store
            # adds the two.  The two forms differ in which of the row index's
            # two mixed-radix parts gets scaled by the output layout.
            body.append(f"    perm_slot = tl.arange(0, {batch_pack})")
            if perm_form == "inner":
                body.append("    perm_batch = perm_base + perm_slot * perm_span")
                body.append("    perm_i0 = perm_batch // perm_span")
                body.append("    perm_i1 = perm_batch - perm_i0 * perm_span")
                body.append("    perm_gbase = perm_i1 * (nbatch // perm_span) + perm_i0")
                body.append("    perm_k_stride = nbatch")
            else:
                body.append("    perm_batch = batch_id + perm_slot")
                body.append("    perm_i0 = perm_batch // perm_span")
                body.append("    perm_i1 = perm_batch - perm_i0 * perm_span")
                body.append(f"    perm_gbase = perm_i0 * ({n} * perm_span) + perm_i1")
                body.append("    perm_k_stride = perm_span")
            body.append("    perm_mask = perm_batch < nbatch")
            if batch_pack == 1:
                # Keep the singleton row address scalar.  On MUSA this avoids
                # a degenerate [1] tensor layout being broadcast into the
                # lane-shaped store pointer.
                if perm_form == "inner":
                    body.append("    perm_batch_scalar = perm_base")
                else:
                    body.append("    perm_batch_scalar = batch_id")
                body.append("    perm_i0_scalar = perm_batch_scalar // perm_span")
                body.append(
                    "    perm_i1_scalar = perm_batch_scalar - "
                    "perm_i0_scalar * perm_span"
                )
                if perm_form == "inner":
                    body.append(
                        "    perm_gbase_scalar = perm_i1_scalar * "
                        "(nbatch // perm_span) + perm_i0_scalar"
                    )
                else:
                    body.append(
                        f"    perm_gbase_scalar = perm_i0_scalar * "
                        f"({n} * perm_span) + perm_i1_scalar"
                    )
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

    if len(factors) > 1 and not portable_exchange:
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
                inner_pack=inner_pack,
                fuse_twiddle_into_row=fuse_twiddle_into_row,
                single_smem_buffer=single_smem_buffer,
                direction=plan.direction,
                dtype=plan.dtype,
                stage_lanes=stage_lanes if uses_cooperative_stage_lanes else None,
                portable_exchange=portable_exchange,
                exchange_size=smem_n,
                exchange_slot_stride=smem_slot_stride,
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
                    inner_pack=inner_pack,
                    fuse_twiddle_into_row=fuse_twiddle_into_row,
                    single_smem_buffer=single_smem_buffer,
                    direction=plan.direction,
                    dtype=plan.dtype,
                    stage_lanes=stage_lanes if uses_cooperative_stage_lanes else None,
                    portable_exchange=portable_exchange,
                    exchange_size=smem_n,
                    exchange_slot_stride=smem_slot_stride,
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
