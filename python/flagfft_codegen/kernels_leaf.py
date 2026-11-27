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
import os
from .backend_profile import current_profile
from typing import Literal

from .kernels_common import (
    _NATURAL_ORDER_CODELET_RADICES,
    _PORTABLE_EXCHANGE_MIN_ELEMENTS,
    _THREAD_LOCAL_MIXED_RADICES,
    _TLE_SMEM_SWIZZLE_SHIFT,
    LeafIoMode,
    LeafPlan,
    _dtype_suffix,
    _is_double_dtype,
    _hcu_backend_active,
    _maca_backend_active,
    _npu_backend_active,
    _mthreads_backend_active,
    _ix_backend_active,
    _ix_permuted_single_smem,
    _portable_leaf_backend_active,
    _maca_knob,
    ix_packed_real_tle,
    _non_nvidia_backend_active,
    _tl_real_dtype,
    _use_single_smem_buffer,
    contiguous_batch_pack_for,
    hcu_packed_cta_batch_pack_for,
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

_IX_BATCH_INTERLEAVE_MODES = {
    "contiguous", "strided", "permuted_store", "strided_permuted_store",
    "contiguous_r2c", "contiguous_c2r", "permuted_r2c", "permuted_store_r2c",
}


def _hcu_vector_io(dtype: str) -> bool:
    if not _hcu_backend_active():
        return False
    setting = os.getenv("FLAGFFT_HCU_VECTOR_IO", "0")
    if setting not in {"0", "1", "packed", "fp32"}:
        raise ValueError("FLAGFFT_HCU_VECTOR_IO must be 0, 1, packed or fp32")
    return setting == "1" or (setting in {"packed", "fp32"} and dtype == "complex64")


def _packed_hcu_fp32_complex_io(dtype: str) -> bool:
    # Preserve one logical lane per complex element through HCU lowering.
    # The pair tensor otherwise introduces a second layout axis at every
    # global access. Bitcasts retain all FP32 bits without arithmetic casts.
    return (dtype == "complex64" and _hcu_backend_active()
            and os.getenv("FLAGFFT_HCU_VECTOR_IO", "0") == "packed")


def _ix_batch_smem_interleave(io_mode: str, portable: bool, pack: int) -> bool:
    if not _ix_backend_active():
        return False
    setting = _maca_knob("BATCH_INTERLEAVE", "0")
    if setting not in {"0", "1"}:
        raise ValueError("FLAGFFT_IX_BATCH_INTERLEAVE must be 0 or 1")
    return setting == "1" and not portable and pack > 1 and io_mode in _IX_BATCH_INTERLEAVE_MODES


def _portable_complex_vector_io(dtype: str = "complex64") -> bool:
    """Whether to vectorize complex IO without backend inline assembly.

    The scalar form issues two 4-byte accesses per complex element, which is
    limited by load/store throughput rather than DRAM.  A block whose
    innermost dimension has stride 1 lets Triton emit one wide access
    natively, without the ``ld.global.v2`` inline asm that the MetaX plugin
    cannot compile.
    """
    return (
        (_maca_backend_active() and _maca_knob("VEC_IO", "0") not in {"", "0"})
        or (_npu_backend_active() and os.environ.get("FLAGFFT_NPU_PORTABLE_VECTOR_IO") == "1")
        or (_ix_backend_active() and _maca_knob("VEC_IO", "0") == "packed")
        or _hcu_vector_io(dtype)
    )


def _packed_maca_fp32_complex_io(dtype: str) -> bool:
    return (
        dtype == "complex64"
        and (_maca_backend_active() or _ix_backend_active())
        and _maca_knob("VEC_IO", "0") == "packed"
    )


def _emit_npu_matrix_load(indent, ptr, tag, real, imag, lanes, pack, active, offset, width, batch_base):
    """Keep the row/column DMA shape explicit until after the global load."""
    return [
        f"{indent}{tag}_row = tl.arange(0, {lanes})[:,None]",
        f"{indent}{tag}_col = four_step_inner_base*2 + tl.arange(0, {2*pack})[None,:]",
        f"{indent}{tag} = tl.load({ptr} + ({batch_base})*2 + ({tag}_row+{offset})*{2*width} + {tag}_col, "
        f"({tag}_row < {active}) & ({tag}_col < {2*width}), 0.0)",
        f"{indent}{real}, {imag} = tl.split(tl.reshape({tag}, ({lanes*pack},2)))",
    ]


def _emit_npu_matrix_store(indent, tag, real, imag, lanes, pack, active, offset, width, transposed, output_rows):
    lines=[f"{indent}{tag} = tl.reshape(tl.join({real}, {imag}), ({lanes},{pack},2))"]
    if transposed:
        lines += [
            f"{indent}{tag} = tl.reshape(tl.permute({tag},1,0,2), ({pack},{2*lanes}))",
            f"{indent}{tag}_row = four_step_inner_base + tl.arange(0,{pack})[:,None]",
            f"{indent}{tag}_col = tl.arange(0,{2*lanes})[None,:]",
            f"{indent}tl.store(out_ptr + four_step_batch_base*2 + {tag}_row*{2*width} + {tag}_col+{2*offset}, "
            f"{tag}, ({tag}_row < {output_rows}) & ({tag}_col < {2*active}))",
        ]
    else:
        lines += [
            f"{indent}{tag} = tl.reshape({tag}, ({lanes},{2*pack}))",
            f"{indent}{tag}_row = tl.arange(0,{lanes})[:,None]",
            f"{indent}{tag}_col = four_step_inner_base*2 + tl.arange(0,{2*pack})[None,:]",
            f"{indent}tl.store(out_ptr + four_step_batch_base*2 + ({tag}_row+{offset})*{2*width} + {tag}_col, "
            f"{tag}, ({tag}_row < {active}) & ({tag}_col < {2*width}))",
        ]
    return lines


def _emit_vectorized_complex_load(
    indent: str,
    ptr: str,
    mask: str,
    dest: str,
    dtype: str,
) -> list[str]:
    if _packed_maca_fp32_complex_io(dtype) or _packed_hcu_fp32_complex_io(dtype):
        pair = "_packed_pair_" + dest.split(",")[0].strip()
        real, imag = (name.strip() for name in dest.split(","))
        return [
            f"{indent}{pair} = tl.load(tl.cast(({ptr}), tl.pointer_type(tl.uint64)), "
            f"mask={mask}, other=0)",
            f"{indent}{real} = tl.cast(tl.cast({pair}, tl.uint32), tl.float32, bitcast=True)",
            f"{indent}{imag} = tl.cast(tl.cast({pair} >> 32, tl.uint32), tl.float32, bitcast=True)",
        ]
    if _portable_complex_vector_io(dtype):
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
    if _packed_maca_fp32_complex_io(dtype) or _packed_hcu_fp32_complex_io(dtype):
        pair = "_packed_store_" + r_name
        return [
            f"{indent}{pair} = tl.cast(tl.cast({r_name}, tl.uint32, bitcast=True), tl.uint64) "
            f"| (tl.cast(tl.cast({i_name}, tl.uint32, bitcast=True), tl.uint64) << 32)",
            f"{indent}tl.store(tl.cast(({ptr}), tl.pointer_type(tl.uint64)), {pair}, "
            f"mask={mask})",
        ]
    if _portable_complex_vector_io(dtype):
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


def _emit_complex_boundary_load(indent, ptr, mask, real, imag, dtype, packed):
    if packed:
        return _emit_vectorized_complex_load(indent,ptr,mask,f'{real}, {imag}',dtype)
    zero='0.0'
    return [f'{indent}{real} = tl.load({ptr}, mask={mask}, other={zero})',
            f'{indent}{imag} = tl.load({ptr} + 1, mask={mask}, other={zero})']


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


def _emit_ix_register_codelet(indent, radix, direction, *, split_radix, decimation_in_frequency=False):
    """Emit the register butterflies used by the dense IX CT leaves."""
    lines = []
    sign = _direction_sign(direction)

    def combine(prefix, a, b, subtract=False):
        op = "-" if subtract else "+"
        pair = (prefix + "r", prefix + "i")
        lines.extend(f"{indent}{dst} = {left} {op} {right}"
                     for dst, left, right in zip(pair, a, b))
        return pair

    def rotate(prefix, pair, numerator, length):
        phase = (4 * numerator) % (4 * length)
        real, imag = pair
        if phase == 0:
            return pair
        if phase == length:
            return (f"({-sign:g} * {imag})", f"({sign:g} * {real})")
        if phase == 2 * length:
            return (f"(-{real})", f"(-{imag})")
        if phase == 3 * length:
            return (f"({sign:g} * {imag})", f"({-sign:g} * {real})")
        angle = sign * 2.0 * math.pi * numerator / length
        wr, wi = math.cos(angle), math.sin(angle)
        result = (prefix + "r", prefix + "i")
        if abs(abs(wr) - abs(wi)) < 1e-14:
            sign_w = "+" if wr * wi > 0 else "-"
            sign_h = "-" if sign_w == "+" else "+"
            c = _fmt_const(wr)
            lines.extend([f"{indent}{result[0]} = {c} * ({real} {sign_h} {imag})",
                          f"{indent}{result[1]} = {c} * ({imag} {sign_w} {real})"])
        else:
            lines.append(f"{indent}{result[0]}, {result[1]} = _cmul("
                         f"{real}, {imag}, {_fmt_const(wr)}, {_fmt_const(wi)})")
        return result

    def fft(values, prefix):
        n = len(values)
        if n == 1:
            return values
        if decimation_in_frequency:
            half = n // 2
            sums = [combine(f"{prefix}s{k}", values[k], values[k + half]) for k in range(half)]
            differences = [
                rotate(f"{prefix}t{k}", combine(f"{prefix}d{k}", values[k], values[k + half], subtract=True), k, n)
                for k in range(half)
            ]
            even = fft(sums, prefix + "e")
            odd = fft(differences, prefix + "o")
            return [pair for group in zip(even, odd) for pair in group]
        even = fft(values[::2], prefix + "e")
        result = [None] * n
        if split_radix and n >= 8:
            odd1 = fft(values[1::4], prefix + "a")
            odd3 = fft(values[3::4], prefix + "b")
            quarter = n // 4
            for k in range(quarter):
                u = rotate(f"{prefix}u{k}", odd1[k], k, n)
                v = rotate(f"{prefix}v{k}", odd3[k], 3 * k, n)
                total = combine(f"{prefix}s{k}", u, v)
                diff = combine(f"{prefix}d{k}", u, v, subtract=True)
                rotated = rotate(f"{prefix}h{k}", diff, 1, 4)
                result[k] = combine(f"{prefix}f{k}", even[k], total)
                result[k + 2 * quarter] = combine(f"{prefix}g{k}", even[k], total, subtract=True)
                result[k + quarter] = combine(f"{prefix}j{k}", even[k + quarter], rotated)
                result[k + 3 * quarter] = combine(f"{prefix}m{k}", even[k + quarter], rotated, subtract=True)
        else:
            odd = fft(values[1::2], prefix + "o")
            for k in range(n // 2):
                t = rotate(f"{prefix}t{k}", odd[k], k, n)
                result[k] = combine(f"{prefix}s{k}", even[k], t)
                result[k + n // 2] = combine(f"{prefix}d{k}", even[k], t, subtract=True)
        return result

    output = fft([(f"r{i}", f"i{i}") for i in range(radix)], "reg")
    for i, (real, imag) in enumerate(output):
        lines.extend([f"{indent}r{i} = {real}", f"{indent}i{i} = {imag}"])
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
    if _npu_backend_active() and os.environ.get("FLAGFFT_NPU_PORTABLE_VECTOR_IO") == "1":
        # The mixed-radix digit reconstruction is the identity on active
        # codelets. Preserve that affine address; the GM load already masks
        # padded lanes and CANN can then issue a contiguous transfer.
        return [f"{indent}input_base = lane + {lanes} * {group_var}"]
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
    if _npu_backend_active() and os.environ.get("FLAGFFT_NPU_PORTABLE_VECTOR_IO") == "1":
        return [f"{indent}output_base = lane + {lanes} * {group_var}"]
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


def _emit_grouped_fp32_permuted_store(
    indent: str,
    factors: tuple[int, ...],
    radix: int,
    pack: int,
    lane_block: int,
) -> list[str]:
    """Transpose a full final radix group once before writing complex pairs."""
    lines: list[str] = []
    pairs = []
    for digit in range(radix):
        pair_name = f"perm_group_pair_flat{digit}"
        pairs.append(pair_name)
        lines.append(
            f"{indent}{pair_name} = "
            f"tl.cast(tl.cast(r{digit}, tl.uint32, bitcast=True), tl.uint64) "
            f"| (tl.cast(tl.cast(i{digit}, tl.uint32, bitcast=True), tl.uint64) << 32)"
        )

    level = pairs
    level_id = 0
    join_stride = radix // 2
    while join_stride > 0:
        next_level = []
        for pair_index in range(join_stride):
            joined = f"perm_group_join_{level_id}_{pair_index}"
            lines.append(
                f"{indent}{joined} = tl.join({level[pair_index]}, "
                f"{level[pair_index + join_stride]})"
            )
            next_level.append(joined)
        level = next_level
        level_id += 1
        join_stride //= 2

    lines.extend(
        [
            f"{indent}perm_group_pair_raw = tl.reshape({level[0]}, "
            f"({pack}, {lane_block * radix}))",
            f"{indent}perm_group_pair_out = tl.trans(perm_group_pair_raw)",
            f"{indent}perm_group_digit = tl.arange(0, {radix})",
            f"{indent}perm_group_base = tl.reshape(output_base_lane[:, None] + "
            f"perm_group_digit[None, :] * {math.prod(factors[:-1])}, "
            f"({lane_block * radix},))",
            f"{indent}perm_group_addr = perm_group_base[:, None] * "
            "perm_k_stride + perm_gbase[None, :]",
            f"{indent}perm_group_mask = tl.reshape(tl.broadcast_to("
            f"perm_store_mask[:, None, :], ({lane_block}, {radix}, {pack})), "
            f"({lane_block * radix}, {pack}))",
            f"{indent}perm_group_ptr = tl.cast(out_ptr + perm_group_addr * 2, "
            "tl.pointer_type(tl.uint64))",
            f"{indent}tl.store(perm_group_ptr, perm_group_pair_out, "
            "mask=perm_group_mask)",
        ]
    )
    return lines


def _emit_permuted_store(
    indent: str,
    digit: int,
    factors: tuple[int, ...],
    pack: int,
    lane_block: int,
    dtype: str,
    compact_length: int | None = None,
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
    vector_store = (current_profile().backend == "cuda" and
                    os.getenv("FLAGFFT_CUDA_PERMSTORE_VEC_STORE") == "1")
    if pack == 1:
        # A one-slot tile has no real batch dimension.  Avoid materializing a
        # degenerate (lane, 1) transpose: MUSA's legacy lowering can associate
        # that singleton layout with the wrong pointer lanes.  The ordinary
        # one-dimensional tensor is both semantically exact and cheaper.
        address = f"{base} * perm_k_stride + perm_gbase_scalar"
        mask = "lane_mask"
        if compact_length is not None:
            mask = f"lane_mask & ({base} < {compact_length})"
        if vector_store:
            return [f"{indent}perm_addr{digit} = {address}"] + _emit_vectorized_complex_store(
                indent, f"out_ptr + perm_addr{digit} * 2", f"r{digit}", f"i{digit}", mask, dtype
            )
        if _packed_maca_fp32_complex_io(dtype):
            return [
                f"{indent}perm_addr{digit} = {address}",
                f"{indent}perm_pair{digit} = "
                f"tl.cast(tl.cast(r{digit}, tl.uint32, bitcast=True), tl.uint64) "
                f"| (tl.cast(tl.cast(i{digit}, tl.uint32, bitcast=True), tl.uint64) << 32)",
                f"{indent}perm_ptr{digit} = tl.cast(out_ptr + perm_addr{digit} * 2, "
                "tl.pointer_type(tl.uint64))",
                f"{indent}tl.store(perm_ptr{digit}, perm_pair{digit}, mask={mask})",
            ]
        mask = ("lane_mask" if compact_length is None else
                f"(lane_mask & ({base} < {compact_length}))")
        return [
            f"{indent}perm_addr{digit} = {address}",
            f"{indent}tl.store(out_ptr + perm_addr{digit} * 2, r{digit}, "
            f"mask={mask})",
            f"{indent}tl.store(out_ptr + perm_addr{digit} * 2 + 1, i{digit}, "
            f"mask={mask})",
        ]
    if (_hcu_backend_active() and len(factors) > 1 and compact_length is None
            and os.getenv("FLAGFFT_HCU_3D_DIRECT_STORE", "0") == "1"):
        address = f"(output_base + {offset}) * perm_k_stride + perm_gbase_vector"
        return [
            f"{indent}perm_addr{digit} = {address}",
            f"{indent}perm_pair{digit} = tl.join(r{digit}, i{digit})",
            f"{indent}perm_pair_addr{digit} = perm_addr{digit}[:, None] * 2 + tl.arange(0, 2)[None, :]",
            f"{indent}tl.store(out_ptr + perm_pair_addr{digit}, perm_pair{digit}, mask=lane_mask[:, None])",
        ]
    if _ix_backend_active() and os.getenv("FLAGFFT_IX_3D_DIRECT_STORE") == "1":
        # Keep the register tile in its original (batch, lane) layout.  This
        # screens whether IX's lowering of tl.trans costs more than scalar
        # stores for this kernel; the row-to-output mapping is unchanged.
        address = f"(output_base + {offset}) * perm_k_stride + perm_gbase_vector"
        return [
            f"{indent}perm_addr{digit} = {address}",
            f"{indent}tl.store(out_ptr + perm_addr{digit} * 2, r{digit}, mask=lane_mask)",
            f"{indent}tl.store(out_ptr + perm_addr{digit} * 2 + 1, i{digit}, mask=lane_mask)",
        ]
    else:
        address = f"{base}[:, None] * perm_k_stride + perm_gbase[None, :]"
        mask = ("perm_store_mask" if compact_length is None else
                f"(perm_store_mask & ({base} < {compact_length})[:, None])")
    pair_store = (
        (_mthreads_backend_active() and os.getenv("FLAGFFT_MUSA_3D_PAIR_STORE", "1") == "1")
        or (_hcu_backend_active() and os.getenv("FLAGFFT_HCU_3D_PAIR_STORE", "1") == "1")
        or (_ix_backend_active() and dtype == "complex64"
            and os.getenv("FLAGFFT_IX_3D_PAIR_STORE", "0") == "1")
    )
    if vector_store:
        return [
            f"{indent}zr{digit} = tl.trans(tl.reshape(r{digit}, ({pack}, {lane_block})))",
            f"{indent}zi{digit} = tl.trans(tl.reshape(i{digit}, ({pack}, {lane_block})))",
            f"{indent}perm_addr{digit} = {address}",
        ] + _emit_vectorized_complex_store(
            indent, f"out_ptr + perm_addr{digit} * 2", f"zr{digit}", f"zi{digit}", mask, dtype
        )
    if pair_store:
        return [
            f"{indent}zr{digit} = tl.trans(tl.reshape(r{digit}, ({pack}, {lane_block})))",
            f"{indent}zi{digit} = tl.trans(tl.reshape(i{digit}, ({pack}, {lane_block})))",
            f"{indent}perm_addr{digit} = {address}",
            f"{indent}pair{digit} = tl.join(zr{digit}, zi{digit})",
            f"{indent}pair_addr{digit} = perm_addr{digit}[:, :, None] * 2 + "
            "tl.arange(0, 2)[None, None, :]",
            f"{indent}tl.store(out_ptr + pair_addr{digit}, pair{digit}, "
            f"mask={mask}[:, :, None])",
        ]
    if _packed_maca_fp32_complex_io(dtype) and math.prod(factors) == 256:
        # The cube final-store path was paying for separate shared-layout
        # conversions of real and imaginary values before combining them.
        # Pack the complex bits per thread first, then transpose one 64-bit
        # tile so the cross-lane conversion moves half as many elements.
        return [
            f"{indent}perm_pair_flat{digit} = "
            f"tl.cast(tl.cast(r{digit}, tl.uint32, bitcast=True), tl.uint64) "
            f"| (tl.cast(tl.cast(i{digit}, tl.uint32, bitcast=True), tl.uint64) << 32)",
            f"{indent}perm_pair{digit} = tl.trans(tl.reshape(perm_pair_flat{digit}, "
            f"({pack}, {lane_block})))",
            f"{indent}perm_addr{digit} = {address}",
            f"{indent}perm_ptr{digit} = tl.cast(out_ptr + perm_addr{digit} * 2, "
            "tl.pointer_type(tl.uint64))",
            f"{indent}tl.store(perm_ptr{digit}, perm_pair{digit}, mask={mask})",
        ]
    if _packed_maca_fp32_complex_io(dtype):
        return [
            f"{indent}zr{digit} = tl.trans(tl.reshape(r{digit}, ({pack}, {lane_block})))",
            f"{indent}zi{digit} = tl.trans(tl.reshape(i{digit}, ({pack}, {lane_block})))",
            f"{indent}perm_addr{digit} = {address}",
            f"{indent}perm_pair{digit} = "
            f"tl.cast(tl.cast(zr{digit}, tl.uint32, bitcast=True), tl.uint64) "
            f"| (tl.cast(tl.cast(zi{digit}, tl.uint32, bitcast=True), tl.uint64) << 32)",
            f"{indent}perm_ptr{digit} = tl.cast(out_ptr + perm_addr{digit} * 2, "
            "tl.pointer_type(tl.uint64))",
            f"{indent}tl.store(perm_ptr{digit}, perm_pair{digit}, mask={mask})",
        ]
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
    indent: str,
    buffer: str,
    index: str,
    digit: int,
    portable: bool,
    direct: bool = False,
    mixed_direct: bool = False,
) -> list[str]:
    if mixed_direct:
        return [
            f"{indent}r{digit} = tl.where(lane_mask, {buffer}_register_r{digit}, 0.0)",
            f"{indent}i{digit} = tl.where(lane_mask, {buffer}_register_i{digit}, 0.0)",
        ]
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
        output_axes = [
            *range(stage + 1, len(factors)),
            stage,
            *range(stage - 1, -1, -1),
        ]
    shape = (pack, *(factors[axis] for axis in source_axes))
    permutation = (0, *(source_axes.index(axis) + 1 for axis in output_axes))
    lines = []
    for component in ("r", "i"):
        name = f"exchange_structured_{component}"
        joined = _distributed_join_tree(
            [f"exchange_{component}{digit}" for digit in range(radix)]
        )
        if interleaved:
            lines.append(
                f"    {name} = tl.reshape({joined}, ({lane_block}, {pack}, {radix}))"
            )
            lines.append(f"    {name} = tl.trans({name}, (1, 0, 2))")
        else:
            lines.append(
                f"    {name} = tl.reshape({joined}, ({pack}, {lane_block}, {radix}))"
            )
        if padding > 1:
            lines.append(
                f"    {name} = tl.reshape({name}, ({pack}, {padding}, {lanes}, {radix}))"
            )
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
        lines.append(
            f"    {prefix} = tl.reshape({buffer}_{component}, ({pack}, {radix}, {lanes}))"
        )
        lines.append(f"    {prefix} = tl.trans({prefix}, (0, 2, 1))")
        # Pad the whole register bank before splitting.  Padding each register
        # separately makes the MACA backend emit two barriers per component
        # and radix output (64 extra barriers on the 2048-point radix-16 stage).
        padded_lanes = lanes
        while padded_lanes < lane_block:
            lines.append(f"    {prefix} = tl.join({prefix}, tl.zeros_like({prefix}))")
            lines.append(f"    {prefix} = tl.trans({prefix}, (0, 3, 1, 2))")
            padded_lanes *= 2
            lines.append(
                f"    {prefix} = tl.reshape({prefix}, ({pack}, {padded_lanes}, {radix}))"
            )
        lines.append(f"    {prefix} = tl.reshape({prefix}, {split_shape})")
        if highest_first:
            # A cross-lane radix bit can otherwise remain until the last
            # split, causing the compiler to convert each two-register pair
            # separately.  Split that high bit while the bank is still whole.
            permutation = (0, 1, *range(len(split_shape) - 1, 1, -1))
            lines.append(f"    {prefix} = tl.trans({prefix}, {permutation})")
        lines.extend(
            _emit_distributed_split_tree(
                "    ",
                prefix,
                [f"{prefix}{digit}" for digit in range(radix)],
                prefix,
                highest_first=highest_first,
            )
        )
        for digit in range(radix):
            name = f"{prefix}{digit}"
            if interleaved:
                lines.append(f"    {name} = tl.trans({name}, (1, 0))")
            lines.append(f"    {name} = tl.reshape({name}, ({pack * lane_block},))")
    return lines


def _mixed_direct_exchange_supported(
    stage: int,
    factors: tuple[int, ...],
    lane_block: int,
    size: int,
    slot_stride: int,
    pack: int,
    register_lane_stride: int,
    register_slot_stride: int,
) -> bool:
    """Opt in only at complete, unswizzled MACA mixed-stage boundaries.

    Producer and consumer must use this same predicate.  Slot padding belongs
    to the discarded leaf buffer; register packing must still be canonical.
    Unknown knob values deliberately retain the existing exchange.
    """
    n = math.prod(factors)
    return (
        _maca_backend_active()
        and _maca_knob("MIXED_EXCHANGE") == "direct"
        and 0 <= stage < len(factors) - 1
        and any(factor & (factor - 1) for factor in factors)
        and all(factor > 0 for factor in factors)
        and pack > 0 and pack & (pack - 1) == 0
        and lane_block > 0 and lane_block & (lane_block - 1) == 0
        and lane_block >= max(n // factors[stage], n // factors[stage + 1])
        and slot_stride >= n
        and size >= slot_stride * pack and size & (size - 1) == 0
        and (register_lane_stride, register_slot_stride)
        in {(1, lane_block), (pack, 1)}
    )


def _emit_mixed_direct_exchange_registers(
    buffer: str,
    stage: int,
    factors: tuple[int, ...],
    lane_block: int,
    pack: int,
    register_lane_stride: int,
    register_slot_stride: int,
) -> list[str]:
    """Compose next-stage loads with the inverse route before gathering.

    Decode the next codelet's lane once.  Only an affine offset depends on
    its radix digit, keeping source growth linear in the two stage radices.
    No full-leaf routed tensor or per-digit reconstruction of the join is
    needed.  Padded output lanes gather index zero and are explicitly zeroed.
    """
    radix = factors[stage]
    next_radix = factors[stage + 1]
    next_lanes = math.prod(factors) // next_radix
    joined_radix = 1 << (radix - 1).bit_length()
    vector_block = lane_block * pack
    interleaved = register_lane_stride > 1
    lines = [
        f"    exchange_target = tl.arange(0, {vector_block})",
        f"    exchange_slot = exchange_target {'%' if interleaved else '//'} "
        f"{pack if interleaved else lane_block}",
        f"    exchange_rem = exchange_target {'//' if interleaved else '%'} "
        f"{pack if interleaved else lane_block}",
        f"    exchange_valid = exchange_rem < {next_lanes}",
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
    lines.append(
        f"    exchange_joined_base = (exchange_codelet * {register_lane_stride} + "
        f"exchange_slot * {register_slot_stride}) * {joined_radix} + exchange_digit"
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
    for digit in range(next_radix):
        offset = digit * stride * register_lane_stride * joined_radix
        lines.append(
            f"    exchange_joined_index = tl.where(exchange_valid, "
            f"exchange_joined_base + {offset}, 0)"
        )
        for component in ("r", "i"):
            lines.append(
                f"    {buffer}_register_{component}{digit} = tl.where(exchange_valid, "
                f"tl.gather(exchange_joined_{component}, exchange_joined_index, 0), 0.0)"
            )
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
    allow_mixed_direct: bool = True,
    next_lane_block: int | None = None,
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
        allow_mixed_direct
        and not natural_order
        and _mixed_direct_exchange_supported(
            stage, factors, lane_block, size, slot_stride, pack,
            register_lane_stride, register_slot_stride,
        )
    ):
        return _emit_mixed_direct_exchange_registers(
            buffer, stage, factors, lane_block, pack,
            register_lane_stride, register_slot_stride,
        )
    if (
        _maca_knob("EXCHANGE") in {"transpose", "direct", "direct_all"}
        and _structured_exchange_supported(factors, size, slot_stride, pack)
        and lane_block >= n // radix
    ):
        lines = _emit_structured_portable_exchange(
            buffer,
            stage,
            factors,
            lane_block,
            pack,
            natural_order,
            interleaved=register_lane_stride > 1,
        )
        if _maca_knob("EXCHANGE") in {"direct", "direct_all"} and not natural_order:
            lines.extend(
                _emit_direct_exchange_registers(
                    buffer,
                    factors[stage + 1],
                    n,
                    next_lane_block if next_lane_block is not None else lane_block,
                    pack,
                    interleaved=register_lane_stride > 1,
                )
            )
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


_IX_DISJOINT_MIXED_LENGTHS = {221, 425, 504, 675, 780, 1500, 1560}


def _emit_ix_packed_c2r_radix2(n: int) -> list[str]:
    # Fuse the Hermitian reconstruction with the first radix-two stage.
    # The fixed Stockham route places its two digits at 2*k and 2*k+1.
    half = n // 2
    shift = int(_maca_knob("SMEM_SWIZZLE_SHIFT", "3"))
    swizzle = _maca_knob("SMEM_SWIZZLE", "0") == "1"
    packed_io = _packed_maca_fp32_complex_io("complex64")
    lines = [
        f"    prepare_k = tl.arange(0, {half // 2}) + 1",
        f"    prepare_q = {half} - prepare_k",
        "    prepare_mask = batch_id < nbatch",
        "    prepare_partner_mask = prepare_mask & (prepare_k != prepare_q)",
    ]
    for frequency, name in (
        ("prepare_k", "a"),
        (f"prepare_k + {half}", "b"),
        ("prepare_q", "c"),
        (f"prepare_q + {half}", "d"),
    ):
        lines.extend(_emit_complex_boundary_load(
            "    ", f"in_ptr + 2 * (input_batch_base + {frequency})",
            "prepare_mask", f"{name}r", f"{name}i", "complex64", packed_io))
    lines.extend(_emit_complex_boundary_load(
        "    ", "packed_twiddle_ptr + 2 * prepare_k", "prepare_mask", "wr", "wi",
        "complex64", packed_io))
    lines.extend([
        "    t_r, t_i = _cmul(ar - dr, ai + di, wr, -wi)",
        "    u_r, u_i = _cmul(br - cr, bi + ci, wi, wr)",
        "    k0_r, k0_i = ar + dr - t_i, ai - di + t_r",
        "    k1_r, k1_i = br + cr - u_i, bi - ci + u_r",
        "    q0_r, q0_i = br + cr + u_i, ci - bi + u_r",
        "    q1_r, q1_i = ar + dr + t_i, di - ai + t_r",
    ])
    for variant, mask in (("k", "prepare_mask"), ("q", "prepare_partner_mask")):
        for digit, op in ((0, "+"), (1, "-")):
            index = f"prepare_{variant}_address{digit}"
            lines.append(f"    {index} = 2 * prepare_{variant} + {digit}")
            address = f"{index} ^ ({index} >> {shift})" if swizzle else index
            for component in ("r", "i"):
                lines.append(
                    f"    tl.store(tle.gpu.local_ptr(smem_b_{component}, ({address},)), "
                    f"{variant}0_{component} {op} {variant}1_{component}, {mask})"
                )
    lines.extend([
        "    dc = tl.load(in_ptr + 2 * input_batch_base, prepare_mask, 0.0)",
        f"    nyquist = tl.load(in_ptr + 2 * (input_batch_base + {n}), prepare_mask, 0.0)",
        f"    middle_r = 2.0 * tl.load(in_ptr + 2 * (input_batch_base + {half}), prepare_mask, 0.0)",
        f"    middle_i = -2.0 * tl.load(in_ptr + 2 * (input_batch_base + {half}) + 1, prepare_mask, 0.0)",
        "    tl.store(tle.gpu.local_ptr(smem_b_r, (0,)), dc + nyquist + middle_r, prepare_mask)",
        "    tl.store(tle.gpu.local_ptr(smem_b_i, (0,)), dc - nyquist + middle_i, prepare_mask)",
        "    tl.store(tle.gpu.local_ptr(smem_b_r, (1,)), dc + nyquist - middle_r, prepare_mask)",
        "    tl.store(tle.gpu.local_ptr(smem_b_i, (1,)), dc - nyquist - middle_i, prepare_mask)",
        "    tl.debug_barrier()",
    ])
    return lines


def _emit_ix_twiddle_tree(indent: str, stage: int, digit: int) -> list[str]:
    if digit == 1:
        return [
            f"{indent}tw_r{digit} = tl.load(tw{stage}_r_ptr + logical_phys{digit}, lane_mask, 0.0)",
            f"{indent}tw_i{digit} = tl.load(tw{stage}_i_ptr + logical_phys{digit}, lane_mask, 0.0)",
        ]
    left = digit // 2
    right = digit - left
    return [f"{indent}tw_r{digit}, tw_i{digit} = _cmul(tw_r{left}, tw_i{left}, tw_r{right}, tw_i{right})"]


def _emit_ix_r2c_finish_pairs(n: int, lanes: int, stage: int,
                              swizzle: bool, shift: int) -> list[str]:
    half = n // 2
    body = ["    partner_mask = finish_store_mask & (partner_k != k)"]
    for prefix in ("a", "b"):
        for digit in (0, 1):
            index = f"{prefix}{digit}_index"
            body.append(f"    {index} = {prefix}_offset + {digit * lanes}")
            address = f"{index} ^ ({index} >> {shift})" if swizzle else index
            for component in ("r", "i"):
                body.append(
                    f"    {prefix}{digit}_{component} = tl.load(tle.gpu.local_ptr("
                    f"smem_b_{component}, ({address},)), finish_mask, 0.0)"
                )
    square_rotation = n == 8192 and _maca_knob("R2C_FINISH_SQUARE_ROTATION", "0") == "1"
    if square_rotation:
        # The last radix-2 butterfly uses W_n^k. The real reconstruction
        # already needs W_(2n)^k, so squaring it removes the stage-table reads.
        body.extend([
            "    wr = tl.load(packed_twiddle_ptr + 2 * k, finish_mask, 0.0)",
            "    wi = tl.load(packed_twiddle_ptr + 2 * k + 1, finish_mask, 0.0)",
            "    twr = wr * wr - wi * wi",
            "    twi = (wr + wr) * wi",
        ])
    else:
        body.extend([
            f"    twr = tl.load(tw{stage}_r_ptr + a1_index, finish_mask, 0.0)",
            f"    twi = tl.load(tw{stage}_i_ptr + a1_index, finish_mask, 0.0)",
        ])
    body.extend([
        "    a1_r, a1_i = _cmul(a1_r, a1_i, twr, twi)",
        "    b1_r, b1_i = _cmul(b1_r, b1_i, -twr, twi)",
        "    ar, ai = a0_r + a1_r, a0_i + a1_i",
        "    br, bi = a0_r - a1_r, a0_i - a1_i",
        "    cr, ci = b0_r + b1_r, b0_i + b1_i",
        "    dr, di = b0_r - b1_r, b0_i - b1_i",
    ])
    if not square_rotation:
        body.extend([
            "    wr = tl.load(packed_twiddle_ptr + 2 * k, finish_mask, 0.0)",
            "    wi = tl.load(packed_twiddle_ptr + 2 * k + 1, finish_mask, 0.0)",
        ])
    for value, partner, frequency, wr, wi, mask in (
        ("a", "d", "k", "wr", "wi", "finish_store_mask"),
        ("b", "c", f"k + {half}", "wi", "-wr", "finish_store_mask"),
        ("c", "b", "partner_k", "-wi", "-wr", "partner_mask"),
        ("d", "a", f"partner_k + {half}", "-wr", "wi", "partner_mask"),
    ):
        body.extend([
            f"    diff_r = {value}r - {partner}r",
            f"    sum_i = {value}i + {partner}i",
            f"    xr_{value} = 0.5 * ({value}r + {partner}r + ({wi}) * diff_r + ({wr}) * sum_i)",
            f"    xi_{value} = 0.5 * ({value}i - {partner}i - ({wr}) * diff_r + ({wi}) * sum_i)",
        ])
        ptr = f"finish_out + 2 * ({frequency})"
        if _packed_maca_fp32_complex_io("complex64"):
            body.extend(_emit_vectorized_complex_store(
                "    ", ptr, f"xr_{value}", f"xi_{value}", mask, "complex64"))
        else:
            body.extend([
                f"    tl.store({ptr}, xr_{value}, {mask})",
                f"    tl.store({ptr} + 1, xi_{value}, {mask})",
            ])
    return body


def _emit_ix_flat_r2c_finish(n: int, lanes: int) -> list[str]:
    # One lane produces k and k+n/2. This gives the real finish two
    # contiguous output bands and a single masked Nyquist store.
    half = n // 2
    body = [
        "    lane_mask = batch_id < nbatch",
        f"    k = tl.arange(0, {half})",
        f"    q = (-k) & {half - 1}",
        "    finish_out = out_ptr + 2 * batch_id * output_distance",
    ]
    for name, index in (("a", "k"), ("b", "q")):
        for digit in (0, 1):
            logical = f"{name}{digit}_index"
            body.append(f"    {logical} = {index} + ({index} & {-lanes}) + {digit*lanes}")
            for component in ("r", "i"):
                body.append(f"    {name}{digit}_{component} = tl.load("
                            f"tle.gpu.local_ptr(smem_b_{component}, ({logical},)), lane_mask, 0.0)")
    body.extend([
        "    wr = tl.load(packed_twiddle_ptr + 2 * k, lane_mask, 0.0)",
        "    wi = tl.load(packed_twiddle_ptr + 2 * k + 1, lane_mask, 0.0)",
    ])
    # The radix-two rotation is the square of the reconstruction root.
    body.extend(["    twr = wr * wr - wi * wi", "    twi = (wr + wr) * wi"])
    body.extend([
        "    a1_r, a1_i = _cmul(a1_r, a1_i, twr, twi)",
        "    b1_r, b1_i = _cmul(b1_r, b1_i, -twr, twi)",
        "    ar, ai = a0_r + a1_r, a0_i + a1_i",
        "    br, bi = a0_r - a1_r, a0_i - a1_i",
        "    cr, ci = b0_r + b1_r, b0_i + b1_i",
        "    dr, di = b0_r - b1_r, b0_i - b1_i",
    ])
    for a, b, wr, wi, offset, edge_r, edge_i in (
        ("a", "d", "wr", "wi", 0, "ar + ai", "0.0"),
        ("b", "c", "wi", "-wr", half, "br", "-bi"),
    ):
        body.extend([
            f"    diff_r = {a}r - {b}r",
            f"    sum_i = {a}i + {b}i",
            f"    out_r = 0.5 * ({a}r + {b}r + ({wi}) * diff_r + ({wr}) * sum_i)",
            f"    out_i = 0.5 * ({a}i - {b}i - ({wr}) * diff_r + ({wi}) * sum_i)",
            f"    out_r = tl.where(k == 0, {edge_r}, out_r)",
            f"    out_i = tl.where(k == 0, {edge_i}, out_i)",
        ])
        pointer = f"finish_out + 2 * (k + {offset})"
        if _packed_maca_fp32_complex_io("complex64"):
            body.extend(_emit_vectorized_complex_store("    ", pointer, "out_r", "out_i", "lane_mask", "complex64"))
        else:
            body.extend([f"    tl.store({pointer}, out_r, lane_mask)",
                         f"    tl.store({pointer} + 1, out_i, lane_mask)"])
    body.extend([
        f"    tl.store(finish_out + {2*n} + 2*k, ar - ai, lane_mask & (k == 0))",
        f"    tl.store(finish_out + {2*n+1} + 2*k, 0.0, lane_mask & (k == 0))",
    ])
    return body


def _emit_ix_packed_r2c_finish(n: int, lanes: int, stage: int) -> list[str]:
    # Both final butterflies for k and n/2-k feed the same packed real
    # reconstruction. Keep intermediate FFT values in the original tile.
    half = n // 2
    shift = int(_maca_knob("SMEM_SWIZZLE_SHIFT", "3"))
    swizzle = _maca_knob("SMEM_SWIZZLE", "0") == "1"
    body = [
        "    finish_mask = batch_id < nbatch",
        "    finish_out = out_ptr + 2 * batch_id * output_distance",
    ]
    if n == 8192:
        # This narrow tile retains the fast lowering away from group
        # boundaries. Compute the four boundary pairs with constant strides.
        body.extend([
            f"    k = tl.arange(0, {half // 2}) + 1",
            f"    partner_k = {half} - k",
            f"    a_offset = k + {lanes} * (k // {lanes})",
            f"    b_offset = partner_k + {lanes} * (partner_k // {lanes})",
            f"    finish_store_mask = finish_mask & ((k & {lanes - 1}) != 0)",
        ])
        body.extend(_emit_ix_r2c_finish_pairs(n, lanes, stage, swizzle, shift))
        body.extend([
            f"    k = (tl.arange(0, {half // (2 * lanes)}) + 1) * {lanes}",
            f"    partner_k = {half} - k",
            "    a_offset = 2 * k",
            "    b_offset = 2 * partner_k",
        ])
    else:
        body.extend([
            f"    k_lane = tl.arange(0, {half // 2})",
            f"    k = tl.where(k_lane == 0, {half // 2}, k_lane)",
            f"    partner_k = {half} - k",
            f"    a_offset = k + (k & {-lanes})",
            f"    b_offset = partner_k + (partner_k & {-lanes})",
        ])
    body.append("    finish_store_mask = finish_mask")
    body.extend(_emit_ix_r2c_finish_pairs(n, lanes, stage, swizzle, shift))
    second = lanes ^ (lanes >> shift) if swizzle else lanes
    for digit, address in enumerate((0, second)):
        for component in ("r", "i"):
            body.append(
                f"    edge{digit}_{component} = tl.load(tle.gpu.local_ptr("
                f"smem_b_{component}, ({address},)), finish_mask, 0.0)"
            )
    body.extend([
        "    dc_r = edge0_r + edge1_r",
        "    dc_i = edge0_i + edge1_i",
        "    tl.store(finish_out, dc_r + dc_i, finish_mask)",
        "    tl.store(finish_out + 1, 0.0, finish_mask)",
        f"    tl.store(finish_out + {n}, edge0_r - edge1_r, finish_mask)",
        f"    tl.store(finish_out + {n + 1}, edge1_i - edge0_i, finish_mask)",
        f"    tl.store(finish_out + {2 * n}, dc_r - dc_i, finish_mask)",
        f"    tl.store(finish_out + {2 * n + 1}, 0.0, finish_mask)",
    ])
    return body


def _ix_full_cta_disjoint(n, factors, io_mode, prime_n):
    if ix_packed_real_tle(n, factors, io_mode):
        return _maca_knob("TLE_DISJOINT_CTA", "0") == "1"
    return (
        n == 16384 and factors == (16, 16, 16, 4)
        and (io_mode == "contiguous" or
             (prime_n == 8191 and io_mode in {"bluestein_prepare_leaf", "bluestein_finish_leaf"}))
        and _maca_knob("TLE_DISJOINT_CTA", "0") == "1"
    )


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
    bluestein_real_kind: str = "",
    permuted_smem_store: bool = False,
    perm_inner: bool = False,
    perm_batched: bool = False,
    packed_smem_alias: bool = False,
    ix_split_frequency: bool = False,
    ix_split_c2r: bool = False,
    ix_compact_tail: bool = False,
    ix_real_ct_codelet: bool = False,
    ix_dif_packed_real: bool = False,
    ix_packed_twiddles: bool = False,
    ix_packed_bank_shifts: tuple[int, ...] = (),
) -> list[str]:
    radix = factors[stage]
    current_lanes = stage_lanes[stage] if stage_lanes is not None else lanes
    ix_short_packed_c2r = (
        io_mode == "packed_c2r" and n == 1024 and not portable_exchange
        and ix_packed_real_tle(n, factors, io_mode)
    )
    next_lanes = (
        stage_lanes[stage + 1]
        if stage_lanes is not None and stage + 1 < len(stage_lanes)
        else current_lanes
    )
    groups = n // (current_lanes * radix)
    next_lane_block = None
    if ix_compact_tail:
        # Keep the first exchange at its existing width. The final radix can
        # use its exact lane count when the preceding exchange emits it.
        if stage == 2:
            lane_block = current_lanes
        elif stage == 1:
            next_lane_block = next_lanes
    compact_npu_stage = (portable_exchange and _npu_backend_active()
        and os.environ.get("FLAGFFT_NPU_COMPACT_STAGES") == "1"
        and any(factor & (factor-1) for factor in factors)
        and io_mode in {"contiguous", "contiguous_r2c", "contiguous_c2r",
                        "four_step_row", "four_step_col", "four_step_real_row",
                        "four_step_hermitian_row", "four_step_r2c_col", "four_step_c2r_col"})
    if compact_npu_stage:
        lane_block = 1 << (current_lanes-1).bit_length()
    npu_affine_tables = (portable_exchange and _npu_backend_active()
        and os.environ.get("FLAGFFT_NPU_AFFINE_MATRIX_IO") == "1")
    npu_matrix_io = (npu_affine_tables and inner_pack >= 4 and dtype == "complex64"
        and not fuse_twiddle_into_row and io_mode in {"four_step_row", "four_step_col"})
    npu_batch_io = (npu_affine_tables and dtype == "complex64"
        and io_mode in {"contiguous", "contiguous_r2c", "contiguous_c2r"})
    npu_trig_twiddles = npu_matrix_io and os.environ.get("FLAGFFT_NPU_TRIG_TWIDDLES") == "1"
    hcu_swizzle_override = os.getenv("FLAGFFT_HCU_3D_SMEM_SWIZZLE")
    hcu_swizzle_default = dtype == "complex128" and n >= 64
    hcu_swizzle_enabled = (
        hcu_swizzle_default
        if hcu_swizzle_override in (None, "auto")
        else hcu_swizzle_override == "1"
    )
    hcu_smem_swizzle = (
        _hcu_backend_active()
        and not portable_exchange
        and n & (n - 1) == 0
        and io_mode in {"permuted_store", "strided_permuted_store", "permuted_r2c"}
        and hcu_swizzle_enabled
    )
    hcu_r2c_leaf_swizzle = (
        _hcu_backend_active()
        and not portable_exchange
        and n >= 64
        and n & (n - 1) == 0
        and io_mode == "contiguous_r2c"
        and dtype == "complex128"
        # The default covers only the profiled 256-point FP64 R2C leaf.
        # Other lengths remain available for explicit screening via =1.
        and (
            os.getenv("FLAGFFT_HCU_3D_R2C_LEAF_SWIZZLE", "auto") == "1"
            or (
                os.getenv("FLAGFFT_HCU_3D_R2C_LEAF_SWIZZLE", "auto") == "auto"
                and n == 256
            )
        )
    )
    hcu_smem_swizzle = hcu_smem_swizzle or hcu_r2c_leaf_swizzle
    cuda_swizzle = os.getenv("FLAGFFT_CUDA_SMEM_SWIZZLE", "0")
    if current_profile().backend == "cuda" and cuda_swizzle not in {"0", "3", "4", "5"}:
        raise ValueError("FLAGFFT_CUDA_SMEM_SWIZZLE must be 0, 3, 4 or 5")
    cuda_smem_swizzle = (current_profile().backend == "cuda" and cuda_swizzle != "0" and
                        not portable_exchange and n & (n - 1) == 0)
    smem_swizzle = fuse_twiddle_into_row or hcu_smem_swizzle or cuda_smem_swizzle or (
        _ix_backend_active() and not portable_exchange and n & (n - 1) == 0
        and _maca_knob("SMEM_SWIZZLE", "0") == "1"
    )
    hcu_cyclic_smem = (
        _hcu_backend_active() and single_smem_buffer and smem_pack > 1
        and ((os.getenv("FLAGFFT_HCU_3D_STORE_MODE", "auto") == "cyclic"
              and io_mode in {"permuted_store", "strided_permuted_store",
                              "permuted_r2c", "permuted_store_r2c"})
             or (io_mode.startswith(("four_step_", "bluestein_four_step_"))
                 and os.getenv("FLAGFFT_HCU_FOURSTEP_SINGLE_BUFFER", "0") == "1")
             or (io_mode in {"contiguous", "strided", "contiguous_r2c", "contiguous_c2r",
                             "packed_r2c", "packed_c2r"}
                 and os.getenv("FLAGFFT_HCU_LEAF_SINGLE_BUFFER", "0") == "1"))
    )
    hcu_cyclic_slot = "inner_slot" if inner_pack > 1 else "batch_slot"
    if hcu_cyclic_smem:
        skew = os.getenv("FLAGFFT_HCU_CYCLIC_SKEW", "unit")
        if skew not in {"unit", "wave"}:
            raise ValueError("FLAGFFT_HCU_CYCLIC_SKEW must be unit or wave")
        if skew == "wave":
            # Adjacent threads own adjacent FFT slots after a direct store
            # remap, and throughout a FourStep tile. Shift each slot by one
            # wave's frequency span so that the shared read spans the banks.
            step = max(1, current_profile().warp_size // smem_pack)
            hcu_cyclic_slot = f"{hcu_cyclic_slot} * {step}"
    batch_interleave = _ix_batch_smem_interleave(io_mode, portable_exchange, smem_pack)
    smem_interleave = batch_interleave or (
        _ix_backend_active() and not portable_exchange and inner_pack > 1
        and (io_mode.startswith("four_step_") or io_mode.startswith("bluestein_four_step_"))
        and _maca_knob("SMEM_INTERLEAVE", "0") == "1"
    )
    interleave_slot = "batch_slot" if batch_interleave else "inner_slot"
    io_layout = (
        _ix_backend_active() and not portable_exchange and dtype=="complex64"
        and 64<=n<=256 and len(factors)==2 and inner_pack>1
        and all(radix & (radix-1)==0 for radix in factors)
        and all(n//(radix*width)==1 for radix,width in
                zip(factors,stage_lanes or (lanes,)*len(factors)))
        and (io_mode.startswith("four_step_") or io_mode.startswith("bluestein_four_step_"))
        and not io_mode.endswith("_strided")
        and _maca_knob("TLE_IO_LAYOUT","0")=="1"
    )
    if io_layout:
        smem_swizzle=False
        smem_interleave=True
    io_logical_shift=factors[0].bit_length()-1
    io_slot_shift=max(0,6-(smem_pack.bit_length()-1))
    io_slot_chunk=64//lane_block
    swizzle_shift = _TLE_SMEM_SWIZZLE_SHIFT
    if cuda_smem_swizzle:
        swizzle_shift = int(cuda_swizzle)
    if _ix_backend_active() and smem_swizzle:
        swizzle_shift = int(_maca_knob("SMEM_SWIZZLE_SHIFT", "3"))
        if not 1 <= swizzle_shift <= 8:
            raise ValueError("FLAGFFT_IX_SMEM_SWIZZLE_SHIFT must be in [1, 8]")
    if hcu_smem_swizzle:
        swizzle_shift = int(os.getenv("FLAGFFT_HCU_3D_SMEM_SWIZZLE_SHIFT", "5"))
        if not 1 <= swizzle_shift <= 8:
            raise ValueError("FLAGFFT_HCU_3D_SMEM_SWIZZLE_SHIFT must be in [1, 8]")
    is_last = stage == len(factors) - 1
    read_bank_shift = write_bank_shift = swizzle_shift if smem_swizzle else 0
    if ix_packed_bank_shifts:
        read_bank_shift = ix_packed_bank_shifts[stage - 1] if stage > 0 else 0
        write_bank_shift = ix_packed_bank_shifts[stage] if not is_last else 0
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
    if (io_mode in {"bluestein_full_leaf", "rader_full_leaf"} and bluestein_pass == 1
            and not single_smem_buffer and bluestein_intermediate_buffer == "smem_b"):
        # The pointwise result is still live in B. Starting the next FFT's
        # ping-pong in B corrupts unread inputs when stage zero has groups.
        # Reverse this FFT's buffer parity so each stage reads a distinct
        # buffer from the one it writes.
        source_buffer = {None: None, "smem_a": "smem_b", "smem_b": "smem_a"}[source_buffer]
        dest_buffer = {None: None, "smem_a": "smem_b", "smem_b": "smem_a"}[dest_buffer]
    zero = "0.0"

    # Swizzled/fixed-lane layouts do not have the canonical inverse route.
    # Gate both sides identically so fallback never reads undefined registers.
    mixed_direct_layout = (
        portable_exchange
        and not fuse_twiddle_into_row
        and tuple(stage_lanes or (lanes,) * len(factors))
        == tuple(n // factor for factor in factors)
    )
    mixed_direct_load = mixed_direct_layout and _mixed_direct_exchange_supported(
        stage - 1, factors, lane_block, exchange_size, exchange_slot_stride,
        smem_pack, inner_pack if inner_pack > 1 else 1,
        1 if inner_pack > 1 else lane_block,
    )

    lines: list[str] = []
    if (_hcu_backend_active() and is_last and len(factors) > 1 and smem_pack > 1
            and io_mode in {"permuted_store", "strided_permuted_store"}
            and os.getenv("FLAGFFT_HCU_3D_DIRECT_STORE", "0") == "1"):
        # Change the thread-to-FFT mapping at the shared-memory boundary.
        # Adjacent lanes now own adjacent FFT slots, so the final butterflies
        # write coalesced output directly without a register-tile transpose.
        lines.extend([
            f"    batch_slot = lane_vec % {smem_pack}",
            f"    lane = lane_vec // {smem_pack}",
            "    current_batch = " + ("perm_base + batch_slot * perm_span" if perm_inner else "batch_id + batch_slot"),
            f"    smem_offset = batch_slot * {exchange_slot_stride}",
            f"    base_lane_mask = (lane < {current_lanes}) & (current_batch < nbatch)",
            # The original mask used FFT-major lanes. Uniform stage lanes
            # skip the cooperative-mask refresh below, so update it here
            # after changing to slot-major lanes, including partial CTAs.
            "    lane_mask = base_lane_mask",
            "    perm_i0_vector = current_batch // perm_span",
            "    perm_i1_vector = current_batch - perm_i0_vector * perm_span",
        ])
        if perm_inner:
            if perm_batched:
                lines.append(f"    perm_gbase_vector = (perm_i0_vector // perm_outer) * ({n} * rows_per_transform) + perm_i1_vector * perm_outer + perm_i0_vector % perm_outer")
            else:
                lines.append("    perm_gbase_vector = perm_i1_vector * (nbatch // perm_span) + perm_i0_vector")
        else:
            lines.append(f"    perm_gbase_vector = perm_i0_vector * ({n} * perm_span) + perm_i1_vector")
    if ix_compact_tail:
        lines.extend([f"    lane = tl.arange(0, {lane_block})", f"    base_lane_mask = lane < {current_lanes}"])
    if compact_npu_stage:
        lines.append(f"    lane_vec = tl.arange(0, {lane_block*smem_pack})")
        if inner_pack > 1:
            width = four_step_n2 if io_mode in {"four_step_row","four_step_real_row","four_step_hermitian_row"} else four_step_n1
            lines += [f"    inner_slot = lane_vec % {inner_pack}",
                      f"    lane = lane_vec // {inner_pack}",
                      "    four_step_inner = four_step_inner_base + inner_slot",
                      f"    smem_offset = inner_slot * {exchange_slot_stride}",
                      f"    base_lane_mask = four_step_inner < {width}"]
        elif smem_pack > 1:
            lines += [f"    batch_slot = lane_vec // {lane_block}",
                      f"    lane = lane_vec - batch_slot * {lane_block}",
                      "    current_batch = batch_id + batch_slot",
                      f"    batch_base = current_batch * {n}",
                      f"    smem_offset = batch_slot * {exchange_slot_stride}",
                      "    base_lane_mask = current_batch < nbatch"]
            if io_mode in {"contiguous_r2c","contiguous_c2r"}:
                lines += ["    input_batch_base = current_batch * input_distance",
                          "    output_batch_base = current_batch * output_distance"]
        else:
            lines += ["    lane = lane_vec", "    base_lane_mask = lane >= 0"]
    if io_layout and stage==1 and io_mode.endswith("_row"):
        # Shared memory separates the two register mappings: adjacent threads
        # read adjacent input FFT slots, then write adjacent output frequencies.
        inner_count=four_step_n2 if io_mode.endswith("_row") else four_step_n1
        lines.extend([
            f"    inner_slot = lane_vec // {lane_block}",
            f"    lane = lane_vec % {lane_block}",
            "    four_step_inner = four_step_inner_base + inner_slot",
            f"    lane_mask = (lane < {current_lanes}) & (four_step_inner < {inner_count})",
            "    base_lane_mask = lane_mask",
            f"    smem_offset = inner_slot * {exchange_slot_stride}",
        ])
    if stage_lanes is not None:
        lines.append(f"    lane_mask = base_lane_mask & (lane < {current_lanes})")
    vector_io_allowed = not _non_nvidia_backend_active() or (
        _portable_complex_vector_io(dtype)
        and (_maca_knob("VEC_IO", "0") != "packed" or dtype == "complex64")
    )
    vectorized_four_step_complex_io = (
        (io_mode in {
            "four_step_row",
            "four_step_col",
            "four_step_r2c_col",
            "four_step_c2r_col",
            "four_step_hermitian_row",
        } or (io_mode in {'four_step_real_row','bluestein_four_step_prepare_row',
                          'bluestein_four_step_pointwise_row','bluestein_four_step_finish_col'}
              and _ix_backend_active() and _packed_maca_fp32_complex_io(dtype)))
        and vector_io_allowed
    )
    vectorized_complex_io = (
        io_mode in {"contiguous", "contiguous_c2r", "permuted_store"}
        or (io_mode == "contiguous_r2c" and _npu_backend_active()
            and os.environ.get("FLAGFFT_NPU_PORTABLE_VECTOR_IO") == "1")
        or (io_mode in {"contiguous_r2c","packed_c2r"} and _ix_backend_active()
            and _packed_maca_fp32_complex_io(dtype))
        or vectorized_four_step_complex_io
    ) and vector_io_allowed and (
        not _hcu_backend_active() or io_mode == "permuted_store" or _hcu_vector_io(dtype)
    )
    ix_prime_vector_io = (
        io_mode == "bluestein_full_leaf" and bluestein_real_kind == "r2c"
        and prime_n == 1009 and _ix_backend_active() and dtype == "complex64"
        and _packed_maca_fp32_complex_io(dtype)
        and _maca_knob("BLUESTEIN_LEAF_VEC_IO", "0") == "1"
    )
    ix_rader_vector_io = (
        io_mode == "rader_full_leaf" and _ix_backend_active()
        and _packed_maca_fp32_complex_io(dtype)
    )
    ix_column_twiddle_recurrence = (
        _ix_backend_active() and dtype == "complex64"
        and stage == 0 and io_mode == "four_step_col" and not fuse_twiddle_into_row
        and (four_step_n1, four_step_n2) in {
            (1024, 512), (504, 675), (1560, 425), (425, 1560),
        }
        and _maca_knob("FOURSTEP_COL_TWIDDLE_RECURRENCE", "0") == "1"
    )
    hcu_u64_load = (
        _hcu_backend_active() and dtype == "complex64" and io_mode == "permuted_store"
        and os.getenv("FLAGFFT_HCU_3D_U64_LOAD", "1") == "1"
    )
    vector_suffix = "f64" if _is_double_dtype(dtype) else "f32"
    vector_reg = "d" if _is_double_dtype(dtype) else "f"
    vector_dtype = "tl.float64" if _is_double_dtype(dtype) else "tl.float32"
    if portable_exchange:
        assert groups == 1
        lines.append(f"    group_{stage} = 0")
        indent = "    "
    else:
        # The bounded mixed-radix experiment enumerates every store group so
        # the LLVM dependence check can still require an acyclic function.
        mixed_disjoint = (_ix_backend_active() and dtype == "complex64"
                          and _maca_knob("TLE_DISJOINT_STAGE","0") == "1"
                          and ((n in _IX_DISJOINT_MIXED_LENGTHS
                                and _maca_knob("TLE_DISJOINT_MIXED","0") == "1")
                               or _ix_full_cta_disjoint(n, factors, io_mode, prime_n)))
        loop = "static_range" if mixed_disjoint else "range"
        lines.append(f"    for group_{stage} in tl.{loop}(0, {groups}):")
        indent = "        "

    for j in range(radix):
        lines.append(
            f"{indent}logical_phys{j} = tl.where(lane_mask, lane + "
            f"{current_lanes} * (group_{stage} * {radix} + {j}), 0)"
        )
        if io_layout:
            lines.append(
                f"{indent}phys{j} = (logical_phys{j} ^ (logical_phys{j} >> {io_logical_shift})) * {smem_pack} + "
                f"(inner_slot ^ (((logical_phys{j} >> {io_slot_shift}) * {io_slot_chunk}) & {smem_pack-1}))"
            )
        elif smem_interleave:
            lines.append(f"{indent}phys{j} = logical_phys{j} * {smem_pack} + {interleave_slot}")
        elif hcu_cyclic_smem:
            lines.append(f"{indent}phys{j} = ((logical_phys{j} + {hcu_cyclic_slot}) & {n - 1}) + smem_offset")
        elif smem_pack > 1:
            lines.append(f"{indent}phys{j} = logical_phys{j} + smem_offset")
        else:
            lines.append(f"{indent}phys{j} = logical_phys{j}")
        if read_bank_shift and stage > 0:
            lines.append(
                f"{indent}smem_phys{j} = logical_phys{j} ^ "
                f"(logical_phys{j} >> {read_bank_shift})"
            )
            if smem_interleave:
                lines.append(f"{indent}smem_phys{j} = smem_phys{j} * {smem_pack} + {interleave_slot}")
            elif hcu_cyclic_smem:
                lines.append(f"{indent}smem_phys{j} = ((smem_phys{j} + {hcu_cyclic_slot}) & {n - 1}) + smem_offset")
            elif smem_pack > 1:
                lines.append(f"{indent}smem_phys{j} += smem_offset")
    if stage == 0:
        lines.extend(_emit_input_base(indent, factors, current_lanes, f"group_{stage}"))
    if is_last:
        lines.extend(
            _emit_output_base(indent, factors, current_lanes, f"group_{stage}")
        )
        if io_mode in {"permuted_store", "strided_permuted_store", "permuted_r2c", "permuted_store_r2c"}:
            active_lanes = max(stage_lanes) if stage_lanes is not None else lanes
            lines.append(f"{indent}lane_only = tl.arange(0, {lane_block})")
            # The last stage may use fewer lanes than an earlier stage. The
            # packed store reshapes its lanes into (lane, batch_slot), so its
            # mask must match this stage's valid lanes or padded lanes can
            # overwrite other transforms' output addresses.
            lines.append(f"{indent}perm_lane_mask = lane_only < {current_lanes}")
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

    if npu_batch_io and stage == 0 and io_mode == "contiguous_c2r":
        compact_row = 1 << (2*(n//2+1)-1).bit_length()
        lines += [
            f"{indent}compact_rows = batch_id + tl.arange(0,{smem_pack})[:,None]",
            f"{indent}compact_cols = tl.arange(0,{compact_row})[None,:]",
            f"{indent}compact_input = tl.load(in_ptr + compact_rows*input_distance*2 + compact_cols, "
            f"(compact_rows < nbatch) & (compact_cols < {2*(n//2+1)}), 0.0)",
            f"{indent}compact_input = tl.reshape(compact_input,({smem_pack*compact_row},))",
        ]
    if ix_column_twiddle_recurrence:
        twiddle_stride = _time_major_stride(factors, 0) * four_step_n1
        lines.extend(_emit_complex_boundary_load(
            indent, f"twiddle_ptr + 2 * ({twiddle_stride} + four_step_inner)",
            "lane_mask", "col_step_r", "col_step_i", dtype, vectorized_complex_io))
    for j in range(radix):
        if stage == 0:
            lines.extend(_emit_input_index(indent, f"in{j}", factors, j))
            if npu_batch_io:
                offset = j * math.prod(factors[1:])
                if io_mode == "contiguous_c2r":
                    half_n = n//2+1
                    slot = "batch_slot" if smem_pack > 1 else "0"
                    nyquist = f" | (in{j} == {n//2})" if n%2 == 0 else ""
                    lines += [
                        f"{indent}compact_freq = tl.where(in{j} < {half_n}, in{j}, {n}-in{j})",
                        f"{indent}compact_lookup = tl.where(lane_mask, ({slot})*{compact_row}+compact_freq*2, 0)",
                        f"{indent}r{j} = tl.gather(compact_input, compact_lookup, 0)",
                        f"{indent}i{j} = tl.gather(compact_input, compact_lookup+1, 0)",
                        f"{indent}i{j} = tl.where(in{j} < {half_n}, i{j}, -i{j})",
                        f"{indent}i{j} = tl.where((in{j} == 0){nyquist}, 0.0, i{j})",
                    ]
                else:
                    components = 1 if io_mode == "contiguous_r2c" else 2
                    distance = "input_distance" if io_mode == "contiguous_r2c" else str(n)
                    lines += [
                        f"{indent}batch_rows = batch_id + tl.arange(0,{smem_pack})[:,None]",
                        f"{indent}batch_cols = tl.arange(0,{lane_block*components})[None,:]",
                        f"{indent}batch_input{j} = tl.load(in_ptr + batch_rows*({distance})*{components} "
                        f"+ batch_cols+{offset*components}, (batch_rows < nbatch) & (batch_cols < {current_lanes*components}), 0.0)",
                    ]
                    if components == 1:
                        lines += [f"{indent}r{j} = tl.reshape(batch_input{j},({smem_pack*lane_block},))",
                                  f"{indent}i{j} = tl.zeros_like(r{j})"]
                    else:
                        lines.append(f"{indent}r{j}, i{j} = tl.split(tl.reshape(batch_input{j},({smem_pack*lane_block},2)))")
                continue
            if npu_matrix_io:
                width = four_step_n2 if io_mode == "four_step_row" else four_step_n1
                offset = j * math.prod(factors[1:])
                lines.extend(_emit_npu_matrix_load(indent, "in_ptr", f"matrix_input{j}",
                    f"r{j}", f"i{j}", lane_block, inner_pack, current_lanes, offset, width,
                    "four_step_batch_base"))
                if io_mode == "four_step_col":
                    if npu_trig_twiddles:
                        scale = _direction_sign(direction)*2*math.pi/(four_step_n1*four_step_n2)
                        lines += [
                            f"{indent}outer_angle{j} = in{j}.to(tl.float32)*four_step_inner.to(tl.float32)*{scale:.17g}",
                            f"{indent}tw_r{j} = tl.cos(outer_angle{j})",
                            f"{indent}tw_i{j} = tl.sin(outer_angle{j})",
                        ]
                    else:
                        lines.extend(_emit_npu_matrix_load(indent, "twiddle_ptr", f"matrix_phase{j}",
                            f"tw_r{j}", f"tw_i{j}", lane_block, inner_pack, current_lanes, offset, width, "0"))
                    lines.append(f"{indent}r{j}, i{j} = _cmul(r{j}, i{j}, tw_r{j}, tw_i{j})")
                continue
            if ix_split_frequency:
                other = f"({n} - in{j})" if ix_split_c2r else f"(in{j} + {n})"
                for tag, index in [("a", f"in{j}"), ("b", other),
                                   ("root", f"(in{j} + split_band * {n})")]:
                    ptr = "split_roots_ptr" if tag == "root" else "in_ptr"
                    lines.extend([
                        f"{indent}pair = tl.load(tl.cast({ptr}, tl.pointer_type(tl.uint64)) + {index}, lane_mask, 0)",
                        f"{indent}{tag}r = tl.cast(tl.cast(pair, tl.uint32), tl.float32, bitcast=True)",
                        f"{indent}{tag}i = tl.cast(tl.cast(pair >> 32, tl.uint32), tl.float32, bitcast=True)",
                    ])
                if ix_split_c2r:
                    lines.extend([
                        f"{indent}ai = tl.where(in{j} == 0, 0.0, ai)",
                        f"{indent}bi = tl.where(in{j} == 0, 0.0, -bi)",
                    ])
                lines.extend([
                    f"{indent}sr = ar + tl.where(split_band == 0, br, -br)",
                    f"{indent}si = ai + tl.where(split_band == 0, bi, -bi)",
                    f"{indent}r{j}, i{j} = _cmul(sr, si, rootr, rooti)",
                ])
            elif io_mode in {"contiguous", "permuted_store"}:
                if hcu_u64_load:
                    lines.append(
                        f"{indent}packed{j} = tl.load(tl.cast(in_ptr, tl.pointer_type(tl.uint64)) "
                        f"+ batch_base + in{j}, mask=lane_mask, other=0)"
                    )
                    lines.append(
                        f"{indent}r{j} = tl.cast(tl.cast(packed{j} & 0xffffffff, tl.uint32), "
                        "tl.float32, bitcast=True)"
                    )
                    lines.append(
                        f"{indent}i{j} = tl.cast(tl.cast(packed{j} >> 32, tl.uint32), "
                        "tl.float32, bitcast=True)"
                    )
                elif vectorized_complex_io:
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
            elif io_mode in {"strided", "strided_permuted_store"}:
                stride = "perm_span" if io_mode == "strided_permuted_store" else "outer_stride"
                lines.append(
                    f"{indent}r{j} = tl.load(in_ptr + (batch_base + in{j} * {stride}) * 2, "
                    f"mask=lane_mask, other={zero})"
                )
                lines.append(
                    f"{indent}i{j} = tl.load(in_ptr + (batch_base + in{j} * {stride}) * 2 + 1, "
                    f"mask=lane_mask, other={zero})"
                )
            elif io_mode in {"contiguous_r2c", "permuted_r2c", "permuted_store_r2c"}:
                lines.append(
                    f"{indent}r{j} = tl.load(in_ptr + input_batch_base + in{j}, mask=lane_mask, other={zero})"
                )
                lines.append(f"{indent}i{j} = r{j} * 0.0")
            elif io_mode == "packed_r2c":
                if (_ix_backend_active() and _packed_maca_fp32_complex_io(dtype)
                        and _maca_knob("PACKED_R2C_INPUT", "0") == "1"):
                    # A packed real input contains adjacent even/odd FP32
                    # samples. Their pair has the same aligned eight-byte
                    # representation as the complex I/O load.
                    lines.extend(_emit_vectorized_complex_load(
                        indent, f"in_ptr + input_batch_base + 2 * in{j}",
                        "lane_mask", f"r{j}, i{j}", dtype))
                else:
                    lines.append(
                        f"{indent}r{j} = tl.load(in_ptr + input_batch_base + 2 * in{j}, mask=lane_mask, other={zero})"
                    )
                    lines.append(
                        f"{indent}i{j} = tl.load(in_ptr + input_batch_base + 2 * in{j} + 1, mask=lane_mask, other={zero})"
                    )
            elif io_mode == "packed_c2r":
                lines.extend(
                    [
                        f"{indent}packed_k{j} = tl.where(lane_mask, in{j}, 0)",
                        f"{indent}packed_q{j} = {n} - packed_k{j}",
                        f"{indent}packed_x_ptr{j} = in_ptr + (input_batch_base + packed_k{j}) * 2",
                        f"{indent}packed_q_ptr{j} = in_ptr + (input_batch_base + packed_q{j}) * 2",
                    ]
                )
                if ((_ix_backend_active() and _packed_maca_fp32_complex_io(dtype))
                        or (_hcu_backend_active() and _hcu_vector_io(dtype))):
                    for ptr,dest in ((f"packed_x_ptr{j}",f"packed_xr{j}, packed_xi{j}"),
                                     (f"packed_q_ptr{j}",f"packed_qr{j}, packed_qi{j}"),
                                     (f"packed_twiddle_ptr + packed_k{j} * 2",f"packed_wr{j}, packed_wi{j}")):
                        lines.extend(_emit_vectorized_complex_load(indent,ptr,"lane_mask",dest,dtype))
                    lines.append(f"{indent}packed_qi{j} = -packed_qi{j}")
                else:
                    lines.extend([
                        f"{indent}packed_xr{j} = tl.load(packed_x_ptr{j}, mask=lane_mask, other={zero})",
                        f"{indent}packed_xi{j} = tl.load(packed_x_ptr{j} + 1, mask=lane_mask, other={zero})",
                        f"{indent}packed_qr{j} = tl.load(packed_q_ptr{j}, mask=lane_mask, other={zero})",
                        f"{indent}packed_qi{j} = -tl.load(packed_q_ptr{j} + 1, mask=lane_mask, other={zero})",
                        f"{indent}packed_wr{j} = tl.load(packed_twiddle_ptr + packed_k{j} * 2, mask=lane_mask, other={zero})",
                        f"{indent}packed_wi{j} = tl.load(packed_twiddle_ptr + packed_k{j} * 2 + 1, mask=lane_mask, other={zero})",
                    ])
                if _ix_backend_active() and dtype == "complex64":
                    # DC and Nyquist are real even when the compact input
                    # contains arbitrary imaginary endpoint values.
                    lines.extend([
                        f"{indent}packed_xi{j} = tl.where(packed_k{j} == 0, 0.0, packed_xi{j})",
                        f"{indent}packed_qi{j} = tl.where(packed_k{j} == 0, 0.0, packed_qi{j})",
                    ])
                lines.extend([
                        f"{indent}packed_sum_r{j} = packed_xr{j} + packed_qr{j}",
                        f"{indent}packed_sum_i{j} = packed_xi{j} + packed_qi{j}",
                        f"{indent}packed_diff_r{j} = packed_xr{j} - packed_qr{j}",
                        f"{indent}packed_diff_i{j} = packed_xi{j} - packed_qi{j}",
                        f"{indent}packed_prod_r{j} = packed_diff_r{j} * packed_wr{j} + "
                        f"packed_diff_i{j} * packed_wi{j}",
                        f"{indent}packed_prod_i{j} = packed_diff_i{j} * packed_wr{j} - "
                        f"packed_diff_r{j} * packed_wi{j}",
                        f"{indent}r{j} = packed_sum_r{j} - packed_prod_i{j}",
                        f"{indent}i{j} = packed_sum_i{j} + packed_prod_r{j}",
                    ]
                )
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
            elif io_mode == "rader_prepare_leaf":
                lines.append(f"{indent}inv_idx{j} = tl.where(in{j} == 0, 0, {n} - in{j})")
                lines.append(
                    f"{indent}src_idx{j} = tl.load(idx_ptr + inv_idx{j}, mask=lane_mask, other=0)"
                )
                lines.append(
                    f"{indent}src_ptr{j} = in_ptr + (current_batch * {prime_n} + src_idx{j}) * 2"
                )
                lines.append(f"{indent}r{j} = tl.load(src_ptr{j}, mask=lane_mask, other={zero})")
                lines.append(f"{indent}i{j} = tl.load(src_ptr{j} + 1, mask=lane_mask, other={zero})")
            elif io_mode == "rader_finish_leaf":
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
                lines.append(f"{indent}r{j}, point_i{j} = _cmul(ar{j}, ai{j}, br{j}, bi{j})")
                lines.append(f"{indent}i{j} = -point_i{j}")
            elif io_mode == "bluestein_full_leaf":
                if bluestein_pass == 0:
                    lines.append(
                        f"{indent}prime_mask{j} = lane_mask & (in{j} < {prime_n})"
                    )
                    if bluestein_real_kind == "r2c":
                        lines.append(f"{indent}src_ptr{j} = in_ptr + current_batch * {prime_n} + in{j}")
                        lines.append(f"{indent}r{j} = tl.load(src_ptr{j}, mask=prime_mask{j}, other={zero})")
                        lines.append(f"{indent}i{j} = {zero}")
                    elif bluestein_real_kind == "c2r":
                        lines.append(f"{indent}compact_idx{j} = tl.minimum(in{j}, {prime_n} - in{j})")
                        lines.append(f"{indent}src_ptr{j} = in_ptr + (current_batch * {prime_n // 2 + 1} + compact_idx{j}) * 2")
                        lines.append(f"{indent}r{j} = tl.load(src_ptr{j}, mask=prime_mask{j}, other={zero})")
                        lines.append(f"{indent}loaded_i{j} = tl.load(src_ptr{j} + 1, mask=prime_mask{j}, other={zero})")
                        lines.append(f"{indent}i{j} = tl.where(in{j} == 0, 0.0, tl.where(in{j} <= {prime_n // 2}, loaded_i{j}, -loaded_i{j}))")
                    else:
                        lines.append(f"{indent}src_ptr{j} = in_ptr + (current_batch * {prime_n} + in{j}) * 2")
                        lines.append(f"{indent}r{j} = tl.load(src_ptr{j}, mask=prime_mask{j}, other={zero})")
                        lines.append(f"{indent}i{j} = tl.load(src_ptr{j} + 1, mask=prime_mask{j}, other={zero})")
                    lines.extend(_emit_complex_boundary_load(
                        indent, f"chirp_ptr + in{j} * 2", f"prime_mask{j}",
                        f"chirp_r{j}", f"chirp_i{j}", dtype, ix_prime_vector_io,
                    ))
                    lines.append(f"{indent}r{j}, i{j} = _cmul(r{j}, i{j}, chirp_r{j}, chirp_i{j})")
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
            elif io_mode == "rader_full_leaf":
                if bluestein_pass == 0:
                    lines.append(f"{indent}inv_idx{j} = tl.where(in{j} == 0, 0, {n} - in{j})")
                    lines.append(
                        f"{indent}src_idx{j} = tl.load(idx_ptr + inv_idx{j}, mask=lane_mask, other=0)"
                    )
                    if bluestein_real_kind == "c2r":
                        half = prime_n // 2 + 1
                        lines.extend([
                            f"{indent}reflected{j} = src_idx{j} >= {half}",
                            f"{indent}compact_idx{j} = tl.where(reflected{j}, {prime_n} - src_idx{j}, src_idx{j})",
                            f"{indent}src_ptr{j} = in_ptr + (current_batch * {half} + compact_idx{j}) * 2",
                        ])
                    else:
                        lines.append(
                            f"{indent}src_ptr{j} = in_ptr + (current_batch * {prime_n} + src_idx{j}) * 2"
                        )
                    lines.extend(_emit_complex_boundary_load(
                        indent, f"src_ptr{j}", "lane_mask", f"r{j}", f"i{j}", dtype,
                        ix_rader_vector_io,
                    ))
                    if bluestein_real_kind == "c2r":
                        lines.append(f"{indent}i{j} = tl.where(reflected{j}, -i{j}, i{j})")
                else:
                    intermediate_index = f"in{j}"
                    if smem_pack > 1:
                        lines.append(f"{indent}intermediate_in{j} = in{j} + smem_offset")
                        intermediate_index = f"intermediate_in{j}"
                    lines.extend(
                        _emit_exchange_load(
                            indent, bluestein_intermediate_buffer,
                            intermediate_index, j, portable_exchange,
                        )
                    )
            elif io_mode == "bluestein_four_step_prepare_row":
                lines.append(f"{indent}src_idx{j} = in{j} * {four_step_n2} + four_step_inner")
                lines.append(f"{indent}prime_mask{j} = lane_mask & (src_idx{j} < {prime_n})")
                if bluestein_real_kind == "r2c":
                    lines.append(f"{indent}src_ptr{j} = in_ptr + four_step_batch * {prime_n} + src_idx{j}")
                    lines.append(f"{indent}r{j} = tl.load(src_ptr{j}, mask=prime_mask{j}, other={zero})")
                    lines.append(f"{indent}i{j} = {zero}")
                elif bluestein_real_kind == "c2r":
                    lines.append(f"{indent}compact_idx{j} = tl.minimum(src_idx{j}, {prime_n} - src_idx{j})")
                    lines.append(f"{indent}src_ptr{j} = in_ptr + (four_step_batch * {prime_n // 2 + 1} + compact_idx{j}) * 2")
                    lines.extend(_emit_complex_boundary_load(indent,f"src_ptr{j}",f"prime_mask{j}",
                        f"r{j}",f"loaded_i{j}",dtype,vectorized_complex_io))
                    lines.append(f"{indent}i{j} = tl.where(src_idx{j} == 0, 0.0, tl.where(src_idx{j} <= {prime_n // 2}, loaded_i{j}, -loaded_i{j}))")
                else:
                    lines.append(f"{indent}src_ptr{j} = in_ptr + (four_step_batch * {prime_n} + src_idx{j}) * 2")
                    lines.extend(_emit_complex_boundary_load(indent,f"src_ptr{j}",f"prime_mask{j}",
                        f"r{j}",f"i{j}",dtype,vectorized_complex_io))
                lines.extend(_emit_complex_boundary_load(indent,f"chirp_ptr + src_idx{j} * 2",f"prime_mask{j}",
                    f"chirp_r{j}",f"chirp_i{j}",dtype,vectorized_complex_io))
                lines.append(f"{indent}r{j}, i{j} = _cmul(r{j}, i{j}, chirp_r{j}, chirp_i{j})")
            elif io_mode == "bluestein_four_step_pointwise_row":
                lines.append(f"{indent}src_idx{j} = in{j} * {four_step_n2} + four_step_inner")
                lines.extend(_emit_complex_boundary_load(indent,f"in_ptr + (four_step_batch_base + src_idx{j}) * 2",
                    "lane_mask",f"ar{j}",f"ai{j}",dtype,vectorized_complex_io))
                lines.extend(_emit_complex_boundary_load(indent,f"b_fft_ptr + src_idx{j} * 2",
                    "lane_mask",f"br{j}",f"bi{j}",dtype,vectorized_complex_io))
                lines.append(f"{indent}r{j}, point_i{j} = _cmul(ar{j}, ai{j}, br{j}, bi{j})")
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
                if ix_column_twiddle_recurrence:
                    lines.extend(_emit_complex_boundary_load(
                        indent, f"in_ptr + (four_step_batch_base + src_idx{j}) * 2",
                        "lane_mask", f"r{j}", f"i{j}", dtype, vectorized_complex_io))
                    # Reload the exact table value every four digits to keep
                    # the complex multiplication chain short.
                    if j % 4 == 0:
                        lines.extend(_emit_complex_boundary_load(
                            indent, f"twiddle_ptr + src_idx{j} * 2",
                            "lane_mask", "col_tw_r", "col_tw_i", dtype, vectorized_complex_io))
                    else:
                        lines.append(
                            f"{indent}col_tw_r, col_tw_i = _cmul(col_tw_r, col_tw_i, col_step_r, col_step_i)"
                        )
                    lines.append(f"{indent}r{j}, i{j} = _cmul(r{j}, i{j}, col_tw_r, col_tw_i)")
                elif fuse_twiddle_into_row:
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
            load_index = f"smem_phys{j}" if read_bank_shift else f"phys{j}"
            lines.extend(
                _emit_exchange_load(
                    indent, source_buffer, load_index, j, portable_exchange,
                    mixed_direct=mixed_direct_load,
                    direct=(
                        portable_exchange
                        and _maca_knob("EXCHANGE") in {"direct", "direct_all"}
                        and _structured_exchange_supported(
                            factors, exchange_size, exchange_slot_stride, smem_pack
                        )
                    ),
                )
            )
            recurrence = (
                _ix_backend_active() and _maca_knob("RECURRENCE") == "1"
                and dtype == "complex64"
            )
            if ix_packed_twiddles:
                if j:
                    # The 8192-point shared tile reuses its digit-one root
                    # across all sixteen inputs to the radix.
                    if n == 8192 and j > 1:
                        lines.append(f"{indent}twr, twi = _cmul(twr, twi, root_step_r, root_step_i)")
                    else:
                        lines.extend(_emit_complex_boundary_load(
                            indent, f"tw{stage}_ptr + 2 * logical_phys{j}", "lane_mask",
                            "twr", "twi", dtype, _packed_maca_fp32_complex_io(dtype)))
                        if n == 8192:
                            lines.append(f"{indent}root_step_r, root_step_i = twr, twi")
                    lines.append(f"{indent}r{j}, i{j} = _cmul(r{j}, i{j}, twr, twi)")
            elif recurrence and ix_short_packed_c2r:
                if j:
                    lines.extend(_emit_ix_twiddle_tree(indent, stage, j))
                    lines.append(f"{indent}r{j}, i{j} = _cmul(r{j}, i{j}, tw_r{j}, tw_i{j})")
            elif recurrence:
                # Each butterfly consumes powers of the digit-one root. The
                # digit-zero root is exactly one, so it needs neither a load
                # nor a multiply. Keep the recurrence scoped to measured FP32.
                if j == 1:
                    lines += [
                        f"{indent}tw_step_r = tl.load(tw{stage}_r_ptr + logical_phys1, lane_mask, 0.0)",
                        f"{indent}tw_step_i = tl.load(tw{stage}_i_ptr + logical_phys1, lane_mask, 0.0)",
                        f"{indent}twr = tw_step_r", f"{indent}twi = tw_step_i",
                    ]
                elif j > 1:
                    lines.append(f"{indent}twr, twi = _cmul(twr, twi, tw_step_r, tw_step_i)")
                if j:
                    lines.append(f"{indent}r{j}, i{j} = _cmul(r{j}, i{j}, twr, twi)")
            else:
                tw_index = f"logical_phys{j}"
                if _npu_backend_active() and os.environ.get("FLAGFFT_NPU_PORTABLE_VECTOR_IO") == "1":
                    tw_index = f"(lane + {current_lanes} * (group_{stage} * {radix} + {j}))"
                if npu_trig_twiddles:
                    prefix = math.prod(factors[:stage])
                    scale = _direction_sign(direction)*2*math.pi*j/(prefix*radix)
                    lines += [
                        f"{indent}tw_frequency = lane-(lane//{prefix})*{prefix}",
                        f"{indent}tw_angle = tw_frequency.to(tl.float32)*{scale:.17g}",
                        f"{indent}twr = tl.cos(tw_angle)",
                        f"{indent}twi = tl.sin(tw_angle)",
                    ]
                elif npu_affine_tables:
                    lines += [
                        f"{indent}tw_range = tl.arange(0,{lane_block})",
                        f"{indent}tw_plane_r = tl.load(tw{stage}_r_ptr + {j*current_lanes} + tw_range, tw_range < {current_lanes}, 0.0)",
                        f"{indent}tw_plane_i = tl.load(tw{stage}_i_ptr + {j*current_lanes} + tw_range, tw_range < {current_lanes}, 0.0)",
                        f"{indent}tw_lookup = tl.where(lane_mask, lane, 0)",
                        f"{indent}twr = tl.gather(tw_plane_r, tw_lookup, 0)",
                        f"{indent}twi = tl.gather(tw_plane_i, tw_lookup, 0)",
                    ]
                else:
                    lines.append(
                        f"{indent}twr = tl.load(tw{stage}_r_ptr + {tw_index}, mask=lane_mask, other={zero})"
                    )
                    lines.append(
                        f"{indent}twi = tl.load(tw{stage}_i_ptr + {tw_index}, mask=lane_mask, other={zero})"
                    )
                lines.append(f"{indent}r{j}, i{j} = _cmul(r{j}, i{j}, twr, twi)")

    smem_tail = permuted_smem_store and is_last
    shared_read = stage > 0 or (io_mode in {"bluestein_full_leaf", "rader_full_leaf"} and bluestein_pass == 1)
    shared_write = not is_last or (io_mode in {"bluestein_full_leaf", "rader_full_leaf"} and bluestein_pass == 0)
    if ((single_smem_buffer and shared_read and shared_write) or smem_tail
            or (packed_smem_alias and is_last)):
        # Every old shared-memory input must be read before the final stage
        # reuses that buffer for a different, batch-contiguous layout.
        lines.append(f"{indent}tl.debug_barrier()")

    ix_register_prime = False
    if _ix_backend_active() and dtype=='complex64' and radix in {13,17}:
        setting=_maca_knob('RADER_CODELET','0')
        if setting not in {'0','1'}:
            raise ValueError('FLAGFFT_IX_RADER_CODELET must be 0 or 1')
        ix_register_prime=setting=='1'
    if ix_compact_tail or ix_real_ct_codelet or ix_dif_packed_real:
        lines.extend(_emit_ix_register_codelet(
            indent, radix, direction, split_radix=ix_compact_tail, decimation_in_frequency=ix_dif_packed_real
        ))
    elif ix_register_prime:
        from .ix_prime_codelets import emit_prime_codelet
        lines.extend(emit_prime_codelet(indent,radix,direction))
    elif radix == 16:
        lines.extend(_emit_radix16_codelet_call(indent, direction))
    elif radix == 32:
        lines.extend(_emit_natural_order_radix32_codelet_call(indent, direction))
    elif radix in _THREAD_LOCAL_MIXED_RADICES:
        lines.extend(_emit_local_mixed_codelet_call(indent, radix, direction))
    elif radix in _NATURAL_ORDER_CODELET_RADICES:
        lines.extend(_emit_natural_order_codelet_call(indent, radix, direction))
    else:
        lines.extend(_emit_table_codelet(indent, radix, lane_block, dtype))

    if smem_tail:
        assert not portable_exchange and stage > 0 and groups == 1
        for j in range(radix):
            offset = j * math.prod(factors[:-1])
            lines.append(f"{indent}tail_dst{j} = (output_base + {offset}) * {smem_pack} + batch_slot")
            lines.extend(_emit_exchange_store(indent, source_buffer, f"tail_dst{j}", j,
                                              f"r{j}", f"i{j}", False))
        out_length = n // 2 + 1 if io_mode in {"permuted_r2c", "permuted_store_r2c"} else n
        lines.extend([
            f"{indent}tl.debug_barrier()",
            f"{indent}tail_flat = tl.arange(0, {n * smem_pack})",
            f"{indent}tail_slot = tail_flat % {smem_pack}",
            f"{indent}tail_freq = tail_flat // {smem_pack}",
            f"{indent}tail_batch = " + ("perm_base + tail_slot * perm_span" if perm_inner else "batch_id + tail_slot"),
            f"{indent}tail_i0 = tail_batch // perm_span",
            f"{indent}tail_i1 = tail_batch - tail_i0 * perm_span",
            f"{indent}tail_gbase = " + ("tail_i1 * (nbatch // perm_span) + tail_i0" if perm_inner
                                       else f"tail_i0 * ({out_length} * perm_span) + tail_i1"),
            f"{indent}tail_mask = (tail_batch < nbatch) & (tail_freq < {out_length})",
            f"{indent}tail_r = tl.load(tle.gpu.local_ptr({source_buffer}_r, (tail_flat,)), mask=tail_mask, other=0.0)",
            f"{indent}tail_i = tl.load(tle.gpu.local_ptr({source_buffer}_i, (tail_flat,)), mask=tail_mask, other=0.0)",
            f"{indent}tail_addr = tail_freq * perm_k_stride + tail_gbase",
            f"{indent}tl.store(out_ptr + tail_addr * 2, tail_r, mask=tail_mask)",
            f"{indent}tl.store(out_ptr + tail_addr * 2 + 1, tail_i, mask=tail_mask)",
        ])
        return lines

    grouped_maca_cube_store = (
        is_last
        and io_mode == "permuted_store"
        and _packed_maca_fp32_complex_io(dtype)
        and math.prod(factors) == 256
        and radix in {2, 4, 8, 16, 32}
        and smem_pack > 1
    )
    grouped_hcu_store = (
        _hcu_backend_active()
        and os.getenv("FLAGFFT_HCU_3D_STORE_MODE", "auto") == "grouped"
        and is_last
        and io_mode in {"permuted_store", "strided_permuted_store"}
        and dtype == "complex64"
        and radix in {2, 4, 8, 16, 32}
        and smem_pack > 1
        and n == current_lanes * radix
    )
    grouped_store = grouped_maca_cube_store or grouped_hcu_store
    if grouped_store:
        lines.extend(
            _emit_grouped_fp32_permuted_store(
                indent, factors, radix, smem_pack, lane_block
            )
        )

    for j in range(radix):
        if is_last:
            if (io_mode == "contiguous_r2c" and _npu_backend_active()
                    and j * math.prod(factors[:-1]) >= n//2+1):
                # These frequencies are outside the compact spectrum. Avoid
                # constructing an empty masked vector in the CANN lowering.
                continue
            if io_mode in {"permuted_store", "strided_permuted_store", "permuted_r2c", "permuted_store_r2c"}:
                if grouped_store:
                    continue
                compact_length = (
                    n // 2 + 1
                    if io_mode in {"permuted_r2c", "permuted_store_r2c"}
                    else None
                )
                lines.extend(
                    _emit_permuted_store(
                        indent,
                        j,
                        factors,
                        smem_pack,
                        lane_block,
                        dtype,
                        compact_length=compact_length,
                    )
                )
                continue
            lines.extend(_emit_output_index(indent, f"out_idx{j}", factors, j))
            if npu_batch_io:
                offset = j * math.prod(factors[:-1])
                real_output = io_mode == "contiguous_c2r"
                components = 1 if real_output else 2
                extent = n//2+1 if io_mode == "contiguous_r2c" else n
                distance = "output_distance" if io_mode != "contiguous" else str(n)
                value = f"r{j}" if real_output else f"tl.join(r{j},i{j})"
                lines += [
                    f"{indent}batch_output{j} = tl.reshape({value},({smem_pack},{lane_block*components}))",
                    f"{indent}batch_rows = batch_id + tl.arange(0,{smem_pack})[:,None]",
                    f"{indent}batch_cols = tl.arange(0,{lane_block*components})[None,:]",
                    f"{indent}tl.store(out_ptr + batch_rows*({distance})*{components} + batch_cols+{offset*components}, "
                    f"batch_output{j}, (batch_rows < nbatch) & (batch_cols < {current_lanes*components}) "
                    f"& (batch_cols+{offset*components} < {extent*components}))",
                ]
                continue
            if npu_matrix_io:
                lines.extend(_emit_npu_matrix_store(indent, f"matrix_output{j}",
                    f"r{j}", f"i{j}", lane_block, inner_pack, current_lanes,
                    j * math.prod(factors[:-1]), four_step_n1, io_mode == "four_step_row", four_step_n2))
                continue
            if ix_split_c2r:
                lines.append(f"{indent}tl.store(out_ptr + 2 * out_idx{j} + split_band, r{j}, lane_mask)")
            elif ix_split_frequency:
                lines.extend([
                    f"{indent}split_lo = tl.cast(tl.cast(r{j}, tl.uint32, bitcast=True), tl.uint64)",
                    f"{indent}split_hi = tl.cast(tl.cast(i{j}, tl.uint32, bitcast=True), tl.uint64)",
                    f"{indent}split_dst = tl.cast(out_ptr, tl.pointer_type(tl.uint64)) + 2 * out_idx{j} + split_band",
                    f"{indent}tl.store(split_dst, split_lo | (split_hi << 32), lane_mask)",
                ])
            elif io_mode in {"contiguous", "strided", "packed_c2r", "bluestein_prepare_leaf",
                           "rader_prepare_leaf"}:
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
            elif io_mode == "permuted_r2c":
                half_n = n // 2 + 1
                offset = j * math.prod(factors[:len(factors) - 1])
                base = "output_base_lane" if offset == 0 else f"(output_base_lane + {offset})"
                if smem_pack == 1:
                    lines.append(f"{indent}compact_mask{j} = lane_mask & (out_idx{j} < {half_n})")
                    lines.append(
                        f"{indent}dst_idx{j} = {base} * output_distance + perm_gbase_scalar"
                    )
                    lines.append(f"{indent}tl.store(out_ptr + dst_idx{j} * 2, r{j}, mask=compact_mask{j})")
                    lines.append(f"{indent}tl.store(out_ptr + dst_idx{j} * 2 + 1, i{j}, mask=compact_mask{j})")
                else:
                    lines.append(f"{indent}compact_mask{j} = perm_store_mask & ({base} < {half_n})[:, None]")
                    lines.append(f"{indent}zr{j} = tl.trans(tl.reshape(r{j}, ({smem_pack}, {lane_block})))")
                    lines.append(f"{indent}zi{j} = tl.trans(tl.reshape(i{j}, ({smem_pack}, {lane_block})))")
                    lines.append(
                        f"{indent}dst_idx{j} = {base}[:, None] * output_distance + perm_gbase[None, :]"
                    )
                    lines.append(f"{indent}tl.store(out_ptr + dst_idx{j} * 2, zr{j}, mask=compact_mask{j})")
                    lines.append(f"{indent}tl.store(out_ptr + dst_idx{j} * 2 + 1, zi{j}, mask=compact_mask{j})")
            elif io_mode == "contiguous_r2c":
                lines.append(
                    f"{indent}compact_mask{j} = lane_mask & (out_idx{j} < {n // 2 + 1})"
                )
                lines.append(
                    f"{indent}dst_ptr{j} = out_ptr + (output_batch_base + out_idx{j}) * 2"
                )
                if vectorized_complex_io:
                    lines.extend(_emit_vectorized_complex_store(indent,f"dst_ptr{j}",
                        f"r{j}",f"i{j}",f"compact_mask{j}",dtype))
                else:
                    lines.append(
                        f"{indent}tl.store(dst_ptr{j}, r{j}, mask=compact_mask{j})"
                    )
                    lines.append(
                        f"{indent}tl.store(dst_ptr{j} + 1, i{j}, mask=compact_mask{j})"
                    )
            elif io_mode == "packed_r2c":
                packed_index = f"out_idx{j}"
                if not portable_exchange and smem_pack > 1:
                    lines.append(f"{indent}packed_index{j} = out_idx{j} + smem_offset")
                    packed_index = f"packed_index{j}"
                lines.extend(_emit_exchange_store(
                    indent, "packed", packed_index, j, f"r{j}", f"i{j}", portable_exchange
                ))
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
            elif io_mode == "rader_finish_leaf":
                lines.append(
                    f"{indent}dst_idx{j} = tl.load(idx_ptr + out_idx{j}, mask=lane_mask, other=0)"
                )
                lines.append(
                    f"{indent}dst_ptr{j} = out_ptr + (current_batch * {prime_n} + dst_idx{j}) * 2"
                )
                lines.append(
                    f"{indent}x0_ptr{j} = original_ptr + current_batch * {prime_n} * 2"
                )
                lines.append(
                    f"{indent}x0r{j} = tl.load(x0_ptr{j}, mask=current_batch < nbatch, other={zero})"
                )
                lines.append(
                    f"{indent}x0i{j} = tl.load(x0_ptr{j} + 1, mask=current_batch < nbatch, other={zero})"
                )
                lines.append(
                    f"{indent}tl.store(dst_ptr{j}, x0r{j} + r{j} / {n}, mask=lane_mask)"
                )
                lines.append(
                    f"{indent}tl.store(dst_ptr{j} + 1, x0i{j} - i{j} / {n}, mask=lane_mask)"
                )
                lines.append(f"{indent}dc_mask{j} = lane_mask & (out_idx{j} == 0)")
                lines.append(
                    f"{indent}a0_ptr{j} = in_ptr + batch_base * 2"
                )
                lines.append(
                    f"{indent}dc_ptr{j} = out_ptr + current_batch * {prime_n} * 2"
                )
                lines.append(
                    f"{indent}a0r{j} = tl.load(a0_ptr{j}, mask=current_batch < nbatch, other={zero})"
                )
                lines.append(
                    f"{indent}a0i{j} = tl.load(a0_ptr{j} + 1, mask=current_batch < nbatch, other={zero})"
                )
                lines.append(
                    f"{indent}tl.store(dc_ptr{j} + out_idx{j}, x0r{j} + a0r{j}, mask=dc_mask{j})"
                )
                lines.append(
                    f"{indent}tl.store(dc_ptr{j} + out_idx{j} + 1, x0i{j} + a0i{j}, mask=dc_mask{j})"
                )
            elif io_mode == "bluestein_full_leaf":
                if bluestein_pass == 0:
                    lines.extend(_emit_complex_boundary_load(
                        indent, f"b_fft_ptr + out_idx{j} * 2", "lane_mask",
                        f"br{j}", f"bi{j}", dtype, ix_prime_vector_io,
                    ))
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
                    output_length = prime_n // 2 + 1 if bluestein_real_kind == "r2c" else prime_n
                    lines.append(f"{indent}prime_mask{j} = lane_mask & (out_idx{j} < {output_length})")
                    lines.append(f"{indent}scaled_r{j} = r{j} / {n}")
                    lines.append(f"{indent}scaled_i{j} = -i{j} / {n}")
                    lines.extend(_emit_complex_boundary_load(
                        indent, f"chirp_ptr + out_idx{j} * 2", f"prime_mask{j}",
                        f"chirp_r{j}", f"chirp_i{j}", dtype, ix_prime_vector_io,
                    ))
                    lines.append(
                        f"{indent}final_r{j}, final_i{j} = _cmul(scaled_r{j}, scaled_i{j}, chirp_r{j}, chirp_i{j})"
                    )
                    if bluestein_real_kind == "c2r":
                        lines.append(f"{indent}dst_ptr{j} = out_ptr + current_batch * {prime_n} + out_idx{j}")
                        lines.append(f"{indent}tl.store(dst_ptr{j}, final_r{j}, mask=prime_mask{j})")
                    else:
                        output_distance = prime_n // 2 + 1 if bluestein_real_kind == "r2c" else prime_n
                        lines.append(f"{indent}dst_ptr{j} = out_ptr + (current_batch * {output_distance} + out_idx{j}) * 2")
                        if ix_prime_vector_io:
                            lines.extend(_emit_vectorized_complex_store(
                                indent, f"dst_ptr{j}", f"final_r{j}", f"final_i{j}",
                                f"prime_mask{j}", dtype,
                            ))
                        else:
                            lines += [
                                f"{indent}tl.store(dst_ptr{j}, final_r{j}, mask=prime_mask{j})",
                                f"{indent}tl.store(dst_ptr{j} + 1, final_i{j}, mask=prime_mask{j})",
                            ]
            elif io_mode == "rader_full_leaf":
                compact_inverse = bluestein_real_kind == "c2r"
                input_distance = prime_n // 2 + 1 if compact_inverse else prime_n
                if bluestein_pass == 0:
                    lines.extend(_emit_complex_boundary_load(
                        indent, f"b_fft_ptr + out_idx{j} * 2", "lane_mask",
                        f"br{j}", f"bi{j}", dtype, ix_rader_vector_io,
                    ))
                    lines.append(
                        f"{indent}point_r{j}, point_i{j} = _cmul(r{j}, i{j}, br{j}, bi{j})"
                    )
                    intermediate_index = f"out_idx{j}"
                    if smem_pack > 1:
                        lines.append(f"{indent}intermediate_out{j} = out_idx{j} + smem_offset")
                        intermediate_index = f"intermediate_out{j}"
                    lines.extend(
                        _emit_exchange_store(
                            indent, bluestein_intermediate_buffer,
                            intermediate_index, j, f"point_r{j}", f"-point_i{j}",
                            portable_exchange,
                        )
                    )
                    lines.append(f"{indent}dc_mask{j} = lane_mask & (out_idx{j} == 0)")
                    lines.append(f"{indent}x0_ptr{j} = in_ptr + current_batch * {input_distance} * 2")
                    output_scalars = prime_n if compact_inverse else 2 * prime_n
                    lines.append(f"{indent}dc_ptr{j} = out_ptr + current_batch * {output_scalars}")
                    lines.append(
                        f"{indent}tl.store(dc_ptr{j} + out_idx{j}, r{j} + tl.load(x0_ptr{j}, mask=current_batch < nbatch, other={zero}), mask=dc_mask{j})"
                    )
                    if not compact_inverse:
                        lines.append(
                            f"{indent}tl.store(dc_ptr{j} + out_idx{j} + 1, i{j} + tl.load(x0_ptr{j} + 1, mask=current_batch < nbatch, other={zero}), mask=dc_mask{j})"
                        )
                else:
                    lines.append(
                        f"{indent}dst_idx{j} = tl.load(idx_ptr + out_idx{j}, mask=lane_mask, other=0)"
                    )
                    output_stride = 1 if compact_inverse else 2
                    lines.append(
                        f"{indent}dst_ptr{j} = out_ptr + (current_batch * {prime_n} + dst_idx{j}) * {output_stride}"
                    )
                    lines.append(f"{indent}x0_ptr{j} = in_ptr + current_batch * {input_distance} * 2")
                    lines.append(
                        f"{indent}tl.store(dst_ptr{j}, tl.load(x0_ptr{j}, mask=current_batch < nbatch, other={zero}) + r{j} / {n}, mask=lane_mask)"
                    )
                    if not compact_inverse:
                        lines.append(
                            f"{indent}tl.store(dst_ptr{j} + 1, tl.load(x0_ptr{j} + 1, mask=current_batch < nbatch, other={zero}) - i{j} / {n}, mask=lane_mask)"
                        )
            elif io_mode == "bluestein_four_step_finish_col":
                lines.append(
                    f"{indent}dst_idx{j} = out_idx{j} * {four_step_n1} + four_step_inner"
                )
                output_length = prime_n // 2 + 1 if bluestein_real_kind == "r2c" else prime_n
                lines.append(f"{indent}prime_mask{j} = lane_mask & (dst_idx{j} < {output_length})")
                lines.append(
                    f"{indent}scaled_r{j} = r{j} / {four_step_n1 * four_step_n2}"
                )
                lines.append(
                    f"{indent}scaled_i{j} = -i{j} / {four_step_n1 * four_step_n2}"
                )
                lines.extend(_emit_complex_boundary_load(indent,f"chirp_ptr + dst_idx{j} * 2",f"prime_mask{j}",
                    f"chirp_r{j}",f"chirp_i{j}",dtype,vectorized_complex_io))
                lines.append(
                    f"{indent}final_r{j}, final_i{j} = _cmul(scaled_r{j}, scaled_i{j}, chirp_r{j}, chirp_i{j})"
                )
                if bluestein_real_kind == "c2r":
                    lines.append(f"{indent}dst_ptr{j} = out_ptr + four_step_batch * {prime_n} + dst_idx{j}")
                    lines.append(f"{indent}tl.store(dst_ptr{j}, final_r{j}, mask=prime_mask{j})")
                else:
                    output_distance = prime_n // 2 + 1 if bluestein_real_kind == "r2c" else prime_n
                    lines.append(f"{indent}dst_ptr{j} = out_ptr + (four_step_batch * {output_distance} + dst_idx{j}) * 2")
                    if vectorized_complex_io:
                        lines.extend(_emit_vectorized_complex_store(indent,f"dst_ptr{j}",
                            f"final_r{j}",f"final_i{j}",f"prime_mask{j}",dtype))
                    else:
                        lines.append(f"{indent}tl.store(dst_ptr{j}, final_r{j}, mask=prime_mask{j})")
                        lines.append(f"{indent}tl.store(dst_ptr{j} + 1, final_i{j}, mask=prime_mask{j})")
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
                    if _is_double_dtype(dtype) or _hcu_backend_active():
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
            if write_bank_shift:
                lines.append(
                    f"{indent}smem_dst{j} = dst{j} ^ "
                    f"(dst{j} >> {write_bank_shift})"
                )
                if smem_interleave:
                    lines.append(f"{indent}smem_dst{j} = smem_dst{j} * {smem_pack} + {interleave_slot}")
                elif hcu_cyclic_smem:
                    lines.append(f"{indent}smem_dst{j} = ((smem_dst{j} + {hcu_cyclic_slot}) & {n - 1}) + smem_offset")
                elif smem_pack > 1:
                    lines.append(f"{indent}smem_dst{j} += smem_offset")
                store_index = f"smem_dst{j}"
            elif smem_interleave:
                lines.append(f"{indent}smem_dst{j} = dst{j} * {smem_pack} + {interleave_slot}")
                store_index = f"smem_dst{j}"
            elif hcu_cyclic_smem:
                lines.append(f"{indent}smem_dst{j} = ((dst{j} + {hcu_cyclic_slot}) & {n - 1}) + smem_offset")
                store_index = f"smem_dst{j}"
            elif smem_pack > 1:
                lines.append(f"{indent}smem_dst{j} = dst{j} + smem_offset")
                store_index = f"smem_dst{j}"
            if io_layout:
                # Both XOR maps are bijections within the allocated FFT tile.
                lines.append(
                    f"{indent}io_dst{j} = (dst{j} ^ (dst{j} >> {io_logical_shift})) * {smem_pack} + "
                    f"(inner_slot ^ (((dst{j} >> {io_slot_shift}) * {io_slot_chunk}) & {smem_pack-1}))"
                )
                store_index=f"io_dst{j}"
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
            "packed" if is_last and io_mode == "packed_r2c" else
            bluestein_intermediate_buffer
            if is_last and io_mode in {"bluestein_full_leaf", "rader_full_leaf"} and bluestein_pass == 0
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
                    allow_mixed_direct=mixed_direct_layout,
                    next_lane_block=next_lane_block,
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
    batched_permutation: bool = False,
    ix_packed_twiddles: bool = False,
) -> list[str]:
    params = _leaf_kernel_params(
        plan, include_four_step_twiddle=include_four_step_twiddle
    )
    if io_mode == "packed_r2c":
        params.insert(2, "packed_twiddle_ptr")
        if ix_packed_twiddles:
            params[3:] = [f"tw{stage}_ptr" for stage in range(1, len(plan.factors) - 1)]
    elif io_mode == "packed_c2r":
        params.insert(2, "packed_twiddle_ptr")
    if io_mode == "strided":
        params.append("outer_stride")
    if io_mode in {"permuted_store", "strided_permuted_store"}:
        params.append("perm_span")
        if batched_permutation:
            params.append("rows_per_transform")
    elif io_mode == "permuted_store_r2c":
        params.extend(("input_distance", "perm_span"))
    if io_mode == "bluestein_prepare_leaf":
        params.insert(1, "chirp_ptr")
    elif io_mode == "bluestein_finish_leaf":
        params[1:1] = ["b_fft_ptr", "chirp_ptr"]
    elif io_mode == "bluestein_full_leaf":
        params[1:1] = ["b_fft_ptr", "chirp_ptr"]
    elif io_mode == "rader_full_leaf":
        params[1:1] = ["b_fft_ptr", "idx_ptr"]
    elif io_mode == "rader_prepare_leaf":
        params.insert(1, "idx_ptr")
    elif io_mode == "rader_finish_leaf":
        params[1:1] = ["b_fft_ptr", "idx_ptr", "original_ptr"]
    elif io_mode in {
        "bluestein_four_step_prepare_row",
        "bluestein_four_step_finish_col",
    }:
        params.insert(1, "chirp_ptr")
    elif io_mode == "bluestein_four_step_pointwise_row":
        params.insert(1, "b_fft_ptr")
    if io_mode in {
        "contiguous_r2c",
        "permuted_r2c",
        "packed_r2c",
        "packed_c2r",
        "contiguous_c2r",
        "four_step_real_row",
        "four_step_hermitian_row",
    }:
        params.append("input_distance")
    if io_mode in {
        "contiguous_r2c",
        "permuted_r2c",
        "packed_r2c",
        "contiguous_c2r",
        "four_step_r2c_col",
        "four_step_c2r_col",
    }:
        params.append("output_distance")
    if io_mode in {"four_step_row_strided", "four_step_col_strided"}:
        params.append("outer_stride")
    if io_mode == "permuted_r2c":
        params.append("perm_span")
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
    if io_mode.startswith("bluestein_four_step_") and not (
        os.getenv("FLAGFFT_CUDA_BLUESTEIN_REGISTER") == "1"
        and current_profile().backend == "cuda"
        and four_step_n1 == four_step_n2 == 1024
        and tuple(plan.factors) == (32, 32)
    ):
        return False
    if len(plan.factors) != 2:
        return False
    register_radix, cross_radix = plan.factors
    power2_register = (current_profile().backend == "cuda"
        and os.getenv("FLAGFFT_CUDA_POWER2_REGISTER") == "1"
        and register_radix in {16,32} and cross_radix in {16,32})
    row_modes = {
        "four_step_row",
        "bluestein_four_step_prepare_row",
        "bluestein_four_step_pointwise_row",
        "four_step_real_row",
        "four_step_hermitian_row",
    }
    col_modes = {
        "four_step_col",
        "four_step_r2c_col",
        "four_step_c2r_col",
        "bluestein_four_step_finish_col",
    }
    expected_length = four_step_n1 if io_mode in row_modes else four_step_n2
    return (
        io_mode in row_modes | col_modes
        and plan.dtype in {"complex64", "complex128"}
        and (register_radix in _THREAD_LOCAL_MIXED_RADICES or power2_register)
        and (cross_radix == 32 or power2_register)
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

    even_names = names[: len(names) // 2] if highest_first else names[0::2]
    odd_names = names[len(names) // 2 :] if highest_first else names[1::2]
    even_source = (
        even_names[0] if len(even_names) == 1 else f"{prefix}_even{len(names)}"
    )
    odd_source = odd_names[0] if len(odd_names) == 1 else f"{prefix}_odd{len(names)}"
    lines = [f"{indent}{even_source}, {odd_source} = tl.split({source})"]
    if len(even_names) > 1:
        lines.extend(
            _emit_distributed_split_tree(
                indent,
                even_source,
                even_names,
                f"{prefix}_e",
                highest_first=highest_first,
            )
        )
    if len(odd_names) > 1:
        lines.extend(
            _emit_distributed_split_tree(
                indent,
                odd_source,
                odd_names,
                f"{prefix}_o",
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
    prime_n: int = 0,
    bluestein_real_kind: str = "",
) -> tuple[str, str]:
    # Each thread owns the first composite register FFT, followed by one
    # shared exchange and a full register-only radix-32 FFT.
    register_radix, cross_radix = plan.factors
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
        "bluestein_four_step_prepare_row",
        "bluestein_four_step_pointwise_row",
        "four_step_real_row",
        "four_step_hermitian_row",
    }
    col_modes = {
        "four_step_col",
        "four_step_r2c_col",
        "four_step_c2r_col",
        "bluestein_four_step_finish_col",
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
        f"{io_mode}_{kernel_prefix}_kernel_{register_radix}_{cross_radix}_thread_local"
        f"_n{four_step_n1}_{four_step_n2}_l{plan.lanes}_b32_t{smem_chunk}"
        "_v5g_itwsincos_otwrec_nw4"
    )

    if io_mode.startswith("bluestein_four_step_"):
        kernel_name += f"_p{prime_n}_{bluestein_real_kind or 'complex'}"

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
            f"    input_lane_mask = lane_mask & (fft_thread < {cross_radix})",
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
        body.append(f"    input_idx{idx} = fft_thread + {cross_radix * input_digit}")
        body.append(
            f"    src_idx{idx} = input_idx{idx} * {source_stride} + " "four_step_inner"
        )
        if io_mode == "bluestein_four_step_prepare_row":
            body.append(f"    prime_mask{idx} = lane_mask & (src_idx{idx} < {prime_n})")
            if bluestein_real_kind == "r2c":
                offset = f"four_step_batch * {prime_n} + src_idx{idx}"
                body.append(f"    r{idx} = tl.load(in_ptr + {offset}, mask=prime_mask{idx}, other=0.0)")
                body.append(f"    i{idx} = r{idx} * 0.0")
            else:
                if bluestein_real_kind == "c2r":
                    body.append(f"    compact_idx{idx} = tl.minimum(src_idx{idx}, {prime_n} - src_idx{idx})")
                    offset = f"(four_step_batch * {prime_n // 2 + 1} + compact_idx{idx}) * 2"
                else:
                    offset = f"(four_step_batch * {prime_n} + src_idx{idx}) * 2"
                body.append(f"    input_offset{idx} = {offset}")
                body.append(f"    r{idx} = tl.load(in_ptr + input_offset{idx}, mask=prime_mask{idx}, other=0.0)")
                body.append(f"    i{idx} = tl.load(in_ptr + input_offset{idx} + 1, mask=prime_mask{idx}, other=0.0)")
                if bluestein_real_kind == "c2r":
                    body.append(f"    i{idx} = tl.where(src_idx{idx} == 0, 0.0, tl.where(src_idx{idx} <= {prime_n // 2}, i{idx}, -i{idx}))")
            body.append(f"    chirp_r{idx} = tl.load(chirp_ptr + src_idx{idx} * 2, mask=prime_mask{idx}, other=0.0)")
            body.append(f"    chirp_i{idx} = tl.load(chirp_ptr + src_idx{idx} * 2 + 1, mask=prime_mask{idx}, other=0.0)")
            body.append(f"    r{idx}, i{idx} = _cmul(r{idx}, i{idx}, chirp_r{idx}, chirp_i{idx})")
        elif io_mode == "bluestein_four_step_pointwise_row":
            body.append(f"    input_offset{idx} = (four_step_batch_base + src_idx{idx}) * 2")
            body.append(f"    ar{idx} = tl.load(in_ptr + input_offset{idx})")
            body.append(f"    ai{idx} = tl.load(in_ptr + input_offset{idx} + 1)")
            body.append(f"    br{idx} = tl.load(b_fft_ptr + src_idx{idx} * 2)")
            body.append(f"    bi{idx} = tl.load(b_fft_ptr + src_idx{idx} * 2 + 1)")
            body.append(f"    r{idx}, point_i{idx} = _cmul(ar{idx}, ai{idx}, br{idx}, bi{idx})")
            body.append(f"    i{idx} = -point_i{idx}")
        elif io_mode == "four_step_real_row":
            body.append(
                f"    input_offset{idx} = "
                f"four_step_batch * input_distance + src_idx{idx}"
            )
            body.append(f"    r{idx} = tl.load(in_ptr + input_offset{idx}, mask=input_lane_mask, other=0.0)")
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
            if cross_radix < 32:
                body.extend(_emit_vectorized_complex_load("    ", f"in_ptr + input_offset{idx}",
                    "input_lane_mask", f"r{idx}, i{idx}", plan.dtype))
            else:
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
            if cross_radix < 32:
                body.extend(_emit_vectorized_complex_load("    ", f"in_ptr + input_offset{idx}",
                    "input_lane_mask", f"r{idx}, i{idx}", plan.dtype))
            else:
                body.append(
                    # The selected leaves and pack=4 cover every input lane exactly.
                    f"    r{idx}, i{idx} = tl.inline_asm_elementwise("
                    f'"ld.global.v2.{vector_suffix} {{$0, $1}}, [$2];", '
                    f"{asm_load_constraints}, "
                    f"[tl.cast(in_ptr + input_offset{idx}, tl.uint64)], "
                    f"dtype=({vector_dtype}, {vector_dtype}), is_pure=False, pack=1)"
                )

    body.extend(_emit_radix16_codelet_call("    ",plan.direction) if register_radix == 16
                else _emit_local_mixed_codelet_call("    ", register_radix, plan.direction))

    body.append(
        f"    smem_store_mask = tl.broadcast_to("
        f"tl.reshape(input_lane_mask, {vector_block}, {smem_reshape_dims}), "
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
                            f"tl.load(tw1_r_ptr + inner_tw_idx{idx}, mask=input_lane_mask, other=0.0)"
                        ),
                        (
                            f"    inner_tw_i{idx} = "
                            f"tl.load(tw1_i_ptr + inner_tw_idx{idx}, mask=input_lane_mask, other=0.0)"
                        ),
                        (
                            f"    r{idx}, i{idx} = "
                            f"_cmul(r{idx}, i{idx}, "
                            f"inner_tw_r{idx}, inner_tw_i{idx})"
                        ),
                    ]
                )
            body.append(f"    smem_logical{idx} = {idx * cross_radix} + fft_thread")
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

    for chunk_base in range(0, cross_radix, smem_chunk):
        chunk_indices = range(chunk_base, chunk_base + smem_chunk)
        for idx in chunk_indices:
            second_input = 2 * (idx % 16) + idx // 16 if cross_radix == 32 else idx
            body.append(f"    smem_logical{idx} = fft_thread * {cross_radix} + " f"{second_input}")
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

    body.extend(_emit_radix16_codelet_call("    ",plan.direction) if cross_radix == 16
                else _emit_local_radix32_codelet_call("    ", plan.direction))

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

    for idx in range(cross_radix):
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
            if idx < cross_radix-1:
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
        if io_mode == "bluestein_four_step_finish_col":
            output_length = prime_n // 2 + 1 if bluestein_real_kind == "r2c" else prime_n
            body.append(f"    prime_mask{idx} = output_lane_mask & (dst_idx{idx} < {output_length})")
            body.append(f"    scaled_r{idx} = r{idx} / {four_step_n1 * four_step_n2}")
            body.append(f"    scaled_i{idx} = -i{idx} / {four_step_n1 * four_step_n2}")
            body.append(f"    chirp_r{idx} = tl.load(chirp_ptr + dst_idx{idx} * 2, mask=prime_mask{idx}, other=0.0)")
            body.append(f"    chirp_i{idx} = tl.load(chirp_ptr + dst_idx{idx} * 2 + 1, mask=prime_mask{idx}, other=0.0)")
            body.append(f"    final_r{idx}, final_i{idx} = _cmul(scaled_r{idx}, scaled_i{idx}, chirp_r{idx}, chirp_i{idx})")
            if bluestein_real_kind == "c2r":
                body.append(f"    output_offset{idx} = four_step_batch * {prime_n} + dst_idx{idx}")
                body.append(f"    tl.store(out_ptr + output_offset{idx}, final_r{idx}, mask=prime_mask{idx})")
            else:
                body.append(f"    output_offset{idx} = (four_step_batch * {output_length} + dst_idx{idx}) * 2")
                body.append(f"    tl.store(out_ptr + output_offset{idx}, final_r{idx}, mask=prime_mask{idx})")
                body.append(f"    tl.store(out_ptr + output_offset{idx} + 1, final_i{idx}, mask=prime_mask{idx})")
        elif io_mode == "four_step_r2c_col":
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
    permutation_span: int = 0,
    permutation_batch: int = 0,
    hcu_full_smem: bool = False,
) -> tuple[str, str]:
    ix_band_modes = {"ix_band_c2c": "contiguous",
                     "ix_band_r2c": "contiguous_r2c",
                     "ix_band_r2c_folded": "contiguous_r2c",
                     "ix_band_c2r": "c2r"}
    if _ix_backend_active() and perm_form in ix_band_modes and io_mode == "contiguous":
        from .kernels_ix_warp import build_ix_warp_bands
        return build_ix_warp_bands(plan, permutation_span, ix_band_modes[perm_form], permutation_batch,
                                  fold=perm_form == "ix_band_r2c_folded")
    if (_ix_backend_active() and perm_form == "ix_prime_r2c_last" and
            io_mode == "strided_permuted_store" and plan.length == 16 and
            plan.direction == "forward" and plan.dtype == "complex64"):
        from .kernels_ix_warp import build_ix_prime_r2c_last_source
        return build_ix_prime_r2c_last_source(_leaf_kernel_params_for_io(plan, io_mode=io_mode))
    kernel_perm_form = perm_form
    batched_permutation = perm_form.endswith("_batched")
    if batched_permutation:
        perm_form = perm_form.removesuffix("_batched")
        if io_mode not in {"permuted_store", "strided_permuted_store"} or not perm_form.startswith("inner"):
            raise ValueError("batched permutation requires an inner permuted store")
    kernel_io_mode = io_mode
    ix_prime_packed_rows = (
        _ix_backend_active() and io_mode == "packed_r2c" and perm_form == "outer_packed"
        and plan.length == 32 and plan.dtype == "complex64" and permutation_span == 997
    )
    if permutation_span < 0 or (permutation_span and io_mode not in {
        "permuted_store", "strided_permuted_store", "permuted_r2c", "permuted_store_r2c"
    } and not ix_prime_packed_rows):
        raise ValueError("a static permutation span requires a permuted-store leaf")
    if permutation_batch < 0 or (permutation_batch and not permutation_span):
        raise ValueError("a static permutation batch requires a positive span")
    if ix_prime_packed_rows:
        from .target import ix_warp_fft_pack
        from .kernels_ix_warp import build_ix_warp256_leaf_source
        return build_ix_warp256_leaf_source(
            plan, io_mode, perm_form, _leaf_kernel_params_for_io(plan, io_mode=io_mode),
            ix_warp_fft_pack(), permutation_span)
    from .kernels_hcu_small import small_wave_pack, build_small_wave
    short_pack = small_wave_pack(plan, io_mode)
    if short_pack:
        return build_small_wave(plan, io_mode, _leaf_kernel_params_for_io(plan, io_mode=io_mode), short_pack)
    from .kernels_hcu_warp import (hcu_warp_store_pack, build_hcu_warp_store,
                                   hcu_dense_wave_pack, build_hcu_dense_wave,
                                   hcu_packed_transposed_pack)
    native_packed_cta = (_hcu_backend_active()
                         and os.getenv("FLAGFFT_HCU_PACKED_REAL_CTA", "0") == "1")
    packed_transpose = io_mode == "packed_r2c" and perm_form == "outer_packed"
    wave_transpose = (native_packed_cta and packed_transpose
        and os.getenv("FLAGFFT_HCU_PACKED_WAVE_LAYOUT", "wave") == "resource"
        and hcu_packed_cta_batch_pack_for(plan, io_mode) < (8 if plan.dtype == "complex128" else 16))
    if packed_transpose and (not native_packed_cta or wave_transpose):
        pack = hcu_packed_transposed_pack(plan)
        if not pack:
            raise ValueError("transposed packed real boundary requires a supported HCU wave leaf")
        return build_hcu_dense_wave(plan, io_mode,
            _leaf_kernel_params_for_io(plan, io_mode=io_mode), pack, packed_transpose=True)
    dense_wave_pack = hcu_dense_wave_pack(plan, io_mode, four_step_n1, four_step_n2)
    if dense_wave_pack:
        return build_hcu_dense_wave(
            plan, io_mode, _leaf_kernel_params_for_io(
                plan, io_mode=io_mode, include_four_step_twiddle=io_mode in {
                    "four_step_row", "four_step_real_row", "four_step_hermitian_row"}),
            dense_wave_pack, four_step_n1, four_step_n2,
        )
    hcu_warp_pack = hcu_warp_store_pack(
        plan, io_mode, kernel_perm_form, permutation_span, permutation_batch,
    )
    if hcu_warp_pack:
        return build_hcu_warp_store(
            plan, kernel_perm_form, _leaf_kernel_params_for_io(
                plan, io_mode=io_mode, batched_permutation=batched_permutation),
            hcu_warp_pack, permutation_span, permutation_batch, io_mode,
        )
    bluestein_real_kind = ""
    if (io_mode.startswith("bluestein_") and io_mode.endswith(("_r2c", "_c2r"))
            or io_mode == "rader_full_leaf_c2r"):
        bluestein_real_kind = io_mode.rsplit("_", 1)[1]
        io_mode = io_mode.rsplit("_", 1)[0]
    from .target import ix_warp_fft_pack
    warp_pack = ix_warp_fft_pack()
    from .backend_profile import current_profile
    if (_ix_backend_active() and perm_form == "ix_ct_packed_wave_c2r"
            and plan.dtype == "complex64" and plan.length in {512, 1024}
            and plan.direction == "inverse" and io_mode == "packed_c2r"):
        from .kernels_ix_warp import build_ix_ct_packed_inverse_source
        return build_ix_ct_packed_inverse_source(plan, _leaf_kernel_params_for_io(plan, io_mode=io_mode))
    if (warp_pack and _ix_backend_active() and current_profile().device_arch=='71'
            and current_profile().warp_size==64 and plan.dtype=='complex64'
            and io_mode in {'four_step_row','four_step_col','four_step_real_row',
                            'four_step_hermitian_row','four_step_r2c_col','four_step_c2r_col'}):
        from .kernels_ix_warp import build_ix_warp_four_step_source
        return build_ix_warp_four_step_source(plan,io_mode,four_step_n1,four_step_n2,
            _leaf_kernel_params_for_io(plan,io_mode=io_mode,
                include_four_step_twiddle=io_mode in {'four_step_row','four_step_real_row','four_step_hermitian_row'}),
            warp_pack, dense_io=perm_form == "ix_r2c_dense_fourstep")
    if (warp_pack and _ix_backend_active() and current_profile().device_arch == "71"
            and current_profile().warp_size == 64 and plan.dtype == "complex64" and plan.length in {16,64,128,256,512,1024,2048}
            and io_mode in {"contiguous", "contiguous_c2r", "contiguous_r2c", "packed_r2c", "packed_c2r", "permuted_store",
                            "strided_permuted_store", "permuted_r2c", "permuted_store_r2c"}
            and perm_form.removesuffix("_padinput16").removesuffix("_padoutput16") in {
                "outer", "inner", "outer_dif", "inner_dif", "outer_dif_chunk1",
                "inner_dif_chunk1", "outer_dif_chunk4", "outer_strided",
                "outer_strided_chunk16", "outer_c2r_chunk16"}
            and (io_mode == "strided_permuted_store") == (
                perm_form in {"outer_strided", "outer_strided_chunk16"})):
        from .kernels_ix_warp import build_ix_warp256_leaf_source
        return build_ix_warp256_leaf_source(
            plan,io_mode,perm_form,_leaf_kernel_params_for_io(plan,io_mode=io_mode),
            warp_pack,permutation_span)
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
            prime_n=prime_n,
            bluestein_real_kind=bluestein_real_kind,
        )

    portable_exchange = _portable_leaf_backend_active()
    ix_packed_tle = (plan.dtype == "complex64" and not portable_exchange
                     and ix_packed_real_tle(plan.length, plan.factors, io_mode))
    native_packed_r2c = (io_mode == "packed_r2c" and (
        (current_profile().backend == "cuda" and os.getenv("FLAGFFT_CUDA_FUSED_PACKED_REAL") == "1")
        or (_hcu_backend_active() and os.getenv("FLAGFFT_HCU_FUSED_PACKED_REAL") == "1")
        or native_packed_cta or ix_packed_tle))
    if io_mode == "packed_r2c" and not portable_exchange and not native_packed_r2c:
        raise ValueError("packed R2C leaf requires portable exchange or an enabled native packed boundary")
    factors = emitted_leaf_factors(plan, io_mode)
    n = plan.length
    from .target import ix_ct_batch_default_enabled, ix_fourstep_preset

    ix_short_packed_real = (
        ix_packed_tle and io_mode == "packed_r2c"
        and ix_ct_batch_default_enabled() and ix_fourstep_preset() == 9
        and n == 1024 and factors == (8, 8, 8, 2) and plan.num_warps == 4
    )
    ix_512_packed_real = (
        ix_packed_tle and io_mode == "packed_r2c"
        and ix_ct_batch_default_enabled() and ix_fourstep_preset() == 11
        and n == 512 and factors == (8, 4, 8, 2) and plan.num_warps == 2
    )
    ix_flat_packed_real = ix_short_packed_real or ix_512_packed_real
    inner_perm_form = perm_form in {"inner", "inner_middle", "inner_middle_c2r_cube"}
    smem_n = plan.smem_size
    stage_lanes = (
        tuple(n // radix for radix in factors)
        if portable_exchange or ix_flat_packed_real
        else cooperative_stage_lanes_for(plan)
    )
    uses_cooperative_stage_lanes = portable_exchange or any(
        lanes != plan.lanes for lanes in stage_lanes
    )
    active_lanes = max(stage_lanes[:-1] if ix_flat_packed_real else stage_lanes, default=plan.lanes)
    lane_block = lane_block_for(active_lanes)
    contiguous_modes = {
        "contiguous",
        "strided",
        "permuted_store",
        "strided_permuted_store",
        "permuted_store_r2c",
        "contiguous_r2c",
        "permuted_r2c",
        "packed_r2c",
        "packed_c2r",
        "contiguous_c2r",
        "bluestein_prepare_leaf",
        "bluestein_finish_leaf",
        "bluestein_full_leaf",
        "rader_full_leaf",
        "rader_prepare_leaf",
        "rader_finish_leaf",
    }
    if io_mode in {"permuted_store", "strided_permuted_store", "permuted_r2c", "permuted_store_r2c"}:
        final_pack = None
        if _hcu_backend_active():
            pack_knob = {
                "outer_first": "FLAGFFT_HCU_3D_FIRST_PACK",
                "inner_middle": "FLAGFFT_HCU_3D_MIDDLE_PACK",
                "inner_middle_c2r_cube": "FLAGFFT_HCU_3D_MIDDLE_PACK",
                "outer_last": "FLAGFFT_HCU_3D_FINAL_PACK",
            }.get(perm_form)
            requested_pack = os.getenv(pack_knob, "auto") if pack_knob else "auto"
            if requested_pack != "auto":
                final_pack = int(requested_pack)
            elif _hcu_backend_active() and perm_form == "inner_middle_c2r_cube":
                final_pack = 8
        batch_pack = permuted_store_batch_pack_for(
            plan, force_full_smem=hcu_full_smem, pack_override=final_pack,
            permutation_span=permutation_span, permutation_batch=permutation_batch,
            inner=inner_perm_form,
        )
    elif io_mode in contiguous_modes:
        real_transform = (
            "r2c"
            if io_mode in {"contiguous_r2c", "permuted_r2c", "permuted_store_r2c", "packed_r2c"}
            else "c2r"
            if io_mode in {"packed_c2r", "contiguous_c2r"}
            else None
        )
        batch_pack = contiguous_batch_pack_for(
            plan,
            real_boundary=real_transform is not None,
            real_transform=real_transform,
        )
        batch_pack = hcu_packed_cta_batch_pack_for(plan, io_mode) or batch_pack
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
    ix_packed_r2c_finish = (io_mode == "packed_r2c" and ix_packed_tle
                            and single_smem_buffer and batch_pack == 1)
    ix_large_packed_twiddles = (
        ix_packed_r2c_finish and n == 8192 and ix_fourstep_preset() == 8
        and _maca_knob("R2C_FINISH_SQUARE_ROTATION", "0") == "1"
        and _maca_knob("RECURRENCE") == "1"
    )
    ix_packed_twiddles = ix_flat_packed_real or ix_large_packed_twiddles
    ix_c2r_radix2 = (io_mode == "packed_c2r" and ix_packed_tle
                     and single_smem_buffer and batch_pack == 1)
    packed_smem_alias = (native_packed_cta and native_packed_r2c and single_smem_buffer
                         and len(factors) > 1 and n == stage_lanes[-1] * factors[-1])
    smem_pack = max(batch_pack, inner_pack)
    # Each exchange uses its own permutation. The final radix two reads
    # contiguous frequency bands directly from the unpermuted tile.
    ix_packed_bank_shifts = (
        (5, 3, 0) if ix_flat_packed_real and _maca_knob("SMEM_SWIZZLE", "0") == "1" else ()
    )
    ix_ct_batch_leaf = (
        _ix_backend_active() and current_profile().device_arch == "71"
        and ix_ct_batch_default_enabled() and plan.dtype == "complex64"
        and portable_exchange and smem_pack == 1
    )
    ix_compact_tail = (
        ix_ct_batch_leaf and ix_fourstep_preset() == 9 and n == 2048
        and factors == (16, 8, 16) and plan.num_warps == 4 and io_mode == "contiguous"
        and _maca_knob("EXCHANGE") in {"direct", "direct_all"}
    )
    ix_real_ct_codelet = (
        ix_ct_batch_leaf and ix_fourstep_preset() == 11 and n == 1024
        and factors == (16, 8, 8) and plan.num_warps == 2 and io_mode == "contiguous_r2c"
    )
    if portable_exchange and len(factors) > 1:
        # The MetaX plugin cannot lower the warp-shuffle path, so the exchange
        # must gather from a tensor wider than one 64-thread warp.  Packing
        # widens that tensor for free, whereas raising the lane block idles
        # most lanes on a small leaf -- which is what starved the throughput
        # bound batch and 3D shapes.
        lane_block = max(lane_block, _portable_exchange_lane_floor(smem_pack))
    vector_block = lane_block * smem_pack
    smem_slot_stride = plan.smem_size + 1 if batch_pack >= 4 else plan.smem_size
    if (_hcu_backend_active() and single_smem_buffer and
            os.getenv("FLAGFFT_HCU_LEAF_SINGLE_BUFFER", "0") == "1" and
            io_mode in {"contiguous", "strided", "contiguous_r2c", "contiguous_c2r",
                        "packed_r2c", "packed_c2r"}):
        smem_slot_stride = plan.smem_size
    if (_hcu_backend_active() and single_smem_buffer
            and os.getenv("FLAGFFT_HCU_3D_STORE_MODE", "auto") == "cyclic"
            and io_mode in {"permuted_store", "strided_permuted_store",
                            "permuted_r2c", "permuted_store_r2c"}):
        smem_slot_stride = plan.smem_size
    if current_profile().backend == "cuda" and io_mode in {
        "permuted_store", "strided_permuted_store", "permuted_r2c", "permuted_store_r2c"
    }:
        padding = os.getenv("FLAGFFT_CUDA_PERMSTORE_PAD", "1")
        if padding not in {"0", "1"}:
            raise ValueError("FLAGFFT_CUDA_PERMSTORE_PAD must be 0 or 1")
        if padding == "0":
            smem_slot_stride = plan.smem_size
    if _ix_batch_smem_interleave(io_mode, portable_exchange, batch_pack):
        # Adjacent FFT slots occupy adjacent words. Padding each power-of-two
        # FFT would double the rounded allocation without separating slots.
        smem_slot_stride = plan.smem_size
    smem_n = lane_block_for(smem_slot_stride * smem_pack)
    permuted_smem_store = (
        _ix_backend_active() and os.getenv("FLAGFFT_IX_3D_SMEM_STORE", "0") == "1"
        and _ix_batch_smem_interleave(io_mode, portable_exchange, batch_pack)
        and plan.dtype == "complex64" and (64 <= n <= 256 or _ix_permuted_single_smem(plan))
        and n & (n - 1) == 0
        and len(factors) >= 2 and batch_pack > 1
        and n == stage_lanes[-1] * factors[-1]
        and io_mode in {"permuted_store", "strided_permuted_store", "permuted_r2c", "permuted_store_r2c"}
    )
    # Reuse the final stage's input allocation after every input has been
    # consumed. One batch-contiguous exchange replaces per-digit layout
    # conversions; incomplete register groups and per-cube batching retain
    # the ordinary store until their lifetime/addressing can be guaranteed.
    hcu_smem_store = (
        _hcu_backend_active()
        and os.getenv("FLAGFFT_HCU_3D_STORE_MODE", "auto") == "smem"
        and not portable_exchange
        and not batched_permutation
        and n > 0 and n & (n - 1) == 0
        and len(factors) >= 2 and batch_pack > 1
        and n == stage_lanes[-1] * factors[-1]
        and io_mode in {"permuted_store", "strided_permuted_store"}
    )
    permuted_smem_store = permuted_smem_store or hcu_smem_store
    params = _leaf_kernel_params_for_io(
        plan,
        io_mode=io_mode,
        ix_packed_twiddles=ix_packed_twiddles,
        batched_permutation=batched_permutation,
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

    ix_split_c2r = perm_form == "ix_split_c2r_frequency"
    ix_split_frequency = ix_split_c2r or perm_form == "ix_split_frequency"
    if ix_split_frequency:
        if not (_ix_backend_active() and n == 1024 and plan.dtype == "complex64"
                and io_mode == "contiguous" and batch_pack == 1
                and (not ix_split_c2r or plan.direction == "inverse")):
            raise ValueError("IX split frequencies require a single 1024-point FP32 half transform")
        params.insert(-1, "split_roots_ptr")
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
    elif io_mode == "permuted_r2c":
        perm_tag = f"_{perm_form}" if perm_form != "outer" else ""
        kernel_name = f"r2c_permuted_store{perm_tag}_leaf_kernel_{suffix}_l{plan.lanes}_b{lane_block}"
    elif io_mode == "packed_r2c":
        tag = "_transpose" if packed_transpose else ""
        kernel_name = f"packed_r2c{tag}_leaf_kernel_{suffix}_l{plan.lanes}_b{lane_block}"
    elif io_mode == "packed_c2r":
        kernel_name = f"packed_c2r_leaf_kernel_{suffix}_l{plan.lanes}_b{lane_block}"
    elif io_mode == "contiguous_c2r":
        kernel_name = f"c2r_leaf_kernel_{suffix}_l{plan.lanes}_b{lane_block}"
    elif io_mode == "bluestein_prepare_leaf":
        kernel_name = f"bluestein_prepare_leaf_kernel_{suffix}_n{prime_n}_m{n}_l{plan.lanes}_b{lane_block}"
    elif io_mode == "bluestein_finish_leaf":
        kernel_name = f"bluestein_finish_leaf_kernel_{suffix}_n{prime_n}_m{n}_l{plan.lanes}_b{lane_block}"
    elif io_mode == "bluestein_full_leaf":
        kernel_name = f"bluestein_leaf_kernel_{suffix}_n{prime_n}_m{n}_l{plan.lanes}_b{lane_block}{'_' + bluestein_real_kind if bluestein_real_kind else ''}"
    elif io_mode == "rader_full_leaf":
        real_tag = "_c2r" if bluestein_real_kind == "c2r" else ""
        kernel_name = f"rader_full_leaf_kernel_{suffix}_n{prime_n}_m{n}_l{plan.lanes}_b{lane_block}{real_tag}"
    elif io_mode == "rader_prepare_leaf":
        kernel_name = f"rader_prepare_leaf_kernel_{suffix}_n{prime_n}_m{n}_l{plan.lanes}_b{lane_block}"
    elif io_mode == "rader_finish_leaf":
        kernel_name = f"rader_finish_leaf_kernel_{suffix}_n{prime_n}_m{n}_l{plan.lanes}_b{lane_block}"
    elif io_mode.startswith("bluestein_four_step_"):
        kernel_name = (
            f"{kernel_io_mode}_fft_kernel_{suffix}_p{prime_n}_n{four_step_n1}_{four_step_n2}"
            f"_l{plan.lanes}_b{lane_block}"
        )
    elif io_mode in {"permuted_store", "strided_permuted_store"}:
        kernel_prefix = "ifft" if plan.direction == "inverse" else "fft"
        kernel_name = (
            f"permuted_store_{kernel_perm_form}_{kernel_prefix}_kernel_{suffix}"
            f"_l{plan.lanes}_b{lane_block}"
        )
    elif io_mode == "permuted_store_r2c":
        kernel_name = (
            f"permuted_store_r2c_{perm_form}_{suffix}"
            f"_l{plan.lanes}_b{lane_block}_{_dtype_suffix(plan.dtype)}"
        )
    else:
        kernel_prefix = "ifft" if plan.direction == "inverse" else "fft"
        kernel_name = (
            f"{io_mode}_{kernel_prefix}_kernel_{suffix}_n{four_step_n1}_{four_step_n2}"
            f"_l{plan.lanes}_b{lane_block}"
        )
    if permuted_smem_store:
        kernel_name += "_smstore"
    if ix_split_frequency:
        kernel_name += "_split_frequency"
    if ix_split_c2r:
        kernel_name += "_c2r"
    if ix_packed_r2c_finish:
        kernel_name += "_tle_finish"
    if ix_c2r_radix2:
        kernel_name += "_radix2_prepare"
    if permutation_span:
        kernel_name += f"_s{permutation_span}"
    if permutation_batch:
        kernel_name += f"_nb{permutation_batch}"
    cta_disjoint = (batch_pack == 1 and _ix_full_cta_disjoint(n, factors, io_mode, prime_n)
        and all(n % (radix * width) == 0 for radix, width in zip(factors, stage_lanes)))
    real_disjoint = (
        ((io_mode == "contiguous_r2c" and batch_pack in {1, 16})
         or (io_mode == "contiguous_c2r" and batch_pack == 16))
        and n == 210 and factors == (7,6,5) and _maca_knob("WARPS", "") == str(batch_pack)
        and all(n == radix * width for radix,width in zip(factors,stage_lanes))
        and _maca_knob("TLE_REAL_DISJOINT", "0") == "1"
    )
    disjoint_stage = (
        _ix_backend_active() and not portable_exchange and plan.dtype=="complex64"
        and (real_disjoint or cta_disjoint or (inner_pack>1
        and ((64<=n<=256 and len(factors)==2
              and all(radix & (radix-1)==0 for radix in factors)
              and all(n==radix*lanes for radix,lanes in zip(factors,stage_lanes)))
             or (n in _IX_DISJOINT_MIXED_LENGTHS and 2<=len(factors)<=3
                 and all(n % (radix*lanes)==0 for radix,lanes in zip(factors,stage_lanes))
                 and _maca_knob("TLE_DISJOINT_MIXED","0")=="1"))
        and (io_mode.startswith("four_step_") or io_mode.startswith("bluestein_four_step_"))
        and not io_mode.endswith("_strided")))
        and _maca_knob("TLE_DISJOINT_STAGE","0")=="1"
    )
    if disjoint_stage:
        from .ix_disjoint_stage import pass_tag
        kernel_name += "_" + pass_tag()
    body: list[str] = [
        "@triton.jit",
        f"def {kernel_name}(",
    ]
    for idx, param in enumerate(params):
        suffix = "," if idx < len(params) - 1 else ""
        body.append(f"    {param}{suffix}")
    body.append("):")
    if current_profile().backend == "cuda" and io_mode in {"permuted_store", "strided_permuted_store"}:
        span = int(os.getenv("FLAGFFT_CUDA_PERMUTE_SPAN", "0"))
        batch = int(os.getenv("FLAGFFT_CUDA_PERMUTE_BATCH", "0"))
        if span < 0 or batch < 0 or bool(span) != bool(batch):
            raise ValueError("CUDA permutation specialization requires positive span and batch together")
        if span:
            body.extend([f"    perm_span = {span}", f"    nbatch = {batch}"])
    if permutation_span:
        body.append(f"    perm_span = tl.full((), {permutation_span}, tl.int64)")
    if permutation_batch:
        body.append(f"    nbatch = tl.full((), {permutation_batch}, tl.int64)")
    if _portable_complex_vector_io(plan.dtype):
        body.append(f"    {_COMPLEX_PAIR_OFFSETS} = tl.arange(0, 2)[None, :]")
    from .target import ix_ct_single_default_enabled
    static_single = (
        _ix_backend_active() and ix_ct_single_default_enabled()
        and n in {1024, 2048} and plan.dtype == "complex64"
        and io_mode in {"contiguous", "contiguous_r2c", "contiguous_c2r"} and batch_pack == 1
        and _maca_knob("STATIC_SINGLE", "0") == "1"
    )
    if io_mode in contiguous_modes:
        if ix_split_frequency:
            body.extend(["    split_band = tl.program_id(0)", "    nbatch = 1", "    pid = 0", "    batch_id = 0"])
        elif static_single:
            # The native policy identifies a single transform. Keep the runtime
            # ABI while specializing its batch bound and base in this experiment.
            body.extend(["    nbatch = 1", "    pid = 0", "    batch_id = 0"])
        else:
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
            if io_mode in {"permuted_store", "strided_permuted_store"} and inner_perm_form:
                # The packed path derives perm_base from pid below.  With one
                # batch slot, pid already is the row index, but the common
                # inner-form address equations still consume this name.
                body.append("    perm_base = batch_id")
            if io_mode in {"strided", "strided_permuted_store"}:
                stride = "perm_span" if io_mode == "strided_permuted_store" else "outer_stride"
                body.append(f"    batch_index = current_batch // {stride}")
                body.append(
                    f"    batch_base = batch_index * ({n} * {stride}) + "
                    f"(current_batch - batch_index * {stride})"
                )
        else:
            body.append(f"    batch_slot = lane_vec // {lane_block}")
            body.append(f"    lane = lane_vec - batch_slot * {lane_block}")
            body.append("    current_batch = batch_id + batch_slot")
            if io_mode in {"permuted_store", "strided_permuted_store"} and inner_perm_form:
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
            if io_mode in {"strided", "strided_permuted_store"}:
                stride = "perm_span" if io_mode == "strided_permuted_store" else "outer_stride"
                body.append(f"    batch_index = current_batch // {stride}")
                body.append(
                    f"    batch_base = batch_index * ({n} * {stride}) + "
                    f"(current_batch - batch_index * {stride})"
                )
            else:
                body.append(f"    batch_base = current_batch * {n}")
            body.append(f"    smem_offset = batch_slot * {smem_slot_stride}")
        if batch_pack == 1:
            if io_mode not in {"strided", "strided_permuted_store"}:
                body.append(f"    batch_base = current_batch * {n}")
        if io_mode in {"contiguous_r2c", "permuted_r2c", "permuted_store_r2c", "packed_r2c", "packed_c2r", "contiguous_c2r"}:
            body.append("    input_batch_base = current_batch * input_distance")
        if io_mode in {"contiguous_r2c", "permuted_r2c", "packed_r2c", "contiguous_c2r"}:
            body.append("    output_batch_base = current_batch * output_distance")
        if io_mode in {"permuted_store", "strided_permuted_store", "permuted_r2c", "permuted_store_r2c"}:
            # `perm_gbase` is the output address of each batch slot's row start
            # and `perm_k_stride` the stride of the FFT output index; the store
            # adds the two.  The two forms differ in which of the row index's
            # two mixed-radix parts gets scaled by the output layout.
            body.append(f"    perm_slot = tl.arange(0, {batch_pack})")
            if inner_perm_form:
                body.append("    perm_batch = perm_base + perm_slot * perm_span")
                body.append("    perm_i0 = perm_batch // perm_span")
                body.append("    perm_i1 = perm_batch - perm_i0 * perm_span")
                if batched_permutation:
                    body.append("    perm_outer = rows_per_transform // perm_span")
                    body.append(f"    perm_gbase = (perm_i0 // perm_outer) * ({n} * rows_per_transform) + perm_i1 * perm_outer + perm_i0 % perm_outer")
                    body.append("    perm_k_stride = rows_per_transform")
                else:
                    body.append("    perm_gbase = perm_i1 * (nbatch // perm_span) + perm_i0")
                    body.append("    perm_k_stride = nbatch")
            else:
                body.append("    perm_batch = batch_id + perm_slot")
                body.append("    perm_i0 = perm_batch // perm_span")
                body.append("    perm_i1 = perm_batch - perm_i0 * perm_span")
                out_length = n // 2 + 1 if io_mode in {"permuted_r2c", "permuted_store_r2c"} else n
                body.append(
                    f"    perm_gbase = perm_i0 * ({out_length} * perm_span) + perm_i1"
                )
                body.append("    perm_k_stride = perm_span")
            body.append("    perm_mask = perm_batch < nbatch")
            contiguous_slots = (
                permutation_span > 0 and permutation_batch > 0
                and not batched_permutation and batch_pack > 1
                and (permutation_batch % permutation_span == 0
                     and (permutation_batch // permutation_span) % batch_pack == 0
                     if inner_perm_form else permutation_span % batch_pack == 0)
            )
            if contiguous_slots:
                # The plan proves that no tile crosses an output row.  Each
                # tile starts on a pack boundary and its FFT slots are adjacent.
                body.append(f"    perm_gbase = tl.max_contiguous(tl.multiple_of(perm_gbase, {batch_pack}), {batch_pack})")
            if batch_pack > 1 and _ix_backend_active() and os.getenv("FLAGFFT_IX_3D_DIRECT_STORE") == "1":
                body.append("    perm_i0_vector = current_batch // perm_span")
                body.append("    perm_i1_vector = current_batch - perm_i0_vector * perm_span")
                if inner_perm_form:
                    if batched_permutation:
                        body.append(f"    perm_gbase_vector = (perm_i0_vector // perm_outer) * ({n} * rows_per_transform) + perm_i1_vector * perm_outer + perm_i0_vector % perm_outer")
                    else:
                        body.append("    perm_gbase_vector = perm_i1_vector * (nbatch // perm_span) + perm_i0_vector")
                else:
                    body.append(f"    perm_gbase_vector = perm_i0_vector * ({n} * perm_span) + perm_i1_vector")
            if batch_pack == 1:
                # Keep the singleton row address scalar.  On MUSA this avoids
                # a degenerate [1] tensor layout being broadcast into the
                # lane-shaped store pointer.
                if inner_perm_form:
                    body.append("    perm_batch_scalar = perm_base")
                else:
                    body.append("    perm_batch_scalar = batch_id")
                body.append("    perm_i0_scalar = perm_batch_scalar // perm_span")
                body.append(
                    "    perm_i1_scalar = perm_batch_scalar - "
                    "perm_i0_scalar * perm_span"
                )
                if inner_perm_form:
                    if batched_permutation:
                        body.append(f"    perm_gbase_scalar = (perm_i0_scalar // perm_outer) * ({n} * rows_per_transform) + perm_i1_scalar * perm_outer + perm_i0_scalar % perm_outer")
                    else:
                        body.append(
                            "    perm_gbase_scalar = perm_i1_scalar * "
                            "(nbatch // perm_span) + perm_i0_scalar"
                        )
                else:
                    out_length = n // 2 + 1 if io_mode in {"permuted_r2c", "permuted_store_r2c"} else n
                    body.append(
                        f"    perm_gbase_scalar = perm_i0_scalar * "
                        f"({out_length} * perm_span) + perm_i1_scalar"
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

    if native_packed_r2c and not portable_exchange and not ix_packed_r2c_finish:
        tl_dtype = _tl_real_dtype(plan.dtype)
        if packed_smem_alias:
            body.extend(["    packed_r = smem_b_r", "    packed_i = smem_b_i"])
        else:
            body.append(f"    packed_r = tle.gpu.alloc([{smem_n}], dtype={tl_dtype}, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)")
            body.append(f"    packed_i = tle.gpu.alloc([{smem_n}], dtype={tl_dtype}, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)")

    if io_mode in {"bluestein_full_leaf", "rader_full_leaf"}:
        if len(factors) < 2:
            raise ValueError("fused convolution leaf requires at least two FFT stages")
        last_source = "smem_b" if (len(factors) - 1) % 2 == 1 else "smem_a"
        intermediate_buffer = "smem_b" if single_smem_buffer else ("smem_a" if last_source == "smem_b" else "smem_b")

    for stage in range(len(factors)):
        if ix_c2r_radix2 and stage == 0:
            body.extend(_emit_ix_packed_c2r_radix2(n))
            continue
        if ix_packed_r2c_finish and stage == len(factors) - 1:
            if ix_flat_packed_real:
                body.extend(_emit_ix_flat_r2c_finish(n, stage_lanes[stage]))
            else:
                body.extend(_emit_ix_packed_r2c_finish(n, stage_lanes[stage], stage))
            break
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
                if io_mode in {"bluestein_full_leaf", "rader_full_leaf"}
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
                bluestein_real_kind=bluestein_real_kind,
                permuted_smem_store=permuted_smem_store,
                perm_inner=inner_perm_form,
                perm_batched=batched_permutation,
                packed_smem_alias=packed_smem_alias,
                ix_split_frequency=ix_split_frequency,
                ix_split_c2r=ix_split_c2r,
                ix_compact_tail=ix_compact_tail,
                ix_real_ct_codelet=ix_real_ct_codelet,
                ix_dif_packed_real=ix_short_packed_real,
                ix_packed_twiddles=ix_packed_twiddles,
                ix_packed_bank_shifts=ix_packed_bank_shifts,
            )
        )

    if io_mode == "packed_r2c" and not ix_packed_r2c_finish:
        # The final codelet outputs were gathered into natural frequency order
        # by the portable exchange. Reconstruct the real FFT's half spectrum
        # without a second kernel launch or a global intermediate buffer.
        if not portable_exchange:
            body.append("    tl.debug_barrier()")
        body.extend([
            f"    packed_pos = tl.arange(0, {smem_n})",
            f"    packed_slot = packed_pos // {smem_slot_stride}",
            f"    packed_k = packed_pos % {smem_slot_stride}",
            f"    packed_mask = (packed_slot < {batch_pack}) & (packed_k < {n}) & (batch_id + packed_slot < nbatch)",
            f"    packed_partner = packed_slot * {smem_slot_stride} + tl.where(packed_k == 0, 0, {n} - packed_k)",
            "    packed_partner = tl.where(packed_mask, packed_partner, 0)",
            *( ["    packed_ar = packed_r", "    packed_ai = packed_i",
                "    packed_br = tl.gather(packed_r, packed_partner, 0)",
                "    packed_bi = tl.gather(packed_i, packed_partner, 0)"] if portable_exchange else [
                "    packed_ar = tl.load(tle.gpu.local_ptr(packed_r, (packed_pos,)), mask=packed_mask, other=0.0)",
                "    packed_ai = tl.load(tle.gpu.local_ptr(packed_i, (packed_pos,)), mask=packed_mask, other=0.0)",
                "    packed_br = tl.load(tle.gpu.local_ptr(packed_r, (packed_partner,)), mask=packed_mask, other=0.0)",
                "    packed_bi = tl.load(tle.gpu.local_ptr(packed_i, (packed_partner,)), mask=packed_mask, other=0.0)"]),
            "    packed_twiddle_k = tl.where(packed_mask, packed_k, 0)",
            "    packed_wr = tl.load(packed_twiddle_ptr + packed_twiddle_k * 2, mask=packed_mask, other=0.0)",
            "    packed_wi = tl.load(packed_twiddle_ptr + packed_twiddle_k * 2 + 1, mask=packed_mask, other=0.0)",
            "    packed_dr = packed_ar - packed_br",
            "    packed_di = packed_ai + packed_bi",
            "    packed_xr = 0.5 * (packed_ar + packed_br + packed_wi * packed_dr + packed_wr * packed_di)",
            "    packed_xi = 0.5 * (packed_ai - packed_bi - packed_wr * packed_dr + packed_wi * packed_di)",
            "    packed_dst = (batch_id + packed_slot) * output_distance + packed_k",
            "    tl.store(out_ptr + packed_dst * 2, packed_xr, mask=packed_mask)",
            "    tl.store(out_ptr + packed_dst * 2 + 1, packed_xi, mask=packed_mask)",
            f"    packed_nyquist = (batch_id + packed_slot) * output_distance + {n}",
            "    tl.store(out_ptr + packed_nyquist * 2, packed_ar - packed_ai, mask=packed_mask & (packed_k == 0))",
            "    tl.store(out_ptr + packed_nyquist * 2 + 1, 0.0, mask=packed_mask & (packed_k == 0))",
        ])

        if packed_transpose:
            # Adjacent output lanes own adjacent FFT rows. Source rows may
            # cross matrix boundaries, including a final partial CTA.
            replacements = {
                f"    packed_slot = packed_pos // {smem_slot_stride}":
                    f"    packed_slot = packed_pos % {batch_pack}",
                f"    packed_k = packed_pos % {smem_slot_stride}":
                    f"    packed_k = packed_pos // {batch_pack}",
                "    packed_ar = tl.load(tle.gpu.local_ptr(packed_r, (packed_pos,)), mask=packed_mask, other=0.0)":
                    f"    packed_ar = tl.load(tle.gpu.local_ptr(packed_r, (packed_slot * {smem_slot_stride} + packed_k,)), mask=packed_mask, other=0.0)",
                "    packed_ai = tl.load(tle.gpu.local_ptr(packed_i, (packed_pos,)), mask=packed_mask, other=0.0)":
                    f"    packed_ai = tl.load(tle.gpu.local_ptr(packed_i, (packed_slot * {smem_slot_stride} + packed_k,)), mask=packed_mask, other=0.0)",
                "    packed_dst = (batch_id + packed_slot) * output_distance + packed_k":
                    f"    packed_dst = ((batch_id + packed_slot) // output_distance) * ({n+1} * output_distance) + packed_k * output_distance + (batch_id + packed_slot) % output_distance",
                f"    packed_nyquist = (batch_id + packed_slot) * output_distance + {n}":
                    f"    packed_nyquist = ((batch_id + packed_slot) // output_distance) * ({n+1} * output_distance) + {n} * output_distance + (batch_id + packed_slot) % output_distance",
            }
            body = [replacements.get(line, line) for line in body]

        if ((_ix_backend_active() and _packed_maca_fp32_complex_io(plan.dtype))
                or (_hcu_backend_active() and _hcu_vector_io(plan.dtype))):
            for old_lines,new_lines in (
                (["    packed_wr = tl.load(packed_twiddle_ptr + packed_twiddle_k * 2, mask=packed_mask, other=0.0)",
                  "    packed_wi = tl.load(packed_twiddle_ptr + packed_twiddle_k * 2 + 1, mask=packed_mask, other=0.0)"],
                 _emit_vectorized_complex_load("    ","packed_twiddle_ptr + packed_twiddle_k * 2","packed_mask","packed_wr, packed_wi",plan.dtype)),
                (["    tl.store(out_ptr + packed_dst * 2, packed_xr, mask=packed_mask)",
                  "    tl.store(out_ptr + packed_dst * 2 + 1, packed_xi, mask=packed_mask)"],
                 _emit_vectorized_complex_store("    ","out_ptr + packed_dst * 2","packed_xr","packed_xi","packed_mask",plan.dtype)),
            ):
                index=body.index(old_lines[0])
                assert body[index:index+len(old_lines)]==old_lines
                body[index:index+len(old_lines)]=new_lines

    if io_mode in {"bluestein_full_leaf", "rader_full_leaf"}:
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
                    bluestein_real_kind=bluestein_real_kind,
                )
            )

    source="\n".join(body)
    if disjoint_stage:
        import hashlib
        # Separate source contracts even when historical radix names coincide.
        digest=hashlib.sha256(source.encode()).hexdigest()[:12]
        base,tag=kernel_name.rsplit("_ixdisjoint_",1)
        renamed=base+"_s"+digest+"_ixdisjoint_"+tag
        source=source.replace(f"def {kernel_name}(",f"def {renamed}(",1)
        kernel_name=renamed
    return kernel_name, source


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
