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

import math
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import dedent
from typing import Literal

import triton
import triton.language as tl

_MODULE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parents[1]
_CODELET_DIR = _MODULE_DIR / "codelet"
_FOUR_STEP_TILE_ROWS = 32
_FOUR_STEP_TILE_COLS = 32
_FOUR_STEP_NUM_WARPS = 4
_FOUR_STEP_COL_INNER_PACK = 2
_FOUR_STEP_LARGE_INNER_PACK = 4
_FOUR_STEP_COL_INNER_PACK_MIN_N1 = 128
_FOUR_STEP_ROW_INNER_PACK_MAX_N1 = 512
_FOUR_STEP_PACKED_COL_LEAF_MAX_N2 = 1024
_TLE_FUSED_TWIDDLE_MIN_LENGTH = 1 << 18
_TLE_FUSED_TWIDDLE_MAX_LEAF = 1024
_TLE_SMEM_SWIZZLE_SHIFT = 5
_THREAD_LOCAL_MIXED_RADICES = frozenset({18, 20, 24, 25, 27, 28, 30, 32})
_COOPERATIVE_STAGE_MIN_LENGTH = 512
_COOPERATIVE_STAGE_MAX_LENGTH = 2048
_COOPERATIVE_STAGE_MAX_BASE_LANES = 8
_COOPERATIVE_STAGE_MAX_LANES = 128
_LEAF_PACK_TARGET_THREADS = 32
_LEAF_PACK_SMEM_BUDGET_BYTES = 48 * 1024
_NATURAL_ORDER_CODELET_RADICES = frozenset(
    {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 19}
)
SPECIALIZED_INLINE_CODELET_RADICES: set[int] = set()


def _is_double_dtype(dtype: str) -> bool:
    return dtype in ("complex128", "float64")


def _tl_real_dtype(dtype: str) -> str:
    return "tl.float64" if _is_double_dtype(dtype) else "tl.float32"


def _real_element_bytes(dtype: str) -> int:
    return 8 if _is_double_dtype(dtype) else 4


def _dtype_suffix(dtype: str) -> str:
    return "f64" if _is_double_dtype(dtype) else "f32"


def _zero_other(dtype: str) -> str:
    # auto-promoted by Triton's masked-load lowering; one literal works for fp32 and fp64
    del dtype
    return "0.0"


@dataclass(frozen=True)
class LeafPlan:
    length: int
    factors: tuple[int, ...]
    remainder: int
    lanes: int
    num_warps: int
    generic_radices: tuple[int, ...]
    smem_size: int
    direction: Literal["forward", "inverse"] = "forward"
    dtype: Literal["complex64", "complex128"] = "complex64"
    kind: Literal["ct_leaf"] = field(default="ct_leaf", init=False)


LeafIoMode = Literal[
    "contiguous",
    "contiguous_r2c",
    "contiguous_c2r",
    "four_step_row",
    "four_step_real_row",
    "four_step_hermitian_row",
    "four_step_col",
    "four_step_r2c_col",
    "four_step_c2r_col",
]


def lane_block_for(lanes: int) -> int:
    if lanes <= 1:
        return 1
    value = 1
    while value < lanes:
        value <<= 1
    return value


def cooperative_stage_lanes_for(plan: LeafPlan) -> tuple[int, ...]:
    fixed_lanes_are_compatible = all(
        (plan.length // radix) % plan.lanes == 0 for radix in plan.factors
    )
    if not (
        plan.dtype == "complex64"
        and _COOPERATIVE_STAGE_MIN_LENGTH
        <= plan.length
        <= _COOPERATIVE_STAGE_MAX_LENGTH
        and len(plan.factors) >= 2
        and (
            plan.lanes < _COOPERATIVE_STAGE_MAX_BASE_LANES
            or not fixed_lanes_are_compatible
        )
    ):
        return (plan.lanes,) * len(plan.factors)

    stage_lanes: list[int] = []
    for radix in plan.factors:
        codelets = plan.length // radix
        upper = min(codelets, _COOPERATIVE_STAGE_MAX_LANES)
        stage_lanes.append(
            next(
                candidate
                for candidate in range(upper, 0, -1)
                if codelets % candidate == 0
            )
        )
    return tuple(stage_lanes)


def _floor_power_of_two(value: int) -> int:
    power = 1
    while power * 2 <= value:
        power *= 2
    return power


def contiguous_batch_pack_for(plan: LeafPlan) -> int:
    lane_block = lane_block_for(plan.lanes)
    if lane_block >= _LEAF_PACK_TARGET_THREADS:
        return 1

    thread_pack = max(1, _LEAF_PACK_TARGET_THREADS // lane_block)
    if plan.length <= 128:
        thread_pack = min(thread_pack, 4)
    if len(plan.factors) <= 1:
        return thread_pack

    bytes_per_fft = 4 * (plan.smem_size + 1) * _real_element_bytes(plan.dtype)
    smem_pack = max(1, _LEAF_PACK_SMEM_BUDGET_BYTES // bytes_per_fft)
    return _floor_power_of_two(max(1, min(thread_pack, smem_pack)))


def four_step_col_inner_pack_for(n1: int, n2: int, dtype: str = "complex64") -> int:
    if n1 < _FOUR_STEP_COL_INNER_PACK_MIN_N1:
        return 1
    if use_tle_fused_twiddle(n1, n2, dtype):
        return _FOUR_STEP_LARGE_INNER_PACK
    if n2 > _FOUR_STEP_PACKED_COL_LEAF_MAX_N2:
        return 1
    return _FOUR_STEP_COL_INNER_PACK


def four_step_row_inner_pack_for(n1: int, n2: int, dtype: str = "complex64") -> int:
    if use_tle_fused_twiddle(n1, n2, dtype):
        return _FOUR_STEP_LARGE_INNER_PACK
    if (
        not _is_double_dtype(dtype)
        and n1 <= _FOUR_STEP_ROW_INNER_PACK_MAX_N1
        and n2 > _FOUR_STEP_PACKED_COL_LEAF_MAX_N2
    ):
        return _FOUR_STEP_LARGE_INNER_PACK
    return 1


def use_tle_fused_twiddle(n1: int, n2: int, dtype: str = "complex64") -> bool:
    """Move large FP32 Four-Step twiddles into the row pass.

    Both leaves are capped at 1024 so pack=4 stays within the A100 dynamic
    shared-memory budget for the generated mixed-radix kernels.
    """
    return (
        not _is_double_dtype(dtype)
        and n1 * n2 >= _TLE_FUSED_TWIDDLE_MIN_LENGTH
        and n1 <= _TLE_FUSED_TWIDDLE_MAX_LEAF
        and n2 <= _TLE_FUSED_TWIDDLE_MAX_LEAF
    )


def _use_single_smem_buffer(
    plan: LeafPlan,
    *,
    io_mode: LeafIoMode,
    four_step_n1: int,
    four_step_n2: int,
) -> bool:
    """Reuse one shared buffer between generated mixed-radix stages."""
    return (
        io_mode.startswith("four_step_")
        and use_tle_fused_twiddle(four_step_n1, four_step_n2, plan.dtype)
        and plan.dtype == "complex64"
        and plan.length == 1024
        and len(plan.factors) > 2
    )


@triton.jit
def _cmul(ar, ai, br, bi):
    return ar * br - ai * bi, ai * br + ar * bi


@triton.jit
def _transpose_complex_kernel(
    src_ptr,
    dst_ptr,
    src_batch_stride,
    src_row_stride,
    src_col_stride,
    dst_batch_stride,
    dst_row_stride,
    dst_col_stride,
    rows,
    cols,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    pid_col = tl.program_id(0)
    pid_row = tl.program_id(1)
    pid_batch = tl.program_id(2)

    row_offsets = pid_row * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col_offsets = pid_col * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    mask = (row_offsets[:, None] < rows) & (col_offsets[None, :] < cols)

    src_base = src_ptr + pid_batch * src_batch_stride
    src_offsets = (
        src_base
        + row_offsets[:, None] * src_row_stride
        + col_offsets[None, :] * src_col_stride
    )
    src_real = tl.load(src_offsets, mask=mask, other=0.0)
    src_imag = tl.load(src_offsets + 1, mask=mask, other=0.0)

    dst_base = dst_ptr + pid_batch * dst_batch_stride
    dst_offsets = (
        dst_base
        + col_offsets[None, :] * dst_row_stride
        + row_offsets[:, None] * dst_col_stride
    )
    tl.store(dst_offsets, src_real, mask=mask)
    tl.store(dst_offsets + 1, src_imag, mask=mask)


@triton.jit
def _twiddle_transpose_complex_kernel(
    src_ptr,
    twiddle_ptr,
    dst_ptr,
    src_batch_stride,
    src_row_stride,
    src_col_stride,
    twiddle_row_stride,
    twiddle_col_stride,
    dst_batch_stride,
    dst_row_stride,
    dst_col_stride,
    rows,
    cols,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    pid_col = tl.program_id(0)
    pid_row = tl.program_id(1)
    pid_batch = tl.program_id(2)

    row_offsets = pid_row * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col_offsets = pid_col * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    mask = (row_offsets[:, None] < rows) & (col_offsets[None, :] < cols)

    src_base = src_ptr + pid_batch * src_batch_stride
    src_offsets = (
        src_base
        + row_offsets[:, None] * src_row_stride
        + col_offsets[None, :] * src_col_stride
    )
    src_real = tl.load(src_offsets, mask=mask, other=0.0)
    src_imag = tl.load(src_offsets + 1, mask=mask, other=0.0)

    tw_offsets = (
        twiddle_ptr
        + row_offsets[:, None] * twiddle_row_stride
        + col_offsets[None, :] * twiddle_col_stride
    )
    tw_real = tl.load(tw_offsets, mask=mask, other=0.0)
    tw_imag = tl.load(tw_offsets + 1, mask=mask, other=0.0)
    out_real, out_imag = _cmul(src_real, src_imag, tw_real, tw_imag)

    dst_base = dst_ptr + pid_batch * dst_batch_stride
    dst_offsets = (
        dst_base
        + col_offsets[None, :] * dst_row_stride
        + row_offsets[:, None] * dst_col_stride
    )
    tl.store(dst_offsets, out_real, mask=mask)
    tl.store(dst_offsets + 1, out_imag, mask=mask)


@triton.jit
def _bluestein_prepare_kernel(
    in_ptr,
    chirp_ptr,
    out_ptr,
    n,
    m,
    nbatch,
    BLOCK: tl.constexpr,
):
    pid_block = tl.program_id(0)
    pid_batch = tl.program_id(1)
    offsets = pid_block * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < m
    in_mask = mask & (offsets < n)

    src = in_ptr + (pid_batch * n + offsets) * 2
    xr = tl.load(src, mask=in_mask, other=0.0)
    xi = tl.load(src + 1, mask=in_mask, other=0.0)
    cr = tl.load(chirp_ptr + offsets * 2, mask=in_mask, other=0.0)
    ci = tl.load(chirp_ptr + offsets * 2 + 1, mask=in_mask, other=0.0)
    yr, yi = _cmul(xr, xi, cr, ci)

    dst = out_ptr + (pid_batch * m + offsets) * 2
    tl.store(dst, yr, mask=mask)
    tl.store(dst + 1, yi, mask=mask)


@triton.jit
def _bluestein_pointwise_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    m,
    nbatch,
    BLOCK: tl.constexpr,
):
    pid_block = tl.program_id(0)
    pid_batch = tl.program_id(1)
    offsets = pid_block * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < m

    a = a_ptr + (pid_batch * m + offsets) * 2
    b = b_ptr + offsets * 2
    ar = tl.load(a, mask=mask, other=0.0)
    ai = tl.load(a + 1, mask=mask, other=0.0)
    br = tl.load(b, mask=mask, other=0.0)
    bi = tl.load(b + 1, mask=mask, other=0.0)
    pr, pi = _cmul(ar, ai, br, bi)

    dst = out_ptr + (pid_batch * m + offsets) * 2
    tl.store(dst, pr, mask=mask)
    tl.store(dst + 1, -pi, mask=mask)


@triton.jit
def _bluestein_finalize_kernel(
    in_ptr,
    chirp_ptr,
    out_ptr,
    n,
    m,
    nbatch,
    BLOCK: tl.constexpr,
):
    pid_block = tl.program_id(0)
    pid_batch = tl.program_id(1)
    offsets = pid_block * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n

    src = in_ptr + (pid_batch * m + offsets) * 2
    xr = tl.load(src, mask=mask, other=0.0) / m
    xi = -tl.load(src + 1, mask=mask, other=0.0) / m
    cr = tl.load(chirp_ptr + offsets * 2, mask=mask, other=0.0)
    ci = tl.load(chirp_ptr + offsets * 2 + 1, mask=mask, other=0.0)
    yr, yi = _cmul(xr, xi, cr, ci)

    dst = out_ptr + (pid_batch * n + offsets) * 2
    tl.store(dst, yr, mask=mask)
    tl.store(dst + 1, yi, mask=mask)


def _build_rader_prepare_kernel_source(dtype: str) -> tuple[str, str, list[str]]:
    zero = _zero_other(dtype)
    return (
        "_rader_prepare_kernel",
        dedent(
            f"""
            @triton.jit
            def _rader_prepare_kernel(
                in_ptr,
                idx_ptr,
                out_ptr,
                n,
                m,
                nbatch,
            ):
                pid_block = tl.program_id(0)
                pid_batch = tl.program_id(1)
                offsets = pid_block * 256 + tl.arange(0, 256)
                mask = offsets < m
                inv_offsets = tl.where(offsets == 0, 0, m - offsets)
                src_index = tl.load(idx_ptr + inv_offsets, mask=mask, other=0)

                src = in_ptr + (pid_batch * n + src_index) * 2
                xr = tl.load(src, mask=mask, other={zero})
                xi = tl.load(src + 1, mask=mask, other={zero})

                dst = out_ptr + (pid_batch * m + offsets) * 2
                tl.store(dst, xr, mask=mask)
                tl.store(dst + 1, xi, mask=mask)
            """
        ),
        ["in_ptr", "idx_ptr", "out_ptr", "n", "m", "nbatch"],
    )


def _build_rader_pointwise_kernel_source(dtype: str) -> tuple[str, str, list[str]]:
    zero = _zero_other(dtype)
    return (
        "_rader_pointwise_kernel",
        dedent(
            f"""
            @triton.jit
            def _rader_pointwise_kernel(
                a_ptr,
                b_ptr,
                out_ptr,
                m,
                nbatch,
            ):
                pid_block = tl.program_id(0)
                pid_batch = tl.program_id(1)
                offsets = pid_block * 256 + tl.arange(0, 256)
                mask = offsets < m

                a = a_ptr + (pid_batch * m + offsets) * 2
                b = b_ptr + offsets * 2
                ar = tl.load(a, mask=mask, other={zero})
                ai = tl.load(a + 1, mask=mask, other={zero})
                br = tl.load(b, mask=mask, other={zero})
                bi = tl.load(b + 1, mask=mask, other={zero})
                pr, pi = _cmul(ar, ai, br, bi)

                dst = out_ptr + (pid_batch * m + offsets) * 2
                tl.store(dst, pr, mask=mask)
                tl.store(dst + 1, -pi, mask=mask)
            """
        ),
        ["a_ptr", "b_ptr", "out_ptr", "m", "nbatch"],
    )


def _build_rader_finalize_kernel_source(
    n: int, dtype: str
) -> tuple[str, str, list[str]]:
    zero = _zero_other(dtype)
    div_cast = "tl.cast(m, tl.float64)" if dtype == "complex128" else "m"
    sum_block = 1
    while sum_block < n:
        sum_block <<= 1
    return (
        "_rader_finalize_kernel",
        dedent(
            f"""
            @triton.jit
            def _rader_finalize_kernel(
                input_ptr,
                conv_ptr,
                idx_ptr,
                out_ptr,
                n,
                m,
                nbatch,
            ):
                pid_block = tl.program_id(0)
                pid_batch = tl.program_id(1)
                offsets = pid_block * 256 + tl.arange(0, 256)
                mask = offsets < m

                src = conv_ptr + (pid_batch * m + offsets) * 2
                cr = tl.load(src, mask=mask, other={zero}) / {div_cast}
                ci = -tl.load(src + 1, mask=mask, other={zero}) / {div_cast}
                x0 = input_ptr + pid_batch * n * 2
                x0r = tl.load(x0)
                x0i = tl.load(x0 + 1)

                dst_index = tl.load(idx_ptr + offsets, mask=mask, other=0)
                dst = out_ptr + (pid_batch * n + dst_index) * 2
                tl.store(dst, x0r + cr, mask=mask)
                tl.store(dst + 1, x0i + ci, mask=mask)

                sum_offsets = tl.arange(0, {sum_block})
                sum_mask = sum_offsets < n
                sum_src = input_ptr + (pid_batch * n + sum_offsets) * 2
                sr = tl.load(sum_src, mask=sum_mask, other={zero})
                si = tl.load(sum_src + 1, mask=sum_mask, other={zero})
                out0 = out_ptr + pid_batch * n * 2
                block0 = pid_block == 0
                tl.store(out0, tl.sum(sr, axis=0), mask=block0)
                tl.store(out0 + 1, tl.sum(si, axis=0), mask=block0)
            """
        ),
        ["input_ptr", "conv_ptr", "idx_ptr", "out_ptr", "n", "m", "nbatch"],
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


def _emit_inline_constant_codelet(
    indent: str,
    radix: int,
    lane_block: int,
    direction: Literal["forward", "inverse"],
    dtype: str = "complex64",
) -> list[str]:
    lines: list[str] = []
    sign = _direction_sign(direction)
    for kout in range(radix):
        lines.append(f"{indent}acc_r_{kout} = tl.zeros_like(r0)")
        lines.append(f"{indent}acc_i_{kout} = tl.zeros_like(i0)")

    for kout in range(radix):
        for nin in range(radix):
            angle = sign * 2.0 * math.pi * kout * nin / float(radix)
            wr = _fmt_const(math.cos(angle))
            wi = _fmt_const(math.sin(angle))
            lines.append(f"{indent}pr, pi = _cmul(r{nin}, i{nin}, {wr}, {wi})")
            lines.append(f"{indent}acc_r_{kout} += pr")
            lines.append(f"{indent}acc_i_{kout} += pi")

    for kout in range(radix):
        lines.append(f"{indent}r{kout} = acc_r_{kout}")
        lines.append(f"{indent}i{kout} = acc_i_{kout}")
    return lines


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
                lines.append(
                    f"{indent}r{j} = tl.load(in_ptr + (batch_base + in{j}) * 2, mask=lane_mask, other={zero})"
                )
                lines.append(
                    f"{indent}i{j} = tl.load(in_ptr + (batch_base + in{j}) * 2 + 1, mask=lane_mask, other={zero})"
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
    elif radix in SPECIALIZED_INLINE_CODELET_RADICES:
        lines.extend(
            _emit_inline_constant_codelet(indent, radix, lane_block, direction, dtype)
        )
    else:
        lines.extend(_emit_table_codelet(indent, radix, lane_block, dtype))

    for j in range(radix):
        if is_last:
            lines.extend(_emit_output_index(indent, f"out_idx{j}", factors, j))
            if io_mode == "contiguous":
                lines.append(
                    f"{indent}tl.store(out_ptr + (batch_base + out_idx{j}) * 2, r{j}, mask=lane_mask)"
                )
                lines.append(
                    f"{indent}tl.store(out_ptr + (batch_base + out_idx{j}) * 2 + 1, i{j}, mask=lane_mask)"
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
            elif io_mode in {
                "four_step_row",
                "four_step_real_row",
                "four_step_hermitian_row",
            }:
                lines.append(
                    f"{indent}dst_idx{j} = four_step_inner * {four_step_n1} + out_idx{j}"
                )
                if fuse_twiddle_into_row:
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
    params.append("nbatch")
    return params


def _use_thread_local_mixed_leaf(
    plan: LeafPlan,
    *,
    io_mode: LeafIoMode,
    four_step_n1: int,
    four_step_n2: int,
) -> bool:
    if len(plan.factors) != 2:
        return False
    register_radix, cross_radix = plan.factors
    expected_length = four_step_n1 if io_mode == "four_step_row" else four_step_n2
    return (
        io_mode in {"four_step_row", "four_step_col"}
        and plan.dtype == "complex64"
        and register_radix in _THREAD_LOCAL_MIXED_RADICES
        and cross_radix == 32
        and plan.length == register_radix * cross_radix
        and plan.length == expected_length
        and use_tle_fused_twiddle(four_step_n1, four_step_n2, plan.dtype)
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
    io_mode: Literal["four_step_row", "four_step_col"],
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
    include_outer_twiddle = io_mode == "four_step_row"
    inner_count = four_step_n2 if io_mode == "four_step_row" else four_step_n1
    source_stride = four_step_n2 if io_mode == "four_step_row" else four_step_n1
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
                f"    smem_r = tle.gpu.alloc([{smem_n}], dtype=tl.float32, "
                "layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)"
            ),
            (
                f"    smem_i = tle.gpu.alloc([{smem_n}], dtype=tl.float32, "
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
        body.append(
            f"    input_offset{idx} = " f"(four_step_batch_base + src_idx{idx}) * 2"
        )
        body.append(
            # The selected leaves and pack=4 cover every input lane exactly.
            f"    r{idx}, i{idx} = tl.inline_asm_elementwise("
            '"ld.global.v2.f32 {$0, $1}, [$2];", "=f,=f,l", '
            f"[tl.cast(in_ptr + input_offset{idx}, tl.uint64)], "
            "dtype=(tl.float32, tl.float32), is_pure=False, pack=1)"
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
    inner_twiddle_scale = _direction_sign(plan.direction) * 2.0 * math.pi / plan.length
    for chunk_base in range(0, register_radix, smem_chunk):
        chunk_indices = range(chunk_base, chunk_base + smem_chunk)
        for idx in chunk_indices:
            if idx > 0:
                body.extend(
                    [
                        (
                            f"    inner_angle{idx} = fft_thread * {idx} * "
                            f"{inner_twiddle_scale:.17g}"
                        ),
                        (
                            f"    inner_tw_i{idx}, inner_tw_r{idx} = "
                            "tl.inline_asm_elementwise("
                            '"sin.approx.f32 $0, $2; '
                            'cos.approx.f32 $1, $2;", '
                            f'"=f,=f,f", [inner_angle{idx}], '
                            "dtype=(tl.float32, tl.float32), "
                            "is_pure=True, pack=1)"
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
        outer_twiddle_scale = (
            _direction_sign(plan.direction)
            * 2.0
            * math.pi
            / (four_step_n1 * four_step_n2)
        )
        body.extend(
            [
                ("    outer_base_idx = fft_thread"),
                (
                    "    outer_base_angle = four_step_inner * outer_base_idx * "
                    f"{outer_twiddle_scale:.17g}"
                ),
                (
                    f"    outer_step_angle = four_step_inner * {register_radix} * "
                    f"{outer_twiddle_scale:.17g}"
                ),
                (
                    "    outer_tw_i, outer_tw_r = tl.inline_asm_elementwise("
                    '"sin.approx.f32 $0, $2; cos.approx.f32 $1, $2;", '
                    '"=f,=f,f", [outer_base_angle], '
                    "dtype=(tl.float32, tl.float32), is_pure=True, pack=1)"
                ),
                (
                    "    outer_step_i, outer_step_r = tl.inline_asm_elementwise("
                    '"sin.approx.f32 $0, $2; cos.approx.f32 $1, $2;", '
                    '"=f,=f,f", [outer_step_angle], '
                    "dtype=(tl.float32, tl.float32), is_pure=True, pack=1)"
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
        body.append(
            f"    output_offset{idx} = " f"(four_step_batch_base + dst_idx{idx}) * 2"
        )
        if register_radix == 32:
            body.append(
                f"    output_dummy{idx} = tl.inline_asm_elementwise("
                '"st.global.v2.f32 [$1], {$2, $3}; mov.u32 $0, 0;", '
                '"=r,l,f,f", '
                f"[tl.cast(out_ptr + output_offset{idx}, tl.uint64), "
                f"r{idx}, i{idx}], "
                "dtype=tl.int32, is_pure=False, pack=1)"
            )
        else:
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
    batch_pack = (
        contiguous_batch_pack_for(plan)
        if io_mode in {"contiguous", "contiguous_r2c", "contiguous_c2r"}
        else 1
    )
    row_modes = {
        "four_step_row",
        "four_step_real_row",
        "four_step_hermitian_row",
    }
    col_modes = {"four_step_col", "four_step_r2c_col", "four_step_c2r_col"}
    if io_mode in row_modes:
        inner_pack = four_step_row_inner_pack_for(
            four_step_n1, four_step_n2, plan.dtype
        )
    elif io_mode in col_modes:
        inner_pack = four_step_col_inner_pack_for(
            four_step_n1, four_step_n2, plan.dtype
        )
    else:
        inner_pack = 1
    fuse_twiddle_into_row = use_tle_fused_twiddle(
        four_step_n1, four_step_n2, plan.dtype
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
                "four_step_real_row",
                "four_step_hermitian_row",
            }
            if fuse_twiddle_into_row
            else io_mode in {"four_step_col", "four_step_r2c_col", "four_step_c2r_col"}
        ),
    )

    suffix = "_".join(str(x) for x in factors)
    if io_mode == "contiguous":
        kernel_prefix = "ifft" if plan.direction == "inverse" else "fft"
        kernel_name = f"{kernel_prefix}_kernel_{suffix}_l{plan.lanes}_b{lane_block}"
    elif io_mode == "contiguous_r2c":
        kernel_name = f"r2c_leaf_kernel_{suffix}_l{plan.lanes}_b{lane_block}"
    elif io_mode == "contiguous_c2r":
        kernel_name = f"c2r_leaf_kernel_{suffix}_l{plan.lanes}_b{lane_block}"
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
    if io_mode in {"contiguous", "contiguous_r2c", "contiguous_c2r"}:
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
    if io_mode in {"contiguous", "contiguous_r2c", "contiguous_c2r"}:
        if batch_pack == 1:
            body.append("    current_batch = batch_id")
            body.append("    lane = lane_vec")
            body.append(f"    lane_mask = lane < {active_lanes}")
        else:
            body.append(f"    batch_slot = lane_vec // {lane_block}")
            body.append(f"    lane = lane_vec - batch_slot * {lane_block}")
            body.append("    current_batch = batch_id + batch_slot")
            body.append(
                f"    lane_mask = (lane < {active_lanes}) & (current_batch < nbatch)"
            )
            body.append(f"    batch_base = current_batch * {n}")
            body.append(f"    smem_offset = batch_slot * {smem_slot_stride}")
        if batch_pack == 1:
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

    for stage in range(len(factors)):
        body.extend(
            _emit_stage_block(
                stage,
                factors,
                n,
                plan.lanes,
                lane_block,
                io_mode=io_mode,
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


def _build_direct_dft_kernel_source(
    n: int, direction: Literal["forward", "inverse"], dtype: str
) -> tuple[str, str, list[str]]:
    block = lane_block_for(n)
    acc_dtype = "tl.float64" if dtype == "complex128" else "tl.float32"
    suffix = _dtype_suffix(dtype)
    prefix = "direct_idft" if direction == "inverse" else "direct_dft"
    kernel_name = f"{prefix}_kernel_n{n}_{suffix}_b{block}"
    compensation_init = ""
    accumulation = """
                acc_r += xr * wr - xi * wi
                acc_i += xr * wi + xi * wr
    """
    if dtype == "complex128":
        compensation_init = f"""
            comp_r = tl.zeros(({block},), dtype={acc_dtype})
            comp_i = tl.zeros(({block},), dtype={acc_dtype})
        """
        accumulation = """
                term_r = xr * wr - xi * wi
                term_i = xr * wi + xi * wr
                corrected_r = term_r - comp_r
                corrected_i = term_i - comp_i
                next_r = acc_r + corrected_r
                next_i = acc_i + corrected_i
                comp_r = (next_r - acc_r) - corrected_r
                comp_i = (next_i - acc_i) - corrected_i
                acc_r = next_r
                acc_i = next_i
        """
    source = dedent(
        f"""
        @triton.jit
        def {kernel_name}(
            in_ptr,
            out_ptr,
            dft_r_ptr,
            dft_i_ptr,
            nbatch,
        ):
            pid_batch = tl.program_id(0)
            if pid_batch >= nbatch:
                return
            k = tl.arange(0, {block})
            mask = k < {n}
            acc_r = tl.zeros(({block},), dtype={acc_dtype})
            acc_i = tl.zeros(({block},), dtype={acc_dtype})
            {compensation_init}
            for j in tl.range(0, {n}):
                xr = tl.load(in_ptr + (pid_batch * {n} + j) * 2)
                xi = tl.load(in_ptr + (pid_batch * {n} + j) * 2 + 1)
                wr = tl.load(dft_r_ptr + k * {n} + j, mask=mask, other=0.0)
                wi = tl.load(dft_i_ptr + k * {n} + j, mask=mask, other=0.0)
                {accumulation}
            dst = out_ptr + (pid_batch * {n} + k) * 2
            tl.store(dst, acc_r, mask=mask)
            tl.store(dst + 1, acc_i, mask=mask)
        """
    )
    return (
        kernel_name,
        source,
        ["in_ptr", "out_ptr", "dft_r_ptr", "dft_i_ptr", "nbatch"],
    )


def _build_reshape_pack_kernel_source(
    n1: int, n2: int, dtype: str
) -> tuple[str, list[str], list[str]]:
    """Emit a pointwise (batch, R=n1, C=n2) -> (batch, C, R) transpose pack kernel.

    Used by generic nested four-step both for the row pre-transpose (stage 1) and the
    final natural-order pack (stage 5). Complex elements are stored interleaved
    (real, imag) with stride 2.
    """
    R = n1
    C = n2
    total = R * C
    block = 256
    zero = _zero_other(dtype)
    suffix = _dtype_suffix(dtype)
    kernel_name = f"_reshape_pack_kernel_n{R}_{C}_{suffix}"
    source = dedent(
        f"""
        @triton.jit
        def {kernel_name}(
            in_ptr,
            out_ptr,
            nbatch,
        ):
            pid_block = tl.program_id(0)
            pid_batch = tl.program_id(1)
            offsets = pid_block * {block} + tl.arange(0, {block})
            mask = offsets < {total}
            r = offsets // {C}
            c = offsets - r * {C}
            src = in_ptr + (pid_batch * {total} + offsets) * 2
            xr = tl.load(src, mask=mask, other={zero})
            xi = tl.load(src + 1, mask=mask, other={zero})
            dst_off = c * {R} + r
            dst = out_ptr + (pid_batch * {total} + dst_off) * 2
            tl.store(dst, xr, mask=mask)
            tl.store(dst + 1, xi, mask=mask)
        """
    )
    return kernel_name, source, ["in_ptr", "out_ptr", "nbatch"]


def _build_twiddle_reshape_pack_kernel_source(
    n1: int, n2: int, dtype: str
) -> tuple[str, list[str], list[str]]:
    """Emit a pointwise twiddle-multiply + (batch, R=n1, C=n2) -> (batch, C, R) pack kernel.

    The twiddle table is laid out as (R, C) row-major complex: twiddle[r * C + c].
    Used by generic nested four-step as the stage 3 fused twiddle+transpose pass.
    Caller-side mapping is R=n2, C=n1; the four-step twiddle table built by
    build_raw_four_step_twiddle uses (row=j2, col=k1) → (row * n1 + col) which
    matches twiddle[r * C + c] when (R, C) = (n2, n1).
    """
    R = n1
    C = n2
    total = R * C
    block = 256
    zero = _zero_other(dtype)
    suffix = _dtype_suffix(dtype)
    kernel_name = f"_twiddle_reshape_pack_kernel_n{R}_{C}_{suffix}"
    source = dedent(
        f"""
        @triton.jit
        def {kernel_name}(
            in_ptr,
            twiddle_ptr,
            out_ptr,
            nbatch,
        ):
            pid_block = tl.program_id(0)
            pid_batch = tl.program_id(1)
            offsets = pid_block * {block} + tl.arange(0, {block})
            mask = offsets < {total}
            r = offsets // {C}
            c = offsets - r * {C}
            src = in_ptr + (pid_batch * {total} + offsets) * 2
            xr = tl.load(src, mask=mask, other={zero})
            xi = tl.load(src + 1, mask=mask, other={zero})
            tw = twiddle_ptr + offsets * 2
            tr = tl.load(tw, mask=mask, other={zero})
            ti = tl.load(tw + 1, mask=mask, other={zero})
            yr, yi = _cmul(xr, xi, tr, ti)
            dst_off = c * {R} + r
            dst = out_ptr + (pid_batch * {total} + dst_off) * 2
            tl.store(dst, yr, mask=mask)
            tl.store(dst + 1, yi, mask=mask)
        """
    )
    return kernel_name, source, ["in_ptr", "twiddle_ptr", "out_ptr", "nbatch"]


def _build_real_to_complex_kernel_source(
    n: int, dtype: str
) -> tuple[str, list[str], list[str]]:
    block = 256
    zero = _zero_other(dtype)
    suffix = _dtype_suffix(dtype)
    kernel_name = f"_real_to_complex_kernel_n{n}_{suffix}"
    source = dedent(
        f"""
        @triton.jit
        def {kernel_name}(
            in_ptr,
            out_ptr,
            input_distance,
            nbatch,
        ):
            pid_block = tl.program_id(0)
            pid_batch = tl.program_id(1)
            offsets = pid_block * {block} + tl.arange(0, {block})
            mask = offsets < {n}
            xr = tl.load(in_ptr + pid_batch * input_distance + offsets, mask=mask, other={zero})
            dst = out_ptr + (pid_batch * {n} + offsets) * 2
            tl.store(dst, xr, mask=mask)
            tl.store(dst + 1, 0.0, mask=mask)
        """
    )
    return kernel_name, source, ["in_ptr", "out_ptr", "input_distance", "nbatch"]


def _build_r2c_half_pack_kernel_source(
    n: int, dtype: str
) -> tuple[str, list[str], list[str]]:
    half = n // 2 + 1
    block = 256
    zero = _zero_other(dtype)
    suffix = _dtype_suffix(dtype)
    kernel_name = f"_r2c_half_pack_kernel_n{n}_{suffix}"
    source = dedent(
        f"""
        @triton.jit
        def {kernel_name}(
            in_ptr,
            out_ptr,
            output_distance,
            nbatch,
        ):
            pid_block = tl.program_id(0)
            pid_batch = tl.program_id(1)
            offsets = pid_block * {block} + tl.arange(0, {block})
            mask = offsets < {half}
            src = in_ptr + (pid_batch * {n} + offsets) * 2
            xr = tl.load(src, mask=mask, other={zero})
            xi = tl.load(src + 1, mask=mask, other={zero})
            dst = out_ptr + (pid_batch * output_distance + offsets) * 2
            tl.store(dst, xr, mask=mask)
            tl.store(dst + 1, xi, mask=mask)
        """
    )
    return kernel_name, source, ["in_ptr", "out_ptr", "output_distance", "nbatch"]


def _build_compact_to_hermitian_full_kernel_source(
    n: int, dtype: str
) -> tuple[str, list[str], list[str]]:
    half = n // 2 + 1
    nyquist_guard = f" | (offsets == {n // 2})" if n % 2 == 0 else ""
    block = 256
    zero = _zero_other(dtype)
    suffix = _dtype_suffix(dtype)
    kernel_name = f"_compact_to_hermitian_full_kernel_n{n}_{suffix}"
    source = dedent(
        f"""
        @triton.jit
        def {kernel_name}(
            in_ptr,
            out_ptr,
            input_distance,
            nbatch,
        ):
            pid_block = tl.program_id(0)
            pid_batch = tl.program_id(1)
            offsets = pid_block * {block} + tl.arange(0, {block})
            mask = offsets < {n}
            src_k = tl.where(offsets < {half}, offsets, {n} - offsets)
            src = in_ptr + (pid_batch * input_distance + src_k) * 2
            xr = tl.load(src, mask=mask, other={zero})
            xi = tl.load(src + 1, mask=mask, other={zero})
            xi = tl.where(offsets < {half}, xi, -xi)
            xi = tl.where((offsets == 0){nyquist_guard}, 0.0, xi)
            dst = out_ptr + (pid_batch * {n} + offsets) * 2
            tl.store(dst, xr, mask=mask)
            tl.store(dst + 1, xi, mask=mask)
        """
    )
    return kernel_name, source, ["in_ptr", "out_ptr", "input_distance", "nbatch"]


def _build_complex_to_real_kernel_source(
    n: int, dtype: str
) -> tuple[str, list[str], list[str]]:
    block = 256
    zero = _zero_other(dtype)
    suffix = _dtype_suffix(dtype)
    kernel_name = f"_complex_to_real_kernel_n{n}_{suffix}"
    source = dedent(
        f"""
        @triton.jit
        def {kernel_name}(
            in_ptr,
            out_ptr,
            output_distance,
            nbatch,
        ):
            pid_block = tl.program_id(0)
            pid_batch = tl.program_id(1)
            offsets = pid_block * {block} + tl.arange(0, {block})
            mask = offsets < {n}
            src = in_ptr + (pid_batch * {n} + offsets) * 2
            xr = tl.load(src, mask=mask, other={zero})
            dst = out_ptr + pid_batch * output_distance + offsets
            tl.store(dst, xr, mask=mask)
        """
    )
    return kernel_name, source, ["in_ptr", "out_ptr", "output_distance", "nbatch"]


# NOTE: tile_size default (32) must match the constexpr tile_size in
# src/exec/raw_nodes.cpp CompiledRaw2DNode::execute().
def _build_tiled_transpose_kernel_source(
    n0: int, n1: int, dtype: str, tile_size: int = 32
) -> tuple[str, list[str], list[str]]:
    """Emit a tiled (batch, M=n0, N=n1) -> (batch, N, M) transpose kernel.

    Decomposes the matrix into tile_size x tile_size blocks across the grid — each
    program loads one block from global memory and writes it transposed.  No shared
    memory is used; the tiling is purely for grid parallelism.
    """
    zero = _zero_other(dtype)
    suffix = _dtype_suffix(dtype)
    total_complex = n0 * n1
    total_float = total_complex * 2  # interleaved complex: 2 floats per element
    kernel_name = f"_tiled_transpose_kernel_n{n0}_{n1}_{suffix}"
    source = dedent(
        f"""
        @triton.jit
        def {kernel_name}(
            in_ptr,
            out_ptr,
            nbatch,
        ):
            # Program IDs
            pid_tile_col = tl.program_id(0)
            pid_tile_row = tl.program_id(1)
            pid_batch = tl.program_id(2)

            # Tile offsets
            tile_row_start = pid_tile_row * {tile_size}
            tile_col_start = pid_tile_col * {tile_size}

            # Row and column offsets within tile
            row_offsets = tile_row_start + tl.arange(0, {tile_size})
            col_offsets = tile_col_start + tl.arange(0, {tile_size})

            # Mask for valid elements
            row_mask = row_offsets < {n0}
            col_mask = col_offsets < {n1}
            mask = row_mask[:, None] & col_mask[None, :]

            # Clamp offsets to valid range to avoid out-of-bounds addresses
            safe_row = tl.minimum(row_offsets, {n0 - 1})
            safe_col = tl.minimum(col_offsets, {n1 - 1})

            # Source element offsets in floats (row-major: batch * n0 * n1 * 2 + (row * n1 + col) * 2)
            src_elem_offsets = pid_batch * {total_float} + (safe_row[:, None] * {n1} + safe_col[None, :]) * 2

            # Load from source (complex elements are interleaved: real, imag)
            src_real = tl.load(in_ptr + src_elem_offsets, mask=mask, other={zero})
            src_imag = tl.load(in_ptr + src_elem_offsets + 1, mask=mask, other={zero})

            # Destination element offsets in floats (transposed: batch * n0 * n1 * 2 + (col * n0 + row) * 2)
            dst_elem_offsets = pid_batch * {total_float} + (safe_col[None, :] * {n0} + safe_row[:, None]) * 2

            # Store to destination (transposed)
            tl.store(out_ptr + dst_elem_offsets, src_real, mask=mask)
            tl.store(out_ptr + dst_elem_offsets + 1, src_imag, mask=mask)
        """
    )
    return kernel_name, source, ["in_ptr", "out_ptr", "nbatch"]


__all__ = [
    "LeafPlan",
    "_CODELET_DIR",
    "_FOUR_STEP_NUM_WARPS",
    "_FOUR_STEP_COL_INNER_PACK",
    "four_step_col_inner_pack_for",
    "four_step_row_inner_pack_for",
    "_FOUR_STEP_TILE_COLS",
    "_FOUR_STEP_TILE_ROWS",
    "_build_direct_dft_kernel_source",
    "_build_leaf_kernel_source",
    "_build_four_step_col_kernel_source",
    "_build_four_step_row_kernel_source",
    "_build_compact_to_hermitian_full_kernel_source",
    "_build_complex_to_real_kernel_source",
    "_build_r2c_half_pack_kernel_source",
    "_build_real_to_complex_kernel_source",
    "_build_reshape_pack_kernel_source",
    "_build_twiddle_reshape_pack_kernel_source",
    "_build_tiled_transpose_kernel_source",
    "_transpose_complex_kernel",
    "_twiddle_transpose_complex_kernel",
    "contiguous_batch_pack_for",
    "lane_block_for",
]
