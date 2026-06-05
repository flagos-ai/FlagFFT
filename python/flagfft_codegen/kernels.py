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
_FOUR_STEP_COL_INNER_PACK_MIN_N1 = 128
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


LeafIoMode = Literal["contiguous", "four_step_row", "four_step_col"]


def lane_block_for(lanes: int) -> int:
    if lanes <= 1:
        return 1
    value = 1
    while value < lanes:
        value <<= 1
    return value


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
    if dtype in ("complex128", "float64") and n2 > 1024:
        return 1
    return _FOUR_STEP_COL_INNER_PACK


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
    tl_dtype = _tl_real_dtype(dtype)
    for kout in range(radix):
        lines.append(
            f"{indent}acc_r_{kout} = tl.zeros(({lane_block},), dtype={tl_dtype})"
        )
        lines.append(
            f"{indent}acc_i_{kout} = tl.zeros(({lane_block},), dtype={tl_dtype})"
        )

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
    tl_dtype = _tl_real_dtype(dtype)
    for kout in range(radix):
        lines.append(
            f"{indent}acc_r_{kout} = tl.zeros(({lane_block},), dtype={tl_dtype})"
        )
        lines.append(
            f"{indent}acc_i_{kout} = tl.zeros(({lane_block},), dtype={tl_dtype})"
        )

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
    indent: str, radix: int, direction: Literal["forward", "inverse"]
) -> list[str]:
    lines: list[str] = []
    if direction == "inverse":
        for idx in range(radix):
            lines.append(f"{indent}i{idx} = -i{idx}")
    lines.append(f"{indent}(")
    for idx in range(radix):
        lines.append(f"{indent}    r{idx},")
    for idx in range(radix):
        lines.append(f"{indent}    i{idx},")
    args = ", ".join(
        [*(f"r{idx}" for idx in range(radix)), *(f"i{idx}" for idx in range(radix))]
    )
    lines.append(f"{indent}) = _fwd_rad{radix}_b1({args})")
    if direction == "inverse":
        for idx in range(radix):
            lines.append(f"{indent}i{idx} = -i{idx}")
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
    direction: Literal["forward", "inverse"] = "forward",
    dtype: str = "complex64",
) -> list[str]:
    radix = factors[stage]
    groups = n // (lanes * radix)
    is_last = stage == len(factors) - 1
    source_buffer = None if stage == 0 else ("smem_b" if stage % 2 == 1 else "smem_a")
    dest_buffer = None if is_last else ("smem_b" if stage % 2 == 0 else "smem_a")
    zero = "0.0"

    lines = [f"    for group_{stage} in tl.range(0, {groups}):"]
    indent = "        "

    for j in range(radix):
        lines.append(
            f"{indent}logical_phys{j} = tl.where(lane_mask, lane + {lanes} * (group_{stage} * {radix} + {j}), 0)"
        )
        if smem_pack > 1:
            lines.append(f"{indent}phys{j} = logical_phys{j} + smem_offset")
        else:
            lines.append(f"{indent}phys{j} = logical_phys{j}")
    if stage == 0:
        lines.extend(_emit_input_base(indent, factors, lanes, f"group_{stage}"))
    if is_last:
        lines.extend(_emit_output_base(indent, factors, lanes, f"group_{stage}"))
    else:
        lines.extend(_emit_route_base(indent, stage, factors, lanes, f"group_{stage}"))

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
            else:
                lines.append(
                    f"{indent}src_idx{j} = in{j} * {four_step_n1} + four_step_inner"
                )
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
            lines.append(
                f"{indent}r{j} = tl.load(tle.gpu.local_ptr({source_buffer}_r, (phys{j},)), "
                f"mask=lane_mask, other={zero})"
            )
            lines.append(
                f"{indent}i{j} = tl.load(tle.gpu.local_ptr({source_buffer}_i, (phys{j},)), "
                f"mask=lane_mask, other={zero})"
            )
            lines.append(
                f"{indent}twr = tl.load(tw{stage}_r_ptr + logical_phys{j}, mask=lane_mask, other={zero})"
            )
            lines.append(
                f"{indent}twi = tl.load(tw{stage}_i_ptr + logical_phys{j}, mask=lane_mask, other={zero})"
            )
            lines.append(f"{indent}r{j}, i{j} = _cmul(r{j}, i{j}, twr, twi)")

    if radix == 16:
        if direction == "inverse":
            for idx in range(16):
                lines.append(f"{indent}i{idx} = -i{idx}")
        lines.append(f"{indent}(")
        for idx in range(16):
            lines.append(f"{indent}    r{idx},")
        for idx in range(16):
            lines.append(f"{indent}    i{idx},")
        lines.append(
            f"{indent}) = _fwd_rad16_b1("
            "r0, r8, r4, r12, r2, r10, r6, r14, r1, r9, r5, r13, r3, r11, r7, r15, "
            "i0, i8, i4, i12, i2, i10, i6, i14, i1, i9, i5, i13, i3, i11, i7, i15)"
        )
        if direction == "inverse":
            for idx in range(16):
                lines.append(f"{indent}i{idx} = -i{idx}")
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
            elif io_mode == "four_step_row":
                lines.append(
                    f"{indent}dst_idx{j} = four_step_inner * {four_step_n1} + out_idx{j}"
                )
                lines.append(
                    f"{indent}tl.store(out_ptr + (four_step_batch_base + dst_idx{j}) * 2, r{j}, mask=lane_mask)"
                )
                lines.append(
                    f"{indent}tl.store(out_ptr + (four_step_batch_base + dst_idx{j}) * 2 + 1, i{j}, mask=lane_mask)"
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
            lines.extend(_emit_route_index(indent, f"dst{j}", stage, factors, lanes, j))
            store_index = f"dst{j}"
            if smem_pack > 1:
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
    params.append("nbatch")
    return params


def _build_leaf_kernel_source_for_io(
    plan: LeafPlan,
    *,
    io_mode: LeafIoMode,
    four_step_n1: int = 0,
    four_step_n2: int = 0,
) -> tuple[str, str]:
    factors = plan.factors
    n = plan.length
    smem_n = plan.smem_size
    lane_block = lane_block_for(plan.lanes)
    batch_pack = contiguous_batch_pack_for(plan) if io_mode == "contiguous" else 1
    inner_pack = (
        four_step_col_inner_pack_for(four_step_n1, four_step_n2, plan.dtype)
        if io_mode == "four_step_col"
        else 1
    )
    smem_pack = max(batch_pack, inner_pack)
    vector_block = lane_block * smem_pack
    smem_slot_stride = plan.smem_size + 1 if batch_pack >= 4 else plan.smem_size
    smem_n = lane_block_for(smem_slot_stride * smem_pack)
    params = _leaf_kernel_params(
        plan, include_four_step_twiddle=io_mode == "four_step_col"
    )

    suffix = "_".join(str(x) for x in factors)
    if io_mode == "contiguous":
        kernel_prefix = "ifft" if plan.direction == "inverse" else "fft"
        kernel_name = f"{kernel_prefix}_kernel_{suffix}_l{plan.lanes}_b{lane_block}"
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
    if io_mode == "contiguous":
        body.append("    pid = tl.program_id(0)")
        body.append(f"    batch_id = pid * {batch_pack}")
        body.append("    if batch_id >= nbatch:")
        body.append("        return")
    else:
        if io_mode == "four_step_col" and inner_pack > 1:
            body.append(f"    four_step_inner_base = tl.program_id(0) * {inner_pack}")
        else:
            body.append("    four_step_inner = tl.program_id(0)")
        body.append("    four_step_batch = tl.program_id(1)")
        body.append("    if four_step_batch >= nbatch:")
        body.append("        return")
    body.append(f"    lane_vec = tl.arange(0, {vector_block})")
    if io_mode == "contiguous":
        if batch_pack == 1:
            body.append("    lane = lane_vec")
            body.append(f"    lane_mask = lane < {plan.lanes}")
            body.append(f"    batch_base = batch_id * {n}")
        else:
            body.append(f"    batch_slot = lane_vec // {lane_block}")
            body.append(f"    lane = lane_vec - batch_slot * {lane_block}")
            body.append("    current_batch = batch_id + batch_slot")
            body.append(
                f"    lane_mask = (lane < {plan.lanes}) & (current_batch < nbatch)"
            )
            body.append(f"    batch_base = current_batch * {n}")
            body.append(f"    smem_offset = batch_slot * {smem_slot_stride}")
    else:
        if io_mode == "four_step_col" and inner_pack > 1:
            body.append(f"    inner_slot = lane_vec % {inner_pack}")
            body.append(f"    lane = lane_vec // {inner_pack}")
            body.append("    four_step_inner = four_step_inner_base + inner_slot")
            body.append(
                f"    lane_mask = (lane < {plan.lanes}) & (four_step_inner < {four_step_n1})"
            )
            body.append(f"    smem_offset = inner_slot * {smem_slot_stride}")
        else:
            body.append("    lane = lane_vec")
            body.append(f"    lane_mask = lane < {plan.lanes}")
        body.append(
            f"    four_step_batch_base = four_step_batch * {four_step_n1 * four_step_n2}"
        )

    if len(factors) > 1:
        tl_dtype = _tl_real_dtype(plan.dtype)
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
                direction=plan.direction,
                dtype=plan.dtype,
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
    "_FOUR_STEP_TILE_COLS",
    "_FOUR_STEP_TILE_ROWS",
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
