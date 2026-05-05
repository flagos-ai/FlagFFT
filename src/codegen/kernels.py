from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import triton
import triton.language as tl
import triton.experimental.tle.language as tle

_MODULE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parents[1]
_CODELET_DIR = _PROJECT_ROOT / "src" / "codelet"
_FOUR_STEP_TILE_ROWS = 32
_FOUR_STEP_TILE_COLS = 32
_FOUR_STEP_NUM_WARPS = 4
SPECIALIZED_INLINE_CODELET_RADICES: set[int] = set()


@dataclass(frozen=True)
class LeafPlan:
    length: int
    factors: tuple[int, ...]
    remainder: int
    lanes: int
    num_warps: int
    generic_radices: tuple[int, ...]
    smem_size: int
    kind: Literal["ct_leaf"] = field(default="ct_leaf", init=False)


def lane_block_for(lanes: int) -> int:
    if lanes <= 1:
        return 1
    value = 1
    while value < lanes:
        value <<= 1
    return value


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
    src_offsets = src_base + row_offsets[:, None] * src_row_stride + col_offsets[None, :] * src_col_stride
    src_real = tl.load(src_offsets, mask=mask, other=0.0)
    src_imag = tl.load(src_offsets + 1, mask=mask, other=0.0)

    dst_base = dst_ptr + pid_batch * dst_batch_stride
    dst_offsets = dst_base + col_offsets[None, :] * dst_row_stride + row_offsets[:, None] * dst_col_stride
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
    src_offsets = src_base + row_offsets[:, None] * src_row_stride + col_offsets[None, :] * src_col_stride
    src_real = tl.load(src_offsets, mask=mask, other=0.0)
    src_imag = tl.load(src_offsets + 1, mask=mask, other=0.0)

    tw_offsets = twiddle_ptr + row_offsets[:, None] * twiddle_row_stride + col_offsets[None, :] * twiddle_col_stride
    tw_real = tl.load(tw_offsets, mask=mask, other=0.0)
    tw_imag = tl.load(tw_offsets + 1, mask=mask, other=0.0)
    out_real, out_imag = _cmul(src_real, src_imag, tw_real, tw_imag)

    dst_base = dst_ptr + pid_batch * dst_batch_stride
    dst_offsets = dst_base + col_offsets[None, :] * dst_row_stride + row_offsets[:, None] * dst_col_stride
    tl.store(dst_offsets, out_real, mask=mask)
    tl.store(dst_offsets + 1, out_imag, mask=mask)



def _fmt_const(value: float) -> str:
    if abs(value) < 1e-8:
        value = 0.0
    elif abs(value - 1.0) < 1e-8:
        value = 1.0
    elif abs(value + 1.0) < 1e-8:
        value = -1.0
    return repr(float(value))


def _emit_inline_constant_codelet(indent: str, radix: int, lane_block: int) -> list[str]:
    lines: list[str] = []
    for kout in range(radix):
        lines.append(f"{indent}acc_r_{kout} = tl.zeros(({lane_block},), dtype=tl.float32)")
        lines.append(f"{indent}acc_i_{kout} = tl.zeros(({lane_block},), dtype=tl.float32)")

    for kout in range(radix):
        for nin in range(radix):
            angle = -2.0 * math.pi * kout * nin / float(radix)
            wr = _fmt_const(math.cos(angle))
            wi = _fmt_const(math.sin(angle))
            lines.append(f"{indent}pr, pi = _cmul(r{nin}, i{nin}, {wr}, {wi})")
            lines.append(f"{indent}acc_r_{kout} += pr")
            lines.append(f"{indent}acc_i_{kout} += pi")

    for kout in range(radix):
        lines.append(f"{indent}r{kout} = acc_r_{kout}")
        lines.append(f"{indent}i{kout} = acc_i_{kout}")
    return lines


def _emit_table_codelet(indent: str, radix: int, lane_block: int) -> list[str]:
    lines: list[str] = []
    for kout in range(radix):
        lines.append(f"{indent}acc_r_{kout} = tl.zeros(({lane_block},), dtype=tl.float32)")
        lines.append(f"{indent}acc_i_{kout} = tl.zeros(({lane_block},), dtype=tl.float32)")

    for kout in range(radix):
        for nin in range(radix):
            lines.append(f"{indent}wr = tl.load(dft{radix}_r_ptr + {kout * radix + nin})")
            lines.append(f"{indent}wi = tl.load(dft{radix}_i_ptr + {kout * radix + nin})")
            lines.append(f"{indent}pr, pi = _cmul(r{nin}, i{nin}, wr, wi)")
            lines.append(f"{indent}acc_r_{kout} += pr")
            lines.append(f"{indent}acc_i_{kout} += pi")

    for kout in range(radix):
        lines.append(f"{indent}r{kout} = acc_r_{kout}")
        lines.append(f"{indent}i{kout} = acc_i_{kout}")
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


def _emit_input_index(indent: str, out_var: str, factors: tuple[int, ...], digit: int) -> list[str]:
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


def _emit_output_index(indent: str, out_var: str, factors: tuple[int, ...], digit: int) -> list[str]:
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
        lines.append(f"{indent}digit_route{stage}_{axis} = rem_route{stage} % {factors[axis]}")
        lines.append(f"{indent}rem_route{stage} = rem_route{stage} // {factors[axis]}")
        if stride == 1:
            lines.append(f"{indent}route_codelet_base{stage} += digit_route{stage}_{axis}")
        else:
            lines.append(f"{indent}route_codelet_base{stage} += digit_route{stage}_{axis} * {stride}")
        stride *= factors[axis]

    stride *= factors[stage]

    for axis in range(len(factors) - 1, stage, -1):
        lines.append(f"{indent}digit_route{stage}_{axis} = rem_route{stage} % {factors[axis]}")
        lines.append(f"{indent}rem_route{stage} = rem_route{stage} // {factors[axis]}")
        if axis == stage + 1:
            lines.append(f"{indent}next_digit{stage} = digit_route{stage}_{axis}")
        else:
            if stride == 1:
                lines.append(f"{indent}route_codelet_base{stage} += digit_route{stage}_{axis}")
            else:
                lines.append(f"{indent}route_codelet_base{stage} += digit_route{stage}_{axis} * {stride}")
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
        lines = [f"{indent}next_codelet{stage}_{suffix} = route_codelet_base{stage} + {offset}"]
    lines.append(
        f"{indent}{out_var} = (next_codelet{stage}_{suffix} % {lanes}) + "
        f"{lanes} * ((next_codelet{stage}_{suffix} // {lanes}) * {radix_next} + next_digit{stage})"
    )
    return lines


def _emit_stage_block(
    stage: int, factors: tuple[int, ...], n: int, lanes: int, lane_block: int
) -> list[str]:
    radix = factors[stage]
    groups = n // (lanes * radix)
    is_last = stage == len(factors) - 1
    source_buffer = None if stage == 0 else ("smem_b" if stage % 2 == 1 else "smem_a")
    dest_buffer = None if is_last else ("smem_b" if stage % 2 == 0 else "smem_a")

    lines = [f"    for group_{stage} in tl.range(0, {groups}):"]
    indent = "        "

    for j in range(radix):
        lines.append(
            f"{indent}phys{j} = tl.where(lane_mask, lane + {lanes} * (group_{stage} * {radix} + {j}), 0)"
        )
    if stage == 0:
        lines.extend(_emit_input_base(indent, factors, lanes, f"group_{stage}"))
    if is_last:
        lines.extend(_emit_output_base(indent, factors, lanes, f"group_{stage}"))
    else:
        lines.extend(_emit_route_base(indent, stage, factors, lanes, f"group_{stage}"))

    for j in range(radix):
        if stage == 0:
            lines.extend(_emit_input_index(indent, f"in{j}", factors, j))
            lines.append(
                f"{indent}r{j} = tl.load(in_ptr + (batch_base + in{j}) * 2, mask=lane_mask, other=0.0)"
            )
            lines.append(
                f"{indent}i{j} = tl.load(in_ptr + (batch_base + in{j}) * 2 + 1, mask=lane_mask, other=0.0)"
            )
        else:
            lines.append(
                f"{indent}r{j} = tl.load(tle.gpu.local_ptr({source_buffer}_r, (phys{j},)), mask=lane_mask, other=0.0)"
            )
            lines.append(
                f"{indent}i{j} = tl.load(tle.gpu.local_ptr({source_buffer}_i, (phys{j},)), mask=lane_mask, other=0.0)"
            )
            lines.append(f"{indent}twr = tl.load(tw{stage}_r_ptr + phys{j}, mask=lane_mask, other=0.0)")
            lines.append(f"{indent}twi = tl.load(tw{stage}_i_ptr + phys{j}, mask=lane_mask, other=0.0)")
            lines.append(f"{indent}r{j}, i{j} = _cmul(r{j}, i{j}, twr, twi)")

    if radix == 2:
        lines.append(f"{indent}(r0, r1, i0, i1) = _fwd_rad2_b1(r0, r1, i0, i1)")
    elif radix == 3:
        lines.append(
            f"{indent}(r0, r1, r2, i0, i1, i2) = "
            "_fwd_rad3_b1(r0, r1, r2, i0, i1, i2)"
        )
    elif radix == 4:
        lines.append(
            f"{indent}(r0, r1, r2, r3, i0, i1, i2, i3) = _fwd_rad4_b1(r0, r1, r2, r3, i0, i1, i2, i3)"
        )
    elif radix == 5:
        lines.append(
            f"{indent}(r0, r1, r2, r3, r4, i0, i1, i2, i3, i4) = "
            "_fwd_rad5_b1(r0, r1, r2, r3, r4, i0, i1, i2, i3, i4)"
        )
    elif radix == 6:
        lines.append(
            f"{indent}(r0, r1, r2, r3, r4, r5, i0, i1, i2, i3, i4, i5) = "
            "_fwd_rad6_b1(r0, r1, r2, r3, r4, r5, i0, i1, i2, i3, i4, i5)"
        )
    elif radix == 7:
        lines.append(
            f"{indent}(r0, r1, r2, r3, r4, r5, r6, i0, i1, i2, i3, i4, i5, i6) = "
            "_fwd_rad7_b1(r0, r1, r2, r3, r4, r5, r6, i0, i1, i2, i3, i4, i5, i6)"
        )
    elif radix == 8:
        lines.append(
            f"{indent}(r0, r1, r2, r3, r4, r5, r6, r7, i0, i1, i2, i3, i4, i5, i6, i7) = "
            "_fwd_rad8_b1(r0, r1, r2, r3, r4, r5, r6, r7, i0, i1, i2, i3, i4, i5, i6, i7)"
        )
    elif radix == 9:
        lines.append(
            f"{indent}(r0, r1, r2, r3, r4, r5, r6, r7, r8, i0, i1, i2, i3, i4, i5, i6, i7, i8) = "
            "_fwd_rad9_b1(r0, r1, r2, r3, r4, r5, r6, r7, r8, i0, i1, i2, i3, i4, i5, i6, i7, i8)"
        )
    elif radix == 11:
        lines.append(f"{indent}(")
        for idx in range(11):
            lines.append(f"{indent}    r{idx},")
        for idx in range(11):
            lines.append(f"{indent}    i{idx},")
        lines.append(
            f"{indent}) = _fwd_rad11_b1("
            "r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, "
            "i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10)"
        )
    elif radix == 13:
        lines.append(f"{indent}(")
        for idx in range(13):
            lines.append(f"{indent}    r{idx},")
        for idx in range(13):
            lines.append(f"{indent}    i{idx},")
        lines.append(
            f"{indent}) = _fwd_rad13_b1("
            "r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, "
            "i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12)"
        )
    elif radix == 16:
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
    elif radix in SPECIALIZED_INLINE_CODELET_RADICES:
        lines.extend(_emit_inline_constant_codelet(indent, radix, lane_block))
    else:
        lines.extend(_emit_table_codelet(indent, radix, lane_block))

    for j in range(radix):
        if is_last:
            lines.extend(_emit_output_index(indent, f"out_idx{j}", factors, j))
            lines.append(
                f"{indent}tl.store(out_ptr + (batch_base + out_idx{j}) * 2, r{j}, mask=lane_mask)"
            )
            lines.append(
                f"{indent}tl.store(out_ptr + (batch_base + out_idx{j}) * 2 + 1, i{j}, mask=lane_mask)"
            )
        else:
            lines.extend(_emit_route_index(indent, f"dst{j}", stage, factors, lanes, j))
            lines.append(
                f"{indent}tl.store(tle.gpu.local_ptr({dest_buffer}_r, (dst{j},)), r{j}, mask=lane_mask)"
            )
            lines.append(
                f"{indent}tl.store(tle.gpu.local_ptr({dest_buffer}_i, (dst{j},)), i{j}, mask=lane_mask)"
            )

    if not is_last:
        lines.append("    tl.debug_barrier()")
    return lines


def _build_leaf_kernel_source(plan: LeafPlan) -> tuple[str, str]:
    factors = plan.factors
    n = plan.length
    smem_n = plan.smem_size
    generic_radices = plan.generic_radices
    lane_block = lane_block_for(plan.lanes)
    params = ["in_ptr", "out_ptr"]
    for stage in range(1, len(factors)):
        params.append(f"tw{stage}_r_ptr")
        params.append(f"tw{stage}_i_ptr")
    for radix in generic_radices:
        params.append(f"dft{radix}_r_ptr")
        params.append(f"dft{radix}_i_ptr")
    params.append("nbatch")

    kernel_name = "fft_kernel_" + "_".join(str(x) for x in factors) + f"_l{plan.lanes}_b{lane_block}"
    body: list[str] = [
        "@triton.jit",
        f"def {kernel_name}(",
    ]
    for idx, param in enumerate(params):
        suffix = "," if idx < len(params) - 1 else ""
        body.append(f"    {param}{suffix}")
    body.append("):")
    body.append("    pid = tl.program_id(0)")
    body.append("    if pid >= nbatch:")
    body.append("        return")
    body.append(f"    lane = tl.arange(0, {lane_block})")
    body.append(f"    lane_mask = lane < {plan.lanes}")
    body.append(f"    batch_base = pid * {n}")

    if len(factors) > 1:
        body.append(
            f"    smem_a_r = tle.gpu.alloc([{smem_n}], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)"
        )
        body.append(
            f"    smem_a_i = tle.gpu.alloc([{smem_n}], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)"
        )
        body.append(
            f"    smem_b_r = tle.gpu.alloc([{smem_n}], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)"
        )
        body.append(
            f"    smem_b_i = tle.gpu.alloc([{smem_n}], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)"
        )

    for stage in range(len(factors)):
        body.extend(_emit_stage_block(stage, factors, n, plan.lanes, lane_block))

    return kernel_name, "\n".join(body)

__all__ = [
    "LeafPlan",
    "_CODELET_DIR",
    "_FOUR_STEP_NUM_WARPS",
    "_FOUR_STEP_TILE_COLS",
    "_FOUR_STEP_TILE_ROWS",
    "_build_leaf_kernel_source",
    "_transpose_complex_kernel",
    "_twiddle_transpose_complex_kernel",
    "lane_block_for",
]
