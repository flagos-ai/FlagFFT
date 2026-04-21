from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

from .plan import (
    FFTDecompositionSpec,
    FFTPlan,
    FourStepPlan,
    LeafPlan,
    SPECIALIZED_INLINE_CODELET_RADICES,
    build_fft_plan,
    clear_plan_cache,
    get_fft_plan,
    lane_block_for,
    unique_leaf_plans,
)

_MODULE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parent if _MODULE_DIR.name == "src" else _MODULE_DIR
_CODELET_DIR = _MODULE_DIR / "codelet"
if not _CODELET_DIR.is_dir():
    _CODELET_DIR = _PROJECT_ROOT / "codelet"
_CACHE_DIR = _PROJECT_ROOT / ".tle_cache" / "fft_mixed_radix_tle_generated"

_TABLE_CACHE: dict[tuple[torch.device, tuple[int, ...], int], dict[str, torch.Tensor]] = {}
_KERNEL_CACHE: dict[tuple[tuple[int, ...], int], object] = {}
_TWIDDLE_CACHE: dict[tuple[torch.device, int, int], torch.Tensor] = {}


def _decode_stage_codelet(codelet: int, radices: tuple[int, ...], stage: int) -> tuple[list[int], list[int]]:
    prev_freq: list[int] = []
    rem = codelet
    for axis in range(stage):
        prev_freq.append(rem % radices[axis])
        rem //= radices[axis]

    future_time = [0] * len(radices)
    for axis in range(len(radices) - 1, stage, -1):
        future_time[axis] = rem % radices[axis]
        rem //= radices[axis]

    if rem != 0:
        raise ValueError(f"stage-{stage} codelet decode overflow: rem={rem}")
    return prev_freq, future_time


def _encode_stage_codelet(
    prev_freq: list[int], future_time: list[int], radices: tuple[int, ...], stage: int
) -> int:
    codelet = 0
    stride = 1
    for axis in range(stage):
        codelet += prev_freq[axis] * stride
        stride *= radices[axis]
    for axis in range(len(radices) - 1, stage, -1):
        codelet += future_time[axis] * stride
        stride *= radices[axis]
    return codelet


def _mixed_radix_value(digits: list[int], radices: tuple[int, ...]) -> int:
    value = 0
    stride = 1
    for axis, digit in enumerate(digits):
        value += digit * stride
        stride *= radices[axis]
    return value


def _time_major_value(digits: list[int], radices: tuple[int, ...]) -> int:
    value = 0
    stride = 1
    for axis in range(len(radices) - 1, -1, -1):
        value += digits[axis] * stride
        stride *= radices[axis]
    return value


def _build_stage_route_table(radices: tuple[int, ...], stage: int, lanes: int) -> torch.Tensor:
    n = math.prod(radices)
    radix_cur = radices[stage]
    radix_next = radices[stage + 1]
    codelets_cur = n // radix_cur
    codelets_next = n // radix_next
    groups_cur = codelets_cur // lanes
    groups_next = codelets_next // lanes
    elems_per_lane = n // lanes

    if codelets_cur % lanes != 0 or codelets_next % lanes != 0:
        raise ValueError(f"lanes={lanes} must divide both stage codelet counts")
    if groups_cur * radix_cur != elems_per_lane or groups_next * radix_next != elems_per_lane:
        raise ValueError("fixed-lane invariant violated")

    route = torch.empty(n, dtype=torch.int32)
    for lane in range(lanes):
        for elem in range(elems_per_lane):
            group_cur = elem // radix_cur
            k_cur = elem % radix_cur
            codelet_cur = lane + lanes * group_cur

            prev_freq, future_time = _decode_stage_codelet(codelet_cur, radices, stage)
            prev_freq = list(prev_freq)
            prev_freq.append(k_cur)

            next_digit = future_time[stage + 1]
            codelet_next = _encode_stage_codelet(prev_freq, future_time, radices, stage + 1)
            lane_next = codelet_next % lanes
            group_next = codelet_next // lanes
            elem_next = group_next * radix_next + next_digit

            src_phys = lane + lanes * elem
            dst_phys = lane_next + lanes * elem_next
            route[src_phys] = dst_phys
    return route


def _build_input_map(radices: tuple[int, ...], lanes: int) -> torch.Tensor:
    n = math.prod(radices)
    elems_per_lane = n // lanes
    radix0 = radices[0]
    mapping = torch.empty(n, dtype=torch.int32)
    for lane in range(lanes):
        for elem in range(elems_per_lane):
            group = elem // radix0
            n0 = elem % radix0
            codelet = lane + lanes * group
            _, future_time = _decode_stage_codelet(codelet, radices, 0)
            time_digits = [n0] + [future_time[axis] for axis in range(1, len(radices))]
            mapping[lane + lanes * elem] = _time_major_value(time_digits, radices)
    return mapping


def _build_output_map(radices: tuple[int, ...], lanes: int) -> torch.Tensor:
    n = math.prod(radices)
    elems_per_lane = n // lanes
    last_stage = len(radices) - 1
    radix_last = radices[last_stage]
    mapping = torch.empty(n, dtype=torch.int32)
    for lane in range(lanes):
        for elem in range(elems_per_lane):
            group = elem // radix_last
            k_last = elem % radix_last
            codelet = lane + lanes * group
            prev_freq, _ = _decode_stage_codelet(codelet, radices, last_stage)
            freq_digits = list(prev_freq) + [k_last]
            mapping[lane + lanes * elem] = _mixed_radix_value(freq_digits, radices)
    return mapping


def _build_stage_twiddles(radices: tuple[int, ...], stage: int, lanes: int) -> tuple[torch.Tensor, torch.Tensor]:
    n = math.prod(radices)
    elems_per_lane = n // lanes
    radix = radices[stage]
    denom = math.prod(radices[: stage + 1])
    tw_r = torch.empty(n, dtype=torch.float32)
    tw_i = torch.empty(n, dtype=torch.float32)
    for lane in range(lanes):
        for elem in range(elems_per_lane):
            group = elem // radix
            digit = elem % radix
            codelet = lane + lanes * group
            prev_freq, _ = _decode_stage_codelet(codelet, radices, stage)
            prefix = _mixed_radix_value(prev_freq, radices[:stage])
            angle = -2.0 * math.pi * prefix * digit / float(denom)
            tw_r[lane + lanes * elem] = math.cos(angle)
            tw_i[lane + lanes * elem] = math.sin(angle)
    return tw_r, tw_i


def _build_dft_matrix(radix: int) -> tuple[torch.Tensor, torch.Tensor]:
    n = torch.arange(radix, dtype=torch.float32)
    k = n[:, None]
    angle = -2.0 * math.pi * (k * n[None, :]) / float(radix)
    return torch.cos(angle).reshape(-1), torch.sin(angle).reshape(-1)


def _get_leaf_tables(device: torch.device, plan: LeafPlan) -> dict[str, torch.Tensor]:
    key = (device, plan.factors, plan.lanes)
    cached = _TABLE_CACHE.get(key)
    if cached is not None:
        return cached

    tables: dict[str, torch.Tensor] = {
        "in_map": _build_input_map(plan.factors, plan.lanes).to(device),
        "out_map": _build_output_map(plan.factors, plan.lanes).to(device),
    }

    for stage in range(len(plan.factors) - 1):
        tables[f"route{stage}"] = _build_stage_route_table(plan.factors, stage, plan.lanes).to(device)

    for stage in range(1, len(plan.factors)):
        tw_r, tw_i = _build_stage_twiddles(plan.factors, stage, plan.lanes)
        tables[f"tw{stage}_r"] = tw_r.to(device)
        tables[f"tw{stage}_i"] = tw_i.to(device)

    for radix in plan.generic_radices:
        dft_r, dft_i = _build_dft_matrix(radix)
        tables[f"dft{radix}_r"] = dft_r.to(device)
        tables[f"dft{radix}_i"] = dft_i.to(device)

    _TABLE_CACHE[key] = tables
    return tables


@triton.jit
def _cmul(ar, ai, br, bi):
    return ar * br - ai * bi, ai * br + ar * bi


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

    for j in range(radix):
        if stage == 0:
            lines.append(f"{indent}in{j} = tl.load(in_map_ptr + phys{j}, mask=lane_mask, other=0)")
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
            lines.append(f"{indent}out_idx{j} = tl.load(out_map_ptr + phys{j}, mask=lane_mask, other=0)")
            lines.append(
                f"{indent}tl.store(out_ptr + (batch_base + out_idx{j}) * 2, r{j}, mask=lane_mask)"
            )
            lines.append(
                f"{indent}tl.store(out_ptr + (batch_base + out_idx{j}) * 2 + 1, i{j}, mask=lane_mask)"
            )
        else:
            lines.append(f"{indent}dst{j} = tl.load(route{stage}_ptr + phys{j}, mask=lane_mask, other=0)")
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
    params = ["in_ptr", "out_ptr", "in_map_ptr", "out_map_ptr"]
    for stage in range(len(factors) - 1):
        params.append(f"route{stage}_ptr")
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


def _get_leaf_kernel(plan: LeafPlan):
    key = (plan.factors, plan.lanes)
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached

    if not _CODELET_DIR.is_dir():
        raise RuntimeError(f"missing codelet directory: {_CODELET_DIR}")

    helpers = ""
    utils_path = _CODELET_DIR / "utils.py"
    if utils_path.exists():
        helpers += utils_path.read_text() + "\n\n"

    for codelet_file in sorted(_CODELET_DIR.glob("*.py")):
        if codelet_file.name not in {utils_path.name, Path(__file__).name}:
            helpers += codelet_file.read_text() + "\n\n"

    kernel_name, source = _build_leaf_kernel_source(plan)
    lane_block = lane_block_for(plan.lanes)
    module_name = (
        "fft_mixed_radix_gen_" + "_".join(str(x) for x in plan.factors) + f"_l{plan.lanes}_b{lane_block}"
    )
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    module_path = _CACHE_DIR / f"{module_name}.py"
    module_path.write_text(helpers + "\n\n" + source + "\n")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load generated module {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kernel = getattr(module, kernel_name)
    _KERNEL_CACHE[key] = kernel
    return kernel


def _execute_leaf_plan(x: torch.Tensor, plan: LeafPlan) -> torch.Tensor:
    batch = x.numel() // plan.length
    x_contig = x.contiguous().reshape(batch, plan.length)
    out = torch.empty_like(x_contig)
    tables = _get_leaf_tables(x.device, plan)
    kernel = _get_leaf_kernel(plan)

    args = [
        x_contig.view(torch.float32),
        out.view(torch.float32),
        tables["in_map"],
        tables["out_map"],
    ]
    for stage in range(len(plan.factors) - 1):
        args.append(tables[f"route{stage}"])
    for stage in range(1, len(plan.factors)):
        args.append(tables[f"tw{stage}_r"])
        args.append(tables[f"tw{stage}_i"])
    for radix in plan.generic_radices:
        args.append(tables[f"dft{radix}_r"])
        args.append(tables[f"dft{radix}_i"])
    args.append(batch)

    kernel[(batch,)](*args, num_warps=plan.num_warps, num_stages=1)
    return out


def _get_four_step_twiddle(device: torch.device, n1: int, n2: int) -> torch.Tensor:
    key = (device, n1, n2)
    cached = _TWIDDLE_CACHE.get(key)
    if cached is not None:
        return cached

    row = torch.arange(n2, device=device, dtype=torch.float32)
    col = torch.arange(n1, device=device, dtype=torch.float32)
    angle = -2.0 * math.pi * row[:, None] * col[None, :] / float(n1 * n2)
    twiddle = torch.complex(torch.cos(angle), torch.sin(angle)).to(torch.complex64)
    _TWIDDLE_CACHE[key] = twiddle
    return twiddle


def _execute_four_step_plan(x: torch.Tensor, plan: FourStepPlan) -> torch.Tensor:
    batch = x.numel() // plan.length
    x_contig = x.contiguous().reshape(batch, plan.length)

    stage0 = x_contig.reshape(batch, plan.n1, plan.n2).transpose(1, 2).contiguous()
    stage1 = _execute_plan(stage0.reshape(batch * plan.n2, plan.n1), plan.row_plan)
    stage1 = stage1.reshape(batch, plan.n2, plan.n1)

    twiddle = _get_four_step_twiddle(x.device, plan.n1, plan.n2)
    stage2 = (stage1 * twiddle[None, :, :]).transpose(1, 2).contiguous()

    stage3 = _execute_plan(stage2.reshape(batch * plan.n1, plan.n2), plan.col_plan)
    stage3 = stage3.reshape(batch, plan.n1, plan.n2)

    return stage3.transpose(1, 2).contiguous().reshape_as(x_contig)


def _execute_plan(x: torch.Tensor, plan: FFTPlan) -> torch.Tensor:
    if isinstance(plan, LeafPlan):
        return _execute_leaf_plan(x, plan)
    return _execute_four_step_plan(x, plan)


def prepare_fft_tables(device: torch.device, plan: FFTPlan) -> None:
    for leaf in unique_leaf_plans(plan):
        _get_leaf_tables(device, leaf)
    _prepare_four_step_twiddles(device, plan)


def prepare_fft_kernels(plan: FFTPlan) -> None:
    for leaf in unique_leaf_plans(plan):
        _get_leaf_kernel(leaf)


def _prepare_four_step_twiddles(device: torch.device, plan: FFTPlan) -> None:
    if isinstance(plan, LeafPlan):
        return
    _get_four_step_twiddle(device, plan.n1, plan.n2)
    _prepare_four_step_twiddles(device, plan.row_plan)
    _prepare_four_step_twiddles(device, plan.col_plan)


def clear_exec_caches() -> None:
    _TABLE_CACHE.clear()
    _KERNEL_CACHE.clear()
    _TWIDDLE_CACHE.clear()


def clear_fft_caches() -> None:
    clear_plan_cache()
    clear_exec_caches()


def _resolve_fft_plan(
    n: int,
    plan: FFTPlan | None = None,
    split_spec: FFTDecompositionSpec | None = None,
) -> FFTPlan:
    if plan is not None and split_spec is not None:
        raise ValueError("pass either plan or split_spec, not both")
    if plan is not None:
        return plan
    if split_spec is not None:
        return build_fft_plan(n, split_spec)
    return get_fft_plan(n)


def fft_mixed_radix_triton(
    x: torch.Tensor,
    plan: FFTPlan | None = None,
    split_spec: FFTDecompositionSpec | None = None,
) -> torch.Tensor:
    if x.dtype != torch.complex64:
        raise TypeError("expected complex64 input")
    if not x.is_cuda:
        raise ValueError("expected a CUDA tensor")
    if x.ndim == 0:
        raise ValueError("expected at least a 1-D tensor")

    n = x.shape[-1]
    resolved_plan = _resolve_fft_plan(n, plan=plan, split_spec=split_spec)
    if resolved_plan.length != n:
        raise ValueError(f"plan length mismatch: plan={resolved_plan.length}, tensor={n}")
    out = _execute_plan(x, resolved_plan)
    return out.reshape(x.shape)


def fft_mixed_radix_triton_manual(
    x: torch.Tensor,
    split_spec: FFTDecompositionSpec,
) -> torch.Tensor:
    return fft_mixed_radix_triton(x, split_spec=split_spec)


__all__ = [
    "clear_exec_caches",
    "clear_fft_caches",
    "fft_mixed_radix_triton",
    "fft_mixed_radix_triton_manual",
    "prepare_fft_kernels",
    "prepare_fft_tables",
]
