from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from textwrap import dedent
from typing import Any

from .kernels import (
    _CODELET_DIR,
    LeafPlan,
    _build_compact_to_hermitian_full_kernel_source,
    _build_complex_to_real_kernel_source,
    _build_four_step_col_kernel_source,
    _build_four_step_row_kernel_source,
    _build_leaf_kernel_source,
    _build_r2c_half_pack_kernel_source,
    _build_real_to_complex_kernel_source,
    _build_reshape_pack_kernel_source,
    _build_tiled_transpose_kernel_source,
    _build_twiddle_reshape_pack_kernel_source,
    contiguous_batch_pack_for,
    four_step_col_inner_pack_for,
    lane_block_for,
)

_BLUESTEIN_BLOCK = 256
_BLUESTEIN_NUM_WARPS = 4
_BLUESTEIN_NUM_STAGES = 4
_RADER_BLOCK = 256
_RADER_NUM_WARPS = 4
_RADER_NUM_STAGES = 4
_RESHAPE_NUM_WARPS = 4
_RESHAPE_NUM_STAGES = 1


def _pointer_signature(dtype: str) -> str:
    if dtype == "complex128":
        return "*fp64:16"
    return "*fp32:16"


def _dtype_suffix(dtype: str) -> str:
    return "f64" if dtype == "complex128" else "f32"


def _zero_literal(dtype: str) -> str:
    # `tl.load(..., other=0.0)` is auto-promoted to the pointer's dtype by
    # Triton's masked-load lowering, so a single literal works for fp32 and
    # fp64. Indexing a constexpr (e.g. ``tl.zeros((1,), tl.float64)[0]``) is
    # rejected at IR-build time.
    del dtype
    return "0.0"


def _csv_ints(raw: str) -> tuple[int, ...]:
    if raw == "":
        return ()
    return tuple(int(part) for part in raw.split(",") if part)


def _module_source(kernel_source: str) -> str:
    helpers = (
        "import triton\n"
        "import triton.language as tl\n"
        "import triton.experimental.tle.language as tle\n\n"
    )
    utils_path = _CODELET_DIR / "utils.py"
    if utils_path.exists():
        helpers += utils_path.read_text() + "\n\n"

    for codelet_file in sorted(_CODELET_DIR.glob("*.py")):
        if codelet_file.name not in {utils_path.name, Path(__file__).name}:
            helpers += codelet_file.read_text() + "\n\n"

    return helpers + "\n\n" + kernel_source + "\n"


def _arg_signature(name: str, dtype: str) -> str:
    if name == "nbatch":
        return "i32"
    if name in {"n", "m", "input_distance", "output_distance"}:
        return "i64"
    if name == "idx_ptr":
        return "*i32:16"
    return _pointer_signature(dtype)


def _signature(arg_names: list[str], dtype: str) -> str:
    return ",".join(_arg_signature(name, dtype) for name in arg_names)


def _metadata(
    *,
    module_path: Path,
    kernel_name: str,
    arg_names: list[str],
    plan: LeafPlan,
    kernel_type: str,
    n1: int,
    n2: int,
    dtype: str,
) -> dict[str, Any]:
    batch_per_block = contiguous_batch_pack_for(plan) if kernel_type == "leaf" else 1
    return {
        "module_path": str(module_path),
        "kernel_name": kernel_name,
        "signature": _signature(arg_names, dtype),
        "num_warps": int(plan.num_warps),
        "num_stages": 1,
        "batch_per_block": int(batch_per_block),
        "arg_names": arg_names,
        "kernel_type": kernel_type,
        "length": int(plan.length),
        "lanes": int(plan.lanes),
        "direction": plan.direction,
        "dtype": dtype,
        "n1": int(n1),
        "n2": int(n2),
    }


def _bluestein_kernel_source(kind: str, dtype: str) -> tuple[str, str, list[str]]:
    zero = _zero_literal(dtype)
    div_cast = "tl.cast(m, tl.float64)" if dtype == "complex128" else "m"
    if kind == "bluestein_prepare":
        return (
            "_bluestein_prepare_kernel",
            dedent(
                f"""
                @triton.jit
                def _bluestein_prepare_kernel(
                    in_ptr,
                    chirp_ptr,
                    out_ptr,
                    n,
                    m,
                    nbatch,
                ):
                    pid_block = tl.program_id(0)
                    pid_batch = tl.program_id(1)
                    offsets = pid_block * 256 + tl.arange(0, 256)
                    mask = offsets < m
                    in_mask = mask & (offsets < n)

                    src = in_ptr + (pid_batch * n + offsets) * 2
                    xr = tl.load(src, mask=in_mask, other={zero})
                    xi = tl.load(src + 1, mask=in_mask, other={zero})
                    cr = tl.load(chirp_ptr + offsets * 2, mask=in_mask, other={zero})
                    ci = tl.load(chirp_ptr + offsets * 2 + 1, mask=in_mask, other={zero})
                    yr, yi = _cmul(xr, xi, cr, ci)

                    dst = out_ptr + (pid_batch * m + offsets) * 2
                    tl.store(dst, yr, mask=mask)
                    tl.store(dst + 1, yi, mask=mask)
                """
            ),
            ["in_ptr", "chirp_ptr", "out_ptr", "n", "m", "nbatch"],
        )
    if kind == "bluestein_pointwise":
        return (
            "_bluestein_pointwise_kernel",
            dedent(
                f"""
                @triton.jit
                def _bluestein_pointwise_kernel(
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
    if kind == "bluestein_finalize":
        return (
            "_bluestein_finalize_kernel",
            dedent(
                f"""
                @triton.jit
                def _bluestein_finalize_kernel(
                    in_ptr,
                    chirp_ptr,
                    out_ptr,
                    n,
                    m,
                    nbatch,
                ):
                    pid_block = tl.program_id(0)
                    pid_batch = tl.program_id(1)
                    offsets = pid_block * 256 + tl.arange(0, 256)
                    mask = offsets < n

                    src = in_ptr + (pid_batch * m + offsets) * 2
                    xr = tl.load(src, mask=mask, other={zero}) / {div_cast}
                    xi = -tl.load(src + 1, mask=mask, other={zero}) / {div_cast}
                    cr = tl.load(chirp_ptr + offsets * 2, mask=mask, other={zero})
                    ci = tl.load(chirp_ptr + offsets * 2 + 1, mask=mask, other={zero})
                    yr, yi = _cmul(xr, xi, cr, ci)

                    dst = out_ptr + (pid_batch * n + offsets) * 2
                    tl.store(dst, yr, mask=mask)
                    tl.store(dst + 1, yi, mask=mask)
                """
            ),
            ["in_ptr", "chirp_ptr", "out_ptr", "n", "m", "nbatch"],
        )
    raise ValueError(f"unsupported JIT kernel kind: {kind}")


def _emit_bluestein_jit_kernel(
    *,
    kernel: str,
    n: int,
    m: int,
    dtype: str = "complex64",
    out_dir: Path,
) -> dict[str, Any]:
    kernel_name, kernel_source, arg_names = _bluestein_kernel_source(kernel, dtype)
    suffix = _dtype_suffix(dtype)
    module_name = f"flagfft_jit_{kernel}_n{n}_m{m}_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    module_path = out_dir / f"{module_name}.py"
    module_path.write_text(_module_source(kernel_source))
    metadata = {
        "module_path": str(module_path),
        "kernel_name": kernel_name,
        "signature": _signature(arg_names, dtype),
        "num_warps": _BLUESTEIN_NUM_WARPS,
        "num_stages": _BLUESTEIN_NUM_STAGES,
        "batch_per_block": 1,
        "arg_names": arg_names,
        "kernel_type": kernel,
        "dtype": dtype,
        "bluestein_n": int(n),
        "bluestein_m": int(m),
        "block": _BLUESTEIN_BLOCK,
    }
    (out_dir / f"{module_name}.json").write_text(json.dumps(metadata, sort_keys=True))
    return metadata


def _next_power_of_two(value: int) -> int:
    result = 1
    while result < value:
        result <<= 1
    return result


def _rader_kernel_source(
    kind: str, n: int, m: int, dtype: str
) -> tuple[str, str, list[str]]:
    zero = _zero_literal(dtype)
    div_cast = "tl.cast(m, tl.float64)" if dtype == "complex128" else "m"
    sum_block = _next_power_of_two(n)
    if kind == "rader_prepare":
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
    if kind == "rader_pointwise":
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
    if kind == "rader_finalize":
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
                    yr = x0r + cr
                    yi = x0i + ci

                    dst_index = tl.load(idx_ptr + offsets, mask=mask, other=0)
                    dst = out_ptr + (pid_batch * n + dst_index) * 2
                    tl.store(dst, yr, mask=mask)
                    tl.store(dst + 1, yi, mask=mask)

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
    raise ValueError(f"unsupported JIT kernel kind: {kind}")


def _emit_rader_jit_kernel(
    *,
    kernel: str,
    n: int,
    m: int,
    dtype: str = "complex64",
    out_dir: Path,
) -> dict[str, Any]:
    kernel_name, kernel_source, arg_names = _rader_kernel_source(kernel, n, m, dtype)
    suffix = _dtype_suffix(dtype)
    module_name = f"flagfft_jit_{kernel}_n{n}_m{m}_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    module_path = out_dir / f"{module_name}.py"
    module_path.write_text(_module_source(kernel_source))
    metadata = {
        "module_path": str(module_path),
        "kernel_name": kernel_name,
        "signature": _signature(arg_names, dtype),
        "num_warps": _RADER_NUM_WARPS,
        "num_stages": _RADER_NUM_STAGES,
        "batch_per_block": 1,
        "arg_names": arg_names,
        "kernel_type": kernel,
        "dtype": dtype,
        "rader_n": int(n),
        "rader_m": int(m),
        "block": _RADER_BLOCK,
    }
    (out_dir / f"{module_name}.json").write_text(json.dumps(metadata, sort_keys=True))
    return metadata


def _emit_reshape_jit_kernel(
    *,
    kernel: str,
    n1: int,
    n2: int,
    dtype: str = "complex64",
    out_dir: Path,
) -> dict[str, Any]:
    if kernel == "reshape_pack":
        kernel_name, kernel_source, arg_names = _build_reshape_pack_kernel_source(
            n1, n2, dtype
        )
    elif kernel == "twiddle_reshape_pack":
        (
            kernel_name,
            kernel_source,
            arg_names,
        ) = _build_twiddle_reshape_pack_kernel_source(n1, n2, dtype)
    else:
        raise ValueError(f"unsupported reshape kernel kind: {kernel}")

    suffix = _dtype_suffix(dtype)
    module_name = f"flagfft_jit_{kernel}_n{n1}_{n2}_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    module_path = out_dir / f"{module_name}.py"
    module_path.write_text(_module_source(kernel_source))
    metadata = {
        "module_path": str(module_path),
        "kernel_name": kernel_name,
        "signature": _signature(arg_names, dtype),
        "num_warps": _RESHAPE_NUM_WARPS,
        "num_stages": _RESHAPE_NUM_STAGES,
        "batch_per_block": 1,
        "arg_names": arg_names,
        "kernel_type": kernel,
        "dtype": dtype,
        "reshape_n1": int(n1),
        "reshape_n2": int(n2),
        "block": 256,
    }
    (out_dir / f"{module_name}.json").write_text(json.dumps(metadata, sort_keys=True))
    return metadata


def _emit_r2c_pointwise_jit_kernel(
    *,
    kernel: str,
    n: int,
    dtype: str,
    out_dir: Path,
) -> dict[str, Any]:
    if kernel == "real_to_complex":
        kernel_name, kernel_source, arg_names = _build_real_to_complex_kernel_source(
            n, dtype
        )
    elif kernel == "r2c_half_pack":
        kernel_name, kernel_source, arg_names = _build_r2c_half_pack_kernel_source(
            n, dtype
        )
    elif kernel == "compact_to_hermitian_full":
        (
            kernel_name,
            kernel_source,
            arg_names,
        ) = _build_compact_to_hermitian_full_kernel_source(n, dtype)
    elif kernel == "complex_to_real":
        kernel_name, kernel_source, arg_names = _build_complex_to_real_kernel_source(
            n, dtype
        )
    else:
        raise ValueError(f"unsupported R2C pointwise kernel kind: {kernel}")

    dtype_tag = _dtype_suffix(dtype)
    module_name = f"flagfft_jit_{kernel}_n{n}_{dtype_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    module_path = out_dir / f"{module_name}.py"
    module_path.write_text(_module_source(kernel_source))

    sys.path.insert(0, str(module_path.parent))
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load generated kernel module {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    arg_names = list(getattr(module, kernel_name).arg_names)
    metadata = {
        "module_path": str(module_path),
        "kernel_name": kernel_name,
        "signature": _signature(arg_names, dtype),
        "num_warps": _RESHAPE_NUM_WARPS,
        "num_stages": _RESHAPE_NUM_STAGES,
        "batch_per_block": 1,
        "arg_names": arg_names,
        "kernel_type": kernel,
        "dtype": dtype,
        "length": int(n),
        "block": 256,
    }
    (out_dir / f"{module_name}.json").write_text(json.dumps(metadata, sort_keys=True))
    return metadata


def emit_jit_kernel(
    *,
    kernel: str,
    length: int,
    factors: tuple[int, ...],
    lanes: int,
    num_warps: int,
    generic_radices: tuple[int, ...],
    smem_size: int,
    direction: str,
    dtype: str,
    four_step_n1: int,
    four_step_n2: int,
    out_dir: Path,
) -> dict[str, Any]:
    plan = LeafPlan(
        length=length,
        factors=factors,
        remainder=1,
        lanes=lanes,
        num_warps=num_warps,
        generic_radices=generic_radices,
        smem_size=smem_size,
        direction=direction,
        dtype=dtype,
    )
    if kernel == "leaf":
        kernel_name, kernel_source = _build_leaf_kernel_source(plan)
        n1 = 0
        n2 = 0
    elif kernel == "four_step_row":
        kernel_name, kernel_source = _build_four_step_row_kernel_source(
            plan, four_step_n1, four_step_n2
        )
        n1 = four_step_n1
        n2 = four_step_n2
    elif kernel == "four_step_col":
        kernel_name, kernel_source = _build_four_step_col_kernel_source(
            plan, four_step_n1, four_step_n2
        )
        n1 = four_step_n1
        n2 = four_step_n2
    else:
        raise ValueError(f"unsupported JIT kernel kind: {kernel}")

    lane_block = lane_block_for(lanes)
    direction_tag = "inv" if direction == "inverse" else "fwd"
    factor_tag = "_".join(str(x) for x in factors)
    dtype_tag = _dtype_suffix(dtype)
    if kernel == "leaf":
        module_name = f"flagfft_jit_{direction_tag}_{factor_tag}_l{lanes}_b{lane_block}_{dtype_tag}"
    else:
        module_name = (
            f"flagfft_jit_{kernel}_{direction_tag}_{factor_tag}"
            f"_n{four_step_n1}_{four_step_n2}_l{lanes}_b{lane_block}_{dtype_tag}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    module_path = out_dir / f"{module_name}.py"
    module_path.write_text(_module_source(kernel_source))

    sys.path.insert(0, str(module_path.parent))
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load generated kernel module {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    arg_names = list(getattr(module, kernel_name).arg_names)
    metadata = _metadata(
        module_path=module_path,
        kernel_name=kernel_name,
        arg_names=arg_names,
        plan=plan,
        kernel_type=kernel,
        n1=n1,
        n2=n2,
        dtype=dtype,
    )
    if kernel == "four_step_col":
        metadata["inner_pack"] = four_step_col_inner_pack_for(
            four_step_n1, four_step_n2, dtype
        )
    (out_dir / f"{module_name}.json").write_text(json.dumps(metadata, sort_keys=True))
    return metadata


def _emit_tiled_transpose_jit_kernel(
    *,
    n0: int,
    n1: int,
    dtype: str = "complex64",
    tile_size: int = 32,  # must match constexpr tile_size in raw_nodes.cpp CompiledRaw2DNode::execute()
    out_dir: Path,
) -> dict[str, Any]:
    kernel_name, kernel_source, arg_names = _build_tiled_transpose_kernel_source(
        n0, n1, dtype, tile_size
    )
    suffix = _dtype_suffix(dtype)
    module_name = f"flagfft_jit_tiled_transpose_n{n0}_{n1}_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    module_path = out_dir / f"{module_name}.py"
    module_path.write_text(_module_source(kernel_source))
    metadata = {
        "module_path": str(module_path),
        "kernel_name": kernel_name,
        "signature": _signature(arg_names, dtype),
        "num_warps": 4,
        "num_stages": 1,
        "batch_per_block": 1,
        "arg_names": arg_names,
        "kernel_type": "tiled_transpose",
        "dtype": dtype,
        "reshape_n1": int(n0),
        "reshape_n2": int(n1),
        "tile_size": int(tile_size),
        "block": tile_size * tile_size,
    }
    (out_dir / f"{module_name}.json").write_text(json.dumps(metadata, sort_keys=True))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate FlagFFT libtriton_jit kernel sources"
    )
    parser.add_argument(
        "--kernel",
        choices=(
            "leaf",
            "four_step_row",
            "four_step_col",
            "bluestein_prepare",
            "bluestein_pointwise",
            "bluestein_finalize",
            "rader_prepare",
            "rader_pointwise",
            "rader_finalize",
            "reshape_pack",
            "twiddle_reshape_pack",
            "tiled_transpose",
            "real_to_complex",
            "r2c_half_pack",
            "compact_to_hermitian_full",
            "complex_to_real",
        ),
        required=True,
    )
    parser.add_argument("--length", type=int)
    parser.add_argument("--factors", type=_csv_ints)
    parser.add_argument("--lanes", type=int)
    parser.add_argument("--num-warps", type=int)
    parser.add_argument("--generic-radices", type=_csv_ints, default=())
    parser.add_argument("--smem-size", type=int)
    parser.add_argument(
        "--direction", choices=("forward", "inverse"), default="forward"
    )
    parser.add_argument(
        "--dtype", choices=("complex64", "complex128"), default="complex64"
    )
    parser.add_argument("--four-step-n1", type=int, default=0)
    parser.add_argument("--four-step-n2", type=int, default=0)
    parser.add_argument("--bluestein-n", type=int)
    parser.add_argument("--bluestein-m", type=int)
    parser.add_argument("--rader-n", type=int)
    parser.add_argument("--rader-m", type=int)
    parser.add_argument("--reshape-n1", type=int, default=0)
    parser.add_argument("--reshape-n2", type=int, default=0)
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.kernel.startswith("bluestein_"):
        if args.bluestein_n is None or args.bluestein_m is None:
            parser.error(
                f"--kernel {args.kernel} requires --bluestein-n and --bluestein-m"
            )
        metadata = _emit_bluestein_jit_kernel(
            kernel=args.kernel,
            n=args.bluestein_n,
            m=args.bluestein_m,
            dtype=args.dtype,
            out_dir=args.out_dir,
        )
        print(json.dumps(metadata, sort_keys=True))
        return

    if args.kernel.startswith("rader_"):
        if args.rader_n is None or args.rader_m is None:
            parser.error(f"--kernel {args.kernel} requires --rader-n and --rader-m")
        metadata = _emit_rader_jit_kernel(
            kernel=args.kernel,
            n=args.rader_n,
            m=args.rader_m,
            dtype=args.dtype,
            out_dir=args.out_dir,
        )
        print(json.dumps(metadata, sort_keys=True))
        return

    if args.kernel in {"reshape_pack", "twiddle_reshape_pack"}:
        if args.reshape_n1 <= 0 or args.reshape_n2 <= 0:
            parser.error(
                f"--kernel {args.kernel} requires --reshape-n1 and --reshape-n2"
            )
        metadata = _emit_reshape_jit_kernel(
            kernel=args.kernel,
            n1=args.reshape_n1,
            n2=args.reshape_n2,
            dtype=args.dtype,
            out_dir=args.out_dir,
        )
        print(json.dumps(metadata, sort_keys=True))
        return

    if args.kernel == "tiled_transpose":
        if args.reshape_n1 <= 0 or args.reshape_n2 <= 0:
            parser.error(
                "--kernel tiled_transpose requires --reshape-n1 and --reshape-n2"
            )
        metadata = _emit_tiled_transpose_jit_kernel(
            n0=args.reshape_n1,
            n1=args.reshape_n2,
            dtype=args.dtype,
            tile_size=args.tile_size,
            out_dir=args.out_dir,
        )
        print(json.dumps(metadata, sort_keys=True))
        return

    if args.kernel in {
        "real_to_complex",
        "r2c_half_pack",
        "compact_to_hermitian_full",
        "complex_to_real",
    }:
        if args.length is None or args.length <= 0:
            parser.error(f"--kernel {args.kernel} requires --length")
        metadata = _emit_r2c_pointwise_jit_kernel(
            kernel=args.kernel,
            n=args.length,
            dtype=args.dtype,
            out_dir=args.out_dir,
        )
        print(json.dumps(metadata, sort_keys=True))
        return

    missing = [
        name
        for name in ("length", "factors", "lanes", "num_warps", "smem_size")
        if getattr(args, name) is None
    ]
    if missing:
        parser.error(
            f"--kernel {args.kernel} requires "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )
    if args.kernel in {"four_step_row", "four_step_col"}:
        if args.four_step_n1 <= 0 or args.four_step_n2 <= 0:
            parser.error(
                f"--kernel {args.kernel} requires --four-step-n1 and --four-step-n2"
            )

    metadata = emit_jit_kernel(
        kernel=args.kernel,
        length=args.length,
        factors=args.factors,
        lanes=args.lanes,
        num_warps=args.num_warps,
        generic_radices=args.generic_radices,
        smem_size=args.smem_size,
        direction=args.direction,
        dtype=args.dtype,
        four_step_n1=args.four_step_n1,
        four_step_n2=args.four_step_n2,
        out_dir=args.out_dir,
    )
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
