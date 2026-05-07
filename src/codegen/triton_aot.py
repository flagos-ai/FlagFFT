from __future__ import annotations

import argparse
import binascii
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import triton
import triton.backends

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.codegen.kernels import (
    LeafPlan,
    _CODELET_DIR,
    _FOUR_STEP_NUM_WARPS,
    _FOUR_STEP_TILE_COLS,
    _FOUR_STEP_TILE_ROWS,
    _build_four_step_col_kernel_source,
    _build_four_step_row_kernel_source,
    _build_leaf_kernel_source,
    _bluestein_finalize_kernel,
    _bluestein_pointwise_kernel,
    _bluestein_prepare_kernel,
    _transpose_complex_kernel,
    _twiddle_transpose_complex_kernel,
    contiguous_batch_pack_for,
    four_step_col_inner_pack_for,
    lane_block_for,
)


_POINTER_SIGNATURE = "*fp32"


def _csv_ints(raw: str) -> tuple[int, ...]:
    if raw == "":
        return ()
    return tuple(int(part) for part in raw.split(",") if part)


def _leaf_module_source(plan: LeafPlan) -> tuple[str, str]:
    kernel_name, source = _build_leaf_kernel_source(plan)
    return kernel_name, _module_source(source)


def _module_source(kernel_source: str) -> str:
    helpers = ""
    utils_path = _CODELET_DIR / "utils.py"
    if utils_path.exists():
        helpers += utils_path.read_text() + "\n\n"

    for codelet_file in sorted(_CODELET_DIR.glob("*.py")):
        if codelet_file.name not in {utils_path.name, Path(__file__).name}:
            helpers += codelet_file.read_text() + "\n\n"

    return helpers + "\n\n" + kernel_source + "\n"


def _compile_leaf_source(
    *,
    kernel_name: str,
    source: str,
    module_name: str,
    plan: LeafPlan,
    kernel_type: str,
    grid: dict[str, Any],
    target: str,
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    module_path = out_dir / f"{module_name}.py"
    module_path.write_text(source)

    sys.path.insert(0, str(module_path.parent))
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load generated kernel module {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kernel = getattr(module, kernel_name)

    arg_names = list(kernel.arg_names)
    signature = {name: _POINTER_SIGNATURE for name in arg_names}
    signature["nbatch"] = "i32"
    attrs = {(idx,): [["tt.divisibility", 16]] for idx, name in enumerate(arg_names) if name != "nbatch"}
    artifact = _compile_kernel_artifact(
        kernel=kernel,
        kernel_name=kernel_name,
        arg_names=arg_names,
        signature=signature,
        attrs=attrs,
        constexprs={},
        num_warps=plan.num_warps,
        num_stages=1,
        target=target,
        module_path=str(module_path),
    )
    artifact["kernel_type"] = kernel_type
    artifact["length"] = plan.length
    artifact["lanes"] = plan.lanes
    artifact["batch_per_block"] = contiguous_batch_pack_for(plan) if kernel_type == "leaf" else 1
    artifact["grid"] = grid
    artifact_path = out_dir / f"{module_name}.json"
    artifact_path.write_text(json.dumps(artifact, sort_keys=True))
    return artifact


def compile_leaf_kernel(
    *,
    length: int,
    factors: tuple[int, ...],
    lanes: int,
    num_warps: int,
    generic_radices: tuple[int, ...],
    smem_size: int,
    target: str,
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
    )
    lane_block = lane_block_for(lanes)
    module_name = "flagfft_aot_" + "_".join(str(x) for x in factors) + f"_l{lanes}_b{lane_block}"
    kernel_name, source = _leaf_module_source(plan)
    return _compile_leaf_source(
        kernel_name=kernel_name,
        source=source,
        module_name=module_name,
        plan=plan,
        kernel_type="leaf",
        grid={"x": "nbatch", "y": 1, "z": 1},
        target=target,
        out_dir=out_dir,
    )


def compile_four_step_leaf_kernel(
    *,
    kind: str,
    length: int,
    factors: tuple[int, ...],
    lanes: int,
    num_warps: int,
    generic_radices: tuple[int, ...],
    smem_size: int,
    four_step_n1: int,
    four_step_n2: int,
    target: str,
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
    )
    if kind == "four_step_row":
        kernel_name, kernel_source = _build_four_step_row_kernel_source(
            plan, four_step_n1, four_step_n2
        )
        grid = {"x": "n2", "y": "nbatch", "z": 1}
    elif kind == "four_step_col":
        kernel_name, kernel_source = _build_four_step_col_kernel_source(
            plan, four_step_n1, four_step_n2
        )
        grid = {
            "x": f"ceil_div(n1,{four_step_col_inner_pack_for(four_step_n1, four_step_n2)})",
            "y": "nbatch",
            "z": 1,
        }
    else:
        raise ValueError(f"unsupported four-step leaf kernel kind: {kind}")

    lane_block = lane_block_for(lanes)
    module_name = (
        "flagfft_aot_"
        + kind
        + "_"
        + "_".join(str(x) for x in factors)
        + f"_n{four_step_n1}_{four_step_n2}_l{lanes}_b{lane_block}"
    )
    return _compile_leaf_source(
        kernel_name=kernel_name,
        source=_module_source(kernel_source),
        module_name=module_name,
        plan=plan,
        kernel_type=kind,
        grid=grid,
        target=target,
        out_dir=out_dir,
    )


def _compile_kernel_artifact(
    *,
    kernel: Any,
    kernel_name: str,
    arg_names: list[str],
    signature: dict[str, str],
    attrs: dict[tuple[int], list[list[Any]]],
    constexprs: dict[str, Any],
    num_warps: int,
    num_stages: int,
    target: str,
    module_path: str | None = None,
) -> dict[str, Any]:
    src = triton.compiler.ASTSource(fn=kernel, constexprs=constexprs, signature=signature, attrs=attrs)

    backend_name, arch, warp_size = target.split(":")
    parsed_arch: int | str = int(arch) if arch.isdigit() else arch
    triton_target = triton.backends.compiler.GPUTarget(backend_name, parsed_arch, int(warp_size))
    backend = triton.compiler.make_backend(triton_target)
    options = backend.parse_options({"num_warps": num_warps, "num_stages": num_stages})
    compiled = triton.compile(src, target=triton_target, options=options.__dict__)

    if getattr(compiled.metadata, "global_scratch_size", 0) > 0:
        raise RuntimeError("AOT kernels with global scratch are not supported")
    if getattr(compiled.metadata, "profile_scratch_size", 0) > 0:
        raise RuntimeError("AOT kernels with profile scratch are not supported")

    binary = compiled.asm[backend.binary_ext]
    artifact = {
        "kernel_name": kernel_name,
        "module_path": module_path,
        "target": target,
        "binary_ext": backend.binary_ext,
        "cubin_hex": binascii.hexlify(binary).decode("ascii"),
        "shared": int(compiled.metadata.shared),
        "num_warps": int(num_warps),
        "num_stages": int(num_stages),
        "arg_names": arg_names,
        "signature": signature,
        "constexprs": constexprs,
        "grid": None,
    }
    return artifact


def _compile_four_step_kernel(
    *,
    kind: str,
    target: str,
    out_dir: Path,
) -> dict[str, Any]:
    if kind == "transpose":
        kernel = _transpose_complex_kernel
    elif kind == "twiddle_transpose":
        kernel = _twiddle_transpose_complex_kernel
    else:
        raise ValueError(f"unsupported four-step kernel kind: {kind}")

    constexprs = {
        "BLOCK_ROWS": _FOUR_STEP_TILE_ROWS,
        "BLOCK_COLS": _FOUR_STEP_TILE_COLS,
    }
    arg_names = [name for name in kernel.arg_names if name not in constexprs]
    signature = {
        name: (_POINTER_SIGNATURE if name.endswith("_ptr") else "i64")
        for name in arg_names
    }
    attrs = {
        (idx,): [["tt.divisibility", 16]]
        for idx, name in enumerate(arg_names)
        if name.endswith("_ptr")
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = _compile_kernel_artifact(
        kernel=kernel,
        kernel_name=kernel.fn.__name__,
        arg_names=arg_names,
        signature=signature,
        attrs=attrs,
        constexprs=constexprs,
        num_warps=_FOUR_STEP_NUM_WARPS,
        num_stages=1,
        target=target,
        module_path=None,
    )
    artifact["kernel_type"] = kind
    artifact["tile_rows"] = _FOUR_STEP_TILE_ROWS
    artifact["tile_cols"] = _FOUR_STEP_TILE_COLS
    artifact["batch_per_block"] = 1
    artifact["grid"] = {
        "x": "ceil(cols / BLOCK_COLS)",
        "y": "ceil(rows / BLOCK_ROWS)",
        "z": "batch",
    }
    artifact_path = out_dir / f"four_step_{kind}.json"
    artifact_path.write_text(json.dumps(artifact, sort_keys=True))
    return artifact


def _compile_bluestein_kernel(
    *,
    kind: str,
    n: int,
    m: int,
    target: str,
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if kind == "bluestein_prepare":
        kernel = _bluestein_prepare_kernel
        arg_names = ["in_ptr", "chirp_ptr", "out_ptr", "n", "m", "nbatch"]
        signature = {
            "in_ptr": _POINTER_SIGNATURE,
            "chirp_ptr": _POINTER_SIGNATURE,
            "out_ptr": _POINTER_SIGNATURE,
            "n": "i64",
            "m": "i64",
            "nbatch": "i32",
        }
    elif kind == "bluestein_pointwise":
        kernel = _bluestein_pointwise_kernel
        arg_names = ["a_ptr", "b_ptr", "out_ptr", "m", "nbatch"]
        signature = {
            "a_ptr": _POINTER_SIGNATURE,
            "b_ptr": _POINTER_SIGNATURE,
            "out_ptr": _POINTER_SIGNATURE,
            "m": "i64",
            "nbatch": "i32",
        }
    elif kind == "bluestein_finalize":
        kernel = _bluestein_finalize_kernel
        arg_names = ["in_ptr", "chirp_ptr", "out_ptr", "n", "m", "nbatch"]
        signature = {
            "in_ptr": _POINTER_SIGNATURE,
            "chirp_ptr": _POINTER_SIGNATURE,
            "out_ptr": _POINTER_SIGNATURE,
            "n": "i64",
            "m": "i64",
            "nbatch": "i32",
        }
    else:
        raise ValueError(f"unsupported bluestein kernel kind: {kind}")

    artifact = _compile_kernel_artifact(
        kernel=kernel,
        kernel_name=kernel.__name__,
        arg_names=arg_names,
        signature=signature,
        attrs={(idx,): [["tt.divisibility", 16]] for idx, name in enumerate(arg_names) if name.endswith("_ptr")},
        constexprs={"BLOCK": 256},
        num_warps=4,
        num_stages=4,
        target=target,
    )
    artifact["kernel_type"] = kind
    artifact["bluestein_n"] = n
    artifact["bluestein_m"] = m
    artifact["batch_per_block"] = 1
    artifact["grid"] = {"x": "ceil_div(m,256)", "y": "nbatch", "z": 1}
    artifact_path = out_dir / f"{kind}_n{n}_m{m}.json"
    artifact_path.write_text(json.dumps(artifact, sort_keys=True))
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="AOT compile FlagFFT Triton kernels")
    parser.add_argument(
        "--kernel",
        choices=(
            "leaf",
            "four_step_row",
            "four_step_col",
            "transpose",
            "twiddle_transpose",
            "bluestein_prepare",
            "bluestein_pointwise",
            "bluestein_finalize",
        ),
        default="leaf",
    )
    parser.add_argument("--length", type=int)
    parser.add_argument("--factors", type=_csv_ints)
    parser.add_argument("--lanes", type=int)
    parser.add_argument("--num-warps", type=int)
    parser.add_argument("--generic-radices", type=_csv_ints, default=())
    parser.add_argument("--smem-size", type=int)
    parser.add_argument("--four-step-n1", type=int)
    parser.add_argument("--four-step-n2", type=int)
    parser.add_argument("--bluestein-n", type=int)
    parser.add_argument("--bluestein-m", type=int)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.kernel in {"leaf", "four_step_row", "four_step_col"}:
        missing = [
            name
            for name in ("length", "factors", "lanes", "num_warps", "smem_size")
            if getattr(args, name) is None
        ]
        if args.kernel in {"four_step_row", "four_step_col"}:
            missing.extend(
                name
                for name in ("four_step_n1", "four_step_n2")
                if getattr(args, name) is None
            )
        if missing:
            parser.error(
                f"--kernel {args.kernel} requires "
                + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
            )
        if args.kernel == "leaf":
            artifact = compile_leaf_kernel(
                length=args.length,
                factors=args.factors,
                lanes=args.lanes,
                num_warps=args.num_warps,
                generic_radices=args.generic_radices,
                smem_size=args.smem_size,
                target=args.target,
                out_dir=args.out_dir,
            )
        else:
            artifact = compile_four_step_leaf_kernel(
                kind=args.kernel,
                length=args.length,
                factors=args.factors,
                lanes=args.lanes,
                num_warps=args.num_warps,
                generic_radices=args.generic_radices,
                smem_size=args.smem_size,
                four_step_n1=args.four_step_n1,
                four_step_n2=args.four_step_n2,
                target=args.target,
                out_dir=args.out_dir,
            )
    elif args.kernel in {"transpose", "twiddle_transpose"}:
        artifact = _compile_four_step_kernel(kind=args.kernel, target=args.target, out_dir=args.out_dir)
    else:
        if args.bluestein_n is None or args.bluestein_m is None:
            parser.error(f"--kernel {args.kernel} requires --bluestein-n and --bluestein-m")
        artifact = _compile_bluestein_kernel(
            kind=args.kernel,
            n=args.bluestein_n,
            m=args.bluestein_m,
            target=args.target,
            out_dir=args.out_dir,
        )
    print(json.dumps(artifact, sort_keys=True))


if __name__ == "__main__":
    main()
