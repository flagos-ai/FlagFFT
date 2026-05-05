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

from src.exec import (
    _CODELET_DIR,
    _FOUR_STEP_NUM_WARPS,
    _FOUR_STEP_TILE_COLS,
    _FOUR_STEP_TILE_ROWS,
    _build_leaf_kernel_source,
    _transpose_complex_kernel,
    _twiddle_transpose_complex_kernel,
)
from src.plan import LeafPlan, lane_block_for


_POINTER_SIGNATURE = "*fp32"


def _csv_ints(raw: str) -> tuple[int, ...]:
    if raw == "":
        return ()
    return tuple(int(part) for part in raw.split(",") if part)


def _leaf_module_source(plan: LeafPlan) -> tuple[str, str]:
    helpers = ""
    utils_path = _CODELET_DIR / "utils.py"
    if utils_path.exists():
        helpers += utils_path.read_text() + "\n\n"

    for codelet_file in sorted(_CODELET_DIR.glob("*.py")):
        if codelet_file.name not in {utils_path.name, Path(__file__).name}:
            helpers += codelet_file.read_text() + "\n\n"

    kernel_name, source = _build_leaf_kernel_source(plan)
    return kernel_name, helpers + "\n\n" + source + "\n"


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
    out_dir.mkdir(parents=True, exist_ok=True)
    module_path = out_dir / f"{module_name}.py"

    kernel_name, source = _leaf_module_source(plan)
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
        num_warps=num_warps,
        num_stages=1,
        target=target,
        module_path=str(module_path),
    )
    artifact["kernel_type"] = "leaf"
    artifact["length"] = length
    artifact["lanes"] = lanes
    artifact["grid"] = {"x": "nbatch", "y": 1, "z": 1}
    artifact_path = out_dir / f"{module_name}.json"
    artifact_path.write_text(json.dumps(artifact, sort_keys=True))
    return artifact


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
    artifact["grid"] = {
        "x": "ceil(cols / BLOCK_COLS)",
        "y": "ceil(rows / BLOCK_ROWS)",
        "z": "batch",
    }
    artifact_path = out_dir / f"four_step_{kind}.json"
    artifact_path.write_text(json.dumps(artifact, sort_keys=True))
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="AOT compile FlagFFT Triton kernels")
    parser.add_argument("--kernel", choices=("leaf", "transpose", "twiddle_transpose"), default="leaf")
    parser.add_argument("--length", type=int)
    parser.add_argument("--factors", type=_csv_ints)
    parser.add_argument("--lanes", type=int)
    parser.add_argument("--num-warps", type=int)
    parser.add_argument("--generic-radices", type=_csv_ints, default=())
    parser.add_argument("--smem-size", type=int)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.kernel == "leaf":
        missing = [
            name
            for name in ("length", "factors", "lanes", "num_warps", "smem_size")
            if getattr(args, name) is None
        ]
        if missing:
            parser.error("--kernel leaf requires " + ", ".join(f"--{name.replace('_', '-')}" for name in missing))
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
        artifact = _compile_four_step_kernel(kind=args.kernel, target=args.target, out_dir=args.out_dir)
    print(json.dumps(artifact, sort_keys=True))


if __name__ == "__main__":
    main()
