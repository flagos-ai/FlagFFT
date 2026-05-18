from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.codegen.kernels import (
    LeafPlan,
    _CODELET_DIR,
    _build_four_step_col_kernel_source,
    _build_four_step_row_kernel_source,
    _build_leaf_kernel_source,
    contiguous_batch_pack_for,
    four_step_col_inner_pack_for,
    lane_block_for,
)


_POINTER_SIGNATURE = "*fp32:16"


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


def _signature(arg_names: list[str]) -> str:
    return ",".join("i32" if name == "nbatch" else _POINTER_SIGNATURE for name in arg_names)


def _metadata(
    *,
    module_path: Path,
    kernel_name: str,
    arg_names: list[str],
    plan: LeafPlan,
    kernel_type: str,
    n1: int,
    n2: int,
) -> dict[str, Any]:
    batch_per_block = contiguous_batch_pack_for(plan) if kernel_type == "leaf" else 1
    return {
        "module_path": str(module_path),
        "kernel_name": kernel_name,
        "signature": _signature(arg_names),
        "num_warps": int(plan.num_warps),
        "num_stages": 1,
        "batch_per_block": int(batch_per_block),
        "arg_names": arg_names,
        "kernel_type": kernel_type,
        "length": int(plan.length),
        "lanes": int(plan.lanes),
        "direction": plan.direction,
        "n1": int(n1),
        "n2": int(n2),
    }


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
    if kernel == "leaf":
        module_name = f"flagfft_jit_{direction_tag}_{factor_tag}_l{lanes}_b{lane_block}"
    else:
        module_name = (
            f"flagfft_jit_{kernel}_{direction_tag}_{factor_tag}"
            f"_n{four_step_n1}_{four_step_n2}_l{lanes}_b{lane_block}"
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
    )
    if kernel == "four_step_col":
        metadata["inner_pack"] = four_step_col_inner_pack_for(four_step_n1, four_step_n2)
    (out_dir / f"{module_name}.json").write_text(json.dumps(metadata, sort_keys=True))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate FlagFFT libtriton_jit kernel sources")
    parser.add_argument("--kernel", choices=("leaf", "four_step_row", "four_step_col"), required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--factors", type=_csv_ints, required=True)
    parser.add_argument("--lanes", type=int, required=True)
    parser.add_argument("--num-warps", type=int, required=True)
    parser.add_argument("--generic-radices", type=_csv_ints, default=())
    parser.add_argument("--smem-size", type=int, required=True)
    parser.add_argument("--direction", choices=("forward", "inverse"), default="forward")
    parser.add_argument("--four-step-n1", type=int, default=0)
    parser.add_argument("--four-step-n2", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.kernel in {"four_step_row", "four_step_col"}:
        if args.four_step_n1 <= 0 or args.four_step_n2 <= 0:
            parser.error(f"--kernel {args.kernel} requires --four-step-n1 and --four-step-n2")

    metadata = emit_jit_kernel(
        kernel=args.kernel,
        length=args.length,
        factors=args.factors,
        lanes=args.lanes,
        num_warps=args.num_warps,
        generic_radices=args.generic_radices,
        smem_size=args.smem_size,
        direction=args.direction,
        four_step_n1=args.four_step_n1,
        four_step_n2=args.four_step_n2,
        out_dir=args.out_dir,
    )
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
