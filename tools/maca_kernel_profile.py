#!/usr/bin/env python3
"""Time individual generated MACA kernels.

There is no MetaX profiler in the container and Triton's proton is not built,
so the only way to attribute time inside a plan is to build the generated
kernel directly and launch it in isolation.  The kernels are the exact codegen
output for the given plan; only the launch harness is ours.

Correctness is not checked here -- use the accuracy gate in the acceptance
runner for that.  Twiddle buffers are allocated generously and left
uninitialised because their values do not change the memory access pattern.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

DEVICE = {
    "backend": "maca",
    "device_arch": "102",
    "device_name": "MetaX GPU",
    "warp_size": 64,
    "max_threads_per_block": 1024,
    "max_dynamic_shared_memory": 65536,
    "shared_memory_per_block": 65536,
    "source": "backend_default",
}


def parse_factors(raw: str) -> tuple[int, ...]:
    return tuple(int(v) for v in raw.split(",") if v)


def build(args):
    from flagfft_codegen.backend_profile import BackendProfile, set_profile
    from flagfft_codegen.kernels_common import (
        LeafPlan,
        codelet_radices_for,
        emitted_leaf_factors,
        four_step_col_inner_pack_for,
        four_step_row_inner_pack_for,
        lane_block_for,
    )
    from flagfft_codegen.kernels_leaf import (
        _build_four_step_col_kernel_source,
        _build_four_step_row_kernel_source,
    )
    from flagfft_codegen.metadata import _metadata, _module_source
    from flagfft_codegen.target import set_codegen_target

    set_codegen_target("maca:80:64")
    set_profile(BackendProfile.from_device(DEVICE, args.policy))
    factors = parse_factors(args.factors)
    length = args.n1 if args.kernel == "four_step_row" else args.n2
    plan = LeafPlan(
        length=length,
        factors=factors,
        remainder=1,
        lanes=args.lanes,
        num_warps=args.num_warps,
        generic_radices=(),
        smem_size=lane_block_for(length),
        direction=args.direction,
        dtype=args.dtype,
    )
    if args.kernel == "four_step_row":
        name, source = _build_four_step_row_kernel_source(plan, args.n1, args.n2)
        inner_count = args.n2
    else:
        name, source = _build_four_step_col_kernel_source(plan, args.n1, args.n2)
        inner_count = args.n1
    # Read the launch shape off the same metadata the real codegen uses, so the
    # probe runs the kernel the planner would actually launch.  Launching with
    # the requested --num-warps understates the MACA cooperative leaves, which
    # derive their warp count from the lane block and the pack rather than from
    # the planner hint.
    meta = _metadata(
        module_path=Path("<maca_prof>"),
        kernel_name=name,
        arg_names=("in_ptr",),
        plan=plan,
        kernel_type=args.kernel,
        n1=args.n1,
        n2=args.n2,
        dtype=args.dtype,
    )
    inner_pack = meta["inner_pack"]
    num_warps = meta["num_warps"]
    radices = codelet_radices_for(factors) | codelet_radices_for(
        emitted_leaf_factors(plan)
    )
    module = _module_source(source, tuple(sorted(radices)))
    filename = f"<maca_prof_{name}>"
    import linecache

    linecache.cache[filename] = (len(module), None, module.splitlines(True), filename)
    scope = {"__name__": "maca_kernel_profile"}
    exec(compile(module, filename, "exec"), scope)
    # The twiddle/table argument count varies by kernel family, so read it off
    # the generated signature instead of assuming a fixed list.
    signature = source.split(f"def {name}(", 1)[1].split(")", 1)[0]
    params = [p.strip() for p in signature.replace("\n", " ").split(",") if p.strip()]
    grid = (math.ceil(inner_count / inner_pack), args.batch)
    return scope[name], grid, inner_pack, num_warps, params


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", choices=("four_step_row", "four_step_col"), required=True)
    parser.add_argument("--n1", type=int, required=True)
    parser.add_argument("--n2", type=int, required=True)
    parser.add_argument("--factors", required=True)
    parser.add_argument("--lanes", type=int, default=1)
    parser.add_argument("--num-warps", type=int, default=2)
    parser.add_argument("--dtype", default="complex64")
    parser.add_argument("--direction", default="forward")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--policy", default="legacy")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    import torch

    torch.cuda.set_device(0)
    kernel, grid, inner_pack, num_warps, params = build(args)

    n = args.n1 * args.n2
    total = n * args.batch
    real_dtype = torch.float64 if args.dtype == "complex128" else torch.float32
    # Over-allocate the tables: only the access pattern matters here.
    table_elems = 2 * 4 * n
    dev = "cuda"
    x = torch.zeros(total * 2, dtype=real_dtype, device=dev)
    y = torch.zeros(total * 2, dtype=real_dtype, device=dev)
    tensors = [torch.zeros(table_elems, dtype=real_dtype, device=dev) for _ in params]
    scalars = [args.batch for _ in params]

    def launch():
        argv = [
            x if name.endswith("in_ptr") else
            y if name.endswith("out_ptr") else
            tensors[i] if name.endswith("_ptr") else
            scalars[i]
            for i, name in enumerate(params)
        ]
        kernel[grid](*argv, num_warps=num_warps)

    launch()
    torch.cuda.synchronize()
    for _ in range(args.warmup):
        launch()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        launch()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / args.iters

    moved = total * 8 * 2 / 1e9
    report = {
        "kernel": args.kernel,
        "n1": args.n1,
        "n2": args.n2,
        "factors": args.factors,
        "inner_pack": inner_pack,
        "num_warps": num_warps,
        "batch": args.batch,
        "grid": list(grid),
        "blocks": grid[0] * grid[1],
        "ms": round(ms, 6),
        "gb_per_s": round(moved / ms * 1e3, 1),
        "active_lanes": max(n // r for r in parse_factors(args.factors))
        if args.kernel == "four_step_row"
        else max(args.n2 // r for r in parse_factors(args.factors)),
    }
    if args.json:
        print(json.dumps(report))
    else:
        for key, value in report.items():
            print(f"{key:>14}: {value}")


if __name__ == "__main__":
    main()
