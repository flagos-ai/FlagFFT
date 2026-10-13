#!/usr/bin/env python3
"""Isolated MACA timing for the generated Rader boundary kernels.

This probes launch and memory access cost with a shuffled Rader index table.
The full acceptance runner remains the correctness and end-to-end gate.
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=int, required=True)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--dtype", choices=("complex64", "complex128"), default="complex64")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()
    if args.shape < 3 or args.batch < 1 or args.iters < 1:
        parser.error("shape >= 3, batch >= 1, and iters >= 1 are required")

    import torch
    from flagfft_codegen.emit import _rader_kernel_source
    from flagfft_codegen.metadata import _module_source
    from flagfft_codegen.target import set_codegen_target

    set_codegen_target("maca:80:64")
    torch.cuda.set_device(0)
    n, m = args.shape, args.shape - 1
    grid = (math.ceil(m / 256), args.batch)
    real_dtype = torch.float64 if args.dtype == "complex128" else torch.float32
    input_data = torch.ones((args.batch * n * 2,), device="cuda", dtype=real_dtype)
    fft_data = torch.ones((args.batch * m * 2,), device="cuda", dtype=real_dtype)
    work_data = torch.ones_like(fft_data)
    b_fft = torch.ones((m * 2,), device="cuda", dtype=real_dtype)
    output_data = torch.empty_like(input_data)
    indices = torch.randperm(m, device="cuda", dtype=torch.int32) + 1

    arguments = {
        "rader_prepare": (input_data, indices, work_data, n, m, args.batch),
        "rader_pointwise": (fft_data, b_fft, work_data, input_data, output_data,
                            n, m, args.batch),
        "rader_finalize": (input_data, fft_data, indices, output_data,
                           n, m, args.batch),
    }
    for kind, argv in arguments.items():
        name, source, _ = _rader_kernel_source(kind, n, m, args.dtype)
        module_source = _module_source(source)
        filename = f"<maca_rader_profile_{kind}>"
        import linecache
        linecache.cache[filename] = (len(module_source), None,
                                    module_source.splitlines(True), filename)
        scope = {"__name__": f"maca_rader_profile_{kind}"}
        exec(compile(module_source, filename, "exec"), scope)
        kernel = scope[name]

        def launch():
            kernel[grid](*argv, num_warps=4)

        launch()
        torch.cuda.synchronize()
        for _ in range(args.warmup):
            launch()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            launch()
        stop.record()
        torch.cuda.synchronize()
        print(json.dumps({
            "kernel": kind,
            "shape": n,
            "batch": args.batch,
            "dtype": args.dtype,
            "grid": list(grid),
            "ms": start.elapsed_time(stop) / args.iters,
            "correctness_checked": False,
        }), flush=True)


if __name__ == "__main__":
    main()
