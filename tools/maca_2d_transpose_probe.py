#!/usr/bin/env python3
"""Correctness and paired event timings of generated complex64 transposes.

Run on one visible MACA GPU. The caller redirects JSON-lines to a results
directory. Kernels are generated in memory from the production builders.
"""
from __future__ import annotations

import argparse
import json
import linecache
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", default="2048x2048,2048x1025,1025x2048")
    parser.add_argument("--tiles", default="32")
    parser.add_argument("--warps", default="4")
    parser.add_argument("--variants", default="legacy,register,packed")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    import torch
    import triton
    from flagfft_codegen.kernels_layout import (
        _build_packed_transpose_kernel_source,
        _build_tiled_transpose_kernel_source,
    )
    from flagfft_codegen.metadata import _module_source

    assert torch.cuda.device_count() == 1
    assert triton.runtime.driver.active.get_current_target().backend == "maca"
    torch.cuda.set_device(0)
    for shape in args.shapes.split(","):
        n0, n1 = map(int, shape.split("x"))
        # Integer payload covers arbitrary float bit patterns and checks that
        # NaNs, signed zero and infinities survive without numeric conversion.
        torch.manual_seed(20260922)
        x = torch.randint(-(2**31), 2**31 - 1, (args.batch, n0, n1, 2),
                          device="cuda", dtype=torch.int32)
        y = torch.empty((args.batch, n1, n0, 2), device="cuda", dtype=torch.int32)
        expected = x.permute(0, 2, 1, 3).contiguous()
        for tile in map(int, args.tiles.split(",")):
            for variant in args.variants.split(","):
                if variant == "packed":
                    name, source, _ = _build_packed_transpose_kernel_source(n0, n1, tile)
                else:
                    assert variant in {"legacy", "register"}
                    name, source, _ = _build_tiled_transpose_kernel_source(
                        n0, n1, "complex64", tile,
                        register_transpose=True if variant == "register" else None,
                    )
                module = _module_source(source)
                filename = f"<transpose_{shape}_{tile}_{variant}>"
                linecache.cache[filename] = (len(module), None, module.splitlines(True), filename)
                scope = {"__name__": "maca_2d_transpose_probe"}
                exec(compile(module, filename, "exec"), scope)
                grid = (math.ceil(n1 / tile), math.ceil(n0 / tile), args.batch)
                for warps in map(int, args.warps.split(",")):
                    def launch():
                        return scope[name][grid](x.view(torch.float32), y.view(torch.float32),
                                                 args.batch, num_warps=warps)
                    kernel = launch()
                    torch.cuda.synchronize()
                    assert torch.equal(y, expected), (shape, tile, variant, warps)
                    for _ in range(args.warmup):
                        launch()
                    torch.cuda.synchronize()
                    samples = []
                    for _ in range(args.iters):
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        launch()
                        end.record()
                        end.synchronize()
                        samples.append(start.elapsed_time(end))
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(args.iters):
                        launch()
                    end.record()
                    end.synchronize()
                    print(json.dumps({
                        "shape": shape, "batch": args.batch, "tile": tile,
                        "variant": variant, "warps": warps, "correct": True,
                        "shared": kernel.metadata.shared,
                        "median_ms": statistics.median(samples),
                        "queued_ms": start.elapsed_time(end) / args.iters,
                        "samples_ms": samples,
                    }), flush=True)


if __name__ == "__main__":
    main()
