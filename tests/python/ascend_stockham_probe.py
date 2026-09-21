"""Manual CANN 9 Stockham stage qualification; run under an external timeout.

Example: python ascend_stockham_probe.py --radix 13 --length 46189 --span 1
"""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--radix", type=int, required=True)
    parser.add_argument("--length", type=int, default=46189)
    parser.add_argument("--span", type=int, default=1)
    parser.add_argument("--inverse", action="store_true")
    parser.add_argument("--specialize-span", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    import torch
    import torch_npu  # noqa: F401
    import triton
    from flagfft_codegen.emit import emit_jit_kernel

    n, radix, span = args.length, args.radix, args.span
    assert n % radix == 0 and n % (radix * span) == 0
    rng = np.random.default_rng(20260921)
    host = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64)
    sign = 1 if args.inverse else -1
    twiddle = np.exp(sign * 2j * np.pi * np.arange(n) / n).astype(np.complex64)
    x = torch.from_numpy(host.view(np.float32)).npu()
    tw = torch.from_numpy(twiddle.view(np.float32)).npu()
    y = torch.empty_like(x)
    k = np.arange(n // radix)
    j = k % span
    samples = host.reshape(radix, -1) * twiddle[
        (np.arange(radix)[:, None] * j * (n // (radix * span))) % n
    ]
    transformed = np.fft.ifft(samples, axis=0) * radix if args.inverse else np.fft.fft(samples, axis=0)
    expected = np.empty(n, dtype=np.complex128)
    for digit in range(radix):
        expected[radix * k - (radix - 1) * j + digit * span] = transformed[digit]
    with tempfile.TemporaryDirectory(prefix="flagfft-stockham-") as tmp:
        factors = (radix, span) if args.specialize_span else (radix,)
        meta = emit_jit_kernel(kernel="stockham_stage", length=n, factors=factors,
            lanes=1, num_warps=4, generic_radices=(), smem_size=0,
            direction="inverse" if args.inverse else "forward", dtype="complex64",
            prime_n=0, four_step_n1=0, four_step_n2=0, out_dir=Path(tmp))
        spec = importlib.util.spec_from_file_location("stockham_probe", meta["module_path"])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        kernel = getattr(module, meta["kernel_name"])
        # Match the C++ raw signature: Python otherwise specializes nbatch=1
        # to a constexpr and the generated scalar .to() is no longer valid.
        kernel = triton.jit(kernel.fn, do_not_specialize=["span", "nbatch"])
        run = lambda: kernel[(triton.cdiv(n // radix, 128),)](x, y, tw, span, 1, num_warps=4)
        started = time.perf_counter()
        print(json.dumps({"phase": "compile", "radix": radix}), flush=True)
        if args.compile_only:
            kernel.warmup(x, y, tw, span, 1, grid=(triton.cdiv(n // radix, 128),), num_warps=4)
            print(json.dumps({**vars(args), "compile_s": time.perf_counter() - started}), flush=True)
            return
        run()
        torch.npu.synchronize()
        compile_s = time.perf_counter() - started
        actual = y.cpu().numpy().view(np.complex64)
        error = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
        assert error < 2e-6, error
        for _ in range(5):
            run()
        torch.npu.synchronize()
        timings = []
        for _ in range(30):
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            start.record()
            run()
            end.record()
            end.synchronize()
            timings.append(start.elapsed_time(end))
        print(json.dumps({**vars(args), "compile_s": compile_s,
            "relative_l2": float(error), "median_ms": float(np.median(timings))}), flush=True)


if __name__ == "__main__":
    main()
