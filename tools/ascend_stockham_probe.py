"""Pure Triton radix-2 feasibility probe; NOT a FlagFFT C API implementation."""

import argparse
import json
import time

import numpy as np
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

from ascend_plain_validation import signals


@triton.jit
def stockham_stage(src, dst, twiddle, n: tl.constexpr, batch, span,
                   block: tl.constexpr):
    index = tl.program_id(0) * block + tl.arange(0, block)
    valid = index < batch * (n // 2)
    row = index // (n // 2)
    k = index % (n // 2)
    j = k % span
    a = row * n + k
    b = a + n // 2
    tw = j * (n // (2 * span))
    ar = tl.load(src + 2 * a, valid, 0)
    ai = tl.load(src + 2 * a + 1, valid, 0)
    br = tl.load(src + 2 * b, valid, 0)
    bi = tl.load(src + 2 * b + 1, valid, 0)
    wr = tl.load(twiddle + 2 * tw, valid, 0)
    wi = tl.load(twiddle + 2 * tw + 1, valid, 0)
    cr = br * wr - bi * wi
    ci = br * wi + bi * wr
    out = row * n + 2 * k - j
    tl.store(dst + 2 * out, ar + cr, valid)
    tl.store(dst + 2 * out + 1, ai + ci, valid)
    tl.store(dst + 2 * (out + span), ar - cr, valid)
    tl.store(dst + 2 * (out + span) + 1, ai - ci, valid)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+",
                        default=[16, 128, 256, 1024, 4096, 65536])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--block", type=int, default=256)
    args = parser.parse_args()
    if any(n < 2 or n & (n - 1) for n in args.lengths):
        parser.error("lengths must be powers of two >= 2")
    if min(args.batches) < 1 or args.block < 1 or args.block & (args.block - 1):
        parser.error("positive batches and a power-of-two block are required")
    torch.npu.set_device(args.device)
    stream = torch.npu.Stream(device=args.device)
    count = 0
    worst_l2 = 0.0
    with torch.npu.stream(stream):
        for n in args.lengths:
            for batch in args.batches:
                buffers = [torch.empty((batch, n * 2), device="npu") for _ in range(2)]
                for direction in (-1, 1):
                    tw = np.exp(direction * 2j * np.pi * np.arange(n // 2) / n)
                    twiddle = torch.from_numpy(tw.astype(np.complex64).view(np.float32)).to("npu")
                    for pattern, source in signals(n, batch).items():
                        src = torch.from_numpy(source.view(np.float32).copy()).to("npu")
                        expected = (np.fft.fft(source.astype(np.complex128), axis=1)
                                    if direction == -1 else
                                    n * np.fft.ifft(source.astype(np.complex128), axis=1))
                        begin = time.perf_counter()
                        current = src
                        for stage in range(n.bit_length() - 1):
                            dst = buffers[stage % 2]
                            stockham_stage[(triton.cdiv(batch * (n // 2), args.block),)](
                                current, dst, twiddle, n, batch, 1 << stage, args.block)
                            current = dst
                        stream.synchronize()
                        elapsed = time.perf_counter() - begin
                        actual = current.cpu().numpy().copy().view(np.complex64)
                        error = np.abs(actual - expected)
                        l2 = float(np.linalg.norm(error) / max(np.linalg.norm(expected), 1e-30))
                        # Absolute allowance scales with the input norm; avoids an
                        # unstable relative test at near-zero Fourier coefficients.
                        allowance = 3e-5 * float(np.linalg.norm(source.astype(np.complex128), axis=1).max())
                        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=allowance)
                        assert np.isfinite(actual).all() and l2 < 3e-5
                        worst_l2 = max(worst_l2, l2)
                        count += 1
                        print(json.dumps(dict(n=n, batch=batch, direction=direction,
                                              pattern=pattern, max_abs=float(error.max()),
                                              relative_l2=l2, compile_and_run_seconds=elapsed,
                                              grid=triton.cdiv(batch * (n // 2), args.block))), flush=True)
    print(json.dumps(dict(status="PASS", checks=count, worst_relative_l2=worst_l2,
                          scope="Python Triton only; no C API or performance acceptance")), flush=True)


if __name__ == "__main__":
    main()
