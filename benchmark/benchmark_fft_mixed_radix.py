from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flagfft import fft as flagfft_fft


def _bench_warm_wall_ms(fn, warmup: int, iters: int) -> float:
    if iters <= 0:
        raise ValueError("iters must be positive")

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def _bench_one_cuda_ms(fn) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3


def _require_cpp_core():
    try:
        import _flagfft_core
    except ImportError as exc:
        raise RuntimeError(
            "C++ benchmark backend requires _flagfft_core; run `python -m pip install -e .` first"
        ) from exc
    return _flagfft_core


def _format_plan_node(node: dict, indent: int = 0) -> str:
    prefix = "  " * indent
    kind = node.get("kind", "unknown")
    length = node.get("length", "?")
    if kind == "ct_leaf":
        label = f"{kind}(n={length}, factors={node.get('factors')}, lanes={node.get('lanes')})"
    elif kind == "four_step":
        label = f"{kind}(n={length}, n1={node.get('n1')}, n2={node.get('n2')})"
    else:
        label = f"{kind}(n={length})"
    lines = [prefix + label]
    if kind == "four_step":
        lines.append(_format_plan_node(node["row"], indent + 1))
        lines.append(_format_plan_node(node["col"], indent + 1))
    return "\n".join(lines)


def _profile_flagfft_cpp_warm_ms(x: torch.Tensor, warmup: int, iters: int) -> dict[str, Any]:
    core = _require_cpp_core()
    core.clear_plan_cache()
    plan = core.debug_plan(x)

    first_call_ms = _bench_one_cuda_ms(lambda: flagfft_fft(x))
    after_first = dict(core.cache_info())

    for _ in range(warmup):
        flagfft_fft(x)
    torch.cuda.synchronize()

    warm_median_ms = _bench_warm_wall_ms(lambda: flagfft_fft(x), warmup=0, iters=iters)
    after_warm = dict(core.cache_info())

    return {
        "resolved_plan": plan,
        "flagfft_median_ms": warm_median_ms,
        "flagfft_first_call_ms": first_call_ms,
        "cpp_cache_after_first": after_first,
        "cpp_cache_after_warm": after_warm,
    }


def benchmark_mixed_radix_once(n: int, batch: int = 64, warmup: int = 20, iters: int = 100) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    torch.manual_seed(n)
    xr = torch.randn(batch, n, device="cuda", dtype=torch.float32)
    xi = torch.randn(batch, n, device="cuda", dtype=torch.float32)
    x = torch.complex(xr, xi)

    flagfft_stats = _profile_flagfft_cpp_warm_ms(x=x, warmup=warmup, iters=iters)
    torch_median_ms = _bench_warm_wall_ms(
        lambda: torch.fft.fft(x, dim=-1), warmup=warmup, iters=iters
    )
    resolved_plan = flagfft_stats["resolved_plan"]

    return {
        "length": resolved_plan["root"]["length"],
        "batch": batch,
        "backend": "cpp",
        "plan_source": "cpp_auto",
        "plan": _format_plan_node(resolved_plan["root"]),
        "flagfft_median_ms": flagfft_stats["flagfft_median_ms"],
        "torch_median_ms": torch_median_ms,
        **{
            key: value
            for key, value in flagfft_stats.items()
            if key not in {"resolved_plan", "flagfft_median_ms"}
        },
    }


def run_benchmark(lengths: list[int], batch: int, warmup: int, iters: int) -> None:
    for n in lengths:
        stats = benchmark_mixed_radix_once(n=n, batch=batch, warmup=warmup, iters=iters)
        print(f"[backend={stats['backend']} mode={stats['plan_source']} n={n} batch={batch}]")
        print("  plan:")
        for line in stats["plan"].splitlines():
            print(f"    {line}")
        print(
            f"  warm: flagfft_median_ms={stats['flagfft_median_ms']:.4f} "
            f"torch_median_ms={stats['torch_median_ms']:.4f} "
            f"speedup={stats['torch_median_ms']/stats['flagfft_median_ms']:.2f}x"
        )
        cache = stats["cpp_cache_after_warm"]
        print(
            f"  cpp: first_call_ms={stats['flagfft_first_call_ms']:.4f} "
            f"cache_size={cache['size']} hits={cache['hits']} misses={cache['misses']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark FFT mixed-radix execution through the C++ backend.")
    parser.add_argument("--lengths", type=int, nargs="+", default=[34, 64, 105, 1024, 936, 4096, 8192])
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()
    run_benchmark(args.lengths, args.batch, args.warmup, args.iters)


if __name__ == "__main__":
    main()
