from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flagfft import (
    FFTDecompositionSpec,
    FFTPlan,
    build_fft_plan,
    clear_fft_caches,
    collect_leaf_plans,
    describe_fft_plan,
    fft,
    fft_mixed_radix_triton,
    fft_mixed_radix_triton_manual,
    load_fft_plan,
    max_leaf_smem_bytes,
    plan_depth,
    prepare_fft_kernels,
    prepare_fft_tables,
    save_fft_plan,
    unique_leaf_plans,
)


def _parse_split_spec(raw: str | None) -> FFTDecompositionSpec | None:
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid --split-spec JSON: {exc}") from exc


def _bench_gpu_ms(fn, warmup: int, iters: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
    }


def _build_plan(
    n: int,
    split_spec: FFTDecompositionSpec | None,
    plan: FFTPlan | None,
) -> FFTPlan:
    if plan is not None:
        return build_fft_plan(n, plan)
    return build_fft_plan(n, split_spec=split_spec)


def benchmark_mixed_radix_once(
    n: int,
    batch: int = 64,
    warmup: int = 20,
    iters: int = 100,
    split_spec: FFTDecompositionSpec | None = None,
    plan: FFTPlan | None = None,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    torch.manual_seed(n)
    xr = torch.randn(batch, n, device="cuda", dtype=torch.float32)
    xi = torch.randn(batch, n, device="cuda", dtype=torch.float32)
    x = torch.complex(xr, xi)
    device = x.device

    clear_fft_caches()

    t0 = time.perf_counter()
    resolved_plan = _build_plan(n, split_spec, plan)
    plan_build_ms = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    for _ in range(1000):
        _build_plan(n, split_spec, plan)
    plan_cached_us = (time.perf_counter() - t0) * 1e6 / 1000.0

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    prepare_fft_tables(device, resolved_plan)
    torch.cuda.synchronize()
    table_build_ms = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    for _ in range(200):
        prepare_fft_tables(device, resolved_plan)
    table_cached_us = (time.perf_counter() - t0) * 1e6 / 200.0

    t0 = time.perf_counter()
    prepare_fft_kernels(resolved_plan)
    kernel_build_ms = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    for _ in range(200):
        prepare_fft_kernels(resolved_plan)
    kernel_cached_us = (time.perf_counter() - t0) * 1e6 / 200.0

    clear_fft_caches()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if plan is not None:
        y = fft_mixed_radix_triton(x, plan=resolved_plan)
    elif split_spec is None:
        y = fft(x)
    else:
        y = fft_mixed_radix_triton_manual(x, split_spec=split_spec)
    torch.cuda.synchronize()
    cold_first_call_ms = (time.perf_counter() - t0) * 1e3

    resolved_plan = _build_plan(n, split_spec, plan)
    if plan is not None:
        flagfft_fn = lambda: fft_mixed_radix_triton(x, plan=resolved_plan)
    elif split_spec is None:
        flagfft_fn = lambda: fft(x)
    else:
        flagfft_fn = lambda: fft_mixed_radix_triton_manual(x, split_spec=split_spec)
    flagfft_stats = _bench_gpu_ms(
        flagfft_fn,
        warmup=warmup,
        iters=iters,
    )
    torch_stats = _bench_gpu_ms(lambda: torch.fft.fft(x, dim=-1), warmup=warmup, iters=iters)
    ref = torch.fft.fft(x, dim=-1)
    err = (y - ref).abs()
    leaf_lengths = tuple(leaf.length for leaf in unique_leaf_plans(resolved_plan))

    return {
        "length": resolved_plan.length,
        "batch": batch,
        "plan_source": "json" if plan is not None else ("manual" if split_spec is not None else "auto"),
        "plan": describe_fft_plan(resolved_plan),
        "resolved_plan": resolved_plan,
        "plan_depth": plan_depth(resolved_plan),
        "leaf_lengths": leaf_lengths,
        "leaf_count": len(collect_leaf_plans(resolved_plan)),
        "max_leaf_smem_bytes": max_leaf_smem_bytes(resolved_plan),
        "plan_build_ms": plan_build_ms,
        "plan_cached_us": plan_cached_us,
        "table_build_ms": table_build_ms,
        "table_cached_us": table_cached_us,
        "kernel_build_ms": kernel_build_ms,
        "kernel_cached_us": kernel_cached_us,
        "cold_first_call_ms": cold_first_call_ms,
        "flagfft_mean_ms": flagfft_stats["mean_ms"],
        "flagfft_median_ms": flagfft_stats["median_ms"],
        "flagfft_min_ms": flagfft_stats["min_ms"],
        "torch_mean_ms": torch_stats["mean_ms"],
        "torch_median_ms": torch_stats["median_ms"],
        "torch_min_ms": torch_stats["min_ms"],
        "speedup_vs_torch": torch_stats["mean_ms"] / flagfft_stats["mean_ms"],
        "cold_over_warm": cold_first_call_ms / flagfft_stats["median_ms"],
        "max_abs_err": float(err.max()),
        "rms_err": float(err.square().mean().sqrt()),
    }


def run_benchmark(
    lengths: list[int],
    batch: int,
    warmup: int,
    iters: int,
    split_spec: FFTDecompositionSpec | None = None,
    plan_json: Path | None = None,
    dump_plan: Path | None = None,
) -> None:
    loaded_plan = load_fft_plan(plan_json) if plan_json is not None else None
    for n in lengths:
        stats = benchmark_mixed_radix_once(
            n=n,
            batch=batch,
            warmup=warmup,
            iters=iters,
            split_spec=split_spec,
            plan=loaded_plan,
        )
        if dump_plan is not None:
            if len(lengths) == 1:
                dump_plan.parent.mkdir(parents=True, exist_ok=True)
                save_fft_plan(stats["resolved_plan"], dump_plan)
            else:
                dump_plan.mkdir(parents=True, exist_ok=True)
                save_fft_plan(stats["resolved_plan"], dump_plan / f"fft_plan_{n}.json")
        print(
            f"[mode={stats['plan_source']} n={n} plan={stats['plan']} "
            f"max_leaf_smem={stats['max_leaf_smem_bytes']}B]"
        )
        print(
            f"  cold: plan={stats['plan_build_ms']:.3f} ms "
            f"table={stats['table_build_ms']:.3f} ms "
            f"kernel={stats['kernel_build_ms']:.3f} ms "
            f"first_call={stats['cold_first_call_ms']:.3f} ms"
        )
        print(
            f"  shape: depth={stats['plan_depth']} "
            f"leaf_lengths={stats['leaf_lengths']} "
            f"leaf_count={stats['leaf_count']}"
        )
        print(
            f"  cache: plan={stats['plan_cached_us']:.3f} us "
            f"table={stats['table_cached_us']:.3f} us "
            f"kernel={stats['kernel_cached_us']:.3f} us"
        )
        print(
            f"  warm: flagfft_mean={stats['flagfft_mean_ms']:.4f} ms "
            f"flagfft_median={stats['flagfft_median_ms']:.4f} ms "
            f"torch_mean={stats['torch_mean_ms']:.4f} ms "
            f"speedup={stats['speedup_vs_torch']:.2f}x"
        )
        print(
            f"  err: max={stats['max_abs_err']:.6e} "
            f"rms={stats['rms_err']:.6e} "
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark FFT mixed-radix execution. "
            "--split-spec accepts JSON such as '[32, 3200, null, [32, 100]]' "
            "or '{\"split\": [32, 3200], \"col\": [32, 100]}'. "
            "--plan-json loads an exact executable plan produced by --dump-plan."
        )
    )
    parser.add_argument("--lengths", type=int, nargs="+", default=[34, 105, 936, 4096, 8192])
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--split-spec", type=str, default=None)
    parser.add_argument("--plan-json", type=Path, default=None)
    parser.add_argument("--dump-plan", type=Path, default=None)
    args = parser.parse_args()
    if args.split_spec is not None and args.plan_json is not None:
        parser.error("pass either --split-spec or --plan-json, not both")

    split_spec = _parse_split_spec(args.split_spec)
    run_benchmark(
        args.lengths,
        args.batch,
        args.warmup,
        args.iters,
        split_spec=split_spec,
        plan_json=args.plan_json,
        dump_plan=args.dump_plan,
    )


if __name__ == "__main__":
    main()
