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
    FFTPlanRequest,
    build_fft_plan,
    clear_fft_caches,
    format_plan_tree,
    load_fft_plan,
    save_fft_plan,
)
from src.exec import _execute_plan


def _parse_split_spec(raw: str | None) -> FFTDecompositionSpec | None:
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid --split-spec JSON: {exc}") from exc


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


def _build_plan(
    n: int,
    split_spec: FFTDecompositionSpec | None,
    plan: FFTPlan | None,
    request: FFTPlanRequest | None = None,
) -> FFTPlan:
    if plan is not None:
        return build_fft_plan(n, plan, request=request)
    return build_fft_plan(n, split_spec=split_spec, request=request)


def _make_plan_request(x: torch.Tensor, n: int) -> FFTPlanRequest:
    return FFTPlanRequest(
        length=n,
        dtype=str(x.dtype).removeprefix("torch."),
        device=x.device.type,
        batch=x.numel() // n,
    )


def _profile_flagfft_warm_ms(
    x: torch.Tensor,
    n: int,
    split_spec: FFTDecompositionSpec | None,
    plan: FFTPlan | None,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    if iters <= 0:
        raise ValueError("iters must be positive")

    request = _make_plan_request(x, n)
    resolved_plan = _build_plan(n, split_spec, plan, request=request)

    for _ in range(warmup):
        warm_plan = _build_plan(n, split_spec, plan, request=request)
        _execute_plan(x, warm_plan)
    torch.cuda.synchronize()

    plan_times: list[float] = []
    exec_times: list[float] = []
    total_times: list[float] = []
    for _ in range(iters):
        torch.cuda.synchronize()
        total_t0 = time.perf_counter()

        t0 = time.perf_counter()
        iter_plan = _build_plan(n, split_spec, plan, request=request)
        plan_ms = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        _execute_plan(x, iter_plan).reshape(x.shape)
        torch.cuda.synchronize()
        exec_ms = (time.perf_counter() - t0) * 1e3
        total_ms = (time.perf_counter() - total_t0) * 1e3

        plan_times.append(plan_ms)
        exec_times.append(exec_ms)
        total_times.append(total_ms)

    plan_median_ms = statistics.median(plan_times)
    exec_median_ms = statistics.median(exec_times)
    total_median_ms = statistics.median(total_times)
    profile_total_ms = plan_median_ms + exec_median_ms
    plan_pct = (plan_median_ms / profile_total_ms * 100.0) if profile_total_ms else 0.0
    exec_pct = (exec_median_ms / profile_total_ms * 100.0) if profile_total_ms else 0.0

    return {
        "resolved_plan": resolved_plan,
        "flagfft_median_ms": total_median_ms,
        "flagfft_plan_median_ms": plan_median_ms,
        "flagfft_exec_median_ms": exec_median_ms,
        "flagfft_plan_pct": plan_pct,
        "flagfft_exec_pct": exec_pct,
    }


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

    clear_fft_caches()

    flagfft_stats = _profile_flagfft_warm_ms(
        x=x,
        n=n,
        split_spec=split_spec,
        plan=plan,
        warmup=warmup,
        iters=iters,
    )
    torch_median_ms = _bench_warm_wall_ms(
        lambda: torch.fft.fft(x, dim=-1), warmup=warmup, iters=iters
    )
    resolved_plan = flagfft_stats["resolved_plan"]

    return {
        "length": resolved_plan.length,
        "batch": batch,
        "plan_source": "json" if plan is not None else ("manual" if split_spec is not None else "auto"),
        "plan": format_plan_tree(resolved_plan),
        "resolved_plan": resolved_plan,
        "flagfft_median_ms": flagfft_stats["flagfft_median_ms"],
        "torch_median_ms": torch_median_ms,
        "flagfft_plan_median_ms": flagfft_stats["flagfft_plan_median_ms"],
        "flagfft_exec_median_ms": flagfft_stats["flagfft_exec_median_ms"],
        "flagfft_plan_pct": flagfft_stats["flagfft_plan_pct"],
        "flagfft_exec_pct": flagfft_stats["flagfft_exec_pct"],
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
        print(f"[mode={stats['plan_source']} n={n} batch={batch}]")
        print("  plan:")
        for line in stats["plan"].splitlines():
            print(f"    {line}")
        print(
            f"  warm: flagfft_median_ms={stats['flagfft_median_ms']:.4f} "
            f"torch_median_ms={stats['torch_median_ms']:.4f} "
            f"speedup={stats['torch_median_ms']/stats['flagfft_median_ms']:.2f}x"
        )
        print(
            f"  profile: plan={stats['flagfft_plan_median_ms']:.4f} ms "
            f"({stats['flagfft_plan_pct']:.1f}%) "
            f"exec={stats['flagfft_exec_median_ms']:.4f} ms "
            f"({stats['flagfft_exec_pct']:.1f}%)"
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
    parser.add_argument("--lengths", type=int, nargs="+", default=[34, 64, 105, 1024, 936, 4096, 8192])
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=200)
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
