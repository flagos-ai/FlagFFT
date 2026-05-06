from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _require_core():
    try:
        import _flagfft_core
    except ImportError as exc:
        raise RuntimeError("run `python -m pip install -e .` before tuning") from exc
    return _flagfft_core


def _batch_bucket(batch: int) -> str:
    if batch <= 1:
        return "1"
    if batch <= 8:
        return "2-8"
    if batch <= 64:
        return "9-64"
    if batch <= 512:
        return "65-512"
    return "513+"


def _bench_ms(fn, warmup: int, iters: int) -> tuple[float, float]:
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
    return statistics.median(times), sorted(times)[max(0, int(0.9 * len(times)) - 1)]


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tuned_measurements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            schema_version INTEGER NOT NULL,
            device_arch TEXT NOT NULL,
            fft_length INTEGER NOT NULL,
            batch_bucket TEXT NOT NULL,
            batch INTEGER NOT NULL,
            dtype TEXT NOT NULL,
            direction TEXT NOT NULL,
            norm TEXT NOT NULL,
            input_layout TEXT NOT NULL,
            planner_fingerprint TEXT NOT NULL,
            codegen_fingerprint TEXT NOT NULL,
            runtime_fingerprint TEXT NOT NULL,
            benchmark_fingerprint TEXT NOT NULL,
            plan_key TEXT NOT NULL,
            plan_json TEXT NOT NULL,
            status TEXT NOT NULL,
            rank INTEGER,
            compile_ms REAL,
            first_call_ms REAL,
            median_ms REAL,
            p90_ms REAL,
            max_abs_err REAL,
            rms_err REAL,
            failure_reason TEXT,
            measured_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS tuned_measurements_lookup
        ON tuned_measurements (
            schema_version, status, rank, device_arch, fft_length, batch_bucket, dtype,
            direction, norm, input_layout, planner_fingerprint, codegen_fingerprint,
            runtime_fingerprint
        )
        """
    )
    return conn


def _problem_fields(core: Any, x: torch.Tensor) -> dict[str, Any]:
    request = dict(core.debug_request(x))
    fps = dict(core.tune_fingerprints())
    return {
        "schema_version": int(request.get("schema_version", 1)),
        "device_arch": request["device_arch"],
        "fft_length": int(request["requested_n"]),
        "batch_bucket": _batch_bucket(int(request["batch"])),
        "batch": int(request["batch"]),
        "dtype": request["input_dtype"],
        "direction": request["direction"],
        "norm": request["norm"],
        "input_layout": request["input_layout"],
        "planner_fingerprint": fps["planner"],
        "codegen_fingerprint": fps["codegen"],
        "runtime_fingerprint": fps["runtime"],
        "benchmark_fingerprint": fps["benchmark"],
    }


def _where_problem(fields: dict[str, Any], *, include_benchmark: bool = False) -> tuple[str, list[Any]]:
    keys = [
        "schema_version",
        "device_arch",
        "fft_length",
        "batch_bucket",
        "dtype",
        "direction",
        "norm",
        "input_layout",
        "planner_fingerprint",
        "codegen_fingerprint",
        "runtime_fingerprint",
    ]
    if include_benchmark:
        keys.append("benchmark_fingerprint")
    return " AND ".join(f"{key}=?" for key in keys), [fields[key] for key in keys]


def _existing_winner(conn: sqlite3.Connection, fields: dict[str, Any]) -> sqlite3.Row | None:
    conn.row_factory = sqlite3.Row
    where, values = _where_problem(fields)
    return conn.execute(
        f"SELECT * FROM tuned_measurements WHERE {where} AND status='valid' AND rank=0 "
        "ORDER BY measured_at DESC LIMIT 1",
        values,
    ).fetchone()


def _insert_measurement(
    conn: sqlite3.Connection,
    fields: dict[str, Any],
    plan: dict[str, Any],
    *,
    status: str,
    rank: int | None,
    compile_ms: float | None = None,
    first_call_ms: float | None = None,
    median_ms: float | None = None,
    p90_ms: float | None = None,
    max_abs_err: float | None = None,
    rms_err: float | None = None,
    failure_reason: str | None = None,
) -> None:
    plan_key = plan["plan_key"]["repr"]
    payload = {
        **fields,
        "plan_key": plan_key,
        "plan_json": json.dumps(plan, sort_keys=True),
        "status": status,
        "rank": rank,
        "compile_ms": compile_ms,
        "first_call_ms": first_call_ms,
        "median_ms": median_ms,
        "p90_ms": p90_ms,
        "max_abs_err": max_abs_err,
        "rms_err": rms_err,
        "failure_reason": failure_reason,
    }
    conn.execute(
        """
        INSERT INTO tuned_measurements (
            schema_version, device_arch, fft_length, batch_bucket, batch, dtype, direction,
            norm, input_layout, planner_fingerprint, codegen_fingerprint, runtime_fingerprint,
            benchmark_fingerprint, plan_key, plan_json, status, rank, compile_ms, first_call_ms,
            median_ms, p90_ms, max_abs_err, rms_err, failure_reason
        ) VALUES (
            :schema_version, :device_arch, :fft_length, :batch_bucket, :batch, :dtype,
            :direction, :norm, :input_layout, :planner_fingerprint, :codegen_fingerprint,
            :runtime_fingerprint, :benchmark_fingerprint, :plan_key, :plan_json, :status,
            :rank, :compile_ms, :first_call_ms, :median_ms, :p90_ms, :max_abs_err,
            :rms_err, :failure_reason
        )
        """,
        payload,
    )


def _mark_superseded(conn: sqlite3.Connection, fields: dict[str, Any]) -> None:
    where, values = _where_problem(fields)
    conn.execute(
        f"UPDATE tuned_measurements SET status='superseded' "
        f"WHERE {where} AND status='valid' AND rank=0",
        values,
    )


def _measure_plan(core: Any, x: torch.Tensor, ref: torch.Tensor, plan: dict[str, Any], warmup: int, iters: int) -> dict[str, Any]:
    core.clear_plan_cache()
    t0 = time.perf_counter()
    y = core.fft_with_plan(x, plan)
    torch.cuda.synchronize()
    first_call_ms = (time.perf_counter() - t0) * 1e3
    err = (y - ref).abs()
    max_abs_err = float(err.max())
    rms_err = float(err.square().mean().sqrt())
    if max_abs_err > 5e-3:
        raise RuntimeError(f"correctness failed: max_abs_err={max_abs_err:.6g}")
    median_ms, p90_ms = _bench_ms(lambda: core.fft_with_plan(x, plan), warmup=warmup, iters=iters)
    return {
        "first_call_ms": first_call_ms,
        "median_ms": median_ms,
        "p90_ms": p90_ms,
        "max_abs_err": max_abs_err,
        "rms_err": rms_err,
    }


def tune_length(args: argparse.Namespace, conn: sqlite3.Connection, n: int) -> None:
    core = _require_core()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch.manual_seed(n)
    real = torch.randn(args.batch, n, device="cuda", dtype=torch.float32)
    imag = torch.randn(args.batch, n, device="cuda", dtype=torch.float32)
    x = torch.complex(real, imag)
    fields = _problem_fields(core, x)

    if args.explain_cache:
        winner = _existing_winner(conn, fields)
        if winner is None:
            print(f"n={n}: no valid tuned winner for current problem/fingerprints")
        else:
            print(f"n={n}: valid winner median_ms={winner['median_ms']} plan_key={winner['plan_key']}")
        return

    if _existing_winner(conn, fields) is not None and not args.retune:
        print(f"n={n}: existing valid winner found; pass --retune to overwrite")
        return

    plans = [dict(plan) for plan in core.enumerate_plan_candidates(x)]
    print(f"n={n}: generated {len(plans)} candidate plans")
    if args.dry_run:
        for idx, plan in enumerate(plans):
            print(f"  [{idx}] {plan['plan_key']['repr']}")
        return

    ref = torch.fft.fft(x, dim=-1)
    quick_results: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for plan in plans[: args.static_limit]:
        try:
            stats = _measure_plan(core, x, ref, plan, warmup=1, iters=5)
            _insert_measurement(conn, fields, plan, status="ok", rank=None, **stats)
            quick_results.append((plan, stats))
            print(f"  ok quick {stats['median_ms']:.4f} ms {plan['plan_key']['repr']}")
        except Exception as exc:
            _insert_measurement(conn, fields, plan, status="failed", rank=None, failure_reason=str(exc))
            print(f"  failed {plan['plan_key']['repr']}: {exc}")
        conn.commit()

    quick_results.sort(key=lambda item: (item[1]["median_ms"], item[1]["p90_ms"]))
    finalists = quick_results[: args.finalists]
    final_results: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for plan, _ in finalists:
        stats = _measure_plan(core, x, ref, plan, warmup=args.warmup, iters=args.iters)
        final_results.append((plan, stats))
    final_results.sort(key=lambda item: (item[1]["median_ms"], item[1]["p90_ms"]))
    if not final_results:
        raise RuntimeError(f"n={n}: no valid tuned plan")

    _mark_superseded(conn, fields)
    for rank, (plan, stats) in enumerate(final_results):
        _insert_measurement(conn, fields, plan, status="valid" if rank == 0 else "ok", rank=rank, **stats)
    conn.commit()
    winner, stats = final_results[0]
    print(f"n={n}: winner median_ms={stats['median_ms']:.4f} p90_ms={stats['p90_ms']:.4f}")
    print(f"  {winner['plan_key']['repr']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline FlagFFT plan tuner.")
    parser.add_argument("--lengths", type=int, nargs="+", default=[4096])
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--db", type=Path, default=Path(".flagfft/tuned_plans.sqlite"))
    parser.add_argument("--mode", choices=["quick"], default="quick")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Accepted for workflow compatibility; measurements are always appended.")
    parser.add_argument("--retune", action="store_true")
    parser.add_argument("--explain-cache", action="store_true")
    parser.add_argument("--static-limit", type=int, default=32)
    parser.add_argument("--finalists", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    os.environ.setdefault("FLAGFFT_TUNE_DB", str(args.db))
    with _connect(args.db) as conn:
        for n in args.lengths:
            tune_length(args, conn, n)


if __name__ == "__main__":
    main()
