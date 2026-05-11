from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import torch

DEFAULT_TUNE_DB = Path(".flagfft/tuned_plans.sqlite")

FFT_API_NAMES = (
    "fft",
    "ifft",
    "fft2",
    "ifft2",
    "fftn",
    "ifftn",
    "rfft",
    "irfft",
    "rfft2",
    "irfft2",
    "rfftn",
    "irfftn",
    "hfft",
    "ihfft",
    "hfft2",
    "ihfft2",
    "hfftn",
    "ihfftn",
    "fftfreq",
    "rfftfreq",
    "fftshift",
    "ifftshift",
)
IMPLEMENTED_TUNE_APIS = {"fft", "ifft"}


@dataclass(frozen=True)
class TuneConfig:
    api: str = "fft"
    lengths: tuple[int, ...] = (4096,)
    batch: int = 256
    dtype: str = "complex64"
    device: str = "cuda"
    dim: int = -1
    norm: str | None = None
    db: Path = DEFAULT_TUNE_DB
    retune: bool = False
    dry_run: bool = False
    explain_cache: bool = False
    static_limit: int = 32
    finalists: int = 3
    warmup: int = 5
    iters: int = 200


def _require_core():
    try:
        import _flagfft_core
    except ImportError as exc:  # pragma: no cover - depends on extension build state
        raise RuntimeError("run `python -m pip install -e .` before tuning") from exc
    return _flagfft_core


def _api_name(api: str | Callable[..., Any]) -> str:
    name = api if isinstance(api, str) else getattr(api, "__name__", None)
    if not isinstance(name, str):
        raise TypeError("flagfft.tune expected an API function or API name")
    if name not in FFT_API_NAMES:
        raise ValueError(f"unknown flagfft API for tuning: {name}")
    return name


def _ensure_implemented_api(api_name: str) -> None:
    if api_name not in IMPLEMENTED_TUNE_APIS:
        raise NotImplementedError(
            f"flagfft.tune currently supports flagfft.fft and flagfft.ifft only, got flagfft.{api_name}"
        )


def _direction_for_api(api_name: str) -> str:
    return "inverse" if api_name == "ifft" else "forward"


def _torch_fft_api(api_name: str) -> Callable[..., torch.Tensor]:
    if api_name == "ifft":
        return torch.fft.ifft
    return torch.fft.fft


def _core_with_plan(core: Any, api_name: str) -> Callable[..., torch.Tensor]:
    return core.ifft_with_plan if api_name == "ifft" else core.fft_with_plan


def _normalize_dtype(dtype: str | torch.dtype) -> str:
    if isinstance(dtype, torch.dtype):
        dtype = str(dtype)
    return str(dtype).removeprefix("torch.")


def _torch_dtype(dtype: str) -> torch.dtype:
    if dtype == "complex64":
        return torch.complex64
    if dtype == "float32":
        return torch.float32
    raise NotImplementedError(f"flagfft.tune currently supports complex64 and float32 fft inputs, got {dtype}")


def _normalize_norm(norm: str | None) -> str | None:
    if norm in {None, ""}:
        return None
    value = str(norm)
    if value in {"none", "null", "None"}:
        return None
    if value not in {"backward", "forward", "ortho"}:
        raise ValueError(f"unsupported FFT norm: {value}")
    return value


def _normalize_lengths(lengths: int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(lengths, int):
        values = (lengths,)
    else:
        values = tuple(int(length) for length in lengths)
    if not values:
        raise ValueError("at least one FFT length is required")
    if any(length <= 0 for length in values):
        raise ValueError(f"FFT lengths must be positive, got {values}")
    return values


def _validate_config(config: TuneConfig) -> TuneConfig:
    api_name = _api_name(config.api)
    normalized = replace(
        config,
        api=api_name,
        lengths=_normalize_lengths(config.lengths),
        batch=int(config.batch),
        dtype=_normalize_dtype(config.dtype),
        device=str(config.device),
        dim=int(config.dim),
        norm=_normalize_norm(config.norm),
        db=Path(config.db),
        static_limit=int(config.static_limit),
        finalists=int(config.finalists),
        warmup=int(config.warmup),
        iters=int(config.iters),
    )
    if normalized.batch <= 0:
        raise ValueError(f"batch must be positive, got {normalized.batch}")
    if normalized.static_limit <= 0:
        raise ValueError(f"static_limit must be positive, got {normalized.static_limit}")
    if normalized.finalists <= 0:
        raise ValueError(f"finalists must be positive, got {normalized.finalists}")
    if normalized.warmup < 0:
        raise ValueError(f"warmup must be non-negative, got {normalized.warmup}")
    if normalized.iters <= 0:
        raise ValueError(f"iters must be positive, got {normalized.iters}")
    return normalized


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


def _bench_ms(fn: Callable[[], Any], warmup: int, iters: int) -> tuple[float, float]:
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


def _problem_fields(core: Any, x: torch.Tensor, config: TuneConfig) -> dict[str, Any]:
    direction = _direction_for_api(config.api)
    request = dict(core.debug_request(x, None, config.dim, config.norm, direction))
    plan_schema_version = int(
        core.debug_plan(x, None, config.dim, config.norm, direction)["schema_version"]
    )
    fps = dict(core.tune_fingerprints())
    return {
        "schema_version": plan_schema_version,
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


def _make_fft_input(config: TuneConfig, n: int) -> torch.Tensor:
    if config.dim not in {-1, 1}:
        raise NotImplementedError("flagfft.tune currently supports fft on the last dimension only")
    device = torch.device(config.device)
    if device.type != "cuda":
        raise NotImplementedError(f"flagfft.tune currently benchmarks CUDA tensors only, got {config.device}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    torch.manual_seed(n)
    dtype = _torch_dtype(config.dtype)
    if dtype is torch.complex64:
        real = torch.randn(config.batch, n, device=device, dtype=torch.float32)
        imag = torch.randn(config.batch, n, device=device, dtype=torch.float32)
        return torch.complex(real, imag)
    return torch.randn(config.batch, n, device=device, dtype=dtype)


def _measure_plan(
    core: Any,
    x: torch.Tensor,
    ref: torch.Tensor,
    plan: dict[str, Any],
    config: TuneConfig,
) -> dict[str, Any]:
    core_api_with_plan = _core_with_plan(core, config.api)
    core.clear_plan_cache()
    t0 = time.perf_counter()
    y = core_api_with_plan(x, plan, None, config.dim, config.norm)
    torch.cuda.synchronize()
    first_call_ms = (time.perf_counter() - t0) * 1e3
    err = (y - ref).abs()
    max_abs_err = float(err.max())
    rms_err = float(err.square().mean().sqrt())
    if max_abs_err > 5e-3:
        raise RuntimeError(f"correctness failed: max_abs_err={max_abs_err:.6g}")
    median_ms, p90_ms = _bench_ms(
        lambda: core_api_with_plan(x, plan, None, config.dim, config.norm),
        warmup=config.warmup,
        iters=config.iters,
    )
    return {
        "first_call_ms": first_call_ms,
        "median_ms": median_ms,
        "p90_ms": p90_ms,
        "max_abs_err": max_abs_err,
        "rms_err": rms_err,
    }


def _tune_fft_length(config: TuneConfig, conn: sqlite3.Connection, n: int) -> None:
    core = _require_core()
    x = _make_fft_input(config, n)
    fields = _problem_fields(core, x, config)

    if config.explain_cache:
        winner = _existing_winner(conn, fields)
        if winner is None:
            print(f"n={n}: no valid tuned winner for current problem/fingerprints")
        else:
            print(f"n={n}: valid winner median_ms={winner['median_ms']} plan_key={winner['plan_key']}")
        return

    if _existing_winner(conn, fields) is not None and not config.retune:
        print(f"n={n}: existing valid winner found; pass --retune to overwrite")
        return

    direction = _direction_for_api(config.api)
    plans = [
        dict(plan)
        for plan in core.enumerate_plan_candidates(x, None, config.dim, config.norm, direction)
    ]
    print(f"n={n}: generated {len(plans)} candidate plans")
    if config.dry_run:
        for idx, plan in enumerate(plans):
            print(f"  [{idx}] {plan['plan_key']['repr']}")
        return

    ref = _torch_fft_api(config.api)(x, dim=config.dim, norm=config.norm)
    quick_results: list[tuple[dict[str, Any], dict[str, Any]]] = []
    quick_config = replace(config, warmup=1, iters=5)
    for plan in plans[: config.static_limit]:
        try:
            stats = _measure_plan(core, x, ref, plan, quick_config)
            _insert_measurement(conn, fields, plan, status="ok", rank=None, **stats)
            quick_results.append((plan, stats))
            print(f"  ok quick {stats['median_ms']:.4f} ms {plan['plan_key']['repr']}")
        except Exception as exc:
            _insert_measurement(conn, fields, plan, status="failed", rank=None, failure_reason=str(exc))
            print(f"  failed {plan['plan_key']['repr']}: {exc}")
        conn.commit()

    quick_results.sort(key=lambda item: (item[1]["median_ms"], item[1]["p90_ms"]))
    finalists = quick_results[: config.finalists]
    final_results: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for plan, _ in finalists:
        stats = _measure_plan(core, x, ref, plan, config)
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


def run_config(config: TuneConfig) -> None:
    config = _validate_config(config)
    _ensure_implemented_api(config.api)
    os.environ.setdefault("FLAGFFT_TUNE_DB", str(config.db))
    with _connect(config.db) as conn:
        for n in config.lengths:
            _tune_fft_length(config, conn, n)


def run_configs(configs: Sequence[TuneConfig]) -> None:
    for config in configs:
        run_config(config)


def tune(
    api: str | Callable[..., Any],
    *,
    lengths: int | Sequence[int] = (4096,),
    batch: int = 256,
    dtype: str | torch.dtype = "complex64",
    device: str = "cuda",
    dim: int = -1,
    norm: str | None = None,
    db: str | Path = DEFAULT_TUNE_DB,
    retune: bool = False,
    dry_run: bool = False,
    explain_cache: bool = False,
    static_limit: int = 32,
    finalists: int = 3,
    warmup: int = 5,
    iters: int = 200,
) -> None:
    config = TuneConfig(
        api=_api_name(api),
        lengths=_normalize_lengths(lengths),
        batch=batch,
        dtype=_normalize_dtype(dtype),
        device=device,
        dim=dim,
        norm=norm,
        db=Path(db),
        retune=retune,
        dry_run=dry_run,
        explain_cache=explain_cache,
        static_limit=static_limit,
        finalists=finalists,
        warmup=warmup,
        iters=iters,
    )
    run_config(config)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline FlagFFT plan tuner.")
    parser.add_argument("--api", choices=FFT_API_NAMES, default="fft")
    parser.add_argument("--lengths", type=int, nargs="+", default=[4096])
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--dtype", default="complex64")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dim", type=int, default=-1)
    parser.add_argument("--norm", choices=["backward", "forward", "ortho", "none", "null"], default=None)
    parser.add_argument("--db", type=Path, default=DEFAULT_TUNE_DB)
    parser.add_argument("--mode", choices=["quick"], default="quick")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Accepted for workflow compatibility; measurements are always appended.")
    parser.add_argument("--retune", action="store_true")
    parser.add_argument("--explain-cache", action="store_true")
    parser.add_argument("--static-limit", type=int, default=32)
    parser.add_argument("--finalists", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--json", dest="json_values", action="append", default=[], help="JSON problem object or object list.")
    parser.add_argument("--json-file", dest="json_files", type=Path, action="append", default=[], help="Path to a JSON problem object or object list.")
    return parser


def _base_config_from_args(args: argparse.Namespace) -> TuneConfig:
    return _validate_config(
        TuneConfig(
            api=args.api,
            lengths=tuple(args.lengths),
            batch=args.batch,
            dtype=args.dtype,
            device=args.device,
            dim=args.dim,
            norm=args.norm,
            db=args.db,
            retune=args.retune,
            dry_run=args.dry_run,
            explain_cache=args.explain_cache,
            static_limit=args.static_limit,
            finalists=args.finalists,
            warmup=args.warmup,
            iters=args.iters,
        )
    )


def _json_problem_objects(text: str) -> list[dict[str, Any]]:
    payload = json.loads(text)
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        return payload
    raise ValueError("tune JSON must be an object or a list of objects")


def _config_from_mapping(base: TuneConfig, mapping: dict[str, Any]) -> TuneConfig:
    allowed = {field.name for field in fields(TuneConfig)}
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"unknown tune JSON field(s): {', '.join(unknown)}")
    values = {field.name: getattr(base, field.name) for field in fields(TuneConfig)}
    values.update(mapping)
    values["lengths"] = _normalize_lengths(values["lengths"])
    values["db"] = Path(values["db"])
    return _validate_config(TuneConfig(**values))


def configs_from_argv(argv: Sequence[str] | None = None) -> list[TuneConfig]:
    args = _parser().parse_args(argv)
    base = _base_config_from_args(args)
    objects: list[dict[str, Any]] = []
    for value in args.json_values:
        objects.extend(_json_problem_objects(value))
    for path in args.json_files:
        objects.extend(_json_problem_objects(path.read_text()))
    if not objects:
        return [base]
    return [_config_from_mapping(base, obj) for obj in objects]


def main(argv: Sequence[str] | None = None) -> int:
    run_configs(configs_from_argv(argv))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
