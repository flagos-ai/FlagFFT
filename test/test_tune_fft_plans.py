from __future__ import annotations

import sqlite3

from benchmark.tune_fft_plans import _connect, _existing_winner, _insert_measurement, _mark_superseded


def _fields() -> dict[str, object]:
    return {
        "schema_version": 1,
        "device_arch": "sm_80",
        "fft_length": 4096,
        "batch_bucket": "65-512",
        "batch": 256,
        "dtype": "complex64",
        "direction": "forward",
        "norm": "backward",
        "input_layout": "contiguous",
        "planner_fingerprint": "planner",
        "codegen_fingerprint": "codegen",
        "runtime_fingerprint": "runtime",
        "benchmark_fingerprint": "benchmark",
    }


def _plan(repr_value: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "source": "cpp_tune_candidate",
        "plan_key": {"repr": repr_value},
        "root": {"kind": "ct_leaf", "length": 4096, "factors": [16, 16, 16]},
    }


def test_retune_supersedes_existing_winner(tmp_path) -> None:
    db = tmp_path / "tuned.sqlite"
    fields = _fields()
    with _connect(db) as conn:
        _insert_measurement(conn, fields, _plan("old"), status="valid", rank=0, median_ms=1.0)
        assert _existing_winner(conn, fields)["plan_key"] == "old"

        _mark_superseded(conn, fields)
        _insert_measurement(conn, fields, _plan("new"), status="valid", rank=0, median_ms=0.8)
        conn.commit()

        assert _existing_winner(conn, fields)["plan_key"] == "new"
        statuses = dict(conn.execute("SELECT plan_key, status FROM tuned_measurements").fetchall())
        assert statuses == {"old": "superseded", "new": "valid"}


def test_fingerprint_mismatch_does_not_return_winner(tmp_path) -> None:
    db = tmp_path / "tuned.sqlite"
    fields = _fields()
    with _connect(db) as conn:
        _insert_measurement(conn, fields, _plan("winner"), status="valid", rank=0, median_ms=1.0)
        stale_fields = {**fields, "codegen_fingerprint": "changed"}

        assert _existing_winner(conn, stale_fields) is None
