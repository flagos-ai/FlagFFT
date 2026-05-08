from __future__ import annotations

import json
import sqlite3

import pytest

import flagfft
from src import tuning
from src.tuning import TuneConfig, _connect, _existing_winner, _insert_measurement, _mark_superseded


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


def test_cli_parses_command_line_config(tmp_path) -> None:
    db = tmp_path / "plans.sqlite"

    [config] = tuning.configs_from_argv(
        [
            "--api",
            "fft",
            "--lengths",
            "16",
            "32",
            "--batch",
            "4",
            "--dtype",
            "float32",
            "--device",
            "cuda:0",
            "--dim",
            "-1",
            "--norm",
            "ortho",
            "--db",
            str(db),
            "--retune",
            "--dry-run",
            "--static-limit",
            "4",
            "--finalists",
            "2",
            "--warmup",
            "1",
            "--iters",
            "3",
        ]
    )

    assert config == TuneConfig(
        api="fft",
        lengths=(16, 32),
        batch=4,
        dtype="float32",
        device="cuda:0",
        dim=-1,
        norm="ortho",
        db=db,
        retune=True,
        dry_run=True,
        static_limit=4,
        finalists=2,
        warmup=1,
        iters=3,
    )


def test_cli_parses_json_string_and_merges_cli_defaults(tmp_path) -> None:
    db = tmp_path / "json.sqlite"
    [config] = tuning.configs_from_argv(
        [
            "--batch",
            "7",
            "--json",
            json.dumps({"api": "fft", "lengths": [64], "dry_run": True, "db": str(db)}),
        ]
    )

    assert config.api == "fft"
    assert config.lengths == (64,)
    assert config.batch == 7
    assert config.dry_run is True
    assert config.db == db


def test_cli_parses_json_file_object_list(tmp_path) -> None:
    problems = tmp_path / "problems.json"
    problems.write_text(
        json.dumps(
            [
                {"api": "fft", "lengths": [8], "batch": 1},
                {"api": "ifft", "lengths": 16, "dry_run": True},
            ]
        )
    )

    configs = tuning.configs_from_argv(["--json-file", str(problems)])

    assert [config.api for config in configs] == ["fft", "ifft"]
    assert configs[0].lengths == (8,)
    assert configs[0].batch == 1
    assert configs[1].lengths == (16,)
    assert configs[1].dry_run is True


def test_cli_rejects_unknown_json_fields() -> None:
    with pytest.raises(ValueError, match="unknown tune JSON field"):
        tuning.configs_from_argv(["--json", json.dumps({"api": "fft", "unknown": 1})])


def test_run_config_dispatches_fft_lengths_without_python_fft_fallback(tmp_path, monkeypatch) -> None:
    calls: list[int] = []

    def fake_tune_length(config, conn, n):
        calls.append(n)

    monkeypatch.setattr(tuning, "_tune_fft_length", fake_tune_length)

    tuning.run_config(TuneConfig(api="fft", lengths=(8, 16), db=tmp_path / "tuned.sqlite"))

    assert calls == [8, 16]


def test_run_config_rejects_unimplemented_api_before_benchmark(tmp_path) -> None:
    with pytest.raises(NotImplementedError, match="flagfft.fft only"):
        tuning.run_config(TuneConfig(api="ifft", lengths=(8,), db=tmp_path / "tuned.sqlite"))


def test_tune_accepts_flagfft_api_callable(tmp_path, monkeypatch) -> None:
    captured: list[TuneConfig] = []
    monkeypatch.setattr(tuning, "run_config", lambda config: captured.append(config))

    tuning.tune(flagfft.fft, lengths=8, dry_run=True, db=tmp_path / "tuned.sqlite")

    assert captured == [
        TuneConfig(api="fft", lengths=(8,), dry_run=True, db=tmp_path / "tuned.sqlite")
    ]
