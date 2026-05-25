from __future__ import annotations

import sqlite3
import subprocess

import pytest


def test_help(flagfft_cli) -> None:
    result = subprocess.run(
        [str(flagfft_cli), "--help"], text=True, capture_output=True, check=False
    )
    assert result.returncode == 0
    assert "test --suite" in result.stdout
    assert "flagfft-cli bench" in result.stdout
    assert "flagfft-cli tune" in result.stdout
    assert "--launches-per-sample" in result.stdout


def test_plan_contract(invoke_cli) -> None:
    result, report = invoke_cli("test", "--suite", "plan")
    assert result.returncode == 0
    assert report["status"] == "passed"
    assert report["suites"][0]["name"] == "plan"


@pytest.mark.parametrize(
    "arguments",
    [
        ("test", "--suite", "correctness", "--shape", "16garbage"),
        ("test", "--suite", "correctness", "--shape", "16x4tail"),
        ("test", "--suite", "correctness", "--batch", "2garbage"),
        ("bench", "--shape", "16", "--iters", "1tail"),
    ],
)
def test_rejects_integer_options_with_trailing_characters(
    invoke_cli, arguments
) -> None:
    result, report = invoke_cli(*arguments)
    assert result.returncode == 1
    assert report["status"] == "failed"


def test_expressible_unsupported_case(invoke_cli, tmp_path) -> None:
    result, report = invoke_cli(
        "tune",
        "--api",
        "r2c",
        "--shape",
        "16",
        "--db",
        str(tmp_path / "tune.sqlite"),
        "--warmup",
        "0",
        "--iters",
        "1",
        "--static-limit",
        "1",
        "--finalists",
        "1",
    )
    assert result.returncode == 77
    assert report["status"] == "unsupported"


def test_unsupported_rank(invoke_cli) -> None:
    result, report = invoke_cli(
        "bench", "--api", "c2c", "--shape", "16x16", "--iters", "1"
    )
    assert result.returncode == 77
    assert report["status"] == "unsupported"


def test_api_error_contract(invoke_cli) -> None:
    result, report = invoke_cli("test", "--suite", "api-errors")
    assert result.returncode == 0
    assert report["suites"][0]["status"] == "passed"


@pytest.mark.parametrize(
    "arguments",
    [
        ("--api", "c2c", "--direction", "forward"),
        ("--api", "z2z", "--direction", "inverse", "--stream"),
        ("--api", "r2c", "--direction", "forward"),
        (
            "--api",
            "d2z",
            "--direction",
            "forward",
            "--placement",
            "in-place",
            "--plan-api",
            "planmany",
        ),
        (
            "--api",
            "c2r",
            "--direction",
            "inverse",
            "--placement",
            "in-place",
            "--plan-api",
            "planmany",
        ),
        ("--api", "z2d", "--direction", "inverse", "--placement", "in-place"),
    ],
)
def test_correctness_supported_apis(invoke_cli, arguments) -> None:
    result, report = invoke_cli(
        "test", "--suite", "correctness", "--shape", "16", "--batch", "2", *arguments
    )
    assert result.returncode == 0, report
    assert report["cases"][0]["correctness"]["passed"]
    assert report["cases"][0]["correctness"]["non_finite_values"] == 0


def test_correctness_native_routes(invoke_cli) -> None:
    result, report = invoke_cli(
        "test",
        "--suite",
        "correctness",
        "--api",
        "c2c",
        "--direction",
        "forward",
        "--shape",
        "331",
        "--shape",
        "8192",
        "--batch",
        "1",
    )
    assert result.returncode == 0, report
    assert all(case["correctness"]["passed"] for case in report["cases"])


def test_all_suite_builtin_collection(invoke_cli) -> None:
    result, report = invoke_cli("test", "--suite", "all")
    assert result.returncode == 0, report
    assert len(report["cases"]) == 8


@pytest.mark.parametrize(
    "api,direction",
    [
        ("c2c", "forward"),
        ("z2z", "inverse"),
        ("r2c", "forward"),
        ("d2z", "forward"),
        ("c2r", "inverse"),
        ("z2d", "inverse"),
    ],
)
def test_bench_smoke_apis(invoke_cli, api, direction) -> None:
    result, report = invoke_cli(
        "bench",
        "--api",
        api,
        "--direction",
        direction,
        "--shape",
        "16",
        "--batch",
        "1",
        "--warmup",
        "0",
        "--iters",
        "1",
        "--launches-per-sample",
        "1",
        "--print-path",
    )
    assert result.returncode == 0, report
    case = report["cases"][0]
    assert case["correctness"]["passed"]
    assert "speedup" in case["timing"]
    assert "FlagFFT Plan" in case["plan_description"]


def test_tune_retune_and_runtime_lookup(invoke_cli, tmp_path) -> None:
    db = tmp_path / "tuned_plans.sqlite"
    tune_args = (
        "tune",
        "--api",
        "c2c",
        "--shape",
        "16",
        "--batch",
        "1",
        "--db",
        str(db),
        "--warmup",
        "0",
        "--iters",
        "1",
        "--static-limit",
        "1",
        "--finalists",
        "1",
    )
    result, report = invoke_cli(*tune_args)
    assert result.returncode == 0, report
    result, report = invoke_cli(*tune_args, "--retune")
    assert result.returncode == 0, report
    with sqlite3.connect(db) as connection:
        count = connection.execute(
            "SELECT count(*) FROM tuned_measurements WHERE status='valid' AND rank=0"
        ).fetchone()[0]
    assert count == 1
    env = {"FLAGFFT_TUNE_DB": str(db)}
    result, report = invoke_cli(
        "bench",
        "--api",
        "c2c",
        "--shape",
        "16",
        "--batch",
        "1",
        "--warmup",
        "0",
        "--iters",
        "1",
        "--launches-per-sample",
        "1",
        "--print-path",
        env=env,
    )
    assert result.returncode == 0, report
    assert report["cases"][0]["correctness"]["passed"]


def test_tune_inverse_smoke(invoke_cli, tmp_path) -> None:
    result, report = invoke_cli(
        "tune",
        "--api",
        "c2c",
        "--direction",
        "inverse",
        "--shape",
        "16",
        "--db",
        str(tmp_path / "inverse.sqlite"),
        "--warmup",
        "0",
        "--iters",
        "1",
        "--static-limit",
        "1",
        "--finalists",
        "1",
    )
    assert result.returncode == 0, report
    assert report["cases"][0]["winner"]["max_abs_error"] >= 0
