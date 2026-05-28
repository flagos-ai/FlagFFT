from __future__ import annotations

import json
import subprocess

import pytest


def test_help(flagfft_cli) -> None:
    result = subprocess.run(
        [str(flagfft_cli), "--help"], text=True, capture_output=True, check=False
    )
    assert result.returncode == 0
    assert "flagfft-cli bench" in result.stdout
    assert "flagfft-cli tune" in result.stdout
    assert "--rank" in result.stdout
    assert "--shape" in result.stdout
    assert "test --suite" not in result.stdout
    assert "--plan-api" not in result.stdout
    assert "--launches-per-sample" not in result.stdout


@pytest.mark.parametrize(
    "arguments",
    [
        ("bench", "--rank", "1", "--shape", "16garbage"),
        ("bench", "--rank", "2", "--shape", "16x4tail"),
        ("bench", "--rank", "1", "--shape", "16", "--batch", "2garbage"),
        ("bench", "--rank", "1", "--shape", "16", "--iters", "1tail"),
    ],
)
def test_rejects_integer_options_with_trailing_characters(
    flagfft_cli, arguments
) -> None:
    result = subprocess.run(
        [str(flagfft_cli), *arguments],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1
    assert "invalid" in result.stderr.lower()


def test_bench_basic_table(flagfft_cli) -> None:
    result = subprocess.run(
        [
            str(flagfft_cli),
            "bench",
            "--rank",
            "1",
            "--shape",
            "16",
            "--warmup",
            "2",
            "--iters",
            "5",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert "speedup" in result.stdout
    assert "flagfft_median_ms" in result.stdout
    assert "ref_median_ms" in result.stdout
    assert "warmup" in result.stdout
    assert "iters" in result.stdout


def test_bench_json(invoke_cli) -> None:
    result, report = invoke_cli(
        "bench",
        "--rank",
        "1",
        "--shape",
        "16",
        "--warmup",
        "2",
        "--iters",
        "5",
    )
    assert result.returncode == 0
    assert report["status"] == "passed"
    assert report["command"] == "bench"
    assert len(report["cases"]) == 1
    case = report["cases"][0]
    assert "timing" in case
    assert "speedup" in case["timing"]
    assert "flagfft_median_ms" in case["timing"]
    assert "ref_median_ms" in case["timing"]


def test_bench_multi_shape(flagfft_cli) -> None:
    result = subprocess.run(
        [
            str(flagfft_cli),
            "bench",
            "--rank",
            "1",
            "--shape",
            "16,32",
            "--warmup",
            "2",
            "--iters",
            "5",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    lines = result.stdout.strip().split("\n")
    assert len(lines) >= 3  # header + 2 data rows


def test_bench_json_multi_shape(invoke_cli) -> None:
    result, report = invoke_cli(
        "bench",
        "--rank",
        "1",
        "--shape",
        "16,32",
        "--warmup",
        "2",
        "--iters",
        "5",
    )
    assert result.returncode == 0
    assert len(report["cases"]) == 2


def test_bench_print_path(flagfft_cli) -> None:
    result = subprocess.run(
        [
            str(flagfft_cli),
            "bench",
            "--rank",
            "1",
            "--shape",
            "16",
            "--warmup",
            "2",
            "--iters",
            "5",
            "--json",
            "--print-path",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    report = json.loads(result.stdout)
    assert "FlagFFT Plan" in report["cases"][0]["plan_description"]


def test_tune_placeholder(flagfft_cli) -> None:
    result = subprocess.run(
        [str(flagfft_cli), "tune"], text=True, capture_output=True, check=False
    )
    assert result.returncode == 1
    assert "not yet supported" in result.stderr.lower()


def test_batch_only_rank1(flagfft_cli) -> None:
    result = subprocess.run(
        [str(flagfft_cli), "bench", "--rank", "2", "--shape", "16x16", "--batch", "4"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1
    assert "batch" in result.stderr.lower()


def test_rank_shape_mismatch(flagfft_cli) -> None:
    result = subprocess.run(
        [str(flagfft_cli), "bench", "--rank", "1", "--shape", "16x16"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1
    assert "dimension" in result.stderr.lower() or "rank" in result.stderr.lower()


def test_unknown_command(flagfft_cli) -> None:
    result = subprocess.run(
        [str(flagfft_cli), "test"], text=True, capture_output=True, check=False
    )
    assert result.returncode == 1
    assert "unknown command" in result.stderr.lower()
