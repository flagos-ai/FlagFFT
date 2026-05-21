from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _benchmark_exe(request) -> Path:
    configured = request.config.getoption("--benchmark-exe")
    if configured:
        return Path(configured)
    env_value = os.environ.get("FLAGFFT_BENCH_EXE")
    if env_value:
        return Path(env_value)
    return ROOT / "build" / "cpp" / "bench_vs_cufft"


def _require_benchmark_exe(request) -> Path:
    exe = _benchmark_exe(request)
    if not exe.exists():
        pytest.skip(f"bench_vs_cufft is not built: {exe}")
    return exe


def test_bench_vs_cufft_help(request) -> None:
    exe = _require_benchmark_exe(request)
    result = subprocess.run(
        [str(exe), "--help"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0
    assert "--tune" in result.stdout
    assert "--retune" in result.stdout
    assert "--tune-db" in result.stdout
    assert "--launches-per-sample" in result.stdout
    assert "--type" in result.stdout
    assert "--inplace" in result.stdout


def test_bench_vs_cufft_smoke(request, tmp_path) -> None:
    exe = _require_benchmark_exe(request)
    lengths = request.config.getoption("--benchmark-lengths")
    command = [
        str(exe),
        "--lengths",
        lengths,
        "--batch",
        str(request.config.getoption("--benchmark-batch")),
        "--warmup",
        str(request.config.getoption("--benchmark-warmup")),
        "--iters",
        str(request.config.getoption("--benchmark-iters")),
        "--launches-per-sample",
        str(request.config.getoption("--benchmark-launches-per-sample")),
        "--direction",
        request.config.getoption("--benchmark-direction"),
        "--tune-db",
        str(tmp_path / "tuned_plans.sqlite"),
        "--tune-static-limit",
        str(request.config.getoption("--benchmark-tune-static-limit")),
        "--tune-finalists",
        str(request.config.getoption("--benchmark-tune-finalists")),
    ]
    if request.config.getoption("--benchmark-retune"):
        command.append("--retune")
    elif request.config.getoption("--benchmark-tune"):
        command.append("--tune")

    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=180,
    )
    if result.returncode == 2 and "no CUDA device available" in result.stderr:
        pytest.skip("CUDA device required")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FlagFFT vs cuFFT benchmark" in result.stdout
    assert "Plan mode:" in result.stdout
    assert "flagfft_ms" in result.stdout
    assert "cufft_ms" in result.stdout


@pytest.mark.parametrize("extra", [["--type", "r2c"], ["--type", "c2r", "--inplace"]])
def test_bench_vs_cufft_real_type_smoke(request, tmp_path, extra) -> None:
    exe = _require_benchmark_exe(request)
    command = [
        str(exe),
        "--lengths",
        "16",
        "--batch",
        "1",
        "--warmup",
        "0",
        "--iters",
        "1",
        "--launches-per-sample",
        "1",
        "--tune-db",
        str(tmp_path / "tuned_plans.sqlite"),
        *extra,
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=180,
    )
    if result.returncode == 2 and "no CUDA device available" in result.stderr:
        pytest.skip("CUDA device required")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FlagFFT vs cuFFT benchmark" in result.stdout
