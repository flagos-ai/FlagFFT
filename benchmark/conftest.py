from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def pytest_addoption(parser):
    parser.addoption(
        "--flagfft-cli",
        default=None,
        help="Path to flagfft-cli. Defaults to FLAGFFT_CLI_EXE or build/flagfft-cli.",
    )
    parser.addoption(
        "--bench-warmup",
        type=int,
        default=5,
        help="Number of warmup iterations (default: 5)",
    )
    parser.addoption(
        "--bench-iters",
        type=int,
        default=10,
        help="Number of benchmark iterations (default: 10)",
    )
    parser.addoption(
        "--bench-launches-per-sample",
        type=int,
        default=1,
        help="Launches per sample for bench (default: 1)",
    )


@pytest.fixture(scope="session")
def flagfft_cli(request) -> Path:
    configured = request.config.getoption("--flagfft-cli")
    path = Path(
        configured or os.environ.get("FLAGFFT_CLI_EXE", ROOT / "build" / "flagfft-cli")
    )
    if not path.exists():
        pytest.skip(f"flagfft-cli is not built: {path}")
    return path


@pytest.fixture
def invoke_cli(flagfft_cli):
    def invoke(*arguments: str, env: dict[str, str] | None = None, timeout: int = 240):
        process_env = os.environ.copy()
        if env:
            process_env.update(env)
        result = subprocess.run(
            [str(flagfft_cli), *arguments, "--json"],
            cwd=ROOT,
            env=process_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
        try:
            report = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            pytest.fail(f"invalid CLI JSON: {result.stdout}\n{result.stderr}\n{error}")
        if report.get("status") == "skipped":
            pytest.skip(report.get("reason", "CLI skipped"))
        return result, report

    return invoke


@pytest.fixture
def run_benchmark(invoke_cli, bench_warmup, bench_iters, bench_launches_per_sample):
    """Invoke the bench subcommand with standard options."""

    def _run(size: int, api: str = "c2c", direction: str = "forward"):
        return invoke_cli(
            "bench",
            "--api",
            api,
            "--direction",
            direction,
            "--shape",
            str(size),
            "--batch",
            "1",
            "--warmup",
            str(bench_warmup),
            "--iters",
            str(bench_iters),
            "--launches-per-sample",
            str(bench_launches_per_sample),
            "--print-path",
        )

    return _run


@pytest.fixture(scope="session")
def bench_warmup(request) -> int:
    return request.config.getoption("--bench-warmup")


@pytest.fixture(scope="session")
def bench_iters(request) -> int:
    return request.config.getoption("--bench-iters")


@pytest.fixture(scope="session")
def bench_launches_per_sample(request) -> int:
    return request.config.getoption("--bench-launches-per-sample")
