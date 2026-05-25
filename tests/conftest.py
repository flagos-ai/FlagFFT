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
            assert result.returncode == 77
            pytest.skip(report.get("reason", "CLI skipped"))
        return result, report

    return invoke
