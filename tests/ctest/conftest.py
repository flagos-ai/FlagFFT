from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def pytest_addoption(parser):
    parser.addoption(
        "--ctest-build-dir",
        default=None,
        help="Path to ctest build directory. Defaults to build/ctest.",
    )


@pytest.fixture(scope="session")
def ctest_build_dir(request) -> Path:
    configured = request.config.getoption("--ctest-build-dir")
    path = Path(configured or ROOT / "build" / "ctest")
    if not path.is_dir():
        pytest.skip(f"ctest build directory not found: {path}")
    return path


@pytest.fixture
def run_ctest(ctest_build_dir: Path):
    def run(target: str, *, timeout: int = 300) -> subprocess.CompletedProcess:
        exe = ctest_build_dir / target
        if not exe.is_file():
            pytest.skip(f"ctest executable not found: {exe}")
        env = os.environ.copy()
        result = subprocess.run(
            [str(exe)],
            cwd=ctest_build_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
        return result

    return run
