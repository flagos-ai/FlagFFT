from __future__ import annotations

import pytest

CTEST_TARGETS = [
    "test_plan",
    "test_exec_c2c",
    "test_exec_z2z",
    "test_exec_r2c",
    "test_exec_c2r",
    "test_exec_d2z",
    "test_exec_z2d",
    "test_exec_r2c_c2r",
    "test_exec_d2z_z2d",
]


@pytest.mark.parametrize("target", CTEST_TARGETS)
def test_ctest_correctness(target: str, run_ctest) -> None:
    result = run_ctest(target)
    assert result.returncode == 0, (
        f"ctest {target} failed (exit {result.returncode})\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
