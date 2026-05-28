from __future__ import annotations

import pytest

CTEST_TARGETS = [
    "test_plan",
    "test_exec_c2c_fwd_ct_s",
    "test_exec_c2c_fwd_ct_b",
    "test_exec_c2c_fwd_bs_s",
    "test_exec_c2c_fwd_bs_b",
    "test_exec_c2c_inv_ct_s",
    "test_exec_c2c_inv_ct_b",
    "test_exec_c2c_inv_bs_s",
    "test_exec_c2c_inv_bs_b",
    "test_exec_z2z_fwd_ct_s",
    "test_exec_z2z_fwd_ct_b",
    "test_exec_z2z_fwd_bs_s",
    "test_exec_z2z_fwd_bs_b",
    "test_exec_z2z_inv_ct_s",
    "test_exec_z2z_inv_ct_b",
    "test_exec_z2z_inv_bs_s",
    "test_exec_z2z_inv_bs_b",
    "test_exec_r2c_ct_s",
    "test_exec_r2c_ct_b",
    "test_exec_r2c_bs_s",
    "test_exec_r2c_bs_b",
    "test_exec_c2r_ct_s",
    "test_exec_c2r_ct_b",
    "test_exec_c2r_bs_s",
    "test_exec_c2r_bs_b",
    "test_exec_d2z_ct_s",
    "test_exec_d2z_ct_b",
    "test_exec_d2z_bs_s",
    "test_exec_d2z_bs_b",
    "test_exec_z2d_ct_s",
    "test_exec_z2d_ct_b",
    "test_exec_z2d_bs_s",
    "test_exec_z2d_bs_b",
    "test_exec_r2c_c2r_ct_s",
    "test_exec_r2c_c2r_ct_b",
    "test_exec_r2c_c2r_bs_s",
    "test_exec_r2c_c2r_bs_b",
    "test_exec_d2z_z2d_ct_s",
    "test_exec_d2z_z2d_ct_b",
    "test_exec_d2z_z2d_bs_s",
    "test_exec_d2z_z2d_bs_b",
]


@pytest.mark.parametrize("target", CTEST_TARGETS)
def test_ctest_correctness(target: str, run_ctest) -> None:
    result = run_ctest(target)
    assert result.returncode == 0, (
        f"ctest {target} failed (exit {result.returncode})\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
