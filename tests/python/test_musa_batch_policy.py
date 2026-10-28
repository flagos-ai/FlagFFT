from pathlib import Path

import pytest

from flagfft_codegen.backend_profile import BackendProfile, reset_profile, set_profile
from flagfft_codegen.kernels_common import (
    LeafPlan,
    contiguous_batch_pack_for,
    four_step_col_inner_pack_for,
    four_step_row_inner_pack_for,
)
from flagfft_codegen.kernels_leaf import _build_leaf_kernel_source_for_io
from flagfft_codegen.metadata import _metadata
from flagfft_codegen.target import (
    reset_musa_1d_batch_default,
    set_musa_1d_batch_default,
)


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_musa_1d_batch_pack_is_scoped_resource_bounded_and_matches_metadata(
    monkeypatch, dtype
):
    monkeypatch.delenv("FLAGFFT_MUSA_1D_BATCH_PACK", raising=False)
    profile_token = set_profile(
        BackendProfile.from_device(
            {
                "backend": "musa",
                "device_arch": "31",
                "warp_size": 32,
                "max_threads_per_block": 1024,
                "max_dynamic_shared_memory": 196608,
            },
            policy="legacy",
        )
    )
    policy_token = set_musa_1d_batch_default(True)
    plan = LeafPlan(1024, (32, 32), 1, 32, 1, (), 1024, dtype=dtype)
    try:
        expected_pack = 4 if dtype == "complex128" else 1
        assert contiguous_batch_pack_for(plan) == expected_pack
        assert contiguous_batch_pack_for(plan, real_boundary=True) == 1
        expected_c2r_pack = 4 if dtype == "complex128" else 1
        expected_r2c_pack = 2 if dtype == "complex128" else 1
        assert contiguous_batch_pack_for(
            plan, real_boundary=True, real_transform="c2r"
        ) == expected_c2r_pack
        assert contiguous_batch_pack_for(
            plan, real_boundary=True, real_transform="r2c"
        ) == expected_r2c_pack

        name, source = _build_leaf_kernel_source_for_io(plan, io_mode="contiguous")
        for kernel_type, expected in (
            ("leaf_c2r", expected_c2r_pack),
            ("leaf_r2c", expected_r2c_pack),
        ):
            metadata = _metadata(
                module_path=Path("unused.py"),
                kernel_name=name,
                arg_names=[],
                plan=plan,
                kernel_type=kernel_type,
                n1=0,
                n2=0,
                dtype=dtype,
            )
            assert metadata["batch_per_block"] == expected

        for io_mode, expected in (
            ("contiguous_c2r", expected_c2r_pack),
            ("contiguous_r2c", expected_r2c_pack),
        ):
            _, real_source = _build_leaf_kernel_source_for_io(plan, io_mode=io_mode)
            assert f"batch_id = pid * {expected}" in real_source

        metadata = _metadata(
            module_path=Path("unused.py"),
            kernel_name=name,
            arg_names=[],
            plan=plan,
            kernel_type="leaf",
            n1=0,
            n2=0,
            dtype=dtype,
        )
        assert f"batch_id = pid * {expected_pack}" in source
        assert metadata["batch_per_block"] == expected_pack

        monkeypatch.setenv("FLAGFFT_MUSA_1D_BATCH_PACK", "1")
        monkeypatch.setenv("FLAGFFT_MUSA_1D_BATCH_WARPS", "2")
        metadata = _metadata(
            module_path=Path("unused.py"),
            kernel_name=name,
            arg_names=[],
            plan=plan,
            kernel_type="leaf",
            n1=0,
            n2=0,
            dtype=dtype,
        )
        assert metadata["num_warps"] == 2

        monkeypatch.setenv("FLAGFFT_MUSA_1D_BATCH_PACK", "2")
        assert contiguous_batch_pack_for(plan) == 2
        _, source = _build_leaf_kernel_source_for_io(plan, io_mode="contiguous")
        assert "batch_id = pid * 2" in source

        monkeypatch.setenv("FLAGFFT_MUSA_1D_BATCH_PACK", "8")
        bounded_profile = set_profile(
            BackendProfile.from_device(
                {
                    "backend": "musa",
                    "device_arch": "31",
                    "warp_size": 32,
                    "max_threads_per_block": 1024,
                    "max_dynamic_shared_memory": 65536,
                },
                policy="legacy",
            )
        )
        try:
            assert contiguous_batch_pack_for(plan) == (
                2 if dtype == "complex64" else 1
            )
        finally:
            reset_profile(bounded_profile)
    finally:
        reset_musa_1d_batch_default(policy_token)
        reset_profile(profile_token)


def test_musa_1d_batch_pack_does_not_change_unscoped_leaves(monkeypatch):
    monkeypatch.delenv("FLAGFFT_MUSA_1D_BATCH_PACK", raising=False)
    profile_token = set_profile(
        BackendProfile.from_device(
            {
                "backend": "musa",
                "device_arch": "31",
                "warp_size": 32,
                "max_threads_per_block": 1024,
                "max_dynamic_shared_memory": 196608,
            },
            policy="legacy",
        )
    )
    try:
        plan = LeafPlan(1024, (32, 32), 1, 32, 1, (), 1024)
        assert contiguous_batch_pack_for(plan) == 1
    finally:
        reset_profile(profile_token)


@pytest.mark.parametrize(
    "length,factors,lanes",
    [(512, (32, 16), 32), (128, (8, 4, 4), 16), (64, (4, 4, 4), 16)],
)
def test_musa_1d_batch_warp_override_covers_short_fourstep_leaves(
    monkeypatch, length, factors, lanes
):
    monkeypatch.setenv("FLAGFFT_MUSA_1D_BATCH_WARPS", "2")
    profile_token = set_profile(
        BackendProfile.from_device(
            {
                "backend": "musa",
                "device_arch": "31",
                "warp_size": 32,
                "max_threads_per_block": 1024,
                "max_dynamic_shared_memory": 196608,
            },
            policy="legacy",
        )
    )
    policy_token = set_musa_1d_batch_default(True)
    plan = LeafPlan(length, factors, 1, lanes, 1, (), length, dtype="complex64")
    try:
        for kernel_type in ("four_step_hermitian_row", "four_step_c2r_col"):
            metadata = _metadata(
                module_path=Path("unused.py"),
                kernel_name=kernel_type,
                arg_names=[],
                plan=plan,
                kernel_type=kernel_type,
                n1=length,
                n2=8192 // length,
                dtype="complex64",
            )
            assert metadata["num_warps"] == 2
    finally:
        reset_musa_1d_batch_default(policy_token)
        reset_profile(profile_token)


@pytest.mark.parametrize("pack", [1, 2, 4, 8])
def test_musa_fourstep_inner_pack_override_is_bounded_and_scoped(monkeypatch, pack):
    profile_token = set_profile(
        BackendProfile.from_device(
            {
                "backend": "musa",
                "device_arch": "31",
                "warp_size": 32,
                "max_threads_per_block": 1024,
                "max_dynamic_shared_memory": 196608,
            },
            policy="legacy",
        )
    )
    plan = LeafPlan(512, (32, 16), 1, 32, 1, (), 512, dtype="complex64")
    other_plan = LeafPlan(32, (8, 4), 1, 16, 1, (), 32, dtype="complex64")
    policy_token = set_musa_1d_batch_default(True)
    try:
        monkeypatch.delenv("FLAGFFT_MUSA_1D_FOURSTEP_PACK", raising=False)
        other_row_pack = four_step_row_inner_pack_for(
            256, 1024, "complex64", other_plan
        )
        other_col_pack = four_step_col_inner_pack_for(
            256, 1024, "complex64", other_plan
        )
        monkeypatch.setenv("FLAGFFT_MUSA_1D_FOURSTEP_PACK", str(pack))
        assert four_step_row_inner_pack_for(512, 512, "complex64", plan) == pack
        assert four_step_col_inner_pack_for(512, 512, "complex64", plan) == pack
        assert (
            four_step_row_inner_pack_for(256, 1024, "complex64", other_plan)
            == other_row_pack
        )
        assert (
            four_step_col_inner_pack_for(256, 1024, "complex64", other_plan)
            == other_col_pack
        )
    finally:
        reset_musa_1d_batch_default(policy_token)
        reset_profile(profile_token)


def test_musa_fourstep_row_col_pack_overrides_can_be_screened_independently(monkeypatch):
    monkeypatch.delenv("FLAGFFT_MUSA_1D_FOURSTEP_PACK", raising=False)
    monkeypatch.setenv("FLAGFFT_MUSA_1D_FOURSTEP_ROW_PACK", "1")
    monkeypatch.setenv("FLAGFFT_MUSA_1D_FOURSTEP_COL_PACK", "8")
    profile_token = set_profile(
        BackendProfile.from_device(
            {
                "backend": "musa",
                "device_arch": "31",
                "warp_size": 32,
                "max_threads_per_block": 1024,
                "max_dynamic_shared_memory": 196608,
            },
            policy="legacy",
        )
    )
    policy_token = set_musa_1d_batch_default(True)
    plan = LeafPlan(512, (32, 16), 1, 32, 1, (), 512, dtype="complex64")
    try:
        assert four_step_row_inner_pack_for(512, 512, "complex64", plan) == 1
        assert four_step_col_inner_pack_for(512, 512, "complex64", plan) == 8
    finally:
        reset_musa_1d_batch_default(policy_token)
        reset_profile(profile_token)


def test_musa_fourstep_inner_pack_override_covers_short_leaves(monkeypatch):
    monkeypatch.delenv("FLAGFFT_MUSA_1D_FOURSTEP_PACK", raising=False)
    monkeypatch.setenv("FLAGFFT_MUSA_1D_FOURSTEP_ROW_PACK", "2")
    monkeypatch.setenv("FLAGFFT_MUSA_1D_FOURSTEP_COL_PACK", "8")
    profile_token = set_profile(
        BackendProfile.from_device(
            {
                "backend": "musa",
                "device_arch": "31",
                "warp_size": 32,
                "max_threads_per_block": 1024,
                "max_dynamic_shared_memory": 196608,
            },
            policy="legacy",
        )
    )
    policy_token = set_musa_1d_batch_default(True)
    row_plan = LeafPlan(128, (8, 4, 4), 1, 16, 1, (), 128, dtype="complex64")
    col_plan = LeafPlan(64, (4, 4, 4), 1, 16, 1, (), 64, dtype="complex64")
    other_plan = LeafPlan(32, (8, 4), 1, 16, 1, (), 32, dtype="complex64")
    try:
        assert four_step_row_inner_pack_for(128, 64, "complex64", row_plan) == 2
        assert four_step_col_inner_pack_for(128, 64, "complex64", col_plan) == 8
        assert four_step_row_inner_pack_for(128, 64, "complex64", other_plan) != 2
        assert four_step_col_inner_pack_for(128, 64, "complex64", other_plan) != 8
    finally:
        reset_musa_1d_batch_default(policy_token)
        reset_profile(profile_token)
