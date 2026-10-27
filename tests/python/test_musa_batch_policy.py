from pathlib import Path

import pytest

from flagfft_codegen.backend_profile import BackendProfile, reset_profile, set_profile
from flagfft_codegen.kernels_common import LeafPlan, contiguous_batch_pack_for
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
        name, source = _build_leaf_kernel_source_for_io(plan, io_mode="contiguous")
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
