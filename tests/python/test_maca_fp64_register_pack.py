"""Bounds for the opt-in balanced FP64 four-step resource experiment."""
from pathlib import Path

import pytest

from flagfft_codegen.backend_profile import BackendProfile, reset_profile, set_profile
from flagfft_codegen.kernels_common import LeafPlan, four_step_col_inner_pack_for
from flagfft_codegen.kernels_leaf import _build_four_step_col_kernel_source
from flagfft_codegen.metadata import _metadata
from flagfft_codegen.target import set_codegen_target


@pytest.mark.parametrize("exchange", ["direct", "direct_all"])
def test_fp64_register_trial_is_bounded(monkeypatch, exchange):
    profile = BackendProfile.from_device({
        "backend": "maca", "device_arch": "102", "warp_size": 64,
        "max_threads_per_block": 512, "max_dynamic_shared_memory": 65536,
    }, "legacy")
    token = set_profile(profile)
    set_codegen_target("maca:80:64")
    monkeypatch.setenv("FLAGFFT_MACA_EXCHANGE", exchange)
    monkeypatch.setenv("FLAGFFT_MACA_INNER_PACK", "4")
    monkeypatch.setenv("FLAGFFT_MACA_MAX_WARPS", "8")
    plan = LeafPlan(1024, (16, 8, 8), 1, 64, 2, (), 1024, dtype="complex128")
    try:
        monkeypatch.setenv("FLAGFFT_MACA_FP64_REGISTER_PACK", "0")
        assert four_step_col_inner_pack_for(1024, 1024, "complex128", plan) == 2
        monkeypatch.setenv("FLAGFFT_MACA_FP64_REGISTER_PACK", "1")
        assert four_step_col_inner_pack_for(1024, 1024, "complex128", plan) == 4
        name, source = _build_four_step_col_kernel_source(plan, 1024, 1024)
        meta = _metadata(module_path=Path("unused.py"), kernel_name=name,
                         arg_names=[], plan=plan, kernel_type="four_step_col",
                         n1=1024, n2=1024, dtype=plan.dtype)
        assert meta["inner_pack"] == 4
        assert meta["num_warps"] <= 8
        assert "tl.gather" not in source
        oversized = LeafPlan(2048, (16, 16, 8), 1, 128, 4, (), 2048,
                             dtype="complex128")
        assert four_step_col_inner_pack_for(512, 2048, "complex128", oversized) == 1
        mixed = LeafPlan(476, (17, 7, 4), 1, 1, 2, (), 512, dtype="complex128")
        baseline = four_step_col_inner_pack_for(390, 476, "complex128", mixed)
        monkeypatch.setenv("FLAGFFT_MACA_FP64_REGISTER_PACK", "0")
        assert four_step_col_inner_pack_for(390, 476, "complex128", mixed) == baseline
    finally:
        reset_profile(token)
        set_codegen_target("")
