"""Contract tests for the default-on MACA 1D single policy."""

from __future__ import annotations


def _maca_profile():
    from flagfft_codegen.backend_profile import BackendProfile

    return BackendProfile.from_device(
        {
            "backend": "maca",
            "device_arch": "102",
            "warp_size": 64,
            "max_threads_per_block": 1024,
            "max_dynamic_shared_memory": 65536,
        }
    )


def test_maca_1d_single_defaults_are_scoped_and_overridable(monkeypatch):
    from flagfft_codegen.backend_profile import reset_profile, set_profile
    from flagfft_codegen.kernels_common import _maca_knob
    from flagfft_codegen.target import (
        reset_maca_1d_single_default,
        set_maca_1d_single_default,
    )

    for name in ("EXCHANGE", "INNER_PACK", "MAX_WARPS", "SPLIT_ORDER", "VEC_IO"):
        monkeypatch.delenv(f"FLAGFFT_MACA_{name}", raising=False)

    profile_token = set_profile(_maca_profile())
    policy_token = set_maca_1d_single_default(True)
    try:
        assert _maca_knob("EXCHANGE") == "direct_all"
        assert _maca_knob("INNER_PACK") == "8"
        assert _maca_knob("INNER_PACK") == "8"
        assert _maca_knob("MAX_WARPS") == "8"
        assert _maca_knob("SPLIT_ORDER") == "lsb"
        assert _maca_knob("VEC_IO", "0") == "0"

        monkeypatch.setenv("FLAGFFT_MACA_EXCHANGE", "direct")
        assert _maca_knob("EXCHANGE") == "direct"
    finally:
        reset_maca_1d_single_default(policy_token)
        reset_profile(profile_token)


def test_maca_policy_off_preserves_codegen_defaults(monkeypatch):
    from flagfft_codegen.backend_profile import reset_profile, set_profile
    from flagfft_codegen.kernels_common import _maca_knob
    from flagfft_codegen.target import (
        reset_maca_1d_single_default,
        set_maca_1d_single_default,
    )

    monkeypatch.delenv("FLAGFFT_MACA_EXCHANGE", raising=False)
    monkeypatch.delenv("FLAGFFT_MACA_INNER_PACK", raising=False)
    profile_token = set_profile(_maca_profile())
    policy_token = set_maca_1d_single_default(False)
    try:
        assert _maca_knob("EXCHANGE") == ""
        assert _maca_knob("INNER_PACK") == ""
    finally:
        reset_maca_1d_single_default(policy_token)
        reset_profile(profile_token)


def test_maca_1d_batch_policy_matches_leaf_launch_and_respects_override(monkeypatch):
    from pathlib import Path

    from flagfft_codegen.backend_profile import reset_profile, set_profile
    from flagfft_codegen.kernels_common import LeafPlan, _maca_knob
    from flagfft_codegen.kernels_leaf import _build_leaf_kernel_source_for_io
    from flagfft_codegen.metadata import _metadata
    from flagfft_codegen.target import (
        reset_maca_1d_batch_default,
        set_maca_1d_batch_default,
    )

    monkeypatch.delenv("FLAGFFT_MACA_BATCH_PACK", raising=False)
    monkeypatch.delenv("FLAGFFT_MACA_EXCHANGE", raising=False)
    profile_token = set_profile(_maca_profile())
    policy_token = set_maca_1d_batch_default(True)
    try:
        plan = LeafPlan(16, (16,), 1, 1, 2, (), 0)
        for kernel, mode in (("leaf", "contiguous"), ("leaf_r2c", "contiguous_r2c"),
                             ("leaf_c2r", "contiguous_c2r")):
            name, source = _build_leaf_kernel_source_for_io(plan, io_mode=mode)
            meta = _metadata(module_path=Path("unused.py"), kernel_name=name,
                             arg_names=[], plan=plan, kernel_type=kernel,
                             n1=0, n2=0, dtype=plan.dtype)
            assert "batch_id = pid * 1" in source
            assert meta["batch_per_block"] == 1

        assert _maca_knob("EXCHANGE") == "direct_all"
        monkeypatch.setenv("FLAGFFT_MACA_EXCHANGE", "")
        assert _maca_knob("EXCHANGE") == ""
        monkeypatch.setenv("FLAGFFT_MACA_BATCH_PACK", "4")
        name, source = _build_leaf_kernel_source_for_io(plan, io_mode="contiguous")
        meta = _metadata(module_path=Path("unused.py"), kernel_name=name,
                         arg_names=[], plan=plan, kernel_type="leaf",
                         n1=0, n2=0, dtype=plan.dtype)
        assert "batch_id = pid * 4" in source
        assert meta["batch_per_block"] == 4
    finally:
        reset_maca_1d_batch_default(policy_token)
        reset_profile(profile_token)


def test_maca_1d_batch_four_step_pack_is_resource_bounded(monkeypatch):
    from dataclasses import replace

    from flagfft_codegen.backend_profile import reset_profile, set_profile
    from flagfft_codegen.kernels_common import LeafPlan, _maca_knob, four_step_row_inner_pack_for
    from flagfft_codegen.target import reset_maca_1d_batch_default, set_maca_1d_batch_default

    monkeypatch.delenv("FLAGFFT_MACA_EXCHANGE", raising=False)
    monkeypatch.delenv("FLAGFFT_MACA_INNER_PACK", raising=False)
    monkeypatch.delenv("FLAGFFT_MACA_FP64_REGISTER_PACK", raising=False)
    profile_token = set_profile(replace(_maca_profile(), policy="legacy"))
    policy_token = set_maca_1d_batch_default(True)
    try:
        large = LeafPlan(512, (8, 4, 4, 4), 1, 64, 2, (), 512)
        large_fp64 = replace(large, dtype="complex128")
        short = LeafPlan(128, (8, 8, 2), 1, 64, 2, (), 128)
        assert four_step_row_inner_pack_for(512, 1024, "complex64", large) == 8
        assert _maca_knob("FP64_REGISTER_PACK") == "1"
        assert four_step_row_inner_pack_for(512, 1024, "complex128", large_fp64) == 4
        assert four_step_row_inner_pack_for(128, 128, "complex64", short) == 4
        monkeypatch.setenv("FLAGFFT_MACA_INNER_PACK", "4")
        assert four_step_row_inner_pack_for(512, 1024, "complex64", large) == 4
        monkeypatch.setenv("FLAGFFT_MACA_FP64_REGISTER_PACK", "0")
        assert _maca_knob("FP64_REGISTER_PACK") == "0"
    finally:
        reset_maca_1d_batch_default(policy_token)
        reset_profile(profile_token)
