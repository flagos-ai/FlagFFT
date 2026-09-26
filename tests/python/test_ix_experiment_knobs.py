"""IX diagnostic switches must not silently change another backend."""
import pytest

from flagfft_codegen.backend_profile import BackendProfile, set_profile, reset_profile
from flagfft_codegen.kernels_common import (
    four_step_row_inner_pack_for, four_step_col_inner_pack_for,
)
from flagfft_codegen.kernels_leaf import _portable_complex_vector_io


@pytest.fixture
def ix_profile():
    token = set_profile(BackendProfile.from_device({
        "backend": "ix", "device_arch": "71", "warp_size": 64,
        "max_threads_per_block": 4096, "max_dynamic_shared_memory": 131072,
    }))
    yield
    reset_profile(token)


def test_ix_vector_io_experiment_is_not_enabled(ix_profile, monkeypatch):
    monkeypatch.delenv("FLAGFFT_IX_VEC_IO", raising=False)
    assert not _portable_complex_vector_io()
    monkeypatch.setenv("FLAGFFT_IX_VEC_IO", "1")
    assert not _portable_complex_vector_io()


@pytest.mark.parametrize("pack", [1, 2, 4, 8])
def test_ix_tle_pack_override(ix_profile, monkeypatch, pack):
    monkeypatch.setenv("FLAGFFT_IX_PORTABLE_LEAF", "0")
    monkeypatch.setenv("FLAGFFT_IX_TLE_INNER_PACK", str(pack))
    assert four_step_row_inner_pack_for(1024, 1024) == pack
    assert four_step_col_inner_pack_for(1024, 1024) == pack


def test_ix_tle_pack_rejects_zero(ix_profile, monkeypatch):
    monkeypatch.setenv("FLAGFFT_IX_PORTABLE_LEAF", "0")
    monkeypatch.setenv("FLAGFFT_IX_TLE_INNER_PACK", "0")
    with pytest.raises(ValueError):
        four_step_row_inner_pack_for(1024, 1024)


def test_ix_swizzle_changes_both_exchange_sides(ix_profile, monkeypatch):
    from flagfft_codegen.kernels_common import LeafPlan
    from flagfft_codegen.kernels_leaf import _build_leaf_kernel_source
    monkeypatch.setenv("FLAGFFT_IX_PORTABLE_LEAF", "0")
    monkeypatch.setenv("FLAGFFT_IX_SMEM_SWIZZLE", "1")
    monkeypatch.setenv("FLAGFFT_IX_SMEM_SWIZZLE_SHIFT", "3")
    plan = LeafPlan(1024, (8, 8, 4, 4), 1, 128, 4, (), 1024, "forward", "complex64")
    _, source = _build_leaf_kernel_source(plan)
    assert "smem_dst0 = dst0 ^ (dst0 >> 3)" in source
    assert "smem_phys0 = logical_phys0 ^ (logical_phys0 >> 3)" in source
    compile(source, "<ix-swizzle>", "exec")


def test_ix_four_step_interleave_is_symmetric(ix_profile, monkeypatch):
    from flagfft_codegen.kernels_common import LeafPlan
    from flagfft_codegen.kernels_leaf import _build_four_step_row_kernel_source
    monkeypatch.setenv("FLAGFFT_IX_PORTABLE_LEAF", "0")
    monkeypatch.setenv("FLAGFFT_IX_SMEM_INTERLEAVE", "1")
    monkeypatch.setenv("FLAGFFT_IX_TLE_INNER_PACK", "8")
    plan = LeafPlan(1024, (32, 32), 1, 32, 2, (32,), 1024, "forward", "complex64")
    assert four_step_row_inner_pack_for(1024, 1024, plan=plan) == 8
    _, source = _build_four_step_row_kernel_source(plan, 1024, 1024)
    assert "phys0 = logical_phys0 * 8 + inner_slot" in source
    assert "smem_dst0 = dst0 * 8 + inner_slot" in source
    compile(source, "<ix-interleave>", "exec")


def test_ix_tle_presets_are_scoped_and_overridable(ix_profile, monkeypatch):
    from flagfft_codegen.kernels_common import _maca_knob
    from flagfft_codegen.target import set_ix_ct_single_tle_default, reset_ix_ct_single_tle_default
    for policy in (1, 2):
        token = set_ix_ct_single_tle_default(policy)
        try:
            assert _maca_knob("SMEM_INTERLEAVE") == "1"
            assert _maca_knob("RECURRENCE") == "1"
            assert _maca_knob("PORTABLE_LEAF") != "1"
            if policy == 2:
                assert _maca_knob("TLE_INNER_PACK") == "8"
                assert _maca_knob("WARPS") == "4"
            monkeypatch.setenv("FLAGFFT_IX_SMEM_INTERLEAVE", "0")
            assert _maca_knob("SMEM_INTERLEAVE") == "0"
            monkeypatch.delenv("FLAGFFT_IX_SMEM_INTERLEAVE")
        finally:
            reset_ix_ct_single_tle_default(token)
    assert _maca_knob("TLE_INNER_PACK") == ""


def test_ix_1024_single_warp_default_and_override(ix_profile, monkeypatch):
    from pathlib import Path
    from flagfft_codegen.kernels_common import LeafPlan
    from flagfft_codegen.metadata import _metadata
    from flagfft_codegen.target import set_ix_ct_single_default, reset_ix_ct_single_default
    token = set_ix_ct_single_default(True)
    monkeypatch.delenv("FLAGFFT_IX_WARPS", raising=False)
    try:
        plan = LeafPlan(1024, (16, 8, 8), 1, 128, 4, (), 1024, "forward", "complex64")
        args = dict(module_path=Path("unused.py"), kernel_name="fft_kernel", arg_names=[],
                    plan=plan, kernel_type="leaf", n1=0, n2=0, dtype="complex64")
        assert _metadata(**args)["num_warps"] == 1
        monkeypatch.setenv("FLAGFFT_IX_WARPS", "2")
        assert _metadata(**args)["num_warps"] == 2
    finally:
        reset_ix_ct_single_default(token)


def test_ix_ct_batch_portable_defaults_and_override(ix_profile, monkeypatch):
    from flagfft_codegen.kernels_common import _maca_knob
    from flagfft_codegen.target import set_ix_ct_batch_default, reset_ix_ct_batch_default

    token = set_ix_ct_batch_default(True)
    try:
        assert _maca_knob("PORTABLE_LEAF") == "1"
        assert _maca_knob("RECURRENCE") == "1"
        assert _maca_knob("WARPS") == "4"
        assert _maca_knob("LANE_MIN") == "1"
        monkeypatch.setenv("FLAGFFT_IX_WARPS", "2")
        assert _maca_knob("WARPS") == "2"
    finally:
        reset_ix_ct_batch_default(token)
    assert _maca_knob("PORTABLE_LEAF") != "1"
