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


def test_ix_vector_io_is_opt_in(ix_profile, monkeypatch):
    monkeypatch.delenv("FLAGFFT_IX_VEC_IO", raising=False)
    assert not _portable_complex_vector_io()
    monkeypatch.setenv("FLAGFFT_IX_VEC_IO", "1")
    assert _portable_complex_vector_io()


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
