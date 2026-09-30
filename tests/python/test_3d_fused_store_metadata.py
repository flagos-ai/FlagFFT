from pathlib import Path

import pytest

from flagfft_codegen.backend_profile import BackendProfile, reset_profile, set_profile
from flagfft_codegen.kernels_common import LeafPlan
from flagfft_codegen.kernels_common import permuted_store_batch_pack_for
from flagfft_codegen.kernels_leaf import _build_leaf_kernel_source_for_io
from flagfft_codegen.metadata import _metadata


@pytest.mark.parametrize("pack", (1, 2, 4, 8, 16, 32))
def test_hcu_permuted_store_grid_matches_generated_batch_pack(monkeypatch, pack):
    monkeypatch.setenv("FLAGFFT_HCU_3D_PACK", str(pack))
    token = set_profile(
        BackendProfile(
            backend="hcu",
            device_arch="gfx936",
            warp_size=64,
            max_threads_per_block=1024,
            max_dynamic_shared_memory=65536,
            policy="native",
        )
    )
    try:
        plan = LeafPlan(
            length=64,
            factors=(4, 4, 4),
            remainder=1,
            lanes=16,
            num_warps=1,
            generic_radices=(),
            smem_size=64,
        )
        kernel_name, source = _build_leaf_kernel_source_for_io(
            plan, io_mode="permuted_store", perm_form="outer"
        )
        metadata = _metadata(
            module_path=Path("generated.py"),
            kernel_name=kernel_name,
            arg_names=["in_ptr", "out_ptr", "perm_span", "nbatch"],
            plan=plan,
            kernel_type="leaf_permuted_store",
            n1=0,
            n2=0,
            dtype="complex64",
        )
        assert metadata["batch_per_block"] == pack
        assert f"batch_id = pid * {pack}" in source
    finally:
        reset_profile(token)


@pytest.mark.parametrize("warps", (1, 2, 4, 8))
def test_hcu_permuted_store_warp_override(monkeypatch, warps):
    monkeypatch.setenv("FLAGFFT_HCU_3D_FUSED_WARPS", str(warps))
    token = set_profile(
        BackendProfile(
            backend="hcu",
            device_arch="gfx936",
            warp_size=64,
            max_threads_per_block=1024,
            max_dynamic_shared_memory=65536,
            policy="native",
        )
    )
    try:
        plan = LeafPlan(
            length=256,
            factors=(16, 16),
            remainder=1,
            lanes=16,
            num_warps=1,
            generic_radices=(),
            smem_size=256,
        )
        metadata = _metadata(
            module_path=Path("generated.py"),
            kernel_name="permuted_store_outer_fft_kernel",
            arg_names=["in_ptr", "out_ptr", "perm_span", "nbatch"],
            plan=plan,
            kernel_type="leaf_permuted_store",
            n1=0,
            n2=0,
            dtype="complex64",
        )
        assert metadata["num_warps"] == warps
    finally:
        reset_profile(token)


@pytest.mark.parametrize("dtype, requested, expected", [
    ("complex64", "16", 16),
    ("complex128", "8", 8),
])
def test_hcu_permuted_store_full_shared_memory_pack(monkeypatch, dtype, requested, expected):
    monkeypatch.setenv("FLAGFFT_HCU_3D_PACK", requested)
    monkeypatch.setenv("FLAGFFT_HCU_3D_FULL_SMEM", "1")
    token = set_profile(
        BackendProfile(
            backend="hcu", device_arch="gfx936", warp_size=64,
            max_threads_per_block=1024, max_dynamic_shared_memory=65536,
            policy="native",
        )
    )
    try:
        plan = LeafPlan(
            length=256, factors=(16, 16), remainder=1, lanes=16,
            num_warps=1, generic_radices=(), smem_size=256, dtype=dtype,
        )
        assert permuted_store_batch_pack_for(plan) == expected
    finally:
        reset_profile(token)


def test_hcu_full_shared_memory_pack_keeps_three_stage_budget(monkeypatch):
    monkeypatch.setenv("FLAGFFT_HCU_3D_PACK", "32")
    monkeypatch.setenv("FLAGFFT_HCU_3D_FULL_SMEM", "1")
    token = set_profile(
        BackendProfile(
            backend="hcu", device_arch="gfx936", warp_size=64,
            max_threads_per_block=1024, max_dynamic_shared_memory=65536,
            policy="native",
        )
    )
    try:
        plan = LeafPlan(
            length=128, factors=(8, 4, 4), remainder=1, lanes=16,
            num_warps=1, generic_radices=(), smem_size=128,
        )
        assert permuted_store_batch_pack_for(plan) == 16
    finally:
        reset_profile(token)


def test_hcu_r2c_permuted_store_writes_compact_transposed_rows(monkeypatch):
    token = set_profile(
        BackendProfile(
            backend="hcu", device_arch="gfx936", warp_size=64,
            max_threads_per_block=1024, max_dynamic_shared_memory=65536,
            policy="native",
        )
    )
    try:
        plan = LeafPlan(
            length=256, factors=(16, 16), remainder=1, lanes=16,
            num_warps=1, generic_radices=(), smem_size=256,
        )
        kernel_name, source = _build_leaf_kernel_source_for_io(
            plan, io_mode="permuted_r2c"
        )
        metadata = _metadata(
            module_path=Path("generated.py"), kernel_name=kernel_name,
            arg_names=["in_ptr", "out_ptr", "input_distance", "output_distance", "perm_span", "nbatch"],
            plan=plan, kernel_type="leaf_r2c_permuted_store", n1=0, n2=0,
            dtype="complex64",
        )
        compile(source, "<r2c_permuted_store>", "exec")
        assert metadata["batch_per_block"] == 4
        assert "perm_gbase = perm_i0 * (129 * perm_span) + perm_i1" in source
        assert "output_base_lane < 129" in source
        assert "input_batch_base + in0" in source
    finally:
        reset_profile(token)


def test_hcu_r2c_permuted_store_uses_first_axis_pack_override(monkeypatch):
    monkeypatch.setenv("FLAGFFT_HCU_3D_FIRST_PACK", "8")
    token = set_profile(
        BackendProfile(
            backend="hcu", device_arch="gfx936", warp_size=64,
            max_threads_per_block=1024, max_dynamic_shared_memory=65536,
            policy="native",
        )
    )
    try:
        plan = LeafPlan(
            length=256, factors=(16, 16), remainder=1, lanes=16,
            num_warps=1, generic_radices=(), smem_size=256,
        )
        kernel_name, source = _build_leaf_kernel_source_for_io(
            plan, io_mode="permuted_r2c", perm_form="outer_first"
        )
        metadata = _metadata(
            module_path=Path("generated.py"), kernel_name=kernel_name,
            arg_names=["in_ptr", "out_ptr", "input_distance", "output_distance", "perm_span", "nbatch"],
            plan=plan, kernel_type="leaf_r2c_permuted_store", n1=0, n2=0,
            dtype="complex64",
        )
        compile(source, "<r2c_first_axis_pack>", "exec")
        assert "permuted_store_outer_first_" in kernel_name
        assert metadata["batch_per_block"] == 8
        assert "batch_id = pid * 8" in source
    finally:
        reset_profile(token)
