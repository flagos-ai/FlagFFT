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


def test_hcu_middle_warp_override_is_scoped_to_inner_permuted_store(monkeypatch):
    monkeypatch.setenv("FLAGFFT_HCU_3D_FUSED_WARPS", "2")
    monkeypatch.setenv("FLAGFFT_HCU_3D_MIDDLE_WARPS", "4")
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
            length=2048,
            factors=(8, 16, 16),
            remainder=1,
            lanes=128,
            num_warps=2,
            generic_radices=(),
            smem_size=2048,
            dtype="complex128",
        )
        common = {
            "module_path": Path("generated.py"),
            "arg_names": ["in_ptr", "out_ptr", "outer_stride", "perm_span", "nbatch"],
            "plan": plan,
            "kernel_type": "leaf_strided_permuted_store",
            "n1": 0,
            "n2": 0,
            "dtype": "complex128",
        }
        middle = _metadata(kernel_name="permuted_store_inner_ifft_kernel", **common)
        final = _metadata(kernel_name="permuted_store_outer_last_ifft_kernel", **common)
        assert middle["num_warps"] == 4
        assert final["num_warps"] == 2
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


def test_hcu_strided_permuted_store_uses_middle_axis_pack_override(monkeypatch):
    monkeypatch.setenv("FLAGFFT_HCU_3D_MIDDLE_PACK", "8")
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
            plan, io_mode="strided_permuted_store", perm_form="inner_middle"
        )
        metadata = _metadata(
            module_path=Path("generated.py"), kernel_name=kernel_name,
            arg_names=["in_ptr", "out_ptr", "outer_stride", "perm_span", "nbatch"],
            plan=plan, kernel_type="leaf_strided_permuted_store", n1=0, n2=0,
            dtype="complex64",
        )
        compile(source, "<strided_middle_axis_pack>", "exec")
        assert "permuted_store_inner_middle_" in kernel_name
        assert metadata["batch_per_block"] == 8
        assert "batch_id = pid * 8" in source
    finally:
        reset_profile(token)


@pytest.mark.parametrize(
    "dtype,length,factors,expected",
    (
        ("complex128", 64, (4, 4, 4), True),
        ("complex128", 256, (16, 16), True),
        ("complex128", 32, (4, 8), False),
        ("complex64", 256, (16, 16), False),
    ),
)
def test_hcu_f64_permuted_store_auto_swizzles_large_power_of_two_leaves(
    monkeypatch, dtype, length, factors, expected
):
    monkeypatch.delenv("FLAGFFT_HCU_3D_SMEM_SWIZZLE", raising=False)
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
            length=length,
            factors=factors,
            remainder=1,
            lanes=16,
            num_warps=1,
            generic_radices=(),
            smem_size=length,
            dtype=dtype,
        )
        _, source = _build_leaf_kernel_source_for_io(
            plan, io_mode="permuted_store", perm_form="outer"
        )
        assert ("smem_phys0 = logical_phys0 ^" in source) is expected
    finally:
        reset_profile(token)


@pytest.mark.parametrize("override,dtype,expected", (("0", "complex128", False), ("1", "complex64", True)))
def test_hcu_smem_swizzle_override_wins_over_auto(monkeypatch, override, dtype, expected):
    monkeypatch.setenv("FLAGFFT_HCU_3D_SMEM_SWIZZLE", override)
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
            dtype=dtype,
        )
        _, source = _build_leaf_kernel_source_for_io(
            plan, io_mode="permuted_store", perm_form="outer"
        )
        assert ("smem_phys0 = logical_phys0 ^" in source) is expected
    finally:
        reset_profile(token)


@pytest.mark.parametrize(
    "io_mode,dtype,override,length,expected",
    (
        ("contiguous_r2c", "complex128", "0", 256, False),
        ("contiguous_r2c", "complex128", "1", 256, True),
        ("contiguous_r2c", "complex128", "auto", 256, True),
        ("contiguous_r2c", "complex128", None, 256, True),
        ("contiguous_r2c", "complex128", "auto", 64, False),
        ("contiguous_r2c", "complex64", "1", 256, False),
        ("permuted_r2c", "complex128", "1", 256, True),
    ),
)
def test_hcu_r2c_leaf_swizzle_default_is_narrow_and_fp64_contiguous_only(
    monkeypatch, io_mode, dtype, override, length, expected
):
    if override is None:
        monkeypatch.delenv("FLAGFFT_HCU_3D_R2C_LEAF_SWIZZLE", raising=False)
    else:
        monkeypatch.setenv("FLAGFFT_HCU_3D_R2C_LEAF_SWIZZLE", override)
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
            length=length,
            factors=(16, 16) if length == 256 else (8, 8),
            remainder=1,
            lanes=16,
            num_warps=1,
            generic_radices=(),
            smem_size=256,
            dtype=dtype,
        )
        _, source = _build_leaf_kernel_source_for_io(plan, io_mode=io_mode)
        assert ("smem_phys0 = logical_phys0 ^" in source) is expected
    finally:
        reset_profile(token)


@pytest.mark.parametrize("pack", (1, 2, 4, 8, 16, 32))
def test_hcu_2048_middle_leaf_grid_matches_batch_pack(monkeypatch, pack):
    monkeypatch.setenv("FLAGFFT_HCU_3D_MIDDLE_BATCH_PACK", str(pack))
    token = set_profile(
        BackendProfile(
            backend="hcu", device_arch="gfx936", warp_size=64,
            max_threads_per_block=1024, max_dynamic_shared_memory=65536,
            policy="native",
        )
    )
    try:
        plan = LeafPlan(
            length=2048, factors=(16, 16, 8), remainder=1, lanes=128,
            num_warps=8, generic_radices=(), smem_size=2048,
        )
        kernel_name, source = _build_leaf_kernel_source_for_io(
            plan, io_mode="contiguous"
        )
        metadata = _metadata(
            module_path=Path("generated.py"), kernel_name=kernel_name,
            arg_names=["in_ptr", "out_ptr", "nbatch"], plan=plan,
            kernel_type="leaf", n1=0, n2=0, dtype="complex64",
        )
        assert metadata["batch_per_block"] == pack
        assert f"batch_id = pid * {pack}" in source
    finally:
        reset_profile(token)
