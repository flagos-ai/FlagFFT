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


@pytest.fixture
def ix_balanced_profile(ix_profile):
    from dataclasses import replace
    from flagfft_codegen.backend_profile import current_profile
    token = set_profile(replace(current_profile(),policy="balanced"))
    yield
    reset_profile(token)


def test_ix_vector_io_experiment_is_not_enabled(ix_profile, monkeypatch):
    monkeypatch.delenv("FLAGFFT_IX_VEC_IO", raising=False)
    assert not _portable_complex_vector_io()
    monkeypatch.setenv("FLAGFFT_IX_VEC_IO", "1")
    assert not _portable_complex_vector_io()


def test_ix_packed_complex_io_uses_portable_bitcasts(ix_profile, monkeypatch):
    from flagfft_codegen.kernels_common import LeafPlan
    from flagfft_codegen.kernels_leaf import _build_leaf_kernel_source_for_io
    monkeypatch.setenv("FLAGFFT_IX_VEC_IO", "packed")
    for dtype in ("complex64", "complex128"):
        plan = LeafPlan(256, (16, 16), 1, 16, 2, (), 256, "forward", dtype)
        _, source = _build_leaf_kernel_source_for_io(plan, io_mode="permuted_store")
        assert "inline_asm_elementwise" not in source
        if dtype == "complex64":
            assert "tl.pointer_type(tl.uint64)" in source
            assert "bitcast=True" in source
        else:
            assert "tl.pointer_type(tl.uint64)" not in source
        compile(source, "<ix-packed-io>", "exec")


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


@pytest.mark.parametrize("shared_limit, expected_pack", [(16384, 4), (32768, 8), (131072, 8)])
def test_ix_portable_power_two_pack_respects_exchange_budget(monkeypatch, shared_limit, expected_pack):
    from flagfft_codegen.kernels_common import LeafPlan
    from flagfft_codegen.kernels_leaf import _build_four_step_col_kernel_source
    from flagfft_codegen.metadata import _metadata
    from pathlib import Path

    token = set_profile(BackendProfile.from_device({
        "backend": "ix", "device_arch": "71", "warp_size": 64,
        "max_threads_per_block": 4096, "max_dynamic_shared_memory": shared_limit,
    }))
    monkeypatch.setenv("FLAGFFT_IX_PORTABLE_LEAF", "1")
    monkeypatch.setenv("FLAGFFT_IX_INNER_PACK", "8")
    try:
        plan = LeafPlan(1024, (32, 32), 1, 32, 1, (), 1024, "forward", "complex64")
        name, source = _build_four_step_col_kernel_source(plan, 512, 1024)
        meta = _metadata(module_path=Path("unused.py"), kernel_name=name,
                         arg_names=[], plan=plan, kernel_type="four_step_col",
                         n1=512, n2=1024, dtype="complex64")
        assert meta["inner_pack"] == expected_pack
        assert f"four_step_inner_base = tl.program_id(0) * {expected_pack}" in source
    finally:
        reset_profile(token)


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


def test_ix_permuted_pack_matches_source_and_launch(ix_profile, monkeypatch):
    from pathlib import Path
    from flagfft_codegen.kernels_common import LeafPlan
    from flagfft_codegen.kernels_leaf import _build_leaf_kernel_source_for_io
    from flagfft_codegen.metadata import _metadata

    monkeypatch.setenv("FLAGFFT_IX_3D_PACK", "8")
    plan = LeafPlan(256, (16, 16), 1, 16, 2, (), 256, "forward", "complex64")
    name, source = _build_leaf_kernel_source_for_io(plan, io_mode="permuted_store")
    meta = _metadata(module_path=Path("unused.py"), kernel_name=name,
                     arg_names=[], plan=plan, kernel_type="leaf_permuted_store",
                     n1=0, n2=0, dtype="complex64")
    assert meta["batch_per_block"] == 8
    assert "batch_id = pid * 8" in source
    monkeypatch.setenv("FLAGFFT_IX_3D_PACK", "3")
    with pytest.raises(ValueError, match="FLAGFFT_IX_3D_PACK"):
        _build_leaf_kernel_source_for_io(plan, io_mode="permuted_store")


@pytest.mark.parametrize("shared_limit,expected", [(16384,2),(32768,4),(65536,8),(131072,16)])
def test_ix_explicit_permuted_pack_uses_rounded_hardware_bound(monkeypatch, shared_limit, expected):
    from flagfft_codegen.kernels_common import LeafPlan, permuted_store_batch_pack_for
    token = set_profile(BackendProfile.from_device({
        "backend":"ix", "device_arch":"71", "warp_size":64,
        "max_threads_per_block":4096, "max_dynamic_shared_memory":shared_limit,
    }))
    monkeypatch.setenv("FLAGFFT_IX_3D_PACK", "32")
    try:
        plan = LeafPlan(256,(16,16),1,16,2,(),256,"forward","complex64")
        assert permuted_store_batch_pack_for(plan) == expected
    finally:
        reset_profile(token)


def test_ix_permuted_pair_store_is_explicit(ix_profile, monkeypatch):
    from flagfft_codegen.kernels_common import LeafPlan
    from flagfft_codegen.kernels_leaf import _build_leaf_kernel_source_for_io

    monkeypatch.delenv("FLAGFFT_IX_3D_PAIR_STORE", raising=False)
    plan = LeafPlan(256, (16, 16), 1, 16, 2, (), 256, "forward", "complex64")
    _, scalar = _build_leaf_kernel_source_for_io(plan, io_mode="permuted_store")
    assert "pair0 = tl.join(zr0, zi0)" not in scalar
    monkeypatch.setenv("FLAGFFT_IX_3D_PAIR_STORE", "1")
    _, paired = _build_leaf_kernel_source_for_io(plan, io_mode="permuted_store")
    assert "pair0 = tl.join(zr0, zi0)" in paired


@pytest.mark.parametrize("mode", ["contiguous", "strided", "permuted_store", "permuted_store_r2c"])
def test_ix_batch_interleave_has_no_padding_or_cross_slot_aliases(ix_balanced_profile, monkeypatch, mode):
    from flagfft_codegen.kernels_common import LeafPlan
    from flagfft_codegen.kernels_leaf import _build_leaf_kernel_source_for_io
    monkeypatch.setenv("FLAGFFT_IX_BATCH_INTERLEAVE", "1")
    monkeypatch.setenv("FLAGFFT_IX_PORTABLE_LEAF", "0")
    plan = LeafPlan(256,(16,16),1,16,2,(),256,"forward","complex64")
    _, source = _build_leaf_kernel_source_for_io(plan, io_mode=mode)
    assert "phys0 = logical_phys0 * 4 + batch_slot" in source
    assert "smem_dst0 = dst0 * 4 + batch_slot" in source
    assert "tle.gpu.alloc([1024]" in source
    assert "+ inner_slot" not in source
    physical = [logical*4+slot for logical in range(256) for slot in range(4)]
    assert sorted(physical) == list(range(1024))
    compile(source,"<ix-batch-interleave>","exec")


def test_ix_packed_transpose_preserves_native_pointer_abi(ix_profile, monkeypatch, tmp_path):
    from pathlib import Path
    from flagfft_codegen.emit import _emit_tiled_transpose_jit_kernel
    monkeypatch.setenv("FLAGFFT_IX_2D_TRANSPOSE", "packed")
    for dtype, packed in (("complex64",True),("complex128",False)):
        meta = _emit_tiled_transpose_jit_kernel(n0=129,n1=256,dtype=dtype,out_dir=tmp_path/dtype)
        source = Path(meta["module_path"]).read_text()
        assert ("tl.pointer_type(tl.uint64)" in source) == packed
        assert meta["signature"].endswith(",i32")
        assert "inline_asm_elementwise" not in source


@pytest.mark.parametrize("length, factors, lanes, kind, expected", [
    (512, (32, 16), 16, "four_step_row", 2),
    (1024, (32, 32), 32, "four_step_col", 4),
])
def test_ix_resource_warps_and_explicit_override(ix_profile, monkeypatch, length, factors, lanes, kind, expected):
    from pathlib import Path
    from flagfft_codegen.kernels_common import LeafPlan
    from flagfft_codegen.metadata import _metadata

    monkeypatch.setenv("FLAGFFT_IX_PORTABLE_LEAF", "1")
    monkeypatch.setenv("FLAGFFT_IX_INNER_PACK", "8")
    monkeypatch.setenv("FLAGFFT_IX_RESOURCE_WARPS", "1")
    monkeypatch.delenv("FLAGFFT_IX_WARPS", raising=False)
    plan = LeafPlan(length, factors, 1, lanes, 1, (), length, "forward", "complex64")
    params = dict(module_path=Path("unused.py"), kernel_name="resource_kernel",
                  arg_names=[], plan=plan, kernel_type=kind, n1=512, n2=1024,
                  dtype="complex64")
    assert _metadata(**params)["num_warps"] == expected
    monkeypatch.setenv("FLAGFFT_IX_WARPS", "8")
    assert _metadata(**params)["num_warps"] == 8


def test_ix_short_register_pack_keeps_launch_and_source_consistent(ix_profile, monkeypatch):
    from pathlib import Path
    from flagfft_codegen.kernels_common import LeafPlan, _maca_p8_register_leaf_supported
    from flagfft_codegen.kernels_leaf import _build_four_step_col_kernel_source
    from flagfft_codegen.metadata import _metadata

    monkeypatch.setenv("FLAGFFT_IX_PORTABLE_LEAF", "1")
    monkeypatch.setenv("FLAGFFT_IX_INNER_PACK", "16")
    monkeypatch.setenv("FLAGFFT_IX_MAX_PACK", "16")
    plan = LeafPlan(128, (16, 8), 1, 16, 4, (), 128, "forward", "complex64")
    assert not _maca_p8_register_leaf_supported(plan)
    monkeypatch.setenv("FLAGFFT_IX_SHORT_REGISTER_PACK", "1")
    assert _maca_p8_register_leaf_supported(plan)
    name, source = _build_four_step_col_kernel_source(plan, 128, 128)
    meta = _metadata(module_path=Path("unused.py"), kernel_name=name, arg_names=[],
                     plan=plan, kernel_type="four_step_col", n1=128, n2=128,
                     dtype="complex64")
    assert meta["inner_pack"] == 16
    assert "four_step_inner_base = tl.program_id(0) * 16" in source
    for length, factors, dtype in ((210, (7, 6, 5), "complex64"),
                                    (128, (16, 8), "complex128")):
        unsupported = LeafPlan(length, factors, 1, 16, 4, (), length, "forward", dtype)
        assert not _maca_p8_register_leaf_supported(unsupported)


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


def test_ix_1024_single_two_warp_default_and_override(ix_profile, monkeypatch):
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
        assert _metadata(**args)["num_warps"] == 2
        monkeypatch.setenv("FLAGFFT_IX_WARPS", "1")
        assert _metadata(**args)["num_warps"] == 1
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
