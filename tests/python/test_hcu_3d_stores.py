# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from flagfft_codegen.backend_profile import BackendProfile, reset_profile, set_profile
from flagfft_codegen.kernels_common import LeafPlan
from flagfft_codegen.kernels_leaf import _build_leaf_kernel_source_for_io


@pytest.fixture(autouse=True)
def hcu_profile():
    token = set_profile(BackendProfile(
        backend="hcu", device_arch="gfx936", warp_size=64,
        max_dynamic_shared_memory=65536, policy="native",
    ))
    yield
    reset_profile(token)


def emit(dtype, mode="permuted_store", form="outer"):
    plan = LeafPlan(256, (16, 16), 1, 16, 1, (), 256, dtype=dtype)
    _, source = _build_leaf_kernel_source_for_io(plan, io_mode=mode, perm_form=form)
    compile(source, "<hcu-store-layout>", "exec")
    return source


@pytest.mark.parametrize("form,span,rows", [
    ("outer", 256, 65536), ("inner", 256, 65536),
    ("outer", 129, 33024), ("inner", 129, 33024),
    ("inner_batched", 256, 262144),
])
def test_static_geometry_hints_only_proven_aligned_tiles(monkeypatch, form, span, rows):
    from flagfft_codegen.kernels_common import permuted_store_batch_pack_for
    monkeypatch.setenv("FLAGFFT_HCU_3D_STORE_MODE", "cyclic")
    plan = LeafPlan(256, (16, 16), 1, 16, 1, (), 256, dtype="complex64")
    pack = permuted_store_batch_pack_for(plan)
    _, source = _build_leaf_kernel_source_for_io(
        plan, io_mode="permuted_store", perm_form=form,
        permutation_span=span, permutation_batch=rows,
    )
    compile(source, "<hcu-static-permutation>", "exec")
    assert f"nbatch = tl.full((), {rows}, tl.int64)" in source
    hinted = "tl.max_contiguous(tl.multiple_of(perm_gbase," in source
    if form.endswith("_batched"):
        assert not hinted
    elif hinted:
        for pid in range((rows + pack - 1) // pack):
            batch = ((pid // span) * pack * span + pid % span
                     if form == "inner" else pid * pack)
            bases = []
            for slot in range(pack):
                current = batch + (slot * span if form == "inner" else slot)
                i0, i1 = divmod(current, span)
                bases.append(i1 * (rows // span) + i0 if form == "inner"
                             else i0 * (plan.length * span) + i1)
            assert bases == list(range(bases[0], bases[0] + pack))
            assert bases[0] % pack == 0
    else:
        assert form == "outer" and span % pack != 0


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("mode", ["permuted_store", "strided_permuted_store"])
@pytest.mark.parametrize("form", ["outer", "inner"])
def test_shared_store_reuses_read_buffer_after_a_barrier(monkeypatch, dtype, mode, form):
    monkeypatch.setenv("FLAGFFT_HCU_3D_STORE_MODE", "smem")
    source = emit(dtype, mode, form)
    first_write = source.index("tail_dst0 =")
    # All last-stage inputs must be read before the input buffer is reused.
    assert source.rfind("tl.load(tle.gpu.local_ptr(smem_b_i", 0, first_write) < source.rfind(
        "tl.debug_barrier()", 0, first_write
    )
    first_readback = source.index("tail_r = tl.load")
    assert first_write < source.rfind("tl.debug_barrier()", 0, first_readback)
    assert "tail_mask = (tail_batch < nbatch) & (tail_freq < 256)" in source
    assert "tail_addr = tail_freq * perm_k_stride + tail_gbase" in source


def test_shared_store_keeps_per_cube_batch_addressing_on_generic_path(monkeypatch):
    monkeypatch.setenv("FLAGFFT_HCU_3D_STORE_MODE", "smem")
    source = emit("complex64", form="inner_batched")
    assert "rows_per_transform" in source
    assert "tail_flat" not in source


def test_grouped_store_preserves_fp64_and_compact_spectrum_fallbacks(monkeypatch):
    monkeypatch.setenv("FLAGFFT_HCU_3D_STORE_MODE", "auto")
    fp64 = emit("complex128")
    compact = emit("complex64", mode="permuted_store_r2c")
    monkeypatch.setenv("FLAGFFT_HCU_3D_STORE_MODE", "grouped")
    assert emit("complex128") == fp64
    assert emit("complex64", mode="permuted_store_r2c") == compact
    source = emit("complex64")
    assert "perm_group_pair_out = tl.trans" in source
    assert "perm_group_mask" in source
    assert "tl.pointer_type(tl.uint64)" in source


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("mode", ["permuted_store", "strided_permuted_store"])
@pytest.mark.parametrize("form", ["outer", "inner", "inner_batched"])
def test_direct_store_switches_mapping_before_reading_last_stage(monkeypatch, dtype, mode, form):
    monkeypatch.setenv("FLAGFFT_HCU_3D_STORE_MODE", "cyclic")
    monkeypatch.setenv("FLAGFFT_HCU_3D_DIRECT_STORE", "1")
    source = emit(dtype, mode=mode, form=form)
    switch = source.index("batch_slot = lane_vec %")
    assert source.index("for group_0") < switch < source.index("for group_1")
    assert "perm_pair_addr0 = perm_addr0[:, None] * 2" in source
    assert "zr0 = tl.trans" not in source
    if form.endswith("_batched"):
        assert "perm_gbase_vector = (perm_i0_vector // perm_outer)" in source


def test_direct_store_keeps_compact_boundary_on_original_mapping(monkeypatch):
    monkeypatch.setenv("FLAGFFT_HCU_3D_STORE_MODE", "cyclic")
    original = emit("complex64", mode="permuted_store_r2c")
    monkeypatch.setenv("FLAGFFT_HCU_3D_DIRECT_STORE", "1")
    assert emit("complex64", mode="permuted_store_r2c") == original


@pytest.mark.parametrize("outer_rows", [1, 3, 6, 8, 20, 32])
def test_inner_store_pack_covers_small_and_partial_outer_axes(monkeypatch, tmp_path, outer_rows):
    from flagfft_codegen.emit import emit_jit_kernel
    from flagfft_codegen.kernels_common import permuted_store_batch_pack_for
    monkeypatch.setenv("FLAGFFT_HCU_3D_STORE_MODE", "cyclic")
    span = 64
    rows = outer_rows * span
    plan = LeafPlan(256, (16, 16), 1, 16, 1, (), 256, dtype="complex64")
    pack = permuted_store_batch_pack_for(
        plan, permutation_span=span, permutation_batch=rows, inner=True,
    )
    assert outer_rows % pack == 0
    visited = [
        (pid // span) * pack * span + pid % span + slot * span
        for pid in range(rows // pack) for slot in range(pack)
    ]
    assert sorted(visited) == list(range(rows))
    metadata = emit_jit_kernel(
        kernel="leaf_permuted_store", length=256, factors=(16, 16), lanes=16,
        num_warps=1, generic_radices=(), smem_size=256, direction="forward",
        dtype="complex64", prime_n=0, four_step_n1=0, four_step_n2=0,
        perm_form="inner", permutation_span=span, permutation_batch=rows,
        out_dir=tmp_path,
    )
    assert metadata["batch_per_block"] == pack
    assert f"batch_id = pid * {pack}" in Path(metadata["module_path"]).read_text()


@pytest.mark.parametrize("n", [64, 128, 256, 512, 1024, 2048])
@pytest.mark.parametrize("form", ["inner", "outer"])
def test_wave_store_retains_leaf_signature_and_bounds_output_tiles(monkeypatch, tmp_path, n, form):
    from flagfft_codegen.emit import emit_jit_kernel
    monkeypatch.setenv("FLAGFFT_HCU_3D_WARP_FFT", "1")
    factors = (16, n // 16)
    metadata = emit_jit_kernel(
        kernel="leaf_permuted_store", length=n, factors=factors, lanes=16,
        num_warps=2, generic_radices=(), smem_size=n, direction="forward",
        dtype="complex64", prime_n=0, four_step_n1=0, four_step_n2=0,
        perm_form=form, permutation_span=128, permutation_batch=16384,
        out_dir=tmp_path,
    )
    source = Path(metadata["module_path"]).read_text()
    compile(source, "<hcu-wave-fft>", "exec")
    assert metadata["kernel_name"].startswith("hcu_warp")
    assert metadata["num_warps"] == metadata["batch_per_block"] <= 8
    assert "perm_span" in metadata["arg_names"] and "nbatch" in metadata["arg_names"]
    assert "nbatch=tl.full((),16384,tl.int64)" in source
    assert "fast_cosf" not in source


@pytest.mark.parametrize("dtype,mode,form", [
    ("complex128", "permuted_store", "outer"),
    ("complex64", "strided_permuted_store", "inner"),
    ("complex64", "permuted_store", "inner_batched"),
    ("complex64", "permuted_store_r2c", "outer"),
])
def test_wave_store_preserves_unsupported_layout_fallbacks(monkeypatch, dtype, mode, form):
    from flagfft_codegen.kernels_hcu_warp import hcu_warp_store_pack
    monkeypatch.setenv("FLAGFFT_HCU_3D_WARP_FFT", "1")
    plan = LeafPlan(256, (16, 16), 1, 16, 1, (), 256, dtype=dtype)
    assert hcu_warp_store_pack(plan, mode, form, 128, 16384) == 0


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("variant", ["wide", "cyclic"])
@pytest.mark.parametrize("n,factors,lanes", [
    (64, (4, 4, 4), 16), (128, (8, 4, 4), 16),
    (256, (16, 16), 16), (2048, (16, 16, 8), 128),
    (2048, (8, 16, 16), 128),
])
def test_wide_store_bounds_the_rounded_allocation_and_complete_stages(monkeypatch, dtype, variant, n, factors, lanes):
    from flagfft_codegen.kernels_common import (
        cooperative_stage_lanes_for, lane_block_for, permuted_store_batch_pack_for,
    )
    monkeypatch.setenv("FLAGFFT_HCU_3D_STORE_MODE", variant)
    plan = LeafPlan(n, factors, 1, lanes, 1, (), n, dtype=dtype)
    pack = permuted_store_batch_pack_for(plan)
    scalar_bytes = 8 if dtype == "complex128" else 4
    padded = n + int(pack >= 4 and variant != "cyclic")
    assert 2 * lane_block_for(padded * pack) * scalar_bytes <= 65536
    assert lane_block_for(max(cooperative_stage_lanes_for(plan))) * pack <= 512
    _, source = _build_leaf_kernel_source_for_io(plan, io_mode="permuted_store")
    compile(source, "<hcu-wide-store>", "exec")
    assert "smem_a_r = tle.gpu.alloc" not in source
    assert "smem_b_r = tle.gpu.alloc" in source
    assert "tail_flat" not in source
    if variant == "cyclic" and pack > 1:
        assert f"smem_offset = batch_slot * {n}" in source
        assert "phys0 = ((logical_phys0 + batch_slot)" in source
        assert ("smem_dst0 = ((smem_dst0 + batch_slot)" if dtype == "complex128"
                else "smem_dst0 = ((dst0 + batch_slot)") in source


@pytest.mark.parametrize("n", [64, 128, 256, 512, 2048])
@pytest.mark.parametrize("pack", [2, 8, 16])
def test_cyclic_shared_indices_are_bijective_without_padding(n, pack):
    for shift in (None, 5):
        addresses = {
            slot * n + (((index if shift is None else index ^ (index >> shift)) + slot) & (n - 1))
            for slot in range(pack) for index in range(n)
        }
        assert addresses == set(range(n * pack))


def test_wide_store_rejects_an_overwriting_stage_with_multiple_groups(monkeypatch):
    from flagfft_codegen.kernels_common import _use_single_smem_buffer
    monkeypatch.setenv("FLAGFFT_HCU_3D_STORE_MODE", "wide")
    plan = LeafPlan(2048, (8, 8, 32), 1, 64, 2, (), 2048, dtype="complex128")
    assert not _use_single_smem_buffer(plan, io_mode="permuted_store")


@pytest.mark.parametrize("order", ["021", "210", "201", "120"])
@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_packed_transpose_preserves_component_axis_and_partial_tile_masks(monkeypatch, tmp_path, order, dtype):
    from flagfft_codegen.emit import _emit_tiled_transpose3d_jit_kernel
    monkeypatch.setenv("FLAGFFT_HCU_3D_TRANSPOSE_MODE", "packed")
    metadata = _emit_tiled_transpose3d_jit_kernel(
        n0=9, n1=33, n2=65, order=order, dtype=dtype, out_dir=tmp_path,
    )
    source = Path(metadata["module_path"]).read_text()
    compile(source, "<hcu-paired-transpose>", "exec")
    assert "safe_rows = tl.minimum" in source
    assert "safe_cols = tl.minimum" in source
    if dtype == "complex128":
        assert "tl.permute(values, (1, 0, 2))" in source
        assert "component = tl.arange(0, 2)" in source
    else:
        assert "tl.pointer_type(tl.uint64)" in source
