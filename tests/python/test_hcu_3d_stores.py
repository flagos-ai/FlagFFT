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
