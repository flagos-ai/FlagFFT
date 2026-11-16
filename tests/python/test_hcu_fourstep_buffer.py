# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared-buffer reuse must respect stage lifetimes and physical tile bounds."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from flagfft_codegen.backend_profile import BackendProfile, reset_profile, set_profile
from flagfft_codegen.emit import emit_jit_kernel
from flagfft_codegen.kernels_common import LeafPlan, hcu_four_step_single_buffer
from flagfft_codegen.kernels_common import hcu_leaf_single_buffer
from flagfft_codegen.kernels_leaf import _build_leaf_kernel_source_for_io


@pytest.fixture(autouse=True)
def profile(monkeypatch):
    token = set_profile(BackendProfile(backend="hcu", device_arch="gfx936", warp_size=64,
                                     max_dynamic_shared_memory=65536, policy="native"))
    monkeypatch.setenv("FLAGFFT_HCU_FOURSTEP_SINGLE_BUFFER", "1")
    yield
    reset_profile(token)


@pytest.mark.parametrize("n,factors,lanes", [
    (128, (8, 4, 4), 16), (256, (16, 16), 16),
    (512, (32, 16), 16), (512, (8, 8, 8), 64),
    (1024, (32, 32), 32), (1024, (16, 16, 4), 64),
])
@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("kind", ["four_step_row", "four_step_col"])
def test_dense_fourstep_uses_one_bounded_cyclic_buffer(tmp_path, n, factors, lanes, dtype, kind):
    plan = LeafPlan(n, factors, 1, lanes, 1, (), n, dtype=dtype)
    assert hcu_four_step_single_buffer(plan)
    metadata = emit_jit_kernel(
        kernel=kind, length=n, factors=factors, lanes=lanes, num_warps=1,
        generic_radices=(32,) if 32 in factors else (), smem_size=n,
        direction="inverse", dtype=dtype, prime_n=0, four_step_n1=n,
        four_step_n2=n, out_dir=tmp_path,
    )
    source = Path(metadata["module_path"]).read_text()
    compile(source, "<hcu-fourstep-cyclic>", "exec")
    assert "smem_a_r = tle.gpu.alloc" not in source
    assert "smem_b_r = tle.gpu.alloc" in source
    pack = metadata["inner_pack"]
    assert 2 * n * pack * (8 if dtype == "complex128" else 4) <= 65536
    if pack > 1:
        assert "logical_phys0 + inner_slot" in source
        assert "logical_phys0 + batch_slot" not in source
    # Every FFT owns exactly one bijective cyclic region, including its
    # final shared read. No slot can address another FFT's live values.
    addresses = [slot*n + ((position+slot) & (n-1))
                 for slot in range(pack) for position in range(n)]
    assert sorted(addresses) == list(range(n*pack))
    if len(factors) > 2:
        begin = source.index("for group_1")
        end = source.index("for group_2")
        stage = source[begin:end]
        last_read = stage.rfind("tl.load(tle.gpu.local_ptr(smem_b_i")
        first_write = stage.find("tl.store(tle.gpu.local_ptr(smem_b_r")
        assert last_read < stage.find("tl.debug_barrier()", last_read) < first_write


def test_partial_overwriting_stage_keeps_ping_pong_buffers():
    # With 64 lanes, the middle radix-8 stage has two groups. Reusing
    # one buffer would overwrite inputs before later groups consumed them.
    plan = LeafPlan(1024, (16, 8, 8), 1, 64, 1, (), 1024, dtype="complex64")
    assert not hcu_four_step_single_buffer(plan)


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("mode", ["contiguous", "strided", "contiguous_r2c", "contiguous_c2r"])
def test_leaf_reuses_storage_only_after_complete_shared_reads(monkeypatch, dtype, mode):
    monkeypatch.setenv("FLAGFFT_HCU_LEAF_SINGLE_BUFFER", "1")
    plan = LeafPlan(2048, (16, 16, 8), 1, 128, 1, (), 2048, dtype=dtype)
    assert hcu_leaf_single_buffer(plan, mode)
    _, source = _build_leaf_kernel_source_for_io(plan, io_mode=mode)
    compile(source, "<hcu-leaf-single-buffer>", "exec")
    assert "smem_a_r = tle.gpu.alloc" not in source
    first = source.index("for group_1")
    last = source.index("for group_2")
    stage = source[first:last]
    read = stage.rfind("tl.load(tle.gpu.local_ptr(smem_b_i")
    assert read < stage.find("tl.debug_barrier()", read) < stage.find("tl.store(tle.gpu.local_ptr(smem_b_r")
    small = LeafPlan(64, (4, 4, 4), 1, 16, 1, (), 64, dtype=dtype)
    assert not hcu_leaf_single_buffer(small, mode)
