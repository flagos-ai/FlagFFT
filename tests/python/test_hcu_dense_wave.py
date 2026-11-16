# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from flagfft_codegen.backend_profile import BackendProfile, reset_profile, set_profile
from flagfft_codegen.emit import emit_jit_kernel
from flagfft_codegen.kernels_common import LeafPlan
from flagfft_codegen.kernels_hcu_warp import hcu_dense_wave_pack


@pytest.fixture(autouse=True)
def hcu_profile(monkeypatch):
    token = set_profile(BackendProfile(backend="hcu", device_arch="gfx936", warp_size=64,
                                     max_dynamic_shared_memory=65536, policy="native"))
    monkeypatch.setenv("FLAGFFT_HCU_DENSE_WAVE", "1")
    yield
    reset_profile(token)


@pytest.mark.parametrize("n", [64, 128, 256, 512, 1024, 2048])
@pytest.mark.parametrize("kind,direction", [
    ("leaf", "forward"), ("leaf", "inverse"),
    ("four_step_row", "forward"), ("four_step_col", "inverse"),
    ("four_step_real_row", "forward"), ("four_step_r2c_col", "forward"),
    ("four_step_hermitian_row", "inverse"), ("four_step_c2r_col", "inverse"),
])
def test_dense_wave_preserves_grid_distances_and_lds_budget(tmp_path, n, kind, direction):
    fourstep = kind != "leaf"
    metadata = emit_jit_kernel(kernel=kind, length=n, factors=(16, n // 16),
        lanes=16, num_warps=1, generic_radices=(), smem_size=n, direction=direction,
        dtype="complex64", prime_n=0, four_step_n1=n if fourstep else 0,
        four_step_n2=n if fourstep else 0, out_dir=tmp_path)
    source = Path(metadata["module_path"]).read_text()
    compile(source, "<hcu-dense-wave>", "exec")
    assert metadata["kernel_name"].startswith("hcu_warp")
    assert metadata["num_warps"] == 8
    assert metadata["batch_per_block"] == (1 if fourstep else 8)
    assert metadata["inner_pack"] == (8 if fourstep else 1)
    assert bool(metadata["tle_fused_twiddle"]) == fourstep
    name = metadata["kernel_name"]
    chunk = int(name.split("_chunk", 1)[1].split("_", 1)[0]) if "_chunk" in name else n // 64
    assert 8 * 64 * chunk * 8 <= 32768
    assert "pid=tl.program_id(0).to(tl.int64)" in source
    if fourstep:
        assert "tl.program_id(1).to(tl.int64)*" in source
        assert f"row[:,None]<{n}" in source
    else:
        assert "row[:,None]<nbatch" in source


@pytest.mark.parametrize("dtype,mode,n", [
    ("complex128", "contiguous", 256), ("complex64", "contiguous", 210),
    ("complex64", "strided", 256), ("complex64", "permuted_store", 256),
    ("complex64", "packed_r2c", 256), ("complex64", "contiguous", 4096),
])
def test_dense_wave_retains_fallbacks(dtype, mode, n):
    plan = LeafPlan(n, (16, n // 16), 1, 16, 1, (), n, dtype=dtype)
    assert hcu_dense_wave_pack(plan, mode) == 0
