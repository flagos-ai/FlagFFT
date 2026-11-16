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
    ("complex64", "packed_r2c", 2048), ("complex64", "contiguous", 4096),
])
def test_dense_wave_retains_fallbacks(dtype, mode, n):
    plan = LeafPlan(n, (16, n // 16), 1, 16, 1, (), n, dtype=dtype)
    assert hcu_dense_wave_pack(plan, mode) == 0


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_fourstep_codelets_only_replace_supported_local_butterflies(monkeypatch, dtype):
    monkeypatch.setenv("FLAGFFT_HCU_DENSE_WAVE", "fourstep_codelets")
    monkeypatch.setenv("FLAGFFT_HCU_DENSE_WAVE_FP64", "1")
    monkeypatch.setenv("FLAGFFT_HCU_WAVE_CODELET", "1")
    for n in (64, 128, 256, 512, 1024, 2048):
        plan = LeafPlan(n, (16, n // 16), 1, 16, 1, (), n, dtype=dtype)
        assert hcu_dense_wave_pack(plan, "contiguous") == 0
        selected = hcu_dense_wave_pack(plan, "four_step_col")
        assert bool(selected) == (n >= 1024 and (dtype == "complex64" or n <= 1024))
    monkeypatch.setenv("FLAGFFT_HCU_WAVE_CODELET", "0")
    assert hcu_dense_wave_pack(plan, "four_step_col") == 0


@pytest.mark.parametrize("n", [64, 128, 256, 512, 1024])
@pytest.mark.parametrize("kind,direction", [
    ("leaf", "forward"), ("leaf", "inverse"),
    ("leaf_r2c", "forward"), ("leaf_c2r", "inverse"),
    ("four_step_row", "forward"), ("four_step_col", "inverse"),
    ("four_step_real_row", "forward"), ("four_step_r2c_col", "forward"),
    ("four_step_hermitian_row", "inverse"), ("four_step_c2r_col", "inverse"),
])
def test_wide_wave_uses_supplied_double_roots_and_interleaved_double_stores(tmp_path, monkeypatch, n, kind, direction):
    monkeypatch.setenv("FLAGFFT_HCU_DENSE_WAVE_FP64", "1")
    fourstep = kind.startswith("four_step_")
    metadata = emit_jit_kernel(kernel=kind, length=n, factors=(16, n // 16),
        lanes=16, num_warps=1, generic_radices=(), smem_size=n, direction=direction,
        dtype="complex128", prime_n=0, four_step_n1=n if fourstep else 0,
        four_step_n2=n if fourstep else 0, out_dir=tmp_path)
    source = Path(metadata["module_path"]).read_text()
    compile(source, "<hcu-wide-wave>", "exec")
    assert metadata["num_warps"] == 4
    assert metadata["batch_per_block"] == (1 if fourstep else 4)
    assert metadata["inner_pack"] == (4 if fourstep else 1)
    assert f"wave_tw_r_ptr=tw1_r_ptr+{n}" in source
    assert f"wave_tw_i_ptr=tw1_i_ptr+{n}" in source
    assert "tl.uint32" not in source and "tl.float32" not in source
    assert "tl.cos(" not in source and "tl.sin(" not in source
    if kind in {"leaf_c2r", "four_step_c2r_col"}:
        assert "tl.store(out+address,z,mask)" in source or "tl.store(ptr+address,tl.trans(z),mask)" in source
    else:
        assert "tl.join(" in source and "address[:,:,None]*2" in source
    name = metadata["kernel_name"]
    chunk = int(name.split("_chunk", 1)[1].split("_", 1)[0]) if "_chunk" in name else n // 64
    assert 4 * 64 * chunk * 16 <= 32768


def test_wide_redistribution_keeps_native_fallback_above_register_budget(monkeypatch):
    monkeypatch.setenv("FLAGFFT_HCU_DENSE_WAVE_FP64", "1")
    plan = LeafPlan(2048, (16, 16, 8), 1, 128, 2, (), 2048, dtype="complex128")
    assert hcu_dense_wave_pack(plan, "contiguous") == 0
    assert hcu_dense_wave_pack(plan, "four_step_row") == 0


def test_leaf_scope_retains_fourstep_kernels(monkeypatch):
    monkeypatch.setenv("FLAGFFT_HCU_DENSE_WAVE", "leaves")
    plan = LeafPlan(1024, (32, 32), 1, 32, 1, (32,), 1024, dtype="complex64")
    assert hcu_dense_wave_pack(plan, "contiguous") == 8
    assert hcu_dense_wave_pack(plan, "four_step_row") == 0
    assert hcu_dense_wave_pack(plan, "four_step_col") == 0


@pytest.mark.parametrize("n", [64, 128, 256, 512, 1024])
@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("kind,direction", [("leaf_packed_r2c", "forward"), ("leaf_packed_c2r", "inverse")])
def test_packed_wave_keeps_boundary_abi_and_payload_bound(tmp_path, monkeypatch, n, dtype, kind, direction):
    monkeypatch.setenv("FLAGFFT_HCU_DENSE_WAVE_FP64", "1")
    metadata = emit_jit_kernel(kernel=kind, length=n, factors=(16, n // 16),
        lanes=16, num_warps=1, generic_radices=(), smem_size=n, direction=direction,
        dtype=dtype, prime_n=0, four_step_n1=0, four_step_n2=0, out_dir=tmp_path)
    source = Path(metadata["module_path"]).read_text()
    compile(source, "<hcu-packed-wave>", "exec")
    assert metadata["kernel_name"].startswith("hcu_warp")
    assert "packed_twiddle_ptr" in metadata["arg_names"]
    assert "input_distance" in metadata["arg_names"]
    assert n * 4 <= 64 * 64
    if kind == "leaf_packed_r2c":
        if dtype == "complex64": assert "input_distance%2==0" in source
        assert "output_distance" in metadata["arg_names"]
        assert "nyquist_r" in source
    else:
        assert f"output_distance={n}" in source
    if dtype == "complex128":
        assert "tl.uint32" not in source and "tl.float32" not in source
        assert "tl.cos(" not in source and "tl.sin(" not in source
        assert "tl.load(packed_twiddle_ptr+" in source


@pytest.mark.parametrize("n,mode,expected", [(512, "contiguous", 16), (2048, "contiguous", 8),
                                           (512, "packed_r2c", 16), (1024, "packed_c2r", 8)])
def test_wider_row_packs_bound_the_physical_register_payload(monkeypatch, n, mode, expected):
    monkeypatch.setenv("FLAGFFT_HCU_DENSE_WAVE_PACK", "16")
    plan = LeafPlan(n, (16, n // 16), 1, 16, 1, (), n, dtype="complex64")
    assert hcu_dense_wave_pack(plan, mode) == expected


@pytest.mark.parametrize("kind,direction", [("leaf", "forward"), ("leaf", "inverse"),
    ("four_step_row", "forward"), ("four_step_col", "inverse"),
    ("four_step_real_row", "forward"), ("four_step_hermitian_row", "inverse"),
    ("leaf_packed_r2c", "forward"), ("leaf_packed_c2r", "inverse")])
def test_fp32_wave_table_roots_remove_device_trigonometry(tmp_path, monkeypatch, kind, direction):
    monkeypatch.setenv("FLAGFFT_HCU_WAVE_ROOT_TABLE", "1")
    fourstep = kind.startswith("four_step_")
    metadata = emit_jit_kernel(kernel=kind, length=512, factors=(16, 32),
        lanes=16, num_warps=1, generic_radices=(32,), smem_size=512, direction=direction,
        dtype="complex64", prime_n=0, four_step_n1=512 if fourstep else 0,
        four_step_n2=512 if fourstep else 0, out_dir=tmp_path)
    source = Path(metadata["module_path"]).read_text()
    assert "_roots_" in metadata["kernel_name"]
    assert "wave_tw_r_ptr=tw1_r_ptr+512" in source
    assert "tl.cos(" not in source and "tl.sin(" not in source
    assert "tl.uint32" in source
    compile(source, "<hcu-fp32-table-roots>", "exec")


def test_permuted_wave_uses_the_same_supplied_root_abi(tmp_path, monkeypatch):
    monkeypatch.setenv("FLAGFFT_HCU_DENSE_WAVE", "0")
    monkeypatch.setenv("FLAGFFT_HCU_3D_WARP_FFT", "1")
    monkeypatch.setenv("FLAGFFT_HCU_WAVE_ROOT_TABLE", "1")
    metadata = emit_jit_kernel(kernel="leaf_permuted_store", length=512, factors=(16, 32),
        lanes=16, num_warps=1, generic_radices=(32,), smem_size=512, direction="forward",
        dtype="complex64", prime_n=0, four_step_n1=0, four_step_n2=0,
        permutation_span=128, permutation_batch=16384, out_dir=tmp_path)
    source = Path(metadata["module_path"]).read_text()
    assert "_roots_" in metadata["kernel_name"]
    assert "wave_tw_r_ptr=tw1_r_ptr+512" in source
    assert "tl.cos(" not in source and "tl.sin(" not in source
    compile(source, "<hcu-permuted-table-roots>", "exec")
