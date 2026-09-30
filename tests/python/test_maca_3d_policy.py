"""MACA rank-3 packing remains scoped to rank-3 code generation."""

import pytest

from flagfft_codegen.backend_profile import BackendProfile, reset_profile, set_profile
from flagfft_codegen.kernels_common import LeafPlan, _maca_knob, contiguous_batch_pack_for
from flagfft_codegen.target import reset_maca_3d_default, set_maca_3d_default


def test_maca_3d_packing_scope_and_override(monkeypatch):
    for name in ("FLAGFFT_MACA_BATCH_PACK", "FLAGFFT_MACA_3D_N64_PACK",
                 "FLAGFFT_MACA_3D_N32_PACK", "FLAGFFT_MACA_3D_N128_PACK",
                 "FLAGFFT_MACA_EXCHANGE"):
        monkeypatch.delenv(name, raising=False)
    profile = set_profile(
        BackendProfile.from_device(
            {"backend": "maca", "device_arch": "102", "warp_size": 64,
             "max_threads_per_block": 512, "max_dynamic_shared_memory": 65536}
        )
    )
    short = LeafPlan(64, (4, 4, 4), 1, 16, 2, (), 64, dtype="complex64")
    double_short = LeafPlan(64, (4, 4, 4), 1, 16, 2, (), 64, dtype="complex128")
    single32 = LeafPlan(32, (32,), 1, 1, 2, (), 0, dtype="complex64")
    single = LeafPlan(256, (4, 4, 4, 4), 1, 64, 2, (), 256, dtype="complex64")
    double = LeafPlan(256, (4, 4, 4, 4), 1, 64, 2, (), 256, dtype="complex128")
    single128 = LeafPlan(128, (4, 4, 4, 2), 1, 32, 2, (), 128, dtype="complex64")
    try:
        off = set_maca_3d_default(False)
        try:
            assert contiguous_batch_pack_for(short) == 1
            assert contiguous_batch_pack_for(double_short) == 1
            assert contiguous_batch_pack_for(single) == 1
            assert contiguous_batch_pack_for(double) == 1
            assert _maca_knob("EXCHANGE") == ""
        finally:
            reset_maca_3d_default(off)
        on = set_maca_3d_default(True)
        try:
            assert contiguous_batch_pack_for(single32) == 32
            monkeypatch.setenv("FLAGFFT_MACA_3D_N32_PACK", "16")
            assert contiguous_batch_pack_for(single32) == 16
            monkeypatch.setenv("FLAGFFT_MACA_3D_N32_PACK", "64")
            assert contiguous_batch_pack_for(single32) == 64
            assert contiguous_batch_pack_for(short) == 8
            assert contiguous_batch_pack_for(double_short) == 8
            assert contiguous_batch_pack_for(single) == 2
            assert contiguous_batch_pack_for(double) == 2
            assert contiguous_batch_pack_for(single128) == 2
            monkeypatch.setenv("FLAGFFT_MACA_3D_N128_PACK", "1")
            assert contiguous_batch_pack_for(single128) == 1
            monkeypatch.setenv("FLAGFFT_MACA_3D_N128_PACK", "4")
            assert contiguous_batch_pack_for(single128) == 4
            assert _maca_knob("EXCHANGE") == "direct_all"
            monkeypatch.setenv("FLAGFFT_MACA_3D_N64_PACK", "4")
            assert contiguous_batch_pack_for(short) == 4
            assert contiguous_batch_pack_for(double_short) == 4
            monkeypatch.setenv("FLAGFFT_MACA_BATCH_PACK", "1")
            assert contiguous_batch_pack_for(single) == 1
            assert contiguous_batch_pack_for(double) == 1
            monkeypatch.setenv("FLAGFFT_MACA_EXCHANGE", "join")
            assert _maca_knob("EXCHANGE") == "join"
        finally:
            reset_maca_3d_default(on)
    finally:
        reset_profile(profile)


def test_maca_3d_transpose_defaults(tmp_path, monkeypatch):
    from flagfft_codegen import emit
    from flagfft_codegen.target import set_codegen_target

    for name in ("FLAGFFT_MACA_TRANSPOSE3D", "FLAGFFT_MACA_TRANSPOSE3D_FP64",
                 "FLAGFFT_MACA_TRANSPOSE3D_WARPS"):
        monkeypatch.delenv(name, raising=False)
    set_codegen_target("maca:102:64")
    try:
        fp32 = emit._emit_tiled_transpose3d_jit_kernel(
            n0=256, n1=256, n2=256, order="021", dtype="complex64", out_dir=tmp_path
        )
        fp64 = emit._emit_tiled_transpose3d_jit_kernel(
            n0=256, n1=256, n2=256, order="021", dtype="complex128", out_dir=tmp_path
        )
        assert fp32["kernel_name"].endswith("_t16_tile_pair")
        assert fp32["num_warps"] == 8
        assert fp64["kernel_name"].endswith("_t16_tile_vec")
        assert fp64["num_warps"] == 4
    finally:
        set_codegen_target("")


def test_maca_3d_n32_warp_override(monkeypatch):
    from pathlib import Path

    from flagfft_codegen.metadata import _metadata
    from flagfft_codegen.target import set_codegen_target

    monkeypatch.delenv("FLAGFFT_MACA_3D_N32_WARPS", raising=False)
    profile = set_profile(
        BackendProfile.from_device(
            {"backend": "maca", "device_arch": "102", "warp_size": 64,
             "max_threads_per_block": 512, "max_dynamic_shared_memory": 65536}
        )
    )
    enabled = set_maca_3d_default(True)
    set_codegen_target("maca:102:64")
    plan = LeafPlan(32, (32,), 1, 1, 2, (), 0, dtype="complex64")
    args = {
        "module_path": Path("unused.py"),
        "kernel_name": "fft_kernel_32_l1_b1",
        "arg_names": ["in_ptr", "out_ptr"],
        "plan": plan,
        "kernel_type": "leaf_strided",
        "n1": 0,
        "n2": 0,
        "dtype": "complex64",
    }
    try:
        assert _metadata(**args)["num_warps"] == 2
        monkeypatch.setenv("FLAGFFT_MACA_3D_N32_WARPS", "4")
        assert _metadata(**args)["num_warps"] == 4
        monkeypatch.setenv("FLAGFFT_MACA_3D_N32_WARPS", "1")
        with pytest.raises(ValueError, match="must be 2, 4 or 8"):
            _metadata(**args)
    finally:
        set_codegen_target("")
        reset_maca_3d_default(enabled)
        reset_profile(profile)
