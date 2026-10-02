"""MACA rank-3 packing remains scoped to rank-3 code generation."""

from flagfft_codegen.backend_profile import BackendProfile, reset_profile, set_profile
from flagfft_codegen.kernels_common import (
    LeafPlan,
    _maca_knob,
    contiguous_batch_pack_for,
    permuted_store_batch_pack_for,
)
from flagfft_codegen.target import reset_maca_3d_default, set_maca_3d_default


def test_maca_3d_packing_scope_and_override(monkeypatch):
    for name in ("FLAGFFT_MACA_BATCH_PACK", "FLAGFFT_MACA_3D_N64_PACK",
                 "FLAGFFT_MACA_3D_PRIME_2048_PACK",
                 "FLAGFFT_MACA_3D_N128_PACK", "FLAGFFT_MACA_EXCHANGE",
                 "FLAGFFT_MACA_VEC_IO", "FLAGFFT_MACA_3D_PERMSTORE_PACK"):
        monkeypatch.delenv(name, raising=False)
    profile = set_profile(
        BackendProfile.from_device(
            {"backend": "maca", "device_arch": "102", "warp_size": 64,
             "max_threads_per_block": 512, "max_dynamic_shared_memory": 65536}
        )
    )
    short = LeafPlan(64, (4, 4, 4), 1, 16, 2, (), 64, dtype="complex64")
    double_short = LeafPlan(64, (4, 4, 4), 1, 16, 2, (), 64, dtype="complex128")
    single = LeafPlan(256, (4, 4, 4, 4), 1, 64, 2, (), 256, dtype="complex64")
    double = LeafPlan(256, (4, 4, 4, 4), 1, 64, 2, (), 256, dtype="complex128")
    single128 = LeafPlan(128, (4, 4, 4, 2), 1, 32, 2, (), 128, dtype="complex64")
    long_middle = LeafPlan(2048, (16, 8, 16), 1, 128, 4, (), 2048, dtype="complex64")
    prime_middle = LeafPlan(2048, (16, 16, 8), 1, 128, 4, (), 2048, dtype="complex128")
    try:
        off = set_maca_3d_default(False)
        try:
            assert contiguous_batch_pack_for(short) == 1
            assert contiguous_batch_pack_for(double_short) == 1
            assert contiguous_batch_pack_for(single) == 1
            assert contiguous_batch_pack_for(double) == 1
            assert permuted_store_batch_pack_for(short) == 4
            assert permuted_store_batch_pack_for(single128) == 4
            assert _maca_knob("EXCHANGE") == ""
            assert _maca_knob("VEC_IO", "0") == "0"
        finally:
            reset_maca_3d_default(off)
        on = set_maca_3d_default(True)
        try:
            assert contiguous_batch_pack_for(short) == 8
            assert contiguous_batch_pack_for(double_short) == 8
            assert contiguous_batch_pack_for(single) == 2
            assert contiguous_batch_pack_for(double) == 2
            assert contiguous_batch_pack_for(single128) == 2
            assert contiguous_batch_pack_for(prime_middle, prime_n=997) == 1
            monkeypatch.setenv("FLAGFFT_MACA_3D_PRIME_2048_PACK", "2")
            assert contiguous_batch_pack_for(prime_middle, prime_n=997) == 2
            assert contiguous_batch_pack_for(prime_middle) == 1
            from pathlib import Path

            from flagfft_codegen.metadata import _metadata

            metadata = _metadata(
                module_path=Path("candidate.py"),
                kernel_name="bluestein_prepare_leaf_kernel",
                arg_names=["in_ptr"],
                plan=prime_middle,
                kernel_type="leaf_bluestein_prepare",
                n1=0,
                n2=0,
                dtype="complex128",
                prime_n=997,
            )
            assert metadata["batch_per_block"] == 2
            assert metadata["num_warps"] == 8
            monkeypatch.setenv("FLAGFFT_MACA_3D_PRIME_2048_PACK", "4")
            try:
                contiguous_batch_pack_for(prime_middle, prime_n=997)
            except ValueError as error:
                assert "FLAGFFT_MACA_3D_PRIME_2048_PACK must be 1 or 2" in str(error)
            else:
                raise AssertionError("unsupported MACA 3D prime pack was accepted")
            monkeypatch.delenv("FLAGFFT_MACA_3D_PRIME_2048_PACK")
            assert permuted_store_batch_pack_for(short) == 8
            assert permuted_store_batch_pack_for(single128) == 16
            monkeypatch.setenv("FLAGFFT_MACA_3D_PERMSTORE_PACK", "16")
            assert permuted_store_batch_pack_for(short) == 16
            monkeypatch.setenv("FLAGFFT_MACA_3D_PERMSTORE_PACK", "auto")
            assert permuted_store_batch_pack_for(short) == 8
            monkeypatch.setenv("FLAGFFT_MACA_3D_PERMSTORE_PACK", "8")
            assert permuted_store_batch_pack_for(short) == 8
            assert permuted_store_batch_pack_for(single128) == 8
            monkeypatch.setenv("FLAGFFT_MACA_3D_PERMSTORE_PACK", "32")
            assert permuted_store_batch_pack_for(short) == 32
            assert permuted_store_batch_pack_for(single128) == 16
            monkeypatch.setenv("FLAGFFT_MACA_3D_PERMSTORE_PACK", "3")
            try:
                permuted_store_batch_pack_for(short)
            except ValueError as error:
                assert "must be 1, 2, 4, 8, 16, 32 or auto" in str(error)
            else:
                raise AssertionError("unsupported MACA permuted-store pack was accepted")
            monkeypatch.delenv("FLAGFFT_MACA_3D_PERMSTORE_PACK")
            # The 2048-point leaf consumes enough shared memory to cap its
            # permuted-store pack at one; the long-axis hybrid avoids that.
            assert permuted_store_batch_pack_for(long_middle) == 1
            assert _maca_knob("VEC_IO", "0") == "packed"
            from flagfft_codegen.kernels_leaf import _packed_maca_fp32_complex_io
            assert _packed_maca_fp32_complex_io("complex64")
            assert not _packed_maca_fp32_complex_io("complex128")
            monkeypatch.setenv("FLAGFFT_MACA_VEC_IO", "0")
            assert _maca_knob("VEC_IO", "0") == "0"
            assert not _packed_maca_fp32_complex_io("complex64")
            monkeypatch.setenv("FLAGFFT_MACA_VEC_IO", "1")
            assert _maca_knob("VEC_IO", "0") == "1"
            assert not _packed_maca_fp32_complex_io("complex64")
            monkeypatch.delenv("FLAGFFT_MACA_VEC_IO")
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
                 "FLAGFFT_MACA_TRANSPOSE3D_WARPS",
                 "FLAGFFT_MACA_TRANSPOSE3D_SLICE_GROUP"):
        monkeypatch.delenv(name, raising=False)
    set_codegen_target("maca:102:64")
    try:
        fp32 = emit._emit_tiled_transpose3d_jit_kernel(
            n0=256, n1=256, n2=256, order="021", dtype="complex64", out_dir=tmp_path
        )
        fp64 = emit._emit_tiled_transpose3d_jit_kernel(
            n0=256, n1=256, n2=256, order="021", dtype="complex128", out_dir=tmp_path
        )
        assert fp32["kernel_name"].endswith("_t16_tile_pair_sliceg2seq_rmajor")
        assert fp32["num_warps"] == 8
        assert fp64["kernel_name"].endswith("_t16_tile_vec")
        assert fp64["num_warps"] == 4

        long_fp32 = emit._emit_tiled_transpose3d_jit_kernel(
            n0=128, n1=2048, n2=64, order="201", dtype="complex64", out_dir=tmp_path
        )
        assert long_fp32["kernel_name"].endswith(
            "_t16_tile_pair_sliceg2seq_rmajor"
        )
        long_fp64 = emit._emit_tiled_transpose3d_jit_kernel(
            n0=128, n1=2048, n2=64, order="201", dtype="complex128", out_dir=tmp_path
        )
        assert long_fp64["kernel_name"].endswith("_t16_tile_vec")

        monkeypatch.setenv("FLAGFFT_MACA_TRANSPOSE3D_SLICE_GROUP", "1")
        rollback = emit._emit_tiled_transpose3d_jit_kernel(
            n0=256, n1=256, n2=256, order="021", dtype="complex64", out_dir=tmp_path
        )
        assert rollback["kernel_name"].endswith("_t16_tile_pair_rmajor")
        long_rollback = emit._emit_tiled_transpose3d_jit_kernel(
            n0=128, n1=2048, n2=64, order="201", dtype="complex64", out_dir=tmp_path
        )
        assert long_rollback["kernel_name"].endswith("_t16_tile_pair_rmajor")
    finally:
        set_codegen_target("")


def test_maca_leaf_warp_override_is_validated(monkeypatch):
    from pathlib import Path

    from flagfft_codegen.metadata import _metadata
    from flagfft_codegen.target import set_codegen_target

    profile = set_profile(
        BackendProfile.from_device(
            {"backend": "maca", "device_arch": "102", "warp_size": 64,
             "max_threads_per_block": 512, "max_dynamic_shared_memory": 65536}
        )
    )
    plan = LeafPlan(256, (4, 4, 4, 4), 1, 64, 2, (), 256, dtype="complex64")
    set_codegen_target("maca:102:64")

    def make_metadata():
        return _metadata(
            module_path=Path("kernel.py"),
            kernel_name="fft_kernel",
            arg_names=[],
            plan=plan,
            kernel_type="leaf",
            n1=0,
            n2=0,
            dtype="complex64",
        )

    try:
        monkeypatch.delenv("FLAGFFT_MACA_WARPS", raising=False)
        assert make_metadata()["num_warps"] == 2
        monkeypatch.setenv("FLAGFFT_MACA_WARPS", "4")
        assert make_metadata()["num_warps"] == 4
        monkeypatch.setenv("FLAGFFT_MACA_WARPS", "8")
        assert make_metadata()["num_warps"] == 8
        monkeypatch.setenv("FLAGFFT_MACA_WARPS", "16")
        try:
            make_metadata()
        except ValueError as error:
            assert "FLAGFFT_MACA_WARPS must be 2, 4 or 8" in str(error)
        else:
            raise AssertionError("unsupported MACA warp override was accepted")
    finally:
        set_codegen_target("")
        reset_profile(profile)


def test_maca_environment_fingerprint_separates_variants():
    from flagfft_codegen.cli import _maca_environment_fingerprint

    pack1 = {"FLAGFFT_MACA_3D_N128_PACK": "1"}
    pack4 = {"FLAGFFT_MACA_3D_N128_PACK": "4"}
    other_backend = {"FLAGFFT_IX_WARPS": "4"}
    assert _maca_environment_fingerprint({}) == ""
    assert _maca_environment_fingerprint(pack1) != _maca_environment_fingerprint(pack4)
    assert _maca_environment_fingerprint(pack1) == _maca_environment_fingerprint(
        {**pack1, **other_backend}
    )
