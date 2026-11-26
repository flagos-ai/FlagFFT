"""2D policy lifetime, independent 1D scope and explicit rollback."""

from pathlib import Path

import pytest


def test_2d_policy_restores_parent_and_respects_override(monkeypatch, tmp_path):
    from flagfft_codegen import emit
    from flagfft_codegen.kernels_common import _maca_knob
    from flagfft_codegen.target import (
        maca_1d_single_default_enabled,
        reset_maca_1d_single_default,
        reset_maca_2d_single_default,
        set_maca_1d_single_default,
        set_maca_2d_single_default,
    )

    for name in ("EXCHANGE", "INNER_PACK", "2D_TRANSPOSE"):
        monkeypatch.delenv(f"FLAGFFT_MACA_{name}", raising=False)
    monkeypatch.setattr(emit, "_maca_backend_active", lambda: True)
    outer_1d = set_maca_1d_single_default(False)
    outer_2d = set_maca_2d_single_default(False)
    try:
        assert _maca_knob("EXCHANGE") == ""
        token = set_maca_2d_single_default(True)
        try:
            assert not maca_1d_single_default_enabled()
            assert _maca_knob("EXCHANGE") == "direct_all"
            assert _maca_knob("INNER_PACK") == "8"
            meta = emit._emit_tiled_transpose_jit_kernel(
                n0=2048, n1=1025, out_dir=tmp_path
            )
            assert "packed_transpose" in meta["kernel_name"]
            monkeypatch.setenv("FLAGFFT_MACA_2D_TRANSPOSE", "legacy")
            fallback = emit._emit_tiled_transpose_jit_kernel(
                n0=2048, n1=1025, out_dir=tmp_path
            )
            assert "packed_transpose" not in fallback["kernel_name"]
            assert "tl.uint64" not in Path(fallback["module_path"]).read_text()
        finally:
            reset_maca_2d_single_default(token)
        assert _maca_knob("EXCHANGE") == ""
    finally:
        reset_maca_1d_single_default(outer_1d)
        reset_maca_2d_single_default(outer_2d)


@pytest.mark.parametrize(
    "dtype,shape,tile,warps",
    [
        ("complex64", (1025, 2048), 32, 8),
        ("complex64", (2048, 1025), 32, 4),
        ("complex128", (1025, 2048), 16, 4),
        ("complex128", (2048, 1025), 16, 4),
        ("complex128", (256, 512), 32, 4),
    ],
)
def test_c550_tuned_transpose_metadata(monkeypatch, tmp_path, dtype, shape, tile, warps):
    from flagfft_codegen import emit
    from flagfft_codegen.backend_profile import BackendProfile, reset_profile, set_profile

    monkeypatch.setenv("FLAGFFT_MACA_2D_TRANSPOSE", "tuned")
    token = set_profile(BackendProfile.from_device({
        "backend": "maca", "device_arch": "102", "warp_size": 64,
    }))
    try:
        rows, cols = shape
        meta = emit._emit_tiled_transpose_jit_kernel(
            n0=rows, n1=cols, dtype=dtype, out_dir=tmp_path,
        )
        assert meta["tile_size"] == tile
        assert meta["num_warps"] == warps
        assert meta["kernel_name"].endswith(f"_t{tile}")
    finally:
        reset_profile(token)


@pytest.mark.parametrize(
    "shape,layout,grid",
    [
        ((2048, 1025), "_flat_t16", (None, None)),
        ((1025, 2048), "_column_t16", (65, 128)),
        ((2048, 2048), "_flat_t16", (None, None)),
        ((1024, 2048), "_f64_t16", (None, None)),
    ],
)
def test_c550_vector_fp64_transpose_layout(monkeypatch, tmp_path, shape, layout, grid):
    from flagfft_codegen import emit
    from flagfft_codegen.backend_profile import BackendProfile, reset_profile, set_profile

    monkeypatch.setenv("FLAGFFT_MACA_2D_TRANSPOSE", "vector")
    token = set_profile(BackendProfile.from_device({
        "backend": "maca", "device_arch": "102", "warp_size": 64,
    }))
    try:
        meta = emit._emit_tiled_transpose_jit_kernel(
            n0=shape[0], n1=shape[1], dtype="complex128", out_dir=tmp_path,
        )
        assert meta["kernel_name"].endswith(layout)
        assert meta["tile_size"] == 16
        assert (meta.get("grid_x_override"), meta.get("grid_y_override")) == grid
    finally:
        reset_profile(token)


def test_maca_dense_policy_respects_override_and_scope(monkeypatch):
    from flagfft_codegen.backend_profile import BackendProfile, reset_profile, set_profile
    from flagfft_codegen.kernels_common import _maca_knob
    from flagfft_codegen.target import reset_maca_2d_dense_default, set_maca_2d_dense_default

    monkeypatch.delenv("FLAGFFT_MACA_EXCHANGE", raising=False)
    monkeypatch.delenv("FLAGFFT_MACA_2D_TRANSPOSE", raising=False)
    profile = set_profile(BackendProfile.from_device({
        "backend": "maca", "device_arch": "102", "warp_size": 64,
    }))
    try:
        policy = set_maca_2d_dense_default(True)
        try:
            assert _maca_knob("EXCHANGE") == "direct_all"
            assert _maca_knob("2D_TRANSPOSE") == "vector"
            with monkeypatch.context() as override:
                override.setenv("FLAGFFT_MACA_EXCHANGE", "transpose")
                override.setenv("FLAGFFT_MACA_2D_TRANSPOSE", "legacy")
                assert _maca_knob("EXCHANGE") == "transpose"
                assert _maca_knob("2D_TRANSPOSE") == "legacy"
        finally:
            reset_maca_2d_dense_default(policy)
        assert _maca_knob("EXCHANGE") == ""
        assert _maca_knob("2D_TRANSPOSE") == ""
    finally:
        reset_profile(profile)


@pytest.mark.parametrize("direction", ("forward", "inverse"))
@pytest.mark.parametrize(
    "length,factors,perm_form,kernel_stem,warps,batch_pack",
    [
        (2048, (16, 16, 8), "maca_fp64_dif", "maca_fp64_dif2048", 2, 1),
        (64, (4, 4, 4), "maca_fp64_dif_first64", "maca_fp64_dif_first64", 8, 32),
        (128, (8, 4, 4), "maca_fp64_ct128_prepare", "maca_fp64_ct128_prepare", 4, 4),
        (128, (8, 4, 4), "maca_fp64_ct128_finish", "maca_fp64_ct128_finish", 4, 8),
    ],
)
def test_c550_fp64_specialized_leaf_launch_abi(
    tmp_path, direction, length, factors, perm_form, kernel_stem, warps, batch_pack,
):
    from flagfft_codegen.emit import emit_jit_kernel
    from flagfft_codegen.backend_profile import BackendProfile, reset_profile, set_profile

    token = set_profile(BackendProfile.from_device({
        "backend": "maca", "device_arch": "102", "warp_size": 64,
    }))
    try:
        metadata = emit_jit_kernel(
            kernel="leaf", length=length, factors=factors, lanes=64, num_warps=4,
            generic_radices=(), smem_size=length, direction=direction, dtype="complex128",
            prime_n=0, four_step_n1=0, four_step_n2=0, perm_form=perm_form, out_dir=tmp_path,
        )
        assert metadata["kernel_name"] == f"{kernel_stem}_{direction}"
        assert metadata["num_warps"] == warps
        assert metadata["batch_per_block"] == batch_pack
        assert metadata["arg_names"] == ["in_ptr", "out_ptr", "roots_r_ptr", "roots_i_ptr", "nbatch"]
    finally:
        reset_profile(token)


@pytest.mark.parametrize(
    "length,factors,lanes,expected",
    [
        (48, (4, 4, 3), 4, 4),
        (64, (4, 4, 4), 16, 4),
        (128, (8, 4, 4), 16, 4),
        (2048, (16, 16, 8), 128, 1),
    ],
)
def test_real_row_pack_matches_metadata(monkeypatch, length, factors, lanes, expected):
    from flagfft_codegen.backend_profile import (
        BackendProfile,
        reset_profile,
        set_profile,
    )
    from flagfft_codegen.kernels_common import LeafPlan, contiguous_batch_pack_for
    from flagfft_codegen.kernels_leaf import _build_leaf_kernel_source_for_io
    from flagfft_codegen.metadata import _metadata
    from flagfft_codegen.target import (
        set_maca_2d_single_default,
        reset_maca_2d_single_default,
    )

    monkeypatch.delenv("FLAGFFT_MACA_BATCH_PACK", raising=False)
    profile = set_profile(
        BackendProfile.from_device(
            {
                "backend": "maca",
                "device_arch": "102",
                "warp_size": 64,
                "max_threads_per_block": 512,
                "max_dynamic_shared_memory": 65536,
            }
        )
    )
    policy = set_maca_2d_single_default(True)
    try:
        plan = LeafPlan(length, factors, 1, lanes, 2, (), length)
        # Complex leaves used by small RC matrices must not inherit row packing.
        assert contiguous_batch_pack_for(plan) == 1
        for kernel, mode in (
            ("leaf_r2c", "contiguous_r2c"),
            ("leaf_c2r", "contiguous_c2r"),
        ):
            name, source = _build_leaf_kernel_source_for_io(plan, io_mode=mode)
            meta = _metadata(
                module_path=Path("unused.py"),
                kernel_name=name,
                arg_names=[],
                plan=plan,
                kernel_type=kernel,
                n1=0,
                n2=0,
                dtype=plan.dtype,
            )
            assert meta["batch_per_block"] == expected
            assert f"batch_id = pid * {expected}" in source
        monkeypatch.setenv("FLAGFFT_MACA_BATCH_PACK", "1")
        assert contiguous_batch_pack_for(plan, real_boundary=True) == 1
    finally:
        reset_maca_2d_single_default(policy)
        reset_profile(profile)
