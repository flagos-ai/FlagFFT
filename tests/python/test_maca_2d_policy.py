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
