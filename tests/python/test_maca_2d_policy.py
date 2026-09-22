"""2D policy lifetime, independent 1D scope and explicit rollback."""

from pathlib import Path


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
