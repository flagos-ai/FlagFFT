"""Contract tests for the default-on MACA 1D single policy."""

from __future__ import annotations


def _maca_profile():
    from flagfft_codegen.backend_profile import BackendProfile

    return BackendProfile.from_device(
        {
            "backend": "maca",
            "device_arch": "102",
            "warp_size": 64,
            "max_threads_per_block": 1024,
            "max_dynamic_shared_memory": 65536,
        }
    )


def test_maca_1d_single_defaults_are_scoped_and_overridable(monkeypatch):
    from flagfft_codegen.backend_profile import reset_profile, set_profile
    from flagfft_codegen.kernels_common import _maca_knob
    from flagfft_codegen.target import (
        reset_maca_1d_single_default,
        set_maca_1d_single_default,
    )

    for name in ("EXCHANGE", "INNER_PACK", "MAX_WARPS", "SPLIT_ORDER", "VEC_IO"):
        monkeypatch.delenv(f"FLAGFFT_MACA_{name}", raising=False)

    profile_token = set_profile(_maca_profile())
    policy_token = set_maca_1d_single_default(True)
    try:
        assert _maca_knob("EXCHANGE") == "direct_all"
        assert _maca_knob("INNER_PACK") == "8"
        assert _maca_knob("MAX_WARPS") == "8"
        assert _maca_knob("SPLIT_ORDER") == "lsb"
        assert _maca_knob("VEC_IO", "0") == "0"

        monkeypatch.setenv("FLAGFFT_MACA_EXCHANGE", "direct")
        assert _maca_knob("EXCHANGE") == "direct"
    finally:
        reset_maca_1d_single_default(policy_token)
        reset_profile(profile_token)


def test_maca_policy_off_preserves_codegen_defaults(monkeypatch):
    from flagfft_codegen.backend_profile import reset_profile, set_profile
    from flagfft_codegen.kernels_common import _maca_knob
    from flagfft_codegen.target import (
        reset_maca_1d_single_default,
        set_maca_1d_single_default,
    )

    monkeypatch.delenv("FLAGFFT_MACA_EXCHANGE", raising=False)
    monkeypatch.delenv("FLAGFFT_MACA_INNER_PACK", raising=False)
    profile_token = set_profile(_maca_profile())
    policy_token = set_maca_1d_single_default(False)
    try:
        assert _maca_knob("EXCHANGE") == ""
        assert _maca_knob("INNER_PACK") == ""
    finally:
        reset_maca_1d_single_default(policy_token)
        reset_profile(profile_token)
