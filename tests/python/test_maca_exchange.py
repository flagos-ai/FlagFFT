# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU semantic checks for alternative MACA register exchange expressions."""
from __future__ import annotations

import math
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from flagfft_codegen.kernels_leaf import _emit_portable_exchange


class TensorLanguage:
    arange = staticmethod(np.arange)
    where = staticmethod(np.where)
    full = staticmethod(np.full)
    reshape = staticmethod(np.reshape)
    gather = staticmethod(np.take)
    trans = staticmethod(np.transpose)
    zeros_like = staticmethod(np.zeros_like)

    @staticmethod
    def join(a, b):
        return np.stack((a, b), axis=-1)

    @staticmethod
    def split(a):
        return a[..., 0], a[..., 1]


@pytest.mark.parametrize("factors", [
    (16,), (16, 8, 8), (16, 16, 8), (2,) * 10, (8, 8), (19, 16), (19, 11),
    (13, 6, 5), (17, 7, 4), (10, 9, 5), (9, 9, 9), (7, 6, 3, 3),
    (10, 10, 9), (5, 5, 5, 3), (17, 13, 8), (7, 6, 6, 4),
])
@pytest.mark.parametrize("pack,inner,padded", [(1, False, False), (4, False, True), (4, True, False)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_joined_exchange_matches_original(monkeypatch, factors, pack, inner, padded, dtype):
    n = math.prod(factors)
    lanes = max(128, 1 << (max(n // r for r in factors) - 1).bit_length())
    slot_stride = (1 << (n - 1).bit_length()) + int(padded)
    size = 1 << (slot_stride * pack - 1).bit_length()
    rng = np.random.default_rng(731)
    for stage, radix in enumerate(factors):
        registers = {
            f"exchange_{component}{digit}": rng.normal(size=lanes * pack).astype(dtype)
            for component in ("r", "i") for digit in range(radix)
        }
        outputs = []
        for method in ("", "join", "transpose", "direct", "direct_all"):
            monkeypatch.setenv("FLAGFFT_MACA_EXCHANGE", method)
            lines = _emit_portable_exchange(
                "smem", stage, factors, lanes, size, slot_stride, pack,
                natural_order=stage == len(factors) - 1,
                register_lane_stride=pack if inner else 1,
                register_slot_stride=1 if inner else lanes,
            )
            scope = {"tl": TensorLanguage, **registers}
            exec(textwrap.dedent("\n".join(lines)), scope)
            outputs.append((scope["smem_r"], scope["smem_i"]))
            if method == "join" and radix & (radix - 1) == 0:
                assert sum("tl.gather" in line for line in lines) == 2
            if method in {"transpose", "direct", "direct_all"} and not padded and n & (n - 1) == 0:
                assert not any("tl.gather" in line for line in lines)
                if method in {"direct", "direct_all"} and stage < len(factors) - 1:
                    next_radix = factors[stage + 1]
                    next_lanes = n // next_radix
                    for component in ("r", "i"):
                        routed = scope[f"smem_{component}"].reshape(pack, next_radix, next_lanes)
                        for digit in range(next_radix):
                            expected = np.zeros((pack, lanes), dtype=dtype)
                            expected[:, :next_lanes] = routed[:, digit, :]
                            if inner:
                                expected = expected.T
                            np.testing.assert_array_equal(
                                scope[f"smem_register_{component}{digit}"], expected.reshape(-1)
                            )
            if method == "direct_all" and (padded or n & (n - 1)):
                assert sum("tl.gather" in line for line in lines) == 2
            if radix & (radix - 1) and method != "direct_all":
                assert not any("exchange_joined" in line for line in lines)
        for result in outputs[1:]:
            for old, new in zip(outputs[0], result):
                np.testing.assert_array_equal(old, new)


@pytest.mark.parametrize("n,factors", [(1024, (16, 8, 8)), (2048, (16, 16, 8))])
def test_direct_leaf_eliminates_all_gathers(monkeypatch, n, factors):
    import ast
    from flagfft_codegen.backend_profile import BackendProfile, reset_profile, set_profile
    from flagfft_codegen.kernels_common import LeafPlan
    from flagfft_codegen.kernels_leaf import _build_leaf_kernel_source

    profile = BackendProfile.from_device(
        {"backend": "maca", "warp_size": 64, "device_arch": "102"}, "legacy"
    )
    token = set_profile(profile)
    monkeypatch.setenv("FLAGFFT_MACA_EXCHANGE", "direct")
    try:
        plan = LeafPlan(n, factors, 1, n // factors[0], 2, (), n)
        _, source = _build_leaf_kernel_source(plan)
        ast.parse(source)
        assert "tl.gather" not in source
        assert "smem_b_register_r0" in source
        single = LeafPlan(16, (16,), 1, 1, 2, (), 16)
        _, direct_single = _build_leaf_kernel_source(single)
        monkeypatch.delenv("FLAGFFT_MACA_EXCHANGE")
        _, original_single = _build_leaf_kernel_source(single)
        assert direct_single == original_single
    finally:
        reset_profile(token)
