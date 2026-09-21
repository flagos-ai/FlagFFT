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

    @staticmethod
    def join(a, b):
        return np.stack((a, b), axis=-1)

    @staticmethod
    def split(a):
        return a[..., 0], a[..., 1]


@pytest.mark.parametrize("factors", [(16, 8, 8), (16, 16, 8), (8, 8), (19, 16)])
@pytest.mark.parametrize("pack,inner,padded", [(1, False, False), (4, False, True), (4, True, False)])
def test_joined_exchange_matches_original(monkeypatch, factors, pack, inner, padded):
    n = math.prod(factors)
    lanes = max(128, 1 << (max(n // r for r in factors) - 1).bit_length())
    slot_stride = (1 << (n - 1).bit_length()) + int(padded)
    size = 1 << (slot_stride * pack - 1).bit_length()
    rng = np.random.default_rng(731)
    for stage, radix in enumerate(factors):
        registers = {
            f"exchange_{component}{digit}": rng.normal(size=lanes * pack).astype(np.float32)
            for component in ("r", "i") for digit in range(radix)
        }
        outputs = []
        for method in ("", "join", "transpose"):
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
            if method == "transpose" and not padded and n & (n - 1) == 0:
                assert not any("tl.gather" in line for line in lines)
            if radix & (radix - 1):
                assert not any("exchange_joined" in line for line in lines)
        for result in outputs[1:]:
            for old, new in zip(outputs[0], result):
                np.testing.assert_array_equal(old, new)
