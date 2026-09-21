"""Execute the emitted vector stage using NumPy primitives against FFT.

This checks signs, mirrored outputs, batch boundaries and masked tails without
requiring an NPU. CANN lowering is covered by ascend_stockham_probe.py.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))


class Tensor(np.ndarray):
    def to(self, dtype):
        return self.astype(dtype).view(Tensor)


class Pointer:
    def __init__(self, values, offset=0):
        self.values, self.offset = values, offset

    def __add__(self, offset):
        return Pointer(self.values, self.offset + offset)


class NumpyLanguage:
    float32 = np.float32
    int64 = np.int64
    static_range = staticmethod(range)
    pid = 0

    def program_id(self, axis):
        assert axis == 0
        return np.asarray(self.pid).view(Tensor)

    @staticmethod
    def arange(start, end):
        return np.arange(start, end).view(Tensor)

    @staticmethod
    def full(shape, value, dtype):
        return np.full(shape, value, dtype).view(Tensor)

    @staticmethod
    def load(pointer, mask=True, other=0):
        safe = np.where(mask, pointer.offset, 0)
        return np.where(mask, pointer.values[safe], other).view(Tensor)

    @staticmethod
    def store(pointer, values, mask):
        pointer.values[np.asarray(pointer.offset)[mask]] = np.asarray(values)[mask]


@pytest.mark.parametrize("radix", [13, 17, 19])
@pytest.mark.parametrize("span", [1, 5, 35])
@pytest.mark.parametrize("direction", ["forward", "inverse"])
@pytest.mark.parametrize("block,batch", [(16, 1), (128, 3)])
def test_emitted_vector_stage_matches_numpy(radix, span, direction, block, batch):
    from flagfft_codegen.kernels_stockham import build_stockham_stage

    n = radix * 35  # Deliberately leave a masked tail for both tile sizes.
    name, source = build_stockham_stage(n, radix, direction, "complex64", span, block)
    language = NumpyLanguage()
    namespace = {"tl": language,
        "_cmul": lambda xr, xi, wr, wi: (xr * wr - xi * wi, xr * wi + xi * wr)}
    exec(source.replace("@triton.jit\n", ""), namespace)
    rng = np.random.default_rng(20260921)
    host = (rng.normal(size=(batch, n)) + 1j * rng.normal(size=(batch, n))).astype(np.complex64)
    sign = 1 if direction == "inverse" else -1
    table = np.exp(sign * 2j * np.pi * np.arange(n) / n).astype(np.complex64)
    output = np.full(batch * n, np.nan + 1j * np.nan, dtype=np.complex64)
    for pid in range((batch * (n // radix) + block - 1) // block):
        language.pid = pid
        namespace[name](Pointer(host.reshape(-1).view(np.float32)),
            Pointer(output.view(np.float32)), Pointer(table.view(np.float32)),
            span, np.asarray(batch).view(Tensor))

    k = np.arange(n // radix)
    j = k % span
    samples = host.reshape(batch, radix, -1) * table[
        np.arange(radix)[:, None] * j * (n // (radix * span))]
    transformed = np.fft.fft(samples, axis=1) if direction == "forward" else np.fft.ifft(samples, axis=1) * radix
    expected = np.empty((batch, n), dtype=np.complex128)
    for digit in range(radix):
        expected[:, radix * k - (radix - 1) * j + digit * span] = transformed[:, digit]
    assert np.isfinite(output).all()
    assert np.linalg.norm(output.reshape(batch, n) - expected) / np.linalg.norm(expected) < 3e-7
