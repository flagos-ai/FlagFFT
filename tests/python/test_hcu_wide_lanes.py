# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Execute the emitted wide FFT arithmetic against an independent NumPy FFT."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from flagfft_codegen.backend_profile import BackendProfile, reset_profile, set_profile
from flagfft_codegen.emit import emit_jit_kernel
from flagfft_codegen.kernels_ix_warp import build_ix_warp_source


class Pointer:
    def __init__(self, data, offset=0):
        self.data = data.reshape(-1)
        self.offset = offset

    def __add__(self, offset):
        return Pointer(self.data, self.offset + offset)


class NumpyTL:
    float64 = np.float64
    arange = staticmethod(np.arange)
    full = staticmethod(np.full)
    broadcast_to = staticmethod(np.broadcast_to)
    gather = staticmethod(np.take_along_axis)
    where = staticmethod(np.where)
    reshape = staticmethod(np.reshape)
    join = staticmethod(lambda a, b: np.stack((a, b), axis=-1))

    @staticmethod
    def program_id(axis):
        return 0

    @staticmethod
    def multiple_of(value, hint):
        return value

    max_contiguous = multiple_of

    @staticmethod
    def load(ptr, mask=True, other=0):
        index, valid = np.broadcast_arrays(ptr.offset, mask)
        result = np.full(index.shape, other, dtype=ptr.data.dtype)
        result[valid] = ptr.data[index[valid]]
        return result

    @staticmethod
    def store(ptr, value, mask=True):
        index, value, valid = np.broadcast_arrays(ptr.offset, value, mask)
        ptr.data[index[valid]] = value[valid]


@pytest.mark.parametrize("mode,inverse", [("contiguous", False), ("contiguous", True),
                                          ("contiguous_r2c", False), ("c2r", True)])
def test_two_wave_generated_fft_matches_numpy_with_masked_tail(mode, inverse):
    n, batch, pack = 2048, 3, 4
    rng = np.random.default_rng(716)
    real = rng.normal(size=(batch, n))
    if mode == "contiguous_r2c":
        data = real
        expected = np.fft.rfft(real, axis=-1)
    elif mode == "c2r":
        data = np.fft.rfft(real, axis=-1)
        expected = real * n
    else:
        data = real + 1j * rng.normal(size=(batch, n))
        expected = np.fft.ifft(data, axis=-1) * n if inverse else np.fft.fft(data, axis=-1)
    input_distance = data.shape[-1]
    output_distance = expected.shape[-1]
    output = np.full(expected.shape, np.nan, dtype=expected.dtype)
    name, source = build_ix_warp_source(n, pack, mode, inverse, wide=True,
                                      lane_width=128, output_radix_chunk=4)
    source = "\n".join(line for line in source.splitlines()
                       if not line.startswith(("import ", "@triton.jit")))
    roots = np.exp((1j if inverse else -1j) * 2 * np.pi * np.arange(n) / n)
    namespace = {"tl": NumpyTL, "wave_tw_r_ptr": Pointer(roots.real.copy()),
                 "wave_tw_i_ptr": Pointer(roots.imag.copy())}
    exec(compile(source, "<emitted-wide-fft>", "exec"), namespace)
    namespace[name](Pointer(data.view(np.float64)), Pointer(output.view(np.float64)),
                    1, batch, input_distance, output_distance)
    np.testing.assert_allclose(output, expected, rtol=5e-13, atol=5e-11)


def test_wide_logical_lanes_share_launch_and_memory_bounds(tmp_path, monkeypatch):
    monkeypatch.setenv("FLAGFFT_HCU_DENSE_WAVE", "1")
    monkeypatch.setenv("FLAGFFT_HCU_DENSE_WAVE_FP64", "1")
    monkeypatch.setenv("FLAGFFT_HCU_WIDE_WAVE_LANES", "auto")
    token = set_profile(BackendProfile(backend="hcu", device_arch="gfx936", warp_size=64,
                                      max_dynamic_shared_memory=65536, policy="native"))
    try:
        metadata = emit_jit_kernel(kernel="leaf", length=2048, factors=(16, 16, 8),
            lanes=128, num_warps=4, generic_radices=(), smem_size=2048,
            direction="forward", dtype="complex128", prime_n=0,
            four_step_n1=0, four_step_n2=0, out_dir=tmp_path)
    finally:
        reset_profile(token)
    assert "_lanes128_" in metadata["kernel_name"]
    assert metadata["num_warps"] == 8
    assert metadata["batch_per_block"] == 4
    assert 4 * 128 * 4 * 16 <= 32768
