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
    pid = 0
    float64 = np.float64
    arange = staticmethod(np.arange)
    full = staticmethod(np.full)
    broadcast_to = staticmethod(np.broadcast_to)
    gather = staticmethod(np.take_along_axis)
    where = staticmethod(np.where)
    reshape = staticmethod(np.reshape)
    join = staticmethod(lambda a, b: np.stack((a, b), axis=-1))
    trans = staticmethod(np.transpose)

    @staticmethod
    def program_id(axis):
        return NumpyTL.pid

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
        shape = np.shape(ptr.offset)
        assert np.broadcast_shapes(shape, np.shape(value), np.shape(mask)) == shape
        index, value, valid = np.broadcast_arrays(ptr.offset, value, mask)
        ptr.data[index[valid]] = value[valid]


@pytest.mark.parametrize("mode,inverse", [("contiguous", False), ("contiguous", True),
                                          ("contiguous_r2c", False), ("c2r", True),
                                          ("packed_r2c", False), ("packed_c2r", True)])
@pytest.mark.parametrize("period", [0, 4, 8])
def test_two_wave_generated_fft_matches_numpy_with_masked_tail(mode, inverse, period):
    packed = mode.startswith("packed_")
    n, batch, pack = (1024 if packed else 2048), 3, 4
    rng = np.random.default_rng(716)
    real = rng.normal(size=(batch, 2*n if packed else n))
    if mode in {"contiguous_r2c", "packed_r2c"}:
        data = real
        expected = np.fft.rfft(real, axis=-1)
    elif mode in {"c2r", "packed_c2r"}:
        data = np.fft.rfft(real, axis=-1)
        expected = real * real.shape[-1]
        if packed:
            expected = expected.view(np.complex128)
    else:
        data = real + 1j * rng.normal(size=(batch, n))
        expected = np.fft.ifft(data, axis=-1) * n if inverse else np.fft.fft(data, axis=-1)
    input_distance = data.shape[-1]
    output_distance = expected.shape[-1]
    output = np.full(expected.shape, np.nan, dtype=expected.dtype)
    name, source = build_ix_warp_source(n, pack, mode, inverse, wide=True,
                                      lane_width=64 if packed or period else 128,
                                      output_radix_chunk=4, root_period=period)
    source = "\n".join(line for line in source.splitlines()
                       if not line.startswith(("import ", "@triton.jit")))
    roots = np.exp((1j if inverse else -1j) * 2 * np.pi * np.arange(n) / n)
    namespace = {"tl": NumpyTL, "wave_tw_r_ptr": Pointer(roots.real.copy()),
                 "wave_tw_i_ptr": Pointer(roots.imag.copy())}
    packed_roots = np.exp(-1j * np.pi * np.arange(n) / n)
    namespace["packed_twiddle_ptr"] = Pointer(packed_roots.view(np.float64))
    exec(compile(source, "<emitted-wide-fft>", "exec"), namespace)
    namespace[name](Pointer(data.view(np.float64)), Pointer(output.view(np.float64)),
                    1, batch, input_distance, output_distance)
    np.testing.assert_allclose(output, expected, rtol=5e-13, atol=5e-11)
    error = output - expected
    assert np.max(np.linalg.norm(error, axis=-1) / np.linalg.norm(expected, axis=-1)) < 4e-15
    assert np.max(np.max(np.abs(error), axis=-1) / np.max(np.abs(expected), axis=-1)) < 6e-15


@pytest.mark.parametrize("n", [256, 2048])
@pytest.mark.parametrize("mode,inverse", [("outer", False), ("outer", True),
                                          ("inner", False), ("inner", True), ("r2c", False)])
@pytest.mark.parametrize("period", [0, 4])
def test_wide_transposed_fft_matches_numpy_with_partial_tiles(monkeypatch, n, mode, inverse, period):
    pack = 4
    span = 4 if mode == "r2c" else 3
    batch = 10 if mode == "r2c" else 15 if mode == "inner" else 9
    rng = np.random.default_rng(20261116)
    data = rng.normal(size=(batch, n))
    if mode == "r2c":
        transformed = np.fft.rfft(data, axis=-1)
    else:
        data = data + 1j * rng.normal(size=(batch, n))
        transformed = np.fft.ifft(data, axis=-1) * n if inverse else np.fft.fft(data, axis=-1)
    if mode == "inner":
        expected = transformed.reshape(batch // span, span, n).transpose(2, 1, 0).ravel()
        programs = ((batch + pack * span - 1) // (pack * span)) * span
    else:
        rows = ((batch + span - 1) // span) * span
        padded = np.full((rows, transformed.shape[-1]), np.nan, dtype=np.complex128)
        padded[:batch] = transformed
        expected = padded.reshape(-1, span, transformed.shape[-1]).transpose(0, 2, 1).ravel()
        programs = (rows // span) * ((span + pack - 1) // pack)
    output = np.full(expected.shape, np.nan, dtype=np.complex128)
    lanes = 128 if n > 1024 and not period else 64
    name, source = build_ix_warp_source(n, pack, mode, inverse, wide=True,
                                      lane_width=lanes, output_radix_chunk=4, root_period=period)
    source = "\n".join(line for line in source.splitlines()
                       if not line.startswith(("import ", "@triton.jit")))
    roots = np.exp((1j if inverse else -1j) * 2 * np.pi * np.arange(n) / n)
    namespace = {"tl": NumpyTL, "wave_tw_r_ptr": Pointer(roots.real.copy()),
                 "wave_tw_i_ptr": Pointer(roots.imag.copy())}
    exec(compile(source, "<emitted-wide-permutation>", "exec"), namespace)
    for pid in range(programs):
        monkeypatch.setattr(NumpyTL, "pid", pid)
        namespace[name](Pointer(data.view(np.float64)), Pointer(output.view(np.float64)),
                        span, batch, n, transformed.shape[-1])
    np.testing.assert_allclose(output, expected, rtol=5e-13, atol=5e-11)


@pytest.mark.parametrize("kind,n,pack,lanes", [("leaf_permuted_store", 256, 16, 64),
    ("leaf_permuted_store", 1024, 8, 64),
    ("leaf_r2c_permuted_store", 256, 16, 64)])
def test_wide_permutation_preserves_double_roots_and_payload_bounds(tmp_path, monkeypatch, kind, n, pack, lanes):
    monkeypatch.setenv("FLAGFFT_HCU_3D_WARP_FFT", "1")
    monkeypatch.setenv("FLAGFFT_HCU_3D_WARP_FP64", "1")
    token = set_profile(BackendProfile(backend="hcu", device_arch="gfx936", warp_size=64,
                                      max_dynamic_shared_memory=65536, policy="native"))
    factors = (16, 16, 8) if n == 2048 else (16, 8, 8) if n == 1024 else (16, 16)
    try:
        metadata = emit_jit_kernel(kernel=kind, length=n, factors=factors,
            lanes=16, num_warps=1, generic_radices=(), smem_size=n,
            direction="forward", dtype="complex128", prime_n=0,
            four_step_n1=0, four_step_n2=0, perm_form="outer",
            permutation_span=128, permutation_batch=32768, out_dir=tmp_path)
    finally:
        reset_profile(token)
    source = Path(metadata["module_path"]).read_text()
    compile(source, "<hcu-wide-permutation>", "exec")
    assert metadata["kernel_name"].startswith("hcu_warp")
    assert "tl.float32" not in source and "tl.uint64" not in source
    assert f"wave_tw_r_ptr=tw{len(factors)-1}_r_ptr+" in source
    assert metadata["batch_per_block"] == pack
    assert metadata["num_warps"] == min(8, pack * lanes // 64)
    assert n * pack * 2 <= metadata["num_warps"] * 64 * 32


def test_wide_permutation_keeps_native_fallback_above_one_wave_payload(monkeypatch):
    from flagfft_codegen.kernels_common import LeafPlan
    from flagfft_codegen.kernels_hcu_warp import hcu_warp_store_pack
    monkeypatch.setenv("FLAGFFT_HCU_3D_WARP_FFT", "1")
    monkeypatch.setenv("FLAGFFT_HCU_3D_WARP_FP64", "1")
    token = set_profile(BackendProfile(backend="hcu", device_arch="gfx936", warp_size=64,
                                      max_dynamic_shared_memory=65536, policy="native"))
    try:
        plan = LeafPlan(2048, (16, 16, 8), 1, 128, 4, (), 2048, dtype="complex128")
        assert hcu_warp_store_pack(plan, "permuted_store", "inner", 64, 32768) == 0
        plan = LeafPlan(256, (16, 16), 1, 16, 1, (), 256, dtype="complex128")
        assert hcu_warp_store_pack(plan, "permuted_store_r2c", "outer", 997, 997*16) == 0
    finally:
        reset_profile(token)


@pytest.mark.parametrize("kind,pack", [("leaf", 4), ("leaf_permuted_store", 8)])
def test_periodic_coefficients_keep_large_wide_fft_inside_one_wave(tmp_path, monkeypatch, kind, pack):
    monkeypatch.setenv("FLAGFFT_HCU_DENSE_WAVE", "leaves")
    monkeypatch.setenv("FLAGFFT_HCU_DENSE_WAVE_FP64", "1")
    monkeypatch.setenv("FLAGFFT_HCU_3D_WARP_FFT", "1")
    monkeypatch.setenv("FLAGFFT_HCU_3D_WARP_FP64", "1")
    monkeypatch.setenv("FLAGFFT_HCU_WAVE_ROOT_PERIOD", "4")
    monkeypatch.setenv("FLAGFFT_HCU_WIDE_WAVE_LANES", "64")
    token = set_profile(BackendProfile(backend="hcu", device_arch="gfx936", warp_size=64,
                                      max_dynamic_shared_memory=65536, policy="native"))
    try:
        metadata = emit_jit_kernel(kernel=kind, length=2048, factors=(16, 16, 8),
            lanes=128, num_warps=4, generic_radices=(), smem_size=2048,
            direction="inverse", dtype="complex128", prime_n=0,
            four_step_n1=0, four_step_n2=0, perm_form="outer",
            permutation_span=1025 if kind.endswith("store") else 0,
            permutation_batch=4100 if kind.endswith("store") else 0, out_dir=tmp_path)
    finally:
        reset_profile(token)
    source = Path(metadata["module_path"]).read_text()
    assert "_period4_" in metadata["kernel_name"]
    assert "_lanes128" not in metadata["kernel_name"]
    assert metadata["batch_per_block"] == pack
    assert metadata["num_warps"] == pack
    assert "base_wr=wr" in source and "wr*base_wr-wi*base_wi" in source


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
