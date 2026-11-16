# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from flagfft_codegen.kernels_hcu_3d_ct import build_ct_source, ct_geometry


class Pointer:
    def __init__(self, data, offset=0):
        self.data, self.offset = data, offset

    def __add__(self, offset):
        return Pointer(self.data, self.offset + offset)


class NumpyTL:
    arange = staticmethod(np.arange)
    reshape = staticmethod(np.reshape)
    zeros_like = staticmethod(np.zeros_like)
    broadcast_to = staticmethod(np.broadcast_to)
    gather = staticmethod(np.take_along_axis)
    where = staticmethod(np.where)
    cos = staticmethod(np.cos)
    sin = staticmethod(np.sin)

    def program_id(self, axis):
        return self.pid if axis == 0 else self.batch

    def load(self, ptr):
        return ptr.data[ptr.offset]

    def store(self, ptr, value):
        ptr.data[ptr.offset] = value


@pytest.mark.parametrize("shape", [(64, 32, 64), (32, 64, 64)])
@pytest.mark.parametrize("direction", ["forward", "inverse"])
def test_generated_two_pass_ct_matches_numpy_for_every_batch(shape, direction):
    n0, n1, n2 = shape
    h0, h1, _ = ct_geometry(*shape)
    rng = np.random.default_rng(21)
    data = (rng.standard_normal((2, *shape)) + 1j * rng.standard_normal((2, *shape))).astype(np.complex64)
    raw = data.view(np.float32).reshape(-1)
    scratch = np.zeros_like(raw)
    output = np.zeros_like(raw)
    tl = NumpyTL()
    for first, inp, out in [(True, raw, scratch), (False, scratch, output)]:
        name, source, grid = build_ct_source(n0, n1, n2, h0, h1, direction, first)
        source = source.replace("@triton.jit\n", "").replace(
            "tl.program_id(2).to(tl.int64)", "tl.program_id(2)")
        namespace = {"tl": tl}
        exec(compile(source, "<hcu-ct-numpy-emulation>", "exec"), namespace)
        kernel = namespace[name]
        for tl.batch in range(2):
            for tl.pid in range(grid):
                kernel(Pointer(inp), Pointer(out), 2)
    expected = np.fft.fftn(data, axes=(1, 2, 3)) if direction == "forward" else np.fft.ifftn(data, axes=(1, 2, 3)) * np.prod(shape)
    actual = output.view(np.complex64).reshape(data.shape)
    np.testing.assert_allclose(actual, expected, atol=2e-3, rtol=8e-5)
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 8e-7


@pytest.mark.parametrize("shape", [(256, 256, 256), (128, 2048, 64),
                                  (128, 128, 128), (64, 512, 32)])
def test_ct_geometry_bounds_both_passes_without_case_tables(shape):
    n0, n1, n2 = shape
    h0, h1, tile = ct_geometry(*shape)
    assert n0 % h0 == n1 % h1 == 0
    assert h0 * h1 * n2 <= 8192
    assert (n0 // h0) * (n1 // h1) * tile <= 8192
    assert tile >= 4 and n2 % tile == 0


@pytest.mark.parametrize("shape", [(127, 256, 64), (128, 2048, 997), (4096, 4096, 64)])
def test_ct_geometry_preserves_fallback_for_unsupported_or_excessive_tiles(shape):
    assert ct_geometry(*shape) is None


@pytest.mark.parametrize("shape", [(128, 128, 128), (64, 512, 32), (256, 256, 256)])
def test_ct_metadata_launches_exactly_one_block_per_local_tile(tmp_path, shape):
    from flagfft_codegen.backend_profile import BackendProfile, reset_profile, set_profile
    from flagfft_codegen.kernels_hcu_3d_ct import emit_ct_kernel
    token = set_profile(BackendProfile(backend="hcu", device_arch="gfx936", warp_size=64,
                                     max_dynamic_shared_memory=65536, policy="native"))
    try:
        n0, n1, n2 = shape
        h0, h1, tile = ct_geometry(*shape)
        total = n0 * n1 * n2
        for first in (True, False):
            metadata = emit_ct_kernel(n0=n0, n1=n1, n2=n2, h0=h0, h1=h1,
                                      direction="forward", first=first,
                                      dtype="complex64", out_dir=tmp_path)
            points_per_block = h0 * h1 * n2 if first else (n0 // h0) * (n1 // h1) * tile
            assert metadata["grid_x_override"] * points_per_block == total
            assert metadata["warp_size"] == 64
            assert metadata["num_warps"] * metadata["warp_size"] <= 1024
    finally:
        reset_profile(token)
