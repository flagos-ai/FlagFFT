"""Check the isolated NPU transpose without changing FFT scheduling."""

import ast
from pathlib import Path

import numpy as np
import pytest

from flagfft_codegen.kernels_layout import (
    _build_complex_pair_transpose_kernel_source,
    _build_tiled_transpose_kernel_source,
)
from flagfft_codegen.metadata import _module_source


class Index(np.ndarray):
    def to(self, dtype):
        return self.astype(dtype).view(Index)


class Pointer:
    def __init__(self, values, offset=0):
        self.values, self.offset = values, offset

    def __add__(self, offset):
        return Pointer(self.values, self.offset + offset)


class Language:
    ids = (0, 0, 0)
    int32 = np.int32
    int64 = np.int64
    arange = staticmethod(np.arange)
    split = staticmethod(lambda values: (values[..., 0], values[..., 1]))
    join = staticmethod(lambda real, imag: np.stack((real, imag), axis=-1))
    trans = staticmethod(np.transpose)

    def program_id(self, axis):
        return np.asarray(self.ids[axis]).view(Index)

    @staticmethod
    def load(pointer, mask, other):
        return np.where(mask, pointer.values[np.where(mask, pointer.offset, 0)], other)

    @staticmethod
    def store(pointer, values, mask):
        offsets, values, mask = np.broadcast_arrays(pointer.offset, values, mask)
        pointer.values[offsets[mask]] = values[mask]


@pytest.mark.parametrize("shape", [(1, 3), (17, 33), (65, 17), (257, 129)])
@pytest.mark.parametrize("tile", [16, 32, 64])
def test_pair_transpose_preserves_bits_and_guards(shape, tile):
    rows, cols = shape
    batch = 3
    rng = np.random.default_rng(20261017)
    # Include NaNs, signed zeros and arbitrary bit patterns without arithmetic.
    source = rng.integers(0, 2**32, size=(batch, rows, cols, 2), dtype=np.uint32)
    expected = source.transpose(0, 2, 1, 3).copy()
    name, source_code, names = _build_complex_pair_transpose_kernel_source(rows, cols, tile)
    assert names == ["in_ptr", "out_ptr", "nbatch"]
    tree = ast.parse(source_code)
    tree.body[0].decorator_list = []
    language = Language()
    namespace = {"tl": language}
    exec(compile(tree, "generated_transpose2d", "exec"), namespace)
    guard, marker = 128, np.uint32(0x3fadff12)
    output = np.full(source.size + 2 * guard, marker, dtype=np.uint32)
    # An extra batch program must be fully masked.
    for z in range(batch + 1):
        for y in range((rows + tile - 1) // tile):
            for x in range((cols + tile - 1) // tile):
                language.ids = (x, y, z)
                namespace[name](Pointer(source.reshape(-1).view(np.float32)),
                                Pointer(output.view(np.float32), guard), batch)
    np.testing.assert_array_equal(output[guard:-guard], expected.reshape(-1))
    assert np.all(output[:guard] == marker) and np.all(output[-guard:] == marker)


@pytest.mark.parametrize("backend", ["npu", "cuda", "ix", "musa", "ppu", "hcu", "maca"])
@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_pair_transpose_is_scoped_to_npu_fp32(monkeypatch, tmp_path, backend, dtype):
    from flagfft_codegen import emit

    monkeypatch.setattr(emit, "_npu_backend_active", lambda: backend == "npu")
    monkeypatch.setattr(emit, "_maca_backend_active", lambda: False)
    metadata = emit._emit_tiled_transpose_jit_kernel(n0=257, n1=129, dtype=dtype, out_dir=tmp_path)
    selected = backend == "npu" and dtype == "complex64"
    assert metadata["kernel_name"].startswith("_complex_pair_transpose") == selected
    assert metadata["tile_size"] == 32 and metadata["batch_per_block"] == 1
    assert metadata["signature"].endswith(",i32")
    actual = Path(metadata["module_path"]).read_text()
    compile(actual, metadata["module_path"], "exec")
    if not selected:
        _, previous_source, _ = _build_tiled_transpose_kernel_source(257, 129, dtype)
        assert actual == _module_source(previous_source)


@pytest.mark.parametrize("rows,cols,tile", [(0, 2, 32), (2, 0, 32), (2, 2, 0), (2, 2, 3)])
def test_invalid_tile_rejected(rows, cols, tile):
    with pytest.raises(ValueError):
        _build_complex_pair_transpose_kernel_source(rows, cols, tile)


@pytest.mark.parametrize("shape,index_type", [((2048, 2048), "int32"), ((65536, 65536), "int64")])
def test_coordinate_width_keeps_outer_batch_wide(shape, index_type):
    _, source, _ = _build_complex_pair_transpose_kernel_source(*shape, 64)
    assert f"tl.program_id(1).to(tl.{index_type})" in source
    assert "batch = tl.program_id(2).to(tl.int64)" in source
    assert "input_batch = in_ptr + batch *" in source
