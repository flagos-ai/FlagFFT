"""Check the isolated NPU transpose without changing FFT scheduling."""

import ast
import builtins
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from flagfft_codegen.kernels_layout import (
    _build_complex_pair_transpose_kernel_source,
    _build_tiled_transpose_kernel_source,
)
from flagfft_codegen.metadata import _module_source
from flagfft_codegen.kernels_npu_transpose2d import build_npu_table_transpose2d_source


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
    grid = (1, 1, 1)
    int32 = np.int32
    int64 = np.int64
    arange = staticmethod(np.arange)
    reshape = staticmethod(np.reshape)
    gather = staticmethod(lambda values, indices, axis: np.take_along_axis(values, indices, axis))
    split = staticmethod(lambda values: (values[..., 0], values[..., 1]))
    join = staticmethod(lambda real, imag: np.stack((real, imag), axis=-1))
    trans = staticmethod(np.transpose)

    def program_id(self, axis):
        return np.asarray(self.ids[axis]).view(Index)

    def num_programs(self, axis):
        return self.grid[axis]

    @staticmethod
    def load(pointer, mask=True, other=0):
        return np.where(mask, pointer.values[np.where(mask, pointer.offset, 0)], other)

    @staticmethod
    def store(pointer, values, mask):
        offsets, values, mask = np.broadcast_arrays(pointer.offset, values, mask)
        pointer.values[offsets[mask]] = values[mask]


@pytest.mark.parametrize("shape", [(1, 3), (17, 33), (65, 17), (257, 129), (129, 257)])
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
    selected = backend == "npu" and dtype == "complex64"
    metadata = emit._emit_tiled_transpose_jit_kernel(n0=257, n1=129, dtype=dtype,
        npu_vector_transpose=selected, out_dir=tmp_path)
    assert metadata["kernel_name"].startswith("npu_table_transpose2d_") == selected
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


@pytest.mark.parametrize("shape,tile", [((128, 128), 32), ((2048, 2048), 64),
    ((2048, 1025), 64), ((32, 46189), 64), ((16, 131072), 32)])
def test_npu_transpose_grid_matches_selected_tile(monkeypatch, tmp_path, shape, tile):
    from flagfft_codegen import emit
    monkeypatch.setattr(emit, "_npu_backend_active", lambda: True)
    rows, cols = shape
    meta = emit._emit_tiled_transpose_jit_kernel(n0=rows, n1=cols,
        npu_vector_transpose=True, out_dir=tmp_path)
    assert meta["tile_size"] == tile
    assert meta["grid_x_override"] == min(40, ((cols + tile - 1) // tile) * ((rows + tile - 1) // tile))
    assert meta["grid_y_override"] == 1
    assert meta["arg_names"] == ["in_ptr", "out_ptr", "idx_ptr", "nbatch"]
    assert meta["signature"] == "*fp32:16,*fp32:16,*i32:16,i32"
    assert meta["npu_table_transpose"] is True
    source = Path(meta["module_path"]).read_text()
    assert "tl.gather(data, index, 0)" in source
    assert "tl.split(" not in source and "tl.join(" not in source
    tree = ast.parse(source)
    # Tile decoding is scalar; the full local vector uses the shared GM table.
    assert "index = tl.load(idx_ptr + lane) >> 2" in source
    assert "range(pid," in source and "tl.num_programs(0)" in source


def test_existing_npu_transpose_source_is_preserved(monkeypatch, tmp_path):
    from flagfft_codegen import emit
    monkeypatch.setattr(emit, "_npu_backend_active", lambda: True)
    monkeypatch.setattr(emit, "_maca_backend_active", lambda: False)
    meta = emit._emit_tiled_transpose_jit_kernel(n0=2048, n1=1025, out_dir=tmp_path)
    _, source, _ = _build_tiled_transpose_kernel_source(2048, 1025, "complex64")
    assert Path(meta["module_path"]).read_text() == _module_source(source)
    assert "grid_x_override" not in meta and "grid_y_override" not in meta


@pytest.mark.parametrize("backend,dtype", [("cuda", "complex64"), ("ix", "complex64"),
    ("npu", "complex128")])
def test_explicit_vector_transpose_rejects_wrong_backend_or_dtype(monkeypatch, tmp_path, backend, dtype):
    from flagfft_codegen import emit
    monkeypatch.setattr(emit, "_npu_backend_active", lambda: backend == "npu")
    with pytest.raises(ValueError, match="NPU complex64"):
        emit._emit_tiled_transpose_jit_kernel(n0=128, n1=128, dtype=dtype,
            npu_vector_transpose=True, out_dir=tmp_path)


@pytest.mark.parametrize("vector", [False, True])
def test_npu_transpose_cli_selects_only_explicit_replacement(tmp_path, vector):
    root = Path(__file__).resolve().parents[2]
    command = [sys.executable, "-m", "flagfft_codegen.jit_source", "--kernel", "tiled_transpose",
        "--target", "npu:source-contract:32", "--reshape-n1", "2048", "--reshape-n2", "1025",
        "--out-dir", str(tmp_path)]
    if vector:
        command.append("--npu-vector-transpose")
    environment = dict(os.environ, PYTHONPATH=str(root / "python"), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(command, env=environment, text=True, capture_output=True, check=True)
    meta = json.loads(result.stdout)
    assert meta["kernel_name"].startswith("npu_table_transpose2d_") == vector
    assert meta["tile_size"] == (64 if vector else 32)
    assert ("grid_x_override" in meta) == vector


@pytest.mark.parametrize("shape", [(1, 3), (17, 33), (65, 17), (257, 129), (129, 257)])
@pytest.mark.parametrize("tile", [16, 32, 64])
@pytest.mark.parametrize("budget", [1, 3, 40])
def test_table_transpose_reuses_indices_across_tiles_and_masks_tail(shape, tile, budget):
    rows, cols = shape
    batch = 3
    rng = np.random.default_rng(20261018)
    source = rng.integers(0, 2**32, size=(batch, rows, cols, 2), dtype=np.uint32)
    expected = source.transpose(0, 2, 1, 3).copy().reshape(-1)
    rr, cc = np.meshgrid(np.arange(tile), np.arange(tile), indexing="ij")
    # First byte-offset table produced by the unchanged native C++ host builder.
    table = (((cc * tile + rr)[:, :, None] * 2 + np.arange(2)) * 4).astype(np.int32).reshape(-1)
    name, text, arguments, programs = build_npu_table_transpose2d_source(rows, cols, tile, budget)
    assert arguments == ["in_ptr", "out_ptr", "idx_ptr", "nbatch"]
    tree = ast.parse(text)
    tree.body[0].decorator_list = []
    language = Language()
    language.grid = (programs, 1, batch + 1)
    def tensor_range(start, stop, step):
        return (np.asarray(k).view(Index) for k in builtins.range(int(start), int(stop), int(step)))
    namespace = {"tl": language, "range": tensor_range}
    exec(compile(tree, "generated_table_transpose", "exec"), namespace)
    guard, marker = 128, np.uint32(0x3FADFF12)
    output = np.full(source.size + 2 * guard, marker, dtype=np.uint32)
    for b in range(batch + 1):
        for pid in range(programs):
            language.ids = (pid, 0, b)
            namespace[name](Pointer(source.reshape(-1).view(np.float32)),
                            Pointer(output.view(np.float32), guard), Pointer(table), batch)
    np.testing.assert_array_equal(output[guard:-guard], expected)
    assert np.all(output[:guard] == marker) and np.all(output[-guard:] == marker)


@pytest.mark.parametrize("shape,index_type", [((2048, 2048), "int32"), ((65536, 65536), "int64")])
def test_table_transpose_keeps_wide_outer_addresses(shape, index_type):
    _, source, _, _ = build_npu_table_transpose2d_source(*shape, 64)
    assert f").to(tl.{index_type})" in source
    assert "batch = tl.program_id(2).to(tl.int64)" in source


@pytest.mark.parametrize("arguments", [(0, 2, 32, 40), (2, 0, 32, 40), (2, 2, 3, 40), (2, 2, 32, 0)])
def test_table_transpose_rejects_invalid_contract(arguments):
    with pytest.raises(ValueError):
        build_npu_table_transpose2d_source(*arguments)
