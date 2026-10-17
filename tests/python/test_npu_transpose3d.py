"""Check the isolated native-table 3D transpose and its backend boundary."""

import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from flagfft_codegen.kernels_npu_layout import build_npu_table_transpose3d_source
class Index(np.ndarray):
    def to(self, dtype):
        return self.astype(dtype).view(Index)


class Pointer:
    def __init__(self, values, offset=0):
        self.values, self.offset = values, offset

    def __add__(self, offset):
        return Pointer(self.values, self.offset + offset)


class TableLanguage:
    ids = (0, 0, 0)
    int32 = np.int32
    int64 = np.int64
    arange = staticmethod(np.arange)
    reshape = staticmethod(np.reshape)
    gather = staticmethod(lambda values, indices, axis: np.take_along_axis(values, indices, axis))

    def program_id(self, axis):
        return np.asarray(self.ids[axis]).view(Index)

    @staticmethod
    def load(pointer, mask=True, other=0):
        return np.where(mask, pointer.values[np.where(mask, pointer.offset, 0)], other)

    @staticmethod
    def store(pointer, values, mask):
        offsets, values, mask = np.broadcast_arrays(pointer.offset, values, mask)
        pointer.values[offsets[mask]] = values[mask]


def native_table(order):
    axes = tuple(map(int, order))
    output = np.indices((16, 16, 16)).reshape(3, -1)
    original = output[[axes.index(axis) for axis in range(3)]]
    index = (original[0] * 16 + original[1]) * 16 + original[2]
    return np.stack((index * 8, index * 8 + 4), axis=-1).astype(np.uint32).reshape(-1)


@pytest.mark.parametrize("dims,batch", [((16, 16, 16), 3), ((17, 33, 65), 3),
    ((1, 17, 5), 2), ((4, 7, 9), 5), ((5, 1, 31), 2)])
@pytest.mark.parametrize("order", ["021", "210", "201", "120"])
def test_table_transpose_bits_tails_and_extra_batch(dims, batch, order):
    rng = np.random.default_rng(20261018)
    data = rng.integers(0, 2**32, (batch, *dims, 2), dtype=np.uint32)
    name, source, blocks = build_npu_table_transpose3d_source(*dims, order)
    tree = ast.parse(source)
    tree.body[0].decorator_list = []
    language = TableLanguage()
    namespace = {"tl": language}
    exec(compile(tree, "generated_transpose3d", "exec"), namespace)
    marker = np.uint32(0x3FA3FFFF)
    output = np.full(data.size + 128, marker, dtype=np.uint32)
    index = native_table(order)
    for z in range(batch + 1):
        for x in range(blocks):
            language.ids = (x, 0, z)
            namespace[name](Pointer(data.reshape(-1).view(np.float32)),
                            Pointer(output.view(np.float32), 64), Pointer(index), batch)
    expected = data.transpose(0, *[int(axis) + 1 for axis in order], 4).copy()
    np.testing.assert_array_equal(output[64:-64], expected.reshape(-1))
    assert np.all(output[:64] == marker) and np.all(output[-64:] == marker)
    assert "batch = tl.program_id(2).to(tl.int64)" in source
    assert "(c >> 1)" not in source and "(c & 1)" not in source


@pytest.mark.parametrize("dims,order", [((0, 2, 3), "021"), ((2, -1, 3), "021"),
    ((2, 3, 4), "102"), ((65536, 65536, 1), "021")])
def test_unsupported_table_contract_rejected(dims, order):
    with pytest.raises(ValueError):
        build_npu_table_transpose3d_source(*dims, order)


@pytest.mark.parametrize("backend,dtype", [("npu", "complex64"), ("cuda", "complex64"),
    ("ix", "complex64"), ("maca", "complex64"), ("npu", "complex128")])
def test_cli_table_transpose_scope(tmp_path, backend, dtype):
    repo = Path(__file__).resolve().parents[2]
    command = [sys.executable, "-m", "flagfft_codegen.jit_source", "--kernel", "transpose3d",
        "--npu-table-transpose3d", "--transpose3d-n0", "17", "--transpose3d-n1", "33",
        "--transpose3d-n2", "65", "--transpose3d-order", "021",
        "--target", f"{backend}:source-contract:1", "--dtype", dtype, "--out-dir", str(tmp_path)]
    result = subprocess.run(command, text=True, capture_output=True,
        env=dict(os.environ, PYTHONPATH=str(repo / "python"), PYTHONDONTWRITEBYTECODE="1"))
    if backend != "npu" or dtype != "complex64":
        assert result.returncode != 0 and "NPU FP32" in result.stderr
        return
    assert result.returncode == 0, result.stderr
    meta = json.loads(result.stdout)
    assert meta["signature"] == "*fp32:16,*fp32:16,*i32:16,i32"
    assert meta["grid_x_override"] == 2 * 3 * 5
    assert meta["tile_size"] == 16 and meta["num_warps"] == 4
    compile(Path(meta["module_path"]).read_text(), meta["module_path"], "exec")
