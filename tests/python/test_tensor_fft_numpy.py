"""Execute generated register graphs with NumPy, including the IO contract."""

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from flagfft_codegen.kernels_fft_tiles import TensorFFTConfig, emit_tensor_fft


class Tensor(np.ndarray):
    def to(self, dtype):
        return self.astype(dtype).view(Tensor)


class Pointer:
    def __init__(self, values, offset=0):
        self.values, self.offset = values, offset

    def __add__(self, offset):
        return Pointer(self.values, self.offset + offset)


class Language:
    float32 = np.float32
    int32 = np.int32
    int64 = np.int64
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

    where = staticmethod(np.where)
    minimum = staticmethod(np.minimum)
    maximum = staticmethod(np.maximum)
    gather = staticmethod(np.take_along_axis)
    cos = staticmethod(np.cos)
    sin = staticmethod(np.sin)
    trans = staticmethod(np.transpose)
    reshape = staticmethod(np.reshape)
    broadcast_to = staticmethod(np.broadcast_to)
    split = staticmethod(lambda values: (values[..., 0], values[..., 1]))
    join = staticmethod(lambda even, odd: np.stack((even, odd), axis=-1))

    @staticmethod
    def load(pointer, mask=True, other=0):
        return np.where(mask, pointer.values[np.where(mask, pointer.offset, 0)], other).view(Tensor)

    @staticmethod
    def store(pointer, values, mask):
        offset, values, mask = np.broadcast_arrays(pointer.offset, values, mask)
        pointer.values[offset[mask]] = values[mask]


@pytest.mark.parametrize("length,factors", [(1, ()), (2, ()), (16, ()), (32, (8, 4)), (64, ()),
    (12, (3, 4)), (45, (5, 9)), (209, (11, 19))])
@pytest.mark.parametrize("mode,direction", [("complex", "forward"), ("complex", "inverse"),
    ("r2c", "forward"), ("c2r", "inverse")])
@pytest.mark.parametrize("stride,permuted", [(1, 0), (3, 0), (1, 7)])
@pytest.mark.parametrize("exchange", ["static", "split_register", "split_tensor"])
def test_register_graph_matches_numpy(tmp_path, length, factors, mode, direction, stride, permuted, exchange):
    if exchange in {"split_register", "split_tensor"} and length & (length - 1):
        pytest.skip("NPU register exchange is qualified only for powers of two")
    config = TensorFFTConfig(length, 8, mode=mode, direction=direction, input_stride=stride,
        output_stride=stride if not permuted else 1, output_rows_per_batch=permuted,
        exchange=exchange, factors=factors)
    metadata = emit_tensor_fft(config, tmp_path)
    tree = ast.parse(Path(metadata["module_path"]).read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    for function in functions:
        function.decorator_list = []
    language = Language()
    namespace = {"tl": language}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "generated_fft", "exec"), namespace)
    rows = 19  # Leaves a tail in both the program pack and permuted batch.
    rng = np.random.default_rng(20261017)
    real = rng.normal(size=(rows, length)).astype(np.float32)
    if mode == "complex":
        logical = (real + 1j * rng.normal(size=real.shape)).astype(np.complex64)
        expected = np.fft.fft(logical) if direction == "forward" else np.fft.ifft(logical) * length
    elif mode == "r2c":
        logical, expected = real, np.fft.rfft(real)
    else:
        logical = np.fft.rfft(real).astype(np.complex64)
        expected = np.fft.irfft(logical, n=length) * length
    in_length, out_length = logical.shape[1], expected.shape[1]
    source = np.zeros(((rows + stride - 1) // stride) * in_length * stride, logical.dtype)
    for row in range(rows):
        base = row // stride * in_length * stride + row % stride
        source[base:base + in_length * stride:stride] = logical[row]
    output_step = permuted or stride
    output = np.full(((rows + output_step - 1) // output_step) * output_step * out_length,
        np.nan, np.float32 if mode == "c2r" else np.complex64)
    untouched = np.ones(output.size, dtype=bool)
    kernel = namespace[metadata["kernel_name"]]
    sign = -1 if direction == "forward" else 1
    twiddle = np.exp(sign * 2j * np.pi * np.arange(length // 2) / length).astype(np.complex64)
    # Two host chunks exercise row_offset with global input/output indexing.
    programs = ((rows + metadata["row_group_extent"] - 1) // metadata["row_group_extent"]) * metadata["programs_per_row_group"]
    for offset, count in ((0, 1), (1, programs - 1)):
        for pid in range(count):
            language.pid = pid
            kernel(Pointer(source.view(np.float32)), Pointer(output.view(np.float32)),
                Pointer(twiddle.view(np.float32)), rows, offset)
    actual = np.empty_like(expected)
    for row in range(rows):
        base = row // output_step * out_length * output_step + row % output_step
        indices = np.arange(base, base + out_length * output_step, output_step)
        actual[row] = output[indices]
        untouched[indices] = False
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-4)
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 5e-7
    assert np.isnan(output[untouched]).all()
    assert metadata["signature"].split(",")[-2:] == ["i32", "i64"]
    assert metadata["batch_per_block"] == config.rows_per_program
    assert json.loads(Path(metadata["module_path"]).with_suffix(".json").read_text()) == metadata


@pytest.mark.parametrize("options", [dict(length=23), dict(length=8, factors=(3, 3)),
    dict(length=8, mode="c2r"), dict(length=8, rows_per_program=3),
    dict(length=8, input_stride=0), dict(length=8, output_stride=2, output_rows_per_batch=7)])
def test_invalid_config_is_rejected(options):
    with pytest.raises(ValueError):
        TensorFFTConfig(**options)
