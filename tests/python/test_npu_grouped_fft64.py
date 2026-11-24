"""Check emitted grouped FFT bodies on CPU before remote device validation."""

import ast
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from flagfft_codegen.kernels_npu_grouped import (
    build_npu_grouped_fft64_source, emit_npu_grouped_fft64,
)
from test_npu_transpose2d import Index, Pointer


def tables(group, stride, inverse=False, mode=0):
    assert group in (4, 8) and stride in (1, 64) and mode in (0, 1, 2)
    assert mode == 0 or stride == 1
    n = 256
    second = 15*n
    imag = second + (n if group == 8 else 0)
    index = np.zeros(imag + (n*(2 if group == 8 else 1) if mode == 2 else 0), np.int32)
    twiddle = np.zeros(12*n + (n if mode == 2 else 0), np.float32)
    sign = 1 if inverse else -1
    for part in range(group//4):
        for g in range(4):
            for i in range(64):
                rev = int(f'{i:06b}'[::-1], 2)
                destination = g*64+i
                row = g+4*part
                if mode == 1:
                    source = row*64+rev
                elif mode == 2:
                    source = 2*(row*33+min(rev,64-rev))
                    index[imag+part*n+destination] = (source+1)*4
                    twiddle[12*n+destination] = -1 if rev > 32 else 1
                else:
                    source = 2*(rev*group+row if stride == 64 else row*64+rev)
                index[(0 if part == 0 else second)+destination] = source*4
    if mode == 2:
        index[n:2*n] = np.arange(n)*4
    elif mode == 1:
        values = [4*(g*64+k+component*n) for g in range(4) for k in range(33) for component in range(2)]
        index[n:n+len(values)] = values
    else:
        for i in range(2*n):
            component = i & 1
            if stride == 64:
                row, col = i//8, (i//2)%4
                source = col*64+row+component*n
            else:
                source = (i//128)*64+(i//2)%64+component*n
            index[n+i] = source*4
    for stage in range(6):
        length = 1 << (stage+1)
        half = length//2
        for g in range(4):
            for i in range(64):
                offset = i%length
                a = (i//length)*length+(offset-half if offset>=half else offset)
                destination = g*64+i
                index[3*n+stage*n+destination] = (g*64+a)*4
                index[9*n+stage*n+destination] = (g*64+a+half)*4
                angle = sign*2*math.pi*(offset%half)/length
                negate = -1 if offset>=half else 1
                twiddle[stage*n+destination] = negate*np.float32(math.cos(angle))
                twiddle[6*n+stage*n+destination] = negate*np.float32(math.sin(angle))
    return index, twiddle


class Language:
    pid = 0
    int64 = np.int64
    float32 = np.float32
    arange = staticmethod(np.arange)
    reshape = staticmethod(np.reshape)
    gather = staticmethod(lambda values, indices, axis: np.take(values, indices, axis=axis))
    static_range = staticmethod(range)
    full = staticmethod(np.full)
    zeros = staticmethod(np.zeros)

    def program_id(self, axis):
        assert axis == 0
        return np.asarray(self.pid).view(Index)

    @staticmethod
    def load(pointer, mask=True, other=0):
        return np.where(mask, pointer.values[np.where(mask, pointer.offset, 0)], other)

    @staticmethod
    def store(pointer, values, mask=True):
        offsets, values, mask = np.broadcast_arrays(pointer.offset, values, mask)
        pointer.values[offsets[mask]] = values[mask]


def insert_slice(destination, source, offsets, sizes, strides):
    assert len(offsets) == 1 and strides == [1]
    result = destination.copy()
    result[offsets[0]:offsets[0] + sizes[0]] = source
    return result


@pytest.mark.parametrize("group", [4, 8])
@pytest.mark.parametrize("stride,inverse,mode", [
    (1, False, 0), (1, True, 0), (64, False, 0), (64, True, 0),
    (1, False, 1), (1, True, 2),
])
def test_emitted_body_preserves_grouped_table_contract(group, stride, inverse, mode):
    rng = np.random.default_rng(20261017)
    count = 64 if stride == 64 else group * 3
    if mode == 1:
        data = rng.standard_normal((count, 64)).astype(np.float32)
        expected = np.fft.rfft(data, axis=1).astype(np.complex64).view(np.float32)
    elif mode == 2:
        real = rng.standard_normal((count, 64)).astype(np.float32)
        data = np.fft.rfft(real, axis=1).astype(np.complex64).view(np.float32)
        expected = (np.fft.irfft(data.view(np.complex64), n=64, axis=1) * 64).astype(np.float32)
    else:
        z = (rng.standard_normal((count, 64))
             + 1j * rng.standard_normal((count, 64))).astype(np.complex64)
        data = z.view(np.float32)
        axis = 0 if stride == 64 else 1
        expected = ((np.fft.ifft(z, axis=axis) * 64) if inverse
                    else np.fft.fft(z, axis=axis)).astype(np.complex64).view(np.float32)
    source = build_npu_grouped_fft64_source(group=group, stride=stride, mode=mode)
    assert "//" not in source and "%" not in source
    tree = ast.parse(source)
    tree.body[0].decorator_list = []
    language = Language()
    namespace = {"tl": language, "tle": SimpleNamespace(dsa=SimpleNamespace(insert_slice=insert_slice))}
    exec(compile(tree, "generated_fft64", "exec"), namespace)
    index, twiddle = tables(group, stride, inverse, mode)
    guard, marker = 64, np.float32(31415.25)
    output = np.full(expected.size + 2 * guard, marker, dtype=np.float32)
    for pid in range(count // group):
        language.pid = pid
        namespace["npu_grouped_fft64_kernel"](
            Pointer(data.reshape(-1)), Pointer(output, guard),
            Pointer(index), Pointer(twiddle), count // group,
        )
    np.testing.assert_allclose(output[guard:-guard], expected.reshape(-1), rtol=2e-5, atol=8e-6)
    assert np.all(output[:guard] == marker) and np.all(output[-guard:] == marker)


@pytest.mark.parametrize("group,stride,mode", [(1, 1, 0), (16, 1, 0), (4, 2, 0),
    (8, 64, 1), (4, 64, 2), (8, 1, 3)])
def test_unsupported_contract_is_rejected(group, stride, mode):
    with pytest.raises(ValueError):
        build_npu_grouped_fft64_source(group=group, stride=stride, mode=mode)


@pytest.mark.parametrize("mode,direction", [(1, "inverse"), (2, "forward"), (0, "other")])
def test_inconsistent_real_direction_is_rejected(tmp_path, mode, direction):
    with pytest.raises(ValueError):
        emit_npu_grouped_fft64(group=4, stride=1, mode=mode, direction=direction, out_dir=tmp_path)


@pytest.mark.parametrize("backend,dtype", [("npu", "complex64"), ("cuda", "complex64"),
    ("ix", "complex64"), ("maca", "complex64"), ("npu", "complex128")])
def test_cli_is_scoped_to_npu_fp32(tmp_path, backend, dtype):
    root = Path(__file__).resolve().parents[2]
    command = [sys.executable, "-m", "flagfft_codegen.jit_source", "--kernel", "npu_grouped_fft64",
        "--npu-fft-group", "8", "--npu-fft-stride", "1", "--npu-fft-mode", "2",
        "--direction", "inverse", "--target", f"{backend}:source-contract:32", "--dtype", dtype,
        "--out-dir", str(tmp_path)]
    result = subprocess.run(command, text=True, capture_output=True,
        env=dict(os.environ, PYTHONPATH=str(root / "python"), PYTHONDONTWRITEBYTECODE="1"))
    if backend != "npu" or dtype != "complex64":
        assert result.returncode != 0 and "NPU FP32" in result.stderr
        return
    assert result.returncode == 0, result.stderr
    meta = json.loads(result.stdout)
    assert meta["signature"] == "*fp32:16,*fp32:16,*i32:16,*fp32:16,i32"
    assert (meta["batch_per_block"], meta["num_warps"], meta["num_stages"]) == (8, 4, 1)
    compile(Path(meta["module_path"]).read_text(), meta["module_path"], "exec")
