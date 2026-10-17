"""Preserve the generic layout ABI, bit patterns, tails and backend scope."""
import ast
from types import SimpleNamespace

import numpy as np
import pytest

from flagfft_codegen import kernels_layout
from flagfft_codegen.kernels_npu_fourstep_layout import build_npu_tiled_reshape_source
from test_npu_transpose2d import Index, Language, Pointer
from test_npu_grouped_fft64 import insert_slice


class Grid(Language):
    blocks = 1
    float32 = np.float32
    zeros = staticmethod(np.zeros)

    def num_programs(self, axis):
        assert axis == 0
        return self.blocks


@pytest.mark.parametrize('rows,cols', [(1, 3), (17, 33), (32, 16), (65, 17), (209, 221), (221, 209)])
@pytest.mark.parametrize('twiddle', [False, True])
@pytest.mark.parametrize('grid_mode', ['single', 'abi', 'extra'])
def test_generated_layout_bits_guards_and_grid(rows, cols, twiddle, grid_mode):
    rng = np.random.default_rng(20261018)
    batch, guard = 3, 128
    if twiddle:
        source = rng.standard_normal((batch, rows, cols, 2)).astype(np.float32)
    else:
        source = rng.integers(0, 2**32, (batch, rows, cols, 2), np.uint32).view(np.float32)
    tw = rng.standard_normal((rows, cols, 2)).astype(np.float32)
    ref = source.copy()
    if twiddle:
        a, b = source[..., 0], source[..., 1]
        ref[..., 0] = a * tw[..., 0] - b * tw[..., 1]
        ref[..., 1] = a * tw[..., 1] + b * tw[..., 0]
    ref = ref.transpose(0, 2, 1, 3).copy()
    marker = np.uint32(0x3fadff12)
    output = np.full(source.size + guard * 2, marker, np.uint32)
    name, body, _ = build_npu_tiled_reshape_source(rows, cols, twiddle=twiddle)
    tree = ast.parse(body); tree.body[0].decorator_list = []
    tl = Grid()
    tl.blocks = 1 if grid_mode == 'single' else (rows * cols + 255) // 256 + (3 if grid_mode == 'extra' else 0)
    scope = dict(tl=tl, tle=SimpleNamespace(dsa=SimpleNamespace(insert_slice=insert_slice)),
                 range=lambda start, stop, step: (np.asarray(i).view(Index) for i in range(start, stop, step)),
                 _cmul=lambda ar, ai, br, bi: (ar * br - ai * bi, ar * bi + ai * br))
    exec(compile(tree, name, 'exec'), scope)
    # Include one invalid outer batch; every load/store must remain masked.
    for b in range(batch + 1):
        for pid in range(tl.blocks):
            tl.ids = (pid, b, 0)
            args = [Pointer(source.reshape(-1))]
            if twiddle:
                args.append(Pointer(tw.reshape(-1)))
            scope[name](*args, Pointer(output.view(np.float32), guard), batch)
    np.testing.assert_array_equal(output[guard:-guard], ref.reshape(-1).view(np.uint32))
    assert np.all(output[:guard] == marker) and np.all(output[-guard:] == marker)


@pytest.mark.parametrize('backend', ['npu', 'cuda', 'ix', 'musa', 'ppu', 'hcu', 'maca'])
@pytest.mark.parametrize('dtype', ['complex64', 'complex128'])
@pytest.mark.parametrize('twiddle', [False, True])
def test_isolated_default_off_route(monkeypatch, backend, dtype, twiddle):
    build = (kernels_layout._build_twiddle_reshape_pack_kernel_source if twiddle
             else kernels_layout._build_reshape_pack_kernel_source)
    monkeypatch.setattr(kernels_layout, '_npu_backend_active', lambda: backend == 'npu')
    monkeypatch.delenv('FLAGFFT_NPU_TRITON_RESHAPE_TILE', raising=False)
    default = build(209, 221, dtype)
    monkeypatch.setenv('FLAGFFT_NPU_TRITON_RESHAPE_TILE', '0')
    assert build(209, 221, dtype) == default
    monkeypatch.setenv('FLAGFFT_NPU_TRITON_RESHAPE_TILE', '1')
    changed = build(209, 221, dtype)
    if backend == 'npu' and dtype == 'complex64':
        assert changed == build_npu_tiled_reshape_source(209, 221, twiddle=twiddle)
    else:
        assert changed == default


def test_large_local_coordinate_width():
    assert 'tl.int32' in build_npu_tiled_reshape_source(209, 221)[1]
    assert 'tl.int64' in build_npu_tiled_reshape_source(65536, 65536)[1]
    with pytest.raises(ValueError):
        build_npu_tiled_reshape_source(0, 7)
