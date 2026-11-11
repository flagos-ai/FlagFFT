"""Public-plan admission and live accuracy for the batched dense prefix.

Plan comparisons toggle the new option in one process, exercising cache
identity as well as selection. The large outside-domain case is not an
isolated test of the dense m limit: the overall DIF length limit also applies.
"""
from contextlib import contextmanager
import ctypes
import math
import os
import re

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torch_npu')


@pytest.fixture(autouse=True)
def selected_options(monkeypatch):
    for option in ('DIF_CUBE_PREFIX', 'DIF_LATE_AUTO_BLOCK', 'DIF_PAD_PREFIX',
                   'DIF_FUSED_FINISH', 'DIF_VECTOR_FIRST17', 'DIF_VECTOR_SECOND15',
                   'DIF_GROUP_BRANCHES', 'SYMMETRIC_CODELETS', 'DIF_FIRST_AUTO_BLOCK',
                   'DIF_BATCH_PACK', 'DIF_COMPONENT17', 'DIF_MATRIX_FIRST17',
                   'DIF_CODELET_REUSE', 'DIF_VECTOR_BASIS', 'DIF_FACTOR_COST',
                   'DIF_PREFIX_PITCH', 'DIF_BATCH_FINISH'):
        monkeypatch.setenv('FLAGFFT_NPU_' + option, '1')
    monkeypatch.setenv('FLAGFFT_NPU_DIF_DENSE_SHORT_PREFIX', '0')
    monkeypatch.setenv('FLAGFFT_NPU_DIF_GLOBAL_INVERSE', '1')
    # Isolate the prefix policy's two-pass reduction. The subsequent late
    # pair fusion has its own admission and live-execution checks.
    monkeypatch.setenv('FLAGFFT_NPU_DIF_DENSE_LATE_PAIR', '0')


@pytest.fixture(scope='module')
def library():
    path = os.environ.get('FLAGFFT_LIBRARY')
    if not path:
        pytest.skip('Set FLAGFFT_LIBRARY to a built NPU library')
    lib = ctypes.CDLL(path)
    lib.flagfftPlanMany.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.flagfftPlanMany.restype = ctypes.c_int
    lib.flagfftGetPlanDescription.argtypes = [ctypes.c_void_p]
    lib.flagfftGetPlanDescription.restype = ctypes.c_char_p
    lib.flagfftSetStream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.flagfftSetStream.restype = ctypes.c_int
    lib.flagfftExecC2C.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.flagfftExecC2C.restype = ctypes.c_int
    lib.flagfftDestroy.argtypes = [ctypes.c_void_p]
    lib.flagfftDestroy.restype = ctypes.c_int
    torch.npu.set_device(0)
    return lib


@contextmanager
def _plan(lib, shape, batch, kind=0x29):
    n = math.prod(shape)
    compact = math.prod(shape[:-1]) * (shape[-1] // 2 + 1)
    idist = compact if kind == 0x2c else n
    odist = compact if kind == 0x2a else n
    dimensions = (ctypes.c_int * len(shape))(*shape)
    handle = ctypes.c_void_p()
    assert lib.flagfftPlanMany(ctypes.byref(handle), len(shape), dimensions,
        None, 1, idist, None, 1, odist, kind, batch) == 0
    try:
        description = lib.flagfftGetPlanDescription(handle)
        assert description
        yield handle, description.decode()
    finally:
        torch.npu.synchronize()
        assert lib.flagfftDestroy(handle) == 0


def _description(lib, shape, batch, kind=0x29):
    with _plan(lib, shape, batch, kind) as (_, description):
        return description


def _passes(description):
    return tuple(map(int, re.findall(r'triton_passes=(\d+)', description)))


@pytest.mark.parametrize('n,batch,admitted', [
    (185640, 1, False), (185640, 3, False), (185640, 15, False),
    (185640, 16, True), (185640, 64, True),
    (255 * 507, 16, False), (255 * 520, 16, True),
    (255 * 4095, 16, True),
    pytest.param(255 * 4160, 16, False, id='outside-overall-dif-domain'),
    (255 * 512, 16, False),  # No radix7/13 vector-basis suffix.
    (340200, 16, False),     # A different prefix family.
])
def test_batch_dense_plan_selection(library, monkeypatch, n, batch, admitted):
    monkeypatch.setenv('FLAGFFT_NPU_DIF_BATCH_DENSE_PREFIX', '0')
    off = _description(library, (n,), batch)
    monkeypatch.setenv('FLAGFFT_NPU_DIF_BATCH_DENSE_PREFIX', '1')
    on = _description(library, (n,), batch)
    if admitted:
        # A C API plan describes both its forward and inverse executions.
        before, after = _passes(off), _passes(on)
        assert len(before) == len(after) == 2
        assert all(old - new == 2 for old, new in zip(before, after))
    else:
        assert on == off


def test_batch_dense_cache_optout_round_trip(library, monkeypatch):
    descriptions = []
    for enabled in ('1', '0', '1'):
        monkeypatch.setenv('FLAGFFT_NPU_DIF_BATCH_DENSE_PREFIX', enabled)
        descriptions.append(_description(library, (185640,), 64))
    assert descriptions[0] == descriptions[2]
    enabled, disabled = _passes(descriptions[0]), _passes(descriptions[1])
    assert len(enabled) == len(disabled) == 2
    assert all(old - new == 2 for old, new in zip(disabled, enabled))


@pytest.mark.parametrize('shape,kind', [((663000,), 0x2a), ((663000,), 0x2c), ((2, 185640), 0x29)])
def test_real_and_rank2_plans_keep_their_schedule(library, monkeypatch, shape, kind):
    monkeypatch.setenv('FLAGFFT_NPU_DIF_BATCH_DENSE_PREFIX', '0')
    off = _description(library, shape, 16, kind)
    monkeypatch.setenv('FLAGFFT_NPU_DIF_BATCH_DENSE_PREFIX', '1')
    assert _description(library, shape, 16, kind) == off


# Include both acceptance lengths and shapes/batches outside that matrix.
_LIVE_CASES = [(185640, 16, 'random', False), (185640, 65, 'tone', True),
               (663000, 16, 'basis', True), (663000, 64, 'cancellation', False),
               (132600, 16, 'random', False), (212160, 32, 'random', False),
               (185640, 128, 'random', False)]


@pytest.mark.parametrize('n,batch,pattern,in_place', _LIVE_CASES)
@pytest.mark.parametrize('direction', [-1, 1])
def test_batch_dense_live(library, monkeypatch, n, batch, pattern, in_place, direction):
    _execute(library, monkeypatch, n, batch, pattern, in_place, direction)


def test_batch_dense_inverse_without_global_conjugation(library, monkeypatch):
    monkeypatch.setenv('FLAGFFT_NPU_DIF_GLOBAL_INVERSE', '0')
    _execute(library, monkeypatch, 132600, 16, 'random', False, 1)


def _execute(lib, monkeypatch, n, batch, pattern, in_place, direction):
    monkeypatch.setenv('FLAGFFT_NPU_DIF_BATCH_DENSE_PREFIX', '1')
    torch.manual_seed(20261110)
    streams = [torch.npu.current_stream(), torch.npu.Stream()]
    with _plan(lib, (n,), batch) as (handle, description):
        assert _passes(description)
        for step, scale in enumerate((1., 1e-6, 1e6, -.25)):
            value = torch.randn(batch, n, 2)
            if pattern == 'tone':
                phase = 2 * math.pi * 37 * torch.arange(n, dtype=torch.float64) / n
                value = torch.view_as_real(torch.polar(torch.ones_like(phase), phase)).float()[None].expand(batch, n, 2).clone()
            elif pattern == 'basis':
                value.zero_()
                value[:, 1009, 1] = 1.
            elif pattern == 'cancellation':
                value[:, n // 2:2 * (n // 2)] = -value[:, :n // 2].flip(1)
                value[:, -1] *= 1e-6
            value *= scale
            z = torch.view_as_complex(value).cdouble()
            expected = torch.fft.fft(z) if direction == -1 else torch.fft.ifft(z) * n
            stream = streams[step % 2]
            with torch.npu.stream(stream):
                output_guard = torch.full((batch * n * 2 + 256,), 31415.25, device='npu')
                y = output_guard[128:-128].reshape(batch, n, 2)
                input_guard = torch.full_like(output_guard, 27182.125) if not in_place else None
                x = y if in_place else input_guard[128:-128].reshape(batch, n, 2)
                if not in_place:
                    y.fill_(float('nan'))
                x.copy_(value.npu())
                assert lib.flagfftSetStream(handle, ctypes.c_void_p(stream.npu_stream)) == 0
                assert lib.flagfftExecC2C(handle, ctypes.c_void_p(x.data_ptr()),
                                         ctypes.c_void_p(y.data_ptr()), direction) == 0
            stream.synchronize()
            actual = torch.view_as_complex(y.cpu()).cdouble()
            errors = torch.linalg.vector_norm(actual - expected, dim=1) / torch.linalg.vector_norm(expected, dim=1)
            assert torch.isfinite(actual).all() and (errors < 5e-7).all()
            assert (output_guard[:128].cpu() == 31415.25).all() and (output_guard[-128:].cpu() == 31415.25).all()
            if not in_place:
                assert torch.equal(x.cpu(), value)
                assert (input_guard[:128].cpu() == 27182.125).all() and (input_guard[-128:].cpu() == 27182.125).all()
