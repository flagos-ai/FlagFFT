"""Three-dimensional real FFT256 rows within the 910B UB budget."""
import ctypes
import math
import os

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torch_npu')


@pytest.mark.parametrize('shape', [(8, 8, 256), (16, 8, 256)])
@pytest.mark.parametrize('batch', [1, 4])
@pytest.mark.parametrize('inverse', [False, True])
def test_3d_real_row256_grouped(shape, batch, inverse, monkeypatch):
    path = os.environ.get('FLAGFFT_LIBRARY')
    if not path:
        pytest.skip('Set FLAGFFT_LIBRARY to a built NPU library')
    # Force the general transpose decomposition for compact test geometries.
    monkeypatch.setenv('FLAGFFT_NPU_3D_LEAF', '0')
    lib = ctypes.CDLL(path)
    lib.flagfftPlanMany.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.flagfftPlanMany.restype = ctypes.c_int
    lib.flagfftGetPlanDescription.argtypes = [ctypes.c_void_p]
    lib.flagfftGetPlanDescription.restype = ctypes.c_char_p
    lib.flagfftSetStream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.flagfftSetStream.restype = ctypes.c_int
    execute = lib.flagfftExecC2R if inverse else lib.flagfftExecR2C
    execute.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    execute.restype = ctypes.c_int
    lib.flagfftDestroy.argtypes = [ctypes.c_void_p]
    lib.flagfftDestroy.restype = ctypes.c_int
    torch.npu.set_device(0)
    torch.manual_seed(20261112)
    n = math.prod(shape)
    half = shape[-1] // 2 + 1
    compact = math.prod(shape[:-1]) * half
    dims = (ctypes.c_int * 3)(*shape)
    plan = ctypes.c_void_p()
    assert lib.flagfftPlanMany(ctypes.byref(plan), 3, dims, None, 1, compact if inverse else n,
        None, 1, n if inverse else compact, 0x2c if inverse else 0x2a, batch) == 0
    description = lib.flagfftGetPlanDescription(plan).decode()
    assert 'CompiledRawNpuAivFFT256(group_size=4' in description
    assert ('[c2r-row]' if inverse else '[r2c-row]') in description
    words = batch * (n if inverse else 2 * compact)
    guarded = torch.full((words + 128,), 31415.25, device='npu')
    output = guarded[64:-64].reshape((batch,) + shape if inverse else (batch, *shape[:-1], half, 2))
    stream = torch.npu.current_stream()
    assert lib.flagfftSetStream(plan, ctypes.c_void_p(stream.npu_stream)) == 0
    try:
        for scale in (1.0, 1e-6, 1e6):
            real = torch.randn((batch,) + shape, dtype=torch.float32) * scale
            spectrum = torch.fft.rfftn(real.double(), dim=(1, 2, 3))
            if inverse:
                spectrum = spectrum.to(torch.complex64)
                data = torch.view_as_real(spectrum).npu()
                reference = torch.fft.irfftn(spectrum.cdouble(), s=shape, dim=(1, 2, 3)) * n
            else:
                data = real.npu()
                reference = torch.view_as_real(spectrum)
            output.fill_(float('nan'))
            assert execute(plan, ctypes.c_void_p(data.data_ptr()), ctypes.c_void_p(output.data_ptr())) == 0
            stream.synchronize()
            actual = output.cpu().double()
            errors = torch.linalg.vector_norm((actual - reference).reshape(batch, -1), dim=1) / \
                     torch.linalg.vector_norm(reference.reshape(batch, -1), dim=1)
            assert torch.isfinite(actual).all() and (errors < 5e-7).all()
            assert (guarded[:64].cpu() == 31415.25).all()
            assert (guarded[-64:].cpu() == 31415.25).all()
    finally:
        torch.npu.synchronize()
        assert lib.flagfftDestroy(plan) == 0
