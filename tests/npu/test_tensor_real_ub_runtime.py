"""Real 3-D inverses around the fused tensor UB limit on Ascend."""
import ctypes
import math
import os

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torch_npu')


@pytest.mark.parametrize('shape', [(16, 16, 16), (32, 8, 16), (8, 16, 32), (16, 8, 16)])
@pytest.mark.parametrize('batch', [1, 4])
def test_3d_inverse_uses_resource_bounded_plan(shape, batch):
    path = os.environ.get('FLAGFFT_LIBRARY')
    if not path:
        pytest.skip('Set FLAGFFT_LIBRARY to a built NPU library')
    assert os.environ.get('FLAGFFT_NPU_TENSOR_FFT') == '1'
    lib = ctypes.CDLL(path)
    lib.flagfftPlanMany.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.flagfftPlanMany.restype = ctypes.c_int
    lib.flagfftGetPlanDescription.argtypes = [ctypes.c_void_p]
    lib.flagfftGetPlanDescription.restype = ctypes.c_char_p
    lib.flagfftSetStream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.flagfftSetStream.restype = ctypes.c_int
    lib.flagfftExecC2R.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    lib.flagfftExecC2R.restype = ctypes.c_int
    lib.flagfftDestroy.argtypes = [ctypes.c_void_p]
    lib.flagfftDestroy.restype = ctypes.c_int
    torch.npu.set_device(0)
    torch.manual_seed(20261112)
    n = math.prod(shape)
    half = shape[-1] // 2 + 1
    compact = math.prod(shape[:-1]) * half
    dims = (ctypes.c_int * 3)(*shape)
    plan = ctypes.c_void_p()
    assert lib.flagfftPlanMany(ctypes.byref(plan), 3, dims, None, 1, compact,
        None, 1, n, 0x2c, batch) == 0
    description = lib.flagfftGetPlanDescription(plan).decode()
    fused = 'npu_tensor_' + 'x'.join(map(str, shape)) + '_m2_inverse'
    assert (fused in description) == (n <= 2048)
    guarded = torch.full((batch * n + 128,), 31415.25, device='npu')
    output = guarded[64:-64].reshape((batch,) + shape)
    stream = torch.npu.current_stream()
    try:
        for scale in (1.0, 1e-6, 1e6):
            real = torch.randn((batch,) + shape, dtype=torch.float64) * scale
            spectrum = torch.fft.rfftn(real, dim=(1, 2, 3)).to(torch.complex64)
            reference = torch.fft.irfftn(spectrum.cdouble(), s=shape, dim=(1, 2, 3)) * n
            data = torch.view_as_real(spectrum).npu()
            output.fill_(float('nan'))
            # Reading npu_stream drains torch_npu's host task queue before
            # the external launch, without synchronizing device execution.
            assert lib.flagfftSetStream(plan, ctypes.c_void_p(stream.npu_stream)) == 0
            assert lib.flagfftExecC2R(plan, ctypes.c_void_p(data.data_ptr()),
                                      ctypes.c_void_p(output.data_ptr())) == 0
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
