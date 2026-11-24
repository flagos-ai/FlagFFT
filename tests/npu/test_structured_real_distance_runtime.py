"""Padded real rows, including aliased batches, through the public NPU API."""
import ctypes
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")


@pytest.mark.parametrize("n", [16, 128, 2048])
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("alias", [False, True])
def test_structured_real_padded_rows(n, inverse, alias):
    path = os.environ.get("FLAGFFT_LIBRARY")
    if not path:
        pytest.skip("Set FLAGFFT_LIBRARY to a built NPU library")
    lib = ctypes.CDLL(path)
    lib.flagfftPlanMany.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.flagfftPlanMany.restype = ctypes.c_int
    lib.flagfftGetPlanDescription.argtypes = [ctypes.c_void_p]
    lib.flagfftGetPlanDescription.restype = ctypes.c_char_p
    lib.flagfftDestroy.argtypes = [ctypes.c_void_p]
    lib.flagfftDestroy.restype = ctypes.c_int
    lib.flagfftSetStream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.flagfftSetStream.restype = ctypes.c_int
    execute = lib.flagfftExecC2R if inverse else lib.flagfftExecR2C
    execute.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    execute.restype = ctypes.c_int

    torch.npu.set_device(0)
    torch.manual_seed(20261112)
    batch, half = 3, n // 2 + 1
    real_pitch = 2 * half
    dims, handle = (ctypes.c_int * 1)(n), ctypes.c_void_p()
    idist, odist = (half, real_pitch) if inverse else (real_pitch, half)
    assert lib.flagfftPlanMany(ctypes.byref(handle), 1, dims, None, 1, idist,
        None, 1, odist, 0x2c if inverse else 0x2a, batch) == 0
    stream = torch.npu.current_stream()
    description = lib.flagfftGetPlanDescription(handle).decode()
    assert "CompiledRawNpuAivFFTSmall(" in description
    words = batch * real_pitch
    input_guard = torch.full((words + 128,), 27182.125, device="npu")
    output_guard = input_guard if alias else torch.full((words + 128,), 31415.25, device="npu")
    xbase, ybase = input_guard[64:-64], output_guard[64:-64]
    x = xbase.reshape(batch, half, 2) if inverse else xbase.reshape(batch, real_pitch)[:, :n]
    y = ybase.reshape(batch, real_pitch)[:, :n] if inverse else ybase.reshape(batch, half, 2)
    try:
        for scale in (1.0, 1e-6, 1e6):
            value = torch.randn(batch, half, 2) * scale if inverse else torch.randn(batch, n) * scale
            if inverse:
                reference = torch.fft.irfft(torch.view_as_complex(value).cdouble(), n=n) * n
            else:
                reference = torch.fft.rfft(value.double())
            x.copy_(value.npu())
            assert lib.flagfftSetStream(handle, ctypes.c_void_p(stream.npu_stream)) == 0
            assert execute(handle, ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(y.data_ptr())) == 0
            stream.synchronize()
            got = y.cpu()
            got = got.double() if inverse else torch.view_as_complex(got).cdouble()
            errors = torch.linalg.vector_norm(got - reference, dim=1) / torch.linalg.vector_norm(reference, dim=1)
            assert torch.isfinite(got).all() and (errors < 5e-7).all()
            if not alias:
                assert torch.equal(x.cpu(), value)
            assert (input_guard[:64].cpu() == 27182.125).all()
            assert (input_guard[-64:].cpu() == 27182.125).all()
            marker = 27182.125 if alias else 31415.25
            assert (output_guard[:64].cpu() == marker).all()
            assert (output_guard[-64:].cpu() == marker).all()
    finally:
        torch.npu.synchronize()
        assert lib.flagfftDestroy(handle) == 0
