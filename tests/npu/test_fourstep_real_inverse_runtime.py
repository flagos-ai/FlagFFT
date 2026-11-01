"""Fused real inverse stores with batch pitches, fresh buffers and live streams."""
import ctypes
import os
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torch_npu')


@pytest.mark.parametrize('n,batch', [(323, 1), (2352, 3), (46189, 1), (46189, 3)])
@pytest.mark.parametrize('extra', [0, 8])
@pytest.mark.parametrize('in_place', [False, True])
def test_fourstep_real_inverse_live(n, batch, extra, in_place):
    path = os.environ.get('FLAGFFT_LIBRARY')
    if not path:
        pytest.skip('Set FLAGFFT_LIBRARY to a built NPU library')
    assert os.environ.get('FLAGFFT_NPU_MATRIX_REAL_FOURSTEP') == '1'
    lib = ctypes.CDLL(path)
    lib.flagfftPlanMany.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int]
    lib.flagfftSetStream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.flagfftExecC2R.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    lib.flagfftDestroy.argtypes = [ctypes.c_void_p]
    torch.npu.set_device(0)
    torch.manual_seed(20261101)
    half = n // 2 + 1
    idist = half + extra // 2
    odist = (2 * half if in_place else n) + extra
    dims = (ctypes.c_int * 1)(n)
    handle = ctypes.c_void_p()
    status = lib.flagfftPlanMany(ctypes.byref(handle), 1, dims, None, 1, idist,
                                None, 1, odist, 0x2c, batch)
    if extra:
        # The public 1D API accepts tight and standard in-place pitches only.
        # An arbitrary gap must be rejected before compiling or executing.
        assert status == 14 and handle.value is None
        return
    assert status == 0
    streams = [torch.npu.current_stream(), torch.npu.Stream()]
    try:
        for step, scale in enumerate((1., 1e-6, 1e6, -.25)):
            value = torch.randn(batch, half, 2) * scale
            reference = torch.fft.irfft(torch.view_as_complex(value), n=n) * n
            stream = streams[step % 2]
            with torch.npu.stream(stream):
                guard = torch.full((batch * odist + 128,), 31415.25, device='npu')
                output = guard[64:-64].reshape(batch, odist)
                storage = output if in_place else torch.full(
                    (batch, 2 * idist), 31415.25, device='npu')
                storage[:, :2 * half].copy_(value.reshape(batch, 2 * half).npu())
                saved_padding = output[:, n:].clone() if in_place else None
                if not in_place:
                    output[:, :n].fill_(float('nan'))
                assert lib.flagfftSetStream(handle, ctypes.c_void_p(stream.npu_stream)) == 0
                assert lib.flagfftExecC2R(handle, ctypes.c_void_p(storage.data_ptr()),
                                         ctypes.c_void_p(output.data_ptr())) == 0
            stream.synchronize()
            actual = output[:, :n].cpu()
            assert torch.isfinite(actual).all()
            assert torch.linalg.vector_norm(actual - reference) / torch.linalg.vector_norm(reference) < 5e-7
            padding = output[:, n:].cpu()
            assert torch.equal(padding, saved_padding.cpu()) if in_place else (padding == 31415.25).all()
            if not in_place:
                assert torch.equal(storage[:, :2 * half].cpu(), value.reshape(batch, 2 * half))
            assert (guard[:64].cpu() == 31415.25).all()
            assert (guard[-64:].cpu() == 31415.25).all()
    finally:
        torch.npu.synchronize()
        assert lib.flagfftDestroy(handle) == 0
