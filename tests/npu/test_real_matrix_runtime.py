"""Real matrix FFT execution with live graph inputs and overlapping buffers."""
import ctypes
import os
import pytest

torch=pytest.importorskip('torch')
pytest.importorskip('torch_npu')


@pytest.mark.parametrize('inverse',[False,True])
@pytest.mark.parametrize('n',[128,210,256])
def test_real_matrix_replay_and_in_place(inverse,n):
    path=os.environ.get('FLAGFFT_LIBRARY')
    if not path:pytest.skip('Set FLAGFFT_LIBRARY to a built NPU library')
    lib=ctypes.CDLL(path);batch=64;half=n//2+1
    lib.flagfftPlanMany.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int),ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int]
    lib.flagfftPlanMany.restype=ctypes.c_int
    lib.flagfftDestroy.argtypes=[ctypes.c_void_p];lib.flagfftDestroy.restype=ctypes.c_int
    lib.flagfftSetStream.argtypes=[ctypes.c_void_p,ctypes.c_void_p];lib.flagfftSetStream.restype=ctypes.c_int
    execute=lib.flagfftExecC2R if inverse else lib.flagfftExecR2C
    execute.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p];execute.restype=ctypes.c_int
    torch.npu.set_device(0);torch.manual_seed(20261029)
    dims=(ctypes.c_int*1)(n)
    for in_place in (False,True):
        idist=half if inverse else 2*half if in_place else n
        odist=2*half if inverse else half
        handle=ctypes.c_void_p()
        assert lib.flagfftPlanMany(ctypes.byref(handle),1,dims,None,1,idist,None,1,odist,0x2c if inverse else 0x2a,batch)==0
        raw=torch.full((batch*2*half+128,),31415.25,device='npu')
        storage=raw[64:-64].reshape(batch,2*half)
        if in_place:
            x=storage.reshape(batch,half,2) if inverse else storage[:,:n]
            y=storage[:,:n] if inverse else storage.reshape(batch,half,2)
        else:
            x=torch.empty((batch,half,2) if inverse else (batch,n),device='npu')
            y=storage[:,:n] if inverse else storage.reshape(batch,half,2)
        try:
            for scale in (1.,1e-6,1e6,-.25):
                value=torch.randn((batch,half,2) if inverse else (batch,n))*scale
                if not in_place:y.fill_(float('nan'))
                x.copy_(value.npu())
                ref=torch.fft.irfft(torch.view_as_complex(value),n=n)*n if inverse else torch.view_as_real(torch.fft.rfft(value))
                stream=torch.npu.current_stream()
                assert lib.flagfftSetStream(handle,ctypes.c_void_p(stream.npu_stream))==0
                assert execute(handle,ctypes.c_void_p(x.data_ptr()),ctypes.c_void_p(y.data_ptr()))==0
                stream.synchronize();got=y.cpu()
                assert torch.isfinite(got).all()
                assert torch.linalg.vector_norm(got-ref)/torch.linalg.vector_norm(ref)<2e-6
                assert (raw[:64].cpu()==31415.25).all() and (raw[-64:].cpu()==31415.25).all()
        finally:
            torch.npu.synchronize();assert lib.flagfftDestroy(handle)==0
