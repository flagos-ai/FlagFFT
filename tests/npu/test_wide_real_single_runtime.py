"""Half-length mixed real FFTs with live buffers, streams and padded aliases."""
import ctypes
import os
import pytest

torch=pytest.importorskip('torch')
pytest.importorskip('torch_npu')

@pytest.mark.parametrize('n,batch',[(25935,1),(46189,1),(23,64),(210,64),(23,1),(29,3),(30,5)])
@pytest.mark.parametrize('inverse',[False,True])
@pytest.mark.parametrize('in_place',[False,True])
def test_wide_real_single_live(n,batch,inverse,in_place):
    path=os.environ.get('FLAGFFT_LIBRARY')
    if not path:pytest.skip('Set FLAGFFT_LIBRARY to a built NPU library')
    assert os.environ.get('FLAGFFT_NPU_MATRIX_WIDE_SINGLE')=='1'
    lib=ctypes.CDLL(path)
    lib.flagfftPlanMany.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int),ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int]
    lib.flagfftPlanMany.restype=ctypes.c_int
    lib.flagfftSetStream.argtypes=[ctypes.c_void_p,ctypes.c_void_p];lib.flagfftSetStream.restype=ctypes.c_int
    lib.flagfftDestroy.argtypes=[ctypes.c_void_p];lib.flagfftDestroy.restype=ctypes.c_int
    execute=lib.flagfftExecC2R if inverse else lib.flagfftExecR2C
    execute.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p];execute.restype=ctypes.c_int
    torch.npu.set_device(0);torch.manual_seed(20261029);half=n//2+1
    idist=half if inverse else 2*half if in_place else n
    odist=(2*half if in_place else n) if inverse else half
    dims=(ctypes.c_int*1)(n);handle=ctypes.c_void_p()
    assert lib.flagfftPlanMany(ctypes.byref(handle),1,dims,None,1,idist,None,1,odist,0x2c if inverse else 0x2a,batch)==0
    words=batch*(2*half if in_place or not inverse else n)
    guard=torch.full((words+128,),31415.25,device='npu')
    if in_place:
        storage=guard[64:-64].reshape(batch,2*half)
        x=storage.reshape(batch,half,2) if inverse else storage[:,:n]
        y=storage[:,:n] if inverse else storage.reshape(batch,half,2)
    else:
        x=torch.empty((batch,half,2) if inverse else (batch,n),device='npu')
        y=guard[64:-64].reshape((batch,n) if inverse else (batch,half,2))
    streams=[torch.npu.current_stream(),torch.npu.Stream()]
    try:
        for step,scale in enumerate((1.,1e-6,1e6,-.25)):
            value=torch.randn((batch,half,2) if inverse else (batch,n))*scale
            ref=torch.fft.irfft(torch.view_as_complex(value),n=n)*n if inverse else torch.view_as_real(torch.fft.rfft(value))
            stream=streams[step%2]
            with torch.npu.stream(stream):
                if not in_place:y.fill_(float('nan'))
                x.copy_(value.npu())
                assert lib.flagfftSetStream(handle,ctypes.c_void_p(stream.npu_stream))==0
                assert execute(handle,ctypes.c_void_p(x.data_ptr()),ctypes.c_void_p(y.data_ptr()))==0
            stream.synchronize();got=y.cpu()
            assert torch.isfinite(got).all()
            assert torch.linalg.vector_norm(got-ref)/torch.linalg.vector_norm(ref)<5e-7
            assert (guard[:64].cpu()==31415.25).all() and (guard[-64:].cpu()==31415.25).all()
    finally:
        torch.npu.synchronize();assert lib.flagfftDestroy(handle)==0
