"""Aligned mixed-radix FFTs for small batches and live C API buffers."""
import ctypes
import os
import pytest

torch=pytest.importorskip('torch')
pytest.importorskip('torch_npu')

@pytest.mark.parametrize('n,batch',[(262144,3),(524288,1),(1048576,2)])
@pytest.mark.parametrize('direction',[-1,1])
@pytest.mark.parametrize('in_place',[False,True])
def test_flat_power_dif_live(n,batch,direction,in_place):
    path=os.environ.get('FLAGFFT_LIBRARY')
    if not path:pytest.skip('Set FLAGFFT_LIBRARY to a built NPU library')
    assert os.environ.get('FLAGFFT_NPU_FLAT_POWER_DIF')=='1'
    lib=ctypes.CDLL(path)
    lib.flagfftPlanMany.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int),ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int]
    lib.flagfftPlanMany.restype=ctypes.c_int
    lib.flagfftSetStream.argtypes=[ctypes.c_void_p,ctypes.c_void_p];lib.flagfftSetStream.restype=ctypes.c_int
    lib.flagfftDestroy.argtypes=[ctypes.c_void_p];lib.flagfftDestroy.restype=ctypes.c_int
    lib.flagfftExecC2C.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int]
    lib.flagfftExecC2C.restype=ctypes.c_int
    torch.npu.set_device(0);torch.manual_seed(20261029)
    dims=(ctypes.c_int*1)(n);handle=ctypes.c_void_p()
    assert lib.flagfftPlanMany(ctypes.byref(handle),1,dims,None,1,n,None,1,n,0x29,batch)==0
    streams=[torch.npu.current_stream(),torch.npu.Stream()]
    try:
        for step,scale in enumerate((1.,1e-6,1e6,-.25)):
            value=torch.randn(batch,n,2)*scale;z=torch.view_as_complex(value)
            ref=torch.view_as_real(torch.fft.fft(z) if direction==-1 else torch.fft.ifft(z)*n)
            stream=streams[step%2]
            with torch.npu.stream(stream):
                guard=torch.full((batch*n*2+128,),31415.25,device='npu')
                y=guard[64:-64].reshape(batch,n,2)
                x=y if in_place else torch.empty_like(y)
                if not in_place:y.fill_(float('nan'))
                x.copy_(value.npu())
                assert lib.flagfftSetStream(handle,ctypes.c_void_p(stream.npu_stream))==0
                assert lib.flagfftExecC2C(handle,ctypes.c_void_p(x.data_ptr()),ctypes.c_void_p(y.data_ptr()),direction)==0
            stream.synchronize();got=y.cpu()
            assert torch.isfinite(got).all()
            assert torch.linalg.vector_norm(got-ref)/torch.linalg.vector_norm(ref)<5e-7
            assert (guard[:64].cpu()==31415.25).all() and (guard[-64:].cpu()==31415.25).all()
    finally:
        torch.npu.synchronize();assert lib.flagfftDestroy(handle)==0
