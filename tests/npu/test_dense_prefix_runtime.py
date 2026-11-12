"""Cube DIF with fresh C API buffers, streams, scales and in-place transforms."""
import ctypes
import os
import pytest

torch=pytest.importorskip('torch')
pytest.importorskip('torch_npu')

@pytest.mark.parametrize('n,batch,pattern',[(n,1,pattern) for n in (185640,663000) for pattern in ('random','tone','basis','cancellation')])
@pytest.mark.parametrize('direction',[-1,1])
@pytest.mark.parametrize('in_place',[False,True])
def test_dense_prefix_live(n,batch,pattern,direction,in_place,monkeypatch):
    monkeypatch.setenv("FLAGFFT_NPU_DIF_DENSE_SHORT_PREFIX","1")
    path=os.environ.get('FLAGFFT_LIBRARY')
    if not path:pytest.skip('Set FLAGFFT_LIBRARY to a built NPU library')
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
    lib.flagfftGetPlanDescription.argtypes=[ctypes.c_void_p];lib.flagfftGetPlanDescription.restype=ctypes.c_char_p
    description=lib.flagfftGetPlanDescription(handle).decode()
    assert 'triton_passes='+str(4 if n==185640 else 5) in description
    streams=[torch.npu.current_stream(),torch.npu.Stream()]
    try:
        for step,scale in enumerate((1.,1e-6,1e6,-.25)):
            value=torch.randn(batch,n,2)
            if pattern=='tone':
                phase=2*torch.pi*7*torch.arange(n,dtype=torch.float64)/n
                value=torch.view_as_real(torch.polar(torch.ones_like(phase),phase)).float()[None].expand(batch,n,2).clone()
            elif pattern=='basis':
                value.zero_();value[:,1009,1]=1.
            elif pattern=='cancellation':
                value[:,n//2:2*(n//2)]=-value[:,:n//2].flip(1)
                value[:,-1]*=1e-6
            value*=scale;z=torch.view_as_complex(value).to(torch.complex128)
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
            if not in_place:assert torch.equal(x.cpu(),value)
            assert torch.isfinite(got).all()
            assert torch.linalg.vector_norm(got-ref)/torch.linalg.vector_norm(ref)<5e-7
            assert (guard[:64].cpu()==31415.25).all() and (guard[-64:].cpu()==31415.25).all()
    finally:
        torch.npu.synchronize();assert lib.flagfftDestroy(handle)==0
