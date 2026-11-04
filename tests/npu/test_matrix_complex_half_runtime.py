"""Cube DIF with fresh C API buffers, streams, scales and in-place transforms."""
import ctypes
import os
import pytest

torch=pytest.importorskip('torch')
pytest.importorskip('torch_npu')

@pytest.mark.parametrize('n,batch,pattern,default_policy',
    [(46189,1,pattern,policy) for pattern in ('random','tone','basis','cancellation')
     for policy in (False,True)]+
    [(46189,17,pattern,False) for pattern in ('tone','basis','cancellation')]+
    [(46189,17,'random',False),(46189,33,'random',False)])
@pytest.mark.parametrize('direction',[-1,1])
@pytest.mark.parametrize('in_place',[False,True])
def test_complex_half_matrix_live(n,batch,pattern,default_policy,direction,in_place,monkeypatch):
    if default_policy:monkeypatch.delenv("FLAGFFT_NPU_MATRIX_SINGLE_HALF_LAST",raising=False)
    elif batch==1:monkeypatch.setenv("FLAGFFT_NPU_MATRIX_SINGLE_HALF_LAST","1")
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
    if default_policy:
        forward,inverse=description.split('-- Inverse execution --')
        assert 'complex_half_last_only=0' in forward
        assert 'complex_half_last_only=0' in inverse
    else:
        assert 'complex_half_spectra=1' in description
        assert ('complex_half_last_only=1' in description)==(batch==1)
    if batch>=16:
        expected=os.environ.get('FLAGFFT_NPU_MATRIX_DIRECT_INPUT','1')!='0'
        assert ('complex_direct_input=1' in description)==expected
        combined=expected and os.environ.get('FLAGFFT_NPU_MATRIX_DIRECT_COMBINED','1')!='0'
        assert ('complex_direct_combined=1' in description)==combined
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
