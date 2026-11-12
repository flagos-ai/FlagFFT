"""Native FFT2 real boundaries must consume fresh inputs under graph replay."""
import ctypes
import math
import os
import pytest

torch=pytest.importorskip('torch')
pytest.importorskip('torch_npu')


@pytest.mark.parametrize('inverse',[False,True])
@pytest.mark.parametrize('batch',[1,4])
def test_real_fft2_live_graph(batch,inverse,capfd):
    path=os.environ.get('FLAGFFT_LIBRARY')
    if not path:pytest.skip('Set FLAGFFT_LIBRARY to a built NPU library')
    assert os.environ.get('FLAGFFT_NPU_GRAPH')=='1'
    lib=ctypes.CDLL(path);shape=(128,128);half=shape[1]//2+1;n=math.prod(shape)
    lib.flagfftPlanMany.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int),ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int]
    lib.flagfftPlanMany.restype=ctypes.c_int
    lib.flagfftSetStream.argtypes=[ctypes.c_void_p,ctypes.c_void_p];lib.flagfftSetStream.restype=ctypes.c_int
    lib.flagfftDestroy.argtypes=[ctypes.c_void_p];lib.flagfftDestroy.restype=ctypes.c_int
    execute=lib.flagfftExecC2R if inverse else lib.flagfftExecR2C
    execute.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p];execute.restype=ctypes.c_int
    torch.npu.set_device(0);torch.manual_seed(20261029)
    compact=shape[0]*half
    dims=(ctypes.c_int*2)(*shape);handle=ctypes.c_void_p()
    assert lib.flagfftPlanMany(ctypes.byref(handle),2,dims,None,1,compact if inverse else n,None,1,n if inverse else compact,0x2c if inverse else 0x2a,batch)==0
    output_words=batch*(n if inverse else 2*compact)
    guarded=torch.full((output_words+128,),31415.25,device='npu')
    dst=guarded[64:-64].reshape((batch,)+shape if inverse else (batch,shape[0],half,2))
    source=torch.empty((batch,shape[0],half,2) if inverse else (batch,)+shape,device='npu')
    values=[]
    for scale in (1.,1e-6,1e6,-.25):
        real=torch.randn((batch,)+shape)*scale
        spec=torch.fft.rfft2(real)
        value=torch.view_as_real(spec) if inverse else real
        reference=torch.fft.irfft2(spec,s=shape)*n if inverse else torch.view_as_real(spec)
        values.append((value.npu(),reference))
    torch.npu.synchronize()
    try:
        for index,(value,reference) in enumerate(values):
            source.copy_(value);dst.fill_(float('nan'))
            stream=torch.npu.current_stream()
            assert lib.flagfftSetStream(handle,ctypes.c_void_p(stream.npu_stream))==0
            assert execute(handle,ctypes.c_void_p(source.data_ptr()),ctypes.c_void_p(dst.data_ptr()))==0
            stream.synchronize();got=dst.cpu()
            assert torch.isfinite(got).all()
            assert torch.linalg.vector_norm(got-reference)/torch.linalg.vector_norm(reference)<2e-6
            assert (guarded[:64].cpu()==31415.25).all() and (guarded[-64:].cpu()==31415.25).all()
            if index==0:assert '[flagfft] NPU graph captured' in capfd.readouterr().err
    finally:
        torch.npu.synchronize();assert lib.flagfftDestroy(handle)==0
