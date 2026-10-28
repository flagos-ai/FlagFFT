"""Native C API replay checks, including fresh inputs and pointer/stream changes."""
import ctypes
import os
import pytest

torch=pytest.importorskip('torch')
pytest.importorskip('torch_npu')

@pytest.mark.parametrize('shape,batch',[((16,),1),((16,),64),((210,),1),((210,),64),((1024,),1),((64,64),1),((64,64),4),((128,128),1)])
def test_graph_replays_live_buffers(shape,batch,capfd):
    path=os.environ.get('FLAGFFT_LIBRARY')
    if not path:pytest.skip('Set FLAGFFT_LIBRARY to a built NPU library')
    assert os.environ.get('FLAGFFT_NPU_GRAPH')=='1'
    assert os.environ.get('FLAGFFT_NPU_GRAPH_TRACE')=='1'
    lib=ctypes.CDLL(path)
    lib.flagfftPlanMany.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int),ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int]
    lib.flagfftPlanMany.restype=ctypes.c_int
    lib.flagfftSetStream.argtypes=[ctypes.c_void_p,ctypes.c_void_p];lib.flagfftSetStream.restype=ctypes.c_int
    lib.flagfftExecC2C.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int];lib.flagfftExecC2C.restype=ctypes.c_int
    lib.flagfftDestroy.argtypes=[ctypes.c_void_p];lib.flagfftDestroy.restype=ctypes.c_int
    torch.npu.set_device(0)
    import math
    length=math.prod(shape)
    dims=(ctypes.c_int*len(shape))(*shape)
    handle=ctypes.c_void_p();assert lib.flagfftPlanMany(ctypes.byref(handle),len(shape),dims,None,1,length,None,1,length,0x29,batch)==0
    stream=torch.npu.current_stream()
    def run(x,y,direction=-1):
        assert lib.flagfftSetStream(handle,ctypes.c_void_p(stream.npu_stream))==0
        assert lib.flagfftExecC2C(handle,ctypes.c_void_p(x.data_ptr()),ctypes.c_void_p(y.data_ptr()),direction)==0
        stream.synchronize()
    def check(x,y,direction=-1):
        ref=torch.fft.fftn(x.cpu(),dim=tuple(range(1,len(shape)+1))) if direction==-1 else torch.fft.ifftn(x.cpu(),dim=tuple(range(1,len(shape)+1)))*length
        got=y.cpu()
        assert torch.isfinite(got).all()
        assert torch.linalg.vector_norm(got-ref)/torch.linalg.vector_norm(ref)<5e-7
    storage=torch.full((batch*length*2+128,),31415.25,device='npu')
    y=torch.view_as_complex(storage[64:-64].reshape((batch,)+shape+(2,)))
    x=torch.complex(torch.randn((batch,)+shape),torch.randn((batch,)+shape)).npu()
    try:
        # First execution prepares/captures. Subsequent calls must consume
        # newly written input values and overwrite a poisoned output.
        run(x,y);check(x,y)
        assert '[flagfft] NPU graph captured' in capfd.readouterr().err
        for scale in (2.0,-0.5,0.125):
            torch.view_as_real(x).mul_(scale);torch.view_as_real(y).fill_(float('nan'));run(x,y);check(x,y)
        assert (storage[:64].cpu()==31415.25).all() and (storage[-64:].cpu()==31415.25).all()
        other_x=x.clone();other_y=torch.empty_like(y);run(other_x,other_y);check(other_x,other_y)
        original_stream=stream;stream=torch.npu.Stream();run(x,y);check(x,y)
        stream=original_stream;torch.view_as_real(x).mul_(0.25);run(x,y);check(x,y)
        run(x,y,1);check(x,y,1)
        # In-place calls bypass the cached out-of-place graph.
        saved=x.clone();run(x,x);check(saved,x)
    finally:
        torch.npu.synchronize();assert lib.flagfftDestroy(handle)==0
