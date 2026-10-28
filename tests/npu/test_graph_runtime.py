"""Native C API replay checks, including fresh inputs and pointer/stream changes."""
import ctypes
import os
import pytest

torch=pytest.importorskip('torch')
pytest.importorskip('torch_npu')

@pytest.mark.parametrize('length,batch',[(16,1),(16,64),(1024,1)])
def test_graph_replays_live_buffers(length,batch,capfd):
    path=os.environ.get('FLAGFFT_LIBRARY')
    if not path:pytest.skip('Set FLAGFFT_LIBRARY to a built NPU library')
    assert os.environ.get('FLAGFFT_NPU_GRAPH')=='1'
    assert os.environ.get('FLAGFFT_NPU_GRAPH_TRACE')=='1'
    lib=ctypes.CDLL(path)
    lib.flagfftPlan1d.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_int,ctypes.c_int,ctypes.c_int]
    lib.flagfftPlan1d.restype=ctypes.c_int
    lib.flagfftSetStream.argtypes=[ctypes.c_void_p,ctypes.c_void_p];lib.flagfftSetStream.restype=ctypes.c_int
    lib.flagfftExecC2C.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int];lib.flagfftExecC2C.restype=ctypes.c_int
    lib.flagfftDestroy.argtypes=[ctypes.c_void_p];lib.flagfftDestroy.restype=ctypes.c_int
    torch.npu.set_device(0)
    handle=ctypes.c_void_p();assert lib.flagfftPlan1d(ctypes.byref(handle),length,0x29,batch)==0
    stream=torch.npu.current_stream()
    def run(x,y,direction=-1):
        assert lib.flagfftSetStream(handle,ctypes.c_void_p(stream.npu_stream))==0
        assert lib.flagfftExecC2C(handle,ctypes.c_void_p(x.data_ptr()),ctypes.c_void_p(y.data_ptr()),direction)==0
        stream.synchronize()
    def check(x,y,direction=-1):
        ref=torch.fft.fft(x.cpu()) if direction==-1 else torch.fft.ifft(x.cpu())*length
        got=y.cpu()
        assert torch.isfinite(got).all()
        assert torch.linalg.vector_norm(got-ref)/torch.linalg.vector_norm(ref)<5e-7
    storage=torch.full((batch*length*2+128,),31415.25,device='npu')
    y=torch.view_as_complex(storage[64:-64].reshape(batch,length,2))
    x=torch.complex(torch.randn(batch,length),torch.randn(batch,length)).npu()
    try:
        # First execution prepares/captures. Subsequent calls must consume
        # newly written input values and overwrite a poisoned output.
        run(x,y);check(x,y)
        assert '[flagfft] NPU graph captured' in capfd.readouterr().err
        for scale in (2.0,-0.5,0.125):
            x.mul_(scale);y.fill_(complex(float('nan'),0));run(x,y);check(x,y)
        assert (storage[:64].cpu()==31415.25).all() and (storage[-64:].cpu()==31415.25).all()
        other_x=x.clone();other_y=torch.empty_like(y);run(other_x,other_y);check(other_x,other_y)
        original_stream=stream;stream=torch.npu.Stream();run(x,y);check(x,y)
        stream=original_stream;x.mul_(0.25);run(x,y);check(x,y)
        run(x,y,1);check(x,y,1)
        # In-place calls bypass the cached out-of-place graph.
        saved=x.clone();run(x,x);check(saved,x)
    finally:
        torch.npu.synchronize();assert lib.flagfftDestroy(handle)==0
