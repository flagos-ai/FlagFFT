"""Packed real boundaries against FFT oracles, arbitrary endpoint values, and gaps."""
import math
import pytest

torch=pytest.importorskip('torch')
from flagfft_codegen.kernels_real import _build_npu_packed_boundary
from test_npu_structured_fft import Pointer,execute_module


@pytest.mark.parametrize('n',[512,1010,16384,524288])
@pytest.mark.parametrize('inverse',[False,True])
@pytest.mark.parametrize('width',[256,1024,2048])
@pytest.mark.parametrize('paired',[False,True])
def test_packed_real_windows(tmp_path,n,inverse,width,paired):
    torch.manual_seed(20261029);batch=3;packed=n//2;half=packed+1;distance=half+5
    name,body,args,rows=_build_npu_packed_boundary(n,inverse,width,paired)
    path=tmp_path/'boundary.py';path.write_text(body);ns,pid=execute_module(path)
    angle=-2*math.pi*torch.arange(half,dtype=torch.float64)/n
    roots=torch.stack((angle.cos().float(),angle.sin().float()),-1)
    if inverse:
        values=torch.randn(batch,half,2)
        reference_real=torch.fft.irfft(torch.view_as_complex(values),n=n)*n
        reference=torch.view_as_real(torch.fft.fft(torch.view_as_complex(reference_real.reshape(batch,packed,2)))/packed)
        src=torch.full((batch,distance,2),-77.25);src[:,:half]=values
        out_words=batch*packed*2
    else:
        values=torch.randn(batch,n)
        src=torch.view_as_real(torch.fft.fft(torch.view_as_complex(values.reshape(batch,packed,2))))
        reference=torch.view_as_real(torch.fft.rfft(values));out_words=batch*distance*2
    saved=src.clone();out=torch.full((out_words+128,),31415.25)
    for b in range(batch):
        pid[1]=b
        for tile in range(((packed//2+1 if paired else packed if inverse else half)+width-1)//width):
            pid[0]=tile;ns[name](Pointer(src),Pointer(roots),Pointer(out,64),distance,batch)
    got=out[64:-64].reshape(batch,packed if inverse else distance,2)
    if not inverse:
        assert (got[:,half:]==31415.25).all();got=got[:,:half]
    assert torch.isfinite(got).all()
    assert torch.linalg.vector_norm(got-reference)/torch.linalg.vector_norm(reference)<1e-6
    assert torch.equal(src,saved)
    assert (out[:64]==31415.25).all() and (out[-64:]==31415.25).all()
    assert rows==1 and args[-2] == ('input_distance' if inverse else 'output_distance')
