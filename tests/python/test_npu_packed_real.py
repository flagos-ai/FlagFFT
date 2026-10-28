"""Packed real FFT algebra, partial groups, launch offsets and output guards."""
import math
import pytest

torch=pytest.importorskip('torch')
from flagfft_codegen.kernels_npu_packed_real import build_npu_packed_real_source,packed_real_layout
from flagfft_codegen.metadata import _module_source
from test_npu_structured_fft import roots,execute_module,Pointer


def packed_roots(n,g):
    a,b,_,_=packed_real_layout(n,g);m=n//2
    index=torch.arange(m*g)//g
    angle=-2*math.pi*(index//b).double()*(index%b).double()/m
    k=-2*math.pi*torch.arange(m+1,dtype=torch.float64)/n
    return torch.cat((roots(a,b*g),roots(b,a*g),angle.cos().float(),angle.sin().float(),k.cos().float(),k.sin().float()))


@pytest.mark.parametrize('n',[128,256,512,1024,2048])
@pytest.mark.parametrize('group',[1,4])
@pytest.mark.parametrize('mode',[1,2])
def test_packed_real(tmp_path,n,group,mode):
    name,body=build_npu_packed_real_source(n,group,mode)
    path=tmp_path/'packed.py';path.write_text(_module_source(body))
    ns,pid=execute_module(path)
    torch.manual_seed(20261028)
    batch=group+1;x=torch.randn(batch,n)
    source=x if mode==1 else torch.view_as_real(torch.fft.rfft(x))
    reference=torch.view_as_real(torch.fft.rfft(x)) if mode==1 else torch.fft.irfft(torch.view_as_complex(source),n=n)*n
    saved=source.clone();out=torch.full((reference.numel()+128,),31415.25)
    table=packed_roots(n,group)
    for offset in range(0,batch,group):
        pid[0]=0
        ns[name](Pointer(source),Pointer(out,64),Pointer(torch.zeros(1,dtype=torch.int32)),Pointer(table),batch,1,offset)
    actual=out[64:-64].reshape(reference.shape)
    assert torch.isfinite(actual).all()
    assert torch.linalg.vector_norm(actual-reference)/torch.linalg.vector_norm(reference)<5e-7
    assert (out[:64]==31415.25).all() and (out[-64:]==31415.25).all()
    assert torch.equal(source,saved)
