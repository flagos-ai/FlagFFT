"""Check recovered DC, cancellation, scaling, inverse and partial input tiles."""
import math
import pytest

torch=pytest.importorskip('torch')
from flagfft_codegen.npu_radix17_matrix import half_matrix_first_source
from test_npu_structured_fft import Pointer,execute_module


@pytest.mark.parametrize('direction',['forward','inverse'])
@pytest.mark.parametrize('conjugate',[False,True])
@pytest.mark.parametrize('pattern',['basis','random','cancellation','tone'])
def test_half_matrix17_axis(tmp_path,direction,conjugate,pattern):
    m=1033;n=17*m;columns=512
    source=half_matrix_first_source({'name':'first'},n,columns,conjugate)
    path=tmp_path/'half17.py';path.write_text(source);ns,pid=execute_module(path)
    ns['tl'].dot=lambda a,b,input_precision:a@b
    batch=17 if pattern=='basis' else 3
    torch.manual_seed(20261104);x=torch.randn(batch,17,m,2)
    if pattern=='basis':
        x.zero_()
        for b in range(batch):x[b,b,(61*b)%m,b%2]=1.
    elif pattern=='cancellation':
        x[:,9:]=-x[:,1:9].flip(1);x[:,0]*=1e-6
    elif pattern=='tone':
        angle=2*torch.pi*torch.arange(17,dtype=torch.float64)/17
        x=torch.view_as_real(torch.polar(torch.ones_like(angle),angle)).float()[None,:,None,:].expand(batch,17,m,2).clone()
    if batch==3:x*=torch.tensor([1e-6,1.,1e6])[:,None,None,None]
    sign=-1 if direction=='forward' else 1
    angle=sign*2*math.pi*torch.arange(1,9,dtype=torch.float64)[:,None]*torch.arange(1,17)[None,:]/17
    weights=torch.cat((angle.cos().float(),angle.sin().float()),0)
    saved=x.clone();guard=torch.full((2*batch*n+128,),31415.25)
    for tile in range(math.ceil(m/columns)):
        for b in range(batch):
            pid[:]=[tile,b];ns['first'](Pointer(x),Pointer(weights),Pointer(guard,64))
    planes=guard[64:-64].reshape(batch,2,17,m)
    got=torch.complex(planes[:,0],planes[:,1]).to(torch.complex128)
    z=torch.view_as_complex(x).to(torch.complex128)
    ref=torch.fft.fft(z,dim=1) if direction=='forward' else torch.fft.ifft(z,dim=1)*17
    if conjugate:ref=ref.conj()
    assert torch.isfinite(got).all()
    assert torch.linalg.vector_norm(got-ref)/torch.linalg.vector_norm(ref)<5e-7
    assert torch.equal(x,saved)
    assert (guard[:64]==31415.25).all() and (guard[-64:]==31415.25).all()
