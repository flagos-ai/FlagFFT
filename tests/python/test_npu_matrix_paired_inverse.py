"""Odd inverse FFTs, discarded DC imaginary data and padded input/output rows."""
import math
import pytest

torch=pytest.importorskip('torch')
from flagfft_codegen.kernels_npu_matrix_paired_inverse import build_paired_inverse,emit_paired_inverse
from test_npu_structured_fft import Pointer,execute_module


@pytest.mark.parametrize('a,b',[(17,19),(129,131),(209,221),(221,209)])
@pytest.mark.parametrize('scale',[1.,1e-6,1e6])
def test_paired_inverse_whole(tmp_path,a,b,scale):
    n=a*b;half=n//2+1;h=b//2+1;batch=3
    pa=1<<(a-1).bit_length();ph=1<<(h-1).bit_length();pb=1<<(b-1).bit_length()
    path=tmp_path/'paired.py';path.write_text(build_paired_inverse(a,b))
    ns,pid=execute_module(path);pid.append(0)
    ns['tl'].broadcast_to=torch.broadcast_to
    ns['tl'].dot=lambda x,w,input_precision:x@w
    angle=2*math.pi*torch.arange(a,dtype=torch.float64)[:,None]*torch.arange(a)[None,:]/a
    w0=torch.zeros(2*pa,2*pa)
    w0[:a,:2*a:2]=angle.cos().float();w0[:a,1:2*a:2]=angle.sin().float()
    w0[pa:pa+a,:2*a:2]=-angle.sin().float();w0[pa:pa+a,1:2*a:2]=angle.cos().float()
    angle=2*math.pi*torch.arange(h,dtype=torch.float64)[:,None]*torch.arange(b)[None,:]/b
    weight=torch.full((h,1),2.,dtype=torch.float64);weight[0]=1
    w1=torch.zeros(2*ph,pb)
    w1[:h,:b]=(weight*angle.cos()).float();w1[ph:ph+h,:b]=(-weight*angle.sin()).float();w1[ph,:]=0
    angle=2*math.pi*torch.arange(h,dtype=torch.float64)[:,None]*torch.arange(a)[None,:]/n
    tw=torch.cat((angle.cos().float().flatten(),angle.sin().float().flatten()))
    torch.manual_seed(20261104)
    value=torch.randn(batch,half,2)*scale
    idist=half+5;odist=n+7
    src=torch.full((batch,idist,2),float('nan'));src[:,:half]=value
    saved=src[:,:half].clone()
    p=torch.full((batch,h,2*pa),float('nan'));q=torch.full((batch,a,2*ph),float('nan'))
    guard=torch.full((batch*odist+128,),31415.25)
    steps=[('pack_inverse',(math.ceil(pa/64),math.ceil(h/64),batch),(Pointer(src),Pointer(p),idist)),
           ('cube0_inverse_half_pack',(math.ceil(h/128),math.ceil(2*a/128),batch),(Pointer(p),Pointer(w0),Pointer(tw),Pointer(q))),
           ('cube1_inverse_half_real',(math.ceil(a/64),math.ceil(b/64),batch),(Pointer(q),Pointer(w1),Pointer(guard,64),odist))]
    for name,grid,args in steps:
        for ix in range(grid[0]):
            for iy in range(grid[1]):
                for iz in range(grid[2]):
                    pid[:]=[ix,iy,iz];ns[name](*args)
    out=guard[64:-64].reshape(batch,odist)
    ref=torch.fft.irfft(torch.view_as_complex(value).to(torch.complex128),n=n)*n
    assert torch.isfinite(out[:,:n]).all()
    assert torch.linalg.vector_norm(out[:,:n]-ref)/torch.linalg.vector_norm(ref)<5e-7
    assert torch.equal(src[:,:half],saved)
    assert (out[:,n:]==31415.25).all()
    assert (guard[:64]==31415.25).all() and (guard[-64:]==31415.25).all()


def test_paired_inverse_metadata(tmp_path):
    signatures={'pack_inverse':'*fp32:16,*fp32:16,i64',
                'cube0_inverse_half_pack':'*fp32:16,*fp32:16,*fp32:16,*fp32:16',
                'cube1_inverse_half_real':'*fp32:16,*fp32:16,*fp32:16,i64'}
    for phase,signature in signatures.items():
        meta=emit_paired_inverse(209,221,phase,'inverse',tmp_path)
        assert meta['signature']==signature
    with pytest.raises(ValueError):emit_paired_inverse(209,221,'pack_inverse','forward',tmp_path)
    with pytest.raises(ValueError):build_paired_inverse(128,221)
