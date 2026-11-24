"""Arbitrary complex FFTs reconstructed from real-component half spectra."""
import math
import pytest

torch=pytest.importorskip('torch')
from flagfft_codegen.kernels_npu_matrix_complex_half import build_complex_half,emit_complex_half
from test_npu_structured_fft import Pointer,execute_module


@pytest.mark.parametrize('a,b',[(17,19),(128,129),(209,221),(221,209)])
@pytest.mark.parametrize('direction',['forward','inverse'])
@pytest.mark.parametrize('scale',[1.,1e-6,1e6])
def test_complex_half_whole(tmp_path,a,b,direction,scale):
    n=a*b;batch=2;pa=1<<(a-1).bit_length();pb=1<<(b-1).bit_length()
    path=tmp_path/'complex_half.py';path.write_text(build_complex_half(a,b))
    ns,pid=execute_module(path);pid.append(0);ns['tl'].dot=lambda x,w,input_precision:x@w
    sign=1 if direction=='inverse' else -1
    weights=[]
    for length,pad in ((a,pa),(b,pb)):
        h=length//2+1;hp=1<<(h-1).bit_length()
        angle=sign*2*math.pi*torch.arange(length,dtype=torch.float64)[:,None]*torch.arange(h)[None,:]/length
        w=torch.zeros(pad,2*hp);w[:length,:2*h:2]=angle.cos().float();w[:length,1:2*h:2]=angle.sin().float();weights.append(w)
    angle=sign*2*math.pi*torch.arange(b,dtype=torch.float64)[:,None]*torch.arange(a)[None,:]/n
    tw=torch.cat((angle.cos().float().flatten(),angle.sin().float().flatten()))
    torch.manual_seed(20261104);x=torch.randn(batch,n,2)*scale;saved=x.clone()
    view=x.reshape(batch,a,b,2);p=torch.zeros(batch,b,2*pa)
    p[:,:,:a]=view[:,:,:,0].transpose(1,2);p[:,:,pa:pa+a]=view[:,:,:,1].transpose(1,2)
    q=torch.full((batch,a,2*pb),float('nan'));guard=torch.full((batch*n*2+128,),31415.25)
    steps=[('cube0_complex_half_pack',(math.ceil(pb/128),math.ceil((a//2+1)/32),batch),(Pointer(p),Pointer(weights[0]),Pointer(tw),Pointer(q))),
           ('cube1_complex_half_finish',(math.ceil(a/64),math.ceil((b//2+1)/32),batch),(Pointer(q),Pointer(weights[1]),Pointer(guard,64)))]
    for name,grid,args in steps:
        for ix in range(grid[0]):
            for iy in range(grid[1]):
                for iz in range(grid[2]):pid[:]=[ix,iy,iz];ns[name](*args)
    got=torch.view_as_complex(guard[64:-64].reshape(batch,n,2));z=torch.view_as_complex(x).to(torch.complex128)
    reference=torch.fft.fft(z) if direction=='forward' else torch.fft.ifft(z)*n
    assert torch.isfinite(got).all()
    assert torch.linalg.vector_norm(got-reference)/torch.linalg.vector_norm(reference)<5e-7
    assert torch.equal(x,saved)
    assert (q[:,:,b:pb]==0).all() and (q[:,:,pb+b:]==0).all()
    assert (guard[:64]==31415.25).all() and (guard[-64:]==31415.25).all()


def test_complex_half_metadata(tmp_path):
    for phase,count in [('cube0_complex_half_pack',4),('cube1_complex_half_finish',3)]:
        meta=emit_complex_half(209,221,phase,'forward',tmp_path)
        assert meta['signature']==','.join(['*fp32:16']*count)
        assert len(meta['arg_names'])==count
    with pytest.raises(ValueError):emit_complex_half(209,221,'transpose','forward',tmp_path)
    with pytest.raises(ValueError):build_complex_half(15,221)
