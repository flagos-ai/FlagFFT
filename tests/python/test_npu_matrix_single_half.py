"""Narrow first DFT followed by a real-component half-spectrum final DFT."""
import math
import pytest

torch=pytest.importorskip('torch')
from flagfft_codegen.kernels_npu_matrix_fourstep import build_matrix_fourstep
from flagfft_codegen.kernels_npu_matrix_complex_half import build_complex_half,emit_complex_half
from test_npu_structured_fft import Pointer,execute_module


@pytest.mark.parametrize('a,b',[(128,129),(209,221),(221,209)])
@pytest.mark.parametrize('direction',['forward','inverse'])
@pytest.mark.parametrize('pattern',['random','basis','cancellation','tone'])
def test_single_last_half_whole(tmp_path,a,b,direction,pattern):
    n=a*b;batch=3;pa=1<<(a-1).bit_length();pb=1<<(b-1).bit_length();hp=1<<(b//2).bit_length();sign=-1 if direction=='forward' else 1
    path=tmp_path/'last_half.py';path.write_text(build_matrix_fourstep(a,b)+build_complex_half(a,b,last_rows=32))
    ns,pid=execute_module(path);pid.append(0);ns['tl'].dot=lambda x,w,input_precision:x@w
    angle=sign*2*math.pi*torch.arange(a,dtype=torch.float64)[:,None]*torch.arange(a)[None,:]/a
    w0=torch.zeros(2*pa,2*pa);w0[:a,:2*a:2]=angle.cos().float();w0[:a,1:2*a:2]=angle.sin().float();w0[pa:pa+a,:2*a:2]=-angle.sin().float();w0[pa:pa+a,1:2*a:2]=angle.cos().float()
    angle=sign*2*math.pi*torch.arange(b,dtype=torch.float64)[:,None]*torch.arange(b//2+1)[None,:]/b
    hw=torch.zeros(pb,2*hp);hw[:b,:b//2*2+2:2]=angle.cos().float();hw[:b,1:b//2*2+2:2]=angle.sin().float()
    angle=sign*2*math.pi*torch.arange(b,dtype=torch.float64)[:,None]*torch.arange(a)[None,:]/n
    tw=torch.cat((angle.cos().float().flatten(),angle.sin().float().flatten()))
    torch.manual_seed(20261104);x=torch.randn(batch,n,2)
    if pattern=='basis':
        x.zero_()
        for j in range(batch):x[j,(j*1009)%n,j%2]=1.
    elif pattern=='cancellation':
        x[:,n//2:2*(n//2)]=-x[:,:n//2].flip(1)
        if n%2:x[:,-1]*=1e-6
    elif pattern=='tone':
        phase=2*math.pi*7*torch.arange(n,dtype=torch.float64)/n
        x=torch.view_as_real(torch.polar(torch.ones_like(phase),phase)).float()[None,:,:].expand(batch,n,2).clone()
    x*=torch.tensor([1e-6,1.,1e6])[:,None,None];saved=x.clone()
    p=torch.empty(batch,b,2*pa);mid=torch.empty(batch,n,2);q=torch.empty(batch,a,2*pb);guard=torch.full((2*batch*n+128,),31415.25)
    steps=[('pack0',(math.ceil(pa/16),math.ceil(b/16),batch),(Pointer(x),Pointer(p),Pointer(tw))),
           ('cube0',(math.ceil(b/32),math.ceil(2*a/64),batch),(Pointer(p),Pointer(w0),Pointer(mid))),
           ('pack1',(math.ceil(pb/16),math.ceil(a/16),batch),(Pointer(mid),Pointer(q),Pointer(tw))),
           ('cube1_complex_half_finish',(math.ceil(a/32),math.ceil((b//2+1)/32),batch),(Pointer(q),Pointer(hw),Pointer(guard,64)))]
    for name,grid,args in steps:
        for gx in range(grid[0]):
            for gy in range(grid[1]):
                for gz in range(grid[2]):pid[:]=[gx,gy,gz];ns[name](*args)
    got=torch.view_as_complex(guard[64:-64].reshape(batch,n,2));z=torch.view_as_complex(x).to(torch.complex128)
    ref=torch.fft.fft(z) if direction=='forward' else torch.fft.ifft(z)*n
    errors=torch.linalg.vector_norm(got-ref,dim=1)/torch.linalg.vector_norm(ref,dim=1)
    assert bool((errors<5e-7).all()),errors
    assert torch.equal(x,saved)
    assert (guard[:64]==31415.25).all() and (guard[-64:]==31415.25).all()


def test_single_half_metadata(tmp_path):
    meta=emit_complex_half(209,221,'cube1_complex_half_finish_narrow','forward',tmp_path)
    assert meta['signature']=='*fp32:16,*fp32:16,*fp32:16'
    assert meta['kernel_name']=='cube1_complex_half_finish'
