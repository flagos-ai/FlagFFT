"""Frequency tiles, padded coefficient rows and masked final output tiles."""
import math
import pytest
torch=pytest.importorskip('torch')
from flagfft_codegen.kernels_npu_frequency_dft import emit_frequency_dft
from test_npu_structured_fft import Pointer,execute_module

@pytest.mark.parametrize('n',[64,90,120,210,231,252,256])
@pytest.mark.parametrize('direction',['forward','inverse'])
def test_frequency_dft_body(tmp_path,n,direction):
    meta=emit_frequency_dft(n,direction,tmp_path);ns,pid=execute_module(meta['module_path'])
    ns['tl'].sum=lambda value,axis:torch.sum(value,dim=axis)
    p=1<<(n-1).bit_length();sign=-1 if direction=='forward' else 1
    phase=sign*2*math.pi*torch.arange(n,dtype=torch.float64)[:,None]*torch.arange(p,dtype=torch.float64)[None,:]/n
    w=torch.cat((phase.cos().float().flatten(),phase.sin().float().flatten()))
    torch.manual_seed(20261030);x=torch.randn(3,n,2);saved=x.clone();guard=torch.full((x.numel()+128,),31415.25)
    for b in range(3):
        for tile in range((n+7)//8):
            pid[:]=[tile,b];ns[meta['kernel_name']](Pointer(x),Pointer(w),Pointer(guard,64))
    got=guard[64:-64].reshape_as(x);z=torch.view_as_complex(x)
    ref=torch.view_as_real(torch.fft.fft(z) if sign<0 else torch.fft.ifft(z)*n)
    assert torch.isfinite(got).all()
    assert torch.linalg.vector_norm(got-ref)/torch.linalg.vector_norm(ref)<5e-7
    assert torch.equal(x,saved)
    assert (guard[:64]==31415.25).all() and (guard[-64:]==31415.25).all()
