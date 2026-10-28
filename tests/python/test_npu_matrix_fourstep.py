"""Whole mixed FFT oracle, with independent batches and poisoned work buffers."""
import math
import pytest

torch=pytest.importorskip('torch')
from flagfft_codegen.kernels_npu_matrix_fourstep import build_matrix_fourstep
from test_npu_structured_fft import Pointer,execute_module


@pytest.mark.parametrize('a,b',[(17,19),(48,49),(209,221)])
@pytest.mark.parametrize('batch',[1,3])
@pytest.mark.parametrize('direction',['forward','inverse'])
def test_full_matrix_fourstep(tmp_path,a,b,batch,direction):
    path=tmp_path/'fourstep.py';path.write_text(build_matrix_fourstep(a,b))
    ns,_=execute_module(path);pid=[0,0,0]
    ns['tl'].program_id=lambda axis:torch.tensor(pid[axis])
    ns['tl'].dot=lambda x,w,input_precision:x@w
    sign=1 if direction=='inverse' else -1
    weights=[];packed=[]
    for length,width in ((a,b),(b,a)):
        p=1<<(length-1).bit_length()
        angle=sign*2*math.pi*torch.arange(length,dtype=torch.float64)[:,None]*torch.arange(length)[None,:]/length
        c,s=angle.cos().float(),angle.sin().float();w=torch.zeros(2*p,2*p)
        w[:length,:2*length:2]=c;w[:length,1:2*length:2]=s
        w[p:p+length,:2*length:2]=-s;w[p:p+length,1:2*length:2]=c
        weights.append(w);packed.append(torch.full((batch,width,2*p),float('nan')))
    angle=sign*2*math.pi*torch.arange(b,dtype=torch.float64)[:,None]*torch.arange(a)[None,:]/(a*b)
    tw=torch.cat((angle.cos().float().flatten(),angle.sin().float().flatten()))
    torch.manual_seed(20261029);x=torch.randn(batch,a*b,2);saved=x.clone()
    mid=torch.full_like(x,float('nan'));last=torch.full_like(x,float('nan'))
    guarded=torch.full((x.numel()+128,),31415.25)
    for bz in range(batch):
        pid[2]=bz
        for stage,(length,width) in enumerate(((a,b),(b,a))):
            p=1<<(length-1).bit_length()
            for r in range((p+15)//16):
                pid[0]=r
                for col in range((width+15)//16):
                    pid[1]=col
                    ns[f'pack{stage}'](Pointer(x if stage==0 else mid),Pointer(packed[stage]),Pointer(tw))
            for r in range((width+31)//32):
                pid[0]=r
                for col in range((2*length+63)//64):
                    pid[1]=col
                    ns[f'cube{stage}'](Pointer(packed[stage]),Pointer(weights[stage]),Pointer(mid if stage==0 else last))
        for r in range((a+15)//16):
            pid[0]=r
            for col in range((b+15)//16):
                pid[1]=col;ns['transpose'](Pointer(last),Pointer(guarded,64))
    actual=torch.view_as_complex(guarded[64:-64].reshape(batch,a*b,2))
    ref=torch.fft.ifft(torch.view_as_complex(x))*a*b if direction=='inverse' else torch.fft.fft(torch.view_as_complex(x))
    assert torch.isfinite(actual).all()
    assert torch.linalg.vector_norm(actual-ref)/torch.linalg.vector_norm(ref)<2e-6
    assert torch.equal(x,saved)
    assert (guarded[:64]==31415.25).all() and (guarded[-64:]==31415.25).all()
