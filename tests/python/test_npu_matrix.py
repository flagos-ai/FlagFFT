"""Independent FFT oracle for matrix embedding and distance-aware masked I/O."""
import math
import pytest

torch=pytest.importorskip('torch')
from flagfft_codegen.kernels_npu_matrix import build_matrix_source,matrix_layout
from test_npu_structured_fft import Pointer,execute_module


@pytest.mark.parametrize('n',[48,49,210])
@pytest.mark.parametrize('batch',[3,64])
@pytest.mark.parametrize('mode,direction',[(0,'forward'),(0,'inverse'),(1,'forward'),(2,'inverse')])
def test_real_matrix_embedding(tmp_path,n,batch,mode,direction):
    p,ins,kin,outs,op,bm,bn=matrix_layout(n,mode)
    pack,cube,source=build_matrix_source(n,mode,direction)
    path=tmp_path/'matrix.py';path.write_text(source);ns,pid=execute_module(path)
    ns['tl'].dot=lambda x,w,input_precision:x@w
    torch.manual_seed(20261029)
    if mode==0:
        x=torch.randn(batch,n,2)
        z=torch.view_as_complex(x)
        ref=torch.fft.fft(z) if direction=='forward' else torch.fft.ifft(z)*n
        expected=torch.view_as_real(ref)
    elif mode==1:
        x=torch.randn(batch,n);expected=torch.view_as_real(torch.fft.rfft(x))
    else:
        x=torch.randn(batch,n//2+1,2);expected=torch.fft.irfft(torch.view_as_complex(x),n=n)*n
    input_distance,output_distance=ins+6,outs+10
    src=torch.full((batch,input_distance),-97.25);src[:,:ins]=x.reshape(batch,ins);saved=src.clone()
    w=torch.zeros(kin,op)
    if mode in (0,1):
        frequencies=n if mode==0 else n//2+1
        angle=(1 if direction=='inverse' else -1)*2*math.pi*torch.arange(n,dtype=torch.float64)[:,None]*torch.arange(frequencies)[None,:]/n
        c,s=angle.cos().float(),angle.sin().float()
        w[:n,:2*frequencies:2]=c;w[:n,1:2*frequencies:2]=s
        if mode==0:w[p:p+n,:2*frequencies:2]=-s;w[p:p+n,1:2*frequencies:2]=c
    else:
        j=torch.arange(n//2+1)[:,None];k=torch.arange(n)[None,:]
        angle=2*math.pi*j.double()*k/n
        endpoints=(j==0)|((n%2==0)&(j==n//2))
        amplitude=torch.where(endpoints,1.,2.)
        w[:ins:2,:n]=amplitude*angle.cos().float()
        w[1:ins:2,:n]=torch.where(endpoints,0.,-amplitude*angle.sin().float())
    work=torch.empty(batch,kin)
    out=torch.full((batch*output_distance+128,),31415.25)
    for b in range((batch+1)//2):
        pid[0]=b;ns[pack](Pointer(src),Pointer(work),input_distance,batch)
    for b in range((batch+bm-1)//bm):
        pid[0]=b
        for col in range((op+bn-1)//bn):
            pid[1]=col;ns[cube](Pointer(work if mode==0 else src),Pointer(w),Pointer(out,64),kin if mode==0 else input_distance,output_distance,batch)
    rows=out[64:-64].reshape(batch,output_distance);got=rows[:,:outs]
    assert torch.isfinite(got).all()
    assert torch.linalg.vector_norm(got-expected.reshape(batch,outs))/torch.linalg.vector_norm(expected)<2e-6
    assert torch.equal(src,saved)
    assert (rows[:,outs:]==31415.25).all()
    assert (out[:64]==31415.25).all() and (out[-64:]==31415.25).all()
