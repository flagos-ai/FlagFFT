"""Independent FFT oracle for matrix embedding and distance-aware masked I/O."""
import math
import pytest

torch=pytest.importorskip('torch')
from flagfft_codegen.kernels_npu_matrix import build_matrix_source,matrix_layout
from test_npu_structured_fft import Pointer,execute_module


@pytest.mark.parametrize('snapshot',[False,True])
@pytest.mark.parametrize('n',[23,48,49,128,210,256])
@pytest.mark.parametrize('batch',[3,64])
@pytest.mark.parametrize('mode,direction',[(0,'forward'),(0,'inverse'),(1,'forward'),(2,'inverse')])
def test_real_matrix_embedding(tmp_path,n,batch,mode,direction,snapshot):
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
        if mode==0:
            w[:2*n:2,:2*frequencies:2]=c;w[:2*n:2,1:2*frequencies:2]=s
            w[1:2*n:2,:2*frequencies:2]=-s;w[1:2*n:2,1:2*frequencies:2]=c
        else:w[:n,:2*frequencies:2]=c;w[:n,1:2*frequencies:2]=s
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
            pid[1]=col;ns[cube](Pointer(work if snapshot else src),Pointer(w),Pointer(out,64),kin if snapshot else input_distance,output_distance,batch)
    rows=out[64:-64].reshape(batch,output_distance);got=rows[:,:outs]
    assert torch.isfinite(got).all()
    assert torch.linalg.vector_norm(got-expected.reshape(batch,outs))/torch.linalg.vector_norm(expected)<2e-6
    assert torch.equal(src,saved)
    assert (rows[:,outs:]==31415.25).all()
    assert (out[:64]==31415.25).all() and (out[-64:]==31415.25).all()


@pytest.mark.parametrize('n', [210, 256])
@pytest.mark.parametrize('direction', ['forward', 'inverse'])
def test_long_dot_preserves_small_signal_beside_large_impulse(tmp_path, n, direction):
    """A serial FP32 reduction loses the tail; independent partials retain it."""
    _, ins, kin, outs, columns, bm, bn = matrix_layout(n, 0)
    pack, cube, source = build_matrix_source(n, 0, direction)
    path = tmp_path/'matrix_dynamic_range.py'
    path.write_text(source)
    ns, pid = execute_module(path)

    def serial_ieee_dot(a, b, input_precision):
        assert input_precision == 'ieee'
        assert a.dtype == b.dtype == torch.float32
        value = torch.zeros((a.shape[0], b.shape[1]), dtype=torch.float32)
        for index in range(a.shape[1]):
            value = torch.addcmul(value, a[:, index, None], b[index, None, :])
        return value

    ns['tl'].dot = serial_ieee_dot
    batch = 3
    data = torch.zeros(batch, n, 2)
    data[:, 0, 0] = 1e8
    data[:, n//2:, 0] = 1.
    saved = data.clone()
    value = torch.view_as_complex(data).cdouble()
    reference = torch.fft.fft(value) if direction == 'forward' else torch.fft.ifft(value)*n
    sign = 1 if direction == 'inverse' else -1
    angle = sign*2*math.pi*torch.arange(n, dtype=torch.float64)[:, None]*torch.arange(n)[None, :]/n
    cosine, sine = angle.cos().float(), angle.sin().float()
    weights = torch.zeros(kin, columns)
    weights[:2*n:2, :2*n:2] = cosine
    weights[:2*n:2, 1:2*n:2] = sine
    weights[1:2*n:2, :2*n:2] = -sine
    weights[1:2*n:2, 1:2*n:2] = cosine
    work = torch.empty(batch, kin)
    guarded = torch.full((batch*outs+128,), 31415.25)
    for block in range((batch+1)//2):
        pid[0] = block
        ns[pack](Pointer(data), Pointer(work), ins, batch)
    for block in range((batch+bm-1)//bm):
        pid[0] = block
        for column in range((columns+bn-1)//bn):
            pid[1] = column
            ns[cube](Pointer(work), Pointer(weights), Pointer(guarded, 64), kin, outs, batch)
    actual = torch.view_as_complex(guarded[64:-64].reshape(batch, n, 2)).cdouble()
    assert torch.linalg.vector_norm(actual-reference)/torch.linalg.vector_norm(reference) < 5e-7
    assert (actual-reference).abs().max()/reference.abs().max() < 5e-7
    assert torch.equal(data, saved)
    assert (guarded[:64] == 31415.25).all() and (guarded[-64:] == 31415.25).all()
