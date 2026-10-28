"""Whole mixed FFT oracle, with independent batches and poisoned work buffers."""
import math
import pytest

torch=pytest.importorskip('torch')
from flagfft_codegen.kernels_npu_matrix_fourstep import build_matrix_fourstep
from test_npu_structured_fft import Pointer,execute_module


@pytest.mark.parametrize('a,b',[(17,19),(48,49),(209,221)])
@pytest.mark.parametrize('batch',[1,3])
@pytest.mark.parametrize('wide',[False,True])
@pytest.mark.parametrize('direction',['forward','inverse'])
def test_full_matrix_fourstep(tmp_path,a,b,batch,direction,wide):
    path=tmp_path/'fourstep.py';path.write_text(build_matrix_fourstep(a,b,wide=wide))
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
    tile=32 if wide else 16
    bm,bn=(64,128) if wide else (32,64)
    for stage,(length,width) in enumerate(((a,b),(b,a))):
        p=1<<(length-1).bit_length()
        for bz in range(batch):
            pid[2]=bz
            for r in range((p+tile-1)//tile):
                pid[0]=r
                for col in range((width+tile-1)//tile):
                    pid[1]=col
                    ns[f'pack{stage}'](Pointer(x if stage==0 else mid),Pointer(packed[stage]),Pointer(tw))
        for bz in range(1 if wide else batch):
            pid[2]=bz
            for r in range(((batch if wide else 1)*width+bm-1)//bm):
                pid[0]=r
                for col in range((2*length+bn-1)//bn):
                    pid[1]=col
                    args=[Pointer(packed[stage]),Pointer(weights[stage]),Pointer(mid if stage==0 else last)]
                    if wide:args.append(batch)
                    ns[f'cube{stage}'](*args)
    for bz in range(batch):
        pid[2]=bz
        for r in range((a+tile-1)//tile):
            pid[0]=r
            for col in range((b+tile-1)//tile):
                pid[1]=col;ns['transpose'](Pointer(last),Pointer(guarded,64))
    actual=torch.view_as_complex(guarded[64:-64].reshape(batch,a*b,2))
    ref=torch.fft.ifft(torch.view_as_complex(x))*a*b if direction=='inverse' else torch.fft.fft(torch.view_as_complex(x))
    assert torch.isfinite(actual).all()
    assert torch.linalg.vector_norm(actual-ref)/torch.linalg.vector_norm(ref)<2e-6
    assert torch.equal(x,saved)
    assert (guarded[:64]==31415.25).all() and (guarded[-64:]==31415.25).all()


@pytest.mark.parametrize('wide',[False,True])
def test_emitted_fourstep_launcher_metadata(tmp_path,wide):
    from flagfft_codegen.kernels_npu_matrix_fourstep import emit_matrix_fourstep
    for phase in ('pack0','cube0','pack1','cube1','transpose'):
        meta=emit_matrix_fourstep(209,221,phase,'inverse',tmp_path,wide=wide)
        assert meta['inner_pack']==1 and meta['tle_fused_twiddle'] is True
        assert meta['signature'].count('*fp32:16')==(2 if phase=='transpose' else 3)
        if phase.startswith('cube') and wide:
            assert meta['arg_names'][-1]=='NB' and meta['signature'].endswith(',i32')
