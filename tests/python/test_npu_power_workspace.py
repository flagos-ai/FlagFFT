"""Verify compact API buffers and untouched gaps in padded DIF workspaces."""
import math
import pytest
torch=pytest.importorskip('torch')
from flagfft_codegen.kernels_npu_power_dif import build
from test_npu_structured_fft import Pointer,execute_module,roots

@pytest.mark.parametrize('factors',[(8,8,4),(16,16,16),(32,32,32,16)])
@pytest.mark.parametrize('mode',['forward','inverse','global_inverse'])
def test_padded_workspace_complete(tmp_path,factors,mode):
    global_inverse=mode=='global_inverse';direction='forward' if mode=='forward' else 'inverse'
    source,specs=build(factors,direction,terminal8=True,global_inverse=global_inverse,workspace_padding=512)
    path=tmp_path/'padded.py';path.write_text(source);ns,pid=execute_module(path)
    n=math.prod(factors);pn=n+512;batch=3;torch.manual_seed(20261107)
    x=torch.randn(batch,n,2);saved=x.clone();current=x
    work=[torch.full((batch*2*pn,),21828.5) for _ in range(2)]
    guard=torch.full((x.numel()+256,),31415.25);sign=-1 if direction=='forward' or global_inverse else 1
    for k,s in enumerate(specs):
        w=roots(s['r'],s['c'])
        if k:
            digit=torch.arange(s['prefix'])//s['previous_prefix'];sample=torch.arange(s['r']*s['m'])
            angle=sign*2*math.pi*sample[:,None].double()*digit[None,:]/(factors[k-1]*s['r']*s['m'])
            w=torch.cat((angle.cos().float().flatten(),angle.sin().float().flatten(),w))
        final=k+1==len(specs);out=Pointer(guard,128) if final else Pointer(work[k%2])
        for gx in range(s['grid'][0]):
            for b in range(batch):pid[:]=[gx,b];ns[s['name']](Pointer(current),Pointer(w),out)
        if not final:
            value=work[k%2].reshape(batch,2,pn)
            assert (value[:,:,n:]==21828.5).all()
            current=work[k%2]
    got=torch.view_as_complex(guard[128:-128].reshape_as(x));z=torch.view_as_complex(x).cdouble()
    ref=torch.fft.fft(z) if direction=='forward' else torch.fft.ifft(z)*n
    assert torch.isfinite(got).all()
    assert torch.linalg.vector_norm(got-ref)/torch.linalg.vector_norm(ref)<5e-7
    assert torch.equal(x,saved)
    assert (guard[:128]==31415.25).all() and (guard[-128:]==31415.25).all()

@pytest.mark.parametrize('padding',[-1,32,4096,True])
def test_reject_unsupported_padding(padding):
    with pytest.raises(ValueError,match='workspace padding'):
        build((16,16),'forward',workspace_padding=padding)

def test_padding_emitter_layout_identity(tmp_path):
    from pathlib import Path
    from flagfft_codegen.kernels_npu_dif import emit_vector_dif
    plain=emit_vector_dif((16,16),'forward',0,tmp_path,flat_power2=True)
    padded=emit_vector_dif((16,16),'forward',0,tmp_path,flat_power2=True,workspace_padding=512)
    assert plain['module_path']!=padded['module_path']
    assert Path(plain['module_path']).exists()
    assert padded['workspace_length']==768
    assert padded['signature']==plain['signature']=='*fp32:16,*fp32:16,*fp32:16'
    with pytest.raises(ValueError,match='requires flat power DIF'):
        emit_vector_dif((8,8,4),'forward',0,tmp_path,workspace_padding=512)
