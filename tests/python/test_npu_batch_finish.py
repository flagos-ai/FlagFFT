"""Final-stage FFT oracle with arbitrary complex phases and partial batches."""
import math
import pytest
torch=pytest.importorskip('torch')
from flagfft_codegen.metadata import _module_source
from flagfft_codegen.npu_batch_finish import replace_finish
from flagfft_codegen.kernels_npu_dif import emit_vector_dif
from test_npu_structured_fft import Pointer,execute_module


@pytest.mark.parametrize('radix,initial,padded,groups',[(8,255,256,3),(16,127,128,5)])
@pytest.mark.parametrize('batch',[1,5,8])
@pytest.mark.parametrize('mode,conjugate',[('forward',False),('inverse',False),('forward',True)])
def test_batched_finish_oracle(tmp_path,radix,initial,padded,groups,batch,mode,conjugate):
    prefix=groups*initial;pp=groups*padded;pn=radix*pp;n=radix*prefix
    spec=dict(name='fused_finish',r=radix,prefix=prefix,physical_prefix=pp,
              padded_prefix=padded,initial_prefix=initial,physical_n=pn)
    source,passes=replace_finish('@triton.jit\ndef fused_finish(X,TW,Y):\n    pass\n',[spec],mode,conjugate)
    source=_module_source(source,(radix,))+'\nimport triton.language.extra.cann.extension as al\n'
    path=tmp_path/'finish.py';path.write_text(source);ns,pid=execute_module(path)
    torch.manual_seed(20261108)
    x=torch.randn(batch,2,radix,groups,padded);saved=x.clone()
    angle=torch.randn(radix,groups,padded);angle[0]=0
    tw=torch.stack((angle.cos(),angle.sin()))
    rows=passes[-1]['finish_rows'];points=4*rows*padded
    words=torch.arange(2*rows*padded)
    take=torch.where(words<2*rows*initial,(words&1)*points+
                     words//(2*initial)*padded+(words//2)%initial,0).float()
    weights=torch.cat((tw.flatten(),take))
    guard=torch.full((batch*2*n+256,),31415.25)
    for gx in range(passes[-1]['grid'][0]):
        for by in range(math.ceil(batch/4)):
            pid[:]=[gx,by]
            ns['fused_finish'](Pointer(x),Pointer(weights),Pointer(guard,128),batch)
    value=torch.complex(x[:,0],x[:,1])*torch.complex(tw[0],tw[1])
    expected=torch.fft.fft(value.cdouble(),dim=1) if mode=='forward' else torch.fft.ifft(value.cdouble(),dim=1)*radix
    if conjugate:expected=expected.conj()
    expected=expected[...,:initial].reshape(batch,n)
    got=torch.view_as_complex(guard[128:-128].reshape(batch,n,2))
    assert torch.linalg.vector_norm(got-expected)/torch.linalg.vector_norm(expected)<5e-7
    assert torch.equal(x,saved)
    assert (guard[:128]==31415.25).all() and (guard[-128:]==31415.25).all()


def test_finish_emitter_identity(tmp_path,monkeypatch):
    monkeypatch.setenv('FLAGFFT_NPU_DIF_FUSED_FINISH','1')
    kw=dict(aligned=True,cube_prefix=True,batch_pack=True,vector_first17=True,
            vector_second15=True,matrix_first17=True,prefix_pitch=32,codelet_reuse=True)
    factors=(17,15,13,7,8)
    plain=emit_vector_dif(factors,'forward',5,tmp_path,**kw)
    packed=emit_vector_dif(factors,'forward',5,tmp_path,finish_batch4=True,**kw)
    assert plain['module_path']!=packed['module_path']
    assert plain['batch_per_block']==1 and packed['batch_per_block']==4
    assert packed['signature'].endswith(',i32') and packed['arg_names'][-1]=='NB'
    with pytest.raises(ValueError,match='requires batch-grouped'):
        emit_vector_dif(factors,'forward',0,tmp_path,finish_batch4=True)
