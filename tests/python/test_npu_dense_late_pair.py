"""Complete FFT layout, inverse conventions and merged frequency twiddles."""
import math
from pathlib import Path
import pytest

torch=pytest.importorskip('torch')
from test_npu_batch_dense_prefix import _build,_weights,selected_options
from test_npu_structured_fft import Pointer,execute_module
from flagfft_codegen.npu_dense_late_pair import replace_late_pair
from flagfft_codegen.kernels_npu_dif import emit_vector_dif


@pytest.mark.parametrize('factors',[
    (17,15,13,7,8),(17,15,13,7,3,2),(17,15,13,7,8,2),(17,15,7,13,8)])
@pytest.mark.parametrize('batch_reuse',[1,8])
@pytest.mark.parametrize('mode',['forward','inverse','global_inverse'])
def test_complete_fft(tmp_path,factors,mode,batch_reuse):
    global_inverse=mode=='global_inverse'
    direction='inverse' if global_inverse else mode
    sign=1 if direction=='inverse' and not global_inverse else -1
    source,specs=_build(factors,direction,finish4=factors[-1]==8,global_inverse=global_inverse)
    tables=_weights(factors,specs,direction,global_inverse=global_inverse)
    pn=specs[0]['physical_n'];following=specs[3];prefix=specs[1]['prefix']
    tables[0]=torch.cat((tables[0].flatten(),tables[1][:2*pn]))
    freq=torch.arange(64,dtype=torch.float64)[:,None]
    d=torch.arange(128)[None,:]
    angle=sign*2*math.pi*freq*d/91
    tables[1]=torch.cat((torch.where(d<91,angle.cos(),0).float(),
                         torch.where(d<91,angle.sin(),0).float())).flatten()
    pf=torch.arange(following['physical_prefix'])
    logical=pf//following['padded_prefix']*following['initial_prefix']+pf%following['padded_prefix']
    length=following['r']*following['m']
    angle=sign*2*math.pi*torch.arange(length,dtype=torch.float64)[:,None]*(logical//prefix)[None,:]/(91*length)
    tables[3]=torch.cat((angle.cos().float().flatten(),angle.sin().float().flatten(),tables[3][2*pn:]))
    del tables[2]
    source,specs=replace_late_pair(source,specs,batch_reuse)
    path=tmp_path/'late.py';path.write_text(source);ns,pid=execute_module(path);pid.append(0)
    ns['tl'].dot=lambda a,b,acc=None,**kw:(a.double()@b.double()+(acc.double() if acc is not None else 0)).float()
    n=math.prod(factors);batch=3
    torch.manual_seed(20261111);x=torch.randn(batch,n,2);x[1]*=1e-6;x[2]*=1e6
    saved=x.clone();current=x
    for i,s in enumerate(specs):
        final=i+1==len(specs)
        guarded=torch.full((2*batch*(n if final else pn)+128,),31415.25)
        guarded[64:-64]=float('nan')
        for gx in range(s['grid'][0]):
            for gy in range(math.ceil(batch/s.get('batch_pack',1))):
                pid[:]=[gx,gy,0]
                args=[Pointer(current),Pointer(tables[i]),Pointer(guarded,64)]
                if s.get('batch_pack',1)>1 or s.get('finish_map'):args.append(batch)
                ns[s['name']](*args)
        assert (guarded[:64]==31415.25).all() and (guarded[-64:]==31415.25).all()
        current=guarded[64:-64]
    got=torch.view_as_complex(current.reshape_as(x)).cdouble();z=torch.view_as_complex(x).cdouble()
    expected=torch.fft.fft(z) if direction=='forward' else torch.fft.ifft(z)*n
    error=torch.linalg.vector_norm(got-expected,dim=1)/torch.linalg.vector_norm(expected,dim=1)
    assert torch.isfinite(got).all() and (error<5e-7).all()
    assert torch.equal(x,saved)


def test_metadata_and_rejected_geometry(tmp_path):
    kwargs=dict(aligned=True,cube_prefix=True,batch_pack=True,vector_first17=True,
        vector_second15=True,matrix_first17=True,prefix_pitch=32,dense_prefix=True,
        dense_prefix_columns=64,dense_prefix_frequencies=64,codelet_reuse=True,vector_basis=True)
    factors=(17,15,13,7,8)
    before=emit_vector_dif(factors,'forward',1,tmp_path,**kwargs)
    text=Path(before['module_path']).read_text()
    after=emit_vector_dif(factors,'forward',1,tmp_path,**kwargs,dense_late_pair=True)
    assert after['module_path']!=before['module_path']
    assert Path(before['module_path']).read_text()==text
    assert after['kernel_name']=='dense_late_pair' and after['num_stages']==2
    assert after['batch_per_block']==1 and after['signature']=='*fp32:16,*fp32:16,*fp32:16'
    grouped=emit_vector_dif(factors,'forward',1,tmp_path,**kwargs,
                           dense_late_pair=True,late_pair_batch=8)
    assert grouped['module_path']!=after['module_path']
    assert grouped['kernel_name']=='dense_late_pair_batch'
    assert grouped['batch_per_block']==8 and grouped['arg_names']==['X','TW','Y','NB']
    assert grouped['signature']==after['signature']+',i32'
    with pytest.raises(ValueError,match='requires late-pair fusion'):
        emit_vector_dif(factors,'forward',1,tmp_path,**kwargs,late_pair_batch=8)
    for factors in ((17,15,13,10,10,2),(17,15,13,7)):
        source,specs=_build(factors,'forward')
        with pytest.raises(ValueError,match='13 x 7 pair'):replace_late_pair(source,specs)
    with pytest.raises(ValueError,match='batched dense Cube'):
        emit_vector_dif((17,15,13,7,8),'forward',0,tmp_path,dense_late_pair=True)


def test_invalid_batch_groups():
    for group in (0,2,4,16,True):
        with pytest.raises(ValueError,match='Unsupported late matrix batch group'):
            replace_late_pair('',[],group)
