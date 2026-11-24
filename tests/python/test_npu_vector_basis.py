"""Check the mixed butterfly basis and spatial grids against a complete FFT."""
import math
import pytest

torch=pytest.importorskip('torch')
from flagfft_codegen.kernels_npu_dif_cube import build_cube_prefix_dif
from flagfft_codegen.kernels_npu_dif_coarse import build_coarse_dif
from flagfft_codegen.npu_vector_basis import optimize_vector_basis
from flagfft_codegen.kernels_npu_dif import emit_vector_dif
from test_npu_structured_fft import Pointer,execute_module
from test_npu_dif_cube import _tables


@pytest.mark.parametrize('factors,coarse',[
    ((17,15,13,7,2),False),((17,15,13,10,2),False),
    ((8,15,7,9,5),True),((8,15,7,9,9,5),True)])
@pytest.mark.parametrize('direction',['forward','inverse'])
def test_complete_basis(tmp_path,monkeypatch,factors,coarse,direction):
    monkeypatch.setenv('FLAGFFT_NPU_DIF_FUSED_FINISH','1')
    inverse=direction=='inverse'
    if coarse:
        source,specs=build_coarse_dif(factors,direction,global_inverse=inverse,good_thomas=True)
    else:
        source,specs=build_cube_prefix_dif(factors,direction,vector_first17=True,
            batch_pack=True,vector_second15=True,global_inverse=inverse,
            matrix_first17=True,prefix_pitch=32,codelet_reuse=True)
    tables=_tables(factors,specs,-1)
    if coarse:
        n=math.prod(factors);initial=factors[0]*15
        for i,s in enumerate(specs[1:],1):
            pf=torch.arange(s['physical_prefix']);local=pf%s['padded_prefix']
            logical=pf//s['padded_prefix']*initial+((local%factors[0])*15+(local//factors[0])*factors[0])%initial
            length=s['r']*s['m'];freq=logical if s['deferred_twiddle'] else logical//s['previous_prefix']
            denom=n if s['deferred_twiddle'] else factors[s['stage']-1]*length
            phase=-2*math.pi*torch.arange(length,dtype=torch.float64)[:,None]*freq[None,:]/denom
            tables[i]=torch.cat((phase.cos().float().flatten(),phase.sin().float().flatten()))
    else:
        angle=(-1 if direction=='forward' else 1)*2*math.pi*torch.arange(1,9,dtype=torch.float64)[:,None]*torch.arange(1,17)[None,:]/17
        tables[0]=torch.cat((angle.cos().float(),angle.sin().float()),0)
    source,updated=optimize_vector_basis(source,specs)
    if coarse:
        assert specs[1]['batch_pack']==4
        assert updated[1]['batch_pack']==1
        assert updated[1]['grid']==(math.ceil(updated[1]['m']/4),)
    p=tmp_path/'basis.py';p.write_text(source);ns,pid=execute_module(p);pid.append(0)
    ns['tl'].dot=lambda a,b,**kw:(a.double()@b.double()).float()
    n=math.prod(factors);batch=1 if n>100000 else 3
    torch.manual_seed(20261109);x=torch.randn(batch,n,2);saved=x.clone();current=x
    guard=torch.full((2*batch*n+128,),31415.25)
    for i,s in enumerate(updated):
        final=i+1==len(updated);output=guard if final else torch.empty(2*batch*s['physical_n'])
        grid=s['grid']+(batch,) if s['phase']=='transpose' else (s['grid'][0],math.ceil(batch/s.get('batch_pack',1)),1)
        for gx in range(grid[0]):
            for gy in range(grid[1]):
                for gz in range(grid[2]):
                    pid[:]=[gx,gy,gz]
                    args=[Pointer(current),Pointer(tables[i]),Pointer(output,64 if final else 0)]
                    if s.get('batch_pack',1)>1:args.append(batch)
                    ns[s['name']](*args)
        current=output
    actual=torch.view_as_complex(guard[64:-64].reshape(batch,n,2));z=torch.view_as_complex(x).to(torch.complex128)
    reference=torch.fft.ifft(z)*n if inverse else torch.fft.fft(z)
    assert torch.linalg.vector_norm(actual-reference)/torch.linalg.vector_norm(reference)<5e-7
    assert torch.equal(x,saved)
    assert (guard[:64]==31415.25).all() and (guard[-64:]==31415.25).all()


def test_other_schedule_identity():
    source='def untouched():\n    return 1\n'
    specs=[dict(name='untouched',r=5,phase='late',batch_pack=4,physical_prefix=128)]
    result,updated=optimize_vector_basis(source,specs)
    assert result==source and updated==specs


def test_emit_scope_and_identity(tmp_path,monkeypatch):
    monkeypatch.setenv('FLAGFFT_NPU_DIF_FUSED_FINISH','1')
    kwargs=dict(aligned=True,cube_prefix=True,vector_first17=True,batch_pack=True,
        vector_second15=True,matrix_first17=True,prefix_pitch=32,codelet_reuse=True)
    base=emit_vector_dif((17,15,13,7,2),'forward',4,tmp_path/'old',**kwargs)
    new=emit_vector_dif((17,15,13,7,2),'forward',4,tmp_path/'new',vector_basis=True,**kwargs)
    assert '_vectorbasis' in new['module_path'] and '_vectorbasis' not in base['module_path']
    assert new['arg_names']==base['arg_names']
    with pytest.raises(ValueError,match='public batch'):
        emit_vector_dif((32,32,16),'forward',0,tmp_path,flat_power2=True,vector_basis=True)
