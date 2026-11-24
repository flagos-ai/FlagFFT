"""Independent FFT oracle for every pass, direction, batch, and axis exchange."""
import math
import os
import pytest

torch=pytest.importorskip('torch')
from flagfft_codegen.kernels_npu_dif import build_vector_dif,emit_vector_dif
from test_npu_structured_fft import Pointer,execute_module


@pytest.mark.parametrize('aligned',[False,True])
@pytest.mark.parametrize('factors,batch',[
    ((5,5,3,8),2),((19,17,13,11),3),
    ((17,13,7,15,8),2),((17,13,15,10,10,2),1)])
@pytest.mark.parametrize('tile',[32,64])
@pytest.mark.parametrize('direction',['forward','inverse'])
def test_vector_dif_passes(tmp_path,monkeypatch,factors,batch,direction,tile,aligned):
    monkeypatch.setenv("FLAGFFT_NPU_DIF_TRANSPOSE_TILE",str(tile))
    n=math.prod(factors);sign=-1 if direction=='forward' else 1
    if aligned:
        from flagfft_codegen.kernels_npu_dif_aligned import build_aligned_vector_dif
        source,specs=build_aligned_vector_dif(factors,direction)
    else:
        source,specs=build_vector_dif(factors,direction)
    workspace_n=specs[0].get('physical_n',n)
    path=tmp_path/'dif.py';path.write_text(source);ns,pid=execute_module(path);pid.append(0)
    torch.manual_seed(20261029);x=torch.randn(batch,n,2);saved=x.clone();current=x
    for index,spec in enumerate(specs):
        if spec['phase'] in ('transpose','finish') or spec['stage']==0:
            weights=torch.zeros(1)
        else:
            stage=spec['stage'];previous=factors[stage-1];length=spec['r']*spec['m']
            if spec['phase']=='early':
                phase=sign*2*math.pi*torch.arange(previous,dtype=torch.float64)[:,None]*torch.arange(length)[None,:]/(previous*length)
            else:
                pf=torch.arange(spec.get('physical_prefix',spec['prefix']))
                if aligned:
                    pf=pf//spec['padded_prefix']*spec['initial_prefix']+pf%spec['padded_prefix']
                pf=pf[None,:];sample=torch.arange(length)[:,None]
                phase=sign*2*math.pi*(pf//spec['previous_prefix']).double()*sample/(previous*length)
            weights=torch.cat((phase.cos().float().flatten(),phase.sin().float().flatten()))
        out_length=n if index+1==len(specs) else workspace_n
        out=torch.full((2*batch*out_length+128,),31415.25)
        for b in range(batch):
            for gx in range(spec['grid'][0]):
                for gy in range(spec['grid'][1] if spec['phase']=='transpose' else 1):
                    pid[:]=[gx,gy,b] if spec['phase']=='transpose' else [gx,b,0]
                    ns[spec['name']](Pointer(current),Pointer(weights),Pointer(out,64))
        assert (out[:64]==31415.25).all() and (out[-64:]==31415.25).all()
        current=out[64:-64].clone()
    got=torch.view_as_complex(current.reshape(batch,n,2));z=torch.view_as_complex(x)
    reference=torch.fft.fft(z) if direction=='forward' else torch.fft.ifft(z)*n
    assert torch.isfinite(got).all()
    threshold=5e-7 if os.environ.get('FLAGFFT_NPU_DIF_FUSED_FINISH')=='1' else 1e-6
    assert torch.linalg.vector_norm(got-reference)/torch.linalg.vector_norm(reference)<threshold
    assert torch.equal(x,saved)
    metadata=emit_vector_dif(factors,direction,len(specs)-1,tmp_path/'metadata',aligned=aligned)
    assert metadata['kernel_name']==specs[-1]['name']
    assert metadata['signature']=='*fp32:16,*fp32:16,*fp32:16'


@pytest.mark.parametrize('factors,batch',[
    ((17,13,7,15,8),2),((17,13,15,10,10,2),1),
    ((17,13,15,10,10),2),((7,15,10,9,9,4),2)])
@pytest.mark.parametrize('direction',['forward','inverse'])
def test_fused_finish(tmp_path,monkeypatch,factors,batch,direction):
    monkeypatch.setenv('FLAGFFT_NPU_DIF_FUSED_FINISH','1')
    # Reuse the independent full FFT oracle and per-pass guard checks.
    test_vector_dif_passes(tmp_path,monkeypatch,factors,batch,direction,64,True)
    from flagfft_codegen.kernels_npu_dif_aligned import build_aligned_vector_dif
    _,specs=build_aligned_vector_dif(factors,direction)
    assert specs[-1]['name']=='fused_finish'
    assert len(specs)==len(factors)+1
