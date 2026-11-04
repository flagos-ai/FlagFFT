"""Whole FFT comparison with a double precision oracle and poisoned buffers."""
import math
import pytest

torch=pytest.importorskip('torch')
from flagfft_codegen.kernels_npu_dif_coarse import build_coarse_dif
from test_npu_structured_fft import Pointer,execute_module


@pytest.mark.parametrize('factors',[(8,15,7,9,9,5),(4,15,7,9,9,5),(8,15,7,9,5),(4,15,8,4,3,2)])
@pytest.mark.parametrize('direction',['forward','inverse','global_inverse'])
def test_coarse_full_fft(tmp_path,factors,direction):
    global_inverse=direction=='global_inverse'
    if global_inverse:direction='inverse'
    source,specs=build_coarse_dif(factors,direction,global_inverse=global_inverse)
    path=tmp_path/'coarse.py';path.write_text(source);ns,pid=execute_module(path);pid.append(0)
    n=math.prod(factors);batch=3;a,b=factors[:2];initial=a*b;padded=a*16
    sign=-1 if direction=='forward' or global_inverse else 1
    phase=-2*math.pi*torch.arange(a,dtype=torch.float64)[:,None]*torch.arange(16)[None,:]/initial
    tables=[torch.cat((phase.cos().float().flatten(),phase.sin().float().flatten()))]
    for spec in specs[1:]:
        pp=spec['physical_prefix'];prefix=spec['prefix'];r=spec['r'];m=spec['m']
        pf=torch.arange(pp);logical=pf//padded*initial+pf%padded
        sample=torch.arange(r*m,dtype=torch.float64)[:,None]
        phase=(sign*2*math.pi*sample*logical[None,:]/n if spec['stage']==2 else
            sign*2*math.pi*sample*(logical[None,:]//spec['previous_prefix'])/(factors[spec['stage']-1]*r*m))
        tables.append(torch.cat((phase.cos().float().flatten(),phase.sin().float().flatten())))
    torch.manual_seed(20261104);x=torch.randn(batch,n,2);saved=x.clone();current=x
    guard=torch.full((2*batch*n+128,),31415.25)
    for index,spec in enumerate(specs):
        last=index+1==len(specs)
        y=guard if last else torch.full((2*batch*specs[0]['physical_n'],),float('nan'))
        for gx in range(spec['grid'][0]):
            for gy in range(math.ceil(batch/spec.get('batch_pack',1))):
                pid[:]=[gx,gy,0]
                args=[Pointer(current),Pointer(tables[index]),Pointer(y,64 if last else 0)]
                if spec.get('batch_pack',1)>1:args.append(batch)
                ns[spec['name']](*args)
        current=y
    z=torch.view_as_complex(x).to(torch.complex128)
    reference=torch.fft.fft(z) if direction=='forward' else torch.fft.ifft(z)*n
    got=torch.view_as_complex(guard[64:-64].reshape(batch,n,2))
    assert torch.isfinite(got).all()
    assert torch.linalg.vector_norm(got-reference)/torch.linalg.vector_norm(reference)<5e-7
    assert torch.equal(x,saved)
    assert (guard[:64]==31415.25).all() and (guard[-64:]==31415.25).all()
