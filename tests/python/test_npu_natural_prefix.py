"""126-point family: complete layouts, local/global inverse and emitter identity."""
import math
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
from flagfft_codegen.npu_natural_prefix import build_natural_dif
from flagfft_codegen.kernels_npu_dif import emit_vector_dif
from flagfft_codegen.npu_vector_basis import optimize_vector_basis
from test_npu_dif_cube import _tables
from test_npu_structured_fft import Pointer, execute_module


@pytest.mark.parametrize('factors', [
    (7,18,9,10,10,3), (7,18,9,10,8,3), (7,18,9,10,8,6), (7,18,9,10,10,6)])
@pytest.mark.parametrize('mode', ['forward', 'inverse', 'global_inverse'])
def test_complete_fft(tmp_path, factors, mode):
    global_inverse = mode == 'global_inverse'
    direction = 'inverse' if global_inverse else mode
    source, specs = build_natural_dif(factors, direction,global_inverse=global_inverse)
    source, specs = optimize_vector_basis(source, specs, 'forward' if global_inverse else direction)
    path = tmp_path / 'dense.py'
    path.write_text(source)
    ns, pid = execute_module(path)
    pid.append(0)
    # Correct rounding is only a CPU layout oracle. Native FP32 device tests
    # independently enforce the original 5e-7 accuracy requirement.
    ns['tl'].dot = lambda a,b,**kw: (a.double() @ b.double()).float()
    tables = _tables(factors, specs, 1 if direction=='inverse' and not global_inverse else -1)
    angle = (1 if direction=='inverse' else -1)*2*math.pi*torch.arange(64,dtype=torch.float64)[:,None]*torch.arange(128)[None,:]/126
    valid = torch.arange(128)[None,:] < 126
    tables[0] = torch.cat((torch.where(valid,angle.cos(),0).float(),
                           torch.where(valid,angle.sin(),0).float())).flatten()
    n = math.prod(factors)
    batch = 1 if n>100000 else 3
    torch.manual_seed(20261111)
    x = torch.randn(batch,n,2)
    saved = x.clone()
    current = x
    guard = torch.full((2*batch*n+128,),31415.25)
    for i, spec in enumerate(specs):
        final = i+1 == len(specs)
        output = guard if final else torch.full((2*batch*spec['physical_n'],),float('nan'))
        for gx in range(spec['grid'][0]):
            for gy in range(math.ceil(batch/spec.get('batch_pack',1))):
                pid[:] = [gx,gy,0]
                args = [Pointer(current),Pointer(tables[i]),Pointer(output,64 if final else 0)]
                if spec.get('batch_pack',1)>1: args.append(batch)
                ns[spec['name']](*args)
        current = output
    got = torch.view_as_complex(guard[64:-64].reshape(batch,n,2))
    z = torch.view_as_complex(x).cdouble()
    reference = torch.fft.fft(z) if direction=='forward' else torch.fft.ifft(z)*n
    assert torch.isfinite(got).all()
    assert torch.linalg.vector_norm(got-reference)/torch.linalg.vector_norm(reference)<5e-7
    assert torch.equal(x,saved)
    assert (guard[:64]==31415.25).all() and (guard[-64:]==31415.25).all()



def test_metadata_and_rejection(tmp_path):
    factors=(7,18,9,10,10,3)
    for phase in range(len(factors)-1):
        meta=emit_vector_dif(factors,'inverse',phase,tmp_path,coarse=True,
            coarse_dense=True,natural_prefix=True,global_inverse=True,vector_basis=True)
        assert meta['signature']=='*fp32:16,*fp32:16,*fp32:16'+(',i32' if phase in (1,2,3) else '')
        assert meta['batch_per_block']==(4 if phase in (1,2,3) else 1)
        if phase==0:assert meta['kernel_name']=='coarse_dense_prefix'
    for bad in ((8,15,7,9),(7,18,13,8)):
        with pytest.raises(ValueError,match='7 x 18'):build_natural_dif(bad)
    with pytest.raises(ValueError,match='batched dense coarse'):
        emit_vector_dif(factors,'forward',0,tmp_path,natural_prefix=True)
