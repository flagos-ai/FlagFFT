"""Natural-frequency coarse prefixes against an independent complete FFT."""
import math
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
from flagfft_codegen.kernels_npu_dif_coarse import build_coarse_dif
from flagfft_codegen.kernels_npu_dif import emit_vector_dif
from flagfft_codegen.npu_vector_basis import optimize_vector_basis
from test_npu_dif_cube import _tables
from test_npu_structured_fft import Pointer, execute_module


@pytest.mark.parametrize('factors', [
    (8,15,7,9,5), (8,15,7,9,4), (8,15,7,9,8), (8,15,7,9,9,5)])
@pytest.mark.parametrize('mode', ['forward', 'inverse', 'global_inverse'])
def test_complete_fft(tmp_path, factors, mode):
    global_inverse = mode == 'global_inverse'
    direction = 'inverse' if global_inverse else mode
    source, specs = build_coarse_dif(factors, direction,
        global_inverse=global_inverse, good_thomas=True, dense_prefix=True)
    source, specs = optimize_vector_basis(source, specs)
    path = tmp_path / 'dense.py'
    path.write_text(source)
    ns, pid = execute_module(path)
    pid.append(0)
    # Correct rounding is only a CPU layout oracle. Native FP32 device tests
    # independently enforce the original 5e-7 accuracy requirement.
    ns['tl'].dot = lambda a,b,**kw: (a.double() @ b.double()).float()
    tables = _tables(factors, specs, 1 if direction=='inverse' and not global_inverse else -1)
    angle = (1 if direction=='inverse' else -1)*2*math.pi*torch.arange(64,dtype=torch.float64)[:,None]*torch.arange(128)[None,:]/120
    valid = torch.arange(128)[None,:] < 120
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


def test_emitted_phase_identity(tmp_path):
    factors = (8,15,7,9,9,5)
    for phase in range(len(factors)-1):
        old = emit_vector_dif(factors,'inverse',phase,tmp_path,
            coarse=True,coarse_gt=True,global_inverse=True,vector_basis=True)
        before = Path(old['module_path']).read_text()
        new = emit_vector_dif(factors,'inverse',phase,tmp_path,
            coarse=True,coarse_dense=True,global_inverse=True,vector_basis=True)
        assert new['module_path'] != old['module_path']
        assert Path(old['module_path']).read_text() == before
        assert old['arg_names'] == new['arg_names']
        assert old['batch_per_block'] == new['batch_per_block']
        if phase==0:
            assert new['kernel_name']=='coarse_dense_prefix'
            assert new['signature']=='*fp32:16,*fp32:16,*fp32:16'
            assert new['num_stages']==1


def test_reject_unsupported_schedules(tmp_path):
    with pytest.raises(ValueError,match='batched 8 x 15'):
        build_coarse_dif((4,15,7,9,9,5),dense_prefix=True)
    with pytest.raises(ValueError,match='batched 8 x 15'):
        build_coarse_dif((8,15,7,9,9,5),single_batch=True,dense_prefix=True)
    with pytest.raises(ValueError,match='requires coarse DIF'):
        emit_vector_dif((8,15,7,9,9,5),'forward',0,tmp_path,coarse_dense=True)
