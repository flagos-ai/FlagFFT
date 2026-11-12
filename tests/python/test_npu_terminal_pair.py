"""Terminal matrix layouts and public emitter identity, including both inverses."""
import math

import pytest

torch = pytest.importorskip('torch')
from flagfft_codegen.kernels_npu_dif import emit_vector_dif
from flagfft_codegen.npu_natural_prefix import build_natural_dif
from flagfft_codegen.npu_terminal_pair import replace_terminal_pair
from flagfft_codegen.npu_vector_basis import optimize_vector_basis
from test_npu_dif_cube import _tables
from test_npu_structured_fft import Pointer, execute_module


@pytest.mark.parametrize('factors', [
    (7,18,9,10,10,3), (7,18,10,10,10,3), (7,18,9,10,8,4)])
@pytest.mark.parametrize('mode', ['forward', 'inverse', 'global_inverse'])
def test_complete_fft(tmp_path, factors, mode):
    global_inverse = mode == 'global_inverse'
    direction = 'inverse' if global_inverse else mode
    sign = 1 if direction == 'inverse' and not global_inverse else -1
    source, specs = build_natural_dif(factors, direction, global_inverse)
    source, specs = optimize_vector_basis(source, specs, 'forward' if global_inverse else direction)
    tables = _tables(factors, specs, sign)
    angle = (1 if direction == 'inverse' else -1)*2*math.pi*torch.arange(64,dtype=torch.float64)[:,None]*torch.arange(128)[None,:]/126
    valid = torch.arange(128)[None,:] < 126
    tables[0] = torch.cat((torch.where(valid,angle.cos(),0).float(),
                          torch.where(valid,angle.sin(),0).float())).flatten()
    radix = factors[-2]*factors[-1]
    previous = factors[-3]
    f = torch.arange(32,dtype=torch.float64)[None,:,None]
    d = torch.arange(32,dtype=torch.float64)[None,None,:]
    shift = torch.arange(previous,dtype=torch.float64)[:,None,None]/previous
    phase = sign*2*math.pi*(f+shift)*d/radix
    co = torch.where((f<radix)&(d<radix), phase.cos(), 0).float()
    si = torch.where((f<radix)&(d<radix), phase.sin(), 0).float()
    tables[-2:] = [torch.cat((torch.cat((co,-si),2),torch.cat((si,co),2)),1).flatten()]
    source, specs = replace_terminal_pair(source, specs, global_inverse)
    path = tmp_path/'terminal.py'
    path.write_text(source)
    ns, pid = execute_module(path)
    # This is a layout oracle. Native tests check the actual IEEE FP32 dots.
    ns['tl'].dot = lambda a,b,acc=None,**kw: ((a.double()@b.double()) + (0 if acc is None else acc.double())).float()
    n = math.prod(factors)
    torch.manual_seed(20261112)
    x = torch.randn(1,n,2)
    saved = x.clone()
    guard = torch.full((2*n+128,),31415.25)
    current = x
    for i, spec in enumerate(specs):
        final = i+1 == len(specs)
        output = guard if final else torch.full((2*spec['physical_n'],),float('nan'))
        for gx in range(spec['grid'][0]):
            pid[:] = [gx,0,0]
            args = [Pointer(current),Pointer(tables[i]),Pointer(output,64 if final else 0)]
            if spec.get('batch_pack',1)>1: args.append(1)
            ns[spec['name']](*args)
        current = output
    got = torch.view_as_complex(guard[64:-64].reshape(1,n,2))
    z = torch.view_as_complex(x).cdouble()
    reference = torch.fft.fft(z) if direction=='forward' else torch.fft.ifft(z)*n
    assert torch.isfinite(got).all()
    assert torch.linalg.vector_norm(got-reference)/torch.linalg.vector_norm(reference)<5e-7
    assert torch.equal(x,saved)
    assert (guard[:64]==31415.25).all() and (guard[-64:]==31415.25).all()


def test_emitter_and_rejection(tmp_path):
    factors = (7,18,9,10,10,3)
    options = dict(coarse=True,coarse_dense=True,natural_prefix=True,vector_basis=True)
    old = emit_vector_dif(factors,'inverse',3,tmp_path,global_inverse=True,**options)
    new = emit_vector_dif(factors,'inverse',3,tmp_path,global_inverse=True,terminal_pair=True,**options)
    assert old['module_path'] != new['module_path']
    assert new['kernel_name'] == 'terminal_pair_matrix_batch'
    assert new['signature'] == '*fp32:16,*fp32:16,*fp32:16,i32'
    assert new['batch_per_block'] == 8 and new['num_stages'] == 2
    with pytest.raises(ValueError,match='Invalid DIF phase'):
        emit_vector_dif(factors,'forward',4,tmp_path,terminal_pair=True,**options)
    with pytest.raises(ValueError,match='natural prefix'):
        emit_vector_dif(factors,'forward',0,tmp_path,terminal_pair=True)
    for bad in ((7,18,9,10,8,3), (7,18,10,3)):
        with pytest.raises(ValueError,match='natural suffix'):
            replace_terminal_pair(*build_natural_dif(bad))
