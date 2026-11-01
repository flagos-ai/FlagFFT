"""Execute the complete chirped DIF pipeline against an independent FFT."""
import math

import pytest

torch = pytest.importorskip('torch')
from flagfft_codegen.kernels_npu_dif_cube import build_cube_prefix_dif
from test_npu_structured_fft import Pointer, execute_module


def _tables(factors, specs, sign):
    n = math.prod(factors)
    tables = []
    for s in specs:
        if s['phase'] in ('prepare', 'transpose', 'finish'):
            tables.append(torch.zeros(1))
            continue
        stage, radix = s['stage'], s['r']
        if s.get('planar_cube') or s.get('chirped_cube'):
            pr = s['pr']
            count = factors[0] if stage else 1
            w = torch.zeros(count, 2*pr, 2*pr)
            for pf in range(count):
                shift = pf/count if stage else 0.
                angle = sign*2*math.pi*(torch.arange(radix, dtype=torch.float64)[:, None]+shift)*torch.arange(radix)[None, :]/radix
                co, si = angle.cos().float(), angle.sin().float()
                imag = radix if stage else pr
                w[pf, :radix, :radix] = co
                w[pf, :radix, pr:pr+radix] = -si
                w[pf, imag:imag+radix, :radix] = si
                w[pf, imag:imag+radix, pr:pr+radix] = co
            tables.append(w)
            continue
        if stage == 0:
            tables.append(torch.zeros(1))
            continue
        length = radix*s['m']
        pf = torch.arange(s['physical_prefix'])
        logical = pf//s['padded_prefix']*s['initial_prefix']+pf%s['padded_prefix']
        freq = logical if s.get('deferred_twiddle') else logical//s['previous_prefix']
        denom = n if s.get('deferred_twiddle') else factors[stage-1]*length
        angle = sign*2*math.pi*torch.arange(length, dtype=torch.float64)[:, None]*freq[None, :]/denom
        tables.append(torch.cat((angle.cos().float().flatten(), angle.sin().float().flatten())))
    return tables


@pytest.mark.parametrize('factors', [(17,13,3,4), (19,13,3,4), (17,15,7,2), (17,13,15,2), (7,15,3,4), (8,15,3,4), (7,15,10,4), (7,15,4,3), (17,13,4,5), (17,13,2,15), (7,15,3,9,4), (17,13,3,10,2), (17,15,13,2), (17,15,13,7,2)])
@pytest.mark.parametrize('direction', ['forward', 'inverse'])
@pytest.mark.parametrize('dense_transpose', ['0', '1'])
def test_cube_prefix_complete_fft(tmp_path, monkeypatch, factors, direction, dense_transpose):
    monkeypatch.setenv("FLAGFFT_NPU_DIF_DENSE_TRANSPOSE", dense_transpose)
    monkeypatch.setenv('FLAGFFT_NPU_DIF_FUSED_FINISH', '1')
    monkeypatch.setenv('FLAGFFT_NPU_DIF_TRANSPOSE_TILE', '64')
    source, specs = build_cube_prefix_dif(factors, direction)
    path = tmp_path/'cube.py'
    path.write_text(source)
    ns, pid = execute_module(path)
    pid.append(0)
    ns['tl'].dot = lambda a,b,acc=None,input_precision=None: a@b if acc is None else a@b+acc
    sign = -1 if direction == 'forward' else 1
    tables = _tables(factors, specs, sign)
    n, batch = math.prod(factors), 3
    torch.manual_seed(20261029)
    x = torch.randn(batch,n,2)
    saved = x.clone()
    guard = torch.full((x.numel()+128,),31415.25)
    current = x
    for index, spec in enumerate(specs):
        final = index+1 == len(specs)
        output = guard if final else torch.empty(batch*2*spec['physical_n'])
        grid = spec['grid']+(batch,) if spec['phase']=='transpose' else (spec['grid'][0],batch,1)
        for gx in range(grid[0]):
            for gy in range(grid[1]):
                for gz in range(grid[2] if len(grid)>2 else 1):
                    pid[:] = [gx,gy,gz]
                    ns[spec['name']](Pointer(current),Pointer(tables[index]),Pointer(output,64 if final else 0))
        current = output
    got = torch.view_as_complex(guard[64:-64].reshape(batch,n,2))
    z = torch.view_as_complex(x)
    ref = torch.fft.fft(z) if sign<0 else torch.fft.ifft(z)*n
    assert torch.isfinite(got).all()
    assert torch.linalg.vector_norm(got-ref)/torch.linalg.vector_norm(ref)<5e-7
    assert torch.equal(x,saved)
    assert (guard[:64]==31415.25).all() and (guard[-64:]==31415.25).all()
