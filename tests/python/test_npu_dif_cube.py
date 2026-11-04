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


@pytest.mark.parametrize('factors', [(17,13,3,4), (19,13,3,4), (17,15,7,2), (17,13,15,2), (7,15,3,4), (8,15,3,4), (7,15,10,4), (7,15,4,3), (17,13,4,5), (17,13,2,15), (7,15,3,9,4), (17,13,3,10,2), (17,15,13,2), (17,15,13,7,2), (19,15,3,2)])
@pytest.mark.parametrize('direction', ['forward', 'inverse'])
@pytest.mark.parametrize('dense_transpose,transpose_points', [('0','4096'), ('1','4096'), ('1','8192')])
def test_cube_prefix_complete_fft(tmp_path, monkeypatch, factors, direction, dense_transpose, transpose_points, vector_first17=False, batch_pack=False, vector_second15=False,global_inverse=False,matrix_first17=False):
    monkeypatch.setenv("FLAGFFT_NPU_DIF_DENSE_TRANSPOSE", dense_transpose)
    monkeypatch.setenv('FLAGFFT_NPU_DIF_TRANSPOSE_POINTS', transpose_points)
    monkeypatch.setenv('FLAGFFT_NPU_DIF_FUSED_FINISH', '1')
    monkeypatch.setenv('FLAGFFT_NPU_DIF_TRANSPOSE_TILE', '64')
    source, specs = build_cube_prefix_dif(factors, direction, vector_first17=vector_first17, batch_pack=batch_pack, vector_second15=vector_second15,global_inverse=global_inverse,matrix_first17=matrix_first17)
    path = tmp_path/'cube.py'
    path.write_text(source)
    ns, pid = execute_module(path)
    pid.append(0)
    ns['tl'].dot = lambda a,b,acc=None,input_precision=None: a@b if acc is None else a@b+acc
    sign = -1 if direction == 'forward' or global_inverse else 1
    tables = _tables(factors, specs, sign)
    if matrix_first17:
        angle=(-1 if direction=='forward' else 1)*2*math.pi*torch.arange(1,9,dtype=torch.float64)[:,None]*torch.arange(1,17)[None,:]/17
        tables[0]=torch.cat((angle.cos().float(),angle.sin().float()),dim=0)
    n, batch = math.prod(factors), 3
    torch.manual_seed(20261029)
    x = torch.randn(batch,n,2)
    saved = x.clone()
    guard = torch.full((x.numel()+128,),31415.25)
    current = x
    for index, spec in enumerate(specs):
        final = index+1 == len(specs)
        output = guard if final else torch.empty(batch*2*spec['physical_n'])
        grid = spec['grid']+(batch,) if spec['phase']=='transpose' else (spec['grid'][0],math.ceil(batch/spec.get('batch_pack',1)),1)
        for gx in range(grid[0]):
            for gy in range(grid[1]):
                for gz in range(grid[2] if len(grid)>2 else 1):
                    pid[:] = [gx,gy,gz]
                    args=[Pointer(current),Pointer(tables[index]),Pointer(output,64 if final else 0)]
                    if spec.get('batch_pack',1)>1:args.append(batch)
                    ns[spec['name']](*args)
        current = output
    got = torch.view_as_complex(guard[64:-64].reshape(batch,n,2))
    z = torch.view_as_complex(x)
    ref = torch.fft.fft(z) if direction=='forward' else torch.fft.ifft(z)*n
    assert torch.isfinite(got).all()
    assert torch.linalg.vector_norm(got-ref)/torch.linalg.vector_norm(ref)<5e-7
    assert torch.equal(x,saved)
    assert (guard[:64]==31415.25).all() and (guard[-64:]==31415.25).all()


@pytest.mark.parametrize('factors', [(8,15,3,4), (9,13,3,4), (7,15,10,4), (7,15,9,4),
    (17,15,13,2), (9,13,13,2), (9,13,11,2), (9,13,12,2)])
@pytest.mark.parametrize('direction', ['forward', 'inverse'])
@pytest.mark.parametrize('optimized', ['0', '1'])
def test_branch_dma_and_finish_options(tmp_path, monkeypatch, factors, direction, optimized):
    monkeypatch.setenv('FLAGFFT_NPU_DIF_GROUP_BRANCHES', optimized)
    monkeypatch.setenv('FLAGFFT_NPU_DIF_SINGLE_GATHER_FINISH', optimized)
    test_cube_prefix_complete_fft(tmp_path, monkeypatch, factors, direction, '1', '8192')


@pytest.mark.parametrize('factors', [(17,15,13,2), (17,13,3,4), (17,15,7,2)])
@pytest.mark.parametrize('direction', ['forward', 'inverse'])
def test_vector17_prefix_complete_fft(tmp_path, monkeypatch, factors, direction):
    monkeypatch.setenv('FLAGFFT_NPU_DIF_GROUP_BRANCHES', '1')
    test_cube_prefix_complete_fft(tmp_path, monkeypatch, factors, direction, '1', '8192', True)


@pytest.mark.parametrize('factors', [(7,15,9,4), (7,15,10,9,4), (17,15,13,10,2), (7,15,9,10,4), (17,15,13,4), (17,15,13,7,2)])
@pytest.mark.parametrize('direction', ['forward','inverse'])
def test_batch_grouped_masked_fft(tmp_path, monkeypatch, factors, direction):
    # Three batches exercise the masked final group for both two and four.
    test_cube_prefix_complete_fft(tmp_path, monkeypatch, factors, direction,
                                  '1', '8192', factors[0]==17, True)


@pytest.mark.parametrize('factors', [(7,15,15,10,4),(7,15,10,9,9,2)])
@pytest.mark.parametrize('direction', ['forward','inverse'])
def test_costed_suffix_complete_fft(tmp_path, monkeypatch, factors, direction):
    test_cube_prefix_complete_fft(tmp_path, monkeypatch, factors, direction,
                                  '1', '8192', False, True)


@pytest.mark.parametrize('factors',[(17,15,13,2),(7,15,9,4),(7,15,10,9,4)])
@pytest.mark.parametrize('direction',['forward','inverse'])
def test_factored_second_axis_complete_fft(tmp_path,monkeypatch,factors,direction):
    test_cube_prefix_complete_fft(tmp_path,monkeypatch,factors,direction,
                                  '1','8192',factors[0]==17,True,True)


@pytest.mark.parametrize('factors',[(7,15,9,4),(7,15,9,9)])
@pytest.mark.parametrize('direction',['forward','inverse'])
@pytest.mark.parametrize('wide',['0','1'])
def test_wide7_complete_fft(tmp_path,monkeypatch,factors,direction,wide):
    monkeypatch.setenv('FLAGFFT_NPU_DIF_FIRST7_WIDE',wide)
    monkeypatch.setenv('FLAGFFT_NPU_DIF_FIRST_AUTO_BLOCK','1')
    _,specs=build_cube_prefix_dif(factors,direction,batch_pack=True,vector_second15=True)
    width=1024 if wide=='1' else 512
    assert specs[0]['grid']==(math.ceil((math.prod(factors)//7)/width),)
    test_cube_prefix_complete_fft(tmp_path,monkeypatch,factors,direction,
                                  '1','8192',False,True,True)


@pytest.mark.parametrize('direction',['forward','inverse'])
@pytest.mark.parametrize('pattern',['basis','random','cancellation','tone'])
def test_component17_first_axis(tmp_path,monkeypatch,direction,pattern):
    # A partial DMA tile exercises every masked branch. The mathematical
    # reference transforms the original input directly along its radix axis.
    monkeypatch.setenv('FLAGFFT_NPU_DIF_COMPONENT17','1')
    monkeypatch.setenv('FLAGFFT_NPU_DIF_GROUP_BRANCHES','1')
    factors=(17,15,13,2);n=math.prod(factors);m=n//17
    source,specs=build_cube_prefix_dif(factors,direction,vector_first17=True,batch_pack=True)
    path=tmp_path/'component17.py';path.write_text(source);ns,pid=execute_module(path)
    batch=17 if pattern=='basis' else 3
    torch.manual_seed(20261103);x=torch.randn(batch,17,m,2)
    if pattern=='basis':
        x.zero_()
        for b in range(batch):x[b,b,(31*b)%m,b%2]=1.
    elif pattern=='cancellation':
        x[:,9:]=-x[:,1:9].flip(1)
        x[:,0]*=1e-6
    elif pattern=='tone':
        angle=2*torch.pi*torch.arange(17,dtype=torch.float64)/17
        x=torch.view_as_real(torch.polar(torch.ones_like(angle),angle)).float()[None,:,None,:].expand(batch,17,m,2).clone()
    if batch==3:x*=torch.tensor([1e-6,1.,1e6])[:,None,None,None]
    saved=x.clone();guard=torch.full((2*batch*n+128,),31415.25)
    first=specs[0]
    for gx in range(first['grid'][0]):
        for b in range(batch):
            pid[:]=[gx,b];ns[first['name']](Pointer(x),Pointer(torch.zeros(1)),Pointer(guard,64))
    planes=guard[64:-64].reshape(batch,2,17,m)
    got=torch.complex(planes[:,0],planes[:,1]).to(torch.complex128)
    z=torch.view_as_complex(x).to(torch.complex128)
    ref=torch.fft.fft(z,dim=1) if direction=='forward' else torch.fft.ifft(z,dim=1)*17
    assert torch.isfinite(got).all()
    assert torch.linalg.vector_norm(got-ref)/torch.linalg.vector_norm(ref)<5e-7
    assert torch.equal(x,saved)
    assert (guard[:64]==31415.25).all() and (guard[-64:]==31415.25).all()


@pytest.mark.parametrize('factors',[(17,15,13,7,8),(17,15,13,10,10,2),(17,15,13,10,10),(17,13,15,2)])
def test_global_inverse_complete(tmp_path,monkeypatch,factors):
    monkeypatch.setenv('FLAGFFT_NPU_DIF_COMPONENT17','1')
    monkeypatch.setenv('FLAGFFT_NPU_DIF_GROUP_BRANCHES','1')
    test_cube_prefix_complete_fft(tmp_path,monkeypatch,factors,'inverse','1','8192',
        True,True,factors[1]==15,True)


def test_global_inverse_metadata(tmp_path,monkeypatch):
    from flagfft_codegen.kernels_npu_dif import emit_vector_dif
    monkeypatch.setenv('FLAGFFT_NPU_DIF_FUSED_FINISH','1')
    meta=emit_vector_dif((17,15,13,2),'inverse',0,tmp_path,aligned=True,
        cube_prefix=True,vector_first17=True,batch_pack=True,global_inverse=True)
    assert meta['arg_names']==['X','TW','Y']


@pytest.mark.parametrize('factors',[(17,13,15,7,8),(17,15,13,10,10,2),(17,15,13,10,10)])
@pytest.mark.parametrize('direction,global_inverse',[('forward',False),('inverse',False),('inverse',True)])
def test_half_matrix17_complete(tmp_path,monkeypatch,factors,direction,global_inverse):
    monkeypatch.setenv('FLAGFFT_NPU_DIF_COMPONENT17','1')
    monkeypatch.setenv('FLAGFFT_NPU_DIF_GROUP_BRANCHES','1')
    test_cube_prefix_complete_fft(tmp_path,monkeypatch,factors,direction,'1','8192',
        True,True,factors[1]==15,global_inverse,True)
