"""Execute the generated mixed-radix leaves on CPU against an independent FFT."""
import math
import pytest

torch=pytest.importorskip('torch')
from flagfft_codegen.kernels_common import LeafPlan,codelet_radices_for
from flagfft_codegen.kernels_leaf import _build_leaf_kernel_source_for_io
from flagfft_codegen.metadata import _module_source
from flagfft_codegen import target
from test_npu_structured_fft import Pointer,execute_module


@pytest.fixture
def npu_policy(monkeypatch):
    old=target._target.get();target.set_codegen_target('npu:Ascend910B4:1')
    for key,value in dict(FLAGFFT_NPU_FOURSTEP_LEAF='1',FLAGFFT_NPU_2D_EXCHANGE='direct_all',
            FLAGFFT_NPU_2D_LANE_MIN='1',FLAGFFT_NPU_PORTABLE_VECTOR_IO='1',
            FLAGFFT_NPU_VECTOR_REMAINDER='1',FLAGFFT_NPU_COMPACT_STAGES='1',FLAGFFT_NPU_AFFINE_MATRIX_IO='1').items():
        monkeypatch.setenv(key,value)
    yield
    target.set_codegen_target(old)


def tables(factors,direction):
    n=math.prod(factors);result=[]
    for stage in range(1,len(factors)):
        radix=factors[stage];prefix=math.prod(factors[:stage]);count=n//radix
        angle=(1 if direction=='inverse' else -1)*2*math.pi*torch.arange(radix,dtype=torch.float64)[:,None]*(torch.arange(count)[None,:]%prefix)/(prefix*radix)
        result.extend((angle.cos().float(),angle.sin().float()))
    return [Pointer(x) for x in result]


@pytest.mark.parametrize('factors',[(7,6,5),(19,11),(17,13),(13,17,3),(17,13,2)])
@pytest.mark.parametrize('pack',[1,2,4])
@pytest.mark.parametrize('mode,direction',[('contiguous','forward'),('contiguous','inverse'),('contiguous_r2c','forward'),('contiguous_c2r','inverse')])
def test_compact_leaf(npu_policy,monkeypatch,tmp_path,factors,pack,mode,direction):
    monkeypatch.setenv('FLAGFFT_NPU_2D_BATCH_PACK',str(pack))
    n=math.prod(factors);batch=pack+1
    plan=LeafPlan(n,factors,1,1,4,(),1<<(n-1).bit_length(),direction,'complex64')
    name,body=_build_leaf_kernel_source_for_io(plan,io_mode=mode)
    path=tmp_path/'kernel.py';path.write_text(_module_source(body,tuple(sorted(codelet_radices_for(factors)))))
    ns,pid=execute_module(path)
    torch.manual_seed(20261028)
    z=torch.complex(torch.randn(batch,n),torch.randn(batch,n))
    if mode=='contiguous_r2c':
        x=z.real.contiguous();ref=torch.fft.rfft(x)
    elif mode=='contiguous_c2r':
        x=torch.fft.rfft(z.real).contiguous();ref=torch.fft.irfft(x,n=n)*n
    else:
        x=z;ref=torch.fft.fft(x) if direction=='forward' else torch.fft.ifft(x)*n
    src=x if not x.is_complex() else torch.view_as_real(x)
    expected=ref if not ref.is_complex() else torch.view_as_real(ref)
    saved=src.clone();out=torch.full((expected.numel()+128,),31415.25)
    args=[Pointer(src),Pointer(out,64),*tables(factors,direction)]
    if mode!='contiguous':args.extend((x.shape[-1],ref.shape[-1]))
    args.append(batch)
    for block in range((batch+pack-1)//pack):
        pid[0]=block;ns[name](*args)
    got=out[64:-64].reshape(expected.shape)
    assert torch.isfinite(got).all()
    assert torch.linalg.vector_norm(got-expected)/torch.linalg.vector_norm(expected)<2e-6
    assert torch.equal(src,saved)
    assert (out[:64]==31415.25).all() and (out[-64:]==31415.25).all()


@pytest.mark.parametrize('row_factors,col_factors',[
    ((19,11),(17,13)),((7,6,5),(8,4)),((13,17,3),(8,4)),((17,13,2),(4,4))])
@pytest.mark.parametrize('pack',['1','4','auto'])
@pytest.mark.parametrize('direction',['forward','inverse'])
def test_mixed_fourstep_contract(npu_policy,monkeypatch,tmp_path,row_factors,col_factors,pack,direction):
    from flagfft_codegen.kernels_common import emitted_leaf_factors,four_step_row_inner_pack_for,four_step_col_inner_pack_for
    monkeypatch.setenv('FLAGFFT_NPU_FOURSTEP_PACK',pack)
    a,b=math.prod(row_factors),math.prod(col_factors);n=a*b
    torch.manual_seed(20261028)
    x=torch.randn(1,n,2);saved=x.clone();temp=torch.empty_like(x)
    out=torch.full((x.numel()+128,),31415.25)
    j2,k1=torch.arange(b)[:,None],torch.arange(a)[None,:]
    angle=(1 if direction=='inverse' else -1)*2*math.pi*j2.double()*k1.double()/n
    cross=torch.stack((angle.cos(),angle.sin()),-1).float()
    for first,factors in ((True,row_factors),(False,col_factors)):
        length=math.prod(factors)
        plan=LeafPlan(length,factors,1,1,4,(),1<<(length-1).bit_length(),direction,'complex64')
        mode='four_step_row' if first else 'four_step_col'
        name,body=_build_leaf_kernel_source_for_io(plan,io_mode=mode,four_step_n1=a,four_step_n2=b)
        deps=codelet_radices_for(factors)|codelet_radices_for(emitted_leaf_factors(plan))
        path=tmp_path/(mode+'.py');path.write_text(_module_source(body,tuple(sorted(deps))))
        ns,pid=execute_module(path)
        inner=(four_step_row_inner_pack_for if first else four_step_col_inner_pack_for)(a,b,'complex64',plan)
        args=([Pointer(x),Pointer(temp)] if first else [Pointer(temp),Pointer(cross),Pointer(out,64)])
        args.extend(tables(factors,direction));args.append(1)
        for block in range(((b if first else a)+inner-1)//inner):
            pid[0]=block;ns[name](*args)
    got=torch.view_as_complex(out[64:-64].reshape(x.shape))
    ref=torch.fft.fft(torch.view_as_complex(x)) if direction=='forward' else torch.fft.ifft(torch.view_as_complex(x))*n
    assert torch.linalg.vector_norm(got-ref)/torch.linalg.vector_norm(ref)<2e-6
    assert torch.equal(x,saved)
    assert (out[:64]==31415.25).all() and (out[-64:]==31415.25).all()
