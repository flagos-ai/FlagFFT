"""Accuracy and scope of batch-DIF-only FP32 butterfly variants."""
import ast
import math
import pytest

torch=pytest.importorskip('torch')
from flagfft_codegen import metadata
from flagfft_codegen.kernels_common import codelet_radices_for
from flagfft_codegen.npu_dif_codelets import dif_codelet_source


def transform(x,radix):
    source=metadata._module_source('',tuple(sorted(codelet_radices_for((radix,)))))
    source+='\n'+dif_codelet_source((radix,))
    functions=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)]
    for fn in functions:fn.decorator_list=[]
    namespace={}
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions,type_ignores=[])), '<npu-dif-codelet>','exec'),namespace)
    values=namespace[f'_fwd_rad{radix}_b1'](*x.real.unbind(0),*x.imag.unbind(0))
    return torch.complex(torch.stack(values[:radix]),torch.stack(values[radix:]))


@pytest.fixture(autouse=True)
def npu_helpers(monkeypatch):
    monkeypatch.setattr(metadata,'_npu_backend_active',lambda:True)
    monkeypatch.setenv('FLAGFFT_NPU_SYMMETRIC_CODELETS','1')


@pytest.mark.parametrize('radix',[13,15])
@pytest.mark.parametrize('scale',[1e-6,1.,1e6])
@pytest.mark.parametrize('inverse',[False,True])
def test_dif_codelet_patterns(radix,scale,inverse):
    torch.manual_seed(20261105)
    basis=torch.eye(radix,dtype=torch.complex64)
    tones=torch.exp(2j*math.pi*torch.arange(radix)[:,None].double()*torch.arange(radix)[None,:].double()/radix).cfloat()
    random=torch.complex(torch.randn(radix,1033),torch.randn(radix,1033))
    cancellation=torch.ones(radix,1,dtype=torch.complex64)
    cancellation[1::2]=-1
    value=torch.cat((basis,1j*basis,tones,random,cancellation),1)*scale
    saved=value.clone()
    expected=torch.fft.ifft(value.cdouble(),dim=0)*radix if inverse else torch.fft.fft(value.cdouble(),dim=0)
    got=transform(value.conj() if inverse else value,radix)
    if inverse:got=got.conj()
    assert torch.linalg.vector_norm(got-expected)/torch.linalg.vector_norm(expected)<3.5e-7
    assert torch.equal(saved,value)


@pytest.mark.parametrize('radix,depth',[(13,3),(13,5),(15,3),(15,4)])
def test_dif_codelet_accumulated_error(radix,depth):
    torch.manual_seed(20261105)
    shape=(radix,)*depth
    value=torch.complex(torch.randn(shape),torch.randn(shape))
    got=value
    for axis in range(depth):
        got=transform(got.movedim(axis,0),radix).movedim(0,axis)
    expected=torch.fft.fftn(value.cdouble())
    assert torch.linalg.vector_norm(got-expected)/torch.linalg.vector_norm(expected)<5e-7


@pytest.mark.parametrize('direction',['forward','inverse'])
def test_dif_codelet_selection_survives_layout_wrappers(direction):
    from flagfft_codegen.kernels_npu_dif_cube import build_cube_prefix_dif
    args=dict(vector_first17=True,batch_pack=True,vector_second15=True,matrix_first17=True,
              prefix_pitch=32,global_inverse=direction=='inverse')
    old,old_specs=build_cube_prefix_dif((17,15,13,7,8),direction,**args)
    new,new_specs=build_cube_prefix_dif((17,15,13,7,8),direction,codelet_reuse=True,**args)
    assert old_specs==new_specs
    assert '_npu_dif_rad3' in new and '_npu_dif_rad3' not in old
    assert 'dc_r' in new and 'dc_r' not in old


def test_dif_codelet_requires_batch_grouping():
    from flagfft_codegen.kernels_npu_dif import emit_vector_dif
    with pytest.raises(ValueError,match='batch-grouped'):
        emit_vector_dif((17,15,13,7,8),'forward',0,'unused',codelet_reuse=True)
