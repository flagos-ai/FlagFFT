"""Whole multidimensional FFT oracle and compact real boundary checks."""
import pytest

torch=pytest.importorskip('torch')
from flagfft_codegen.kernels_npu_tensor import build_npu_tensor_source,tensor_fft_layout
from flagfft_codegen.metadata import _module_source
from test_npu_structured_fft import Pointer,roots,execute_module


def tensor_roots(shape):
    n,_=tensor_fft_layout(shape)
    return torch.cat([roots(length,n//length) for length in shape])


@pytest.mark.parametrize('shape',[(8,16),(16,32),(64,64),(8,8,16),(16,16,16),(8,16,32)])
@pytest.mark.parametrize('mode,direction',[(0,'forward'),(0,'inverse'),(1,'forward'),(2,'inverse')])
def test_fused_tensor(tmp_path,shape,mode,direction):
    name,body=build_npu_tensor_source(shape,mode,direction)
    path=tmp_path/'tensor.py';path.write_text(_module_source(body))
    ns,pid=execute_module(path);torch.manual_seed(20261028)
    batch=2;axes=tuple(range(1,len(shape)+1));x=torch.randn((batch,)+shape)
    if mode==0:
        x=torch.complex(x,torch.randn_like(x))
        ref=torch.fft.fftn(x,dim=axes) if direction=='forward' else torch.fft.ifftn(x,dim=axes)*x[0].numel()
        source=torch.view_as_real(x);expected=torch.view_as_real(ref)
    elif mode==1:
        source=x;expected=torch.view_as_real(torch.fft.rfftn(x,dim=axes))
    else:
        source=torch.view_as_real(torch.fft.rfftn(x,dim=axes))
        expected=torch.fft.irfftn(torch.view_as_complex(source),s=shape,dim=axes)*x[0].numel()
    saved=source.clone();out=torch.full((expected.numel()+128,),31415.25);table=tensor_roots(shape)
    for offset in range(batch):
        pid[0]=0
        ns[name](Pointer(source),Pointer(out,64),Pointer(torch.zeros(1,dtype=torch.int32)),Pointer(table),batch,1,offset)
    got=out[64:-64].reshape(expected.shape)
    assert torch.isfinite(got).all()
    assert torch.linalg.vector_norm(got-expected)/torch.linalg.vector_norm(expected)<5e-7
    assert torch.equal(source,saved)
    assert (out[:64]==31415.25).all() and (out[-64:]==31415.25).all()
