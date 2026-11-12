"""Check fixed convolutions, cancellation inputs and accumulated FP32 error."""
import numpy as np
import pytest

torch=pytest.importorskip('torch')
from flagfft_codegen.npu_codelets import codelet_source


def transform(x,radix):
    ns={}
    exec(codelet_source(radix).replace('@triton.jit\n',''),ns)
    parts=ns[f'_fwd_rad{radix}_b1'](*x.real.unbind(0),*x.imag.unbind(0))
    return torch.complex(torch.stack(parts[:radix]),torch.stack(parts[radix:]))


@pytest.mark.parametrize('radix',[5,13])
@pytest.mark.parametrize('scale',[1e-6,1.,1e6])
def test_symmetric_codelet_vectors(radix,scale):
    torch.manual_seed(20261101)
    basis=torch.eye(radix,dtype=torch.complex64)
    tones=torch.exp(2j*torch.pi*torch.arange(radix)[:,None]*torch.arange(radix)[None,:]/radix)
    random=torch.complex(torch.randn(radix,1033),torch.randn(radix,1033))
    x=torch.cat((basis,1j*basis,torch.ones(radix,1),tones,random),1)*scale
    saved=x.clone()
    expected=torch.fft.fft(x.to(torch.complex128),dim=0)
    got=transform(x,radix)
    assert torch.linalg.vector_norm(got-expected)/torch.linalg.vector_norm(expected)<3.5e-7
    inverse=transform(got.conj(),radix).conj()
    assert torch.linalg.vector_norm(inverse-radix*x)/torch.linalg.vector_norm(radix*x)<5e-7
    assert torch.equal(x,saved)


@pytest.mark.parametrize('radix,depth',[(5,3),(5,8),(13,3),(13,5)])
def test_repeated_symmetric_axes(radix,depth):
    torch.manual_seed(20261101)
    shape=(radix,)*depth
    x=torch.complex(torch.randn(shape),torch.randn(shape))
    got=x
    for axis in range(depth):
        got=transform(got.movedim(axis,0),radix).movedim(0,axis)
    # NumPy supports the rank-eight case that some Torch/MKL builds reject.
    expected=torch.from_numpy(np.fft.fftn(x.numpy().astype(np.complex128)))
    assert torch.linalg.vector_norm(got-expected)/torch.linalg.vector_norm(expected)<5e-7


def test_optimized_helpers_are_npu_scoped(monkeypatch):
    from flagfft_codegen import metadata
    monkeypatch.setattr(metadata,'_npu_backend_active',lambda:True)
    monkeypatch.setenv('FLAGFFT_NPU_SYMMETRIC_CODELETS','1')
    optimized=metadata._module_source('',(5,13))
    monkeypatch.setattr(metadata,'_npu_backend_active',lambda:False)
    ordinary=metadata._module_source('',(5,13))
    assert '_npu_cos6' in optimized and '_npu_cos6' not in ordinary
    monkeypatch.setattr(metadata,'_npu_backend_active',lambda:True)
    monkeypatch.setenv('FLAGFFT_NPU_SYMMETRIC_CODELETS','0')
    assert metadata._module_source('',(5,13)).rstrip()==ordinary.rstrip()
