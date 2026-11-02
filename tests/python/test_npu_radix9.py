"""Check the nine-point DFT against an independent high precision FFT."""
import pytest

torch=pytest.importorskip('torch')
from flagfft_codegen.npu_radix9 import radix9_source


def transform(x):
    ns={}
    exec(radix9_source().replace('@triton.jit\n',''),ns)
    parts=ns['_fwd_rad9_b1'](*x.real.unbind(0),*x.imag.unbind(0))
    return torch.complex(torch.stack(parts[:9]),torch.stack(parts[9:]))


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('scale',[1e-6,1.,1e6])
def test_radix9_vectors(dtype,scale):
    torch.manual_seed(20261103)
    basis=torch.eye(9,dtype=dtype)
    tones=torch.exp(2j*torch.pi*torch.arange(9,dtype=dtype)[:,None]*torch.arange(9)[None,:]/9)
    random=torch.complex(torch.randn(9,1033,dtype=dtype),torch.randn(9,1033,dtype=dtype))
    x=torch.cat((basis,1j*basis,torch.ones(9,1),tones,random),1)*scale
    saved=x.clone();expected=torch.fft.fft(x.to(torch.complex128),dim=0)
    got=transform(x)
    tolerance=3.5e-7 if dtype==torch.float32 else 1e-14
    assert torch.linalg.vector_norm(got-expected)/torch.linalg.vector_norm(expected)<tolerance
    inverse=transform(got.conj()).conj()
    assert torch.linalg.vector_norm(inverse-9*x)/torch.linalg.vector_norm(9*x)<2*tolerance
    assert torch.equal(x,saved)


@pytest.mark.parametrize('depth',[2,4])
def test_radix9_accumulated_error(depth):
    torch.manual_seed(20261103)
    shape=(9,)*depth;x=torch.complex(torch.randn(shape),torch.randn(shape));got=x
    for axis in range(depth):got=transform(got.movedim(axis,0)).movedim(0,axis)
    expected=torch.fft.fftn(x.to(torch.complex128))
    assert torch.linalg.vector_norm(got-expected)/torch.linalg.vector_norm(expected)<5e-7
