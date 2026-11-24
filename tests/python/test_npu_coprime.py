"""Independent DFT checks for the coprime register permutations."""
from pathlib import Path
import pytest

torch=pytest.importorskip('torch')
from flagfft_codegen import kernels_common
from flagfft_codegen.npu_codelets import codelet_source
from flagfft_codegen.npu_coprime import coprime_source


def transform(x,a,b):
    codelets=Path(kernels_common.__file__).parent/'codelet'
    source='\n'.join((codelets/f'radix{r}.py').read_text() for r in (2,3))
    source+=codelet_source(5)+coprime_source(a,b)
    ns={};exec(source.replace('@triton.jit\n',''),ns)
    parts=ns[f'_fwd_rad{a*b}_b1'](*x.real.unbind(0),*x.imag.unbind(0))
    return torch.complex(torch.stack(parts[:a*b]),torch.stack(parts[a*b:]))


@pytest.mark.parametrize('a,b',[(2,5),(3,5)])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('scale',[1e-6,1.,1e6])
def test_coprime_vectors(a,b,dtype,scale):
    n=a*b;torch.manual_seed(20261103)
    basis=torch.eye(n,dtype=dtype)
    tones=torch.exp(2j*torch.pi*torch.arange(n,dtype=dtype)[:,None]*torch.arange(n)[None,:]/n)
    random=torch.complex(torch.randn(n,1033,dtype=dtype),torch.randn(n,1033,dtype=dtype))
    x=torch.cat((basis,1j*basis,torch.ones(n,1),tones,random),1)*scale
    saved=x.clone();expected=torch.fft.fft(x.to(torch.complex128),dim=0)
    got=transform(x,a,b);tolerance=3.5e-7 if dtype==torch.float32 else 1e-14
    assert torch.linalg.vector_norm(got-expected)/torch.linalg.vector_norm(expected)<tolerance
    inverse=transform(got.conj(),a,b).conj()
    assert torch.linalg.vector_norm(inverse-n*x)/torch.linalg.vector_norm(n*x)<2*tolerance
    assert torch.equal(saved,x)
