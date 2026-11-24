"""Check the fixed convolutions against double FFTs and accumulated FP32 error."""
import pytest

torch = pytest.importorskip('torch')
from flagfft_codegen.npu_radix17 import radix17_source


def transform(x):
    namespace = {}
    exec(radix17_source().replace('@triton.jit\n', ''), namespace)
    parts = namespace['_fwd_rad17_b1'](*x.real.unbind(0), *x.imag.unbind(0))
    return torch.complex(torch.stack(parts[:17]), torch.stack(parts[17:]))


@pytest.mark.parametrize('scale', [1e-6, 1., 1e6])
def test_basis_tones_scales_and_roundtrip(scale):
    torch.manual_seed(20261101)
    basis = torch.eye(17, dtype=torch.complex64)
    positions = torch.arange(17, dtype=torch.float64)
    tones = torch.exp(2j*torch.pi*positions[:, None]*positions[None, :]/17).to(torch.complex64)
    random = torch.complex(torch.randn(17, 1033), torch.randn(17, 1033))
    x = torch.cat((basis, 1j*basis, torch.ones(17, 1), tones, random), 1)*scale
    expected = torch.fft.fft(x.to(torch.complex128), dim=0)
    got = transform(x)
    assert torch.linalg.vector_norm(got-expected)/torch.linalg.vector_norm(expected) < 3.5e-7
    recovered = transform(got.conj()).conj()
    assert torch.linalg.vector_norm(recovered-17*x)/torch.linalg.vector_norm(17*x) < 5e-7


@pytest.mark.parametrize('depth', [3, 4, 5])
def test_accumulated_error(depth):
    torch.manual_seed(20261101+depth)
    x = torch.complex(torch.randn((17,)*depth), torch.randn((17,)*depth))
    got = x
    for axis in range(depth):
        got = transform(got.movedim(axis, 0)).movedim(0, axis)
    expected = torch.fft.fftn(x.to(torch.complex128))
    assert torch.linalg.vector_norm(got-expected)/torch.linalg.vector_norm(expected) < 5e-7
