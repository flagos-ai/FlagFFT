"""Independent tiny FFT oracles with partial batch tiles and output guards."""
import math
import pytest

torch = pytest.importorskip('torch')
from flagfft_codegen.kernels_npu_direct_prefetch import (
    build_prefetched_direct_dft,
    build_single_direct_dft,
)
from test_npu_structured_fft import Pointer, execute_module


@pytest.mark.parametrize("n", [17, 23, 31])
@pytest.mark.parametrize("direction", ["forward", "inverse"])
def test_single_direct_dft(tmp_path, n, direction):
    name, source, _ = build_single_direct_dft(n, direction)
    path = tmp_path / "single_dft.py"
    path.write_text(source)
    namespace, _ = execute_module(path)
    namespace["tl"].sum = lambda value, axis: torch.sum(value, dim=axis)
    k = torch.arange(n, dtype=torch.float64)
    sign = -1 if direction == "forward" else 1
    angle = sign * 2 * math.pi * k[:, None] * k[None, :] / n
    real = torch.zeros((32, 32))
    imag = torch.zeros_like(real)
    real[:n, :n] = torch.cos(angle)
    imag[:n, :n] = torch.sin(angle)
    x = torch.randn((n, 2), generator=torch.Generator().manual_seed(23))
    saved = x.clone()
    guarded = torch.full((2 * n + 64,), 31415.25)
    namespace[name](Pointer(x), Pointer(guarded, 32), Pointer(real), Pointer(imag), 1)
    z = torch.view_as_complex(x).to(torch.complex128)
    expected = torch.fft.fft(z) if direction == "forward" else torch.fft.ifft(z) * n
    got = torch.view_as_complex(guarded[32:-32].reshape(n, 2))
    assert torch.linalg.vector_norm(got - expected) / torch.linalg.vector_norm(expected) < 5e-7
    assert torch.equal(x, saved)
    assert (guarded[:32] == 31415.25).all() and (guarded[-32:] == 31415.25).all()


@pytest.mark.parametrize('pair', [False, True])
@pytest.mark.parametrize('n', [16, 17, 23, 31, 32])
@pytest.mark.parametrize('batch', [1, 19, 64])
@pytest.mark.parametrize('mode,direction', [('c2c','forward'), ('c2c','inverse'),
                                           ('r2c','forward'), ('c2r','inverse')])
def test_prefetched_direct_fft(tmp_path, n, batch, mode, direction, pair):
    from flagfft_codegen.metadata import _module_source
    name, source, _ = build_prefetched_direct_dft(n, mode, direction, 8, pair=pair)
    path = tmp_path / 'prefetch.py'
    path.write_text(_module_source(source))
    ns, pid = execute_module(path)
    angle = (1 if direction == 'inverse' else -1) * 2 * math.pi * torch.arange(n, dtype=torch.float64)[:,None] * torch.arange(n)[None,:] / n
    wr, wi = angle.cos().float(), angle.sin().float()
    torch.manual_seed(20261029)
    if mode == 'r2c':
        x = torch.randn(batch, n)
        expected = torch.fft.rfft(x)
    elif mode == 'c2r':
        x = torch.randn(batch, n//2+1, 2)
        expected = torch.fft.irfft(torch.view_as_complex(x), n=n) * n
    else:
        x = torch.randn(batch, n, 2)
        z = torch.view_as_complex(x)
        expected = torch.fft.ifft(z) * n if direction == 'inverse' else torch.fft.fft(z)
    saved = x.clone()
    shape = expected.shape if mode == 'c2r' else expected.shape + (2,)
    out = torch.full((math.prod(shape)+128,), 31415.25)
    for block in range((batch+7)//8):
        pid[0] = block
        ns[name](Pointer(x), Pointer(out,64), Pointer(wr), Pointer(wi), batch)
    got = out[64:-64].reshape(shape)
    if mode != 'c2r': got = torch.view_as_complex(got)
    if mode == 'r2c':
        assert (got[:,0].imag == 0).all()
        if n%2 == 0: assert (got[:,-1].imag == 0).all()
    assert torch.isfinite(got).all()
    assert torch.linalg.vector_norm(got-expected)/torch.linalg.vector_norm(expected) < 5e-7
    assert torch.equal(saved, x)
    assert (out[:64] == 31415.25).all() and (out[-64:] == 31415.25).all()
