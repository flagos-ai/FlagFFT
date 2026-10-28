"""Independent FFT oracle for register-only prime convolution codelets."""
import pytest

torch=pytest.importorskip('torch')
from flagfft_codegen import target
from flagfft_codegen.metadata import _module_source
from test_npu_structured_fft import execute_module


@pytest.mark.parametrize('prime',[17,19])
@pytest.mark.parametrize('scale',[1e-6,1.,1e6])
@pytest.mark.parametrize('inverse',[False,True])
def test_rader_prime(tmp_path,monkeypatch,prime,scale,inverse):
    old=target._target.get();target.set_codegen_target('npu:Ascend910B4:1')
    monkeypatch.setenv('FLAGFFT_NPU_RADER_CODELETS','1')
    try:source=_module_source('',(prime,))
    finally:target.set_codegen_target(old)
    path=tmp_path/'prime.py';path.write_text(source);ns,_=execute_module(path)
    torch.manual_seed(20261029);x=torch.complex(torch.randn(64,prime),torch.randn(64,prime))*scale
    z=x.conj() if inverse else x
    result=ns[f'_fwd_rad{prime}_rader'](*[z.real[:,k] for k in range(prime)],*[z.imag[:,k] for k in range(prime)])
    got=torch.complex(torch.stack(result[:prime],1),torch.stack(result[prime:],1))
    if inverse:got=got.conj()
    ref=torch.fft.ifft(x)*prime if inverse else torch.fft.fft(x)
    assert torch.isfinite(got).all()
    assert torch.linalg.vector_norm(got-ref)/torch.linalg.vector_norm(ref)<5e-7
