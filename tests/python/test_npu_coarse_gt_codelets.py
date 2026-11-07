"""Check factored FP32 rotations on cancellation inputs and repeated FFTs."""
import pytest

torch=pytest.importorskip('torch')
from flagfft_codegen.npu_coarse_gt import radix8_source
from test_npu_structured_fft import execute_module


@pytest.mark.parametrize('order',['0','1'])
@pytest.mark.parametrize('scale',[1e-6,1.,1e6])
@pytest.mark.parametrize('kind',['random','impulse','dc','alternating','tone'])
@pytest.mark.parametrize('depth',[1,4])
def test_factored_radix8(tmp_path,scale,kind,depth,order,monkeypatch):
    monkeypatch.setenv("FLAGFFT_NPU_RADIX8_ORDER",order)
    path=tmp_path/'radix8.py';path.write_text(radix8_source())
    ns,_=execute_module(path)
    torch.manual_seed(20261107)
    x=torch.randn(8,97,2)
    if kind=='impulse':x.zero_();x[3,:,0]=1.
    elif kind=='dc':x.fill_(1.)
    elif kind=='alternating':
        x.zero_();x[:,:,0]=((-1.)**torch.arange(8))[:,None]
    elif kind=='tone':
        angle=2*torch.pi*torch.arange(8,dtype=torch.float64)[:,None]*torch.arange(1,98,dtype=torch.float64)[None,:]/8
        x=torch.stack((angle.cos(),angle.sin()),dim=-1).float()
    x*=scale
    real,imag=list(x[:,:,0].unbind(0)),list(x[:,:,1].unbind(0))
    reference=torch.view_as_complex(x).cdouble()
    for _ in range(depth):
        result=ns['_fwd_rad8_b1'](*real,*imag)
        real,imag=result[:8],result[8:]
        reference=torch.fft.fft(reference,dim=0)
    got=torch.complex(torch.stack(real),torch.stack(imag))
    assert torch.isfinite(got).all()
    assert torch.linalg.vector_norm(got-reference)/torch.linalg.vector_norm(reference)<5e-7
