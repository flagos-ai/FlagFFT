"""Independent FFT oracle for the shared grouped/strided/real Triton primitive."""
import pytest

torch=pytest.importorskip('torch')
from flagfft_codegen.kernels_npu_grouped import emit_npu_grouped_fft64
from test_npu_structured_fft import Pointer, roots, execute_module


def layout(x,stride):
    if stride==1:
        return x.contiguous()
    return x.reshape(x.shape[0]//stride,stride,x.shape[1]).transpose(1,2).contiguous()


@pytest.mark.parametrize('n,group',[(16,1),(16,8),(32,1),(32,4),(256,1),(256,4),(256,8)])
@pytest.mark.parametrize('mode,direction',[(0,'forward'),(0,'inverse'),(1,'forward'),(2,'inverse')])
def test_contiguous_real_and_complex(tmp_path,n,group,mode,direction):
    check(tmp_path,n,group,mode,direction,1,1,2*group+1)


@pytest.mark.parametrize('n,group',[(16,1),(16,8),(32,4),(256,8)])
@pytest.mark.parametrize('strides',[(16,16),(16,1),(1,16),(1,0),(17,17),(9,9)])
@pytest.mark.parametrize('direction',['forward','inverse'])
def test_strided_and_transposed_with_launch_offsets(tmp_path,n,group,strides,direction):
    check(tmp_path,n,group,0,direction,*strides,2*strides[0] if strides[0]>1 else 32)


def check(tmp_path,n,group,mode,direction,input_stride,output_stride,batch):
    meta=emit_npu_grouped_fft64(group=group,stride=input_stride,mode=mode,direction=direction,
        out_dir=tmp_path,length=n,structured=True,output_stride=output_stride)
    ns,pid=execute_module(meta['module_path'])
    torch.manual_seed(20261028)
    z=torch.complex(torch.randn(batch,n),torch.randn(batch,n))
    if mode==1:
        src=z.real.contiguous();ref=torch.fft.rfft(src)
    elif mode==2:
        src=torch.fft.rfft(z.real).contiguous();ref=torch.fft.irfft(src,n=n)*n
    else:
        src=layout(z,input_stride)
        ref=torch.fft.fft(z) if direction=='forward' else torch.fft.ifft(z)*n
    src=src if not src.is_complex() else torch.view_as_real(src)
    saved=src.clone()
    out_stride=output_stride if output_stride else batch
    ref=layout(ref,out_stride) if mode==0 else ref
    expected=ref if not ref.is_complex() else torch.view_as_real(ref)
    output=torch.full((expected.numel()+128,),31415.25)
    for offset in range(0,batch,group):
        pid[0]=0
        ns[meta['kernel_name']](Pointer(src),Pointer(output,64),Pointer(torch.zeros(1,dtype=torch.int32)),Pointer(roots(n,group)),batch,out_stride,offset)
    assert torch.allclose(output[64:-64].reshape(expected.shape),expected,atol=3e-4,rtol=3e-5)
    assert torch.equal(src,saved)
    assert (output[:64]==31415.25).all() and (output[-64:]==31415.25).all()
