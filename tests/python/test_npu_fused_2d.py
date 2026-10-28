"""A pair of existing FFT primitives with transposed stores must equal FFT2."""
import pytest

torch=pytest.importorskip('torch')
from flagfft_codegen.kernels_npu_grouped import emit_npu_grouped_fft64
from test_npu_structured_fft import execute_module,Pointer,axis_roots


@pytest.mark.parametrize('pack',[8,16])
@pytest.mark.parametrize('shape',[(64,128),(128,128),(128,256),(32,64),(16,2048)])
@pytest.mark.parametrize('batch',[1,4])
@pytest.mark.parametrize('direction',['forward','inverse'])
def test_two_transposed_fft_stores(tmp_path,shape,batch,direction,pack):
    a,b=shape;torch.manual_seed(20261029);x=torch.randn(batch,a,b,2);saved=x.clone()
    current=x
    for stage,(n,stride) in enumerate(((b,a),(a,b))):
        group=min(pack,4096//n)
        meta=emit_npu_grouped_fft64(group=group,stride=1,mode=0,direction=direction,
            out_dir=tmp_path/str(stage),length=n,structured=True,output_stride=stride,recursive=True)
        ns,pid=execute_module(meta['module_path']);count=batch*stride
        output=torch.full((x.numel()+128,),31415.25)
        for offset in range(0,count,group):
            pid[0]=0
            ns[meta['kernel_name']](Pointer(current),Pointer(output,64),Pointer(torch.zeros(1,dtype=torch.int32)),Pointer(axis_roots(n,group,True)),count,stride,offset)
        assert (output[:64]==31415.25).all() and (output[-64:]==31415.25).all()
        current=output[64:-64].clone()
    got=torch.view_as_complex(current.reshape(batch,a,b,2));z=torch.view_as_complex(x)
    ref=torch.fft.fft2(z) if direction=='forward' else torch.fft.ifft2(z)*a*b
    assert torch.isfinite(got).all()
    assert torch.linalg.vector_norm(got-ref)/torch.linalg.vector_norm(ref)<1e-6
    assert torch.equal(x,saved)
