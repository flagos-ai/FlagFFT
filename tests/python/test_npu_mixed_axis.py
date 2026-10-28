"""Independent FFT oracle for packed mixed axes and their masked matrix edges."""
import math
import pytest

torch=pytest.importorskip('torch')
from flagfft_codegen import target
from flagfft_codegen.kernels_npu_mixed_axis import build_mixed_axis, padded
from test_npu_structured_fft import Pointer, execute_module


def root_table(factors):
    pads=tuple(map(padded,factors));tables=[]
    for stage,radix in enumerate(factors[:-1]):
        later=padded(math.prod(factors[1:])) if stage==0 else math.prod(pads[stage+1:])
        j=torch.arange(later,dtype=torch.int64)
        actual=j if stage==0 else sum(((j//math.prod(pads[k+1:]))%pads[k])*math.prod(factors[k+1:]) for k in range(stage+1,len(factors)))
        phase=-2*math.pi*torch.arange(radix,dtype=torch.float64)[:,None]*actual[None,:]/math.prod(factors[stage:])
        tables.extend((phase.cos().float().flatten(),phase.sin().float().flatten()))
    return torch.cat(tables)


@pytest.mark.parametrize('factors',[(19,11),(17,13),(13,11),(19,17),(28,15),
    (17,13,2),(27,20),(15,7,6),(17,13,3),(25,10,4)])
@pytest.mark.parametrize('first',[False,True])
@pytest.mark.parametrize('direction',['forward','inverse'])
def test_packed_mixed_axis(tmp_path,monkeypatch,factors,first,direction):
    old=target._target.get();target.set_codegen_target('npu:Ascend910B4:1')
    monkeypatch.setenv('FLAGFFT_NPU_VECTOR_REMAINDER','1')
    try:name,source,_,pack=build_mixed_axis(factors,17,first,direction)
    finally:target.set_codegen_target(old)
    path=tmp_path/'axis.py';path.write_text(source);ns,pid=execute_module(path)
    length=math.prod(factors);batch=2;width=17
    torch.manual_seed(20261029);x=torch.randn(batch,length,width,2);saved=x.clone()
    phase=(1 if direction=='inverse' else -1)*2*math.pi*torch.arange(length,dtype=torch.float64)[:,None]*torch.arange(width)[None,:]/(length*width)
    cross=torch.cat((phase.cos().float().flatten(),phase.sin().float().flatten()))
    z=torch.view_as_complex(x)
    ref=torch.fft.fft(z,dim=1) if direction=='forward' else torch.fft.ifft(z,dim=1)*length
    if first:ref=(ref*torch.complex(phase.cos().float(),phase.sin().float())).transpose(1,2).contiguous()
    out=torch.full((x.numel()+128,),31415.25)
    for b in range(batch):
        pid[1]=b
        for g in range((width+pack-1)//pack):
            pid[0]=g
            args=[Pointer(x)]
            if first:args.append(Pointer(cross))
            ns[name](*args,Pointer(out,64),Pointer(root_table(factors)),batch)
    got=torch.view_as_complex(out[64:-64].reshape(ref.shape+(2,)))
    assert torch.isfinite(got).all()
    assert torch.linalg.vector_norm(got-ref)/torch.linalg.vector_norm(ref)<1e-6
    assert torch.equal(x,saved)
    assert (out[:64]==31415.25).all() and (out[-64:]==31415.25).all()
