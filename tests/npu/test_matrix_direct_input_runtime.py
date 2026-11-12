"""Native ordinary 2D dot input and padded reflected phase-table row."""
import importlib.util
import math
import sys
import pytest

torch=pytest.importorskip('torch')
pytest.importorskip('torch_npu')
from flagfft_codegen import target
from flagfft_codegen.kernels_npu_matrix_direct_input import build_direct_first

@pytest.mark.parametrize('a,b',[(128,129),(256,129),(129,256),(209,221)])
@pytest.mark.parametrize('direction',['forward','inverse'])
@pytest.mark.parametrize('combined',[False,True])
def test_direct_first_axis(tmp_path,a,b,combined,direction):
    torch.npu.set_device(0);torch.manual_seed(20261105);target.set_codegen_target('npu:Ascend910B4:1')
    pa=1<<(a-1).bit_length();pb=1<<(b-1).bit_length();h=a//2+1;hp=1<<(h-1).bit_length();batch=3
    sign=-1 if direction=='forward' else 1
    angle=sign*2*math.pi*torch.arange(h,dtype=torch.float64)[:,None]*torch.arange(a)[None,:]/a
    w=torch.zeros(2*hp,pa);w[:h,:a]=angle.cos().float();w[hp:hp+h,:a]=angle.sin().float()
    if combined:
        packed=torch.zeros_like(w)
        for tile in range(hp//32):
            packed[tile*64:tile*64+32]=w[tile*32:tile*32+32]
            packed[tile*64+32:tile*64+64]=w[hp+tile*32:hp+tile*32+32]
        w=packed
    angle=sign*2*math.pi*torch.arange(a,dtype=torch.float64)[:,None]*torch.arange(b)[None,:]/(a*b)
    tr=torch.zeros(pa+1,pb);ti=torch.zeros_like(tr);tr[:a,:b]=angle.cos().float();ti[:a,:b]=angle.sin().float()
    tw=torch.cat((tr.flatten(),ti.flatten())).npu();w=w.npu()
    x=torch.randn(batch,a,b,2);z=torch.view_as_complex(x).to(torch.complex128)
    reference=(torch.fft.fft(z,dim=1) if direction=='forward' else torch.fft.ifft(z,dim=1)*a)*torch.polar(torch.ones_like(angle),angle)[None]
    source=x.npu();guard=torch.full((batch*a*2*pb+128,),31415.25,device='npu');out=guard[64:-64].reshape(batch,a,2*pb)
    path=tmp_path/'direct_axis.py';path.write_text(build_direct_first(a,b,combined=combined))
    spec=importlib.util.spec_from_file_location('direct_axis_'+str(a)+str(b)+direction,path);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    def run():
        fn=getattr(module,'cube0_complex_direct_combined' if combined else 'cube0_complex_direct_pack')
        k=fn[(math.ceil(h/32),pb//32,batch)](source,w,tw,out,num_warps=4,num_stages=1,multibuffer=False,enable_auto_blockify=True,enable_fp_fusion=False)
        assert not getattr(k.metadata,'uses_computed_dot_transfer',False)
    run();graph=torch.npu.NPUGraph()
    with torch.npu.graph(graph):run()
    for scale in (1.,1e-6,1e6):
        source.copy_((x*scale).npu());out.fill_(float('nan'));graph.replay();torch.npu.synchronize();value=out.cpu()
        got=torch.complex(value[:,:,:b],value[:,:,pb:pb+b]);ref=reference*scale
        assert torch.isfinite(value).all()
        assert torch.linalg.vector_norm(got-ref)/torch.linalg.vector_norm(ref)<5e-7
        assert (value[:,:,b:pb]==0).all() and (value[:,:,pb+b:]==0).all()
        assert (guard[:64].cpu()==31415.25).all() and (guard[-64:].cpu()==31415.25).all()
        assert torch.equal(source.cpu(),x*scale)
