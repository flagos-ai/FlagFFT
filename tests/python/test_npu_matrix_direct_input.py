"""Direct complex matrix input, aligned phase rows and reflected edge bounds."""
import math
import pytest

torch=pytest.importorskip('torch')
from flagfft_codegen.kernels_npu_matrix_direct_input import build_direct_first,emit_direct_first
from flagfft_codegen.kernels_npu_matrix_complex_half import build_complex_half
from test_npu_structured_fft import Pointer,execute_module

@pytest.mark.parametrize('a,b',[(128,129),(129,256),(256,129),(209,221),(221,209)])
@pytest.mark.parametrize('direction',['forward','inverse'])
@pytest.mark.parametrize('scale',[1.,1e-6,1e6])
def test_direct_input_whole(tmp_path,a,b,direction,scale):
    pa=1<<(a-1).bit_length();pb=1<<(b-1).bit_length();ha=a//2+1;hb=b//2+1
    hpa=1<<(ha-1).bit_length();hpb=1<<(hb-1).bit_length();batch=2;n=a*b
    path=tmp_path/'direct.py';path.write_text(build_direct_first(a,b)+build_complex_half(a,b))
    ns,pid=execute_module(path);pid.append(0)
    # Correctly rounded FP32 dot model checks layout and reconstruction. Actual
    # NPU IEEE arithmetic is independently checked by the native C API tests.
    ns['tl'].dot=lambda x,w,input_precision:(x.double()@w.double()).float()
    sign=-1 if direction=='forward' else 1
    first_angle=sign*2*math.pi*torch.arange(ha,dtype=torch.float64)[:,None]*torch.arange(a)[None,:]/a
    dw=torch.zeros(2*hpa,pa);dw[:ha,:a]=first_angle.cos().float();dw[hpa:hpa+ha,:a]=first_angle.sin().float()
    last_angle=sign*2*math.pi*torch.arange(b,dtype=torch.float64)[:,None]*torch.arange(hb)[None,:]/b
    hw=torch.zeros(pb,2*hpb);hw[:b,:2*hb:2]=last_angle.cos().float();hw[:b,1:2*hb:2]=last_angle.sin().float()
    angle=sign*2*math.pi*torch.arange(a,dtype=torch.float64)[:,None]*torch.arange(b)[None,:]/n
    tr=torch.zeros(pa+1,pb);ti=torch.zeros_like(tr);tr[:a,:b]=angle.cos().float();ti[:a,:b]=angle.sin().float();tw=torch.cat((tr.flatten(),ti.flatten()))
    torch.manual_seed(20261105);x=torch.randn(batch,n,2)*scale;saved=x.clone()
    q=torch.full((batch,a,2*pb),float('nan'));guard=torch.full((batch*n*2+128,),31415.25)
    steps=[('cube0_complex_direct_pack',(math.ceil(ha/32),pb//32,batch),(Pointer(x),Pointer(dw),Pointer(tw),Pointer(q))),
           ('cube1_complex_half_finish',(math.ceil(a/64),math.ceil(hb/32),batch),(Pointer(q),Pointer(hw),Pointer(guard,64)))]
    for name,grid,args in steps:
        for ix in range(grid[0]):
            for iy in range(grid[1]):
                for iz in range(grid[2]):pid[:]=[ix,iy,iz];ns[name](*args)
    got=torch.view_as_complex(guard[64:-64].reshape(batch,n,2));z=torch.view_as_complex(x).to(torch.complex128)
    reference=torch.fft.fft(z) if direction=='forward' else torch.fft.ifft(z)*n
    assert torch.isfinite(got).all()
    assert torch.linalg.vector_norm(got-reference)/torch.linalg.vector_norm(reference)<5e-7
    assert torch.equal(x,saved)
    assert (q[:,:,b:pb]==0).all() and (q[:,:,pb+b:]==0).all()
    assert (guard[:64]==31415.25).all() and (guard[-64:]==31415.25).all()

def test_direct_input_metadata(tmp_path):
    meta=emit_direct_first(209,221,'cube0_complex_direct_pack','forward',tmp_path)
    assert meta['arg_names']==['X','W','T','Y']
    assert meta['signature']==','.join(['*fp32:16']*4)
    with pytest.raises(ValueError):build_direct_first(127,221)
    with pytest.raises(ValueError):emit_direct_first(209,221,'pack0','forward',tmp_path)
