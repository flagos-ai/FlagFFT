"""Full FFT checks for the compact table ABI and flattened logical tiles."""
import math
import pytest
torch=pytest.importorskip('torch')
from flagfft_codegen.kernels_npu_power_dif import build
from test_npu_structured_fft import Pointer,execute_module,roots


def test_three_axis_emitter(tmp_path):
 from flagfft_codegen.kernels_npu_dif import emit_vector_dif
 for phase in range(3):
  metadata=emit_vector_dif((64,64,128),'forward',phase,tmp_path,flat_power2=True)
  assert metadata['kernel_name']==f'axis_{phase}'
 with pytest.raises(ValueError,match='Unsupported DIF factors'):
  emit_vector_dif((64,64,128),'forward',0,tmp_path)

def test_terminal_emitter(tmp_path):
 from flagfft_codegen.kernels_npu_dif import emit_vector_dif
 metadata=emit_vector_dif((32,32,32,16),'forward',0,tmp_path,
                          flat_power2=True,terminal8=True)
 from pathlib import Path
 text=Path(metadata['module_path']).read_text()
 assert 'power_fft_axis(r,i,W,4096,32,128,5)' in text
 assert metadata['signature']=='*fp32:16,*fp32:16,*fp32:16'
 with pytest.raises(ValueError,match='require flat power DIF'):
  emit_vector_dif((16,16,16),'forward',0,tmp_path,terminal8=True)

@pytest.mark.parametrize('factors',[(64,64),(128,64),(128,128),(16,8,8),(16,16,16),(32,16,16),(16,32,16),(32,32,16,16),(32,32,32,16),(64,64,64),(64,64,128)])
@pytest.mark.parametrize('direction',['forward','inverse'])
@pytest.mark.parametrize('terminal8',[False,True])
def test_power_dif_complete(tmp_path,factors,direction,terminal8):
 source,specs=build(factors,direction,terminal8=terminal8);path=tmp_path/'power.py';path.write_text(source);ns,pid=execute_module(path)
 n,batch=math.prod(factors),2;sign=-1 if direction=='forward' else 1
 torch.manual_seed(20261030);x=torch.randn(batch,n,2);saved=x.clone();current=x
 guard=torch.full((x.numel()+128,),31415.25);scratch=[torch.empty_like(x),torch.empty_like(x)]
 for k,s in enumerate(specs):
  w=roots(s['r'],s['c'])
  if k:
   digit=torch.arange(s['prefix'])//s['previous_prefix'];sample=torch.arange(s['r']*s['m'])
   angle=sign*2*math.pi*sample[:,None].double()*digit[None,:]/(factors[k-1]*s['r']*s['m'])
   w=torch.cat((angle.cos().float().flatten(),angle.sin().float().flatten(),w))
  final=k+1==len(specs);out=Pointer(guard,64) if final else Pointer(scratch[k%2])
  for gx in range(s['grid'][0]):
   for b in range(batch):
    pid[:]=[gx,b];ns[s['name']](Pointer(current),Pointer(w),out)
  current=guard[64:-64].reshape_as(x) if final else scratch[k%2]
 got=torch.view_as_complex(current);z=torch.view_as_complex(x);ref=torch.fft.fft(z) if sign<0 else torch.fft.ifft(z)*n
 assert torch.isfinite(got).all()
 assert torch.linalg.vector_norm(got-ref)/torch.linalg.vector_norm(ref)<5e-7
 assert torch.equal(x,saved)
 assert (guard[:64]==31415.25).all() and (guard[-64:]==31415.25).all()
