"""Three-axis FFT layouts, inverse conventions and bounded matrix packing."""
import math
from pathlib import Path
import pytest

torch=pytest.importorskip('torch')
from flagfft_codegen.kernels_npu_power_dif import build
from flagfft_codegen.kernels_npu_dif import emit_vector_dif
from test_npu_structured_fft import Pointer,execute_module,roots


@pytest.mark.parametrize('batch_reuse',[1,8,32])
@pytest.mark.parametrize('factors',[(64,32,128),(64,64,128)])
@pytest.mark.parametrize('mode',['forward','inverse','global_inverse'])
def test_complete_fft(tmp_path,factors,mode,batch_reuse):
    global_inverse=mode=='global_inverse'
    direction='inverse' if global_inverse else mode
    sign=1 if direction=='inverse' and not global_inverse else -1
    source,specs=build(factors,direction,terminal8=True,global_inverse=global_inverse,
                       workspace_padding=512,cube_finish=True,cube_finish_batch=batch_reuse)
    path=tmp_path/'power.py';path.write_text(source);ns,pid=execute_module(path)
    ns['tl'].dot=lambda a,b,acc=None,**kw:(a.double()@b.double()+(acc.double() if acc is not None else 0)).float()
    n=math.prod(factors);batch=2
    torch.manual_seed(20261111);x=torch.randn(batch,n,2);saved=x.clone();current=x
    guard=torch.full((x.numel()+128,),31415.25)
    for stage,s in enumerate(specs):
        final=stage+1==len(specs)
        if final:
            r=s['r'];previous=factors[stage-1]
            shift=torch.arange(previous,dtype=torch.float64)[:,None,None]/previous
            f=torch.arange(r,dtype=torch.float64)[None,:,None]
            d=torch.arange(r,dtype=torch.float64)[None,None,:]
            angle=sign*2*math.pi*(f+shift)*d/r
            co,si=angle.cos().float(),angle.sin().float()
            w=torch.cat((torch.cat((co,-si),2),torch.cat((si,co),2)),1).flatten()
        else:
            w=roots(s['r'],s['c'])
            if stage:
                digit=torch.arange(s['prefix'])//s['previous_prefix']
                sample=torch.arange(s['r']*s['m'],dtype=torch.float64)
                angle=sign*2*math.pi*sample[:,None]*digit[None,:]/(factors[stage-1]*s['r']*s['m'])
                w=torch.cat((angle.cos().float().flatten(),angle.sin().float().flatten(),w))
        y=guard if final else torch.full((2*batch*s['physical_n'],),float('nan'))
        for gx in range(s['grid'][0]):
            for b in range(math.ceil(batch/s.get('batch_pack',1))):
                pid[:]=[gx,b]
                args=[Pointer(current),Pointer(w),Pointer(y,64 if final else 0)]
                if s.get('batch_pack',1)>1:args.append(batch)
                ns[s['name']](*args)
        current=y
    actual=torch.view_as_complex(guard[64:-64].reshape_as(x)).cdouble()
    z=torch.view_as_complex(x).cdouble()
    expected=torch.fft.fft(z) if direction=='forward' else torch.fft.ifft(z)*n
    assert torch.isfinite(actual).all()
    assert torch.linalg.vector_norm(actual-expected)/torch.linalg.vector_norm(expected)<5e-7
    assert torch.equal(x,saved)
    assert (guard[:64]==31415.25).all() and (guard[-64:]==31415.25).all()


def test_metadata_and_unsupported_geometry(tmp_path):
    factors=(64,64,128)
    for phase in range(3):
        old=emit_vector_dif(factors,'inverse',phase,tmp_path,flat_power2=True,
                            workspace_padding=512,global_inverse=True)
        original=Path(old['module_path']).read_text()
        new=emit_vector_dif(factors,'inverse',phase,tmp_path,flat_power2=True,
                            workspace_padding=512,global_inverse=True,power_cube_finish=True)
        assert new['module_path']!=old['module_path']
        assert Path(old['module_path']).read_text()==original
        assert new['signature']==old['signature']=='*fp32:16,*fp32:16,*fp32:16'
        if phase==2:assert new['kernel_name']=='power_cube_finish'
    for bad in ((64,128,128),(32,128,128),(64,64,64)):
        with pytest.raises(ValueError,match='bounded vector axes'):
            build(bad,workspace_padding=512,cube_finish=True)
    with pytest.raises(ValueError,match='padded workspace'):
        build(factors,cube_finish=True)
    with pytest.raises(ValueError,match='requires flat power DIF'):
        emit_vector_dif((32,32,16),'forward',0,tmp_path,power_cube_finish=True)


@pytest.mark.parametrize('group',[8,32])
def test_batch_reuse_metadata(tmp_path,group):
    for phase in range(3):
        old=emit_vector_dif((64,64,128),'forward',phase,tmp_path,flat_power2=True,
                            workspace_padding=512,power_cube_finish=True)
        new=emit_vector_dif((64,64,128),'forward',phase,tmp_path,flat_power2=True,
                            workspace_padding=512,power_cube_finish=True,power_cube_batch=group)
        assert old['module_path']!=new['module_path']
        assert new['batch_per_block']==(group if phase==2 else 1)
        assert new['signature']==old['signature']+(',i32' if phase==2 else '')
        if phase==2:assert new['kernel_name']=='power_cube_finish_batch'
    with pytest.raises(ValueError,match='requires Cube finish'):
        emit_vector_dif((64,64,128),'forward',0,tmp_path,flat_power2=True,power_cube_batch=group)


def test_invalid_batch_group():
    for group in (0,2,16,64,True):
        with pytest.raises(ValueError,match='Unsupported power Cube batch group'):
            build((64,64,128),workspace_padding=512,cube_finish=True,cube_finish_batch=group)
