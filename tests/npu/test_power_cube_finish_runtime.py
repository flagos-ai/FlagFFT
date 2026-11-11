"""Public admission, aliases, partial batches, streams and FP64 references."""
import pytest
from test_batch_dense_prefix_runtime import library,_description,_passes,_execute,torch


@pytest.fixture(autouse=True)
def power_options(monkeypatch):
    for name in ('FLAT_POWER_DIF','FLAT_POWER_WORKSPACE_PAD','FLAT_POWER_TERMINAL8',
                 'FLAT_POWER_CUBE_FINISH','FLAT_POWER_CUBE_BATCH','DIF_GLOBAL_INVERSE'):
        monkeypatch.setenv('FLAGFFT_NPU_'+name,'1')


@pytest.mark.parametrize('n,batch,admitted',[
    (262144,17,True),(524288,64,True),(524288,1,False),
    (1048576,16,False),(16384,64,False)])
def test_plan_selection(library,monkeypatch,n,batch,admitted):
    descriptions=[]
    for value in ('0','1','0'):
        monkeypatch.setenv('FLAGFFT_NPU_FLAT_POWER_CUBE_FINISH',value)
        descriptions.append(_description(library,(n,),batch))
    assert descriptions[0]==descriptions[2]
    if admitted:
        assert _passes(descriptions[0])==(4,4)
        assert _passes(descriptions[1])==(3,3)
    else:
        assert descriptions[0]==descriptions[1]


class _PreparedLibrary:
    def __init__(self,lib):self.lib=lib
    def __getattr__(self,name):return getattr(self.lib,name)
    def flagfftExecC2C(self,*args):
        # PyTorch may queue preparation on a host submission thread. Fence
        # it before directly invoking a foreign C API on the selected stream.
        torch.npu.synchronize()
        return self.lib.flagfftExecC2C(*args)


@pytest.mark.parametrize('n,batch,pattern,in_place',[
    (262144,17,'random',False),(262144,17,'basis',True),
    (262144,65,'tone',True),
    (524288,65,'tone',True),(524288,17,'cancellation',False)])
@pytest.mark.parametrize('direction',[-1,1])
def test_complete_public_fft(library,monkeypatch,n,batch,pattern,in_place,direction):
    _execute(_PreparedLibrary(library),monkeypatch,n,batch,pattern,in_place,direction)
