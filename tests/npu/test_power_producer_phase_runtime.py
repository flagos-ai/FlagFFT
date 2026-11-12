"""Producer rotation admission and inverse/partial-batch public execution."""
import pytest
from test_batch_dense_prefix_runtime import library,_description,_passes,_execute
from test_power_cube_finish_runtime import power_options,_PreparedLibrary


@pytest.fixture(autouse=True)
def producer_options(monkeypatch,power_options):
    monkeypatch.setenv('FLAGFFT_NPU_FLAT_POWER_FIRST_PIPELINE','1')
    monkeypatch.setenv('FLAGFFT_NPU_FLAT_POWER_FIRST_PHASE','1')


@pytest.mark.parametrize('n,batch,admitted',[
    (262144,32,True),(524288,33,True),(262144,17,False),(1048576,32,False)])
def test_selection_and_cache(library,monkeypatch,n,batch,admitted):
    descriptions=[]
    for value in ('0','1','0'):
        monkeypatch.setenv('FLAGFFT_NPU_FLAT_POWER_FIRST_PHASE',value)
        descriptions.append(_description(library,(n,),batch))
    assert descriptions[0]==descriptions[2]
    if admitted:
        assert 'first_phase=' not in descriptions[0]
        assert descriptions[1].count('first_phase=1')==2
        assert _passes(descriptions[0])==_passes(descriptions[1])==(3,3)
    else:
        assert descriptions[0]==descriptions[1]


@pytest.mark.parametrize('n,batch,pattern,alias,global_inverse',[
    (262144,32,'basis',True,False),(524288,33,'cancellation',False,False),
    (262144,32,'random',False,True)])
@pytest.mark.parametrize('direction',[-1,1])
def test_public_fft(library,monkeypatch,n,batch,pattern,alias,global_inverse,direction):
    monkeypatch.setenv('FLAGFFT_NPU_DIF_GLOBAL_INVERSE','1' if global_inverse else '0')
    _execute(_PreparedLibrary(library),monkeypatch,n,batch,pattern,alias,direction)
