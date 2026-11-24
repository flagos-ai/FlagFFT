"""Public first-axis pipeline admission, cache toggles and partial batch groups."""
import pytest
from test_batch_dense_prefix_runtime import library,_description,_passes,_execute
from test_power_cube_finish_runtime import power_options,_PreparedLibrary


@pytest.fixture(autouse=True)
def first_pipeline_options(monkeypatch,power_options):
    monkeypatch.setenv('FLAGFFT_NPU_FLAT_POWER_FIRST_PIPELINE','1')


@pytest.mark.parametrize('n,batch,group',[
    (262144,32,8),(524288,64,16),(524288,33,16),
    (262144,17,1),(524288,1,1),(1048576,32,1)])
def test_selection_and_cache(library,monkeypatch,n,batch,group):
    descriptions=[]
    for value in ('0','1','0'):
        monkeypatch.setenv('FLAGFFT_NPU_FLAT_POWER_FIRST_PIPELINE',value)
        descriptions.append(_description(library,(n,),batch))
    assert descriptions[0]==descriptions[2]
    if group>1:
        assert 'first_batch=' not in descriptions[0]
        assert descriptions[1].count(f'first_batch={group}')==2
        assert _passes(descriptions[0])==_passes(descriptions[1])==(3,3)
    else:
        assert descriptions[0]==descriptions[1]


@pytest.mark.parametrize('n,batch,pattern,alias,global_inverse',[
    (262144,33,'random',False,True),(262144,65,'tone',True,True),
    (524288,65,'cancellation',True,True),(524288,32,'basis',False,False)])
@pytest.mark.parametrize('direction',[-1,1])
def test_public_fft(library,monkeypatch,n,batch,pattern,alias,global_inverse,direction):
    monkeypatch.setenv('FLAGFFT_NPU_DIF_GLOBAL_INVERSE','1' if global_inverse else '0')
    _execute(_PreparedLibrary(library),monkeypatch,n,batch,pattern,alias,direction)
