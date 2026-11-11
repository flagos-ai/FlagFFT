"""Public family selection, partial batches and independently checked values."""
import pytest
from test_batch_dense_prefix_runtime import library,selected_options,_description,_execute
from test_power_cube_finish_runtime import _PreparedLibrary


@pytest.fixture(autouse=True)
def natural_options(monkeypatch,selected_options):
    monkeypatch.setenv('FLAGFFT_NPU_DIF_COARSE_PREFIX','1')
    monkeypatch.setenv('FLAGFFT_NPU_DIF_COARSE_GT','1')
    monkeypatch.setenv('FLAGFFT_NPU_DIF_COARSE_DENSE_PREFIX','1')
    monkeypatch.setenv('FLAGFFT_NPU_DIF_NATURAL126','1')


@pytest.mark.parametrize('n,batch,admitted',[(340200,64,True),(272160,17,True),
    (544320,32,True),(680400,32,True),(340200,1,False),(340200,15,False),
    (185640,64,False),(663000,64,False)])
def test_public_selection(library,monkeypatch,n,batch,admitted):
    descriptions=[]
    for value in ('0','1','0'):
        monkeypatch.setenv('FLAGFFT_NPU_DIF_NATURAL126',value)
        descriptions.append(_description(library,(n,),batch))
    assert descriptions[0]==descriptions[2]
    if admitted:
        assert 'factors=[7,18,' in descriptions[1]
        assert descriptions[0]!=descriptions[1]
    else:assert descriptions[0]==descriptions[1]


@pytest.mark.parametrize('n,batch,pattern,alias,global_inverse',[
    (340200,65,'cancellation',True,True),(272160,17,'tone',False,True),
    (544320,32,'basis',True,True),(680400,17,'random',False,False)])
@pytest.mark.parametrize('direction',[-1,1])
def test_public_fft(library,monkeypatch,n,batch,pattern,alias,global_inverse,direction):
    monkeypatch.setenv('FLAGFFT_NPU_DIF_GLOBAL_INVERSE','1' if global_inverse else '0')
    _execute(_PreparedLibrary(library),monkeypatch,n,batch,pattern,alias,direction)
