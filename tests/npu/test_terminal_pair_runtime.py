"""Geometry admission, tail batches, aliases and original FP32 accuracy gate."""
import pytest
from test_batch_dense_prefix_runtime import library,selected_options,_description,_passes,_execute
from test_natural_prefix_runtime import natural_options
from test_power_cube_finish_runtime import _PreparedLibrary


@pytest.fixture(autouse=True)
def terminal_options(monkeypatch,natural_options):
    monkeypatch.setenv('FLAGFFT_NPU_DIF_TERMINAL_PAIR','1')


@pytest.mark.parametrize('n,batch,admitted',[
    (340200,64,True),(378000,32,True),(362880,32,True),(340200,17,False),
    (272160,64,False),(544320,32,False),(185640,64,False),(663000,64,False)])
def test_public_selection(library,monkeypatch,n,batch,admitted):
    descriptions = []
    for value in ('0','1','0'):
        monkeypatch.setenv('FLAGFFT_NPU_DIF_TERMINAL_PAIR',value)
        descriptions.append(_description(library,(n,),batch))
    assert descriptions[0] == descriptions[2]
    if admitted:
        assert all(a-b==1 for a,b in zip(_passes(descriptions[0]),_passes(descriptions[1])))
    else:
        assert descriptions[0] == descriptions[1]


@pytest.mark.parametrize('n,batch,pattern,alias,global_inverse',[
    (340200,65,'cancellation',True,True),(378000,32,'random',False,True),
    (362880,33,'tone',True,True),(340200,32,'basis',False,False)])
@pytest.mark.parametrize('direction',[-1,1])
def test_public_fft(library,monkeypatch,n,batch,pattern,alias,global_inverse,direction):
    monkeypatch.setenv('FLAGFFT_NPU_DIF_GLOBAL_INVERSE','1' if global_inverse else '0')
    _execute(_PreparedLibrary(library),monkeypatch,n,batch,pattern,alias,direction)
