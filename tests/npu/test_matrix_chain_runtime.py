"""Public matrix-chain geometry, cache toggles and live FFT execution."""
import pytest
from test_batch_dense_prefix_runtime import library,selected_options,_description,_passes,_execute
from test_power_cube_finish_runtime import _PreparedLibrary


@pytest.fixture(autouse=True)
def matrix_options(monkeypatch,selected_options):
    monkeypatch.setenv('FLAGFFT_NPU_DIF_BATCH_DENSE_PREFIX','1')
    monkeypatch.setenv('FLAGFFT_NPU_DIF_MATRIX_CHAIN','1')


@pytest.mark.parametrize('n,batch,admitted',[
    (663000,64,True),(331500,33,True),(530400,32,True),(663000,32,True),
    (265200,64,False),(185640,64,False),(331500,17,False),(255*26*160,64,False),
    (371280,64,False)])
def test_selection_and_cache(library,monkeypatch,n,batch,admitted):
    descriptions=[]
    for value in ('0','1','0'):
        monkeypatch.setenv('FLAGFFT_NPU_DIF_MATRIX_CHAIN',value)
        descriptions.append(_description(library,(n,),batch))
    assert descriptions[0]==descriptions[2]
    if admitted:
        assert 'matrix_suffix=' not in descriptions[0]
        assert descriptions[1].count(f'matrix_suffix=26x{n//(255*26)}')==2
        assert _passes(descriptions[1])==(3,3)
        assert all(x>3 for x in _passes(descriptions[0]))
    else:
        assert descriptions[0]==descriptions[1]


@pytest.mark.parametrize('n,batch,pattern,alias,global_inverse',[
    (663000,64,'random',False,True),(663000,33,'tone',True,True),
    (331500,33,'cancellation',False,True),(530400,32,'random',True,True),
    (331500,64,'basis',False,True),(663000,32,'random',False,False)])
@pytest.mark.parametrize('direction',[-1,1])
def test_public_fft(library,monkeypatch,n,batch,pattern,alias,global_inverse,direction):
    monkeypatch.setenv('FLAGFFT_NPU_DIF_GLOBAL_INVERSE','1' if global_inverse else '0')
    _execute(_PreparedLibrary(library),monkeypatch,n,batch,pattern,alias,direction)
