"""Public geometry selection, aliases, tails and local inverse execution."""
import pytest
from test_batch_dense_prefix_runtime import library,selected_options,_description,_passes,_execute
from test_power_cube_finish_runtime import _PreparedLibrary


@pytest.fixture(autouse=True)
def late_options(monkeypatch,selected_options):
    monkeypatch.setenv('FLAGFFT_NPU_DIF_DENSE_LATE_PAIR','1')
    monkeypatch.setenv('FLAGFFT_NPU_DIF_BATCH_DENSE_PREFIX','1')


@pytest.mark.parametrize('n,batch,admitted',[
    (139230,17,True),(185640,64,True),(371280,32,True),
    (185640,1,False),(185640,15,False),(663000,64,False),(340200,64,False)])
def test_admission(library,monkeypatch,n,batch,admitted):
    descriptions=[]
    for value in ('0','1','0'):
        monkeypatch.setenv('FLAGFFT_NPU_DIF_DENSE_LATE_PAIR',value)
        descriptions.append(_description(library,(n,),batch))
    assert descriptions[0]==descriptions[2]
    if admitted:
        assert all(a-b==1 for a,b in zip(_passes(descriptions[0]),_passes(descriptions[1])))
    else:assert descriptions[0]==descriptions[1]


@pytest.mark.parametrize('n,batch,pattern,in_place,global_inverse',[
    (185640,65,'random',True,True),(139230,17,'basis',False,True),
    (371280,32,'cancellation',True,True),(185640,17,'tone',False,False),
    (139230,64,'random',False,True),(185640,32,'basis',False,True)])
@pytest.mark.parametrize('direction',[-1,1])
def test_live(library,monkeypatch,n,batch,pattern,in_place,global_inverse,direction):
    monkeypatch.setenv('FLAGFFT_NPU_DIF_GLOBAL_INVERSE','1' if global_inverse else '0')
    _execute(_PreparedLibrary(library),monkeypatch,n,batch,pattern,in_place,direction)


@pytest.mark.parametrize('n,batch',[(185640,65),(139230,64),
    (371280,32),(185640,32),(185640,17)])
def test_batch_reuse_preserves_layout(library,monkeypatch,n,batch):
    descriptions=[]
    for value in ('0','1','0'):
        monkeypatch.setenv('FLAGFFT_NPU_DIF_LATE_PAIR_BATCH',value)
        descriptions.append(_description(library,(n,),batch))
    assert descriptions[0]==descriptions[2]
    assert _passes(descriptions[0])==_passes(descriptions[1])
    # This option changes only scheduling; the public factorization, pass
    # count and workspace description remain identical.
    assert descriptions[0]==descriptions[1]
