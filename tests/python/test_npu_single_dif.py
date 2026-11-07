"""Check sample packing at full and partial final vector tiles."""
import pytest
from test_npu_dif_coarse import test_coarse_full_fft as run_fft

@pytest.mark.parametrize('factors',[(8,15,7,9,9,5),(4,15,7,9,9,5)])
@pytest.mark.parametrize('direction',['forward','inverse','global_inverse'])
@pytest.mark.parametrize('good_thomas',[False,True])
def test_single_sample_pack(tmp_path,factors,direction,good_thomas):
    run_fft(tmp_path,factors,direction,True,good_thomas,single_pack=True)
