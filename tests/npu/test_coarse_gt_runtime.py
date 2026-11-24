"""Exercise the coprime coarse prefix through live complex and real C APIs."""
import pytest
from test_dif_cube_runtime import test_cube_prefix_live as run_complex
from test_packed_mixed_runtime import test_packed_mixed_real_live as run_real


@pytest.mark.parametrize('n,batch',[(340200,1),(340200,17),(170100,17)])
@pytest.mark.parametrize('direction',[-1,1])
@pytest.mark.parametrize('in_place',[False,True])
def test_coarse_complex(n,batch,direction,in_place):
    run_complex(n,batch,direction,in_place)


@pytest.mark.parametrize('inverse',[False,True])
@pytest.mark.parametrize('in_place',[False,True])
def test_coarse_real(inverse,in_place):
    run_real(340200,17,inverse,in_place)
