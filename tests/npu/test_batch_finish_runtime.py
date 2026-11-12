"""Exercise both sides of the public batch gate and a partial final group."""
import pytest
from test_dif_cube_runtime import test_cube_prefix_live as execute_case


@pytest.mark.parametrize('batch',[63,64,65])
@pytest.mark.parametrize('direction',[-1,1])
@pytest.mark.parametrize('in_place',[False,True])
def test_batch_finish_live(batch,direction,in_place):
    execute_case(185640,batch,direction,in_place)
