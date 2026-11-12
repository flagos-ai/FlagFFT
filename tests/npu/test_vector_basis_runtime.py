"""Public batches, partial groups, both directions, aliases, scales and streams."""
import pytest
from test_dif_cube_runtime import test_cube_prefix_live as execute_case

@pytest.mark.parametrize('n,batch',[(185640,17),(340200,17),(663000,17),(185640,65),(340200,65)])
@pytest.mark.parametrize('direction',[-1,1])
@pytest.mark.parametrize('in_place',[False,True])
def test_vector_basis_live(n,batch,direction,in_place):
    execute_case(n,batch,direction,in_place)
