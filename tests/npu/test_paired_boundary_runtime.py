"""Paired inverse boundary scheduling with odd batches and padded aliases."""
import pytest
from test_power_real_runtime import test_flat_power_real_live as run_real_live


@pytest.mark.parametrize('n',[185640,340200,524288,663000])
@pytest.mark.parametrize('in_place',[False,True])
def test_paired_boundary_live(n,in_place):
    run_real_live(n,17,True,in_place)
