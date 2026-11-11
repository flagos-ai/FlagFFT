"""Source contract for the opt-in IX warp-local FFT experiment."""
import ast
import pytest
from flagfft_codegen.kernels_ix_warp import build_ix_warp256_source

@pytest.mark.parametrize('pack',[4,8,16])
@pytest.mark.parametrize('mode,inverse',[
    ('contiguous',False),('contiguous',True),('outer',False),('inner',True),
    ('r2c',False),('c2r',True),
])
def test_warp_fft_source_preserves_generic_pointer_distance_abi(pack,mode,inverse):
    name,source=build_ix_warp256_source(pack,mode,inverse)
    function=ast.parse(source).body[-1]
    assert function.name==name
    assert [v.arg for v in function.args.args]==['inp','out','span','nbatch','input_distance','output_distance']
    assert 'tl.gather' in source and 'tle.gpu.alloc' not in source
    compile(source,'<ix-warp-fft>','exec')

@pytest.mark.parametrize('args',[(3,'outer',False),(8,'missing',False),(8,'r2c',True),(8,'c2r',False)])
def test_warp_fft_rejects_invalid_configuration(args):
    with pytest.raises(ValueError): build_ix_warp256_source(*args)
