"""Source contract for the opt-in IX warp-local FFT experiment."""
import ast
import pytest
from flagfft_codegen.kernels_ix_warp import build_ix_warp256_source

@pytest.mark.parametrize('length',[128,256,512,1024,2048])
@pytest.mark.parametrize('mode,inverse',[('outer',False),('inner',True),('r2c',False),('c2r',True)])
def test_wide_warp_fft_source_is_valid_and_names_length(length,mode,inverse):
    from flagfft_codegen.kernels_ix_warp import build_ix_warp_source
    name,source=build_ix_warp_source(length,4,mode,inverse)
    assert name.startswith(f'warp{length}_')
    assert f'freq=tl.arange(0,{length})' in source
    compile(source,'<ix-wide-warp-fft>','exec')

@pytest.mark.parametrize('length,pack',[(256,16),(1024,8),(2048,4)])
@pytest.mark.parametrize('inverse',[False,True])
def test_manual_warp_store_and_register_codelet_source(length,pack,inverse):
    from flagfft_codegen.kernels_ix_warp import build_ix_warp_source
    name,source=build_ix_warp_source(length,pack,'outer',inverse,fast_math=True,
                                    local_codelet=True,manual_store=True)
    assert name.endswith('_fast_codelet_manual')
    assert 'tle.gpu.alloc' in source and 'tl.debug_barrier()' in source
    compile(source,'<ix-manual-warp-store>','exec')

@pytest.mark.parametrize('pack',[4,8,16,32,64])
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

@pytest.mark.parametrize('mode,direction,kind',[
    ('contiguous','forward','leaf'),('permuted_store','inverse','leaf_permuted_store'),
    ('permuted_store_r2c','forward','leaf_r2c_permuted_store'),
    ('contiguous_c2r','inverse','leaf_c2r'),
    ('contiguous_r2c','forward','leaf_r2c'),
])
@pytest.mark.parametrize('length,pack',[(256,16),(256,32),(512,8),(1024,8),(2048,4)])
def test_warp_route_preserves_raw_leaf_parameters_and_launch(mode,direction,kind,length,pack,tmp_path):
    from flagfft_codegen.backend_profile import BackendProfile,set_profile,reset_profile
    from flagfft_codegen.target import set_ix_warp_fft_pack,reset_ix_warp_fft_pack
    from flagfft_codegen.kernels_common import LeafPlan
    from flagfft_codegen.kernels_leaf import _build_leaf_kernel_source_for_io
    from flagfft_codegen.metadata import _metadata
    ptoken=set_profile(BackendProfile.from_device({'backend':'ix','device_arch':'71','warp_size':64,
                   'max_threads_per_block':4096,'max_dynamic_shared_memory':131072},'balanced'))
    token=set_ix_warp_fft_pack(0)
    try:
        factors={256:(16,16),512:(32,16),1024:(32,32),2048:(16,16,8)}[length]
        plan=LeafPlan(length,factors,1,16,2,(),length,direction,'complex64')
        _,base=_build_leaf_kernel_source_for_io(plan,io_mode=mode)
        base_args=next(v.args for v in ast.parse(base).body if isinstance(v,ast.FunctionDef))
        set_ix_warp_fft_pack(pack)
        name,source=_build_leaf_kernel_source_for_io(plan,io_mode=mode,
                        permutation_span=256 if 'permuted' in mode else 0)
        args=next(v.args for v in ast.parse(source).body if isinstance(v,ast.FunctionDef))
        assert name.startswith(f'ix_warp{length}_')
        assert ast.dump(args)==ast.dump(base_args)
        meta=_metadata(module_path=tmp_path/'kernel.py',kernel_name=name,
                       arg_names=[a.arg for a in args.args],plan=plan,kernel_type=kind,n1=0,n2=0,dtype='complex64')
        assert meta['num_warps']==meta['batch_per_block']==meta['ix_warp_fft_pack']==pack
        assert meta['inner_pack']==1
        compile(source,'<ix-warp-leaf>','exec')
    finally:
        reset_ix_warp_fft_pack(token);reset_profile(ptoken)
