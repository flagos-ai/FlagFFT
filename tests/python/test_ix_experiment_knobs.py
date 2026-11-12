"""IX diagnostic switches must not silently change another backend."""
import pytest

from flagfft_codegen.backend_profile import BackendProfile, set_profile, reset_profile
from flagfft_codegen.kernels_common import (
    four_step_row_inner_pack_for, four_step_col_inner_pack_for,
)
from flagfft_codegen.kernels_leaf import _portable_complex_vector_io


@pytest.fixture
def ix_profile():
    token = set_profile(BackendProfile.from_device({
        "backend": "ix", "device_arch": "71", "warp_size": 64,
        "max_threads_per_block": 4096, "max_dynamic_shared_memory": 131072,
    }))
    yield
    reset_profile(token)


@pytest.fixture
def ix_balanced_profile(ix_profile):
    from dataclasses import replace
    from flagfft_codegen.backend_profile import current_profile
    token = set_profile(replace(current_profile(),policy="balanced"))
    yield
    reset_profile(token)


def test_ix_vector_io_experiment_is_not_enabled(ix_profile, monkeypatch):
    monkeypatch.delenv("FLAGFFT_IX_VEC_IO", raising=False)
    assert not _portable_complex_vector_io()
    monkeypatch.setenv("FLAGFFT_IX_VEC_IO", "1")
    assert not _portable_complex_vector_io()


def test_ix_packed_complex_io_uses_portable_bitcasts(ix_profile, monkeypatch):
    from flagfft_codegen.kernels_common import LeafPlan
    from flagfft_codegen.kernels_leaf import _build_leaf_kernel_source_for_io
    monkeypatch.setenv("FLAGFFT_IX_VEC_IO", "packed")
    for dtype in ("complex64", "complex128"):
        plan = LeafPlan(256, (16, 16), 1, 16, 2, (), 256, "forward", dtype)
        _, source = _build_leaf_kernel_source_for_io(plan, io_mode="permuted_store")
        assert "inline_asm_elementwise" not in source
        if dtype == "complex64":
            assert "tl.pointer_type(tl.uint64)" in source
            assert "bitcast=True" in source
        else:
            assert "tl.pointer_type(tl.uint64)" not in source
        compile(source, "<ix-packed-io>", "exec")


@pytest.mark.parametrize("pack", [1, 2, 4, 8])
def test_ix_tle_pack_override(ix_profile, monkeypatch, pack):
    monkeypatch.setenv("FLAGFFT_IX_PORTABLE_LEAF", "0")
    monkeypatch.setenv("FLAGFFT_IX_TLE_INNER_PACK", str(pack))
    assert four_step_row_inner_pack_for(1024, 1024) == pack
    assert four_step_col_inner_pack_for(1024, 1024) == pack


def test_ix_tle_pack_rejects_zero(ix_profile, monkeypatch):
    monkeypatch.setenv("FLAGFFT_IX_PORTABLE_LEAF", "0")
    monkeypatch.setenv("FLAGFFT_IX_TLE_INNER_PACK", "0")
    with pytest.raises(ValueError):
        four_step_row_inner_pack_for(1024, 1024)


@pytest.mark.parametrize("shared_limit, expected_pack", [(16384, 4), (32768, 8), (131072, 8)])
def test_ix_portable_power_two_pack_respects_exchange_budget(monkeypatch, shared_limit, expected_pack):
    from flagfft_codegen.kernels_common import LeafPlan
    from flagfft_codegen.kernels_leaf import _build_four_step_col_kernel_source
    from flagfft_codegen.metadata import _metadata
    from pathlib import Path

    token = set_profile(BackendProfile.from_device({
        "backend": "ix", "device_arch": "71", "warp_size": 64,
        "max_threads_per_block": 4096, "max_dynamic_shared_memory": shared_limit,
    }))
    monkeypatch.setenv("FLAGFFT_IX_PORTABLE_LEAF", "1")
    monkeypatch.setenv("FLAGFFT_IX_INNER_PACK", "8")
    try:
        plan = LeafPlan(1024, (32, 32), 1, 32, 1, (), 1024, "forward", "complex64")
        name, source = _build_four_step_col_kernel_source(plan, 512, 1024)
        meta = _metadata(module_path=Path("unused.py"), kernel_name=name,
                         arg_names=[], plan=plan, kernel_type="four_step_col",
                         n1=512, n2=1024, dtype="complex64")
        assert meta["inner_pack"] == expected_pack
        assert f"four_step_inner_base = tl.program_id(0) * {expected_pack}" in source
    finally:
        reset_profile(token)


def test_ix_swizzle_changes_both_exchange_sides(ix_profile, monkeypatch):
    from flagfft_codegen.kernels_common import LeafPlan
    from flagfft_codegen.kernels_leaf import _build_leaf_kernel_source
    monkeypatch.setenv("FLAGFFT_IX_PORTABLE_LEAF", "0")
    monkeypatch.setenv("FLAGFFT_IX_SMEM_SWIZZLE", "1")
    monkeypatch.setenv("FLAGFFT_IX_SMEM_SWIZZLE_SHIFT", "3")
    plan = LeafPlan(1024, (8, 8, 4, 4), 1, 128, 4, (), 1024, "forward", "complex64")
    _, source = _build_leaf_kernel_source(plan)
    assert "smem_dst0 = dst0 ^ (dst0 >> 3)" in source
    assert "smem_phys0 = logical_phys0 ^ (logical_phys0 >> 3)" in source
    compile(source, "<ix-swizzle>", "exec")


def test_ix_four_step_interleave_is_symmetric(ix_profile, monkeypatch):
    from flagfft_codegen.kernels_common import LeafPlan
    from flagfft_codegen.kernels_leaf import _build_four_step_row_kernel_source
    monkeypatch.setenv("FLAGFFT_IX_PORTABLE_LEAF", "0")
    monkeypatch.setenv("FLAGFFT_IX_SMEM_INTERLEAVE", "1")
    monkeypatch.setenv("FLAGFFT_IX_TLE_INNER_PACK", "8")
    plan = LeafPlan(1024, (32, 32), 1, 32, 2, (32,), 1024, "forward", "complex64")
    assert four_step_row_inner_pack_for(1024, 1024, plan=plan) == 8
    _, source = _build_four_step_row_kernel_source(plan, 1024, 1024)
    assert "phys0 = logical_phys0 * 8 + inner_slot" in source
    assert "smem_dst0 = dst0 * 8 + inner_slot" in source
    compile(source, "<ix-interleave>", "exec")


def test_ix_permuted_pack_matches_source_and_launch(ix_profile, monkeypatch):
    from pathlib import Path
    from flagfft_codegen.kernels_common import LeafPlan
    from flagfft_codegen.kernels_leaf import _build_leaf_kernel_source_for_io
    from flagfft_codegen.metadata import _metadata

    monkeypatch.setenv("FLAGFFT_IX_3D_PACK", "8")
    plan = LeafPlan(256, (16, 16), 1, 16, 2, (), 256, "forward", "complex64")
    name, source = _build_leaf_kernel_source_for_io(plan, io_mode="permuted_store")
    meta = _metadata(module_path=Path("unused.py"), kernel_name=name,
                     arg_names=[], plan=plan, kernel_type="leaf_permuted_store",
                     n1=0, n2=0, dtype="complex64")
    assert meta["batch_per_block"] == 8
    assert "batch_id = pid * 8" in source
    monkeypatch.setenv("FLAGFFT_IX_3D_PACK", "3")
    with pytest.raises(ValueError, match="FLAGFFT_IX_3D_PACK"):
        _build_leaf_kernel_source_for_io(plan, io_mode="permuted_store")


@pytest.mark.parametrize("shared_limit,expected", [(16384,2),(32768,4),(65536,8),(131072,16)])
def test_ix_explicit_permuted_pack_uses_rounded_hardware_bound(monkeypatch, shared_limit, expected):
    from flagfft_codegen.kernels_common import LeafPlan, permuted_store_batch_pack_for
    token = set_profile(BackendProfile.from_device({
        "backend":"ix", "device_arch":"71", "warp_size":64,
        "max_threads_per_block":4096, "max_dynamic_shared_memory":shared_limit,
    }))
    monkeypatch.setenv("FLAGFFT_IX_3D_PACK", "32")
    try:
        plan = LeafPlan(256,(16,16),1,16,2,(),256,"forward","complex64")
        assert permuted_store_batch_pack_for(plan) == expected
    finally:
        reset_profile(token)


def test_ix_permuted_pair_store_is_explicit(ix_profile, monkeypatch):
    from flagfft_codegen.kernels_common import LeafPlan
    from flagfft_codegen.kernels_leaf import _build_leaf_kernel_source_for_io

    monkeypatch.delenv("FLAGFFT_IX_3D_PAIR_STORE", raising=False)
    plan = LeafPlan(256, (16, 16), 1, 16, 2, (), 256, "forward", "complex64")
    _, scalar = _build_leaf_kernel_source_for_io(plan, io_mode="permuted_store")
    assert "pair0 = tl.join(zr0, zi0)" not in scalar
    monkeypatch.setenv("FLAGFFT_IX_3D_PAIR_STORE", "1")
    _, paired = _build_leaf_kernel_source_for_io(plan, io_mode="permuted_store")
    assert "pair0 = tl.join(zr0, zi0)" in paired


@pytest.mark.parametrize("mode", ["contiguous", "strided", "permuted_store", "permuted_store_r2c"])
def test_ix_batch_interleave_has_no_padding_or_cross_slot_aliases(ix_balanced_profile, monkeypatch, mode):
    from flagfft_codegen.kernels_common import LeafPlan
    from flagfft_codegen.kernels_leaf import _build_leaf_kernel_source_for_io
    monkeypatch.setenv("FLAGFFT_IX_BATCH_INTERLEAVE", "1")
    monkeypatch.setenv("FLAGFFT_IX_PORTABLE_LEAF", "0")
    plan = LeafPlan(256,(16,16),1,16,2,(),256,"forward","complex64")
    _, source = _build_leaf_kernel_source_for_io(plan, io_mode=mode)
    assert "phys0 = logical_phys0 * 4 + batch_slot" in source
    assert "smem_dst0 = dst0 * 4 + batch_slot" in source
    assert "tle.gpu.alloc([1024]" in source
    assert "+ inner_slot" not in source
    physical = [logical*4+slot for logical in range(256) for slot in range(4)]
    assert sorted(physical) == list(range(1024))
    compile(source,"<ix-batch-interleave>","exec")


def test_ix_packed_transpose_preserves_native_pointer_abi(ix_profile, monkeypatch, tmp_path):
    from pathlib import Path
    from flagfft_codegen.emit import _emit_tiled_transpose_jit_kernel
    monkeypatch.setenv("FLAGFFT_IX_2D_TRANSPOSE", "packed")
    for dtype, packed in (("complex64",True),("complex128",False)):
        meta = _emit_tiled_transpose_jit_kernel(n0=129,n1=256,dtype=dtype,out_dir=tmp_path/dtype)
        source = Path(meta["module_path"]).read_text()
        assert ("tl.pointer_type(tl.uint64)" in source) == packed
        assert meta["signature"].endswith(",i32")
        assert "inline_asm_elementwise" not in source


@pytest.mark.parametrize("kernel,n1,n2,dtype,expected", [
    ("four_step_row",512,1024,"complex64",True),
    ("four_step_c2r_col",512,512,"complex64",True),
    ("bluestein_four_step_finish_col_r2c",1024,1024,"complex64",True),
    ("four_step_row_strided",512,1024,"complex64",False),
    ("four_step_row",128,128,"complex64",False),
    ("four_step_row",425,1560,"complex64",False),
    ("four_step_row",512,1024,"complex128",False),
])
def test_ix_large_fourstep_policy_scope(ix_balanced_profile, monkeypatch, kernel,n1,n2,dtype,expected):
    from flagfft_codegen.kernels_common import _ix_large_fourstep_default_for
    assert _ix_large_fourstep_default_for(kernel,n1,n2,dtype) == expected
    monkeypatch.setenv("FLAGFFT_IX_LARGE_FOURSTEP", "0")
    assert not _ix_large_fourstep_default_for(kernel,n1,n2,dtype)


@pytest.mark.parametrize("length,factors,lanes,kind,warps", [
    (512,(32,16),16,"four_step_row",4),
    (1024,(32,32),32,"four_step_col",8),
])
def test_ix_large_fourstep_default_matches_qualified_explicit_policy(ix_balanced_profile, monkeypatch,
                                                                   length,factors,lanes,kind,warps):
    from pathlib import Path
    from flagfft_codegen.kernels_common import LeafPlan
    from flagfft_codegen.kernels_leaf import _build_four_step_row_kernel_source, _build_four_step_col_kernel_source
    from flagfft_codegen.metadata import _metadata
    from flagfft_codegen.target import set_ix_large_fourstep_default,reset_ix_large_fourstep_default
    plan=LeafPlan(length,factors,1,lanes,1,(),length,"forward","complex64")
    builder=_build_four_step_row_kernel_source if kind=="four_step_row" else _build_four_step_col_kernel_source
    def emit():
        name,source=builder(plan,512,1024)
        meta=_metadata(module_path=Path("unused.py"),kernel_name=name,arg_names=[],plan=plan,
                       kernel_type=kind,n1=512,n2=1024,dtype="complex64")
        return source,meta
    settings={"PORTABLE_LEAF":"1","RECURRENCE":"1","INNER_PACK":"16","MAX_PACK":"16",
              "RESOURCE_WARPS":"1","VEC_IO":"packed"}
    for key,value in settings.items(): monkeypatch.setenv("FLAGFFT_IX_"+key,value)
    reference=emit()
    for key in settings: monkeypatch.delenv("FLAGFFT_IX_"+key)
    token=set_ix_large_fourstep_default(True)
    try:
        actual=emit()
        assert actual==reference
        assert actual[1]["inner_pack"]==16 and actual[1]["num_warps"]==warps
    finally:
        reset_ix_large_fourstep_default(token)


@pytest.mark.parametrize("length, factors, lanes, kind, expected", [
    (512, (32, 16), 16, "four_step_row", 2),
    (1024, (32, 32), 32, "four_step_col", 4),
])
def test_ix_resource_warps_and_explicit_override(ix_profile, monkeypatch, length, factors, lanes, kind, expected):
    from pathlib import Path
    from flagfft_codegen.kernels_common import LeafPlan
    from flagfft_codegen.metadata import _metadata

    monkeypatch.setenv("FLAGFFT_IX_PORTABLE_LEAF", "1")
    monkeypatch.setenv("FLAGFFT_IX_INNER_PACK", "8")
    monkeypatch.setenv("FLAGFFT_IX_RESOURCE_WARPS", "1")
    monkeypatch.delenv("FLAGFFT_IX_WARPS", raising=False)
    plan = LeafPlan(length, factors, 1, lanes, 1, (), length, "forward", "complex64")
    params = dict(module_path=Path("unused.py"), kernel_name="resource_kernel",
                  arg_names=[], plan=plan, kernel_type=kind, n1=512, n2=1024,
                  dtype="complex64")
    assert _metadata(**params)["num_warps"] == expected
    monkeypatch.setenv("FLAGFFT_IX_WARPS", "8")
    assert _metadata(**params)["num_warps"] == 8


def test_ix_short_register_pack_keeps_launch_and_source_consistent(ix_profile, monkeypatch):
    from pathlib import Path
    from flagfft_codegen.kernels_common import LeafPlan, _maca_p8_register_leaf_supported
    from flagfft_codegen.kernels_leaf import _build_four_step_col_kernel_source
    from flagfft_codegen.metadata import _metadata

    monkeypatch.setenv("FLAGFFT_IX_PORTABLE_LEAF", "1")
    monkeypatch.setenv("FLAGFFT_IX_INNER_PACK", "16")
    monkeypatch.setenv("FLAGFFT_IX_MAX_PACK", "16")
    plan = LeafPlan(128, (16, 8), 1, 16, 4, (), 128, "forward", "complex64")
    assert not _maca_p8_register_leaf_supported(plan)
    monkeypatch.setenv("FLAGFFT_IX_SHORT_REGISTER_PACK", "1")
    assert _maca_p8_register_leaf_supported(plan)
    name, source = _build_four_step_col_kernel_source(plan, 128, 128)
    meta = _metadata(module_path=Path("unused.py"), kernel_name=name, arg_names=[],
                     plan=plan, kernel_type="four_step_col", n1=128, n2=128,
                     dtype="complex64")
    assert meta["inner_pack"] == 16
    assert "four_step_inner_base = tl.program_id(0) * 16" in source
    for length, factors, dtype in ((210, (7, 6, 5), "complex64"),
                                    (128, (16, 8), "complex128")):
        unsupported = LeafPlan(length, factors, 1, 16, 4, (), length, "forward", dtype)
        assert not _maca_p8_register_leaf_supported(unsupported)


def test_ix_tle_presets_are_scoped_and_overridable(ix_profile, monkeypatch):
    from flagfft_codegen.kernels_common import _maca_knob
    from flagfft_codegen.target import set_ix_ct_single_tle_default, reset_ix_ct_single_tle_default
    for policy in (1, 2):
        token = set_ix_ct_single_tle_default(policy)
        try:
            assert _maca_knob("SMEM_INTERLEAVE") == "1"
            assert _maca_knob("RECURRENCE") == "1"
            assert _maca_knob("PORTABLE_LEAF") != "1"
            if policy == 2:
                assert _maca_knob("TLE_INNER_PACK") == "8"
                assert _maca_knob("WARPS") == "4"
            monkeypatch.setenv("FLAGFFT_IX_SMEM_INTERLEAVE", "0")
            assert _maca_knob("SMEM_INTERLEAVE") == "0"
            monkeypatch.delenv("FLAGFFT_IX_SMEM_INTERLEAVE")
        finally:
            reset_ix_ct_single_tle_default(token)
    assert _maca_knob("TLE_INNER_PACK") == ""


def test_ix_1024_single_two_warp_default_and_override(ix_profile, monkeypatch):
    from pathlib import Path
    from flagfft_codegen.kernels_common import LeafPlan
    from flagfft_codegen.metadata import _metadata
    from flagfft_codegen.target import set_ix_ct_single_default, reset_ix_ct_single_default
    token = set_ix_ct_single_default(True)
    monkeypatch.delenv("FLAGFFT_IX_WARPS", raising=False)
    try:
        plan = LeafPlan(1024, (16, 8, 8), 1, 128, 4, (), 1024, "forward", "complex64")
        args = dict(module_path=Path("unused.py"), kernel_name="fft_kernel", arg_names=[],
                    plan=plan, kernel_type="leaf", n1=0, n2=0, dtype="complex64")
        assert _metadata(**args)["num_warps"] == 2
        monkeypatch.setenv("FLAGFFT_IX_WARPS", "1")
        assert _metadata(**args)["num_warps"] == 1
    finally:
        reset_ix_ct_single_default(token)


def test_ix_ct_batch_portable_defaults_and_override(ix_profile, monkeypatch):
    from flagfft_codegen.kernels_common import _maca_knob
    from flagfft_codegen.target import set_ix_ct_batch_default, reset_ix_ct_batch_default

    token = set_ix_ct_batch_default(True)
    try:
        assert _maca_knob("PORTABLE_LEAF") == "1"
        assert _maca_knob("RECURRENCE") == "1"
        assert _maca_knob("WARPS") == "4"
        assert _maca_knob("LANE_MIN") == "1"
        monkeypatch.setenv("FLAGFFT_IX_WARPS", "2")
        assert _maca_knob("WARPS") == "2"
    finally:
        reset_ix_ct_batch_default(token)
    assert _maca_knob("PORTABLE_LEAF") != "1"


@pytest.mark.parametrize('mode,perm_form', [
    ('permuted_store','outer'), ('permuted_store','inner'),
    ('strided_permuted_store','inner'), ('permuted_store_r2c','outer'),
    ('permuted_r2c','outer'),
])
def test_ix_shared_tail_reuses_buffer_after_all_reads(ix_balanced_profile, monkeypatch, mode, perm_form):
    from flagfft_codegen.kernels_common import LeafPlan
    from flagfft_codegen.kernels_leaf import _build_leaf_kernel_source_for_io
    monkeypatch.setenv('FLAGFFT_IX_BATCH_INTERLEAVE','1')
    monkeypatch.setenv('FLAGFFT_IX_3D_SMEM_STORE','1')
    monkeypatch.setenv('FLAGFFT_IX_3D_PACK','16')
    monkeypatch.setenv('FLAGFFT_IX_PORTABLE_LEAF','0')
    plan=LeafPlan(256,(16,16),1,16,2,(),256,'forward','complex64')
    name,source=_build_leaf_kernel_source_for_io(plan,io_mode=mode,perm_form=perm_form)
    assert name.endswith('_smstore')
    last_read=source.rfind('local_ptr(smem_b_i, (phys15,))')
    first_write=source.index('tail_dst0 =')
    assert last_read >= 0
    assert 'tl.debug_barrier()' in source[last_read:first_write]
    assert 'tl.debug_barrier()' in source[source.index('tail_dst15 ='):source.index('tail_r =')]
    assert 'tail_flat = tl.arange(0, 4096)' in source
    assert ('tail_freq < 129' in source) == ('r2c' in mode)
    assert ('tail_batch = perm_base' in source) == (perm_form=='inner')
    compile(source,'<ix-shared-tail>','exec')
    # The scratch permutation is a bijection of every (frequency, FFT slot),
    # including the compact real boundary. No two radix outputs alias.
    scratch=[(lane+j*16)*16+slot for lane in range(16) for j in range(16) for slot in range(16)]
    assert sorted(scratch)==list(range(4096))
    kept=[p for p in scratch if p//16 < (129 if 'r2c' in mode else 256)]
    assert len(kept)==16*(129 if 'r2c' in mode else 256)


def test_ix_shared_tail_rejects_multiple_unread_groups(ix_balanced_profile, monkeypatch):
    from flagfft_codegen.kernels_common import LeafPlan
    from flagfft_codegen.kernels_leaf import _build_leaf_kernel_source_for_io
    monkeypatch.setenv('FLAGFFT_IX_BATCH_INTERLEAVE','1')
    monkeypatch.setenv('FLAGFFT_IX_3D_SMEM_STORE','1')
    monkeypatch.setenv('FLAGFFT_IX_PORTABLE_LEAF','0')
    # Multiple final groups: overwriting the source buffer would destroy future input.
    plan=LeafPlan(256,(64,1,4),1,4,2,(),256,'forward','complex64')
    name,source=_build_leaf_kernel_source_for_io(plan,io_mode='permuted_store')
    assert not name.endswith('_smstore') and 'tail_flat' not in source


@pytest.mark.parametrize('mode',['bluestein_prepare_leaf','bluestein_finish_leaf'])
def test_ix_bluestein_boundary_single_buffer_allows_only_complete_stages(ix_balanced_profile, monkeypatch, mode):
    from flagfft_codegen.kernels_common import LeafPlan, _use_single_smem_buffer
    from flagfft_codegen.kernels_leaf import _build_leaf_kernel_source_for_io
    plan=LeafPlan(2048,(16,16,8),1,128,4,(),2048,'forward','complex64')
    assert not _use_single_smem_buffer(plan,io_mode=mode)
    monkeypatch.setenv('FLAGFFT_IX_BLUESTEIN_SINGLE_SMEM','1')
    assert _use_single_smem_buffer(plan,io_mode=mode)
    _,source=_build_leaf_kernel_source_for_io(plan,io_mode=mode,prime_n=997)
    assert 'smem_a_r = tle.gpu.alloc' not in source
    assert 'smem_b_r = tle.gpu.alloc' in source
    assert not _use_single_smem_buffer(plan,io_mode='bluestein_full_leaf')
    # Varying lanes is safe only when every overwriting stage reads its entire input.
    incomplete=LeafPlan(4096,(64,1,64),1,64,4,(),4096,'forward','complex64')
    assert not _use_single_smem_buffer(incomplete,io_mode=mode)
    compile(source,'<ix-boundary-single-buffer>','exec')


@pytest.mark.parametrize('length,factors,lanes,requested,expected',[
    (512,(32,16),16,32,32), (1024,(32,32),32,32,16),
    (2048,(16,16,8),128,16,8), (4096,(16,16,16),256,8,4),
])
def test_ix_wide_permuted_single_buffer_pack_matches_queried_memory(ix_balanced_profile,monkeypatch,
                                                                  length,factors,lanes,requested,expected):
    from flagfft_codegen.kernels_common import LeafPlan, _use_single_smem_buffer, permuted_store_batch_pack_for
    from flagfft_codegen.kernels_leaf import _build_leaf_kernel_source_for_io
    monkeypatch.setenv('FLAGFFT_IX_PERMUTED_SINGLE_SMEM','1')
    monkeypatch.setenv('FLAGFFT_IX_BATCH_INTERLEAVE','1')
    monkeypatch.setenv('FLAGFFT_IX_3D_SMEM_STORE','1')
    monkeypatch.setenv('FLAGFFT_IX_3D_PACK',str(requested))
    monkeypatch.setenv('FLAGFFT_IX_PORTABLE_LEAF','0')
    plan=LeafPlan(length,factors,1,lanes,4,(),length,'forward','complex64')
    assert _use_single_smem_buffer(plan,io_mode='permuted_store')
    assert permuted_store_batch_pack_for(plan)==expected
    name,source=_build_leaf_kernel_source_for_io(plan,io_mode='permuted_store')
    assert 'smem_a_r = tle.gpu.alloc' not in source
    complete_last = length == lanes*factors[-1] if lanes>32 else True
    assert name.endswith('_smstore') == complete_last
    assert (f'tail_flat = tl.arange(0, {length*expected})' in source) == complete_last
    assert 2*length*expected*4==131072
    monkeypatch.setenv('FLAGFFT_IX_BATCH_INTERLEAVE','0')
    assert not _use_single_smem_buffer(plan,io_mode='permuted_store')


@pytest.mark.parametrize('mode',['permuted_store','strided_permuted_store','permuted_store_r2c'])
def test_ix_static_permutation_span_keeps_runtime_abi_and_distinct_sources(ix_balanced_profile,tmp_path,mode):
    from flagfft_codegen.kernels_common import LeafPlan
    from flagfft_codegen.kernels_leaf import _build_leaf_kernel_source_for_io
    import ast
    plan=LeafPlan(256,(16,16),1,16,2,(),256,'forward','complex64')
    dynamic_name,dynamic_source=_build_leaf_kernel_source_for_io(plan,io_mode=mode)
    dynamic_args=ast.parse(dynamic_source).body[0].args
    for span in (33,256,65536):
        name,source=_build_leaf_kernel_source_for_io(plan,io_mode=mode,permutation_span=span)
        assert name!=dynamic_name and name.endswith(f'_s{span}')
        assert f'perm_span = tl.full((), {span}, tl.int64)' in source
        assert ast.dump(ast.parse(source).body[0].args)==ast.dump(dynamic_args)
    with pytest.raises(ValueError,match='static permutation span'):
        _build_leaf_kernel_source_for_io(plan,io_mode='contiguous',permutation_span=256)


def test_ix_radix8_permuted_trial_reduces_shared_bytes_per_thread(ix_balanced_profile,monkeypatch):
    from pathlib import Path
    from flagfft_codegen.kernels_common import LeafPlan
    from flagfft_codegen.kernels_leaf import _build_leaf_kernel_source_for_io
    from flagfft_codegen.metadata import _metadata
    monkeypatch.setenv('FLAGFFT_IX_PERMUTED_SINGLE_SMEM','1')
    monkeypatch.setenv('FLAGFFT_IX_BATCH_INTERLEAVE','1')
    plan=LeafPlan(256,(8,8,4),1,32,1,(),256,'forward','complex64')
    name,source=_build_leaf_kernel_source_for_io(plan,io_mode='permuted_store')
    meta=_metadata(module_path=Path('unused.py'),kernel_name=name,arg_names=[],plan=plan,
                   kernel_type='leaf_permuted_store',n1=0,n2=0,dtype='complex64')
    assert meta['batch_per_block']==4 and meta['num_warps']==4
    assert 'smem_a_r = tle.gpu.alloc' not in source
    assert 'smem_b_r = tle.gpu.alloc([1024]' in source
    assert 'tail_flat' not in source
    compile(source,'<ix-radix8-permuted>','exec')


def test_ix_radix8_contiguous_reuses_buffer_with_complete_intermediate_stages(ix_balanced_profile,monkeypatch):
    from flagfft_codegen.kernels_common import LeafPlan, _use_single_smem_buffer
    from flagfft_codegen.kernels_leaf import _build_leaf_kernel_source_for_io
    plan=LeafPlan(256,(8,8,4),1,32,1,(),256,'forward','complex64')
    assert not _use_single_smem_buffer(plan,io_mode='contiguous')
    monkeypatch.setenv('FLAGFFT_IX_RADIX8','1')
    monkeypatch.setenv('FLAGFFT_IX_BATCH_INTERLEAVE','1')
    monkeypatch.setenv('FLAGFFT_IX_BATCH_PACK','4')
    assert _use_single_smem_buffer(plan,io_mode='contiguous')
    _,source=_build_leaf_kernel_source_for_io(plan,io_mode='contiguous')
    assert 'smem_a_r = tle.gpu.alloc' not in source
    assert 'smem_b_r = tle.gpu.alloc([1024]' in source

@pytest.mark.parametrize('n,factors,lanes,pack',[
    (16,(4,4),4,16),(64,(16,4),4,16),(64,(8,8),8,8),
    (128,(16,8),8,4),(256,(16,16),16,4),
])
def test_ix_short_permuted_fill_uses_every_lane_of_one_warp(ix_balanced_profile,monkeypatch,n,factors,lanes,pack):
    from pathlib import Path
    from flagfft_codegen.kernels_common import LeafPlan,permuted_store_batch_pack_for
    from flagfft_codegen.kernels_leaf import _build_leaf_kernel_source_for_io
    from flagfft_codegen.metadata import _metadata
    monkeypatch.setenv('FLAGFFT_IX_WARP_FILL_PACK','1')
    plan=LeafPlan(n,factors,1,lanes,1,(),n,'forward','complex64')
    assert permuted_store_batch_pack_for(plan)==pack
    name,source=_build_leaf_kernel_source_for_io(plan,io_mode='permuted_store')
    meta=_metadata(module_path=Path('unused.py'),kernel_name=name,arg_names=[],plan=plan,
                   kernel_type='leaf_permuted_store',n1=0,n2=0,dtype='complex64')
    assert meta['batch_per_block']==pack and meta['num_warps']==1
    assert f'batch_id = pid * {pack}' in source
    monkeypatch.setenv('FLAGFFT_IX_3D_PACK','2')
    assert permuted_store_batch_pack_for(plan)==2

@pytest.mark.parametrize('mode,direction',[('packed_c2r','inverse'),('packed_r2c','forward')])
def test_ix_packed_real_boundary_pair_io_is_explicit_fp32_only(ix_profile,monkeypatch,mode,direction):
    from flagfft_codegen.kernels_common import LeafPlan
    from flagfft_codegen.kernels_leaf import _build_leaf_kernel_source_for_io
    monkeypatch.setenv("FLAGFFT_IX_PORTABLE_LEAF","1")
    for dtype in ('complex64','complex128'):
        plan=LeafPlan(128,(16,8),1,16,2,(),128,direction,dtype)
        monkeypatch.delenv('FLAGFFT_IX_VEC_IO',raising=False)
        _,original=_build_leaf_kernel_source_for_io(plan,io_mode=mode)
        monkeypatch.setenv('FLAGFFT_IX_VEC_IO','packed')
        _,packed=_build_leaf_kernel_source_for_io(plan,io_mode=mode)
        compile(packed,'<packed-real-io>','exec')
        if dtype=='complex64':
            assert '_packed_pair_' in packed and '_packed_pair_' not in original
        else:
            assert '_packed_pair_' not in packed

@pytest.mark.parametrize('policy,expected',[('balanced','packed'),('legacy','0')])
def test_single_real_vector_io_default_is_scoped_and_has_rollback(ix_profile,monkeypatch,policy,expected):
    from dataclasses import replace
    from flagfft_codegen.backend_profile import current_profile
    from flagfft_codegen.target import ix_real_single_pack_enabled,set_ix_real_single_pack
    from flagfft_codegen.kernels_common import _maca_knob
    previous=ix_real_single_pack_enabled()
    token=set_profile(replace(current_profile(),policy=policy))
    monkeypatch.delenv('FLAGFFT_IX_VEC_IO',raising=False)
    try:
        set_ix_real_single_pack(False)
        assert _maca_knob('VEC_IO','0')=='0'
        set_ix_real_single_pack(True)
        assert _maca_knob('VEC_IO','0')==expected
        monkeypatch.setenv('FLAGFFT_IX_VEC_IO','0')
        assert _maca_knob('VEC_IO','0')=='0'
    finally:
        set_ix_real_single_pack(previous);reset_profile(token)


@pytest.mark.parametrize('length,factors,mode',[(16,(4,4),'permuted_store'),(64,(4,4,4),'permuted_store_r2c'),(2048,(16,16,8),'bluestein_prepare_leaf'),(2048,(16,16,8),'bluestein_finish_leaf')])
def test_prime_3d_context_matches_explicit_knobs_and_preserves_rollback(ix_balanced_profile,monkeypatch,length,factors,mode):
    from flagfft_codegen.target import ix_prime_3d_enabled,set_ix_prime_3d
    from flagfft_codegen.kernels_common import LeafPlan
    from flagfft_codegen.kernels_leaf import _build_leaf_kernel_source_for_io
    previous=ix_prime_3d_enabled()
    knobs={'WARP_FILL_PACK':'1','BATCH_INTERLEAVE':'1','BLUESTEIN_SINGLE_SMEM':'1'}
    for knob in knobs:monkeypatch.delenv('FLAGFFT_IX_'+knob,raising=False)
    plan=LeafPlan(length,factors,1,128 if length==2048 else 16,2 if length==2048 else 1,(),length,'forward','complex64')
    try:
        set_ix_prime_3d(True)
        name,scoped=_build_leaf_kernel_source_for_io(plan,io_mode=mode,prime_n=997 if length==2048 else 0)
        set_ix_prime_3d(False)
        for knob,value in knobs.items():monkeypatch.setenv('FLAGFFT_IX_'+knob,value)
        explicit_name,explicit=_build_leaf_kernel_source_for_io(plan,io_mode=mode,prime_n=997 if length==2048 else 0)
        assert (name,scoped)==(explicit_name,explicit)
        for knob in knobs:monkeypatch.setenv('FLAGFFT_IX_'+knob,'0')
        set_ix_prime_3d(True)
        _,rolled_back=_build_leaf_kernel_source_for_io(plan,io_mode=mode,prime_n=997 if length==2048 else 0)
        set_ix_prime_3d(False)
        _,legacy=_build_leaf_kernel_source_for_io(plan,io_mode=mode,prime_n=997 if length==2048 else 0)
        assert rolled_back==legacy
    finally:set_ix_prime_3d(previous)

@pytest.mark.parametrize('mode',['four_step_real_row','bluestein_four_step_prepare_row',
    'bluestein_four_step_prepare_row_r2c','bluestein_four_step_prepare_row_c2r',
    'bluestein_four_step_pointwise_row','bluestein_four_step_finish_col',
    'bluestein_four_step_finish_col_r2c','bluestein_four_step_finish_col_c2r'])
@pytest.mark.parametrize('dtype',['complex64','complex128'])
def test_packed_fourstep_boundaries_are_explicit_ix_fp32(ix_balanced_profile,monkeypatch,mode,dtype):
    from flagfft_codegen.kernels_common import LeafPlan
    from flagfft_codegen.kernels_leaf import _build_leaf_kernel_source_for_io
    monkeypatch.setenv('FLAGFFT_IX_PORTABLE_LEAF','1')
    plan=LeafPlan(128,(16,8),1,16,2,(),128,'forward',dtype)
    monkeypatch.setenv('FLAGFFT_IX_VEC_IO','0')
    _,scalar=_build_leaf_kernel_source_for_io(plan,io_mode=mode,prime_n=8191,four_step_n1=128,four_step_n2=128)
    monkeypatch.setenv('FLAGFFT_IX_VEC_IO','packed')
    _,packed=_build_leaf_kernel_source_for_io(plan,io_mode=mode,prime_n=8191,four_step_n1=128,four_step_n2=128)
    compile(packed,'<packed-fourstep-boundary>','exec')
    assert 'tl.pointer_type(tl.uint64)' not in scalar
    assert ('tl.pointer_type(tl.uint64)' in packed)==(dtype=='complex64')


def test_ix_fourstep_request_preset_is_nested_and_explicit_knobs_win(ix_balanced_profile,monkeypatch):
    from flagfft_codegen.target import set_ix_fourstep_preset, reset_ix_fourstep_preset
    from flagfft_codegen.kernels_common import _maca_knob
    monkeypatch.delenv('FLAGFFT_IX_PORTABLE_LEAF',raising=False)
    monkeypatch.delenv('FLAGFFT_IX_TLE_INNER_PACK',raising=False)
    original=_maca_knob('PORTABLE_LEAF','off')
    outer=set_ix_fourstep_preset(1)
    try:
        assert _maca_knob('TLE_INNER_PACK')=='32'
        assert _maca_knob('PORTABLE_LEAF')=='0'
        inner=set_ix_fourstep_preset(2)
        try:
            assert _maca_knob('TLE_INNER_PACK')=='16'
            assert _maca_knob('TLE_IO_LAYOUT')=='0'
        finally:reset_ix_fourstep_preset(inner)
        assert _maca_knob('TLE_INNER_PACK')=='32'
        monkeypatch.setenv('FLAGFFT_IX_TLE_INNER_PACK','8')
        assert _maca_knob('TLE_INNER_PACK')=='8'
        monkeypatch.setenv('FLAGFFT_IX_PORTABLE_LEAF','1')
        assert _maca_knob('PORTABLE_LEAF')=='1'
    finally:reset_ix_fourstep_preset(outer)
    monkeypatch.delenv('FLAGFFT_IX_PORTABLE_LEAF')
    assert _maca_knob('PORTABLE_LEAF','off')==original


def test_ix_row_twiddle_moves_the_kernel_argument_without_backend_assembly(ix_balanced_profile,monkeypatch):
    import ast
    from pathlib import Path
    from flagfft_codegen.kernels_common import LeafPlan, use_four_step_row_fused_twiddle
    from flagfft_codegen.kernels_leaf import _build_leaf_kernel_source_for_io
    from flagfft_codegen.metadata import _metadata
    for key,value in {'PORTABLE_LEAF':'0','ROW_TWIDDLE':'1','TLE_INNER_PACK':'32','WARPS':'4','TLE_IO_LAYOUT':'1'}.items():
        monkeypatch.setenv('FLAGFFT_IX_'+key,value)
    plan=LeafPlan(128,(16,8),1,8,1,(),128,'forward','complex64')
    for mode in ('four_step_row','four_step_col'):
        name,source=_build_leaf_kernel_source_for_io(plan,io_mode=mode,four_step_n1=128,four_step_n2=128)
        args=next(node.args.args for node in ast.parse(source).body if isinstance(node,ast.FunctionDef))
        names=[arg.arg for arg in args]
        assert ('twiddle_ptr' in names)==mode.endswith('_row')
        assert 'inline_asm_elementwise' not in source
        meta=_metadata(module_path=Path('unused.py'),kernel_name=name,arg_names=names,plan=plan,kernel_type=mode,n1=128,n2=128,dtype='complex64')
        assert meta['tle_fused_twiddle']
    assert not use_four_step_row_fused_twiddle(512,1024,'complex64')
    assert not use_four_step_row_fused_twiddle(128,128,'complex128')


@pytest.mark.parametrize('preset,kind,factors,warps',[(7,'leaf_packed_c2r',(8,8,8,16),16),(8,'leaf_packed_r2c',(32,16,16),8)])
def test_real_cta_request_preserves_warps_and_batch_tile_without_env(ix_balanced_profile,monkeypatch,preset,kind,factors,warps):
    from pathlib import Path
    from flagfft_codegen.target import set_ix_fourstep_preset,reset_ix_fourstep_preset
    from flagfft_codegen.kernels_common import LeafPlan
    from flagfft_codegen.kernels_leaf import _build_leaf_kernel_source_for_io
    from flagfft_codegen.metadata import _metadata
    for key in ('WARPS','PORTABLE_LEAF','BATCH_PACK'):
        monkeypatch.delenv('FLAGFFT_IX_'+key,raising=False)
    token=set_ix_fourstep_preset(preset)
    try:
        plan=LeafPlan(8192,factors,1,128,16,(),8192,'inverse' if preset==7 else 'forward','complex64')
        name,source=_build_leaf_kernel_source_for_io(plan,io_mode='packed_c2r' if preset==7 else 'packed_r2c')
        meta=_metadata(module_path=Path('unused.py'),kernel_name=name,arg_names=[],plan=plan,kernel_type=kind,n1=0,n2=0,dtype='complex64')
        assert meta['num_warps']==warps
        assert meta['batch_per_block']==1
    finally:reset_ix_fourstep_preset(token)
