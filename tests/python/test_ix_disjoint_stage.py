"""Dependence boundaries for the narrow, optional CoreX FFT postpass."""
from flagfft_codegen.ix_disjoint_stage import rewrite_disjoint_writes
import math


def test_mixed_stage_writes_are_bijections_before_fence_removal():
    from flagfft_codegen.kernels_leaf import _emit_route_base, _emit_route_index
    class ScalarLanguage:
        @staticmethod
        def where(condition, yes, no):
            return yes if condition else no
    for factors in ((17,13),(25,17),(12,13,5),(30,10,5),(24,13,5)):
        n=math.prod(factors)
        # Include the real cooperative cap: later stages can have multiple
        # groups even though all stores together must remain a permutation.
        widths=((50,)*3 if factors==(30,10,5) else
                tuple(next(w for w in range(min(n//r,128),0,-1) if (n//r)%w==0)
                      for r in factors))
        for stage in range(len(factors)-1):
            addresses=[]
            for group in range(n//(widths[stage]*factors[stage])):
                for lane in range(widths[stage]):
                    env={'tl':ScalarLanguage,'lane_mask':True,'lane':lane,'group':group}
                    exec('\n'.join(_emit_route_base('',stage,factors,widths[stage],'group')),env)
                    for digit in range(factors[stage]):
                        exec('\n'.join(_emit_route_index('','dst',stage,factors,widths[stage+1],digit)),env)
                        addresses.append(env['dst'])
            assert sorted(addresses)==list(range(n))


def ir(middle='',tail='',debug=True):
    return '''define iluvatar_kernel void @test() {
entry:
  store float %r0, ptr addrspace(3) %p0
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0), !dbg !10
'''+middle+'''  store float %r1, ptr addrspace(3) %p1
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0), !dbg !11
  %r2 = load float, ptr addrspace(3) %p2
'''+tail+'''  ret void
}
'''+('!10 = !DILocation(line: 20, column: 1, scope: !1)\n!11 = !DILocation(line: 30, column: 1, scope: !1)\n' if debug else '')


def test_only_disjoint_write_write_fence_is_removed():
    source=ir()
    result,count=rewrite_disjoint_writes(source,{20},{30})
    assert count==1
    assert 'i32 0), !dbg !10' not in result
    assert 'i32 0), !dbg !11' in result
    assert 'load float, ptr addrspace(3)' in result


def test_layout_exchange_reads_preserve_implicit_fences():
    source=ir(middle='  %scratch = load float, ptr addrspace(3) %tmp\n')
    assert rewrite_disjoint_writes(source,{20},{30})==(source,0)


def test_explicit_fences_and_missing_source_locations_are_preserved():
    for source,stores,explicit in [(ir(),{20},{20,30}),(ir(debug=False),{20},{30}),(ir(),{99},{30})]:
        assert rewrite_disjoint_writes(source,stores,explicit)==(source,0)


def test_loop_and_hidden_shared_accesses_are_not_supported():
    loop=ir(tail='  br i1 %again, label %entry, label %exit\nexit:\n')
    hidden=ir(middle='  call void @shared(ptr addrspace(3) %tmp)\n')
    atomic=ir(middle='  %old = atomicrmw add ptr %p, i32 1 monotonic\n')
    assembly=ir(middle='  call void asm sideeffect "st.shared.f32 [$0], $1;", "r,f"(i32 %addr, float %r)\n')
    opaque=ir(middle='  call void @instrumented_kernel_callback()\n')
    intrinsic=ir(middle='  call void @llvm.bi.shared.read(i32 %addr)\n')
    for source in (loop,hidden,atomic,assembly,opaque,intrinsic):
        assert rewrite_disjoint_writes(source,{20},{30})==(source,0)


def _registered_stub(tmp_path,monkeypatch,runtime,compilation):
    import sys
    from types import SimpleNamespace
    import flagfft_codegen.ix_disjoint_stage as stage
    path=tmp_path/'generated.py'
    path.write_text('tl.store(tle.gpu.local_ptr(smem_b_r, (x,)), value)\ntl.debug_barrier()\n')
    fn=SimpleNamespace(fn=SimpleNamespace(__name__='test_'+stage.pass_tag(),
        __code__=SimpleNamespace(co_filename=str(path))))
    monkeypatch.setattr(stage,'_REGISTERED',{})
    monkeypatch.setattr(stage,'_INSTALLED',False)
    monkeypatch.setitem(sys.modules,'triton',SimpleNamespace(
        knobs=SimpleNamespace(runtime=runtime,compilation=compilation)))
    stage.register_kernel(fn)
    return stage


def test_missing_sdk_stage_hook_preserves_original_compilation(tmp_path,monkeypatch):
    from types import SimpleNamespace
    runtime=SimpleNamespace()
    stage=_registered_stub(tmp_path,monkeypatch,runtime,SimpleNamespace())
    assert not stage._INSTALLED
    assert not hasattr(runtime,'add_stages_inspection_hook')


def test_instrumented_compilation_keeps_the_sdk_lowering_stage(tmp_path,monkeypatch):
    from types import SimpleNamespace
    runtime=SimpleNamespace(add_stages_inspection_hook=None)
    _registered_stub(tmp_path,monkeypatch,runtime,SimpleNamespace(instrumentation_mode=''))
    original=object();stages={'llir':original}
    backend=SimpleNamespace(target=SimpleNamespace(backend='corex',arch=71,warp_size=64))
    runtime.add_stages_inspection_hook(backend,stages,SimpleNamespace(instrumentation_mode='consan'),None,None)
    assert stages['llir'] is original
