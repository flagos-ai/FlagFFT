"""Dependence boundaries for the narrow, optional CoreX FFT postpass."""
from flagfft_codegen.ix_disjoint_stage import rewrite_disjoint_writes


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
    for source in (loop,hidden,atomic):
        assert rewrite_disjoint_writes(source,{20},{30})==(source,0)
