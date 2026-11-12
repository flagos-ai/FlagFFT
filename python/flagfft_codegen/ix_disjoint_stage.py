"""Narrow CoreX postpass for disjoint writes in generated FFT stages.

The generator opts in only bounded FourStep leaves with one butterfly group
per stage. Their intermediate stores form a permutation of
independent (FFT slot, frequency) addresses. Preserve all read/write fences;
only adjacent write-only intervals can lose an implicit write/write fence.
"""
from functools import lru_cache
import hashlib
from pathlib import Path
import re
import sys

_REGISTERED = {}
_INSTALLED = False
_BARRIER = re.compile(r"\bcall\b.*@llvm\.nvvm\.barrier\.cta\.sync\.aligned\.all\(")
_SHARED_READ = re.compile(r"\bload\b.*addrspace\(3\)")
_SHARED_WRITE = re.compile(r"\bstore\b.*addrspace\(3\)")
_LABEL = re.compile(r"^([\w.$-]+):")


@lru_cache(maxsize=1)
def pass_tag() -> str:
    # Compilation caches must change whenever the postpass implementation does.
    return "ixdisjoint_" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


def _acyclic(body: list[str]) -> bool:
    edges = {"entry": set()}
    current = "entry"
    for line in body:
        if match := _LABEL.match(line):
            current = match[1]
            edges.setdefault(current,set())
        edges[current].update(re.findall(r"label %([\w.$-]+)",line))
    visiting, finished = set(), set()
    def visit(node):
        if node in visiting: return False
        if node in finished: return True
        visiting.add(node)
        if any(not visit(child) for child in edges.get(node,())): return False
        visiting.remove(node)
        finished.add(node)
        return True
    return all(visit(node) for node in edges)


def rewrite_disjoint_writes(llir: str, store_lines: set[int],
                            explicit_fence_lines: set[int]) -> tuple[str,int]:
    """Conservatively remove source-identified WAW fences in an acyclic FFT."""
    lines=llir.splitlines(keepends=True)
    entries=[i for i,line in enumerate(lines) if re.match(r"^define\s+iluvatar_kernel\s+void\s+@",line)]
    if len(entries)!=1: return llir,0
    start=entries[0]
    end=next((i for i in range(start+1,len(lines)) if lines[i].strip()=="}"),None)
    if end is None: return llir,0
    body=lines[start+1:end]
    if not _acyclic(body): return llir,0
    # Hidden shared accesses and atomic/fence operations require a different
    # dependence proof. This pass never handles those kernels.
    for line in body:
        if re.search(r"\b(?:atomicrmw|cmpxchg|fence)\b",line): return llir,0
        if "call " in line and ("addrspace(3)" in line or
                                 re.search(r"(?:\basm\b|ldmatrix|cp\.async|memcpy|memmove)",line)):
            return llir,0
    locations={int(k):int(v) for k,v in re.findall(
        r"!(\d+)\s*=\s*(?:distinct\s+)?!DILocation\(line:\s*(\d+)",llir)}
    barriers=[i for i in range(start+1,end) if _BARRIER.search(lines[i])]
    bounds=[start]+barriers+[end]
    def write_only(lo,hi):
        segment=lines[lo+1:hi]
        return (any(_SHARED_WRITE.search(line) for line in segment) and
                not any(_SHARED_READ.search(line) for line in segment))
    removed=set()
    for j,index in enumerate(barriers,1):
        debug=re.search(r"!dbg !(\d+)",lines[index])
        source_line=locations.get(int(debug[1])) if debug else None
        if source_line not in store_lines or source_line in explicit_fence_lines: continue
        if write_only(bounds[j-1],index) and write_only(index,bounds[j+1]):
            removed.add(index)
    return ''.join(line for i,line in enumerate(lines) if i not in removed),len(removed)


def register_kernel(fn) -> None:
    """Install through the pinned SDK's public stage hook, without SDK edits."""
    global _INSTALLED
    name=fn.fn.__name__
    if not name.endswith(pass_tag()):
        raise ValueError("IX disjoint-stage kernel lacks the current pass identity")
    source=Path(fn.fn.__code__.co_filename).read_text().splitlines()
    stores={i for i,line in enumerate(source,1)
            if re.search(r"tl\.store\(tle\.gpu\.local_ptr\(smem_[ab]_[ri],",line)}
    explicit={i for i,line in enumerate(source,1) if "tl.debug_barrier()" in line}
    if not stores or not explicit:
        raise ValueError("IX disjoint-stage kernel lacks its shared-store/fence contract")
    contract=(stores,explicit)
    if name in _REGISTERED and _REGISTERED[name]!=contract:
        # An ambiguous source location must preserve the original fences.
        _REGISTERED[name]=None
    else:
        _REGISTERED[name]=contract
    if _INSTALLED: return
    from triton import knobs
    previous=knobs.runtime.add_stages_inspection_hook
    def hook(backend,stages,options,language,capability):
        if previous is not None: previous(backend,stages,options,language,capability)
        if (backend.target.backend!='corex' or backend.target.arch!=71 or
                backend.target.warp_size!=64 or
                getattr(options,'instrumentation_mode','') or
                knobs.compilation.instrumentation_mode): return
        original=stages['llir']
        def lower(src,metadata):
            llir=original(src,metadata)
            match=re.search(r"^define\s+iluvatar_kernel\s+void\s+@([\w.$-]+)",llir,re.M)
            contract=_REGISTERED.get(match[1]) if match else None
            if contract is None: return llir
            result,removed=rewrite_disjoint_writes(llir,*contract)
            metadata['flagfft_ix_disjoint_fences_removed']=removed
            if removed:
                print(f"[ix-disjoint-stage] kernel={match[1]} removed={removed}",file=sys.stderr)
            return result
        stages['llir']=lower
    knobs.runtime.add_stages_inspection_hook=hook
    _INSTALLED=True
