# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Fuse bounded Stockham transforms with runtime local-index tables and TLE."""
import json
import math
import re
from pathlib import Path
from .artifacts import write_text_atomic
from .metadata import _module_source
from .kernels_leaf import _emit_natural_order_codelet_call, _emit_radix16_codelet_call
SUPPORTED_RADICES = frozenset((2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,19))

def _power2(n):
    return 1 << (n - 1).bit_length()

def fused_stockham_layout(n, factors, group):
    if (not 1 < n <= 256 or not factors or math.prod(factors) != n
            or any(r not in SUPPORTED_RADICES for r in factors)
            or group not in (1,2,4,8,16,32,64,128) or n*group > 4096):
        raise ValueError("Unsupported bounded NPU Stockham layout")
    size=_power2(n*group)
    indices=[];seen={}
    def add(values):
        key=tuple(values)
        if key not in seen:
            seen[key]=len(indices);indices.extend(key)
        return seen[key]
    incoming=(add(2*k for k in range(size)),add(2*k+1 for k in range(size)))
    stages=[];span=1;tw_base=0
    for radix in factors:
        nd=n//radix;bp=_power2(group*nd);qp=_power2(radix)
        source=[];twiddle=[]
        for digit in range(radix):
            source.append(add(min(k//nd,group-1)*n+k%nd+digit*nd for k in range(bp)))
            if digit and span>1:
                real=add(tw_base+2*(digit*span+(k%nd)%span) for k in range(bp))
                imag=add(tw_base+2*(digit*span+(k%nd)%span)+1 for k in range(bp))
                twiddle.append((real,imag))
            else:twiddle.append(None)
        outgoing=add((((k%n)//span)%radix)*bp+min(k//n,group-1)*nd
            +((k%n)//(radix*span))*span+(k%n)%span for k in range(size))
        stages.append(dict(radix=radix,span=span,bp=bp,qp=qp,source=source,
            twiddle=twiddle,outgoing=outgoing))
        if span>1:tw_base+=2*radix*span
        span*=radix
    packing=add((k>>1)+(k&1)*size for k in range(2*size))
    return dict(size=size,incoming=incoming,stages=stages,packing=packing,
        indices=indices,twiddle_count=tw_base)

def build_npu_fused_stockham_source(n,factors,group,direction):
    if direction not in ("forward","inverse"):
        raise ValueError("Invalid Stockham direction")
    p=fused_stockham_layout(n,factors,group)
    size=p["size"];tc=p["twiddle_count"];tp=_power2(max(tc,1))
    name=f'npu_fused_stockham_n{n}_g{group}_{direction}_f'+'_'.join(map(str,factors))
    lines=['@triton.jit',f'def {name}(in_ptr,out_ptr,idx_ptr,twiddle_ptr,nbatch):',
        '    pid=tl.program_id(0).to(tl.int64)',
        f'    lane=tl.arange(0,{size})',f'    scalar=tl.arange(0,{2*size})',
        f'    twl=tl.arange(0,{tp})',
        f'    twiddles=tl.load(twiddle_ptr+twl,twl<{tc},0.0)',
        f'    for group_base in range(pid*{group},nbatch,tl.num_programs(0).to(tl.int64)*{group}):',
        '        base=group_base.to(tl.int64)',
        f'        valid=tl.minimum(tl.maximum(nbatch-group_base,0),{group}).to(tl.int32)',
        f'        raw=tl.load(in_ptr+base*{2*n}+scalar,scalar<valid*{2*n},0.0)',
        f'        real=tl.gather(raw,tl.load(idx_ptr+{p["incoming"][0]}+lane),0)',
        f'        imag=tl.gather(raw,tl.load(idx_ptr+{p["incoming"][1]}+lane),0)']
    for stage in p["stages"]:
        r,bp,qp,span=[stage[k] for k in ("radix","bp","qp","span")]
        lines += [f'        bf=tl.arange(0,{bp})']
        for d in range(r):
            at=stage["source"][d]
            lines += [f'        input_index=tl.load(idx_ptr+{at}+bf)',
                f'        r{d}=tl.where(bf<{group*(n//r)},tl.gather(real,input_index,0),0.0)',
                f'        i{d}=tl.where(bf<{group*(n//r)},tl.gather(imag,input_index,0),0.0)']
            if stage["twiddle"][d] is not None:
                wr,wi=stage["twiddle"][d]
                lines += [f'        wr=tl.gather(twiddles,tl.load(idx_ptr+{wr}+bf),0)',
                    f'        wi=tl.gather(twiddles,tl.load(idx_ptr+{wi}+bf),0)',
                    f'        r{d},i{d}=_cmul(r{d},i{d},wr,wi)']
        lines += (_emit_radix16_codelet_call('        ',direction) if r==16 else
            _emit_natural_order_codelet_call('        ',r,direction))
        lines += [f'        stage_r=tl.zeros(({qp*bp},),tl.float32)',
            f'        stage_i=tl.zeros(({qp*bp},),tl.float32)']
        for d in range(r):
            lines += [f'        stage_r=tle.dsa.insert_slice(stage_r,r{d},[{d*bp}],[{bp}],[1])',
                f'        stage_i=tle.dsa.insert_slice(stage_i,i{d},[{d*bp}],[{bp}],[1])']
        at=stage["outgoing"]
        lines += [f'        mapping=tl.load(idx_ptr+{at}+lane)',
            f'        real=tl.where(lane<{n*group},tl.gather(stage_r,mapping,0),0.0)',
            f'        imag=tl.where(lane<{n*group},tl.gather(stage_i,mapping,0),0.0)']
    lines += [f'        packed=tl.zeros(({2*size},),tl.float32)',
        f'        packed=tle.dsa.insert_slice(packed,real,[0],[{size}],[1])',
        f'        packed=tle.dsa.insert_slice(packed,imag,[{size}],[{size}],[1])',
        f'        values=tl.gather(packed,tl.load(idx_ptr+{p["packing"]}+scalar),0)',
        f'        tl.store(out_ptr+base*{2*n}+scalar,values,scalar<valid*{2*n})']
    # Give every stage local names. Its butterfly width may change between
    # stages, and Triton must not infer a loop-carried value with two shapes.
    body='\n'.join(lines)+'\n'
    chunks=body.split('        bf=')[1:]
    prefix=body.split('        bf=')[0]
    for i,chunk in enumerate(chunks):
        part='        bf='+chunk
        # Only block-local assignments/uses, excluding invariant lane/scalar,
        # the input/output pointers, real/imag state and final packing.
        if i==len(chunks)-1:
            pos=part.index('        packed=')
            tail=part[pos:];part=part[:pos]
        else:tail=''
        names={'bf','input_index','wr','wi','stage_r','stage_i','mapping'}
        names.update(f'{c}{d}' for d in range(p["stages"][i]["radix"]) for c in 'ri')
        part=re.sub(r'\b('+'|'.join(sorted(names,key=len,reverse=True))+r')\b',
            lambda m:f'{m[0]}_s{i}',part)
        prefix+=part+tail
    return name,prefix

def emit_npu_fused_stockham(*,length,factors,group,direction,out_dir):
    name,source=build_npu_fused_stockham_source(length,factors,group,direction)
    out_dir.mkdir(parents=True,exist_ok=True)
    path=out_dir/f'flagfft_{name}.py'
    dependencies=tuple(sorted(set((2,3,4,5,7,8,9,11,*factors))))
    write_text_atomic(path,_module_source(source,dependencies))
    metadata=dict(module_path=str(path),kernel_name=name,
        signature="*fp32:16,*fp32:16,*u32:16,*fp32:16,i32",
        arg_names=["in_ptr","out_ptr","idx_ptr","twiddle_ptr","nbatch"],
        num_warps=4,num_stages=1,batch_per_block=group,
        kernel_type="npu_fused_stockham",dtype="complex64",direction=direction,
        length=length,factors=factors)
    write_text_atomic(path.with_suffix(".json"),json.dumps(metadata,sort_keys=True))
    return metadata
