# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded mixed-radix axes with affine matrix I/O and compact UB exchange."""
import json
import math
from pathlib import Path
from .artifacts import write_text_atomic
from .kernels_common import codelet_radices_for, _NATURAL_ORDER_CODELET_RADICES
from .kernels_leaf import (_emit_natural_order_codelet_call,
    _emit_local_mixed_codelet_call, _emit_radix16_codelet_call)
from .metadata import _module_source


def padded(n):
    return 1 << (n-1).bit_length()


def mixed_axis_layout(factors):
    factors = tuple(factors)
    if len(factors) not in (2,3) or any(n < 2 for n in factors):
        raise ValueError('Mixed axis requires two or three nontrivial codelets')
    pads = tuple(map(padded, factors))
    size = math.prod(pads)
    if size > 4096:
        raise ValueError('Mixed axis exceeds its padded tensor budget')
    pack = min(8, 4096//size)
    return pads, pack


def build_mixed_axis(factors, width, first, direction, *, fma=False):
    factors=tuple(factors)
    pads, pack=mixed_axis_layout(factors)
    length=math.prod(factors); count=math.prod(pads)*pack
    if direction not in ('forward','inverse') or width < 1:
        raise ValueError('Invalid mixed axis direction or matrix width')
    inverse=direction=='inverse'
    name=f'npu_mixed_axis_{length}_{width}_c{pack}_{int(first)}_{direction}'+('_fma' if fma else '')
    args=['X','TW','Y','R','NB'] if first else ['X','Y','R','NB']
    lines=['@triton.jit',f'def {name}({",".join(args)}):',
        f'    base=tl.program_id(0)*{pack}', '    batch=tl.program_id(1)',
        f'    word=base*2+tl.arange(0,{2*pack})[None,:]']
    root_offset=0; order=list(range(len(factors)))
    for stage, radix in enumerate(factors):
        others=[a for a in order if a!=stage]
        plen=math.prod(pads[a] for a in others)*pack
        if stage==0:
            tail=math.prod(factors[1:]); pt=padded(tail); plen=pt*pack
            lines += [f'    t=tl.arange(0,{pt})[:,None]']
            for d in range(radix):
                lines += [f'    v{d}=tl.load(X+batch*{2*length*width}+(t+{d*tail})*{2*width}+word,(t<{tail})&(word<{2*width}),0.)',
                    f'    r{d},i{d}=tl.split(tl.reshape(v{d},({plen},2)))']
                if inverse:lines += [f'    i{d}=-i{d}']
        else:
            lines += [f'    q=tl.arange(0,{plen})']
            coords={a:f'((q//{math.prod(pads[b] for b in others[k+1:])*pack})&{pads[a]-1})' for k,a in enumerate(others)}
            stride={a:math.prod(pads[b] for b in order[k+1:])*pack for k,a in enumerate(order)}
            index='+'.join([f'({coords[a]})*{stride[a]}' for a in others]+[f'(q&{pack-1})'])
            for d in range(radix):
                lines += [f'    idx{d}={index}+{d*stride[stage]}',
                    f'    r{d}=tl.gather(buf_r,idx{d},0)',f'    i{d}=tl.gather(buf_i,idx{d},0)']
        if radix in _NATURAL_ORDER_CODELET_RADICES:
            lines += _emit_natural_order_codelet_call('    ',radix,'forward')
        elif radix==16:
            lines += _emit_radix16_codelet_call('    ','forward')
        else:
            lines += _emit_local_mixed_codelet_call('    ',radix,'forward')
        if stage < len(factors)-1:
            later=math.prod(pads[stage+1:]) if stage else padded(math.prod(factors[1:]))
            imag=root_offset+radix*later
            lines += [f'    q=tl.arange(0,{plen})',f'    tc=tl.arange(0,{later})']
            for d in range(radix):
                lines += [f'    wr=tl.load(R+{root_offset+d*later}+tc)',f'    wi=tl.load(R+{imag+d*later}+tc)',
                    f'    wr=tl.gather(wr,(q//{pack})&{later-1},0)',f'    wi=tl.gather(wi,(q//{pack})&{later-1},0)',
                    f'    r{d},i{d}=r{d}*wr-i{d}*wi,r{d}*wi+i{d}*wr']
            root_offset += 2*radix*later
            lines += [f'    next_r=tl.zeros(({count},),tl.float32)',f'    next_i=tl.zeros(({count},),tl.float32)']
            block=math.prod(pads[a] for a in others)*pack
            if stage==0 and len(factors)>2:
                lines += [f'    z=tl.arange(0,{block})']
                cs=[f'((z//{math.prod(pads[b] for b in others[k+1:])*pack})&{pads[a]-1})' for k,a in enumerate(others)]
                valid='&'.join(f'({c}<{factors[a]})' for a,c in zip(others,cs))
                idx='+'.join([f'({c})*{math.prod(factors[b] for b in range(a+1,len(factors)))*pack}' for a,c in zip(others,cs)]+[f'(z&{pack-1})'])
                lines += [f'    good={valid}',f'    embed=tl.where(good,{idx},0)']
                for d in range(radix):
                    lines += [f'    r{d}=tl.where(good,tl.gather(r{d},embed,0),0.)',f'    i{d}=tl.where(good,tl.gather(i{d},embed,0),0.)']
            for d in range(radix):
                lines += [f'    next_r=al.insert_slice(next_r,r{d},[{d*block}],[{block}],[1])',
                    f'    next_i=al.insert_slice(next_i,i{d},[{d*block}],[{block}],[1])']
            lines += ['    buf_r=next_r','    buf_i=next_i'];order=[stage]+others
        else:
            prefix=math.prod(factors[:-1]); pp=padded(prefix); pc=pp*pack
            lines += [f'    q=tl.arange(0,{pc})',f'    freq=q//{pack}']
            coords={a:f'((freq//{math.prod(factors[:a])})%{factors[a]})' for a in others}
            index='+'.join([f'({coords[a]})*{math.prod(pads[b] for b in others[k+1:])*pack}' for k,a in enumerate(others)]+[f'(q&{pack-1})'])
            lines += [f'    outidx={index}',f'    row=tl.arange(0,{pp})[:,None]',f'    col=base+tl.arange(0,{pack})[None,:]']
            for d in range(radix):
                lines += [f'    pr=tl.gather(r{d},outidx,0)',f'    pi=tl.gather(i{d},outidx,0)']
                if inverse:lines += ['    pi=-pi']
                if first:
                    lines += [f'    cr=tl.load(TW+(row+{d*prefix})*{width}+col,(row<{prefix})&(col<{width}),0.)',
                        f'    ci=tl.load(TW+{length*width}+(row+{d*prefix})*{width}+col,(row<{prefix})&(col<{width}),0.)',
                        f'    cr=tl.reshape(cr,({pc},))',f'    ci=tl.reshape(ci,({pc},))',
                        '    pr,pi=pr*cr-pi*ci,pr*ci+pi*cr',f'    tr=(q%{pp})*{pack}+q//{pp}',
                        '    pr=tl.gather(pr,tr,0)','    pi=tl.gather(pi,tr,0)',
                        f'    v=tl.reshape(tl.join(pr,pi),({pack},{2*pp}))',
                        f'    orow=base+tl.arange(0,{pack})[:,None]',f'    ow=tl.arange(0,{2*pp})[None,:]',
                        f'    tl.store(Y+batch*{2*length*width}+orow*{2*length}+ow+{2*d*prefix},v,(orow<{width})&(ow<{2*prefix}))']
                else:
                    lines += [f'    v=tl.reshape(tl.join(pr,pi),({pp},{2*pack}))',
                        f'    tl.store(Y+batch*{2*length*width}+(row+{d*prefix})*{2*width}+word,v,(row<{prefix})&(word<{2*width}))']
    source=_module_source('\n'.join(lines)+'\n',tuple(sorted(codelet_radices_for(factors))),fma_helpers=fma)
    return name,source,args,pack


def emit_mixed_axis(factors,n1,n2,first,direction,out_dir,*,fma=False):
    length,width=(n1,n2) if first else (n2,n1)
    if math.prod(factors)!=length:raise ValueError('Mixed axis factors disagree with plan')
    name,source,args,pack=build_mixed_axis(factors,width,first,direction,fma=fma)
    path=Path(out_dir)/(name+'.py');path.parent.mkdir(parents=True,exist_ok=True)
    write_text_atomic(path,source)
    meta=dict(module_path=str(path),kernel_name=name,arg_names=args,
        signature=','.join(['*fp32:16']*(len(args)-1)+['i32']),num_warps=8,
        num_stages=1,batch_per_block=1,inner_pack=pack,tle_fused_twiddle=True,
        kernel_type='four_step_row' if first else 'four_step_col',
        dtype='complex64',direction=direction,length=length)
    write_text_atomic(path.with_suffix('.json'),json.dumps(meta,sort_keys=True))
    return meta
