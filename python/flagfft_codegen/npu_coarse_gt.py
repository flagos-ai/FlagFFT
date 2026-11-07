# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Coprime 4/8 by 15 prefix with its permutation retained in the DIF layout."""
import math
from .kernels_leaf import _emit_natural_order_codelet_call


def prefix_source(factors, direction, global_inverse):
    a,b=factors[:2];initial=a*b;padded=a*16;n=math.prod(factors)
    remaining=n//initial;c=128 if a==4 else 64;state=padded*c
    alpha=b*pow(b,-1,a);beta=a*pow(a,-1,b)
    lines=['@triton.jit','def coarse_prefix(X,TW,Y):',
        f'    base=tl.program_id(0)*{c}',
        '    batch=tl.program_id(1).to(tl.int64)',
        f'    word=base*2+tl.arange(0,{2*c})[None,:]',
        '    t=tl.arange(0,16)[:,None]']
    for d in range(a):
        lines += [f'    row=({alpha*d}+{beta}*t)%{initial}',
            f'    raw=tl.load(X+batch*{2*n}+row*{2*remaining}+word,(t<15)&(word<{2*remaining}),0.)',
            f'    r{d},i{d}=tl.split(tl.reshape(raw,({16*c},2)))']
        if direction=='inverse':lines += [f'    i{d}=-i{d}']
    lines += _emit_natural_order_codelet_call('    ',a,'forward')
    lines += [f'    buf_r=tl.zeros(({state},),tl.float32)',
              f'    buf_i=tl.zeros(({state},),tl.float32)']
    for d in range(a):
        lines += [f'    buf_r=al.insert_slice(buf_r,r{d},[{d*16*c}],[{16*c}],[1])',
                  f'    buf_i=al.insert_slice(buf_i,i{d},[{d*16*c}],[{16*c}],[1])']
    for d in range(b):
        lines += [f'    r{d}=tl.reshape(al.extract_slice(tl.reshape(buf_r,({a},{16*c})),[0,{d*c}],[{a},{c}],[1,1]),({a*c},))',
                  f'    i{d}=tl.reshape(al.extract_slice(tl.reshape(buf_i,({a},{16*c})),[0,{d*c}],[{a},{c}],[1,1]),({a*c},))']
    lines += _emit_natural_order_codelet_call('    ',b,'forward')
    lines += [f'    re=tl.zeros(({state},),tl.float32)',f'    im=tl.zeros(({state},),tl.float32)']
    for d in range(b):
        lines += [f'    re=al.insert_slice(re,r{d},[{d*a*c}],[{a*c}],[1])',
                  f'    im=al.insert_slice(im,i{d},[{d*a*c}],[{a*c}],[1])']
    if direction=='inverse' and not global_inverse:lines += ['    im=-im']
    physical_n=remaining*padded
    lines += [f'    re=tl.trans(tl.reshape(re,({padded},{c})))',
              f'    im=tl.trans(tl.reshape(im,({padded},{c})))',
              f'    sample=base+tl.arange(0,{c})[:,None]',
              f'    freq=tl.arange(0,{padded})[None,:]',f'    out=sample*{padded}+freq',
              f'    tl.store(Y+batch*{2*physical_n}+out,re,sample<{remaining})',
              f'    tl.store(Y+batch*{2*physical_n}+{physical_n}+out,im,sample<{remaining})']
    return '\n'.join(lines)+'\n'


def finish_take(initial,padded,points,single_gather):
    a=initial//15;b=15;k=f'((word//2)%{initial})'
    local=f'(({pow(a,-1,b)}*{k})%{b})*{a}+(({pow(b,-1,a)}*{k})%{a})'
    offset=f'(word//{2*initial})*{padded}+{local}'
    return (f'(word&1)*{points}+{offset}' if single_gather else
            f'2*({offset})+(word&1)')


from .npu_radix8 import radix8_source
