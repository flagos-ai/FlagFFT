# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Share a bounded final butterfly across four public complex FFT batches.

The compact gather map is appended to the final twiddle table by the planner.
Small batches retain the original schedule; no coefficient or floating-point
butterfly arithmetic is approximated.
"""
import math
import re
from .kernels_common import _NATURAL_ORDER_CODELET_RADICES
from .kernels_leaf import (_emit_natural_order_codelet_call,
    _emit_radix16_codelet_call, _emit_local_mixed_codelet_call)

def _source(s, direction, conjugate_output=False):
    bp = 4
    table_offset = 2 * s['physical_n']
    radix, prefix, pp = (s['r'], s['prefix'], s['physical_prefix'])
    padded, initial, pn = (s['padded_prefix'], s['initial_prefix'], s['physical_n'])
    n, groups = (prefix * radix, pp // padded)
    rows = 1
    while rows < 8 and 2 * rows * radix * padded <= 4096:
        rows *= 2
    rows = max(1, rows // bp)
    points, words = (bp * rows * padded, 2 * rows * padded)
    lines = ['@triton.jit', 'def fused_finish(X,TW,Y,NB):', f'    batch=(tl.program_id(1)*{bp}+tl.arange(0,{bp}))[:,None,None]', f'    row=(tl.program_id(0)*{rows}+tl.arange(0,{rows}))[None,:,None]', f'    col=tl.arange(0,{padded})[None,None,:]', f'    off=row*{padded}+col', f'    good=(row<{groups})&(batch<NB)']
    for d in range(radix):
        lines += [f'    r{d}=tl.load(X+batch*{2 * pn}+{d * pp}+off,good,0.)', f'    i{d}=tl.load(X+batch*{2 * pn}+{pn + d * pp}+off,good,0.)']
        if d:
            for v, offset in [('wr', d * pp), ('wi', pn + d * pp)]:
                lines += [f'    {v}=tl.load(TW+{offset}+off,row<{groups},0.)']
            lines += [f'    r{d},i{d}=r{d}*wr-i{d}*wi,r{d}*wi+i{d}*wr']
    mode = direction
    if radix in _NATURAL_ORDER_CODELET_RADICES:
        lines += _emit_natural_order_codelet_call('    ', radix, mode)
    elif radix == 16:
        lines += _emit_radix16_codelet_call('    ', mode)
    else:
        lines += _emit_local_mixed_codelet_call('    ', radix, mode)
    if conjugate_output:
        lines += [f'    i{d}=-i{d}' for d in range(radix)]
    load = f'tl.load(TW+{table_offset}+tl.arange(0,{words}))'
    lines += [f'    word=tl.arange(0,{words})[None,:]', f'    local_batch=tl.arange(0,{bp})[:,None]', f'    take={load}', f'    take=tl.reshape(take,(1,{words})).to(tl.int32)+local_batch*{rows * padded}', f'    valid_word=word<{2 * rows * initial}', f'    out=tl.program_id(0)*{2 * rows * initial}+word', f'    out_batch=tl.program_id(1)*{bp}+local_batch']
    for d in range(radix):
        lines += [f'    value=tl.zeros(({2 * points},),tl.float32)', f'    value=al.insert_slice(value,tl.reshape(r{d},({points},)),[0],[{points}],[1])', f'    value=al.insert_slice(value,tl.reshape(i{d},({points},)),[{points}],[{points}],[1])', f'    packed=tl.gather(value,tl.reshape(take,({bp * words},)),0)', f'    tl.store(Y+out_batch*{2 * n}+{2 * d * prefix}+out,tl.reshape(packed,({bp},{words})),valid_word&(out<{2 * prefix})&(out_batch<NB))']
    spec = dict(s, batch_pack=bp, grid=(math.ceil(groups / rows),), finish_map=True, finish_rows=rows)
    return ('\n'.join(lines) + '\n', spec)


def replace_finish(source, specs, direction, conjugate_output=False):
    s=specs[-1]
    r,padded,initial=s['r'],s['padded_prefix'],s['initial_prefix']
    if (s['name']!='fused_finish' or s.get('coprime_prefix') or r<8 or r&(r-1)
            or initial*100<padded*99 or padded<128 or 8*r*padded>16384):
        raise ValueError('Batch finish requires a dense bounded power-of-two suffix')
    body,last=_source(s,direction,conjugate_output)
    # Promote the batch address before multiplying by a potentially large
    # plane pitch; the public buffer size does not bound addresses to i32.
    body=body.replace('batch=(tl.program_id(1)*4+tl.arange(0,4))[:,None,None]',
                      'batch=(tl.program_id(1).to(tl.int64)*4+tl.arange(0,4))[:,None,None]')
    body=body.replace('out_batch=tl.program_id(1)*4+local_batch',
                      'out_batch=tl.program_id(1).to(tl.int64)*4+local_batch')
    source,count=re.subn(r'@triton.jit\ndef fused_finish\([^\n]*\):\n[\s\S]*?(?=\n@triton.jit|\Z)',lambda _:body,source)
    if count!=1:raise ValueError('Missing compact DIF finish')
    return source,specs[:-1]+[last]
