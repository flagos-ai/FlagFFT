# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Combine a nearly full 32-point suffix with its phase in IEEE FP32 Cube."""
import copy

from .npu_dense_late_pair import _replace_function
from .npu_matrix_batch import batch_matrix_source


def replace_terminal_pair(source, specs, conjugate_output=False):
    specs = copy.deepcopy(specs)
    if len(specs) < 3:
        raise ValueError('Terminal pair requires a bounded natural-prefix suffix')
    first, last = specs[-2:]
    radix = first['r'] * last['r']
    initial, padded = first.get('initial_prefix'), first.get('padded_prefix')
    if (initial != 126 or padded != 128 or not 28 < radix <= 32
            or first.get('deferred_twiddle') or last['m'] != 1
            or first['m'] != last['r'] or last['name'] != 'fused_finish'):
        raise ValueError('Terminal pair requires a near-full 32-point natural suffix')
    prefix, physical = first['prefix'], first['physical_n']
    physical_prefix = first['physical_prefix']
    previous_physical = first['previous_prefix'] // initial * padded
    if not previous_physical or previous_physical % 128 or physical_prefix % 128:
        raise ValueError('Terminal phase digit must span complete 128-column tiles')
    n = prefix * radix
    name = 'terminal_pair_matrix_batch'
    lines = [
        '@triton.jit', f'def {name}(X,TW,Y):',
        '    batch=tl.program_id(1).to(tl.int64)',
        '    tile=tl.program_id(0)',
        f'    kprev=tile//{previous_physical//128}',
        '    col=tile*128+tl.arange(0,128)[None,:]',
        '    d=tl.arange(0,32)[:,None]',
        f'    off=d*{physical_prefix}+col',
        f'    xr=tl.load(X+batch*{2*physical}+off,d<{radix},0.)',
        f'    xi=tl.load(X+batch*{2*physical}+{physical}+off,d<{radix},0.)',
        '    row=tl.arange(0,64)[:,None]',
        '    k=tl.arange(0,32)[None,:]',
        '    basis=TW+kprev*4096+row*64+k',
        '    a=tl.load(basis)', '    b=tl.load(basis+32)',
        '    value=tl.dot(a,xr,input_precision="ieee")',
        '    value=tl.dot(b,xi,acc=value,input_precision="ieee")',
        '    word=tl.arange(0,256)[None,:]',
    ]
    for start in (0, 16):
        lines += [
            f'    re=al.extract_slice(value,[{start},0],[16,128],[1,1])',
            f'    im=al.extract_slice(value,[{32+start},0],[16,128],[1,1])',
        ]
        if conjugate_output:
            lines += ['    im=-im']
        lines += [
            '    out=tl.reshape(tl.join(re,im),(16,256))',
            f'    freq=(tl.arange(0,16)+{start})[:,None]',
            # An explicit rectangular bound lowers to the valid destination
            # slice. A word//2 predicate can lower to an unsafe read/modify/
            # write of neighboring programs' masked-off output elements.
            f'    tl.store(Y+batch*{2*n}+freq*{2*prefix}+tile*252+word,out,(freq<{radix})&(word<252))',
        ]
    body = batch_matrix_source('\n'.join(lines)+'\n', 8)
    source = _replace_function(source, first['name'], body)
    source = _replace_function(source, last['name'], '')
    first.update(name=name, r=radix, m=1, grid=(physical_prefix//128,),
                 batch_pack=8, merged_terminal_pair=True)
    return source, specs[:-1]
