# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Fuse the final power axis and its phase into an IEEE FP32 matrix product."""


def finish_source(spec, conjugate_output=False):
    r, prefix, pn = spec['r'], spec['prefix'], spec['physical_n']
    previous = spec['previous_prefix']
    columns = 64
    n = prefix * r
    lines = [
        '@triton.jit', 'def power_cube_finish(X,TW,Y):',
        '    batch=tl.program_id(1).to(tl.int64)',
        '    tile=tl.program_id(0)',
        f'    kprev=tile//{previous//columns}',
        f'    col=tile*{columns}+tl.arange(0,{columns})[None,:]',
        f'    d=tl.arange(0,{r})[:,None]',
        f'    off=d*{prefix}+col',
        f'    xr=tl.load(X+batch*{2*pn}+off)',
        f'    xi=tl.load(X+batch*{2*pn}+{pn}+off)',
        f'    row=tl.arange(0,{2*r})[:,None]',
        f'    k=tl.arange(0,{r})[None,:]',
        f'    basis=TW+kprev*{4*r*r}+row*{2*r}+k',
        '    a=tl.load(basis)', f'    b=tl.load(basis+{r})',
        '    value=tl.dot(a,xr,input_precision="ieee")',
        '    value=tl.dot(b,xi,acc=value,input_precision="ieee")',
        f'    word=tile*{2*columns}+tl.arange(0,{2*columns})[None,:]',
    ]
    # Pack a bounded row group at a time. Packing the entire matrix output
    # together would unnecessarily keep a second full complex tile in UB.
    for start in range(0, r, 32):
        lines += [
            f'    re=al.extract_slice(value,[{start},0],[32,{columns}],[1,1])',
            f'    im=al.extract_slice(value,[{r+start},0],[32,{columns}],[1,1])',
        ]
        if conjugate_output:
            lines += ['    im=-im']
        lines += [
            f'    out=tl.reshape(tl.join(re,im),(32,{2*columns}))',
            f'    freq=(tl.arange(0,32)+{start})[:,None]',
            f'    tl.store(Y+batch*{2*n}+freq*{2*prefix}+word,out)',
        ]
    return '\n'.join(lines) + '\n'
