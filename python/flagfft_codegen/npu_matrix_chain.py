# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Three bounded IEEE FP32 matrices for a dense prefix and a short suffix.

The 255-point producer owns the deferred rotation. A 26-point middle axis
then uses a constant real basis, and the terminal matrix includes the phase
from that middle axis. All intermediate rows keep their 256-column padding.
"""
import math

from .npu_dense_prefix import replace_prefix
from .npu_dense_late_pair import prephase
from .npu_matrix_batch import batch_matrix_source


def build(factors, direction='forward', global_inverse=False, columns=128):
    if (tuple(factors[:4]) != (17, 15, 13, 2) or len(factors) != 5
            or not 40 <= factors[-1] <= 128 or columns not in (128, 256)):
        raise ValueError('Matrix chain requires a 255 x 26 prefix and a 40..128 terminal axis')
    if direction not in ('forward', 'inverse') or (global_inverse and direction != 'inverse'):
        raise ValueError('Invalid matrix-chain direction')
    n = math.prod(factors)
    radix = factors[-1]
    physical = n//255*256
    source = 'import triton\nimport triton.language as tl\nimport triton.language.extra.cann.extension as al\n'
    # Reuse the established dense producer with its exact physical layout.
    source, specs = replace_prefix(source, [dict(name='prefix_workspace', phase='transpose', physical_n=physical)],
                                   factors, conjugate_output=global_inverse, columns=64, frequencies=64)
    source = prephase(source, specs[0]['name'], 2*2*64*256, physical)
    specs[0].update(num_stages=2)
    middle = f'''@triton.jit
def matrix_middle26_batch(X,TW,Y):
    batch=tl.program_id(1).to(tl.int64)
    j=tl.program_id(0)
    col=tl.arange(0,256)[None,:]
    digit=tl.arange(0,32)[:,None]
    off=(digit*{radix}+j)*256+col
    xr=tl.load(X+batch*{2*physical}+off,digit<26,0.)
    xi=tl.load(X+batch*{2*physical}+{physical}+off,digit<26,0.)
    row=tl.arange(0,32)[:,None]
    k=tl.arange(0,32)[None,:]
    basis=tl.load(TW+row*32+k)
    vr=tl.dot(basis,xr,input_precision="ieee")
    vi=tl.dot(basis,xi,input_precision="ieee")
    cr=al.extract_slice(vr,[0,0],[16,256],[1,1])
    sr=al.extract_slice(vr,[16,0],[16,256],[1,1])
    ci=al.extract_slice(vi,[0,0],[16,256],[1,1])
    si=al.extract_slice(vi,[16,0],[16,256],[1,1])
    freq=tl.arange(0,16)[:,None]
    out=(j*26+freq)*256+col
    tl.store(Y+batch*{2*physical}+out,cr-si)
    tl.store(Y+batch*{2*physical}+{physical}+out,ci+sr)
    lane=tl.arange(0,4096)
    nf=16+lane//256
    take=tl.maximum(0,26-nf)*256+lane%256
    nr=tl.gather(tl.reshape(cr+si,(4096,)),take,0)
    ni=tl.gather(tl.reshape(ci-sr,(4096,)),take,0)
    out=(j*26+16+freq)*256+col
    tl.store(Y+batch*{2*physical}+out,tl.reshape(nr,(16,256)),16+freq<26)
    tl.store(Y+batch*{2*physical}+{physical}+out,tl.reshape(ni,(16,256)),16+freq<26)
'''
    source += '\n'+batch_matrix_source(middle, 8)
    specs.append(dict(name='matrix_middle26_batch', phase='late', stage=2, r=26, m=radix,
                      prefix=255, physical_prefix=256, physical_n=physical,
                      grid=(radix,), batch_pack=8, num_stages=2, auto=True))
    padded = 1 << (radix-1).bit_length()
    tiles = math.ceil(radix/64)
    prefix, physical_prefix = 255*26, 256*26
    # The accumulator is at most 128 KiB. Packing eight rows at a time
    # bounds the extra Vector temporaries for the 256-column alternative.
    pack = 8 if columns == 256 else 16
    lines = ['@triton.jit', 'def matrix_chain_finish_batch(X,TW,Y):',
             '    batch=tl.program_id(1).to(tl.int64)',
             f'    tile=tl.program_id(0)//{tiles}', f'    ftile=tl.program_id(0)%{tiles}',
             f'    kprev=tile//{256//columns}',
             f'    col=tile*{columns}+tl.arange(0,{columns})[None,:]',
             f'    d=tl.arange(0,{padded})[:,None]',
             f'    off=d*{physical_prefix}+col',
             f'    xr=tl.load(X+batch*{2*physical}+off,d<{radix},0.)',
             f'    xi=tl.load(X+batch*{2*physical}+{physical}+off,d<{radix},0.)',
             '    row=tl.arange(0,128)[:,None]', f'    k=tl.arange(0,{padded})[None,:]',
             f'    basis=TW+(kprev*{tiles}+ftile)*{256*padded}+row*{2*padded}+k',
             '    a=tl.load(basis)', f'    b=tl.load(basis+{padded})',
             '    value=tl.dot(a,xr,input_precision="ieee")',
             '    value=tl.dot(b,xi,acc=value,input_precision="ieee")',
             f'    word=tl.arange(0,{2*columns})[None,:]',
             f'    base=(tile*{columns}//256)*510+2*(tile*{columns}%256)',
             f'    validword=word<2*(255-tile*{columns}%256)']
    for start in range(0, 64, pack):
        lines += [f'    re=al.extract_slice(value,[{start},0],[{pack},{columns}],[1,1])',
                  f'    im=al.extract_slice(value,[{64+start},0],[{pack},{columns}],[1,1])']
        if global_inverse:
            lines += ['    im=-im']
        lines += [f'    out=tl.reshape(tl.join(re,im),({pack},{2*columns}))',
                  f'    freq=(ftile*64+tl.arange(0,{pack})+{start})[:,None]',
                  f'    tl.store(Y+batch*{2*n}+freq*{2*prefix}+base+word,out,(freq<{radix})&validword)']
    source += '\n'+batch_matrix_source('\n'.join(lines)+'\n', 8)
    specs.append(dict(name='matrix_chain_finish_batch', phase='late', stage=4, r=radix, m=1,
                      prefix=prefix, physical_prefix=physical_prefix, physical_n=physical,
                      grid=(physical_prefix//columns*tiles,), batch_pack=8, num_stages=1, auto=True))
    return source, specs
