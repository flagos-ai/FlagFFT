# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""IEEE FP32 prefix with natural frequencies in 128-lane aligned rows."""
import math


def prefix_source(factors, global_inverse=False):
    n = math.prod(factors)
    m = n // 120
    pn = m * 128
    # A 64x64 complex tile uses one real 128x128 matrix product. The real
    # cosine/sine basis also supplies frequencies 64..119 by conjugate
    # symmetry of the basis (the complex input itself need not be real).
    positive_imag = '-ci-sr' if global_inverse else 'ci+sr'
    negative_imag = 'sr-ci' if global_inverse else 'ci-sr'
    return f'''@triton.jit
def coarse_dense_prefix(X,TW,Y):
    batch=tl.program_id(1).to(tl.int64)
    base=tl.program_id(0)*64
    digit=tl.arange(0,128)[:,None]
    word=base*2+tl.arange(0,128)[None,:]
    raw=tl.load(X+batch*{2*n}+digit*{2*m}+word,(digit<120)&(word<{2*m}),0.)
    row=tl.arange(0,128)[:,None]
    k=tl.arange(0,128)[None,:]
    basis=tl.load(TW+row*128+k)
    value=tl.dot(basis,raw,input_precision="ieee")
    co=al.extract_slice(value,[0,0],[64,128],[1,1])
    sn=al.extract_slice(value,[64,0],[64,128],[1,1])
    cr,ci=tl.split(tl.reshape(co,(4096,2)))
    sr,si=tl.split(tl.reshape(sn,(4096,2)))
    posr=tl.trans(tl.reshape(cr-si,(64,64)))
    posi=tl.trans(tl.reshape({positive_imag},(64,64)))
    sample=(base+tl.arange(0,64))[:,None]
    freq=tl.arange(0,64)[None,:]
    dst=sample*128+freq
    tl.store(Y+batch*{2*pn}+dst,posr,sample<{m})
    tl.store(Y+batch*{2*pn}+{pn}+dst,posi,sample<{m})
    lane=tl.arange(0,4096)
    nf=64+lane//64
    take=tl.maximum(0,120-nf)*64+lane%64
    nr=tl.gather(cr+si,take,0)
    ni=tl.gather({negative_imag},take,0)
    nr=tl.where(nf<120,nr,0.)
    ni=tl.where(nf<120,ni,0.)
    ndst=sample*128+64+freq
    tl.store(Y+batch*{2*pn}+ndst,tl.trans(tl.reshape(nr,(64,64))),sample<{m})
    tl.store(Y+batch*{2*pn}+{pn}+ndst,tl.trans(tl.reshape(ni,(64,64))),sample<{m})
'''
