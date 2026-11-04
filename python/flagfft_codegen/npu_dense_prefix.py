# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""A bounded short-prefix DFT with direct two-dimensional IEEE operands.

Real-basis symmetry computes both frequency halves from one product. A single
padding lane permits contiguous output rows without a separate padding kernel.
"""
import math,re


def replace_prefix(source,specs,factors,conjugate_output=False):
    n=math.prod(factors);r=factors[0]*factors[1];assert r%2==1 and r<=255
    m=n//r;pad=1<<(r-1).bit_length();h=r//2+1
    c=f=32;ftiles=math.ceil(h/f)
    if pad-r!=1:raise ValueError('Dense prefix requires one padding lane')
    ix=next(i for i,s in enumerate(specs) if s['phase']=='transpose');pn=specs[ix]['physical_n']
    for old in specs[:ix+1]:
        source=re.sub(r'@triton.jit\ndef '+old['name']+r'\([^\n]*\):\n[\s\S]*?(?=\n@triton.jit|\Z)','',source)
    body=f'''@triton.jit
def dense_prefix_half(X,TW,Y):
    batch=tl.program_id(1).to(tl.int64)
    tile=tl.program_id(0)%{ftiles}
    qtile=tl.program_id(0)//{ftiles}
    digit=tl.arange(0,{pad})[:,None]
    word=qtile*{2*c}+tl.arange(0,{2*c})[None,:]
    raw=tl.load(X+batch*{2*n}+digit*{2*m}+word,(digit<{r})&(word<{2*m}),0.)
    row=tl.arange(0,{2*f})[:,None]
    k=tl.arange(0,{pad})[None,:]
    basis=tl.load(TW+tile*{2*f*pad}+row*{pad}+k)
    value=tl.dot(basis,raw,input_precision="ieee")
    co=al.extract_slice(value,[0,0],[{f},{2*c}],[1,1])
    sn=al.extract_slice(value,[{f},0],[{f},{2*c}],[1,1])
    cr,ci=tl.split(tl.reshape(co,({f*c},2)))
    sr,si=tl.split(tl.reshape(sn,({f*c},2)))
    posr=tl.trans(tl.reshape(cr-si,({f},{c})))
    posi=tl.trans(tl.reshape(ci+sr,({f},{c})))
    reverse=tl.arange(0,{f*c})^{(f-1)*c}
    negr=tl.trans(tl.reshape(tl.gather(cr+si,reverse,0),({f},{c})))
    negi=tl.trans(tl.reshape(tl.gather(ci-sr,reverse,0),({f},{c})))
    sample=(qtile*{c}+tl.arange(0,{c}))[:,None]
    freq=tile*{f}+tl.arange(0,{f})[None,:]
    negfreq={r}-(tile*{f}+{f-1})+tl.arange(0,{f})[None,:]
    dst=sample*{pad}+freq
    ndst=sample*{pad}+negfreq
    tl.store(Y+batch*{2*pn}+dst,posr,(sample<{m})&(freq<{h}))
    tl.store(Y+batch*{2*pn}+{pn}+dst,posi,(sample<{m})&(freq<{h}))
'''
    if pad-r==1:
        body+=f'''    negr=tl.where(negfreq<{r},negr,0.)
    negi=tl.where(negfreq<{r},negi,0.)
    tl.store(Y+batch*{2*pn}+ndst,negr,(sample<{m})&(negfreq>={h})&(negfreq<{pad}))
    tl.store(Y+batch*{2*pn}+{pn}+ndst,negi,(sample<{m})&(negfreq>={h})&(negfreq<{pad}))
'''
    if conjugate_output:
        body=body.replace('tl.reshape(ci+sr,','tl.reshape(-ci-sr,').replace('tl.gather(ci-sr,reverse,0)','tl.gather(sr-ci,reverse,0)')
    source+='\n'+body
    first=dict(name='dense_prefix_half',phase='early',stage=0,r=r,m=m,prefix=1,physical_n=pn,grid=(math.ceil(m/c)*ftiles,),auto=True,dense_prefix=(r,pad,c,f,ftiles))
    return source,[first]+specs[ix+1:]
