# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Radix17 from one IEEE 16x16 real basis over interleaved input components.

For each nonzero input digit d, sum(cos(2*pi*k*d/17), k=1..8)=-1/2.
Recover the DC output from these cosine responses instead of padding a
17-point DFT to 32. No input reshape feeds the dot: both operands are 2D.
"""


def half_matrix_first_source(spec,n,columns=512,conjugate_output=False):
 m=n//17;c=columns;points=8*c
 body=f'''@triton.jit
def {spec['name']}(X,TW,Y):
    batch=tl.program_id(1).to(tl.int64)
    digit=(tl.arange(0,16)+1)[:,None]
    word=(tl.program_id(0)*{2*c}+tl.arange(0,{2*c}))[None,:]
    raw=tl.load(X+batch*{2*n}+digit*{2*m}+word,word<{2*m},0.)
    row=tl.arange(0,16)[:,None]
    k=tl.arange(0,16)[None,:]
    w=tl.load(TW+row*16+k)
    value=tl.dot(w,raw,input_precision="ieee")
    co=al.extract_slice(value,[0,0],[8,{2*c}],[1,1])
    si=al.extract_slice(value,[8,0],[8,{2*c}],[1,1])
    x0=tl.load(X+batch*{2*n}+tl.program_id(0)*{2*c}+tl.arange(0,{2*c}),tl.program_id(0)*{2*c}+tl.arange(0,{2*c})<{2*m},0.)
'''
 for j in range(8):body+=f'    a{j}=tl.reshape(al.extract_slice(co,[{j},0],[1,{2*c}],[1,1]),({2*c},))\n'
 body+='    dc=x0-2.*('+'+'.join(f'a{j}' for j in range(8))+')\n'
 body+=f'''    dr,di=tl.split(tl.reshape(dc,({c},2)))
    sample=tl.program_id(0)*{c}+tl.arange(0,{c})
    tl.store(Y+batch*{2*n}+sample,dr,sample<{m})
    tl.store(Y+batch*{2*n}+{n}+sample,di,sample<{m})
    co=co+x0[None,:]
    cr,ci=tl.split(tl.reshape(co,({points},2)))
    sr,si=tl.split(tl.reshape(si,({points},2)))
    pr=tl.reshape(cr-si,(8,{c}))
    pi=tl.reshape(ci+sr,(8,{c}))
    reverse=tl.arange(0,{points})^{7*c}
    nr=tl.reshape(tl.gather(cr+si,reverse,0),(8,{c}))
    ni=tl.reshape(tl.gather(ci-sr,reverse,0),(8,{c}))
    freq=(tl.arange(0,8)+1)[:,None]
    negfreq=tl.arange(0,8)[:,None]+9
    off=freq*{m}+sample[None,:]
    noff=negfreq*{m}+sample[None,:]
    tl.store(Y+batch*{2*n}+off,pr,sample[None,:]<{m})
    tl.store(Y+batch*{2*n}+{n}+off,pi,sample[None,:]<{m})
    tl.store(Y+batch*{2*n}+noff,nr,sample[None,:]<{m})
    tl.store(Y+batch*{2*n}+{n}+noff,ni,sample[None,:]<{m})
'''
 if conjugate_output:
  body=body.replace(',di,sample',',-di,sample').replace(',pi,sample',',-pi,sample').replace(',ni,sample',',-ni,sample')
 return body
