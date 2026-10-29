# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""IEEE Cube prefix for aligned DIF, with deferred column twiddles.

The first conversion is an explicit pipeline pass. The second transform folds
its radix-dependent twiddle into the coefficients; stage two applies the
remaining column phase together with its normal twiddle.
"""
import math,re
from flagfft_codegen.kernels_npu_dif_aligned import build_aligned_vector_dif

def build_cube_prefix_dif(factors,direction='forward'):
 if len(factors)<4 or not (13<=factors[0]<=19 and 11<=factors[1]<=15):
  raise ValueError('Cube DIF prefix needs two large leading radices and at least four stages')
 source,specs=build_aligned_vector_dif(factors,direction)
 source+='\nimport triton.language.extra.cann.extension as al\n'
 n=math.prod(factors);r=factors[0];m=n//r;pr=max(16,1<<(r-1).bit_length());c=512;pn=specs[0]['physical_n'];tiles=(m+c-1)//c
 s0=specs[0];assert s0['stage']==0
 source=re.sub(r'@triton.jit\ndef '+s0['name']+r'\([^\n]*\):\n[\s\S]*?(?=\n@triton.jit|\Z)','',source)
 source+=f'''
@triton.jit
def planar_prepare(X,TW,Y):
    b=tl.program_id(1).to(tl.int64)
    j=tl.program_id(0)*4096+tl.arange(0,4096)
    w=tl.program_id(0)*8192+tl.arange(0,8192)
    raw=tl.load(X+b*{2*n}+w,w<{2*n},0.)
    re,im=tl.split(tl.reshape(raw,(4096,2)))
    tl.store(Y+b*{2*n}+j,re,j<{n})
    tl.store(Y+b*{2*n}+{n}+j,im,j<{n})
@triton.jit
def cube_first(X,TW,Y):
    b=tl.program_id(1).to(tl.int64)
    d=tl.arange(0,{pr})[:,None]
    q=tl.program_id(0)*{c}+tl.arange(0,{c})[None,:]
    good=(d<{r})&(q<{m})
    off=d*{m}+q
    xr=tl.load(X+b*{2*n}+off,good,0.)
    xi=tl.load(X+b*{2*n}+{n}+off,good,0.)
    row=tl.arange(0,{2*pr})
    k=tl.arange(0,{pr})
    w0=tl.load(TW+row[:,None]*{2*pr}+k[None,:])
    w1=tl.load(TW+row[:,None]*{2*pr}+{pr}+k[None,:])
    y=tl.dot(w0,xr,input_precision="ieee")
    y=tl.dot(w1,xi,acc=y,input_precision="ieee")
    yr=al.extract_slice(y,[0,0],[{pr},{c}],[1,1])
    yi=al.extract_slice(y,[{pr},0],[{pr},{c}],[1,1])
    tl.store(Y+b*{2*n}+off,yr,good)
    tl.store(Y+b*{2*n}+{n}+off,yi,good)
'''
 prep=dict(name='planar_prepare',phase='prepare',grid=((n+4095)//4096,),physical_n=pn,auto=True)
 cube=dict(s0,name='cube_first',grid=(tiles,),auto=True,planar_cube=True,pr=pr)
 # Stage one has a scalar previous frequency per program. Absorb its
 # radix-dependent phase into the coefficients, and defer the remaining
 # column-dependent phase to the next stage's twiddle table.
 second=specs[1];rr=second['r'];prev=factors[0];mm=second['m'];pp=max(16,1<<(rr-1).bit_length());cc=1024;tt=(mm+cc-1)//cc
 assert second['stage']==1 and second['phase']=='early' and len([z for z in specs if z['phase']=='early'])==2
 source=re.sub(r'@triton.jit\ndef '+second['name']+r'\([^\n]*\):\n[\s\S]*?(?=\n@triton.jit|\Z)','',source)
 source+=f'''
@triton.jit
def chirped_second(X,TW,Y):
    b=tl.program_id(1).to(tl.int64)
    pf=tl.program_id(0)//{tt}
    d=tl.arange(0,{pp})[:,None]
    q=tl.program_id(0)%{tt}*{cc}+tl.arange(0,{cc})[None,:]
    good=(d<{rr})&(q<{mm})
    off=pf*{rr*mm}+d*{mm}+q
    xr=tl.load(X+b*{2*n}+off,good,0.)
    xi=tl.load(X+b*{2*n}+{n}+off,good,0.)
    row=tl.arange(0,{2*pp})
    k=tl.arange(0,{pp})
    w0=tl.load(TW+pf*{4*pp*pp}+row[:,None]*{2*pp}+k[None,:])
    w1=tl.load(TW+pf*{4*pp*pp}+row[:,None]*{2*pp}+{pp}+k[None,:])
    y=tl.dot(w0,xr,input_precision="ieee")
    y=tl.dot(w1,xi,acc=y,input_precision="ieee")
    outrow=tl.arange(0,{2*pp})[:,None]
    dst=outrow*{prev*mm}+pf*{mm}+q
    tl.store(Y+b*{2*n}+dst,y,(outrow<{2*rr})&(q<{mm}))
'''
 chirp=dict(second,name='chirped_second',grid=(prev*tt,),auto=True,chirped_cube=True,pr=pp)
 result=[prep,cube,chirp]+specs[2:]
 for z in result:
  if z.get('stage')==2:z['deferred_twiddle']=True
 return source,result
