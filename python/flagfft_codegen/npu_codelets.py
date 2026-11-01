# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Symmetric FP32 codelets for the Ascend vector backend.

Radix 5 shares pair sums across all outputs. Radix 13 pairs +/-2**j
modulo 13. Its cosine and sine matrices become length-six cyclic and
negacyclic convolutions, evaluated by polynomial CRT. All constants are
formed on the host; emitted operations remain ordinary Triton arithmetic.
"""
import math
from functools import lru_cache

_RADIX5_SOURCE='''
@triton.jit
def _fwd_rad5_b1(r0,r1,r2,r3,r4,i0,i1,i2,i3,i4):
    s1r,s1i=r1+r4,i1+i4
    s2r,s2i=r2+r3,i2+i3
    d1r,d1i=r1-r4,i1-i4
    d2r,d2i=r2-r3,i2-i3
    sr,si=s1r+s2r,s1i+s2i
    z0r,z0i=r0+sr,i0+si
    ar,ai=r0-0.25*sr,i0-0.25*si
    br,bi=0.5590169943749475*(s1r-s2r),0.5590169943749475*(s1i-s2i)
    a1r,a1i=ar+br,ai+bi
    a2r,a2i=ar-br,ai-bi
    b1r,b1i=0.9510565162951535*d1r+0.5877852522924731*d2r,0.9510565162951535*d1i+0.5877852522924731*d2i
    b2r,b2i=0.5877852522924731*d1r-0.9510565162951535*d2r,0.5877852522924731*d1i-0.9510565162951535*d2i
    return z0r,a1r+b1i,a2r+b2i,a2r-b2i,a1r-b1i,z0i,a1i-b1r,a2i-b2r,a2i+b2r,a1i+b1r
'''

def _quadratic_product(lines,label,u,v,a,b,s):
 """Multiply by a+b*x modulo x*x+s*x+1 with three real products."""
 lines += [f'    {label}m={b!r}*({u}-{v})',f'    {label}u={(a-b)!r}*{u}+{label}m',f'    {label}v={(a+(1-s)*b)!r}*{v}+{label}m']

def _convolution6(name,kernel,negative):
 """Emit CRT for x**6-1 or x**6+1, with inverse scales in constants."""
 k=kernel;lines=['@triton.jit',f'def {name}(x0,x1,x2,x3,x4,x5):']
 if not negative:
  lines += ['    a0,a1,a2=x0+x3,x1+x4,x2+x5','    b0,b1,b2=x0-x3,x1-x4,x2-x5','    f0=a0+a1+a2','    f1=b0-b1+b2','    f2u,f2v=a0-a2,a1-a2','    f3u,f3v=b0-b2,b1+b2']
  h0=sum(k)/6;h1=sum(v*(-1)**i for i,v in enumerate(k))/6
  a=(k[0]+k[3]-k[2]-k[5])/2;b=(k[1]+k[4]-k[2]-k[5])/2
  _quadratic_product(lines,'z2','f2u','f2v',a,b,1)
  a=(k[0]-k[3]-k[2]+k[5])/2;b=(k[1]-k[4]+k[2]-k[5])/2
  _quadratic_product(lines,'z3','f3u','f3v',a,b,-1)
  lines += [f'    z0={h0!r}*f0',f'    z1={h1!r}*f1','    u=z0-(z2u+z2v)*(1.0/3.0)','    v=z1+(z3v-z3u)*(1.0/3.0)','    a0,a1=u+z2u,u+z2v','    b0,b1=v+z3u,z3v-v','    return a0+b0,a1+b1,u+v,a0-b0,a1-b1,u-v']
 else:
  sq=math.sqrt(3)
  lines += ['    a=x0-x2','    b=x1+x5','    c=x3+x5','    d=x2+x4','    aa=a-2.0*x4','    bb=b+2.0*x3',f'    c={sq!r}*c',f'    d={sq!r}*d','    f0u,f0v=a+x4,b-x3','    fpu,fpv=aa+c,bb-d','    fmu,fmv=aa-c,bb+d']
  _quadratic_product(lines,'z0','f0u','f0v',(k[0]-k[2]+k[4])/3,(k[1]-k[3]+k[5])/3,0)
  for label,s,u,v in [('zp',sq,'fpu','fpv'),('zm',-sq,'fmu','fmv')]:
   a=(k[0]-k[2]-2*k[4]+s*(k[3]+k[5]))/6
   b=(k[1]+2*k[3]+k[5]-s*(k[2]+k[4]))/6
   _quadratic_product(lines,label,u,v,a,b,s)
  lines += ['    a=zpu+zmu','    b=zpv+zmv',f'    c={sq!r}*(zpu-zmu)',f'    d={sq!r}*(zmv-zpv)','    y4=z0u-a','    y3=b-z0v','    y2=d-y4','    y5=c-y3','    y0=3.0*z0u+y2-y4','    y1=3.0*z0v+y3-y5','    return y0,y1,y2,y3,y4,y5']
 return '\n'.join(lines)+'\n'

def _radix13_source():
 # 2**6 == -1 modulo 13: cosine is periodic, sine is antiperiodic.
 order=[pow(2,j,13) for j in range(6)];co=[math.cos(2*math.pi*j/13) for j in order];si=[math.sin(2*math.pi*j/13) for j in order]
 source=_convolution6('_npu_cos6',co,False)+'\n'+_convolution6('_npu_sin6',si,True)
 args=[f'r{k}' for k in range(13)]+[f'i{k}' for k in range(13)]
 lines=['@triton.jit','def _fwd_rad13_b1('+','.join(args)+'):']
 for j,k in enumerate(order):lines += [f'    p{j}r,p{j}i=r{k}+r{13-k},i{k}+i{13-k}',f'    m{j}r,m{j}i=r{k}-r{13-k},i{k}-i{13-k}']
 for component in ('r','i'):
  lhs=','.join(f'c{j}{component}' for j in range(6));rhs=','.join(f'p{(-j)%6}{component}' for j in range(6));lines += [f'    {lhs}=_npu_cos6({rhs})']
  lhs=','.join(f's{j}{component}' for j in range(6));rhs=','.join([f'm0{component}']+[f'-m{6-j}{component}' for j in range(1,6)]);lines += [f'    {lhs}=_npu_sin6({rhs})']
 lines += ['    y0r=r0+'+'+'.join(f'p{j}r' for j in range(6)),'    y0i=i0+'+'+'.join(f'p{j}i' for j in range(6))]
 for j,k in enumerate(order):lines += [f'    ar,ai=r0+c{j}r,i0+c{j}i',f'    y{k}r,y{k}i=ar+s{j}i,ai-s{j}r',f'    y{13-k}r,y{13-k}i=ar-s{j}i,ai+s{j}r']
 lines += ['    return '+','.join([f'y{k}r' for k in range(13)]+[f'y{k}i' for k in range(13)])]
 return source+'\n'+'\n'.join(lines)+'\n'


@lru_cache(maxsize=None)
def codelet_source(radix):
    if radix == 5:
        return _RADIX5_SOURCE
    if radix == 13:
        return _radix13_source()
    return None
