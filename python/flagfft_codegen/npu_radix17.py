# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Fixed cyclic/negacyclic length-eight convolutions for symmetric radix17."""
import math

def _product(lines,label,u,v,a,b,s):
 lines += [f'    {label}m={b!r}*({u}-{v})',f'    {label}u={(a-b)!r}*{u}+{label}m',f'    {label}v={(a+(1-s)*b)!r}*{v}+{label}m']

def _cyclic4(name,k):
 lines=['@triton.jit',f'def {name}(x0,x1,x2,x3):','    a,b=x0+x2,x1+x3','    f0,f1=a+b,a-b','    u,v=x0-x2,x1-x3']
 _product(lines,'z','u','v',(k[0]-k[2])/2,(k[1]-k[3])/2,0)
 lines += [f'    a={sum(k)/4!r}*f0',f'    b={(k[0]-k[1]+k[2]-k[3])/4!r}*f1','    p,q=a+b,a-b','    return p+zu,q+zv,p-zu,q-zv']
 return '\n'.join(lines)+'\n'

def _negative4(name,k):
 sq=math.sqrt(2);lines=['@triton.jit',f'def {name}(x0,x1,x2,x3):','    a,b=x0-x2,x1+x3',f'    c,d={sq!r}*x3,{sq!r}*x2','    pu,pv=a+c,b-d','    mu,mv=a-c,b+d']
 for label,s,u,v in [('p',sq,'pu','pv'),('m',-sq,'mu','mv')]:
  _product(lines,label,u,v,(k[0]-k[2]+s*k[3])/2,(k[1]-s*k[2]+k[3])/2,s)
 lines += [f'    y3={(1/sq)!r}*(pu-mu)',f'    y2={(1/sq)!r}*(mv-pv)','    return pu+mu+y2,pv+mv-y3,y2,y3']
 return '\n'.join(lines)+'\n'

def _convolution8(name,k,negative):
 lines=['@triton.jit',f'def {name}(x0,x1,x2,x3,x4,x5,x6,x7):']
 if not negative:
  source=_cyclic4(name+'_lo',[(k[j]+k[j+4])/2 for j in range(4)])+_negative4(name+'_hi',[(k[j]-k[j+4])/2 for j in range(4)])
  lines += ['    a0,a1,a2,a3=x0+x4,x1+x5,x2+x6,x3+x7','    b0,b1,b2,b3=x0-x4,x1-x5,x2-x6,x3-x7',f'    u0,u1,u2,u3={name}_lo(a0,a1,a2,a3)',f'    v0,v1,v2,v3={name}_hi(b0,b1,b2,b3)','    return u0+v0,u1+v1,u2+v2,u3+v3,u0-v0,u1-v1,u2-v2,u3-v3']
 else:
  even=k[::2];odd=k[1::2]
  source=_negative4(name+'_a',even)+_negative4(name+'_b',odd)+_negative4(name+'_sum',[a+b for a,b in zip(even,odd)])
  lines += [f'    u0,u1,u2,u3={name}_a(x0,x2,x4,x6)',f'    v0,v1,v2,v3={name}_b(x1,x3,x5,x7)',f'    w0,w1,w2,w3={name}_sum(x0+x1,x2+x3,x4+x5,x6+x7)','    return u0-v3,w0-u0-v0,u1+v0,w1-u1-v1,u2+v1,w2-u2-v2,u3+v2,w3-u3-v3']
 return source+'\n'+'\n'.join(lines)+'\n'

def radix17_source():
 order=[pow(3,j,17) for j in range(8)]
 co=[math.cos(2*math.pi*j/17) for j in order];si=[math.sin(2*math.pi*j/17) for j in order]
 source=_convolution8('_npu_cos8',co,False)+'\n'+_convolution8('_npu_sin8',si,True)
 args=[f'r{k}' for k in range(17)]+[f'i{k}' for k in range(17)];lines=['@triton.jit','def _fwd_rad17_b1('+','.join(args)+'):']
 for j,k in enumerate(order):lines += [f'    p{j}r,p{j}i=r{k}+r{17-k},i{k}+i{17-k}',f'    m{j}r,m{j}i=r{k}-r{17-k},i{k}-i{17-k}']
 for component in ('r','i'):
  lhs=','.join(f'c{j}{component}' for j in range(8));rhs=','.join(f'p{(-j)%8}{component}' for j in range(8));lines += [f'    {lhs}=_npu_cos8({rhs})']
  lhs=','.join(f's{j}{component}' for j in range(8));rhs=','.join([f'm0{component}']+[f'-m{8-j}{component}' for j in range(1,8)]);lines += [f'    {lhs}=_npu_sin8({rhs})']
 lines += ['    y0r=r0+'+'+'.join(f'p{j}r' for j in range(8)),'    y0i=i0+'+'+'.join(f'p{j}i' for j in range(8))]
 for j,k in enumerate(order):lines += [f'    ar,ai=r0+c{j}r,i0+c{j}i',f'    y{k}r,y{k}i=ar+s{j}i,ai-s{j}r',f'    y{17-k}r,y{17-k}i=ar-s{j}i,ai+s{j}r']
 lines += ['    return '+','.join([f'y{k}r' for k in range(17)]+[f'y{k}i' for k in range(17)])]
 return source+'\n'+'\n'.join(lines)+'\n'


def component_first_source(spec,n,direction,columns=512,conjugate_output=False):
 """Share each real convolution between interleaved real and imaginary lanes."""
 # The cosine/sine convolutions have real coefficients, so they can process
 # both components as adjacent lanes. Split only at the planar output boundary.
 m=n//17;c=columns;points=2*c
 lines=['@triton.jit',f'def {spec["name"]}(X,TW,Y):','    batch=tl.program_id(1).to(tl.int64)',f'    lane=tl.arange(0,{c})',f'    word=tl.arange(0,{points})',f'    base=tl.program_id(0)*{c}']
 start=0;groups=[]
 while start<17:
  g=min(4,1<<((17-start).bit_length()-1));groups.append((start,g));size=g*points
  lines += [f'    d=(tl.arange(0,{g})+{start})[:,None]',f'    raw=tl.load(X+batch*{2*n}+d*{2*m}+base*2+word[None,:],base*2+word[None,:]<{2*m},0.)',f'    raw=tl.reshape(raw,({size},))']
  for j in range(g):lines += [f'    x{start+j}=al.extract_slice(raw,[{j*points}],[{points}],[1])']
  start+=g
 order=[pow(3,j,17) for j in range(8)]
 for j,d in enumerate(order):lines += [f'    p{j}=x{d}+x{17-d}',f'    m{j}=x{d}-x{17-d}']
 lines += ['    '+','.join(f'c{j}' for j in range(8))+'=_npu_cos8('+','.join(f'p{(-j)%8}' for j in range(8))+')', '    '+','.join(f's{j}' for j in range(8))+'=_npu_sin8('+','.join(['m0']+[f'-m{8-j}' for j in range(1,8)])+')','    y0=x0+'+'+'.join(f'p{j}' for j in range(8)),f'    sign=tl.where(word%2==0,{1. if direction=="forward" else -1.},{-1. if direction=="forward" else 1.})']
 # Swapping adjacent components implements the final +/- i rotation.
 for j,d in enumerate(order):lines += [f'    mix=tl.gather(s{j},word^1,0)*sign',f'    y{d}=x0+c{j}+mix',f'    y{17-d}=x0+c{j}-mix']
 for start,g in groups:
  size=g*points;lanes=g*c
  lines += [f'    raw=tl.zeros(({size},),tl.float32)']
  for j in range(g):lines += [f'    raw=al.insert_slice(raw,y{start+j},[{j*points}],[{points}],[1])']
  lines += [f'    real,imag=tl.split(tl.reshape(raw,({lanes},2)))']
  if conjugate_output:lines += ['    imag=-imag']
  lines += [f'    d=(tl.arange(0,{g})+{start})[:,None]',f'    q=base+lane[None,:]',f'    off=d*{m}+q',f'    tl.store(Y+batch*{2*n}+off,tl.reshape(real,({g},{c})),q<{m})',f'    tl.store(Y+batch*{2*n}+{n}+off,tl.reshape(imag,({g},{c})),q<{m})']
 return '\n'.join(lines)+'\n'
