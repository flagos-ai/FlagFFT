# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""IEEE Cube prefix for aligned DIF, with deferred column twiddles.

Large first radices use an explicit planar conversion and a matrix DFT.
Small first radices retain the shared vector codelet. The second transform folds
its radix-dependent twiddle into the coefficients; stage two applies the
remaining column phase together with its normal twiddle.
"""
import math,re,os
from flagfft_codegen.kernels_npu_dif_aligned import build_aligned_vector_dif
from flagfft_codegen.kernels_leaf import _emit_natural_order_codelet_call
from flagfft_codegen.kernels_common import _NATURAL_ORDER_CODELET_RADICES


def _grouped_first_stage(spec,n,direction,columns=512):
 """Group adjacent branches into bounded interleaved DMA tiles."""
 r,m=spec['r'],spec['m'];c=columns;groups=[];start=0
 lines=['@triton.jit',f'def {spec["name"]}(X,TW,Y):',
  '    b=tl.program_id(1).to(tl.int64)',
  f'    q=tl.program_id(0)*{c}+tl.arange(0,{c})[None,:]',
  f'    word=tl.program_id(0)*{2*c}+tl.arange(0,{2*c})[None,:]']
 while start<r:
  g=min(4,1<<((r-start).bit_length()-1));groups.append((start,g));points=g*c
  lines += [f'    d=tl.arange(0,{g})[:,None]+{start}',
   f'    raw=tl.load(X+b*{2*n}+d*{2*m}+word,word<{2*m},0.)',
   f'    real,imag=tl.split(tl.reshape(raw,({points},2)))']
  for j in range(g):
   lines += [f'    r{start+j}=al.extract_slice(real,[{j*c}],[{c}],[1])',
             f'    i{start+j}=al.extract_slice(imag,[{j*c}],[{c}],[1])']
  start+=g
 lines += _emit_natural_order_codelet_call('    ',r,direction)
 for start,g in groups:
  points=g*c
  lines += [f'    real=tl.zeros(({points},),tl.float32)',f'    imag=tl.zeros(({points},),tl.float32)']
  for j in range(g):
   lines += [f'    real=al.insert_slice(real,r{start+j},[{j*c}],[{c}],[1])',
             f'    imag=al.insert_slice(imag,i{start+j},[{j*c}],[{c}],[1])']
  lines += [f'    d=tl.arange(0,{g})[:,None]+{start}',f'    off=d*{m}+q',
   f'    tl.store(Y+b*{2*n}+off,tl.reshape(real,({g},{c})),q<{m})',
   f'    tl.store(Y+b*{2*n}+{n}+off,tl.reshape(imag,({g},{c})),q<{m})']
 return '\n'.join(lines)+'\n'


def _coalesced_late_stage(spec,direction):
 """Transfer all radix branches as matrices before the shared codelet."""
 r,m,pp,pn=spec['r'],spec['m'],spec['physical_prefix'],spec['physical_n']
 pr=1<<(r-1).bit_length();c=1<<(pp-1).bit_length();points=pr*c
 name=spec['name']
 lines=['@triton.jit',f'def {name}(X,TW,Y):',
  '    b=tl.program_id(1).to(tl.int64)','    j=tl.program_id(0)',
  f'    d=tl.arange(0,{pr})[:,None]',f'    pf=tl.arange(0,{c})[None,:]',
  f'    good=(d<{r})&(pf<{pp})',f'    off=(d*{m}+j)*{pp}+pf',
  f'    re=tl.load(X+b*{2*pn}+off,good,0.)',f'    im=tl.load(X+b*{2*pn}+{pn}+off,good,0.)',
  '    wr=tl.load(TW+off,good,0.)',f'    wi=tl.load(TW+{pn}+off,good,0.)',
  '    re,im=re*wr-im*wi,re*wi+im*wr',
  f'    re=tl.reshape(re,({points},))',f'    im=tl.reshape(im,({points},))']
 for d in range(r):lines += [f'    r{d}=al.extract_slice(re,[{d*c}],[{c}],[1])',f'    i{d}=al.extract_slice(im,[{d*c}],[{c}],[1])']
 lines+=_emit_natural_order_codelet_call('    ',r,direction)
 lines += [f'    re=tl.zeros(({points},),tl.float32)',f'    im=tl.zeros(({points},),tl.float32)']
 for d in range(r):lines += [f'    re=al.insert_slice(re,r{d},[{d*c}],[{c}],[1])',f'    im=al.insert_slice(im,i{d},[{d*c}],[{c}],[1])']
 lines += [f'    dst=j*{r*pp}+d*{pp}+pf',f'    tl.store(Y+b*{2*pn}+dst,tl.reshape(re,({pr},{c})),good)',f'    tl.store(Y+b*{2*pn}+{pn}+dst,tl.reshape(im,({pr},{c})),good)']
 return '\n'.join(lines)+'\n'

def _grouped_late_stage(s,direction,c):
 r,pp,m,pn=s['r'],s['physical_prefix'],s['m'],s['physical_n'];tiles=(pp+c-1)//c
 lines=['@triton.jit',f'def {s["name"]}(X,TW,Y):','    batch=tl.program_id(1).to(tl.int64)',f'    j=tl.program_id(0)//{tiles}',f'    pf=(tl.program_id(0)%{tiles}*{c}+tl.arange(0,{c}))[None,:]',f'    good=pf<{pp}']
 groups=[];start=0
 while start<r:
  g=1<<((r-start).bit_length()-1);groups.append((start,g));points=g*c
  lines += [f'    d=tl.arange(0,{g})[:,None]+{start}',f'    off=(d*{m}+j)*{pp}+pf',f'    xr=tl.load(X+batch*{2*pn}+off,good,0.)',f'    xi=tl.load(X+batch*{2*pn}+{pn}+off,good,0.)','    wr=tl.load(TW+off,good,0.)',f'    wi=tl.load(TW+{pn}+off,good,0.)','    re,im=xr*wr-xi*wi,xr*wi+xi*wr',f'    re=tl.reshape(re,({points},))',f'    im=tl.reshape(im,({points},))']
  for d in range(g):lines += [f'    r{start+d}=al.extract_slice(re,[{d*c}],[{c}],[1])',f'    i{start+d}=al.extract_slice(im,[{d*c}],[{c}],[1])']
  start+=g
 lines += _emit_natural_order_codelet_call('    ',r,direction)
 for start,g in groups:
  points=g*c;lines += [f'    re=tl.zeros(({points},),tl.float32)',f'    im=tl.zeros(({points},),tl.float32)']
  for d in range(g):lines += [f'    re=al.insert_slice(re,r{start+d},[{d*c}],[{c}],[1])',f'    im=al.insert_slice(im,i{start+d},[{d*c}],[{c}],[1])']
  lines += [f'    d=tl.arange(0,{g})[:,None]+{start}',f'    out=j*{r*pp}+d*{pp}+pf',f'    tl.store(Y+batch*{2*pn}+out,tl.reshape(re,({g},{c})),good)',f'    tl.store(Y+batch*{2*pn}+{pn}+out,tl.reshape(im,({g},{c})),good)']
 return '\n'.join(lines)+'\n'


def _batch_grouped_late_stage(s,direction,bp,columns=128):
 """Reuse twiddles across batches with a bounded 512-lane butterfly state."""
 r,m,pp,pn=s['r'],s['m'],s['physical_prefix'],s['physical_n']
 c=min(1<<(pp-1).bit_length(),columns);tiles=math.ceil(pp/c)
 s['grid']=(m*tiles,);s['batch_pack']=bp
 lines=['@triton.jit',f'def {s["name"]}(X,TW,Y,NB):',
        f'    batch=(tl.program_id(1)*{bp}+tl.arange(0,{bp}))[None,:,None]',
        f'    j=tl.program_id(0)//{tiles}',
        f'    pf=(tl.program_id(0)%{tiles}*{c}+tl.arange(0,{c}))[None,None,:]',
        f'    good=(pf<{pp})&(batch<NB)']
 groups=[];start=0
 while start<r:
  g=1<<((r-start).bit_length()-1);groups.append((start,g));points=g*bp*c
  lines += [f'    d=(tl.arange(0,{g})+{start})[:,None,None]',
            f'    off=(d*{m}+j)*{pp}+pf',
            f'    xr=tl.load(X+batch*{2*pn}+off,good,0.)',
            f'    xi=tl.load(X+batch*{2*pn}+{pn}+off,good,0.)',
            f'    wr=tl.load(TW+off,pf<{pp},0.)',
            f'    wi=tl.load(TW+{pn}+off,pf<{pp},0.)',
            '    re,im=xr*wr-xi*wi,xr*wi+xi*wr',
            f'    re=tl.reshape(re,({points},))',f'    im=tl.reshape(im,({points},))']
  for d in range(g):
   lines += [f'    r{start+d}=al.extract_slice(re,[{d*bp*c}],[{bp*c}],[1])',
             f'    i{start+d}=al.extract_slice(im,[{d*bp*c}],[{bp*c}],[1])']
  start+=g
 lines += _emit_natural_order_codelet_call('    ',r,direction)
 for start,g in groups:
  points=g*bp*c
  lines += [f'    re=tl.zeros(({points},),tl.float32)',f'    im=tl.zeros(({points},),tl.float32)']
  for d in range(g):
   lines += [f'    re=al.insert_slice(re,r{start+d},[{d*bp*c}],[{bp*c}],[1])',
             f'    im=al.insert_slice(im,i{start+d},[{d*bp*c}],[{bp*c}],[1])']
  lines += [f'    d=(tl.arange(0,{g})+{start})[:,None,None]',
            f'    out=j*{r*pp}+d*{pp}+pf',
            f'    tl.store(Y+batch*{2*pn}+out,tl.reshape(re,({g},{bp},{c})),good)',
            f'    tl.store(Y+batch*{2*pn}+{pn}+out,tl.reshape(im,({g},{bp},{c})),good)']
 return '\n'.join(lines)+'\n'


def build_cube_prefix_dif(factors,direction='forward',vector_first17=False,batch_pack=False,vector_second15=False,global_inverse=False,matrix_first17=False,prefix_pitch=1,dense_prefix=False,codelet_reuse=False):
 if dense_prefix:
  if batch_pack or vector_second15 or matrix_first17 or prefix_pitch!=1:
   raise ValueError('Dense short prefix uses a single-batch Cube DIF layout')
  source,specs=build_cube_prefix_dif(factors,direction,vector_first17=vector_first17,
      global_inverse=global_inverse)
  from .npu_dense_prefix import replace_prefix
  return replace_prefix(source,specs,factors,conjugate_output=global_inverse)
 if prefix_pitch!=1:
  if prefix_pitch!=32 or not (vector_first17 and vector_second15 and batch_pack):
   raise ValueError('Aligned prefix pitch requires a batch-grouped vector 17/15 prefix')
  source,specs=build_cube_prefix_dif(factors,direction,vector_first17=vector_first17,
      batch_pack=batch_pack,vector_second15=vector_second15,global_inverse=global_inverse,
      matrix_first17=matrix_first17,codelet_reuse=codelet_reuse)
  n=math.prod(factors);previous=factors[0];m=n//(previous*factors[1])
  pitch=math.ceil(m/prefix_pitch)*prefix_pitch;storage=previous*factors[1]*pitch
  if pitch==m:return source,specs
  def second(body):
   old=f'dst=d*{previous*m}+pf*{m}+q'
   if old not in body:raise ValueError('Prefix pitch requires a vector second stage')
   body=body.replace(old,f'dst=d*{previous*pitch}+pf*{pitch}+q')
   return body.replace(f'Y+b*{2*n}+{n}+dst',f'Y+b*{2*storage}+{storage}+dst').replace(f'Y+b*{2*n}+dst',f'Y+b*{2*storage}+dst')
  def switch(body):
   body=body.replace(f'off=row[:,None]*{m}+col[None,:]',f'off=row[:,None]*{pitch}+col[None,:]')
   return body.replace(f'X+batch*{2*n}+{n}+off',f'X+batch*{2*storage}+{storage}+off').replace(f'X+batch*{2*n}+off',f'X+batch*{2*storage}+off')
  for name,transform in [(specs[1]['name'],second),('switch_axes',switch)]:
   source,count=re.subn(r'@triton.jit\ndef '+name+r'\([^\n]*\):\n[\s\S]*?(?=\n@triton.jit|\Z)',lambda match:transform(match.group(0)),source)
   if count!=1:raise ValueError('Missing prefix layout stage')
  for stage in (0,1):specs[stage]=dict(specs[stage],physical_n=max(specs[stage]['physical_n'],storage))
  return source,specs
 if matrix_first17 and (not vector_first17 or not batch_pack):
  raise ValueError('Half-matrix radix17 requires batch-grouped vector-17 DIF')
 if global_inverse:
  if direction!='inverse' or (batch_pack and not vector_first17):
   raise ValueError('Global inverse requires inverse grouped vector DIF')
  source,specs=build_cube_prefix_dif(factors,'forward',vector_first17=vector_first17,
      batch_pack=batch_pack,vector_second15=vector_second15,matrix_first17=matrix_first17,codelet_reuse=codelet_reuse)
  from .npu_radix17 import component_first_source
  from .kernels_npu_dif_aligned import _fused_finish
  first=specs[0]
  if matrix_first17:
   from .npu_radix17_matrix import half_matrix_first_source
   body=half_matrix_first_source(first,math.prod(factors),conjugate_output=True)
  elif batch_pack:
   body=component_first_source(first,math.prod(factors),'inverse',conjugate_output=True)
  else:
   if (factors[0] not in _NATURAL_ORDER_CODELET_RADICES or
       (factors[0]>=13 and not vector_first17) or
       os.environ.get('FLAGFFT_NPU_DIF_GROUP_BRANCHES','1')=='0'):
    raise ValueError('Global inverse needs a grouped vector first stage')
   body=_grouped_first_stage(first,math.prod(factors),'forward',512)
   # Negate each interleaved load group once before slicing its codelet lanes.
   body=re.sub(r'(    real,imag=tl.split[^\n]+\n)',r'\1    imag=-imag\n',body)
  source=re.sub(r'@triton.jit\ndef '+first['name']+r'\([^\n]*\):\n[\s\S]*?(?=\n@triton.jit|\Z)',lambda _:body,source)
  last=specs[-1]
  if last['name']!='fused_finish':raise ValueError('Global inverse requires a fused finish')
  body,last=_fused_finish(last,'forward',conjugate_output=True)
  source=re.sub(r'@triton.jit\ndef fused_finish\([^\n]*\):\n[\s\S]*?(?=\n@triton.jit|\Z)',lambda _:body,source)
  specs[-1]=last
  return source,specs
 if len(factors)<4 or not (2<=factors[0]<=19 and 11<=factors[1]<=15 and factors[0]*factors[1]>=64):
  raise ValueError('Chirped DIF prefix needs two early stages and a large second radix')
 source,specs=build_aligned_vector_dif(factors,direction)
 source+='\nimport triton.language.extra.cann.extension as al\n'
 n=math.prod(factors);r=factors[0];m=n//r;pr=max(16,1<<(r-1).bit_length());c=512;pn=specs[0]['physical_n'];tiles=(m+c-1)//c
 s0=specs[0];assert s0['stage']==0
 if vector_first17 and factors[0]!=17:
  raise ValueError('The vector-17 prefix requires first radix 17')
 matrix_first=factors[0]>=13 and not vector_first17
 group_branches=os.environ.get('FLAGFFT_NPU_DIF_GROUP_BRANCHES','1')!='0'
 original_first=re.search(r'@triton.jit\ndef '+s0['name']+r'\([^\n]*\):\n[\s\S]*?(?=\n@triton.jit|\Z)',source).group(0)
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
 second=specs[1];rr=second['r'];prev=factors[0];mm=second['m'];pp=max(16,1<<(rr-1).bit_length());cc=min(1024,max(32,1<<(mm-1).bit_length()));tt=(mm+cc-1)//cc
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
 if vector_second15:
  if rr!=15:raise ValueError('The vector second axis requires radix 15')
  from .npu_vector_second import vector_second_source
  body=vector_second_source(chirp,prev,n,direction,min(512,1<<(mm-1).bit_length()))
  source=re.sub(r'@triton.jit\ndef chirped_second\([^\n]*\):\n[\s\S]*?(?=\n@triton.jit|\Z)',lambda _:body,source)
 result=[prep,cube,chirp]+specs[2:]
 if not matrix_first:
  for name in ('planar_prepare','cube_first'):
   source=re.sub(r'@triton.jit\ndef '+name+r'\([^\n]*\):\n[\s\S]*?(?=\n@triton.jit|\Z)','',source)
  source+='\n'+original_first
  result=[dict(s0,auto=vector_first17)]+result[2:]
  if group_branches and factors[0] in _NATURAL_ORDER_CODELET_RADICES:
   first=result[0]
   wide7=(batch_pack and factors[0]==7 and
          os.environ.get('FLAGFFT_NPU_DIF_FIRST_AUTO_BLOCK','1')!='0' and
          os.environ.get('FLAGFFT_NPU_DIF_FIRST7_WIDE','1')!='0')
   columns=1024 if wide7 else 512
   body=_grouped_first_stage(first,n,direction,columns)
   if (vector_first17 and batch_pack and
       os.environ.get('FLAGFFT_NPU_DIF_COMPONENT17','1')!='0'):
    from .npu_radix17 import component_first_source
    if matrix_first17:
     from .npu_radix17_matrix import half_matrix_first_source
     body=half_matrix_first_source(first,n)
    else:
     body=component_first_source(first,n,direction)
   source=re.sub(r'@triton.jit\ndef '+first['name']+r'\([^\n]*\):\n[\s\S]*?(?=\n@triton.jit|\Z)',lambda _:body,source)
   first['grid']=((m+columns-1)//columns,)
 for z in result:
  if z.get('stage')==2:z['deferred_twiddle']=True
  radix=z.get('r',0);physical=z.get('physical_prefix',0)
  # Keep the whole-radix matrices within the validated 4096-element UB budget.
  coalesced=(radix>=10 and radix in _NATURAL_ORDER_CODELET_RADICES and 0<physical<=256
             and (1<<(radix-1).bit_length())*(1<<(physical-1).bit_length())<=4096)
  # Group radices with substantial power-of-two padding so DMA groups
  # contain actual branches. For whole-prefix tiles, also avoid materializing
  # three or more empty rows before the codelet's slices and inserts.
  # Retain at most 512 contiguous columns.
  grouped=(9<=radix<=16 and radix in _NATURAL_ORDER_CODELET_RADICES
           and physical>0 and (physical>256 or group_branches)
           and (2*(1<<(radix-1).bit_length())>=3*radix or
                (physical<=256 and (1<<(radix-1).bit_length())-radix>=3))
           and radix*512<=8192)
  if z['phase']=='late' and z['name']!='fused_finish' and (coalesced or grouped):
   columns=min(512,1<<(physical-1).bit_length())
   body=_grouped_late_stage(z,direction,columns) if grouped else _coalesced_late_stage(z,direction)
   source=re.sub(r'@triton.jit\ndef '+z['name']+r'\([^\n]*\):\n[\s\S]*?(?=\n@triton.jit|\Z)',lambda _:body,source)
 if batch_pack:
  for z in result:
   packed13=(z.get('r')==13 and z.get('physical_prefix')==256
             and os.environ.get('FLAGFFT_NPU_DIF_BATCH_PACK13','1')!='0')
   if z['phase']=='late' and z['name']!='fused_finish' and (z['r'] in (9,10) or packed13):
    body=_batch_grouped_late_stage(z,direction,2 if packed13 else 4,256 if packed13 else 128)
    source=re.sub(r'@triton.jit\ndef '+z['name']+r'\([^\n]*\):\n[\s\S]*?(?=\n@triton.jit|\Z)',lambda _:body,source)
 if (9 in factors and os.environ.get('FLAGFFT_NPU_DIF_FAST9','1')!='0'
     and os.environ.get('FLAGFFT_NPU_SYMMETRIC_CODELETS','1')!='0'):
  from .npu_radix9 import radix9_source
  source+='\n'+radix9_source()
 if (os.environ.get('FLAGFFT_NPU_DIF_COPRIME','1')!='0'
     and os.environ.get('FLAGFFT_NPU_SYMMETRIC_CODELETS','1')!='0'):
  from .npu_coprime import coprime_source
  if 10 in factors:source+='\n'+coprime_source(2,5)
  if vector_second15:source+='\n'+coprime_source(3,5)
 if codelet_reuse:
  if not batch_pack:raise ValueError('DIF codelet reuse requires batch grouping')
  from .npu_dif_codelets import dif_codelet_source
  source+='\n'+dif_codelet_source(factors)
 if vector_first17:
  from .npu_radix17 import radix17_source
  source+='\n'+radix17_source()
 return source,result
