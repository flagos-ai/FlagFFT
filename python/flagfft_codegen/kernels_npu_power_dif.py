# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded power-of-two DIF axes with contiguous logical vector tiles.

Each program transforms 4096 real lane elements. A tile can include multiple
remaining-coordinate groups while the prefix is narrow; its output permutation
then becomes a dense store. Larger prefixes retain wide contiguous row slices.
Twiddles and structured FFT roots share each pass's read-only table argument.
"""
import math
from flagfft_codegen.kernels_npu_structured import FFT_HELPERS

def build(factors,direction='forward',terminal8=False,global_inverse=False,workspace_padding=0,cube_finish=False,cube_finish_batch=1,middle_radix8=False,first_batch=1,first_phase=False):
 if type(workspace_padding) is not int or workspace_padding not in (0,512):
  raise ValueError('Power DIF workspace padding must be 0 or 512 floats')
 if global_inverse and direction!='inverse':raise ValueError('Global inverse requires inverse DIF')
 if type(cube_finish_batch) is not int or cube_finish_batch not in (1,8,32):
  raise ValueError('Unsupported power Cube batch group')
 if cube_finish_batch!=1 and not cube_finish:raise ValueError('Batch coefficient reuse requires Cube finish')
 if cube_finish and (len(factors)!=3 or factors[0]!=64 or not 32<=factors[1]<=64 or factors[2]!=128 or workspace_padding!=512):
  raise ValueError('Power Cube finish requires bounded vector axes and padded workspace')
 if middle_radix8 and not (cube_finish and factors[1]==64):
  raise ValueError('Radix-eight middle axis requires a 64-point middle Cube-finish schedule')
 if type(first_batch) is not int or first_batch not in (1,8,16):
  raise ValueError('Unsupported power first-axis batch group')
 if first_batch>1 and not cube_finish:
  raise ValueError('First-axis batch pipeline requires Cube finish')
 if first_phase and first_batch==1:
  raise ValueError('Producer phase requires the first-axis batch pipeline')
 n=math.prod(factors);workspace_n=n+workspace_padding;prefix=1;previous=1;specs=[]
 source='import triton\nimport triton.language as tl\nimport triton.language.extra.cann.extension as al\n'+FFT_HELPERS
 if terminal8 or middle_radix8:
  from .npu_power_terminal import terminal8_source
  source+='\n'+terminal8_source()
 if middle_radix8:
  from .npu_power_radix8 import radix8_axis_source
  source+='\n'+radix8_axis_source()
 axis='power_fft_axis' if terminal8 else 'fft_axis_sliced'
 for stage,r in enumerate(factors):
  if r<4 or r>128 or r&(r-1):raise ValueError('Power DIF axes require radix 4..128')
  m=n//(prefix*r);c=4096//r;points=r*c;tiles=(m+c-1)//c if stage==0 else (m*prefix+c-1)//c
  name=f'axis_{stage}';last=stage==len(factors)-1
  if stage==0 and first_batch>1:name+='_batch'+str(first_batch)
  selected_axis='power_fft64_8x8' if middle_radix8 and stage==1 else axis
  if middle_radix8 and stage==1:name+='_radix8'
  if first_phase and stage<2:name+='_prephase'
  if last and cube_finish:
   from .npu_power_cube_finish import finish_source
   spec=dict(name='power_cube_finish_batch' if cube_finish_batch>1 else 'power_cube_finish',
       batch_pack=cube_finish_batch,phase='late',stage=stage,r=r,m=m,
       prefix=prefix,previous_prefix=previous,c=64,grid=(prefix//64,),physical_n=workspace_n)
   source+='\n'+finish_source(spec,conjugate_output=global_inverse)
   specs.append(spec)
   continue
  ls=['@triton.jit',f'def {name}(X,TW,Y):',f'    W=TW+{0 if stage==0 or (stage==1 and first_phase) else 2*n}','    batch=tl.program_id(1).to(tl.int64)',f'    d=tl.arange(0,{r})[:,None]']
  if stage==0:
   ls += [f'    q=tl.program_id(0)*{c}+tl.arange(0,{c})[None,:]',f'    word=tl.program_id(0)*{2*c}+tl.arange(0,{2*c})[None,:]',f'    raw=tl.load(X+batch*{2*n}+d*{2*m}+word,word<{2*m},0.)',f'    re,im=tl.split(tl.reshape(raw,({points},2)))']
  else:
   ls += [f'    q=tl.program_id(0)*{c}+tl.arange(0,{c})[None,:]',f'    j=tl.program_id(0)//{max(1,prefix//c)}',f'    pf=(tl.program_id(0)%{max(1,prefix//c)}*{c}+tl.arange(0,{c}))[None,:]',f'    off=d*{m*prefix}+q',f'    rr=tl.load(X+batch*{2*workspace_n}+off,q<{m*prefix},0.)',f'    ii=tl.load(X+batch*{2*workspace_n}+{workspace_n}+off,q<{m*prefix},0.)']
   if stage==1 and first_phase:
    ls += [f'    re=tl.reshape(rr,({points},))',f'    im=tl.reshape(ii,({points},))']
   else:
    ls += [f'    wr=tl.load(TW+off,q<{m*prefix},0.)',f'    wi=tl.load(TW+{n}+off,q<{m*prefix},0.)',f'    re=tl.reshape(rr*wr-ii*wi,({points},))',f'    im=tl.reshape(rr*wi+ii*wr,({points},))']
  ls += ['    r,i=re,im']
  if direction=='inverse' and (not global_inverse or stage==0):ls += ['    i=-i']
  ls += [f'    r,i={selected_axis}(r,i,W,{points},{r},{c},{r.bit_length()-1})']
  if direction=='inverse' and (not global_inverse or last):ls += ['    i=-i']
  if stage==0:
   ls += [f'    row=tl.program_id(0)*{c}+tl.arange(0,{c})[:,None]',f'    freq=tl.arange(0,{r})[None,:]',f'    dst=row*{r}+freq']
   if first_phase:
    # Move the same rotation across the GM boundary. Apply it to the
    # transposed producer tile so the following butterfly starts at once.
    root_words=6*((r.bit_length()-1)//2)*(points//4)
    ls += [f'    pr=tl.trans(tl.reshape(r,({r},{c})))',f'    pi=tl.trans(tl.reshape(i,({r},{c})))',
           f'    wr=tl.load(TW+{root_words}+dst,row<{m},0.)',
           f'    wi=tl.load(TW+{root_words+n}+dst,row<{m},0.)',
           '    re,im=pr*wr-pi*wi,pr*wi+pi*wr',
           f'    tl.store(Y+batch*{2*workspace_n}+dst,re,row<{m})',
           f'    tl.store(Y+batch*{2*workspace_n}+{workspace_n}+dst,im,row<{m})']
   else:
    ls += [f'    tl.store(Y+batch*{2*workspace_n}+dst,tl.trans(tl.reshape(r,({r},{c}))),row<{m})',f'    tl.store(Y+batch*{2*workspace_n}+{workspace_n}+dst,tl.trans(tl.reshape(i,({r},{c}))),row<{m})']
  elif not last:
   if prefix<c:
    ls += [f'    re=tl.reshape(tl.permute(tl.reshape(r,({r},{c//prefix},{prefix})),1,0,2),({points},))',f'    im=tl.reshape(tl.permute(tl.reshape(i,({r},{c//prefix},{prefix})),1,0,2),({points},))',f'    dst=tl.program_id(0)*{points}+tl.arange(0,{points})',f'    tl.store(Y+batch*{2*workspace_n}+dst,re,dst<{n})',f'    tl.store(Y+batch*{2*workspace_n}+{workspace_n}+dst,im,dst<{n})']
   else:
    ls += [f'    dst=(j*{r}+d)*{prefix}+pf',f'    tl.store(Y+batch*{2*workspace_n}+dst,tl.reshape(r,({r},{c})),pf<{prefix})',f'    tl.store(Y+batch*{2*workspace_n}+{workspace_n}+dst,tl.reshape(i,({r},{c})),pf<{prefix})']
  else:
   chunk=min(1024,points);rows=chunk//c
   for start in range(0,points,chunk):
    ls += [f'    pr=al.extract_slice(r,[{start}],[{chunk}],[1])',f'    pi=al.extract_slice(i,[{start}],[{chunk}],[1])',f'    out=tl.reshape(tl.join(pr,pi),({rows},{2*c}))',f'    freq=(tl.arange(0,{rows})+{start//c})[:,None]',f'    word=(tl.program_id(0)%{tiles}*{2*c}+tl.arange(0,{2*c}))[None,:]',f'    tl.store(Y+batch*{2*n}+freq*{2*prefix}+word,out,word<{2*prefix})']
  if stage==0 and first_batch>1:
   # Keep the 4096-element butterfly state fixed. A bounded batch loop gives
   # the compiler consecutive independent inputs to double buffer in UB;
   # the final group runs only its live batches, with no masked extra FFTs.
   ls[1]=ls[1].replace('):',',NB):')
   start=next(i for i,line in enumerate(ls) if line.strip().startswith('batch='))
   ls=ls[:start]+[f'    begin=tl.program_id(1)*{first_batch}','    step=0',
       f'    while step<tl.minimum({first_batch},NB-begin):',
       '        batch=(begin+step).to(tl.int64)']+['    '+line for line in ls[start+1:]]+['        step+=1']
  source+='\n'+'\n'.join(ls)+'\n'
  spec=dict(name=name,phase='early' if stage==0 else 'late',stage=stage,r=r,m=m,prefix=prefix,previous_prefix=previous,c=c,grid=(tiles,),physical_n=workspace_n)
  if stage==0 and first_batch>1:spec.update(batch_pack=first_batch,num_stages=2)
  specs.append(spec)
  previous=prefix;prefix*=r
 return source,specs
