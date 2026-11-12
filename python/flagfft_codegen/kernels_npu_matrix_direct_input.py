# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Read contiguous API samples directly into an IEEE first-axis matrix DFT.

The half-spectrum real-component reconstruction is shared with the packed
matrix path. Both the DFT basis and cross-axis phase table follow the
accumulator's frequency/sample orientation, including zero sample padding.
"""
import json
from pathlib import Path
from .metadata import _module_source
from .artifacts import write_text_atomic

def build_direct_first(a,b,combined=False):
 if not (128<=a<=256 and 128<=b<=256):raise ValueError('Direct matrix input requires axes 128..256')
 pa=1<<(a-1).bit_length();pb=1<<(b-1).bit_length();h=a//2+1;hp=1<<(h-1).bit_length();m=c=32;n=a*b;points=c*m
 name='cube0_complex_direct_combined' if combined else 'cube0_complex_direct_pack'
 ls=['@triton.jit',f'def {name}(X,W,T,Y):','    batch=tl.program_id(2).to(tl.int64)',f'    freq=tl.program_id(0)*{c}+tl.arange(0,{c})',f'    sample=tl.program_id(1)*{m}+tl.arange(0,{m})',f'    word=tl.program_id(1)*{2*m}+tl.arange(0,{2*m})',f'    k=tl.arange(0,{pa})',f'    x=tl.load(X+batch*{2*n}+k[:,None]*{2*b}+word[None,:],(k[:,None]<{a})&(word[None,:]<{2*b}),0.)']
 if combined:
  ls += [f'    blockrow=tl.program_id(0)*64+tl.arange(0,64)',f'    w=tl.load(W+blockrow[:,None]*{pa}+k[None,:])','    value=tl.dot(w,x,input_precision="ieee")','    co=al.extract_slice(value,[0,0],[32,64],[1,1])','    si=al.extract_slice(value,[32,0],[32,64],[1,1])']
 else:
  ls += [f'    wc=tl.load(W+freq[:,None]*{pa}+k[None,:],freq[:,None]<{h},0.)',f'    ws=tl.load(W+{hp*pa}+freq[:,None]*{pa}+k[None,:],freq[:,None]<{h},0.)','    co=tl.dot(wc,x,input_precision="ieee")','    si=tl.dot(ws,x,input_precision="ieee")']
 ls += [f'    cr,ci=tl.split(tl.reshape(co,({points},2)))',f'    sr,sim=tl.split(tl.reshape(si,({points},2)))',f'    posr=tl.reshape(cr-sim,({c},{m}))',f'    posi=tl.reshape(ci+sr,({c},{m}))',f'    reverse=tl.arange(0,{points})^{(c-1)*m}',f'    negr=tl.reshape(tl.gather(cr+sim,reverse,0),({c},{m}))',f'    negi=tl.reshape(tl.gather(ci-sr,reverse,0),({c},{m}))',f'    negfreq={a}-(tl.program_id(0)*{c}+{c-1})+tl.arange(0,{c})']
 for label,fr in [('pos','freq'),('neg','negfreq')]:
  outgood=f'(sample[None,:]<{pb})&({fr}[:,None]>=0)&({fr}[:,None]<{h})' if label=='pos' else f'(sample[None,:]<{pb})&({fr}[:,None]>={h})&({fr}[:,None]<{a})'
  ls += [f'    off={fr}[:,None]*{pb}+sample[None,:]','    tr=tl.load(T+off)',f'    ti=tl.load(T+{(pa+1)*pb}+off)',f'    re,im={label}r*tr-{label}i*ti,{label}r*ti+{label}i*tr',f'    dst=batch*{a*2*pb}+{fr}[:,None]*{2*pb}+sample[None,:]',f'    tl.store(Y+dst,re,{outgood})',f'    tl.store(Y+{pb}+dst,im,{outgood})']
 return _module_source('\n'.join(ls)+'\n')


def emit_direct_first(a,b,phase,direction,out_dir):
 if phase not in ('cube0_complex_direct_pack','cube0_complex_direct_combined') or direction not in ('forward','inverse'):
  raise ValueError('Invalid direct matrix input phase or direction')
 path=Path(out_dir)/f'npu_matrix_fourstep_{a}_{b}_{phase}_{direction}.py'
 path.parent.mkdir(parents=True,exist_ok=True);write_text_atomic(path,build_direct_first(a,b,combined=phase.endswith("combined")))
 meta=dict(module_path=str(path),kernel_name=phase,arg_names=['X','W','T','Y'],signature=','.join(['*fp32:16']*4),num_warps=4,num_stages=1,batch_per_block=1,inner_pack=1,tle_fused_twiddle=True,kernel_type='four_step_row',dtype='complex64',direction=direction,length=a*b)
 write_text_atomic(path.with_suffix('.json'),json.dumps(meta,sort_keys=True));return meta
