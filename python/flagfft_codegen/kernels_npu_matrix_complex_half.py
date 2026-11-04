# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Factor arbitrary complex matrix DFTs into two real half-spectrum DFTs."""
import json
from pathlib import Path
from .metadata import _module_source
from .artifacts import write_text_atomic

def build_complex_half(a,b,first_rows=128,first_cols=32,last_rows=64,last_cols=32):
 if not (16<=a<=256 and 16<=b<=256):raise ValueError('Complex matrix axes require lengths 16..256')
 source=''
 # U=DFT(real input), V=DFT(imag input): F(k)=U(k)+i*V(k),
 # F(-k)=conj(U(k))+i*conj(V(k)). Positive and reflected output
 # tiles use ascending GM addresses; only the bounded UB tile is reversed.
 for first,length,rows,pitch,m,c,op in [(True,a,b,b,first_rows,first_cols,'cube0_complex_half_pack'),(False,b,a,a,last_rows,last_cols,'cube1_complex_half_finish')]:
  kpad=1<<(length-1).bit_length();rowpad=1<<(rows-1).bit_length();h=length//2+1;hp=1<<(h-1).bit_length();n=a*b;points=m*c
  args='P,W,T,Y' if first else 'P,W,Y'
  ls=['@triton.jit',f'def {op}({args}):','    batch=tl.program_id(2).to(tl.int64)',f'    row=tl.program_id(0)*{m}+tl.arange(0,{m})',f'    k=tl.arange(0,{kpad})',f'    freq=tl.program_id(1)*{c}+tl.arange(0,{c})',f'    word=tl.program_id(1)*{2*c}+tl.arange(0,{2*c})',f'    off=batch*{rows*2*kpad}+row[:,None]*{2*kpad}+k[None,:]',f'    xr=tl.load(P+off,row[:,None]<{rows},0.)',f'    xi=tl.load(P+{kpad}+off,row[:,None]<{rows},0.)',f'    w=tl.load(W+k[:,None]*{2*hp}+word[None,:],word[None,:]<{2*h},0.)','    u=tl.dot(xr,w,input_precision="ieee")','    v=tl.dot(xi,w,input_precision="ieee")',f'    ur,ui=tl.split(tl.reshape(u,({points},2)))',f'    vr,vi=tl.split(tl.reshape(v,({points},2)))',f'    posr=tl.reshape(ur-vi,({m},{c}))',f'    posi=tl.reshape(ui+vr,({m},{c}))',f'    reverse=tl.arange(0,{points})^{c-1}',f'    negr=tl.reshape(tl.gather(ur+vi,reverse,0),({m},{c}))',f'    negi=tl.reshape(tl.gather(vr-ui,reverse,0),({m},{c}))',f'    negfreq={length}-(tl.program_id(1)*{c}+{c-1})+tl.arange(0,{c})']
  if first:
   for label,freq in [('pos','freq'),('neg','negfreq')]:
    valid=f'(row[:,None]<{rows})&({freq}[None,:]>=0)&({freq}[None,:]<{length})'
    ls += [f'    trow=row[:,None]*{length}+{freq}[None,:]',f'    tr=tl.load(T+trow,{valid},0.)',f'    ti=tl.load(T+{n}+trow,{valid},0.)',f'    re,im={label}r*tr-{label}i*ti,{label}r*ti+{label}i*tr',f'    dst=batch*{length*2*rowpad}+{freq}[:,None]*{2*rowpad}+row[None,:]']
    good=f'(row[None,:]<{rowpad})&({freq}[:,None]>=0)&({freq}[:,None]<{h})' if label=='pos' else f'(row[None,:]<{rowpad})&({freq}[:,None]>={h})&({freq}[:,None]<{length})'
    ls += [f'    tl.store(Y+dst,tl.trans(re),{good})',f'    tl.store(Y+{rowpad}+dst,tl.trans(im),{good})']
  else:
   for label,freq in [('pos','freq'),('neg','negfreq')]:
    ls += [f'    re=tl.trans({label}r)',f'    im=tl.trans({label}i)',f'    packed=tl.reshape(tl.join(re,im),({c},{2*m}))',f'    outword=tl.program_id(0)*{2*m}+tl.arange(0,{2*m})[None,:]',f'    dst=batch*{2*n}+{freq}[:,None]*{2*rows}+outword']
    good=f'(outword<{2*rows})&({freq}[:,None]>=0)&({freq}[:,None]<{h})' if label=='pos' else f'(outword<{2*rows})&({freq}[:,None]>={h})&({freq}[:,None]<{length})'
    ls += [f'    tl.store(Y+dst,packed,{good})']
  source+='\n'+'\n'.join(ls)+'\n'
 return _module_source(source)


def emit_complex_half(a,b,phase,direction,out_dir):
 narrow=phase=='cube1_complex_half_finish_narrow'
 if phase not in ('cube0_complex_half_pack','cube1_complex_half_finish','cube1_complex_half_finish_narrow') or direction not in ('forward','inverse'):
  raise ValueError('Invalid complex half-spectrum matrix phase or direction')
 path=Path(out_dir)/f'npu_matrix_fourstep_{a}_{b}_{phase}_{direction}.py'
 path.parent.mkdir(parents=True,exist_ok=True);write_text_atomic(path,build_complex_half(a,b,last_rows=32 if narrow else 64))
 args=['P','W','T','Y'] if phase=='cube0_complex_half_pack' else ['P','W','Y']
 meta=dict(module_path=str(path),kernel_name=phase.removesuffix("_narrow"),arg_names=args,signature=','.join(['*fp32:16']*len(args)),num_warps=4,num_stages=1,batch_per_block=1,inner_pack=1,tle_fused_twiddle=True,kernel_type='four_step_row',dtype='complex64',direction=direction,length=a*b)
 write_text_atomic(path.with_suffix('.json'),json.dumps(meta,sort_keys=True));return meta
