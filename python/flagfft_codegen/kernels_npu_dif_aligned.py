# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""DIF with an aligned physical prefix and a final compact complex store."""
import math,re,os
from flagfft_codegen.kernels_npu_dif import build_vector_dif,pad
from flagfft_codegen.kernels_common import _NATURAL_ORDER_CODELET_RADICES
from flagfft_codegen.kernels_leaf import _emit_natural_order_codelet_call,_emit_radix16_codelet_call,_emit_local_mixed_codelet_call


def _fused_finish(spec, direction):
 """Keep the last butterfly in UB and write compact complex rows directly."""
 radix=spec['r'];prefix=spec['prefix'];physical_prefix=spec['physical_prefix']
 padded=spec['padded_prefix'];initial=spec['initial_prefix']
 physical_n=spec['physical_n'];n=prefix*radix
 rows=4 if radix==2 else 2 if radix<=8 else 1
 groups=physical_prefix//padded;points=rows*padded
 lines=['@triton.jit','def fused_finish(X,TW,Y):',
        '    batch=tl.program_id(1).to(tl.int64)',
        f'    row=tl.program_id(0)*{rows}+tl.arange(0,{rows})[:,None]',
        f'    col=tl.arange(0,{padded})[None,:]',
        f'    off=row*{padded}+col',f'    good=row<{groups}']
 for d in range(radix):
  lines += [f'    r{d}=tl.load(X+batch*{2*physical_n}+{d*physical_prefix}+off,good,0.)',
            f'    i{d}=tl.load(X+batch*{2*physical_n}+{physical_n+d*physical_prefix}+off,good,0.)']
  # The zero sample always has twiddle one, including inverse transforms.
  if d:
   lines += [f'    wr=tl.load(TW+{d*physical_prefix}+off,good,0.)',
             f'    wi=tl.load(TW+{physical_n+d*physical_prefix}+off,good,0.)',
             f'    r{d},i{d}=r{d}*wr-i{d}*wi,r{d}*wi+i{d}*wr']
 if radix in _NATURAL_ORDER_CODELET_RADICES:
  lines += _emit_natural_order_codelet_call('    ',radix,direction)
 elif radix==16:
  lines += _emit_radix16_codelet_call('    ',direction)
 else:
  lines += _emit_local_mixed_codelet_call('    ',radix,direction)
 lines += [f'    word=tl.arange(0,{2*padded})[None,:]']
 for d in range(radix):
  lines += [f'    value=tl.reshape(tl.join(tl.reshape(r{d},({points},)),tl.reshape(i{d},({points},))),({rows},{2*padded}))',
            f'    tl.store(Y+batch*{2*n}+{2*d*prefix}+row*{2*initial}+word,value,(row<{groups})&(word<{2*initial}))']
 return '\n'.join(lines)+'\n',dict(spec,name='fused_finish',grid=((groups+rows-1)//rows,))


def build_aligned_vector_dif(factors,direction='forward'):
 source,old=build_vector_dif(factors,direction);n=math.prod(factors)
 switch=next(s for s in old if s['phase']=='transpose');p0=next(s['prefix'] for s in old if s['phase']=='late');padded=pad(p0);rest=n//p0;physical_n=rest*padded
 first=source.index('@triton.jit\ndef vector_stage_0');chunks=[source[:first]];specs=[]
 fused_finish=os.environ.get('FLAGFFT_NPU_DIF_FUSED_FINISH')=='1'
 for s in old:
  if s['phase']!='early':continue
  match=re.search(r'@triton.jit\ndef '+s['name']+r'\([^\n]*\):\n[\s\S]*?(?=\n@triton.jit|\Z)',source);chunks.append(match.group(0));specs.append(dict(s,physical_n=physical_n))
 tile=int(os.environ.get("FLAGFFT_NPU_DIF_TRANSPOSE_TILE","32"))
 if tile not in (32,64):raise ValueError("DIF transpose tile must be 32 or 64")
 chunks.append(f'''@triton.jit
def switch_axes(X,TW,Y):
    batch=tl.program_id(2).to(tl.int64)
    row=tl.program_id(0)*{tile}+tl.arange(0,{tile})
    col=tl.program_id(1)*{tile}+tl.arange(0,{tile})
    off=row[:,None]*{rest}+col[None,:]
    valid=(row[:,None]<{p0})&(col[None,:]<{rest})
    r=tl.load(X+batch*{2*n}+off,valid,0.)
    i=tl.load(X+batch*{2*n}+{n}+off,valid,0.)
    dst=col[:,None]*{padded}+row[None,:]
    valid=(col[:,None]<{rest})&(row[None,:]<{padded})
    tl.store(Y+batch*{2*physical_n}+dst,tl.trans(r),valid)
    tl.store(Y+batch*{2*physical_n}+{physical_n}+dst,tl.trans(i),valid)
''')
 specs.append(dict(name='switch_axes',phase='transpose',grid=((padded+tile-1)//tile,(rest+tile-1)//tile),physical_n=physical_n))
 for s in old:
  if s['phase']!='late':continue
  i=s['stage'];r=s['r'];m=s['m'];prefix=s['prefix'];pp=prefix//p0*padded;c=min(2048 if r==2 else 1024 if r<=8 else 512,pad(pp));tiles=(pp+c-1)//c
  spec=dict(s,name=f'switched_stage_{i}',grid=(m*tiles,),physical_prefix=pp,initial_prefix=p0,padded_prefix=padded,physical_n=physical_n)
  if fused_finish and i+1==len(factors):
   text,spec=_fused_finish(spec,direction)
   chunks.append(text);specs.append(spec)
   continue
  lines=['@triton.jit',f'def switched_stage_{i}(X,TW,Y):',f'    batch=tl.program_id(1).to(tl.int64)',f'    j=tl.program_id(0)//{tiles}',f'    pf=tl.program_id(0)%{tiles}*{c}+tl.arange(0,{c})',f'    valid=pf<{pp}']
  for d in range(r):
   lines += [f'    off=({d*m}+j)*{pp}+pf',f'    r{d}=tl.load(X+batch*{2*physical_n}+off,valid,0.)',f'    i{d}=tl.load(X+batch*{2*physical_n}+{physical_n}+off,valid,0.)',f'    wr=tl.load(TW+off,valid,0.)',f'    wi=tl.load(TW+{physical_n}+off,valid,0.)',f'    r{d},i{d}=r{d}*wr-i{d}*wi,r{d}*wi+i{d}*wr']
  if r in _NATURAL_ORDER_CODELET_RADICES:lines+=_emit_natural_order_codelet_call('    ',r,direction)
  elif r==16:lines+=_emit_radix16_codelet_call('    ',direction)
  else:lines+=_emit_local_mixed_codelet_call('    ',r,direction)
  for d in range(r):
   lines += [f'    off=j*{r*pp}+{d*pp}+pf',f'    tl.store(Y+batch*{2*physical_n}+off,r{d},valid)',f'    tl.store(Y+batch*{2*physical_n}+{physical_n}+off,i{d},valid)']
  chunks.append('\n'.join(lines)+'\n');specs.append(spec)
 if fused_finish:
  return '\n'.join(chunks),specs
 rows=4 if padded<=256 else 2;points=rows*padded
 chunks.append(f'''@triton.jit
def compact_finish(X,TW,Y):
    batch=tl.program_id(1).to(tl.int64)
    row=tl.program_id(0)*{rows}+tl.arange(0,{rows})[:,None]
    col=tl.arange(0,{padded})[None,:]
    off=row*{padded}+col
    r=tl.load(X+batch*{2*physical_n}+off,row<{rest},0.)
    i=tl.load(X+batch*{2*physical_n}+{physical_n}+off,row<{rest},0.)
    value=tl.reshape(tl.join(tl.reshape(r,({points},)),tl.reshape(i,({points},))),({rows},{2*padded}))
    word=tl.arange(0,{2*padded})[None,:]
    tl.store(Y+batch*{2*n}+row*{2*p0}+word,value,(row<{rest})&(word<{2*p0}))
''')
 specs.append(dict(name='compact_finish',phase='finish',grid=((rest+rows-1)//rows,),physical_n=physical_n))
 return '\n'.join(chunks),specs
