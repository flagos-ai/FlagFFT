# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""DIF with an aligned physical prefix and a final compact complex store."""
import math,re,os
from flagfft_codegen.kernels_npu_dif import build_vector_dif,pad
from flagfft_codegen.kernels_common import _NATURAL_ORDER_CODELET_RADICES
from flagfft_codegen.kernels_leaf import _emit_natural_order_codelet_call,_emit_radix16_codelet_call,_emit_local_mixed_codelet_call


def _fused_finish(spec, direction, conjugate_output=False):
 """Keep the last butterfly in UB and write compact complex rows directly."""
 radix=spec['r'];prefix=spec['prefix'];physical_prefix=spec['physical_prefix']
 padded=spec['padded_prefix'];initial=spec['initial_prefix']
 physical_n=spec['physical_n'];n=prefix*radix
 # Bound butterfly state by the radix and padded row width. Larger
 # row groups amortize the compact output transfer without growing UB
 # past the same 4096-real-element budget used by the coalesced path.
 rows=1
 while rows<8 and 2*rows*radix*padded<=4096:rows*=2
 groups=physical_prefix//padded;points=rows*padded
 single_gather=os.environ.get('FLAGFFT_NPU_DIF_SINGLE_GATHER_FINISH','1')!='0'
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
 if conjugate_output:lines += [f'    i{d}=-i{d}' for d in range(radix)]
 # Compact in UB, then issue a flat store. A masked 2D store would
 # retain the unaligned logical row stride after removing prefix padding.
 lines += [f'    word=tl.arange(0,{2*points})',
           f'    valid_word=word<{rows*2*initial}',
           (f'    take=tl.where(valid_word,(word&1)*{points}+word//2+(word//{2*initial})*{padded-initial},0)' if single_gather else
            f'    take=tl.where(valid_word,word+(word//{2*initial})*{2*(padded-initial)},0)'),
           f'    out=tl.program_id(0)*{rows*2*initial}+word']
 if spec.get('coprime_prefix'):
  from .npu_coarse_gt import finish_take
  lines[-2]='    take=tl.where(valid_word,'+finish_take(initial,padded,points,single_gather)+',0)'
 for d in range(radix):
  if single_gather:
   lines += [f'    value=tl.zeros(({2*points},),tl.float32)',
             f'    value=al.insert_slice(value,tl.reshape(r{d},({points},)),[0],[{points}],[1])',
             f'    value=al.insert_slice(value,tl.reshape(i{d},({points},)),[{points}],[{points}],[1])']
  else:
   lines += [f'    value=tl.reshape(tl.join(tl.reshape(r{d},({points},)),tl.reshape(i{d},({points},))),({2*points},))']
  lines += [f'    packed=tl.gather(value,take,0)',
            f'    tl.store(Y+batch*{2*n}+{2*d*prefix}+out,packed,valid_word&(out<{2*prefix}))']
 return '\n'.join(lines)+'\n',dict(spec,name='fused_finish',grid=((groups+rows-1)//rows,))


def build_aligned_vector_dif(factors,direction='forward'):
 source,old=build_vector_dif(factors,direction);n=math.prod(factors)
 switch=next(s for s in old if s['phase']=='transpose');p0=next(s['prefix'] for s in old if s['phase']=='late');padded=pad(p0);rest=n//p0;physical_n=rest*padded
 first=source.index('@triton.jit\ndef vector_stage_0');chunks=[source[:first]];specs=[]
 fused_finish=os.environ.get('FLAGFFT_NPU_DIF_FUSED_FINISH')=='1'
 if fused_finish and os.environ.get('FLAGFFT_NPU_DIF_SINGLE_GATHER_FINISH','1')!='0':
  chunks.append('import triton.language.extra.cann.extension as al\n')
 for s in old:
  if s['phase']!='early':continue
  match=re.search(r'@triton.jit\ndef '+s['name']+r'\([^\n]*\):\n[\s\S]*?(?=\n@triton.jit|\Z)',source);chunks.append(match.group(0));specs.append(dict(s,physical_n=physical_n))
 tile=int(os.environ.get("FLAGFFT_NPU_DIF_TRANSPOSE_TILE","32"))
 if tile not in (32,64):raise ValueError("DIF transpose tile must be 32 or 64")
 # A complete padded output row is a contiguous GM transfer. Large
 # workspaces retain enough independent tiles to widen the input DMA.
 dense=os.environ.get("FLAGFFT_NPU_DIF_DENSE_TRANSPOSE","1")!="0" and padded<=256
 transpose_points=int(os.environ.get('FLAGFFT_NPU_DIF_TRANSPOSE_POINTS',
     '8192' if physical_n>=262144 else '4096'))
 if transpose_points not in (4096,8192):
  raise ValueError('DIF transpose points must be 4096 or 8192')
 tile_rows=padded if dense else tile
 tile_cols=transpose_points//tile_rows if dense else tile
 chunks.append(f'''@triton.jit
def switch_axes(X,TW,Y):
    batch=tl.program_id(2).to(tl.int64)
    row=tl.program_id(0)*{tile_rows}+tl.arange(0,{tile_rows})
    col=tl.program_id(1)*{tile_cols}+tl.arange(0,{tile_cols})
    off=row[:,None]*{rest}+col[None,:]
    valid=(row[:,None]<{p0})&(col[None,:]<{rest})
    r=tl.load(X+batch*{2*n}+off,valid,0.)
    i=tl.load(X+batch*{2*n}+{n}+off,valid,0.)
    dst=col[:,None]*{padded}+row[None,:]
    valid=(col[:,None]<{rest})&(row[None,:]<{padded})
    tl.store(Y+batch*{2*physical_n}+dst,tl.trans(r),valid)
    tl.store(Y+batch*{2*physical_n}+{physical_n}+dst,tl.trans(i),valid)
''')
 specs.append(dict(name='switch_axes',phase='transpose',grid=((padded+tile_rows-1)//tile_rows,(rest+tile_cols-1)//tile_cols),physical_n=physical_n))
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
