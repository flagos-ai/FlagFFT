# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Mixed-radix DIF FFTs with one switch of the contiguous vector axis.

Early passes vectorize remaining samples. A tiled exchange then makes
frequency prefixes contiguous, so late passes avoid short strided copies.
The codelets and unnormalized inverse convention are shared with other targets.
"""
import json
import os
from pathlib import Path
from .artifacts import write_text_atomic
"""DIF autosort vector FFT with contiguous global vectors in every pass."""
import math
from .kernels_common import codelet_radices_for,_NATURAL_ORDER_CODELET_RADICES
from .kernels_leaf import _emit_natural_order_codelet_call,_emit_local_mixed_codelet_call,_emit_radix16_codelet_call
from .metadata import _module_source


def pad(n):return 1<<(n-1).bit_length()
def floorpow(n):return 1<<(max(1,n).bit_length()-1)


def _flat_source(factors,direction='forward'):
 n=math.prod(factors);prefix=1;previous_prefix=1;lines=[];specs=[]
 for stage,r in enumerate(factors):
  m=n//(prefix*r);first=stage==0;last=stage==len(factors)-1;small=m<128
  g=floorpow(min(max(1,256//m),max(1,4096//(r*m)))) if small else 1
  c=pad(g*m) if small else min(512,pad(m));raw=pad(g*r*m)
  name=f'vector_stage_{stage}'
  lines += ['@triton.jit',f'def {name}(X,TW,Y):',f'    q=tl.arange(0,{c})']
  if small:
   lines += [f'    base=tl.program_id(0)*{g}',f'    w=tl.arange(0,{2*raw if first else raw})']
   if first:
    lines += [f'    v=tl.load(X+base*{2*r*m}+w,(w<{2*g*r*m})&(w<2*({n}-base*{r*m})),0.)',f'    vr,vi=tl.split(tl.reshape(v,({raw},2)))']
   else:
    lines += [f'    vr=tl.load(X+base*{r*m}+w,(w<{g*r*m})&(w<{n}-base*{r*m}),0.)',f'    vi=tl.load(X+{n}+base*{r*m}+w,(w<{g*r*m})&(w<{n}-base*{r*m}),0.)']
   lines += [f'    pf=base+q//{m}',f'    j=q%{m}',f'    outbase=base*{m}',f'    tc=outbase+q',f'    valid=(q<{g*m})&(pf<{prefix})']
   for d in range(r):
    lines += [f'    ix=tl.minimum((q//{m})*{r*m}+{d*m}+j,{raw-1})',f'    r{d}=tl.where(valid,tl.gather(vr,ix,0),0.)',f'    i{d}=tl.where(valid,tl.gather(vi,ix,0),0.)']
  else:
   tiles=(m+c-1)//c
   lines += [f'    pf=tl.program_id(0)//{tiles}',f'    tile=tl.program_id(0)%{tiles}',f'    j=tile*{c}+q',f'    outbase=pf*{m}+tile*{c}',f'    tc=outbase+q',f'    valid=j<{m}']
   for d in range(r):
    if first:
     lines += [f'    w=tile*{2*c}+tl.arange(0,{2*c})',f'    v=tl.load(X+(pf*{r*m}+{d*m})*2+w,w<{2*m},0.)',f'    r{d},i{d}=tl.split(tl.reshape(v,({c},2)))']
    else:
     lines += [f'    r{d}=tl.load(X+pf*{r*m}+{d*m}+j,valid,0.)',f'    i{d}=tl.load(X+{n}+pf*{r*m}+{d*m}+j,valid,0.)']
  if stage:
   previous_r=factors[stage-1];length=r*m
   if small:
    rows=min(previous_r,1+(g-1+previous_prefix-1)//previous_prefix);window=rows*length;pwindow=pad(window)
    lines += [f'    twbase=base//{previous_prefix}*{length}',f'    w=tl.arange(0,{pwindow})',f'    wr=tl.load(TW+twbase+w,(w<{window})&(twbase+w<{previous_r*length}),0.)',f'    wi=tl.load(TW+{previous_r*length}+twbase+w,(w<{window})&(twbase+w<{previous_r*length}),0.)']
   for d in range(r):
    if small:
     lines += [f'    ix=tl.minimum((pf//{previous_prefix}-base//{previous_prefix})*{length}+{d*m}+j,{pwindow-1})',f'    cr=tl.gather(wr,ix,0)',f'    ci=tl.gather(wi,ix,0)']
    else:
     lines += [f'    ix=pf//{previous_prefix}*{length}+{d*m}+j',f'    cr=tl.load(TW+ix,valid,0.)',f'    ci=tl.load(TW+{previous_r*length}+ix,valid,0.)']
    lines += [f'    r{d},i{d}=r{d}*cr-i{d}*ci,r{d}*ci+i{d}*cr']
  if r in _NATURAL_ORDER_CODELET_RADICES:lines += _emit_natural_order_codelet_call('    ',r,direction)
  elif r==16:lines += _emit_radix16_codelet_call('    ',direction)
  else:lines += _emit_local_mixed_codelet_call('    ',r,direction)
  total=prefix*m
  for d in range(r):
   if last:
    tail=f'{2*g*m}' if small else f'2*({m}-tile*{c})'
    lines += [f'    v=tl.reshape(tl.join(r{d},i{d}),({2*c},))',f'    w=tl.arange(0,{2*c})',f'    tl.store(Y+2*({d*total}+outbase)+w,v,(w<{tail})&(w<2*({total}-outbase)))']
   else:
    lines += [f'    tl.store(Y+{d*total}+tc,r{d},valid)',f'    tl.store(Y+{n+d*total}+tc,i{d},valid)']
  specs.append(dict(r=r,m=m,prefix=prefix,previous_prefix=previous_prefix,g=g,c=c,grid=((prefix+g-1)//g if small else prefix*((m+c-1)//c),)))
  previous_prefix=prefix;prefix*=r
 return _module_source('\n'.join(lines)+'\n',tuple(sorted(codelet_radices_for(tuple(factors))))),specs

"""DIF FFT switching the contiguous vector axis once prefix parallelism grows."""
import math,re
from .kernels_common import _NATURAL_ORDER_CODELET_RADICES
from .kernels_leaf import _emit_natural_order_codelet_call,_emit_radix16_codelet_call,_emit_local_mixed_codelet_call


def build_vector_dif(factors,direction='forward'):
 source,old=_flat_source(factors,direction);n=math.prod(factors)
 tile=int(os.environ.get('FLAGFFT_NPU_DIF_TRANSPOSE_TILE','32'))
 if tile not in (32,64):raise ValueError('DIF transpose tile must be 32 or 64')
 switch=next(i for i,s in enumerate(old) if s['prefix']>=64)
 first=re.search(r'@triton.jit\ndef vector_stage_0',source).start();header=source[:first];chunks=[];specs=[]
 for i in range(switch):
  match=re.search(rf'@triton.jit\ndef vector_stage_{i}\([^\n]*\):\n[\s\S]*?(?=\n@triton.jit|\Z)',source)
  offset=f'    batch=tl.program_id(1).to(tl.int64)\n    X=X+batch*{2*n}\n    Y=Y+batch*{2*n}\n'
  text=re.sub(r'(def vector_stage_\d+\([^\n]*\):\n)',lambda m:m.group(0)+offset,match.group(0),count=1)
  chunks.append(text);specs.append(dict(old[i],name=f'vector_stage_{i}',phase='early',stage=i))
 p=old[switch]['prefix'];rest=n//p
 chunks.append(f'''@triton.jit
def switch_axes(X,TW,Y):
    batch=tl.program_id(2).to(tl.int64)
    X=X+batch*{2*n}
    Y=Y+batch*{2*n}
    row=tl.program_id(0)*{tile}+tl.arange(0,{tile})
    col=tl.program_id(1)*{tile}+tl.arange(0,{tile})
    off=row[:,None]*{rest}+col[None,:]
    valid=(row[:,None]<{p})&(col[None,:]<{rest})
    r=tl.load(X+off,valid,0.);i=tl.load(X+{n}+off,valid,0.)
    dst=col[:,None]*{p}+row[None,:]
    mask=(col[:,None]<{rest})&(row[None,:]<{p})
    tl.store(Y+dst,tl.trans(r),mask);tl.store(Y+{n}+dst,tl.trans(i),mask)
''')
 specs.append(dict(name='switch_axes',phase='transpose',grid=((p+tile-1)//tile,(rest+tile-1)//tile)))
 for i in range(switch,len(factors)):
  s=old[i];r=s['r'];m=s['m'];prefix=s['prefix'];c=min(2048 if r==2 else 1024 if r<=8 else 512,pad(prefix));tiles=(prefix+c-1)//c
  lines=['@triton.jit',f'def switched_stage_{i}(X,TW,Y):',f'    batch=tl.program_id(1).to(tl.int64)',f'    X=X+batch*{2*n}',f'    Y=Y+batch*{2*n}',f'    j=tl.program_id(0)//{tiles}',f'    pf=tl.program_id(0)%{tiles}*{c}+tl.arange(0,{c})',f'    valid=pf<{prefix}']
  for d in range(r):
   lines += [f'    off=({d*m}+j)*{prefix}+pf',f'    r{d}=tl.load(X+off,valid,0.)',f'    i{d}=tl.load(X+{n}+off,valid,0.)',f'    wr=tl.load(TW+off,valid,0.)',f'    wi=tl.load(TW+{n}+off,valid,0.)',f'    r{d},i{d}=r{d}*wr-i{d}*wi,r{d}*wi+i{d}*wr']
  if r in _NATURAL_ORDER_CODELET_RADICES:lines+=_emit_natural_order_codelet_call('    ',r,direction)
  elif r==16:lines+=_emit_radix16_codelet_call('    ',direction)
  else:lines+=_emit_local_mixed_codelet_call('    ',r,direction)
  for d in range(r):
   if i+1==len(factors):
    lines += [f'    v=tl.reshape(tl.join(r{d},i{d}),({2*c},))',f'    w=tl.arange(0,{2*c})',f'    tl.store(Y+{2*d*prefix}+tl.program_id(0)*{2*c}+w,v,tl.program_id(0)*{2*c}+w<{2*prefix})']
   else:
    lines += [f'    off=j*{r*prefix}+{d*prefix}+pf',f'    tl.store(Y+off,r{d},valid)',f'    tl.store(Y+{n}+off,i{d},valid)']
  chunks.append('\n'.join(lines)+'\n');specs.append(dict(s,name=f'switched_stage_{i}',phase='late',stage=i,grid=(m*tiles,)))
 return header+'\n'.join(chunks),specs


def emit_vector_dif(factors,direction,phase,out_dir,aligned=False,cube_prefix=False,flat_power2=False,vector_first17=False,batch_pack=False,vector_second15=False,terminal8=False,coarse=False,global_inverse=False,matrix_first17=False,prefix_pitch=1,dense_prefix=False,coarse_single=False,codelet_reuse=False,coarse_gt=False,workspace_padding=0,finish_batch4=False,vector_basis=False,dense_prefix_columns=32,dense_prefix_frequencies=32,coarse_dense=False,power_cube_finish=False):
    valid_factors = (all(4 <= r <= 128 and not (r & (r - 1)) for r in factors)
                     if flat_power2 else
                     all(r in _NATURAL_ORDER_CODELET_RADICES or r in (16,18,20,24,25,27,28,30,32)
                         for r in factors))
    if not factors or not valid_factors:
        raise ValueError('Unsupported DIF factors')
    if direction not in ('forward','inverse'):
        raise ValueError('Unsupported DIF direction')
    if dense_prefix and not cube_prefix:
        raise ValueError('Dense short prefix requires Cube DIF')
    if prefix_pitch!=1 and not cube_prefix:
        raise ValueError('Aligned prefix pitch requires Cube DIF')
    if batch_pack and not cube_prefix:
        raise ValueError('Batch-grouped DIF requires the Cube DIF plan')
    if codelet_reuse and not (cube_prefix and batch_pack):
        raise ValueError('DIF codelet reuse requires a batch-grouped Cube prefix')
    if terminal8 and not flat_power2:
        raise ValueError('Terminal eight-point sections require flat power DIF')
    if vector_first17 and not cube_prefix:
        raise ValueError('The vector-17 prefix requires the Cube DIF plan')
    if vector_second15 and not cube_prefix:
        raise ValueError('The vector second axis requires the Cube DIF plan')
    if global_inverse and (not (cube_prefix or coarse or flat_power2) or direction!='inverse'):
        raise ValueError('Global inverse requires inverse grouped DIF')
    if workspace_padding and not flat_power2:
        raise ValueError('Workspace padding requires flat power DIF')
    if power_cube_finish and not flat_power2:
        raise ValueError('Power Cube finish requires flat power DIF')
    if coarse_gt and not coarse:
        raise ValueError("Coprime coarse prefix requires coarse DIF")
    if coarse_dense and not coarse:
        raise ValueError('Dense coarse prefix requires coarse DIF')
    if coarse_single and not coarse:
        raise ValueError('Single-batch coarse schedule requires coarse DIF')
    if vector_basis and not ((cube_prefix and aligned and batch_pack) or (coarse and not coarse_single)):
        raise ValueError('Vector basis requires a public batch DIF schedule')
    if finish_batch4 and not (cube_prefix and aligned and batch_pack):
        raise ValueError('Batch finish requires batch-grouped aligned Cube DIF')
    if coarse:
        from .kernels_npu_dif_coarse import build_coarse_dif
        source,specs=build_coarse_dif(tuple(factors),direction,global_inverse=global_inverse,single_batch=coarse_single,good_thomas=coarse_gt,dense_prefix=coarse_dense)
    elif flat_power2:
        if aligned or cube_prefix:raise ValueError("Flat power DIF uses a compact layout")
        from .kernels_npu_power_dif import build
        source,specs=build(tuple(factors),direction,terminal8=terminal8,global_inverse=global_inverse,workspace_padding=workspace_padding,cube_finish=power_cube_finish)
    elif cube_prefix:
        if not aligned:raise ValueError('Cube DIF prefix requires an aligned layout')
        from .kernels_npu_dif_cube import build_cube_prefix_dif
        source,specs=build_cube_prefix_dif(tuple(factors),direction,vector_first17=vector_first17,batch_pack=batch_pack,vector_second15=vector_second15,global_inverse=global_inverse,matrix_first17=matrix_first17,prefix_pitch=prefix_pitch,dense_prefix=dense_prefix,codelet_reuse=codelet_reuse,dense_prefix_columns=dense_prefix_columns,dense_prefix_frequencies=dense_prefix_frequencies)
    elif aligned:
        from .kernels_npu_dif_aligned import build_aligned_vector_dif
        source,specs=build_aligned_vector_dif(tuple(factors),direction)
    else:
        source,specs=build_vector_dif(tuple(factors),direction)
    if finish_batch4:
        from .npu_batch_finish import replace_finish
        source,specs=replace_finish(source,specs,'forward' if global_inverse else direction,
                                    conjugate_output=global_inverse)
    if vector_basis:
        from .npu_vector_basis import optimize_vector_basis
        source,specs=optimize_vector_basis(source,specs,'forward' if global_inverse else direction)
    if phase<0 or phase>=len(specs):raise ValueError('Invalid DIF phase')
    n=math.prod(factors);spec=specs[phase]
    dense_tag=f'_dense_f{dense_prefix_frequencies}_c{dense_prefix_columns}' if dense_prefix else ''
    if coarse_dense:dense_tag+='_coarsedense'
    if power_cube_finish:dense_tag+='_powercubefinish'
    path=Path(out_dir)/(('npu_vector_dif_coarse_' if coarse else 'npu_vector_dif_flat_power2_' if flat_power2 else 'npu_vector_dif_aligned_cube_' if cube_prefix else 'npu_vector_dif_aligned_' if aligned else 'npu_vector_dif_')+str(n)+'_'+str(phase)+'_'+direction+('_workspacepad512' if workspace_padding else '')+('_finishbatch4' if finish_batch4 else '')+('_vectorbasis' if vector_basis else '')+dense_tag+'.py')
    path.parent.mkdir(parents=True,exist_ok=True);write_text_atomic(path,source)
    meta=dict(module_path=str(path),kernel_name=spec['name'],arg_names=['X','TW','Y'],
        signature='*fp32:16,*fp32:16,*fp32:16',num_warps=4,num_stages=1,
        batch_per_block=spec.get('batch_pack',1),inner_pack=1,tle_fused_twiddle=True,
        kernel_type='four_step_row',dtype='complex64',direction=direction,length=n)
    # Keep the real/single/nested schedules and the inverse prefix on their
    # established settings. Wider public complex prefixes have enough sample
    # tiles to amortize double-buffering the direct 2D matrix input in CBUF.
    if (phase == 0 and spec.get('r') == 17 and cube_prefix and matrix_first17 and vector_first17
            and batch_pack and codelet_reuse and vector_basis
            and direction == 'forward'
            and os.environ.get('FLAGFFT_NPU_DIF_COMPONENT17', '1') != '0'
            and spec['grid'][0] >= 64):
        pipeline = os.environ.get('FLAGFFT_NPU_DIF_PREFIX_PIPELINE', '1')
        if pipeline not in ('0', '1'):
            raise ValueError('FLAGFFT_NPU_DIF_PREFIX_PIPELINE must be 0 or 1')
        if pipeline == '1':
            meta['num_stages'] = 2
    # The public batched 255-point prefix uses a bounded 64x64 matrix tile.
    # Pipeline only this qualified Cube producer; its AIV suffix keeps the
    # original scheduling and all arithmetic remains IEEE FP32.
    if (phase == 0 and dense_prefix and cube_prefix and batch_pack
            and matrix_first17 and vector_first17 and vector_second15
            and codelet_reuse and vector_basis
            and spec.get('dense_prefix') == (255, 256, 64, 64, 2)
            and 512 <= spec['m'] <= 4096):
        pipeline = os.environ.get('FLAGFFT_NPU_DIF_BATCH_DENSE_PIPELINE', '1')
        if pipeline not in ('0', '1'):
            raise ValueError('FLAGFFT_NPU_DIF_BATCH_DENSE_PIPELINE must be 0 or 1')
        meta['num_stages'] = 2 if pipeline == '1' else 1
    if workspace_padding:meta['workspace_length']=spec['physical_n']
    if dense_prefix:
        meta.update(dense_prefix_columns=dense_prefix_columns,dense_prefix_frequencies=dense_prefix_frequencies)
    if spec.get('batch_pack',1)>1:
        meta['arg_names'].append('NB');meta['signature']+=',i32'
    write_text_atomic(path.with_suffix('.json'),json.dumps(meta,sort_keys=True))
    return meta
