# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Fill a bounded vector tile with adjacent single-transform sample groups."""
import math
from .kernels_leaf import _emit_natural_order_codelet_call

def packed_late_source(s,direction,points=512):
 r,m,pp,pn=s['r'],s['m'],s['physical_prefix'],s['physical_n']
 c=min(points,1<<(pp-1).bit_length());jp=points//c;tiles=math.ceil(pp/c)
 ls=['@triton.jit',f'def {s["name"]}(X,TW,Y):','    batch=tl.program_id(1).to(tl.int64)',
  f'    j=(tl.program_id(0)//{tiles}*{jp}+tl.arange(0,{jp}))[:,None]',
  f'    pf=(tl.program_id(0)%{tiles}*{c}+tl.arange(0,{c}))[None,:]',
  f'    good=(j<{m})&(pf<{pp})']
 for d in range(r):
  ls += [f'    off=({d*m}+j)*{pp}+pf',
   f'    xr=tl.load(X+batch*{2*pn}+off,good,0.)',f'    xi=tl.load(X+batch*{2*pn}+{pn}+off,good,0.)',
   '    wr=tl.load(TW+off,good,0.)',f'    wi=tl.load(TW+{pn}+off,good,0.)',
   f'    r{d}=tl.reshape(xr*wr-xi*wi,({points},))',f'    i{d}=tl.reshape(xr*wi+xi*wr,({points},))']
 ls+=_emit_natural_order_codelet_call('    ',r,direction)
 for d in range(r):ls += [f'    out=(j*{r}+{d})*{pp}+pf',f'    tl.store(Y+batch*{2*pn}+out,tl.reshape(r{d},({jp},{c})),good)',f'    tl.store(Y+batch*{2*pn}+{pn}+out,tl.reshape(i{d},({jp},{c})),good)']
 s['grid']=(math.ceil(m/jp)*tiles,)
 return '\n'.join(ls)+'\n'
