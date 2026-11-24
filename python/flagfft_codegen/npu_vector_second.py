# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Use a factored vector DFT after the prefix-dependent constant phase."""
import math
from .kernels_leaf import _emit_natural_order_codelet_call

def vector_second_source(s,previous,n,direction,c):
 r,m,pr=s['r'],s['m'],s['pr'];tiles=math.ceil(m/c)
 ls=['@triton.jit',f'def {s["name"]}(X,TW,Y):','    b=tl.program_id(1).to(tl.int64)',f'    pf=tl.program_id(0)//{tiles}',f'    q=(tl.program_id(0)%{tiles}*{c}+tl.arange(0,{c}))[None,:]']
 groups=[];start=0
 while start<r:
  g=1<<((r-start).bit_length()-1);groups.append((start,g));points=g*c
  ls += [f'    d=(tl.arange(0,{g})+{start})[:,None]',f'    off=pf*{r*m}+d*{m}+q',
   f'    xr=tl.load(X+b*{2*n}+off,q<{m},0.)',f'    xi=tl.load(X+b*{2*n}+{n}+off,q<{m},0.)',
   f'    wr=tl.load(TW+pf*{4*pr*pr}+d)',f'    wi=tl.load(TW+pf*{4*pr*pr}+{r*2*pr}+d)',
   f'    re=tl.reshape(xr*wr-xi*wi,({points},))',f'    im=tl.reshape(xr*wi+xi*wr,({points},))']
  for j in range(g):ls += [f'    r{start+j}=al.extract_slice(re,[{j*c}],[{c}],[1])',f'    i{start+j}=al.extract_slice(im,[{j*c}],[{c}],[1])']
  start+=g
 ls += _emit_natural_order_codelet_call('    ',r,direction)
 for start,g in groups:
  points=g*c;ls += [f'    re=tl.zeros(({points},),tl.float32)',f'    im=tl.zeros(({points},),tl.float32)']
  for j in range(g):ls += [f'    re=al.insert_slice(re,r{start+j},[{j*c}],[{c}],[1])',f'    im=al.insert_slice(im,i{start+j},[{j*c}],[{c}],[1])']
  ls += [f'    d=(tl.arange(0,{g})+{start})[:,None]',f'    dst=d*{previous*m}+pf*{m}+q',
   f'    tl.store(Y+b*{2*n}+dst,tl.reshape(re,({g},{c})),q<{m})',f'    tl.store(Y+b*{2*n}+{n}+dst,tl.reshape(im,({g},{c})),q<{m})']
 s['grid']=(previous*tiles,);s['auto']=True
 return '\n'.join(ls)+'\n'
