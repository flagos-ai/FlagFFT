# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Real three-product blocks for the symmetric nine-point DFT."""
import math
def radix9_source():
 args=[f'r{k}' for k in range(9)]+[f'i{k}' for k in range(9)]
 ls=['@triton.jit','def _fwd_rad9_b1('+','.join(args)+'):']
 c1,c2=math.cos(2*math.pi/9),math.cos(4*math.pi/9);c4=-c1-c2
 s1,s4=math.sin(2*math.pi/9),math.sin(math.pi/9);s2=s1+s4;s3=math.sqrt(3)/2
 for z in ('r','i'):
  for j in range(1,5):ls += [f'    a{j}{z}={z}{j}+{z}{9-j}',f'    d{j}{z}={z}{j}-{z}{9-j}']
  ls += [f'    sum{z}=a1{z}+a2{z}+a4{z}',
         f'    base{z}={z}0-0.5*a3{z}',
         f'    u{z},v{z}=a1{z}-a4{z},a2{z}-a4{z}',
         f'    p{z}={c2!r}*(u{z}+v{z})',
         f'    q{z}={(c1-c2)!r}*u{z}',
         f'    t{z}={(c4-c2)!r}*v{z}',
         f'    c1{z},c2{z}=p{z}+q{z},p{z}+t{z}',
         f'    c4{z}=base{z}-c1{z}-c2{z}',
         f'    c1{z},c2{z}=base{z}+c1{z},base{z}+c2{z}',
         f'    c3{z}={z}0+a3{z}-0.5*sum{z}',
         f'    y0{z}={z}0+a3{z}+sum{z}',
         f'    p{z}={s1!r}*(d1{z}-d4{z})',
         f'    q{z}={s2!r}*(d2{z}+d4{z})',
         f'    t{z}={s4!r}*(d1{z}+d2{z})',
         f'    z{z}={s3!r}*d3{z}',
         f'    s1{z},s2{z},s4{z}=p{z}+q{z}+z{z},p{z}+t{z}-z{z},t{z}-q{z}+z{z}',
         f'    s3{z}={s3!r}*(d1{z}+d4{z}-d2{z})']
 for k in range(1,5):
  ls += [f'    y{k}r,y{k}i=c{k}r+s{k}i,c{k}i-s{k}r',
         f'    y{9-k}r,y{9-k}i=c{k}r-s{k}i,c{k}i+s{k}r']
 ls += ['    return '+','.join([f'y{k}r' for k in range(9)]+[f'y{k}i' for k in range(9)])]
 return '\n'.join(ls)+'\n'
