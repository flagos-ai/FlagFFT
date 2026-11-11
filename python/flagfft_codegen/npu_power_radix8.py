# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""A bounded 64-point axis from two eight-point IEEE FP32 butterflies."""

def radix8_axis_source():
    ls=['@triton.jit','def power_fft64_8x8(r,i,W,N:tl.constexpr,L:tl.constexpr,C:tl.constexpr,LOG:tl.constexpr):']
    for a in range(8):
        ls += [f'    r{a}=al.extract_slice(r,[{a}*(N//8)],[N//8],[1])',f'    i{a}=al.extract_slice(i,[{a}*(N//8)],[N//8],[1])']
    args=','.join([f'r{k}' for k in range(8)]+[f'i{k}' for k in range(8)])
    ls += [f'    {args}=_npu_power_rad8({args})','    lane=tl.arange(0,N//8)']
    for a in range(1,8):
        ls += [f'    wr=tl.load(W+{a}*(N//8)+lane)',f'    wi=tl.load(W+N+{a}*(N//8)+lane)',
            f'    r{a},i{a}=r{a}*wr-i{a}*wi,r{a}*wi+i{a}*wr']
    ls += ['    buf_r=tl.zeros((8,8*C),tl.float32)','    buf_i=tl.zeros((8,8*C),tl.float32)']
    for a in range(8):
        ls += [f'    buf_r=al.insert_slice(buf_r,tl.reshape(r{a},(1,8*C)),[{a},0],[1,8*C],[1,1])',
            f'    buf_i=al.insert_slice(buf_i,tl.reshape(i{a},(1,8*C)),[{a},0],[1,8*C],[1,1])']
    for b in range(8):
        ls += [f'    r{b}=tl.reshape(al.extract_slice(buf_r,[0,{b}*C],[8,C],[1,1]),(8*C,))',
            f'    i{b}=tl.reshape(al.extract_slice(buf_i,[0,{b}*C],[8,C],[1,1]),(8*C,))']
    ls += [f'    {args}=_npu_power_rad8({args})','    out_r=tl.zeros((N,),tl.float32)','    out_i=tl.zeros((N,),tl.float32)']
    for b in range(8):
        ls += [f'    out_r=al.insert_slice(out_r,r{b},[{b}*8*C],[8*C],[1])',f'    out_i=al.insert_slice(out_i,i{b},[{b}*8*C],[8*C],[1])']
    ls += ['    return out_r,out_i']
    return '\n'.join(ls)+'\n'
