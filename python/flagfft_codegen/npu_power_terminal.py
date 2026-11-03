# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Constant rotations for the terminal eight-point section of a power DIF axis."""
from pathlib import Path


def terminal8_source():
    lines = [
        '@triton.jit',
        'def power_terminal8(r,i,N:tl.constexpr,C:tl.constexpr):',
        '    re=tl.reshape(r,(N//(8*C),8*C))',
        '    im=tl.reshape(i,(N//(8*C),8*C))',
    ]
    for k in range(8):
        for component, value in (('r', 're'), ('i', 'im')):
            lines.append(
                f'    {component}{k}=tl.reshape(al.extract_slice({value},'
                f'[0,{k}*C],[N//(8*C),C],[1,1]),(N//8,))')
    args = ','.join([f'r{k}' for k in range(8)] + [f'i{k}' for k in range(8)])
    lines += [f'    {args}=_npu_power_rad8({args})',
              '    re=tl.zeros((N,),tl.float32)',
              '    im=tl.zeros((N,),tl.float32)']
    for k in range(8):
        lines += [f'    re=al.insert_slice(re,r{k},[{k}*(N//8)],[N//8],[1])',
                  f'    im=al.insert_slice(im,i{k},[{k}*(N//8)],[N//8],[1])']
    lines += [
        '    return re,im', '', '@triton.jit',
        'def power_fft_axis(r,i,W,N:tl.constexpr,L:tl.constexpr,C:tl.constexpr,LOG:tl.constexpr):',
        '    if LOG%2 and LOG>=3:',
        '        for s in tl.static_range(0,(LOG-3)//2):',
        '            r,i=stage4(r,i,W,N,L*C//4**s,s,True,True,C)',
        '        return power_terminal8(r,i,N,C)',
        '    else:',
        '        return fft_axis_sliced(r,i,W,N,L,C,LOG)',
    ]
    # Reuse the validated ordinary FP32 codelet. Its 0, +/-i and sqrt(1/2)
    # rotations avoid loading and multiplying full root vectors in this
    # section. Keep the existing root-table ABI for the preceding sections.
    codelets = Path(__file__).parent / 'codelet'
    codelet = ((codelets / 'utils.py').read_text() + '\n' +
               (codelets / 'radix8.py').read_text())
    codelet = codelet.replace('_fwd_rad8_b1', '_npu_power_rad8')
    codelet = codelet.replace('_cmul', '_npu_power_cmul')
    return codelet + '\n' + '\n'.join(lines) + '\n'
