# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Register-only Rader codelets built from the existing short FFT emitters."""
import cmath
import math
from .kernels_leaf import _emit_radix16_codelet_call, _emit_local_mixed_codelet_call


def rader_codelet(prime):
    if prime not in (17,19):
        raise ValueError('Rader register codelets support primes 17 and 19')
    m=prime-1
    generator=next(g for g in range(2,prime)
        if len({pow(g,k,prime) for k in range(m)})==m)
    inputs=[pow(generator,k,prime) for k in range(m)]
    outputs=[pow(generator,-k,prime) for k in range(m)]
    c=[cmath.exp(-2j*math.pi*pow(generator,-k,prime)/prime) for k in range(m)]
    weights=[sum(c[j]*cmath.exp(-2j*math.pi*j*k/m) for j in range(m))/m for k in range(m)]
    args=[f'r{k}' for k in range(prime)]+[f'i{k}' for k in range(prime)]
    lines=['@triton.jit',f'def _fwd_rad{prime}_rader({",".join(args)}):',
        '    zero_r=r0','    zero_i=i0',
        '    dc_r='+ '+'.join(f'r{k}' for k in range(prime)),
        '    dc_i='+ '+'.join(f'i{k}' for k in range(prime))]
    for j,k in enumerate(inputs):lines += [f'    save_r{j}=r{k}',f'    save_i{j}=i{k}']
    for j in range(m):lines += [f'    r{j}=save_r{j}',f'    i{j}=save_i{j}']
    def fft(direction):
        return (_emit_radix16_codelet_call('    ',direction) if m==16 else
                _emit_local_mixed_codelet_call('    ',m,direction))
    lines += fft('forward')
    for j,w in enumerate(weights):
        # Conjugate symmetry makes alternate components exactly zero.
        w=complex(0. if abs(w.real)<1e-12 else w.real, 0. if abs(w.imag)<1e-12 else w.imag)
        lines += [f'    r{j},i{j}=r{j}*{w.real!r}-i{j}*{w.imag!r},r{j}*{w.imag!r}+i{j}*{w.real!r}']
    lines += fft('inverse')
    for j,k in enumerate(outputs):lines += [f'    out_r{k}=zero_r+r{j}',f'    out_i{k}=zero_i+i{j}']
    lines += ['    out_r0=dc_r','    out_i0=dc_i',
        '    return ('+','.join([f'out_r{k}' for k in range(prime)]+[f'out_i{k}' for k in range(prime)])+')']
    return '\n'.join(lines)+'\n'
