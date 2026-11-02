# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CRT register permutations remove inner twiddles from coprime DFT axes."""
import math
def coprime_source(a, b):
    assert math.gcd(a, b) == 1
    n = a * b
    u = pow(b, -1, a)
    v = pow(a, -1, b)
    args = [f'r{k}' for k in range(n)] + [f'i{k}' for k in range(n)]
    lines = ['@triton.jit', f'def _fwd_rad{n}_b1(' + ','.join(args) + '):']
    for j in range(b):
        indices = [(k * b * u + j * a * v) % n for k in range(a)]
        lhs = [f'a{j}_{k}r' for k in range(a)] + [f'a{j}_{k}i' for k in range(a)]
        rhs = [f'r{k}' for k in indices] + [f'i{k}' for k in indices]
        lines.append('    ' + ','.join(lhs) + f'=_fwd_rad{a}_b1(' + ','.join(rhs) + ')')
    out = [''] * (2 * n)
    for k in range(a):
        lhs = [f'b{k}_{j}r' for j in range(b)] + [f'b{k}_{j}i' for j in range(b)]
        rhs = [f'a{j}_{k}r' for j in range(b)] + [f'a{j}_{k}i' for j in range(b)]
        lines.append('    ' + ','.join(lhs) + f'=_fwd_rad{b}_b1(' + ','.join(rhs) + ')')
        for j in range(b):
            freq = (k * b + j * a) % n
            out[freq] = f'b{k}_{j}r'
            out[n + freq] = f'b{k}_{j}i'
    lines.append('    return ' + ','.join(out))
    return '\n'.join(lines) + '\n'
