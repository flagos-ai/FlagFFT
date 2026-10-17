# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contiguous FP32 AoS transfers with local planar complex arithmetic."""


def build_npu_bluestein_source(kind):
    """Keep the original 256-complex tile, ABI, padding and conjugation order."""
    if kind not in ('bluestein_prepare', 'bluestein_pointwise', 'bluestein_finalize'):
        raise ValueError(f'Unsupported vector complex I/O body: {kind}')
    point = kind == 'bluestein_pointwise'
    final = kind == 'bluestein_finalize'
    name = f'_{kind}_aos_kernel'
    args = (['a_ptr', 'b_ptr', 'out_ptr', 'm', 'nbatch'] if point else
            ['in_ptr', 'chirp_ptr', 'out_ptr', 'n', 'm', 'nbatch'])
    lines = [f'@triton.jit\ndef {name}(' + ', '.join(args) + '):',
        '    tile = tl.program_id(0).to(tl.int64)',
        '    batch = tl.program_id(1).to(tl.int64)',
        '    scalar = tile * 512 + tl.arange(0, 512)',
        '    local = tl.arange(0, 256)',
        f"    mask = scalar < {('n' if final else 'm')} * 2"]
    if point:
        lines += ['    raw_a = tl.load(a_ptr + batch * m * 2 + scalar, mask, 0.0)',
                  '    raw_b = tl.load(b_ptr + scalar, mask, 0.0)']
    else:
        mask = 'mask' if final else 'mask & (scalar < n * 2)'
        distance = 'm' if final else 'n'
        lines += [f'    raw_a = tl.load(in_ptr + batch * {distance} * 2 + scalar, {mask}, 0.0)',
                  f'    raw_b = tl.load(chirp_ptr + scalar, {mask}, 0.0)']
    lines += ['    ar = tl.gather(raw_a, local * 2, 0)',
              '    ai = tl.gather(raw_a, local * 2 + 1, 0)',
              '    br = tl.gather(raw_b, local * 2, 0)',
              '    bi = tl.gather(raw_b, local * 2 + 1, 0)']
    if final:
        lines += ['    ar = ar / m', '    ai = -ai / m']
    lines += ['    pr, pi = _cmul(ar, ai, br, bi)']
    if point:
        lines += ['    pi = -pi']
    lines += ['    planar = tl.zeros((512,), tl.float32)',
              '    planar = tle.dsa.insert_slice(planar, pr, [0], [256], [1])',
              '    planar = tle.dsa.insert_slice(planar, pi, [256], [256], [1])',
              '    out_lane = tl.arange(0, 512)',
              '    interleave = (out_lane >> 1) + (out_lane & 1) * 256',
              '    values = tl.gather(planar, interleave, 0)',
              f"    tl.store(out_ptr + batch * {('n' if final else 'm')} * 2 + scalar, values, mask)"]
    return name, '\n'.join(lines) + '\n', args


REAL_BOUNDARY_KINDS = ('real_to_complex', 'r2c_half_pack', 'compact_to_hermitian_full', 'complex_to_real')


def build_npu_real_boundary_source(kind, n, elements=256):
    if kind not in REAL_BOUNDARY_KINDS or n < 1:
        raise ValueError((kind, n))
    half = n // 2 + 1
    cols = half if kind == 'r2c_half_pack' else n
    block = min(elements, 1 << (cols - 1).bit_length())
    rows = max(1, elements // block)
    distance = 'input_distance' if kind in ('real_to_complex', 'compact_to_hermitian_full') else 'output_distance'
    name = f'_{kind}_aos_kernel_n{n}_f32'
    lines = ['@triton.jit', f'def {name}(in_ptr, out_ptr, {distance}, nbatch):',
        '    tile = tl.program_id(0).to(tl.int64)',
        '    batch_tile = tl.program_id(1).to(tl.int64)',
        f'    row = batch_tile * {rows} + tl.arange(0, {rows})[:, None]',
        f'    local = tl.arange(0, {block})[None, :]',
        f'    col = tile * {block} + local',
        f'    scalar = tl.arange(0, {2 * block})[None, :]',
        f'    scalar_col = tile * {2 * block} + scalar']
    if kind == 'real_to_complex':
        lines += [f'    raw = tl.load(in_ptr + row * input_distance + col, (row < nbatch) & (col < {n}), 0.0)',
            f'    mapping = tl.broadcast_to(scalar >> 1, ({rows}, {2 * block}))',
            '    real = tl.gather(raw, mapping, 1)',
            '    values = tl.where((scalar & 1) == 0, real, 0.0)',
            f'    tl.store(out_ptr + row * {2 * n} + scalar_col, values, (row < nbatch) & (scalar_col < {2 * n}))']
    elif kind == 'r2c_half_pack':
        lines += [f'    mask = (row < nbatch) & (scalar_col < {2 * half})',
            f'    raw = tl.load(in_ptr + row * {2 * n} + scalar_col, mask, 0.0)',
            '    tl.store(out_ptr + row * output_distance * 2 + scalar_col, raw, mask)']
    elif kind == 'complex_to_real':
        lines += [f'    raw = tl.load(in_ptr + row * {2 * n} + scalar_col, (row < nbatch) & (scalar_col < {2 * n}), 0.0)',
            f'    mapping = tl.broadcast_to(local * 2, ({rows}, {block}))',
            '    values = tl.gather(raw, mapping, 1)',
            f'    tl.store(out_ptr + row * output_distance + col, values, (row < nbatch) & (col < {n}))']
    else:
        lines += [f'    lo_start = tl.minimum(tile * {block}, {half - 1})',
            '    lo_scalar = lo_start * 2 + scalar',
            f'    lo = tl.load(in_ptr + row * input_distance * 2 + lo_scalar, (row < nbatch) & (lo_scalar < {2 * half}), 0.0)',
            f'    mirrored_start = {n - block + 1} - tile * {block}',
            f'    mirrored_start = tl.minimum(tl.maximum(mirrored_start, 0), {half - 1})',
            '    mirrored_scalar = mirrored_start * 2 + scalar',
            f'    hi = tl.load(in_ptr + row * input_distance * 2 + mirrored_scalar, (row < nbatch) & (mirrored_scalar < {2 * half}), 0.0)',
            f'    reverse_complex = {n} - tile * {block} - (scalar >> 1) - mirrored_start',
            f'    reverse = tl.minimum(tl.maximum(reverse_complex, 0), {block - 1}) * 2 + (scalar & 1)',
            f'    mapping = tl.broadcast_to(reverse, ({rows}, {2 * block}))',
            '    mirrored = tl.gather(hi, mapping, 1)',
            '    mirrored = tl.where((scalar & 1) != 0, -mirrored, mirrored)',
            f'    values = tl.where(scalar_col < {2 * half}, lo, mirrored)',
            '    boundary = ((scalar_col >> 1) == 0)' + (f' | ((scalar_col >> 1) == {n // 2})' if n % 2 == 0 else ''),
            '    values = tl.where(boundary & ((scalar & 1) != 0), 0.0, values)',
            f'    tl.store(out_ptr + row * {2 * n} + scalar_col, values, (row < nbatch) & (scalar_col < {2 * n}))']
    return name, '\n'.join(lines) + '\n', ['in_ptr', 'out_ptr', distance, 'nbatch'], rows
