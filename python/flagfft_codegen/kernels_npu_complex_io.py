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
