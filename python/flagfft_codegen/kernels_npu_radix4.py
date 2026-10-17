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

"""Generated bodies for the existing Ascend radix4 pair table contract."""

import json
from pathlib import Path

from .artifacts import write_text_atomic
from .metadata import _module_source, _signature


def build_npu_radix4_pair_source(*, length: int, group: int, mode: int):
    """Preserve quarter-vector butterflies, grouped tables and real boundaries.

    TLE insert_slice keeps the native contiguous UB vector partitions between
    pairs. Inverse scaling is owned by the parent plan, as in the native body.
    """
    supported = {64: (1, 4, 8, 16, 32), 128: (4, 8, 16), 2048: (1,)}
    if (length not in supported or group not in supported[length]
            or mode not in (0, 1, 2) or (length == 2048 and mode != 0)):
        raise ValueError("Unsupported Ascend radix4 pair length, group or mode")
    n = length
    count = n * group
    quarter = count // 4
    stages = n.bit_length() - 1
    pairs = stages // 2
    half_spectrum = n // 2 + 1
    input_count = group * (n if mode == 1 else 2 * half_spectrum if mode == 2 else 2 * n)
    input_size = 1 << (input_count - 1).bit_length()
    output_count = group * (n if mode == 2 else 2 * half_spectrum if mode == 1 else 2 * n)
    output_size = 1 << (output_count - 1).bit_length()
    output_base = pairs * count
    expansion_base = output_base + 4 * count
    expansion_sign = pairs * 6 * quarter + 2 * count
    name = f'radix4_pair_fft{n}_g{group}_m{mode}'
    lines = [f'@triton.jit\ndef {name}(in_ptr, out_ptr, idx_ptr, twiddle_ptr, nbatch):',
             '    pid = tl.program_id(0).to(tl.int64)',
             f'    source_lane = tl.arange(0, {input_size})',
             f'    source = tl.load(in_ptr + pid * {input_count} + source_lane, source_lane < {input_count}, 0.0)',
             f'    lane = tl.arange(0, {quarter})']
    if mode == 2:
        lines += [f'    expand_lane = tl.arange(0, {count})',
                  f'    expand_real = tl.load(idx_ptr + {expansion_base} + expand_lane) >> 2',
                  f'    expand_imag = tl.load(idx_ptr + {expansion_base + count} + expand_lane) >> 2',
                  '    real = tl.gather(source, expand_real, 0)',
                  f'    imag = tl.gather(source, expand_imag, 0) * tl.load(twiddle_ptr + {expansion_sign} + expand_lane)']
    for pair in range(pairs):
        for k in range(4):
            lines.append(f'    index{k} = tl.load(idx_ptr + {pair * count + k * quarter} + lane) >> 2')
            if pair == 0 and mode != 2:
                lines.append(f'    x{k}r = tl.gather(source, index{k}, 0)')
                lines.append(f'    x{k}i = tl.zeros(({quarter},), tl.float32)' if mode == 1
                             else f'    x{k}i = tl.gather(source, index{k} + 1, 0)')
            else:
                lines += [f'    x{k}r = tl.gather(real, index{k}, 0)',
                          f'    x{k}i = tl.gather(imag, index{k}, 0)']
        for k in range(3):
            base = pair * 6 * quarter + 2 * k * quarter
            lines += [f'    w{k}r = tl.load(twiddle_ptr + {base} + lane)',
                      f'    w{k}i = tl.load(twiddle_ptr + {base + quarter} + lane)']
        for first, second in ((0, 1), (2, 3)):
            lines += [f'    pr = x{second}r * w0r - x{second}i * w0i',
                      f'    pi = x{second}r * w0i + x{second}i * w0r',
                      f'    a{first}r = x{first}r + pr',
                      f'    a{first}i = x{first}i + pi',
                      f'    a{second}r = x{first}r - pr',
                      f'    a{second}i = x{first}i - pi']
        for first, second, twiddle, low, high in ((0, 2, 1, 0, 2), (1, 3, 2, 1, 3)):
            lines += [f'    pr = a{second}r * w{twiddle}r - a{second}i * w{twiddle}i',
                      f'    pi = a{second}r * w{twiddle}i + a{second}i * w{twiddle}r',
                      f'    y{low}r = a{first}r + pr', f'    y{low}i = a{first}i + pi',
                      f'    y{high}r = a{first}r - pr', f'    y{high}i = a{first}i - pi']
        lines += [f'    real = tl.zeros(({count},), tl.float32)',
                  f'    imag = tl.zeros(({count},), tl.float32)']
        for k in range(4):
            lines += [f'    real = tle.dsa.insert_slice(real, y{k}r, [{k * quarter}], [{quarter}], [1])',
                      f'    imag = tle.dsa.insert_slice(imag, y{k}i, [{k * quarter}], [{quarter}], [1])']
    if stages & 1:
        leftover_index = output_base + 2 * count
        leftover_twiddle = pairs * 6 * quarter
        lines += [f'    next_real = tl.zeros(({count},), tl.float32)',
                  f'    next_imag = tl.zeros(({count},), tl.float32)']
        for tile in range(4):
            at = tile * quarter
            lines += [f'    ia = tl.load(idx_ptr + {leftover_index + at} + lane) >> 2',
                      f'    ib = tl.load(idx_ptr + {leftover_index + count + at} + lane) >> 2',
                      f'    wr = tl.load(twiddle_ptr + {leftover_twiddle + at} + lane)',
                      f'    wi = tl.load(twiddle_ptr + {leftover_twiddle + count + at} + lane)',
                      '    ar = tl.gather(real, ia, 0)', '    ai = tl.gather(imag, ia, 0)',
                      '    br = tl.gather(real, ib, 0)', '    bi = tl.gather(imag, ib, 0)',
                      '    pr = br * wr - bi * wi', '    pi = br * wi + bi * wr',
                      '    yr = ar + pr', '    yi = ai + pi',
                      f'    next_real = tle.dsa.insert_slice(next_real, yr, [{at}], [{quarter}], [1])',
                      f'    next_imag = tle.dsa.insert_slice(next_imag, yi, [{at}], [{quarter}], [1])']
        lines += ['    real = next_real', '    imag = next_imag']
    if mode == 2:
        lines.append('    packed = real')
    else:
        lines += [f'    packed = tl.zeros(({2 * count},), tl.float32)',
                  f'    packed = tle.dsa.insert_slice(packed, real, [0], [{count}], [1])',
                  f'    packed = tle.dsa.insert_slice(packed, imag, [{count}], [{count}], [1])']
    lines += [f'    output_lane = tl.arange(0, {output_size})',
              f'    output_index = tl.load(idx_ptr + {output_base} + output_lane, output_lane < {output_count}, 0) >> 2',
              '    result = tl.gather(packed, output_index, 0)',
              f'    tl.store(out_ptr + pid * {output_count} + output_lane, result, output_lane < {output_count})']
    return name, '\n'.join(lines) + '\n'


def emit_npu_radix4_pair(*, length: int, group: int, mode: int,
                        direction: str, out_dir: Path) -> dict:
    if direction not in ("forward", "inverse"):
        raise ValueError("FFT direction must be forward or inverse")
    if (mode == 1 and direction != "forward") or (mode == 2 and direction != "inverse"):
        raise ValueError("Real FFT mode disagrees with its direction")
    kernel_name, source = build_npu_radix4_pair_source(length=length, group=group, mode=mode)
    basename = f"flagfft_npu_radix4_fft{length}_g{group}_m{mode}_{direction}"
    out_dir.mkdir(parents=True, exist_ok=True)
    module_path = out_dir / f"{basename}.py"
    write_text_atomic(module_path, _module_source(source))
    arg_names = ["in_ptr", "out_ptr", "idx_ptr", "twiddle_ptr", "nbatch"]
    metadata = {
        "module_path": str(module_path), "kernel_name": kernel_name,
        "signature": _signature(arg_names, "complex64"),
        "num_warps": 4, "num_stages": 1, "batch_per_block": group,
        "arg_names": arg_names, "kernel_type": "npu_radix4_pair",
        "dtype": "complex64", "direction": direction, "length": length,
        "group_size": group, "mode": mode,
    }
    write_text_atomic(module_path.with_suffix(".json"), json.dumps(metadata, sort_keys=True))
    return metadata
