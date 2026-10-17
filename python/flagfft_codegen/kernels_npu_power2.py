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

"""Generated replacements for the native single-row radix2 table layouts."""
import json
from pathlib import Path

from .artifacts import write_text_atomic
from .metadata import _module_source, _signature


def build_npu_single_power2_source(n):
    if n not in (128, 1024, 2048):
        raise ValueError("Single-row power2 body supports 128, 1024 and 2048")
    stages = n.bit_length() - 1
    grouped = n == 128
    a = 3 * n if grouped else n
    b = a + stages * n
    output = n if grouped else b + stages * n
    name = f'npu_single_power2_fft{n}'
    lines = ['@triton.jit',
        f'def {name}(in_ptr, out_ptr, idx_ptr, twiddle_ptr, nbatch):',
        '    pid = tl.program_id(0).to(tl.int64)',
        f'    lane = tl.arange(0, {n})',
        f'    scalar = tl.arange(0, {2*n})',
        f'    source = tl.load(in_ptr + pid * {2*n} + scalar)']
    if grouped:
        icount = b + stages * n
        ipow = 1 << (icount - 1).bit_length()
        tcount = stages * 2 * n
        tpow = 1 << (tcount - 1).bit_length()
        lines += [f'    ilane = tl.arange(0, {ipow})',
            f'    tlane = tl.arange(0, {tpow})',
            f'    indices = tl.load(idx_ptr + ilane, ilane < {icount}, 0)',
            f'    twiddles = tl.load(twiddle_ptr + tlane, tlane < {tcount}, 0.0)',
            f'    input_index = tle.dsa.extract_slice(indices, [0], [{n}], [1]) >> 2']
    else:
        lines += ['    input_index = tl.load(idx_ptr + lane) >> 2']
    lines += ['    real = tl.gather(source, input_index, 0)',
              '    imag = tl.gather(source, input_index + 1, 0)']
    for stage in range(stages):
        tr = stage * n if grouped else stage * 2 * n
        ti = (stages + stage) * n if grouped else (stage * 2 + 1) * n
        if grouped:
            lines += [f'    ia = tle.dsa.extract_slice(indices, [{a+stage*n}], [{n}], [1]) >> 2',
                f'    ib = tle.dsa.extract_slice(indices, [{b+stage*n}], [{n}], [1]) >> 2',
                f'    wr = tle.dsa.extract_slice(twiddles, [{tr}], [{n}], [1])',
                f'    wi = tle.dsa.extract_slice(twiddles, [{ti}], [{n}], [1])']
        else:
            lines += [f'    ia = tl.load(idx_ptr + {a+stage*n} + lane) >> 2',
                f'    ib = tl.load(idx_ptr + {b+stage*n} + lane) >> 2',
                f'    wr = tl.load(twiddle_ptr + {tr} + lane)',
                f'    wi = tl.load(twiddle_ptr + {ti} + lane)']
        lines += ['    ar = tl.gather(real, ia, 0)', '    ai = tl.gather(imag, ia, 0)',
            '    br = tl.gather(real, ib, 0)', '    bi = tl.gather(imag, ib, 0)',
            '    pr = br * wr - bi * wi', '    pi = br * wi + bi * wr',
            '    real = ar + pr', '    imag = ai + pi']
    lines += [f'    packed = tl.zeros(({2*n},), tl.float32)',
        f'    packed = tle.dsa.insert_slice(packed, real, [0], [{n}], [1])',
        f'    packed = tle.dsa.insert_slice(packed, imag, [{n}], [{n}], [1])']
    if grouped:
        lines += [f'    output_index = tle.dsa.extract_slice(indices, [{output}], [{2*n}], [1]) >> 2']
    else:
        lines += [f'    output_index = tl.load(idx_ptr + {output} + scalar) >> 2']
    lines += ['    values = tl.gather(packed, output_index, 0)',
              f'    tl.store(out_ptr + pid * {2*n} + scalar, values)']
    return name, '\n'.join(lines) + '\n'


def emit_npu_single_power2(*, length: int, direction: str, out_dir: Path) -> dict:
    if direction not in ("forward", "inverse"):
        raise ValueError("FFT direction must be forward or inverse")
    name, source = build_npu_single_power2_source(length)
    out_dir.mkdir(parents=True, exist_ok=True)
    module_path = out_dir / f"flagfft_npu_single_power2_n{length}_{direction}.py"
    write_text_atomic(module_path, _module_source(source))
    args = ["in_ptr", "out_ptr", "idx_ptr", "twiddle_ptr", "nbatch"]
    metadata = dict(module_path=str(module_path), kernel_name=name,
        signature=_signature(args, "complex64"), arg_names=args,
        num_warps=4, num_stages=1, batch_per_block=1,
        kernel_type="npu_single_power2", dtype="complex64",
        direction=direction, length=length)
    write_text_atomic(module_path.with_suffix(".json"), json.dumps(metadata, sort_keys=True))
    return metadata
