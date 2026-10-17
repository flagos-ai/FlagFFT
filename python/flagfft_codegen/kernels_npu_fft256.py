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

"""Qualified native table layouts for the 256-point group8 FFT bodies."""
import json
from pathlib import Path
from .artifacts import write_text_atomic
from .metadata import _module_source
from .kernels_npu_radix4 import build_npu_radix4_pair_source

def build_npu_fft256_pair_source(mode):
    if mode not in (0, 1, 2):
        raise ValueError(mode)
    count, half = 2048, 1024
    input_count = (4096, 2048, 2064)[mode]
    output_count = (4096, 2064, 2048)[mode]
    input_size = 1 << (input_count - 1).bit_length()
    output_size = 1 << (output_count - 1).bit_length()
    name = f'fft256_pair_g8_m{mode}'
    # The original real body has a bounded launch with a grid-stride loop.
    loop = ['    for row in range(pid * 8, nbatch, tl.num_programs(0) * 8):'] if mode else ['    row = pid * 8']
    indent = '        ' if mode else '    '
    body = [f'source_lane = tl.arange(0, {input_size})',
            f'source = tl.load(in_ptr + row * {input_count // 8} + source_lane, source_lane < {input_count}, 0.0)',
            'lane = tl.arange(0, 2048)',
            'input_index = tl.load(idx_ptr + lane) >> 2',
            'real = tl.gather(source, input_index, 0)']
    if mode == 1:
        body += ['imag = tl.zeros((2048,), tl.float32)']
    else:
        body += ['imag = tl.gather(source, input_index + 1, 0)']
        if mode == 2:
            body += ['imag = imag * tl.load(twiddle_ptr + 16384 + lane)']
    body += ['butterfly = tl.arange(0, 1024)']
    for stage in range(8):
        body += [f'ia = tl.load(idx_ptr + {6144 + stage * half} + butterfly) >> 2',
                 f'ib = tl.load(idx_ptr + {14336 + stage * half} + butterfly) >> 2',
                 f'wr = tl.load(twiddle_ptr + {stage * count} + butterfly)',
                 f'wi = tl.load(twiddle_ptr + {stage * count + half} + butterfly)',
                 'ar = tl.gather(real, ia, 0)', 'ai = tl.gather(imag, ia, 0)',
                 'br = tl.gather(real, ib, 0)', 'bi = tl.gather(imag, ib, 0)',
                 'pr = br * wr - bi * wi', 'pi = br * wi + bi * wr',
                 'low_r = ar + pr', 'low_i = ai + pi',
                 'high_r = ar - pr', 'high_i = ai - pi',
                 'real = tl.zeros((2048,), tl.float32)',
                 'imag = tl.zeros((2048,), tl.float32)',
                 'real = tle.dsa.insert_slice(real, low_r, [0], [1024], [1])',
                 'real = tle.dsa.insert_slice(real, high_r, [1024], [1024], [1])',
                 'imag = tle.dsa.insert_slice(imag, low_i, [0], [1024], [1])',
                 'imag = tle.dsa.insert_slice(imag, high_i, [1024], [1024], [1])']
    if mode == 2:
        body += ['packed = real']
    else:
        body += ['packed = tl.zeros((4096,), tl.float32)',
                 'packed = tle.dsa.insert_slice(packed, real, [0], [2048], [1])',
                 'packed = tle.dsa.insert_slice(packed, imag, [2048], [2048], [1])']
    body += [f'output_lane = tl.arange(0, {output_size})',
             f'output_index = tl.load(idx_ptr + 2048 + output_lane, output_lane < {output_count}, 0) >> 2',
             'result = tl.gather(packed, output_index, 0)',
             f'tl.store(out_ptr + row * {output_count // 8} + output_lane, result, output_lane < {output_count})']
    return name, '\n'.join([f'@triton.jit\ndef {name}(in_ptr, out_ptr, idx_ptr, twiddle_ptr, nbatch):',
        '    pid = tl.program_id(0).to(tl.int64)', *loop, *[indent + line for line in body]]) + '\n'


def emit_npu_fft256_pair(*, mode: int, radix4_store: bool, direction: str,
                        out_dir: Path) -> dict:
    if direction not in ("forward", "inverse") or (mode == 1 and direction != "forward") or (mode == 2 and direction != "inverse"):
        raise ValueError("FFT direction and real mode disagree")
    if radix4_store:
        if mode != 0:
            raise ValueError("Transposed radix4 store requires complex mode")
        name, source = build_npu_radix4_pair_source(length=256, group=8, mode=0,
            transposed_store=True)
    else:
        name, source = build_npu_fft256_pair_source(mode)
    names = ["in_ptr", "out_ptr", "idx_ptr", "twiddle_ptr", "nbatch"]
    signature = "*fp32:16,*fp32:16,*u32:16,*fp32:16,i32"
    if radix4_store:
        names += ["output_row_stride", "output_transform_offset"]
        signature += ",i32,i32"
    out_dir.mkdir(parents=True, exist_ok=True)
    module_path = out_dir / f"flagfft_npu_fft256_{name}_{direction}.py"
    write_text_atomic(module_path, _module_source(source))
    metadata = dict(module_path=str(module_path), kernel_name=name,
        signature=signature, arg_names=names, num_warps=4, num_stages=1,
        batch_per_block=8, kernel_type="npu_fft256_pair", dtype="complex64",
        direction=direction, mode=mode, radix4_store=radix4_store, length=256)
    write_text_atomic(module_path.with_suffix(".json"), json.dumps(metadata, sort_keys=True))
    return metadata
