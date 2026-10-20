# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

"""Triton/TLE bodies for the existing Ascend grouped FFT table contract."""

from __future__ import annotations

import json
from pathlib import Path

from .artifacts import write_text_atomic
from .metadata import _module_source, _signature
from .kernels_npu_radix4 import build_npu_radix4_pair_source


def build_npu_grouped_fft64_source(*, group: int, stride: int, mode: int,
                                 radix4_pair: bool = False) -> str:
    """Preserve the selected native stages, grouping and input/output layout.

    Tables contain byte offsets; all gather indices below are FP32 elements.
    Radix2 group8 retains its two sequential groups of four; radix4 pairs use
    the separate quarter-vector contract. The parent plan owns inverse scale.
    """
    if radix4_pair:
        if stride != 1:
            raise ValueError("Radix4 pair FFT64 requires contiguous rows")
        name, source = build_npu_radix4_pair_source(length=64, group=group, mode=mode)
        return source.replace(f"def {name}(", "def npu_grouped_fft64_kernel(")
    if group not in (4, 8) or stride not in (1, 64) or mode not in (0, 1, 2):
        raise ValueError("FFT64 requires group 4/8, stride 1/64 and mode 0/1/2")
    if mode != 0 and stride != 1:
        raise ValueError("Real FFT64 requires contiguous rows")
    count = 256 if mode == 2 else 264 if mode == 1 else 512
    output_size = 256 if mode == 2 else 512
    input_count = group * (66 if mode == 2 else 64 if mode == 1 else 128)
    input_size = 1 << (input_count - 1).bit_length()
    load = f"""offset = pid * {input_count} + tl.arange(0, {input_size})
    source = tl.load(in_ptr + offset, tl.arange(0, {input_size}) < {input_count}, 0.0)"""
    if stride == 64:
        load = f"""matrix = pid >> {(64 // group).bit_length() - 1}
    column = (pid & {64 // group - 1}) * {group}
    offset = (matrix * 8192 + column * 2
              + tl.arange(0, 64)[:, None] * 128
              + tl.arange(0, {2 * group})[None, :])
    source = tl.reshape(tl.load(in_ptr + offset), ({128 * group},))"""
    imag = "imag = tl.gather(source, input_index + 1, 0)"
    if mode == 1:
        imag = "imag = tl.full((256,), 0.0, tl.float32)"
    elif mode == 2:
        imag = f"""imag_index = tl.load(idx_ptr + {3840 + (256 if group == 8 else 0)}
                             + part * 256 + local) >> 2
        imag = tl.gather(source, imag_index, 0) * tl.load(twiddle_ptr + 3072 + local)"""
    packed = "packed = real" if mode == 2 else """packed = tl.zeros((512,), tl.float32)
        packed = tle.dsa.insert_slice(packed, real, [0], [256], [1])
        packed = tle.dsa.insert_slice(packed, imag, [256], [256], [1])"""
    output = f"""offset = (pid * {group * (64 if mode == 2 else 66 if mode == 1 else 128)}
                  + part * {count} + tl.arange(0, {output_size}))
        tl.store(out_ptr + offset, result, tl.arange(0, {output_size}) < {count})"""
    if stride == 64:
        output = """offset = (matrix * 8192 + (column + part * 4) * 2
                  + tl.arange(0, 64)[:, None] * 128 + tl.arange(0, 8)[None, :])
        tl.store(out_ptr + offset, tl.reshape(result, (64, 8)))"""
    return f"""@triton.jit
def npu_grouped_fft64_kernel(in_ptr, out_ptr, idx_ptr, twiddle_ptr, nbatch):
    pid = tl.program_id(0).to(tl.int64)
    {load}
    local = tl.arange(0, 256)
    for part in tl.static_range(0, {group // 4}):
        input_offset = 0 if part == 0 else 3840
        input_index = tl.load(idx_ptr + input_offset + local) >> 2
        real = tl.gather(source, input_index, 0)
        {imag}
        for stage in tl.static_range(0, 6):
            a = tl.load(idx_ptr + 768 + stage * 256 + local) >> 2
            b = tl.load(idx_ptr + 2304 + stage * 256 + local) >> 2
            wr = tl.load(twiddle_ptr + stage * 256 + local)
            wi = tl.load(twiddle_ptr + 1536 + stage * 256 + local)
            ar = tl.gather(real, a, 0)
            ai = tl.gather(imag, a, 0)
            br = tl.gather(real, b, 0)
            bi = tl.gather(imag, b, 0)
            product_r = br * wr - bi * wi
            product_i = br * wi + bi * wr
            real = ar + product_r
            imag = ai + product_i
        {packed}
        output_index = tl.load(idx_ptr + 256 + tl.arange(0, {output_size})) >> 2
        result = tl.gather(packed, output_index, 0)
        {output}
"""


def emit_npu_grouped_fft64(*, group: int, stride: int, mode: int,
                         direction: str, out_dir: Path, radix4_pair: bool = False) -> dict:
    if direction not in ("forward", "inverse"):
        raise ValueError("FFT direction must be forward or inverse")
    if (mode == 1 and direction != "forward") or (mode == 2 and direction != "inverse"):
        raise ValueError("Real FFT mode disagrees with its direction")
    source = build_npu_grouped_fft64_source(group=group, stride=stride, mode=mode,
                                           radix4_pair=radix4_pair)
    suffix = "_r4" if radix4_pair else ""
    name = f"flagfft_npu_fft64_g{group}_s{stride}_m{mode}_{direction}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    module_path = out_dir / f"{name}.py"
    write_text_atomic(module_path, _module_source(source))
    arg_names = ["in_ptr", "out_ptr", "idx_ptr", "twiddle_ptr", "nbatch"]
    metadata = {
        "module_path": str(module_path),
        "kernel_name": "npu_grouped_fft64_kernel",
        "signature": _signature(arg_names, "complex64"),
        "num_warps": 4, "num_stages": 1, "batch_per_block": group,
        "arg_names": arg_names, "kernel_type": "npu_grouped_fft64",
        "dtype": "complex64", "direction": direction, "length": 64,
        "group_size": group, "stride": stride, "mode": mode,
        "radix4_pair": radix4_pair,
    }
    write_text_atomic(module_path.with_suffix(".json"), json.dumps(metadata, sort_keys=True))
    return metadata
