# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

"""Coalesced layout passes for the existing generic four-step ABI."""


def build_npu_tiled_reshape_source(rows, cols, *, twiddle=False):
    if rows < 1 or cols < 1:
        raise ValueError("reshape dimensions must be positive")
    tile = 16
    nc = (cols + tile - 1) // tile
    nr = (rows + tile - 1) // tile
    local_type = "int32" if 2 * (rows + tile) * (cols + tile) < 2**31 else "int64"
    name = f'npu_tiled_{"twiddle_" if twiddle else ""}reshape_r{rows}_c{cols}_t{tile}'
    names = ['in_ptr', *(['twiddle_ptr'] if twiddle else []), 'out_ptr', 'nbatch']
    lines = ['@triton.jit', f'def {name}({", ".join(names)}):',
        '    batch = tl.program_id(1).to(tl.int64)',
        f'    input_batch = in_ptr + batch * {rows * cols * 2}',
        f'    output_batch = out_ptr + batch * {rows * cols * 2}',
        '    lane = tl.arange(0, 16)', '    word = tl.arange(0, 32)',
        '    z = tl.arange(0, 512)',
        '    transpose_map = ((z >> 5) << 1) + (((z >> 1) & 15) << 5) + (z & 1)',
        f'    for tile_id in range(tl.program_id(0), {nr * nc}, tl.num_programs(0)):',
        f'        row_start = (tile_id // {nc}).to(tl.{local_type}) * 16',
        f'        col_start = (tile_id % {nc}).to(tl.{local_type}) * 16',
        '        row = row_start + lane', '        source_word = col_start * 2 + word',
        f'        offset = row[:, None] * {cols * 2} + source_word[None, :]',
        f'        mask = (row[:, None] < {rows}) & (source_word[None, :] < {cols * 2}) & (batch < nbatch)',
        '        value = tl.load(input_batch + offset, mask, 0.0)',
        '        flat = tl.reshape(value, (512,))']
    if twiddle:
        lines += ['        tw = tl.load(twiddle_ptr + offset, mask, 0.0)',
            '        tw_flat = tl.reshape(tw, (512,))',
            '        pair = tl.arange(0, 256) * 2',
            '        xr = tl.gather(flat, pair, 0)', '        xi = tl.gather(flat, pair + 1, 0)',
            '        wr = tl.gather(tw_flat, pair, 0)', '        wi = tl.gather(tw_flat, pair + 1, 0)',
            '        yr, yi = _cmul(xr, xi, wr, wi)',
            '        combined = tl.zeros((512,), tl.float32)',
            '        combined = tle.dsa.insert_slice(combined, yr, [0], [256], [1])',
            '        combined = tle.dsa.insert_slice(combined, yi, [256], [256], [1])',
            '        flat = tl.gather(combined, (z >> 1) + (z & 1) * 256, 0)']
    lines += ['        result = tl.reshape(tl.gather(flat, transpose_map, 0), (16, 32))',
        '        col = col_start + lane', '        target_word = row_start * 2 + word',
        f'        target = col[:, None] * {rows * 2} + target_word[None, :]',
        f'        mask = (col[:, None] < {cols}) & (target_word[None, :] < {rows * 2}) & (batch < nbatch)',
        '        tl.store(output_batch + target, result, mask)']
    return name, '\n'.join(lines) + '\n', names
