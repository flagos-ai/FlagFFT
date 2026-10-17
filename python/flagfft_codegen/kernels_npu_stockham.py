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

"""The existing Ascend 128-lane Stockham mapping and table ABI in Triton/TLE."""

import json
from pathlib import Path

from .artifacts import write_text_atomic
from .metadata import _module_source


def npu_stockham_layout(n, radix, span):
    if not (n > 0 and 2 <= radix <= 19 and span > 0 and n % (radix * span) == 0):
        raise ValueError((n, radix, span))
    if n > 0xffffffff or span > 0xffffffff:
        raise ValueError("Ascend stage parameters must fit uint32")
    ndiv = n // radix
    extent = ndiv if span == 1 else min(span, 128)
    batch_group = 128 // extent if 0 < extent < 128 and extent % 4 == 0 else 1
    packed_groups = 128 // span if 1 < span < 128 and batch_group == 1 else 1
    group_tiles = (span + 127) // 128 if span > 1 else 1
    groups = ndiv // span if span > 1 else 1
    tiles = ((ndiv + 127) // 128 if span == 1 else
             (groups + packed_groups - 1) // packed_groups * group_tiles)
    contiguous = span <= 128 and packed_groups == 1
    return dict(ndiv=ndiv, extent=extent, batch_group=batch_group,
                packed_groups=packed_groups, group_tiles=group_tiles,
                groups=groups, tiles=tiles, contiguous=contiguous,
                twiddle_stride=128 if packed_groups > 1 else span,
                store_lanes=min(ndiv, 128) if span == 1 else extent)


def build_npu_table_stockham_source(n, radix, span):
    p = npu_stockham_layout(n, radix, span)
    nd, extent, bg, pg, gt, ng, nt = [p[k] for k in
        ('ndiv', 'extent', 'batch_group', 'packed_groups', 'group_tiles', 'groups', 'tiles')]
    rpad = 1 << (radix - 1).bit_length()
    root_width = 1 << (2 * radix * 24 - 1).bit_length()
    packed_width = 1 << (2 * radix * 128 - 1).bit_length()
    bpad = 1 << (bg - 1).bit_length()
    name = f'npu_table_stockham_n{n}_r{radix}_s{span}'
    lines = ['@triton.jit',
        f'def {name}(in_ptr, out_ptr, stage_twiddles, roots_ptr, indices_ptr, store_indices, length_arg, radix_arg, span_arg, nbatch):',
        '    block = tl.program_id(0).to(tl.int64)',
        f'    total_tiles = ((nbatch + {bg - 1}) // {bg}) * {nt}',
        f'    root_lane = tl.arange(0, {root_width})',
        f'    roots = tl.load(roots_ptr + root_lane, root_lane < {2 * radix * 24}, 0.0)',
        '    lane = tl.arange(0, 128)',
        f'    q = tl.arange(0, {rpad})',
        '    for tile_id in range(block, total_tiles, tl.num_programs(0)):',
        f'        batch_base = (tile_id // {nt}) * {bg}',
        f'        local_tile = tile_id % {nt}']
    if span == 1:
        lines += ['        k_start = local_tile * 128', '        j_start = 0',
                  f'        valid = tl.minimum({nd} - k_start, 128)',
                  '        valid_per_group = valid', '        groups_in_tile = 1']
    else:
        lines += [f'        group = (local_tile // {gt}) * {pg}',
                  f'        j_start = (local_tile % {gt}) * 128',
                  f'        k_start = group * {span} + j_start',
                  f'        valid_per_group = tl.minimum({span} - j_start, 128)',
                  f'        groups_in_tile = tl.minimum({ng} - group, {pg})',
                  f'        valid = groups_in_tile * valid_per_group' if pg > 1 else
                  '        valid = valid_per_group']
    lines += [f'        acc_r = tl.zeros(({rpad}, 128), tl.float32)',
              f'        acc_i = tl.zeros(({rpad}, 128), tl.float32)']
    for digit in range(radix):
        if bg == 1:
            lines += ['        scalar = tl.arange(0, 256)',
                f'        src = (batch_base * {n} + k_start + {digit * nd}) * 2',
                '        raw = tl.load(in_ptr + src + scalar, (scalar < valid * 2) & (batch_base < nbatch), 0.0)',
                '        xr = tl.gather(raw, lane * 2, 0)',
                '        xi = tl.gather(raw, lane * 2 + 1, 0)']
        else:
            width = 1 << (2 * extent - 1).bit_length()
            lines += [f'        transform = tl.arange(0, {bpad})',
                f'        scalar = tl.arange(0, {width})',
                f'        src = ((batch_base + transform[:, None]) * {n} + k_start + {digit * nd}) * 2 + scalar[None, :]',
                f'        mask = (transform[:, None] < {bg}) & (batch_base + transform[:, None] < nbatch) & (scalar[None, :] < valid * 2)',
                '        raw2d = tl.load(in_ptr + src, mask, 0.0)',
                f'        raw = tl.reshape(raw2d, ({bpad * width},))',
                f'        local_real = (lane // {extent}) * {width} + (lane % {extent}) * 2',
                f'        xr = tl.where(lane < {bg * extent}, tl.gather(raw, local_real, 0), 0.0)',
                f'        xi = tl.where(lane < {bg * extent}, tl.gather(raw, local_real + 1, 0), 0.0)']
        if digit and span > 1:
            lines += ['        scalar = tl.arange(0, 256)',
                f'        twbase = ({digit * p["twiddle_stride"]} + j_start) * 2',
                '        twraw = tl.load(stage_twiddles + twbase + scalar, scalar < valid * 2, 0.0)']
            if bg == 1:
                lines += ['        wr = tl.gather(twraw, lane * 2, 0)',
                          '        wi = tl.gather(twraw, lane * 2 + 1, 0)']
            else:
                lines += [f'        wr = tl.gather(twraw, (lane % {extent}) * 2, 0)',
                          f'        wi = tl.gather(twraw, (lane % {extent}) * 2 + 1, 0)']
            lines += ['        xr, xi = _cmul(xr, xi, wr, wi)']
        if digit == 0:
            lines += ['        acc_r = acc_r + xr[None, :]',
                      '        acc_i = acc_i + xi[None, :]']
        else:
            lines += [f'        qr = tl.gather(roots, (q % {radix}) * 24 + {digit}, 0)',
                f'        qi = tl.gather(roots, {radix * 24} + (q % {radix}) * 24 + {digit}, 0)',
                '        pr, pi = _cmul(xr[None, :], xi[None, :], qr[:, None], qi[:, None])',
                '        acc_r = acc_r + pr', '        acc_i = acc_i + pi']
    lines += [f'        packed = tl.zeros(({packed_width},), tl.float32)',
        f'        packed = tle.dsa.insert_slice(packed, tl.reshape(acc_r, ({rpad * 128},)), [0], [{rpad * 128}], [1])',
        f'        packed = tle.dsa.insert_slice(packed, tl.reshape(acc_i, ({rpad * 128},)), [{radix * 128}], [{rpad * 128}], [1])']
    if p['contiguous']:
        count = 2 * radix * p['store_lanes']
        stride = (count + 7) & ~7
        width = 1 << (stride - 1).bit_length()
        lines += [f'        transform = tl.arange(0, {bpad})',
            f'        scalar = tl.arange(0, {width})',
            f'        mapping2d = tl.load(store_indices + transform[:, None] * {stride} + scalar[None, :], (transform[:, None] < {bg}) & (scalar[None, :] < {stride}), 0) >> 2',
            f'        mapping = tl.reshape(mapping2d, ({bpad * width},))',
            f'        values = tl.reshape(tl.gather(packed, mapping, 0), ({bpad}, {width}))',
            f'        out_base = ((batch_base + transform[:, None]) * {n} + {radix} * k_start - {radix - 1} * j_start) * 2',
            f'        mask = (transform[:, None] < {bg}) & (batch_base + transform[:, None] < nbatch) & (scalar[None, :] < {2 * radix} * valid)',
            '        tl.store(out_ptr + out_base + scalar[None, :], values, mask)']
    elif pg == 1:
        # The native body writes each distant q segment separately. Keep that
        # transfer when combining q into one tensor would exceed DMA strides.
        for output in range(radix):
            lines += ['        scalar = tl.arange(0, 256)',
                f'        mapping = {output * 128} + (scalar >> 1) + (scalar & 1) * {radix * 128}',
                '        values = tl.gather(packed, mapping, 0)',
                f'        out_base = (batch_base * {n} + {radix} * k_start - {radix - 1} * j_start + {output * span}) * 2',
                '        tl.store(out_ptr + out_base + scalar, values, (batch_base < nbatch) & (scalar < valid_per_group * 2))']
    else:
        # q, packed group and contiguous AoS columns are explicit tensor axes.
        gp = 1 << (pg - 1).bit_length()
        cols = span if pg > 1 else 128
        width = 1 << (2 * cols - 1).bit_length()
        lines += [f'        slot = tl.arange(0, {gp})',
            f'        scalar = tl.arange(0, {width})',
            f'        local_lane = slot[None, :, None] * {span if pg > 1 else 0} + (scalar[None, None, :] >> 1)',
            f'        mapping3d = (q[:, None, None] % {radix}) * 128 + (local_lane % 128) + (scalar[None, None, :] & 1) * {radix * 128}',
            f'        mapping = tl.reshape(mapping3d, ({rpad * gp * width},))',
            f'        values = tl.reshape(tl.gather(packed, mapping, 0), ({rpad}, {gp}, {width}))',
            f'        out_base = (batch_base * {n} + {radix} * k_start - {radix - 1} * j_start + slot[None, :, None] * {radix * span if pg > 1 else 0} + q[:, None, None] * {span}) * 2',
            f'        mask = (batch_base < nbatch) & (q[:, None, None] < {radix}) & (slot[None, :, None] < groups_in_tile) & (scalar[None, None, :] < valid_per_group * 2)',
            '        tl.store(out_ptr + out_base + scalar[None, None, :], values, mask)']
    return name, '\n'.join(lines) + '\n'


def emit_npu_table_stockham(*, length: int, radix: int, span: int,
                            direction: str, out_dir: Path) -> dict:
    if direction not in ("forward", "inverse"):
        raise ValueError("Stockham direction must be forward or inverse")
    name, source = build_npu_table_stockham_source(length, radix, span)
    layout = npu_stockham_layout(length, radix, span)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"flagfft_npu_stockham_n{length}_r{radix}_s{span}_{direction}.py"
    write_text_atomic(path, _module_source(source))
    metadata = {
        "module_path": str(path), "kernel_name": name,
        "signature": ",".join(["*fp32:16"] * 4 + ["*u32:16"] * 2 + ["i32"] * 4),
        "arg_names": ["in_ptr", "out_ptr", "stage_twiddles", "roots_ptr", "indices_ptr",
                      "store_indices", "length_arg", "radix_arg", "span_arg", "nbatch"],
        "kernel_type": "npu_table_stockham", "dtype": "complex64",
        "direction": direction, "length": length, "radix": radix, "span": span,
        "num_warps": 4, "num_stages": 1, "batch_per_block": 1,
        "tile": 128, "batch_group": layout["batch_group"],
        "tiles_per_transform": layout["tiles"],
    }
    write_text_atomic(path.with_suffix(".json"), json.dumps(metadata, sort_keys=True))
    return metadata
