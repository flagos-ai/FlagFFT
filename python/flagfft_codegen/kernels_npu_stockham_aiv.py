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

"""Tile-based Stockham stage for the Ascend AIV table contract.

The native AIV stage processes a 128-lane tile with a q-major radix DFT.
This emitter keeps the same roots/twiddle/store-table contract but uses a
single transform per tile group, which matches the measured 1D CT stage
costs while avoiding the native batch-packing cases. All per-lane inputs and
twiddles are contiguous loads followed by ``tl.gather``.
"""

from __future__ import annotations

import json
from pathlib import Path

from .artifacts import write_text_atomic
from .metadata import _module_source

TILE = 128
ROOT_STRIDE = 24


def build_npu_stockham_aiv_source(*, length: int, radix: int, span: int,
                                  tile: int = TILE) -> tuple[str, str]:
    if tile != TILE:
        raise ValueError("AIV Stockham requires a 128-lane tile")
    if length <= 0 or radix < 2 or length % radix:
        raise ValueError("AIV Stockham length must be divisible by radix")
    ndiv = length // radix
    if span <= 0:
        raise ValueError("AIV Stockham span must be positive")
    if span == 1:
        if ndiv % tile:
            raise ValueError("span=1 AIV Stockham requires ndiv divisible by 128")
        packed = tile
        tiles = (ndiv + tile - 1) // tile
        tiles_per_group = 1
        tw_stride = 1
    elif span <= tile:
        if tile % span or ndiv % span:
            raise ValueError("AIV Stockham span must divide the tile and ndiv")
        packed = tile // span
        groups = ndiv // span
        tiles = (groups + packed - 1) // packed
        tiles_per_group = 1
        tw_stride = tile if packed > 1 else span
    else:
        if span % tile or ndiv % span:
            raise ValueError("long AIV Stockham span must be a tile multiple")
        packed = 1
        tiles_per_group = span // tile
        tiles = (ndiv // span) * tiles_per_group
        tw_stride = span

    name = f"npu_stockham_aiv_n{length}_r{radix}_s{span}_t{tile}"
    if span <= tile:
        group_expr = f"(tile_index * {packed})"
        k_start_expr = f"(group * {span})"
        j_start_expr = "0"
        tw_offset = f"digit * {2 * tw_stride}"
    else:
        group_expr = f"(tile_index // {tiles_per_group})"
        k_start_expr = f"(group * {span} + j_start)"
        j_start_expr = f"(tile_index % {tiles_per_group}) * {tile}"
        tw_offset = f"(digit * {2 * tw_stride} + j_start * 2)"
    tw_condition = "digit != 0" if span > 1 else "False"

    source = f'''@triton.jit
def {name}(in_ptr, out_ptr, twiddle_ptr, roots_ptr, store_ptr, nbatch):
    pid = tl.program_id(0).to(tl.int64)
    local = tl.arange(0, {tile})
    q = tl.arange(0, {radix})
    batch_index = pid // {tiles}
    tile_index = pid % {tiles}
    group = {group_expr}
    j_start = {j_start_expr}
    k_start = {k_start_expr}
    input_base = batch_index * {2 * length} + k_start * 2
    acc_r = tl.zeros(({radix}, {tile}), tl.float32)
    acc_i = tl.zeros(({radix}, {tile}), tl.float32)
    for digit in tl.static_range(0, {radix}):
        source = tl.load(in_ptr + input_base + digit * {2 * ndiv}
                         + tl.arange(0, {2 * tile}))
        xr = tl.gather(source, local * 2, 0)
        xi = tl.gather(source, local * 2 + 1, 0)
        if {tw_condition}:
            tw_source = tl.load(twiddle_ptr + {tw_offset}
                                + tl.arange(0, {2 * tile}))
            wr = tl.gather(tw_source, local * 2, 0)
            wi = tl.gather(tw_source, local * 2 + 1, 0)
            xr, xi = xr * wr - xi * wi, xr * wi + xi * wr
        cr = tl.load(roots_ptr + q * {ROOT_STRIDE} + digit)
        ci = tl.load(roots_ptr + {radix * ROOT_STRIDE} + q * {ROOT_STRIDE} + digit)
        acc_r += xr[None, :] * cr[:, None] - xi[None, :] * ci[:, None]
        acc_i += xr[None, :] * ci[:, None] + xi[None, :] * cr[:, None]
    packed = tl.zeros(({2 * radix * tile},), tl.float32)
    packed = tle.dsa.insert_slice(packed, tl.reshape(acc_r, ({radix * tile},)),
                                  [0], [{radix * tile}], [1])
    packed = tle.dsa.insert_slice(packed, tl.reshape(acc_i, ({radix * tile},)),
                                  [{radix * tile}], [{radix * tile}], [1])
    index = tl.load(store_ptr + tl.arange(0, {2 * radix * tile})) >> 2
    result = tl.gather(packed, index, 0)
    output_base = (batch_index * {2 * length} + {radix} * k_start * 2
                   - {radix - 1} * j_start * 2)
    tl.store(out_ptr + output_base + tl.arange(0, {2 * radix * tile}), result)
'''
    return name, source


def aiv_tiles(*, length: int, radix: int, span: int, tile: int = TILE) -> int:
    ndiv = length // radix
    if span <= tile:
        packed = tile // span
        return (ndiv // span + packed - 1) // packed
    return (ndiv // span) * (span // tile)


def emit_npu_stockham_aiv(*, length: int, radix: int, span: int,
                          direction: str, dtype: str, out_dir: Path,
                          tile: int = TILE) -> dict:
    if dtype != "complex64":
        raise ValueError("AIV Stockham requires complex64")
    name, source = build_npu_stockham_aiv_source(
        length=length, radix=radix, span=span, tile=tile)
    out_dir.mkdir(parents=True, exist_ok=True)
    module_path = out_dir / f"{name}.py"
    write_text_atomic(module_path, _module_source(source))
    arg_names = ["in_ptr", "out_ptr", "twiddle_ptr", "roots_ptr", "store_ptr", "nbatch"]
    metadata = {
        "module_path": str(module_path),
        "kernel_name": name,
        "signature": "*fp32:16,*fp32:16,*fp32:16,*fp32:16,*i32:16,i32",
        "num_warps": 4,
        "num_stages": 1,
        "batch_per_block": 1,
        "butterflies_per_block": tile,
        "arg_names": arg_names,
        "kernel_type": "npu_stockham_aiv",
        "dtype": dtype,
        "direction": direction,
        "length": length,
        "radix": radix,
        "span": span,
        "tile": tile,
    }
    write_text_atomic(module_path.with_suffix(".json"), json.dumps(metadata, sort_keys=True))
    return metadata
