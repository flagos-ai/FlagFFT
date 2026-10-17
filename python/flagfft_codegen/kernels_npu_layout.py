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

"""Isolated generated replacements for the Ascend layout components."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

from .artifacts import write_text_atomic
from .kernels_common import _npu_backend_active
from .metadata import _module_source, _signature


def build_npu_table_transpose3d_source(n0: int, n1: int, n2: int, order: str):
    """Keep the native 16³ tile and byte-offset permutation table contract.

    AoS GM addresses and masks retain their affine scalar coordinate. Rebuilding
    that coordinate from complex index and parity scalarizes accesses in the
    current Ascend compiler. Outer batch addressing remains 64 bit.
    """
    dims = (n0, n1, n2)
    if min(dims) <= 0 or order not in {"021", "210", "201", "120"}:
        raise ValueError("NPU transpose3d requires positive dimensions and a supported order")
    if 2 * math.prod(dims) > 2**31 - 1:
        raise ValueError("NPU transpose3d tile coordinates require a 32-bit matrix offset")
    tile = 16
    axes = tuple(map(int, order))
    output = tuple(dims[axis] for axis in axes)
    tiles = tuple((dim + tile - 1) // tile for dim in output)
    starts = [f"tile{axes.index(axis)} * {tile}" for axis in range(3)]
    name = f"npu_table_transpose3d_n{n0}_{n1}_{n2}_o{order}"
    source = f"""@triton.jit
def {name}(in_ptr, out_ptr, idx_ptr, nbatch):
    pid = tl.program_id(0)
    batch = tl.program_id(2).to(tl.int64)
    tile0 = pid // {tiles[1] * tiles[2]}
    tile1 = (pid // {tiles[2]}) % {tiles[1]}
    tile2 = pid % {tiles[2]}
    a = tl.arange(0, 16)[:, None, None]
    b = tl.arange(0, 16)[None, :, None]
    c = tl.arange(0, 32)[None, None, :]
    i0 = {starts[0]} + a
    i1 = {starts[1]} + b
    input_batch = in_ptr + batch * {2 * n0 * n1 * n2}
    offset = ((i0 * {n1} + i1) * {n2} + ({starts[2]})) * 2 + c
    data = tl.load(input_batch + offset,
                   (i0 < {n0}) & (i1 < {n1})
                   & (c < 2 * ({n2} - ({starts[2]}))) & (batch < nbatch), 0.0)
    data = tl.reshape(data, (8192,))
    index = tl.load(idx_ptr + tl.arange(0, 8192)) >> 2
    result = tl.reshape(tl.gather(data, index, 0), (16, 16, 32))
    o0 = tile0 * 16 + a
    o1 = tile1 * 16 + b
    output_batch = out_ptr + batch * {2 * n0 * n1 * n2}
    destination = ((o0 * {output[1]} + o1) * {output[2]} + tile2 * 16) * 2 + c
    tl.store(output_batch + destination, result,
             (o0 < {output[0]}) & (o1 < {output[1]})
             & (c < 2 * ({output[2]} - tile2 * 16)) & (batch < nbatch))
"""
    return name, source, math.prod(tiles)


def emit_npu_table_transpose3d(*, n0: int, n1: int, n2: int, order: str,
                             dtype: str, out_dir: Path) -> dict:
    if not _npu_backend_active() or dtype != "complex64":
        raise ValueError("Table transpose3d requires an NPU FP32 target")
    name, source, blocks = build_npu_table_transpose3d_source(n0, n1, n2, order)
    warps = os.environ.get("FLAGFFT_NPU_3D_TRANSPOSE_WARPS", "4")
    if warps not in {"4", "8"}:
        raise ValueError("FLAGFFT_NPU_3D_TRANSPOSE_WARPS must be 4 or 8")
    out_dir.mkdir(parents=True, exist_ok=True)
    module_path = out_dir / f"flagfft_jit_{name}.py"
    write_text_atomic(module_path, _module_source(source))
    names = ["in_ptr", "out_ptr", "idx_ptr", "nbatch"]
    metadata = {
        "module_path": str(module_path), "kernel_name": name,
        "signature": _signature(names, dtype), "arg_names": names,
        "num_warps": int(warps), "num_stages": 1, "batch_per_block": 1,
        "kernel_type": "transpose3d", "dtype": dtype, "tile_size": 16,
        "transpose3d_n0": n0, "transpose3d_n1": n1, "transpose3d_n2": n2,
        "transpose3d_order": order, "grid_x_override": blocks,
    }
    write_text_atomic(module_path.with_suffix(".json"), json.dumps(metadata, sort_keys=True))
    return metadata
