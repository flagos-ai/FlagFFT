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

"""Table-based replacement for the existing Ascend complex transpose."""

from textwrap import dedent


# The current NPU build targets Ascend910B4. Bound independent programs to its
# vector-core wave and reuse the byte-offset table across grid-stride tiles.
# This resource policy applies to every eligible matrix and batch.
NPU_TRANSPOSE_PROGRAM_BUDGET = 40


def build_npu_table_transpose2d_source(
    rows: int, cols: int, tile: int, programs: int = NPU_TRANSPOSE_PROGRAM_BUDGET
) -> tuple[str, str, list[str], int]:
    if rows <= 0 or cols <= 0 or tile not in (16, 32, 64) or programs <= 0:
        raise ValueError("NPU transpose needs positive dimensions/programs and tile 16, 32 or 64")
    row_tiles = (rows + tile - 1) // tile
    col_tiles = (cols + tile - 1) // tile
    tiles = row_tiles * col_tiles
    coordinate_type = "int32" if 2 * (rows + tile) * (cols + tile) < 2**31 else "int64"
    tile_type = "int32" if tiles < 2**31 else "int64"
    name = f"npu_table_transpose2d_n{rows}_{cols}_t{tile}_p{programs}"
    source = dedent(f"""
        @triton.jit
        def {name}(in_ptr, out_ptr, idx_ptr, nbatch):
            pid = tl.program_id(0).to(tl.{tile_type})
            batch = tl.program_id(2).to(tl.int64)
            source_base = in_ptr + batch * {2 * rows * cols}
            target_base = out_ptr + batch * {2 * rows * cols}
            lane = tl.arange(0, {2 * tile * tile})
            index = tl.load(idx_ptr + lane) >> 2
            r = tl.arange(0, {tile})
            w = tl.arange(0, {2 * tile})
            for tile_id in range(pid, {tiles}, tl.num_programs(0)):
                tile_row = (tile_id // {col_tiles}).to(tl.{coordinate_type})
                tile_col = (tile_id % {col_tiles}).to(tl.{coordinate_type})
                row = tile_row * {tile} + r
                word = tile_col * {2 * tile} + w
                data = tl.load(source_base + row[:, None] * {2 * cols} + word[None, :],
                               (row[:, None] < {rows}) & (word[None, :] < {2 * cols}) & (batch < nbatch), 0.0)
                data = tl.reshape(data, ({2 * tile * tile},))
                data = tl.reshape(tl.gather(data, index, 0), ({tile}, {2 * tile}))
                col = tile_col * {tile} + r
                word = tile_row * {2 * tile} + w
                tl.store(target_base + col[:, None] * {2 * rows} + word[None, :], data,
                         (col[:, None] < {cols}) & (word[None, :] < {2 * rows}) & (batch < nbatch))
    """)
    return name, source, ["in_ptr", "out_ptr", "idx_ptr", "nbatch"], min(programs, tiles)
