# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

"""Shared plan model, dtype helpers and occupancy/heuristic policy for kernel generation."""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

_MODULE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parents[1]
_CODELET_DIR = _MODULE_DIR / "codelet"
_FOUR_STEP_TILE_ROWS = 32
_FOUR_STEP_TILE_COLS = 32
_FOUR_STEP_NUM_WARPS = 4
_FOUR_STEP_COL_INNER_PACK = 4
_FOUR_STEP_LARGE_INNER_PACK = 4
_FOUR_STEP_COL_INNER_PACK_MIN_N1 = 128
_FOUR_STEP_ROW_INNER_PACK_MAX_N1 = 512
_FOUR_STEP_PACKED_COL_LEAF_MAX_N2 = 1024
_FOUR_STEP_PACK_TARGET_THREADS = 256
_FOUR_STEP_PACK_SMEM_BUDGET_BYTES = 128 * 1024
_TLE_FUSED_TWIDDLE_MIN_LENGTH = 1 << 18
_TLE_FUSED_TWIDDLE_MAX_LEAF = 1024
_TLE_SMEM_SWIZZLE_SHIFT = 5
_THREAD_LOCAL_MIXED_RADICES = frozenset({18, 20, 24, 25, 27, 28, 30, 32})
_COOPERATIVE_STAGE_MIN_LENGTH = 128
_COOPERATIVE_STAGE_MAX_LENGTH = 4096
_COOPERATIVE_STAGE_MAX_BASE_LANES = 32
_COOPERATIVE_STAGE_MAX_LANES = 128
_LEAF_PACK_TARGET_THREADS = 32
_LEAF_PACK_SMEM_BUDGET_BYTES = 48 * 1024
_NATURAL_ORDER_CODELET_RADICES = frozenset(
    {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 19}
)
_THREAD_LOCAL_MIXED_SPLITS = {
    18: (3, 6),
    20: (5, 4),
    24: (3, 8),
    25: (5, 5),
    27: (3, 9),
    28: (7, 4),
    30: (3, 10),
}


def codelet_radices_for(factors: tuple[int, ...]) -> set[int]:
    """Return the bundled ``codelet/radixN.py`` files a leaf needs.

    Radix-16 is emitted through the special factorized call; radix-32 is
    built from two radix-16 codelets; thread-local mixed radices expand to
    their factorized codelets. Generic/table radices have no bundled file.
    """
    needed: set[int] = set()
    for radix in factors:
        if radix in _NATURAL_ORDER_CODELET_RADICES:
            needed.add(radix)
        elif radix in (16, 32):
            needed.add(16)
        elif radix in _THREAD_LOCAL_MIXED_RADICES:
            needed.update(_THREAD_LOCAL_MIXED_SPLITS[radix])
    return needed


def _is_double_dtype(dtype: str) -> bool:
    return dtype in ("complex128", "float64")


def _tl_real_dtype(dtype: str) -> str:
    return "tl.float64" if _is_double_dtype(dtype) else "tl.float32"


def _real_element_bytes(dtype: str) -> int:
    return 8 if _is_double_dtype(dtype) else 4


def _dtype_suffix(dtype: str) -> str:
    return "f64" if _is_double_dtype(dtype) else "f32"


def _zero_other(dtype: str) -> str:
    # auto-promoted by Triton's masked-load lowering; one literal works for fp32 and fp64
    del dtype
    return "0.0"


@dataclass(frozen=True)
class LeafPlan:
    length: int
    factors: tuple[int, ...]
    remainder: int
    lanes: int
    num_warps: int
    generic_radices: tuple[int, ...]
    smem_size: int
    direction: Literal["forward", "inverse"] = "forward"
    dtype: Literal["complex64", "complex128"] = "complex64"
    kind: Literal["ct_leaf"] = field(default="ct_leaf", init=False)


LeafIoMode = Literal[
    "contiguous",
    "strided",
    "contiguous_r2c",
    "contiguous_c2r",
    "bluestein_prepare_leaf",
    "bluestein_finish_leaf",
    "bluestein_full_leaf",
    "bluestein_four_step_prepare_row",
    "bluestein_four_step_pointwise_row",
    "bluestein_four_step_finish_col",
    "four_step_row",
    "four_step_real_row",
    "four_step_hermitian_row",
    "four_step_col",
    "four_step_row_strided",
    "four_step_col_strided",
    "four_step_r2c_col",
    "four_step_c2r_col",
]


def lane_block_for(lanes: int) -> int:
    if lanes <= 1:
        return 1
    value = 1
    while value < lanes:
        value <<= 1
    return value


def cooperative_stage_lanes_for(plan: LeafPlan) -> tuple[int, ...]:
    fixed_lanes_are_compatible = all(
        (plan.length // radix) % plan.lanes == 0 for radix in plan.factors
    )
    if not (
        _COOPERATIVE_STAGE_MIN_LENGTH <= plan.length <= _COOPERATIVE_STAGE_MAX_LENGTH
        and len(plan.factors) >= 2
        and (
            plan.lanes <= _COOPERATIVE_STAGE_MAX_BASE_LANES
            or not fixed_lanes_are_compatible
        )
    ):
        return (plan.lanes,) * len(plan.factors)

    stage_lanes: list[int] = []
    for radix in plan.factors:
        codelets = plan.length // radix
        upper = min(codelets, _COOPERATIVE_STAGE_MAX_LANES)
        stage_lanes.append(
            next(
                candidate
                for candidate in range(upper, 0, -1)
                if codelets % candidate == 0
            )
        )
    if (
        not _is_double_dtype(plan.dtype)
        and fixed_lanes_are_compatible
        and min(stage_lanes) * 4 < max(stage_lanes)
    ):
        return (plan.lanes,) * len(plan.factors)
    return tuple(stage_lanes)


def _floor_power_of_two(value: int) -> int:
    power = 1
    while power * 2 <= value:
        power *= 2
    return power


def _next_power_of_two(value: int) -> int:
    result = 1
    while result < value:
        result <<= 1
    return result


def contiguous_batch_pack_for(plan: LeafPlan) -> int:
    lane_block = lane_block_for(plan.lanes)
    if lane_block >= _LEAF_PACK_TARGET_THREADS:
        return 1

    thread_pack = max(1, _LEAF_PACK_TARGET_THREADS // lane_block)
    tiny_single_stage = plan.length <= 8 and len(plan.factors) == 1
    if not tiny_single_stage and plan.length <= 128:
        thread_pack = min(thread_pack, 4)
    if len(plan.factors) <= 1:
        return thread_pack

    bytes_per_fft = 4 * (plan.smem_size + 1) * _real_element_bytes(plan.dtype)
    smem_pack = max(1, _LEAF_PACK_SMEM_BUDGET_BYTES // bytes_per_fft)
    return _floor_power_of_two(max(1, min(thread_pack, smem_pack)))


def _four_step_resource_inner_pack_for(plan: LeafPlan) -> int:
    if len(plan.factors) <= 1:
        return 1

    stage_lanes = cooperative_stage_lanes_for(plan)
    if not _is_double_dtype(plan.dtype) and all(
        lanes == plan.lanes for lanes in stage_lanes
    ):
        return 1
    active_lanes = max(stage_lanes, default=plan.lanes)
    lane_block = lane_block_for(active_lanes)
    thread_pack = max(1, _FOUR_STEP_PACK_TARGET_THREADS // lane_block)
    bytes_per_fft = 4 * plan.smem_size * _real_element_bytes(plan.dtype)
    smem_pack = max(1, _FOUR_STEP_PACK_SMEM_BUDGET_BYTES // bytes_per_fft)
    return _floor_power_of_two(
        max(1, min(_FOUR_STEP_LARGE_INNER_PACK, thread_pack, smem_pack))
    )


def four_step_col_inner_pack_for(
    n1: int,
    n2: int,
    dtype: str = "complex64",
    plan: LeafPlan | None = None,
) -> int:
    if n1 < _FOUR_STEP_COL_INNER_PACK_MIN_N1:
        return 1
    if use_tle_fused_twiddle(n1, n2, dtype):
        return _FOUR_STEP_LARGE_INNER_PACK
    if plan is not None and (
        _is_double_dtype(dtype) or n2 > _FOUR_STEP_PACKED_COL_LEAF_MAX_N2
    ):
        return _four_step_resource_inner_pack_for(plan)
    if n2 > _FOUR_STEP_PACKED_COL_LEAF_MAX_N2:
        return 1
    return _FOUR_STEP_COL_INNER_PACK


def four_step_row_inner_pack_for(
    n1: int,
    n2: int,
    dtype: str = "complex64",
    plan: LeafPlan | None = None,
) -> int:
    if use_tle_fused_twiddle(n1, n2, dtype):
        return _FOUR_STEP_LARGE_INNER_PACK
    if (
        not _is_double_dtype(dtype)
        and n1 <= _FOUR_STEP_ROW_INNER_PACK_MAX_N1
        and n2 <= _FOUR_STEP_PACKED_COL_LEAF_MAX_N2
    ):
        # Mirror the column kernel: pack two inner columns per program so a
        # 16-lane row leaf occupies a full warp instead of half of one.
        return _FOUR_STEP_COL_INNER_PACK
    if plan is not None and (
        _is_double_dtype(dtype)
        or (
            n1 <= _FOUR_STEP_ROW_INNER_PACK_MAX_N1
            and n2 > _FOUR_STEP_PACKED_COL_LEAF_MAX_N2
        )
    ):
        return _four_step_resource_inner_pack_for(plan)
    if (
        not _is_double_dtype(dtype)
        and n1 <= _FOUR_STEP_ROW_INNER_PACK_MAX_N1
        and n2 > _FOUR_STEP_PACKED_COL_LEAF_MAX_N2
    ):
        return _FOUR_STEP_LARGE_INNER_PACK
    return 1


def use_tle_fused_twiddle(n1: int, n2: int, dtype: str = "complex64") -> bool:
    """Move large FP32 Four-Step twiddles into the row pass.

    Both leaves are capped at 1024 so pack=4 stays within the A100 dynamic
    shared-memory budget for the generated mixed-radix kernels.

    On the MThreads backend the fused-twiddle row kernel exceeds the MTGPU
    LLVM register allocator, so it is disabled there (falls back to the
    precomputed twiddle table path). The PPU toolchain also cannot lower
    the sin/cos.approx PTX inline asm used by the fused-twiddle path.
    """
    if _non_nvidia_backend_active():
        return False
    return (
        not _is_double_dtype(dtype)
        and n1 * n2 >= _TLE_FUSED_TWIDDLE_MIN_LENGTH
        and n1 <= _TLE_FUSED_TWIDDLE_MAX_LEAF
        and n2 <= _TLE_FUSED_TWIDDLE_MAX_LEAF
    )


def use_four_step_row_fused_twiddle(n1: int, n2: int, dtype: str = "complex64") -> bool:
    """Apply the outer twiddle while the row output is still contiguous.

    FP32 uses the existing approximate-TLE path. FP64 keeps full precision by
    loading the precomputed twiddle table in the row pass instead of issuing
    the same reads with the strided column access pattern.
    """
    return use_tle_fused_twiddle(n1, n2, dtype) or (
        _is_double_dtype(dtype) and n1 * n2 >= _TLE_FUSED_TWIDDLE_MIN_LENGTH
    )


def _use_single_smem_buffer(
    plan: LeafPlan,
    *,
    io_mode: LeafIoMode,
    four_step_n1: int,
    four_step_n2: int,
) -> bool:
    """Reuse one shared buffer between generated mixed-radix stages."""
    return (
        io_mode.startswith("four_step_")
        and not io_mode.endswith("_strided")
        and use_tle_fused_twiddle(four_step_n1, four_step_n2, plan.dtype)
        and plan.dtype == "complex64"
        and plan.length == 1024
        and len(plan.factors) > 2
    )

def _triton_plugin_present(plugin: str) -> bool:
    try:
        from triton._C import libtriton

        return hasattr(libtriton, plugin)
    except ImportError:
        return False


def _mthreads_backend_active() -> bool:
    """Whether the installed Triton targets Moore Threads (MUSA/mtgpu)."""
    return _triton_plugin_present("mthreads")


def _ppu_backend_active() -> bool:
    """Whether the installed Triton targets the T-Head PPU (XuanTie GPU).

    The thread-local mixed-radix four-step kernels and the vectorized 3D
    transpose variants rely on PTX inline-asm register patterns that the
    PPU compiler toolchain does not support, so they are disabled there.
    """
    return _triton_plugin_present("ppu")


def _non_nvidia_backend_active() -> bool:
    """Whether the installed Triton is a non-NVIDIA port (MThreads/PPU).

    The thread-local mixed-radix four-step kernels and the vectorized 3D
    transpose variants rely on register/asm patterns that the MThreads
    MTGPU LLVM backend cannot compile (llc register allocation failure)
    and that the PPU toolchain does not support, so they are disabled on
    these backends.
    """
    return _mthreads_backend_active() or _ppu_backend_active()


__all__ = [
    "LeafIoMode",
    "LeafPlan",
    "_CODELET_DIR",
    "_COOPERATIVE_STAGE_MAX_BASE_LANES",
    "_COOPERATIVE_STAGE_MAX_LANES",
    "_COOPERATIVE_STAGE_MAX_LENGTH",
    "_COOPERATIVE_STAGE_MIN_LENGTH",
    "_FOUR_STEP_COL_INNER_PACK",
    "_FOUR_STEP_COL_INNER_PACK_MIN_N1",
    "_FOUR_STEP_LARGE_INNER_PACK",
    "_FOUR_STEP_NUM_WARPS",
    "_FOUR_STEP_PACKED_COL_LEAF_MAX_N2",
    "_FOUR_STEP_PACK_SMEM_BUDGET_BYTES",
    "_FOUR_STEP_PACK_TARGET_THREADS",
    "_FOUR_STEP_ROW_INNER_PACK_MAX_N1",
    "_FOUR_STEP_TILE_COLS",
    "_FOUR_STEP_TILE_ROWS",
    "_LEAF_PACK_SMEM_BUDGET_BYTES",
    "_LEAF_PACK_TARGET_THREADS",
    "_MODULE_DIR",
    "_NATURAL_ORDER_CODELET_RADICES",
    "_PROJECT_ROOT",
    "_THREAD_LOCAL_MIXED_RADICES",
    "_TLE_FUSED_TWIDDLE_MAX_LEAF",
    "_TLE_FUSED_TWIDDLE_MIN_LENGTH",
    "_TLE_SMEM_SWIZZLE_SHIFT",
    "_dtype_suffix",
    "_floor_power_of_two",
    "_four_step_resource_inner_pack_for",
    "_is_double_dtype",
    "_mthreads_backend_active",
    "_next_power_of_two",
    "_non_nvidia_backend_active",
    "_ppu_backend_active",
    "_real_element_bytes",
    "_tl_real_dtype",
    "_triton_plugin_present",
    "_use_single_smem_buffer",
    "_zero_other",
    "codelet_radices_for",
    "contiguous_batch_pack_for",
    "cooperative_stage_lanes_for",
    "four_step_col_inner_pack_for",
    "four_step_row_inner_pack_for",
    "lane_block_for",
    "use_four_step_row_fused_twiddle",
    "use_tle_fused_twiddle",
]
