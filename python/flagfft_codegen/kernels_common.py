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

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

from .backend_profile import current_profile
from .target import (
    ix_ct_single_default_enabled,
    ix_ct_batch_default_enabled,
    ix_large_fourstep_default_enabled,
    ix_ct_single_tle_default,
    ix_real_single_pack_enabled,
    ix_prime_3d_enabled,
    ix_fourstep_preset,
    maca_1d_single_default_enabled,
    maca_1d_batch_default_enabled,
    musa_1d_batch_default_enabled,
    maca_2d_single_default_enabled,
    maca_3d_default_enabled,
    maca_3d_c2c32_single_cube_enabled,
)
from .maca_tail_policy import resource_default

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
# MACA's C550 runtime rejects four-step kernels above 64 KiB of dynamic
# shared memory even though the device properties advertise a 128 KiB opt-in
# value. This is a hard launch limit for the portable exchange path.
_MACA_FOUR_STEP_PACK_SMEM_BUDGET_BYTES = 64 * 1024
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
# The MetaX plugin cannot lower maca.shfl.sync, so the portable exchange must
# gather from a tensor wider than one 64-thread warp.
_PORTABLE_EXCHANGE_MIN_ELEMENTS = 128
# Hard ceiling for the exchange pack.  The collapse this used to encode at
# eight was measured while two sweeps shared the card; re-measured cleanly,
# eight is the best configuration on the 209/221-point leaves (2.567 -> 0.941
# ms at n=46189, correct).  Sixteen is where the generated index overflows.
_PORTABLE_EXCHANGE_MAX_PACK = 8


def _portable_exchange_max_pack() -> int:
    """Ceiling for the exchange pack, overridable for measurement.

    The shipped default reproduces ``_PORTABLE_EXCHANGE_MAX_PACK``; the knob
    exists because that ceiling was measured under a sweep that ran
    concurrently with another one on the same card, so the pack-eight collapse
    it used to encode needed re-checking before being treated as a hardware
    limit.
    """
    override = _maca_knob("MAX_PACK")
    if not override:
        return _PORTABLE_EXCHANGE_MAX_PACK
    return _positive_knob("MAX_PACK", override)


def _cooperative_warp_cap() -> int:
    """Ceiling on the cooperative warp count for a leaf block.

    Eight is both the measured optimum and the hard limit: launching sixteen
    warps fails with ``out of resource: threads, Required: 1024, Hardware
    limit: 512``.  The device profile's ``max_threads_per_block`` of 1024 is a
    static default -- the driver does not expose the attribute, so nothing
    queries it -- and Triton's launcher enforces 512 for this target.
    """
    override = _maca_knob("MAX_WARPS")
    if not override:
        return 8
    return _positive_knob("MAX_WARPS", override)


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


@lru_cache(maxsize=None)
def _codelet_dependencies() -> dict[int, frozenset[int]]:
    """Map each bundled radix codelet to the other codelets it calls."""
    deps: dict[int, frozenset[int]] = {}
    for codelet_path in _CODELET_DIR.glob("radix*.py"):
        text = codelet_path.read_text()
        match = re.search(r"def _fwd_rad(\d+)_b1", text)
        if match is None:
            continue
        radix = int(match.group(1))
        called = {
            int(dep)
            for dep in re.findall(r"_fwd_rad(\d+)_b1", text)
            if int(dep) != radix
        }
        deps[radix] = frozenset(called)
    return deps


def codelet_radices_for(factors: tuple[int, ...]) -> set[int]:
    """Return the bundled ``codelet/radixN.py`` files a leaf needs.

    Radix-16 is emitted through the special factorized call; radix-32 is
    built from two radix-16 codelets; thread-local mixed radices expand to
    their factorized codelets. Composite codelets may call further codelets
    (e.g. radix-10 calls radix-5/radix-2), so the result is transitively
    closed over the bundled codelet dependency graph.
    """
    needed: set[int] = set()
    for radix in factors:
        if radix in _NATURAL_ORDER_CODELET_RADICES:
            needed.add(radix)
        elif radix in (16, 32):
            needed.add(16)
        elif radix in _THREAD_LOCAL_MIXED_RADICES:
            needed.update(_THREAD_LOCAL_MIXED_SPLITS[radix])
    dependencies = _codelet_dependencies()
    stack = list(needed)
    while stack:
        radix = stack.pop()
        for dependency in dependencies.get(radix, ()):
            if dependency not in needed:
                needed.add(dependency)
                stack.append(dependency)
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
    "permuted_store",
    "strided_permuted_store",
    "permuted_r2c",
    "contiguous_r2c",
    "permuted_r2c",
    "packed_r2c",
    "packed_c2r",
    "contiguous_c2r",
    "bluestein_prepare_leaf",
    "bluestein_finish_leaf",
    "bluestein_full_leaf",
    "bluestein_full_leaf_r2c",
    "bluestein_full_leaf_c2r",
    "rader_full_leaf",
    "rader_prepare_leaf",
    "rader_finish_leaf",
    "bluestein_four_step_prepare_row",
    "bluestein_four_step_prepare_row_r2c",
    "bluestein_four_step_prepare_row_c2r",
    "bluestein_four_step_pointwise_row",
    "bluestein_four_step_finish_col",
    "bluestein_four_step_finish_col_r2c",
    "bluestein_four_step_finish_col_c2r",
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


def emitted_leaf_factors(
    plan: LeafPlan, io_mode: str = "contiguous"
) -> tuple[int, ...]:
    if (
        _portable_leaf_backend_active()
        and plan.length == 32
        and plan.dtype == "complex128"
        and plan.factors == (4, 4, 2)
    ):
        return plan.factors
    if (
        _portable_leaf_backend_active()
        and io_mode not in {"bluestein_full_leaf", "rader_full_leaf"}
        and plan.length
        in _NATURAL_ORDER_CODELET_RADICES | _THREAD_LOCAL_MIXED_RADICES | {16}
    ):
        # Keep small FFTs in registers: this SDK cannot parse the plugin's
        # maca.shfl.sync op generated for small tensor layout conversions.
        return (plan.length,)
    return plan.factors


def cooperative_stage_lanes_for(plan: LeafPlan) -> tuple[int, ...]:
    if _portable_leaf_backend_active():
        # The portable exchange evaluates every butterfly in one tensor.
        return tuple(plan.length // radix for radix in plan.factors)
    fixed_lanes_are_compatible = all(
        (plan.length // radix) % plan.lanes == 0 for radix in plan.factors
    )
    wide_cuda = (current_profile().backend == "cuda" and current_profile().device_arch == "sm_80"
                 and os.getenv("FLAGFFT_CUDA_WIDE_STAGE") == "1")
    wide_hcu_convolution = (_hcu_backend_active() and
        os.getenv("FLAGFFT_HCU_CONV_SINGLE_BUFFER", "0") == "1" and
        plan.lanes > _COOPERATIVE_STAGE_MAX_LANES and not fixed_lanes_are_compatible)
    max_lanes = plan.lanes if wide_hcu_convolution else 256 if wide_cuda else _COOPERATIVE_STAGE_MAX_LANES
    min_length = _COOPERATIVE_STAGE_MIN_LENGTH
    short_cooperative = (current_profile().backend == "cuda" and plan.length < min_length and
                         os.getenv("FLAGFFT_CUDA_SHORT_COOPERATIVE") == "1")
    if short_cooperative:
        min_length = 32
    if not (
        min_length <= plan.length <= _COOPERATIVE_STAGE_MAX_LENGTH
        and len(plan.factors) >= 2
        and (
            wide_cuda or plan.lanes <= _COOPERATIVE_STAGE_MAX_BASE_LANES
            or not fixed_lanes_are_compatible
        )
    ):
        return (plan.lanes,) * len(plan.factors)

    stage_lanes: list[int] = []
    for radix in plan.factors:
        codelets = plan.length // radix
        upper = min(codelets, max_lanes)
        stage_lanes.append(
            next(
                candidate
                for candidate in range(upper, 0, -1)
                if codelets % candidate == 0
            )
        )
    if (
        not _is_double_dtype(plan.dtype)
        and not short_cooperative
        and fixed_lanes_are_compatible
        and min(stage_lanes) * 4 < max(stage_lanes)
    ):
        return (plan.lanes,) * len(plan.factors)
    return tuple(stage_lanes)


def _ix_large_fourstep_default_for(kernel: str, n1: int, n2: int, dtype: str) -> bool:
    """Qualified register tiles for the large, dense IX power-of-two pair."""
    from .registry import INNER_PACK_ROW_KERNELS, INNER_PACK_COL_KERNELS, STRIDED_FOUR_STEP_KERNELS
    profile = current_profile()
    if (profile.backend != "ix" or profile.device_arch != "71" or profile.policy != "balanced"
            or (profile.max_dynamic_shared_memory or 0) < 131072 or dtype != "complex64"
            or kernel not in (INNER_PACK_ROW_KERNELS | INNER_PACK_COL_KERNELS) - STRIDED_FOUR_STEP_KERNELS
            or not all(512 <= n <= 1024 and n & (n-1) == 0 for n in (n1,n2))):
        return False
    setting = os.getenv("FLAGFFT_IX_LARGE_FOURSTEP", "1")
    if setting not in {"0","1"}:
        raise ValueError("FLAGFFT_IX_LARGE_FOURSTEP must be 0 or 1")
    return setting == "1"


def _maca_knob(name: str, default: str = "") -> str:
    """Read a MACA code-generation override.

    The native compiler scopes measured 1D single, 1D batch, 2D single and
    3D policies. The 2D scope also covers batched row/column kernels. These
    supply defaults while preserving an explicit environment override for
    A/B testing and rollback.
    Direct Python code-generation calls remain on the historical defaults.
    """
    if _ix_backend_active():
        defaults = {"EXCHANGE": "direct_all", "SPLIT_ORDER": "lsb"}
        if ix_ct_single_default_enabled():
            defaults.update(PORTABLE_LEAF="1", RECURRENCE="1", WARPS="2", INNER_PACK="4", LANE_MIN="1")
        if ix_ct_batch_default_enabled():
            defaults.update(PORTABLE_LEAF="1", RECURRENCE="1", WARPS="4", LANE_MIN="1")
        if ix_real_single_pack_enabled() and current_profile().policy=="balanced":
            defaults.update(VEC_IO="packed", STATIC_SINGLE="1")
        if ix_prime_3d_enabled() and current_profile().policy=="balanced":
            defaults.update(WARP_FILL_PACK="1", BATCH_INTERLEAVE="1", BLUESTEIN_SINGLE_SMEM="1")
        if ix_ct_single_tle_default():
            defaults.update(SMEM_INTERLEAVE="1", RECURRENCE="1")
        if ix_ct_single_tle_default() == 2:
            defaults.update(SMEM_SWIZZLE="1", SMEM_SWIZZLE_SHIFT="5", TLE_INNER_PACK="8", WARPS="4")
        if ix_large_fourstep_default_enabled():
            defaults.update(PORTABLE_LEAF="1", RECURRENCE="1", INNER_PACK="16",
                            MAX_PACK="16", RESOURCE_WARPS="1", VEC_IO="packed")
        preset = ix_fourstep_preset()
        if preset==9 and current_profile().policy=="balanced":
            defaults.update(VEC_IO="packed", PACKED_R2C_INPUT="1")
        elif preset==12 and current_profile().policy=="balanced":
            defaults.update(VEC_IO="packed", RECURRENCE="1")
        elif preset==13 and current_profile().policy=="balanced":
            defaults.update(VEC_IO="packed", RECURRENCE="1", PORTABLE_LEAF="0",
                            WARPS="16", BATCH_PACK="16", TLE_DISJOINT_STAGE="1",
                            TLE_REAL_DISJOINT="1")
        elif preset==10 and current_profile().policy=="balanced":
            defaults.update(VEC_IO="packed", RECURRENCE="1", SMEM_INTERLEAVE="1",
                            TLE_INNER_PACK="8")
            if ix_ct_single_tle_default() == 1:
                defaults.update(TLE_DISJOINT_STAGE="1", TLE_DISJOINT_MIXED="1")
        elif preset==11 and current_profile().policy=="balanced":
            defaults.update(VEC_IO="packed", PORTABLE_LEAF="1", RECURRENCE="1",
                            WARPS="2", BATCH_PACK="1")
        elif preset and current_profile().policy == "balanced":
            defaults.update(RECURRENCE="1", VEC_IO="packed")
            if preset >= 7:
                defaults.update(PORTABLE_LEAF="1", WARPS="16" if preset==7 else "8",
                                BATCH_PACK="1")
            else:
                defaults.update(PORTABLE_LEAF="0", SMEM_INTERLEAVE="1", TLE_DISJOINT_STAGE="1")
            if preset == 4:
                defaults.update(WARPS="16", BATCH_PACK="1", SMEM_SWIZZLE="1",
                                SMEM_SWIZZLE_SHIFT="5", TLE_DISJOINT_CTA="1")
            elif preset <= 3:
                defaults.update(WARPS="4", TLE_INNER_PACK="32" if preset in {1,4} else "16",
                                TLE_IO_LAYOUT="0" if preset==2 else "1")
            elif preset <= 6:
                defaults.update(TLE_INNER_PACK="4", TLE_DISJOINT_MIXED="1",
                                RADER_CODELET="0" if preset==5 else "1")
                if preset==5:
                    defaults.update(REAL_BOUNDARY_BLOCK="256")
        return os.environ.get(f"FLAGFFT_IX_{name}", defaults.get(name, default)).strip().lower()
    if _npu_backend_active() and os.environ.get("FLAGFFT_NPU_FOURSTEP_LEAF") == "1":
        npu_env_name = {
            "EXCHANGE": "FLAGFFT_NPU_2D_EXCHANGE",
            "LANE_MIN": "FLAGFFT_NPU_2D_LANE_MIN",
            "BATCH_PACK": "FLAGFFT_NPU_2D_BATCH_PACK",
            "SPLIT_ORDER": "FLAGFFT_NPU_2D_SPLIT_ORDER",
        }.get(name)
        if npu_env_name is not None and npu_env_name in os.environ:
            return os.environ[npu_env_name].strip().lower()
    env_name = f"FLAGFFT_MACA_{name}"
    if env_name in os.environ:
        return os.environ[env_name].strip().lower()
    tail_default = resource_default(name)
    if tail_default is not None:
        return tail_default
    if maca_3d_default_enabled():
        if name == "EXCHANGE":
            return "direct_all"
        if name == "VEC_IO":
            return "packed"
    if maca_2d_single_default_enabled() and name == "2D_TRANSPOSE":
        return "packed"
    if maca_1d_batch_default_enabled():
        if name == "EXCHANGE":
            return "direct_all"
        if name == "FP64_REGISTER_PACK":
            return "1"
        if name == "INNER_PACK":
            # The resource guard in _maca_four_step_inner_pack keeps short and
            # mixed leaves at their measured pack while allowing P8 for the
            # FP32 512/1024-point power-of-two pair used by N=524288.
            return "8"
    if maca_1d_single_default_enabled() or maca_2d_single_default_enabled():
        defaults = {
            "EXCHANGE": "direct_all",
            "INNER_PACK": "8",
            "MAX_WARPS": "8",
            "SPLIT_ORDER": "lsb",
            "VEC_IO": "0",
        }
        if name in defaults:
            return defaults[name]
    return default.strip().lower()


def _positive_knob(name: str, value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise ValueError(
            f"FLAGFFT_MACA_{name} must be an integer, got {value!r}"
        ) from None
    if parsed < 1:
        raise ValueError(f"FLAGFFT_MACA_{name} must be positive, got {parsed}")
    return parsed


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


def _register_bounded_batch_pack(plan: LeafPlan, pack: int, native_pack: int) -> int:
    profile = current_profile()
    if profile.policy != "balanced":
        return pack
    lanes = lane_block_for(max(cooperative_stage_lanes_for(plan), default=plan.lanes))
    native_warps = max(
        profile.planner_warps(plan.num_warps), profile.warps_for(lanes * native_pack)
    )
    while pack > native_pack:
        warps = max(
            profile.planner_warps(plan.num_warps), profile.warps_for(lanes * pack)
        )
        live_bytes = plan.length * pack * 2 * _real_element_bytes(plan.dtype)
        budget = warps * profile.warp_size * profile.leaf_live_bytes_per_thread
        # Fill spare lanes, but do not add physical warps simply to pack more
        # transforms: that can introduce cross-warp exchanges and barriers.
        if warps <= native_warps and live_bytes <= budget:
            break
        pack //= 2
    return pack


def _profile_batch_pack_for(plan: LeafPlan) -> int:
    profile = current_profile()
    lane_block = lane_block_for(plan.lanes)
    target_threads = profile.leaf_target_threads
    buffers = 4
    if profile.backend == "hcu" and profile.device_arch == "gfx936":
        packing = os.getenv("FLAGFFT_HCU_LEAF_PACKING", "resource")
        if packing not in {"resource", "wave", "two_waves", "four_waves"}:
            raise ValueError("FLAGFFT_HCU_LEAF_PACKING must be resource, wave, two_waves or four_waves")
        if packing != "resource" and len(plan.factors) > 1:
            # Count the lanes actually participating in the emitted stages.
            # Fill a physical wave before adding blocks for otherwise idle
            # lanes; retain the shared-memory bound below for both precisions.
            lane_block = lane_block_for(max(cooperative_stage_lanes_for(plan), default=plan.lanes))
            target_threads = profile.warp_size * {"wave": 1, "two_waves": 2, "four_waves": 4}[packing]
            if os.getenv("FLAGFFT_HCU_LEAF_SINGLE_BUFFER", "0") == "1" and _hcu_buffer_stage_safe(plan):
                buffers = 2

    def pack_for(target_threads):
        thread_pack = max(1, target_threads // lane_block)
        tiny_single_stage = plan.length <= 8 and len(plan.factors) == 1
        if not tiny_single_stage and plan.length <= 128:
            thread_pack = min(thread_pack, 4)
        if len(plan.factors) <= 1:
            return thread_pack
        bytes_per_fft = buffers * (plan.smem_size + 1) * _real_element_bytes(plan.dtype)
        smem_pack = max(
            1, profile.shared_budget(_LEAF_PACK_SMEM_BUDGET_BYTES) // bytes_per_fft
        )
        return _floor_power_of_two(max(1, min(thread_pack, smem_pack)))

    return _register_bounded_batch_pack(
        plan, pack_for(target_threads), pack_for(32)
    )


def _portable_exchange_pack_floor(plan: LeafPlan, pack: int) -> int:
    """Raise the pack so the portable exchange tensor still spans a warp.

    ``vector_block = lane_block * pack`` is what the gather reads, so packing
    widens it without idling lanes.  Leaving the pack at one forces the lane
    block back up to 128, which is what starved the contiguous leaves behind
    2D, 3D and the small batch shapes.
    """
    if _maca_knob("LANE_MIN", "auto") != "auto":
        return pack
    active_lanes = max(cooperative_stage_lanes_for(plan), default=plan.lanes)
    needed = max(1, _PORTABLE_EXCHANGE_MIN_ELEMENTS // lane_block_for(active_lanes))
    return max(pack, min(_portable_exchange_max_pack(), _next_power_of_two(needed)))


def contiguous_batch_pack_for(
    plan: LeafPlan,
    *,
    real_boundary: bool = False,
    real_transform: str | None = None,
) -> int:
    if _ix_backend_active() and not _portable_leaf_backend_active() and _maca_knob("BATCH_PACK"):
        pack = _positive_knob("BATCH_PACK", _maca_knob("BATCH_PACK"))
        limit = current_profile().max_dynamic_shared_memory
        if limit is not None and len(plan.factors) > 1:
            while pack > 1 and 4 * lane_block_for(
                (plan.smem_size + (1 if pack >= 4 else 0)) * pack
            ) * _real_element_bytes(plan.dtype) > limit:
                pack //= 2
        return pack
    if _hcu_backend_active() and plan.length == 2048:
        middle_batch_pack = os.environ.get("FLAGFFT_HCU_3D_MIDDLE_BATCH_PACK", "auto")
        if middle_batch_pack != "auto":
            if middle_batch_pack not in {"1", "2", "4", "8", "16", "32"}:
                raise ValueError(
                    "FLAGFFT_HCU_3D_MIDDLE_BATCH_PACK must be auto, 1, 2, 4, 8, 16 or 32"
                )
            return int(middle_batch_pack)
    if real_boundary and plan.length == 210 and ix_real_single_pack_enabled():
        return 1
    if (_mthreads_backend_active() and musa_1d_batch_default_enabled()
            and plan.length == 1024):
        override = os.getenv("FLAGFFT_MUSA_1D_BATCH_PACK", "auto")
        if override not in {"auto", "1", "2", "4", "8"}:
            raise ValueError("FLAGFFT_MUSA_1D_BATCH_PACK must be auto, 1, 2, 4 or 8")
        # Measured on MTT S5000 at N=1024: pack four for FP64 C2C/C2R, and
        # pack two for FP64 R2C. FP32 keeps pack one; explicit A/B overrides
        # remain available for all of these variants.
        if override == "auto":
            requested_pack = 1
            if plan.dtype == "complex128":
                if not real_boundary or real_transform == "c2r":
                    requested_pack = 4
                elif real_transform == "r2c":
                    requested_pack = 2
        else:
            requested_pack = int(override)
        profile = current_profile()
        resource_limit = profile.max_dynamic_shared_memory or _LEAF_PACK_SMEM_BUDGET_BYTES
        bytes_per_fft = 4 * (plan.smem_size + 1) * _real_element_bytes(plan.dtype)
        resource_pack = _floor_power_of_two(max(1, resource_limit // bytes_per_fft))
        return min(requested_pack, resource_pack)
    if _portable_leaf_backend_active():
        override = _maca_knob("BATCH_PACK")
        if override == "auto":
            return _profile_batch_pack_for(plan)
        if override:
            return _positive_knob("BATCH_PACK", override)
        if plan.length == 64 and len(emitted_leaf_factors(plan)) > 1:
            short_pack = os.getenv("FLAGFFT_MACA_3D_N64_PACK")
            if short_pack is not None:
                if short_pack not in {"1", "2", "4", "8", "16", "32"}:
                    raise ValueError("FLAGFFT_MACA_3D_N64_PACK must be 1, 2, 4, 8, 16 or 32")
                return int(short_pack)
        if maca_3d_default_enabled() and len(emitted_leaf_factors(plan)) > 1:
            if plan.length == 64 and plan.dtype == "complex64":
                return 8
            if plan.length == 64 and plan.dtype == "complex128":
                return 8
            if (plan.length == 32 and plan.dtype == "complex128"
                    and maca_3d_c2c32_single_cube_enabled()):
                # This shape-specific policy was measured on the outer
                # strided leaf of a single 32^3 FP64 C2C transform. Batched
                # length-32 transforms keep the normal rank-3 pack.
                pack_32_fp64 = os.getenv("FLAGFFT_MACA_3D_N32_FP64_PACK", "8")
                if pack_32_fp64 not in {"1", "2", "4", "8"}:
                    raise ValueError(
                        "FLAGFFT_MACA_3D_N32_FP64_PACK must be 1, 2, 4 or 8"
                    )
                return int(pack_32_fp64)
            if plan.length == 128:
                pack_128 = os.getenv("FLAGFFT_MACA_3D_N128_PACK", "2")
                if pack_128 not in {"1", "2", "4"}:
                    raise ValueError("FLAGFFT_MACA_3D_N128_PACK must be 1, 2 or 4")
                return int(pack_128)
            # On C550, the 256-point FP32 3D leaf runs with a 128-thread
            # block.  Packing two transforms fills that block and halves the
            # CTA count; a paired GPU measurement improved 256^3 C2C by 10%.
            if plan.length == 256 and plan.dtype == "complex64":
                return 2
            if plan.length == 256 and plan.dtype == "complex128":
                return 2
        if maca_1d_batch_default_enabled() and plan.length == 16:
            return 1
        # Native 2D only emits these fused real boundary leaves when n0 > 256.
        # Group short rows to avoid tens of thousands of underfilled blocks.
        # Small 2D RC plans use ordinary complex leaves and stay at pack=1;
        # degenerate unit-axis plans are outside the native 2D policy scope.
        if (
            maca_2d_single_default_enabled()
            and real_boundary
            and plan.dtype == "complex64"
            and 16 <= plan.length <= 128
        ):
            return 4
        lane_block = lane_block_for(max(cooperative_stage_lanes_for(plan), default=1))
        if len(emitted_leaf_factors(plan)) > 1:
            return 1
        # Raising this pack was measured slower on 2D 64x64: the row pass only
        # has 64 transforms, so packing eight leaves the grid too small to fill
        # the device, and the extra occupancy does not pay for it.
        return max(
            1,
            min(32 if len(plan.factors) == 1 else 4, 64 // lane_block),
        )
    return _profile_batch_pack_for(plan)


def _ix_permuted_single_smem(plan: LeafPlan) -> bool:
    """Explicit wide transpose-store trial with complete overwriting stages."""
    if (not _ix_backend_active() or _portable_leaf_backend_active()
            or os.getenv("FLAGFFT_IX_PERMUTED_SINGLE_SMEM", "0") != "1"
            or os.getenv("FLAGFFT_IX_BATCH_INTERLEAVE", "0") != "1"
            or plan.dtype != "complex64" or not 64 <= plan.length <= 4096
            or plan.length & (plan.length - 1) or len(plan.factors) < 2):
        return False
    lanes = cooperative_stage_lanes_for(plan)
    return all(plan.length == lanes[i] * plan.factors[i]
               for i in range(len(plan.factors) - 1))


def _hcu_wide_permuted_store(plan: LeafPlan) -> bool:
    if (not _hcu_backend_active()
            or os.getenv("FLAGFFT_HCU_3D_STORE_MODE", "auto") not in {"wide", "cyclic"}
            or plan.length <= 0 or plan.length & (plan.length - 1)
            or len(plan.factors) < 2):
        return False
    lanes = cooperative_stage_lanes_for(plan)
    # Stage zero reads global memory, and the last stage writes global
    # memory. Only intermediate stages overwrite their own shared input.
    return all(plan.length == lanes[i] * plan.factors[i]
               for i in range(1, len(plan.factors) - 1))


def _hcu_buffer_stage_safe(plan: LeafPlan | None) -> bool:
    if (plan is None or len(plan.factors) < 2 or plan.length <= 0
            or plan.length & (plan.length - 1)):
        return False
    lanes = cooperative_stage_lanes_for(plan)
    return all(plan.length == lanes[i] * plan.factors[i]
               for i in range(1, len(plan.factors) - 1))


def hcu_four_step_single_buffer(plan: LeafPlan | None) -> bool:
    """An overwriting stage must read the complete FFT before its barrier."""
    if not _hcu_backend_active():
        return False
    setting = os.getenv("FLAGFFT_HCU_FOURSTEP_SINGLE_BUFFER", "0")
    if setting not in {"0", "1"}:
        raise ValueError("FLAGFFT_HCU_FOURSTEP_SINGLE_BUFFER must be 0 or 1")
    return setting == "1" and _hcu_buffer_stage_safe(plan)


def hcu_leaf_single_buffer(plan: LeafPlan, io_mode: str) -> bool:
    if not _hcu_backend_active():
        return False
    setting = os.getenv("FLAGFFT_HCU_LEAF_SINGLE_BUFFER", "0")
    if setting not in {"0", "1"}:
        raise ValueError("FLAGFFT_HCU_LEAF_SINGLE_BUFFER must be 0 or 1")
    if (setting != "1" or io_mode not in {
            "contiguous", "strided", "contiguous_r2c", "contiguous_c2r",
            "packed_r2c", "packed_c2r"} or not _hcu_buffer_stage_safe(plan)):
        return False
    if (os.getenv("FLAGFFT_HCU_PACKED_REAL_CTA", "0") == "1"
            and io_mode in {"packed_r2c", "packed_c2r"}):
        return True
    # Retain the short-leaf schedule where shared storage is inexpensive.
    # The original allocation must occupy at least one quarter of the CU's
    # queried LDS capacity before screening the extra overwrite barrier.
    pack = contiguous_batch_pack_for(plan)
    scalar_bytes = _real_element_bytes(plan.dtype)
    capacity = current_profile().max_dynamic_shared_memory or 65536
    old_stride = plan.smem_size + (1 if pack >= 4 else 0)
    old_bytes = 4 * lane_block_for(old_stride * pack) * scalar_bytes
    new_bytes = 2 * lane_block_for(plan.smem_size * pack) * scalar_bytes
    return old_bytes >= capacity // 4 and new_bytes <= capacity


def hcu_packed_cta_batch_pack_for(plan: LeafPlan, io_mode: str) -> int:
    """Fill an output transaction while retaining two LDS-resident CTAs."""
    if not _hcu_backend_active() or os.getenv("FLAGFFT_HCU_PACKED_REAL_CTA", "0") != "1":
        return 0
    if io_mode not in {"packed_r2c", "packed_c2r"}:
        return 0
    widths = cooperative_stage_lanes_for(plan)
    single = hcu_leaf_single_buffer(plan, io_mode)
    alias = single and plan.length == widths[-1] * plan.factors[-1]
    buffers = 2 if single else 4
    if io_mode == "packed_r2c" and not alias:
        buffers += 2
    scalar = _real_element_bytes(plan.dtype)
    target = 128 // (2 * scalar)
    capacity = (current_profile().max_dynamic_shared_memory or 65536) // 2
    pack = 1
    while pack * 2 <= target:
        candidate = pack * 2
        stride = plan.smem_size + (1 if candidate >= 4 and not single else 0)
        shared = buffers * lane_block_for(stride * candidate) * scalar
        if max(widths) * candidate > 512 or shared > capacity:
            break
        pack = candidate
    maximum = os.getenv("FLAGFFT_HCU_PACKED_CTA_MAX_PACK", "auto")
    if maximum != "auto":
        if maximum not in {"1", "2", "4", "8", "16", "32"}:
            raise ValueError("FLAGFFT_HCU_PACKED_CTA_MAX_PACK must be auto, 1, 2, 4, 8, 16 or 32")
        pack = min(pack, int(maximum))
    return pack


def hcu_convolution_single_buffer(plan: LeafPlan, io_mode: str) -> bool:
    if not _hcu_backend_active():
        return False
    setting = os.getenv("FLAGFFT_HCU_CONV_SINGLE_BUFFER", "0")
    if setting not in {"0", "1"}:
        raise ValueError("FLAGFFT_HCU_CONV_SINGLE_BUFFER must be 0 or 1")
    if (setting != "1" or io_mode != "bluestein_full_leaf" or len(plan.factors) < 2 or
            plan.length <= 0 or plan.length & (plan.length - 1)):
        return False
    lanes = cooperative_stage_lanes_for(plan)
    # The first FFT's final stage and the second FFT's first stage also
    # overwrite shared input. Every stage must read its entire FFT at once.
    return all(plan.length == radix * width for radix, width in zip(plan.factors, lanes))


def permuted_store_batch_pack_for(
    plan: LeafPlan, *, force_full_smem: bool = False, pack_override: int | None = None,
    permutation_span: int = 0, permutation_batch: int = 0, inner: bool = False,
) -> int:
    """Batch slots per block for the fused permuted store.

    Four is the generic FP32 default.  The validated MACA rank-3 length-64
    first-store path uses eight slots; MUSA's paired complex FP64 store uses
    two. Shared memory still caps the selected pack for large leaves.
    """
    profile = current_profile()
    if _hcu_wide_permuted_store(plan):
        # Aim for a cache-line-sized contiguous output run, bounded by the
        # actual rounded allocation for one real/imaginary shared buffer.
        target = 128 // (2 * _real_element_bytes(plan.dtype))
        fill = os.getenv("FLAGFFT_HCU_PERMUTED_FILL_WAVE", "0")
        if fill not in {"0", "1"}:
            raise ValueError("FLAGFFT_HCU_PERMUTED_FILL_WAVE must be 0 or 1")
        if fill == "1" and profile.policy != "legacy":
            active = lane_block_for(max(cooperative_stage_lanes_for(plan)))
            # A cache-line-sized output run can still leave half a physical
            # wave idle on a short wide-precision leaf. Fill those lanes
            # before adding waves; the existing LDS/thread bounds still win.
            target = max(target, profile.warp_size // active)
        setting = os.getenv("FLAGFFT_HCU_3D_PACK", "auto")
        if setting != "auto":
            if setting not in {"1", "2", "4", "8", "16", "32"}:
                raise ValueError("FLAGFFT_HCU_3D_PACK must be 1, 2, 4, 8, 16 or 32")
            target = int(setting)
        if pack_override is not None:
            target = pack_override
        if inner and permutation_span > 0 and permutation_batch > 0:
            outer_rows = permutation_batch // permutation_span
            # The inner-store grid groups rows across the permuted dimension.
            # A partial group needs a padded grid; choose a divisor instead so
            # the existing compact launch covers every row exactly once.
            while target > 1 and outer_rows % target:
                target //= 2
        thread_limit = min(profile.max_threads_per_block,
                           profile.warp_size * _cooperative_warp_cap())
        lanes = lane_block_for(max(cooperative_stage_lanes_for(plan)))
        shared_limit = profile.max_dynamic_shared_memory or 48 * 1024
        for pack in (32, 16, 8, 4, 2, 1):
            cyclic = os.getenv("FLAGFFT_HCU_3D_STORE_MODE", "auto") == "cyclic"
            padded = plan.smem_size + (1 if pack >= 4 and not cyclic else 0)
            shared = 2 * lane_block_for(padded * pack) * _real_element_bytes(plan.dtype)
            if pack <= target and lanes * pack <= thread_limit and shared <= shared_limit:
                return pack
        return 1
    single_smem = _ix_permuted_single_smem(plan)
    buffer_count = 2 if single_smem else 4
    bytes_per_fft = buffer_count * (plan.smem_size + (0 if single_smem else 1)) * _real_element_bytes(plan.dtype)
    smem_pack = max(
        1, profile.shared_budget(_LEAF_PACK_SMEM_BUDGET_BYTES) // bytes_per_fft
    )
    target_pack = 2 if _mthreads_backend_active() and _is_double_dtype(plan.dtype) else 4
    if _declared_backend() == "cuda":
        override = os.getenv("FLAGFFT_CUDA_PERMSTORE_PACK", "auto")
        if override != "auto":
            if override not in {"1", "2", "4", "8", "16", "32"}:
                raise ValueError("FLAGFFT_CUDA_PERMSTORE_PACK must be auto, 1, 2, 4, 8, 16 or 32")
            target_pack = int(override)
            padding = os.getenv("FLAGFFT_CUDA_PERMSTORE_PAD", "1")
            if padding not in {"0", "1"}:
                raise ValueError("FLAGFFT_CUDA_PERMSTORE_PAD must be 0 or 1")
            cuda_bytes_per_fft = 4 * (plan.smem_size + int(padding)) * _real_element_bytes(plan.dtype)
            smem_pack = max(1, profile.shared_budget(128 * 1024) // cuda_bytes_per_fft)
    # On C550, pack=8 (2 warps / 128 threads) measured about 1% faster than
    # pack=16 (4 warps / 256 threads) for the validated 128x2048x64 C2C
    # first-store path, across single/batch and both directions. The smaller
    # block is the likely source; hardware occupancy counters were not
    # collected. Keep this change to rank-3 FP32 length-64 stores; length 128
    # and ordinary contiguous leaves retain their existing pack choices.
    final_store_cube = (
        plan.length == 256
        and os.getenv("FLAGFFT_MACA_3D_FINAL_STORE") == "1"
    )
    if (
        _declared_backend() == "maca"
        and maca_3d_default_enabled()
        and plan.dtype == "complex64"
        and (plan.length in {64, 128} or final_store_cube)
    ):
        target_pack = 8 if plan.length == 64 else 16 if plan.length == 128 else 4
        override = os.getenv("FLAGFFT_MACA_3D_PERMSTORE_PACK")
        if override is not None and override != "auto":
            allowed = {"1", "2", "4", "8", "16", "32"}
            if final_store_cube:
                allowed = {"1", "2", "4", "8"}
            if override not in allowed:
                raise ValueError(
                    "FLAGFFT_MACA_3D_PERMSTORE_PACK must be 1, 2, 4, 8, 16, 32 or auto"
                )
            target_pack = int(override)
    if _mthreads_backend_active():
        override = os.getenv("FLAGFFT_MUSA_3D_PACK")
        if override is not None and override != "auto":
            if override not in {"1", "2", "4", "8"}:
                raise ValueError("FLAGFFT_MUSA_3D_PACK must be 1, 2, 4 or 8")
            target_pack = int(override)
    if _ix_backend_active():
        fill_pack = _maca_knob("WARP_FILL_PACK", "0")
        if fill_pack not in {"0", "1"}:
            raise ValueError("FLAGFFT_IX_WARP_FILL_PACK must be 0 or 1")
        if fill_pack == "1" and plan.dtype == "complex64" and plan.length <= 256:
            lanes = lane_block_for(max(cooperative_stage_lanes_for(plan)))
            target_pack = max(target_pack, profile.warp_size // lanes)
        override = os.getenv("FLAGFFT_IX_3D_PACK", "auto")
        if override != "auto":
            if override not in {"1", "2", "4", "8", "16", "32"}:
                raise ValueError("FLAGFFT_IX_3D_PACK must be auto, 1, 2, 4, 8, 16 or 32")
            target_pack = int(override)
            if profile.max_dynamic_shared_memory is not None:
                # Explicit IX trials may use the queried device budget.
                # Bound the rounded, padded four-buffer allocation used by
                # the source, rather than the default occupancy budget.
                smem_pack = max(
                    (pack for pack in (1, 2, 4, 8, 16, 32)
                     if buffer_count * lane_block_for(
                         (plan.smem_size + (1 if pack >= 4 and not single_smem else 0)) * pack
                     ) * _real_element_bytes(plan.dtype)
                     <= profile.max_dynamic_shared_memory),
                    default=1,
                )
    if _hcu_backend_active():
        if force_full_smem:
            target_pack = 8
        override = os.getenv("FLAGFFT_HCU_3D_PACK")
        if override is not None and override != "auto":
            if override not in {"1", "2", "4", "8", "16", "32"}:
                raise ValueError("FLAGFFT_HCU_3D_PACK must be 1, 2, 4, 8, 16 or 32")
            target_pack = int(override)
        if pack_override is not None:
            if pack_override not in {1, 2, 4, 8, 16, 32}:
                raise ValueError("HCU final-axis pack must be 1, 2, 4, 8, 16 or 32")
            target_pack = pack_override
        if ((force_full_smem or os.getenv("FLAGFFT_HCU_3D_FULL_SMEM", "0") == "1")
                and len(plan.factors) == 2):
            # The normal 48 KiB budget preserves occupancy.  For this HCU
            # two-stage experiment, admit a pack only when its exchange fits
            # the queried 64 KiB dynamic shared-memory limit.  Three-stage
            # leaves use more buffers; the ordinary budget remains in force.
            smem_limit = profile.shared_budget(64 * 1024)
            smem_pack = max(
                (pack for pack in (1, 2, 4, 8, 16, 32)
                 if lane_block_for((plan.smem_size + (1 if pack >= 4 else 0)) * pack)
                 * 2 * _real_element_bytes(plan.dtype) <= smem_limit),
                default=1,
            )
    return _floor_power_of_two(max(1, min(target_pack, smem_pack)))


def _mthreads_small_mixed_leaf(plan: LeafPlan) -> bool:
    """Pack low-lane mixed leaves without changing power-of-two FFT layouts.

    Eight transforms per CTA improve MUSA's 209/221-point four-step leaves;
    sixteen regressed in measurements. The 256-element cap keeps eight FP64
    transforms within 64 KiB of shared memory; the resource policy below
    still applies the thread and shared-memory limits.
    """
    return (
        _mthreads_backend_active()
        and 128 <= plan.length <= 256
        and plan.lanes == 1
        and len(plan.factors) > 1
    )


def _four_step_resource_inner_pack_for(plan: LeafPlan) -> int:
    if len(plan.factors) <= 1:
        return 1

    stage_lanes = cooperative_stage_lanes_for(plan)
    if not _is_double_dtype(plan.dtype) and all(
        lanes == plan.lanes for lanes in stage_lanes
    ):
        # Legacy heuristic: no cooperative staging means the lane mapping is
        # already uniform and packing was not required on 32-lane devices.
        # Profile-aware backends still benefit from filling the wider block.
        if current_profile().policy == "legacy":
            return 1
    active_lanes = max(stage_lanes, default=plan.lanes)
    lane_block = lane_block_for(active_lanes)
    thread_pack = max(1, _FOUR_STEP_PACK_TARGET_THREADS // lane_block)
    buffers = 2 if hcu_four_step_single_buffer(plan) else 4
    bytes_per_fft = buffers * plan.smem_size * _real_element_bytes(plan.dtype)
    smem_pack = max(
        1,
        current_profile().shared_budget(_FOUR_STEP_PACK_SMEM_BUDGET_BYTES)
        // bytes_per_fft,
    )
    max_pack = _FOUR_STEP_LARGE_INNER_PACK
    if _mthreads_small_mixed_leaf(plan):
        max_pack = 8
    return _floor_power_of_two(max(1, min(max_pack, thread_pack, smem_pack)))


def _maca_four_step_pack_for(plan: LeafPlan) -> int:
    """Pack that fills a 256-thread block on MetaX, bounded to [4, 8].

    Measured on the 1D ct four-step leaves at batch 256.  Eight is the best
    pack when the lane block is 32: the column kernel goes from 2.226 ms at
    85 GB/s to 0.49 ms at 390 GB/s, taking the plan from 2.567 to 0.941 ms.
    Four is best when the lane block is 64 or 128, where eight costs 1.09x and
    1.83x -- a lane block that wide already spans the block, so the extra pack
    only adds registers.  Filling 256 threads picks eight at 32 lanes and four
    at 64, and the shipped four is the floor for the leaves the target would
    push below it.
    """
    active_lanes = max(cooperative_stage_lanes_for(plan), default=plan.lanes)
    target = _FOUR_STEP_PACK_TARGET_THREADS // lane_block_for(active_lanes)
    bounded = min(
        _portable_exchange_max_pack(), max(_FOUR_STEP_LARGE_INNER_PACK, target)
    )
    return _floor_power_of_two(max(1, bounded))


def _maca_p8_register_leaf_supported(plan: LeafPlan) -> bool:
    """Whether the C550 register-routed P8 experiment is a safe choice.

    P8 removes the conservative shared-memory estimate only for the layouts
    that were measured as register-routed.  Mixed-radix joins still materialize
    padded radix tensors, and short leaves do not have enough work to amortize
    the extra register/warp pressure.  Keep both cases on the derived policy
    even when a caller requests P8 globally.
    """
    ix_short_trial = False
    if _ix_backend_active():
        # IX short FourStep axes can otherwise leave most of a four-warp
        # block idle. Screen wider register tiles separately from the MetaX
        # policy; the direct-exchange shared-memory bound still applies.
        short_pack = _maca_knob("SHORT_REGISTER_PACK", "0")
        if short_pack not in {"0", "1"}:
            raise ValueError("FLAGFFT_IX_SHORT_REGISTER_PACK must be 0 or 1")
        ix_short_trial = short_pack == "1" and 64 <= plan.length <= 256
    return (
        _maca_knob("EXCHANGE") in {"direct", "direct_all"}
        and (plan.length >= 512 or ix_short_trial)
        and not _is_double_dtype(plan.dtype)
        and len(plan.factors) > 1
        and all(radix & (radix - 1) == 0 for radix in plan.factors)
    )


def _maca_four_step_smem_pack_limit(plan: LeafPlan) -> int:
    """Return the largest inner pack that fits MACA's launchable SMEM limit."""
    profile_limit = current_profile().max_dynamic_shared_memory
    budget = _MACA_FOUR_STEP_PACK_SMEM_BUDGET_BYTES
    if profile_limit is not None:
        budget = min(budget, profile_limit)

    # The structured direct exchange for an all-power-of-two leaf stays in
    # registers.  The generic four-buffer estimate below is for the portable
    # gather/layout path and incorrectly rejects pack=8 for this case: the
    # generated C550 kernel is only 32 KiB shared at pack=8.  Keep the hard
    # budget for mixed-radix direct_all joins, whose padded register tensors
    # are lowered through shared memory on MACA.
    direct_register_exchange = (
        _maca_knob("EXCHANGE") in {"direct", "direct_all"}
        and len(plan.factors) > 1
        and not _is_double_dtype(plan.dtype)
        and all(radix & (radix - 1) == 0 for radix in plan.factors)
    )
    # Opt-in resource experiment for the balanced FP64 four-step candidate.
    # Do not lift the 2048-point guard: its measured P8 allocation is 128 KiB.
    # Keep the padded tensor bound below and cap the trial at four slots;
    # compiled resource metadata must still be checked before qualification.
    fp64_register_trial = (
        _maca_knob("FP64_REGISTER_PACK") == "1"
        and _maca_knob("EXCHANGE") in {"direct", "direct_all"}
        and _is_double_dtype(plan.dtype)
        and plan.length in {512, 1024}
        and plan.smem_size == plan.length
        and len(plan.factors) > 1
        and all(radix & (radix - 1) == 0 for radix in plan.factors)
    )

    # Four real-valued shared buffers back the complex exchange. Match the
    # codegen's lane-block rounding so a pack that looks legal algebraically
    # cannot become an oversized allocation after padding.
    def shared_bytes(pack: int) -> int:
        smem_elements = lane_block_for(plan.smem_size * pack)
        return 4 * smem_elements * _real_element_bytes(plan.dtype)

    def direct_all_join_bytes(pack: int) -> int:
        if (
            _maca_knob("EXCHANGE") != "direct_all" and not fp64_register_trial
        ) or len(plan.factors) <= 1:
            return 0

        active_lanes = max(cooperative_stage_lanes_for(plan), default=plan.lanes)
        lane_min = _maca_knob("LANE_MIN", "auto")
        if lane_min == "auto":
            exchange_lane_floor = max(
                1, _PORTABLE_EXCHANGE_MIN_ELEMENTS // max(pack, 1)
            )
        else:
            exchange_lane_floor = _positive_knob("LANE_MIN", lane_min)
        exchange_lane_block = max(lane_block_for(active_lanes), exchange_lane_floor)

        # direct_all joins every radix digit into a power-of-two padded tensor.
        # For mixed radix this padding, rather than the base exchange buffer,
        # is the launch-sized allocation (e.g. radix 17 -> 32).
        joined_radix = max(_next_power_of_two(radix) for radix in plan.factors)
        return (
            exchange_lane_block * pack * joined_radix * _real_element_bytes(plan.dtype)
        )

    def fits(pack: int) -> bool:
        return (
            direct_register_exchange
            or (fp64_register_trial and pack <= 4)
            or shared_bytes(pack) <= budget
        ) and direct_all_join_bytes(pack) <= budget

    limit = 1
    while limit < _portable_exchange_max_pack() and fits(limit * 2):
        limit *= 2
    return limit


def _maca_four_step_inner_pack(plan: LeafPlan | None) -> int:
    """Inner transforms packed per four-step row/column block on MACA.

    The bring-up pinned this to one before any C550 measurement. The override
    lets the profile-aware derivation (``auto``) and explicit packs be compared
    against that baseline without rebuilding; neither path may exceed the
    launchable shared-memory limit.
    """
    override = _maca_knob("INNER_PACK", "")
    if (maca_1d_batch_default_enabled() and plan is not None
            and _is_double_dtype(plan.dtype)
            and "FLAGFFT_MACA_INNER_PACK" not in os.environ
            and resource_default("INNER_PACK") is None):
        # The measured FP64 register exchange uses the normal resource-derived
        # pack, while the P8 default is qualified only for FP32 leaves.
        override = ""
    # An explicit number is an experiment setting, but it still goes through
    # the hard launchable shared-memory limit.
    if override not in {"", "auto"}:
        pack = min(
            _positive_knob("INNER_PACK", override), _portable_exchange_max_pack()
        )
        if (
            pack >= 8
            and plan is not None
            and not _maca_p8_register_leaf_supported(plan)
        ):
            # A global P8 request is useful for screening, but it must not
            # force mixed-radix or short leaves into the register experiment.
            # Reuse the normal resource-derived choice for those plans.
            pack = min(
                _maca_four_step_pack_for(plan),
                _four_step_resource_inner_pack_for(plan),
            )
            pack = min(
                _portable_exchange_pack_floor(plan, pack),
                _portable_exchange_max_pack(),
            )
        return (
            min(pack, _maca_four_step_smem_pack_limit(plan))
            if plan is not None
            else pack
        )
    if plan is None:
        return 1
    # Derived packs fill a 256-thread block.  "Just wide enough to span a warp"
    # is not the right target (on the 390/476-point leaves the lane block is
    # already 128, so that rule leaves the pack at one and measured 5x slower
    # than four), and neither is a flat ceiling: the ceiling rule launched 128
    # threads on the 209/221-point leaves and measured 2.7x slower than the
    # 256-thread block the same plan gets on a 32-lane backend.
    pack = _maca_four_step_pack_for(plan)
    if override == "auto":
        pack = min(pack, _four_step_resource_inner_pack_for(plan))
    pack = min(
        _portable_exchange_pack_floor(plan, pack),
        _portable_exchange_max_pack(),
    )
    return min(pack, _maca_four_step_smem_pack_limit(plan))


def _npu_four_step_inner_pack(plan: LeafPlan | None) -> int:
    raw = os.environ.get("FLAGFFT_NPU_FOURSTEP_PACK", "1")
    if raw == "auto":
        requested = 8
    elif raw in {"1","2","4","8"}:
        requested = int(raw)
    else:
        raise ValueError("FLAGFFT_NPU_FOURSTEP_PACK must be auto, 1, 2, 4 or 8")
    if plan is None or requested == 1:
        return 1
    widths = [1 << (plan.length//r-1).bit_length() for r in plan.factors]
    if os.environ.get("FLAGFFT_NPU_COMPACT_STAGES") != "1":
        widths = [max(widths)]*len(widths)
    joined = max(p*(1 << (r-1).bit_length()) for p,r in zip(widths,plan.factors))
    # Four complex exchange components plus the two padded join tensors.
    # Keep 64 KiB of the 192 KiB UB available for codelet temporaries and I/O.
    per_slot = 4*(1 << (plan.length-1).bit_length())*4 + 2*joined*4
    pack = 1
    while pack*2 <= requested and per_slot*pack*2 <= 128*1024:
        pack *= 2
    return pack


def _four_step_col_inner_pack_for(
    n1: int,
    n2: int,
    dtype: str = "complex64",
    plan: LeafPlan | None = None,
) -> int:
    if _npu_backend_active() and os.environ.get("FLAGFFT_NPU_FOURSTEP_LEAF") == "1":
        return _npu_four_step_inner_pack(plan)
    if _portable_leaf_backend_active():
        return _maca_four_step_inner_pack(plan)
    if plan is not None and _mthreads_small_mixed_leaf(plan):
        return _four_step_resource_inner_pack_for(plan)
    if plan is not None and current_profile().policy != "legacy":
        # Hardware-profile-aware packing: the legacy n1 threshold encodes a
        # 32-lane device tradeoff. Derive the pack from the queried warp
        # width, cooperative stage lanes and shared-memory budget instead.
        return _four_step_resource_inner_pack_for(plan)
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


def _four_step_row_inner_pack_for(
    n1: int,
    n2: int,
    dtype: str = "complex64",
    plan: LeafPlan | None = None,
) -> int:
    if _npu_backend_active() and os.environ.get("FLAGFFT_NPU_FOURSTEP_LEAF") == "1":
        return _npu_four_step_inner_pack(plan)
    if _portable_leaf_backend_active():
        return _maca_four_step_inner_pack(plan)
    if plan is not None and _mthreads_small_mixed_leaf(plan):
        return _four_step_resource_inner_pack_for(plan)
    if plan is not None and current_profile().policy != "legacy":
        # Hardware-profile-aware packing, mirroring the column kernel.  The
        # legacy n1/n2 thresholds below encode 32-lane device tradeoffs.
        return _four_step_resource_inner_pack_for(plan)
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


def _bounded_inner_pack(
    pack: int,
    plan: LeafPlan | None,
    *,
    extra_smem_elements: int = 1,
) -> int:
    profile = current_profile()
    if (
        plan is None
        or profile.policy == "legacy"
        or profile.max_dynamic_shared_memory is None
    ):
        return pack
    if (_ix_backend_active() and _portable_leaf_backend_active()
            and _maca_p8_register_leaf_supported(plan)):
        # The direct power-of-two tensor exchange already has a resource
        # bound that accounts for its joined layout. Applying the generic
        # four-buffer TLE estimate again rejects packs that pass that bound:
        # 1024 x 8 is otherwise estimated as 131200 bytes with the padding.
        return min(pack, _maca_four_step_smem_pack_limit(plan))
    # FourStep tiles use batch_pack=1, so their emitted slot stride has
    # no per-batch padding. A two-stage leaf also leaves smem_a unused.
    # Keep this tighter allocation bound explicit while screening IX tiles.
    buffers = 4
    padded_elements = plan.smem_size + extra_smem_elements
    if hcu_four_step_single_buffer(plan):
        buffers = 2
        padded_elements = plan.smem_size
    if (_ix_backend_active() and _maca_knob("TLE_INNER_PACK")
            and not _portable_leaf_backend_active()):
        buffers = 2 if len(plan.factors) == 2 else 4
        padded_elements = lane_block_for(plan.smem_size)
    bytes_per_fft = buffers * padded_elements * _real_element_bytes(plan.dtype)
    if (_hcu_backend_active() and _is_double_dtype(plan.dtype)
            and os.getenv("FLAGFFT_HCU_VECTOR_IO", "0") == "1"):
        # The paired FP64 tensor can require a layout-exchange tile in
        # addition to TLE's explicit FFT buffers. Reserve one complex value
        # per logical lane instead of filling all LDS with those buffers.
        active = max(cooperative_stage_lanes_for(plan), default=plan.lanes)
        bytes_per_fft += 2 * lane_block_for(active) * _real_element_bytes(plan.dtype)
    return _floor_power_of_two(
        max(1, min(pack, profile.max_dynamic_shared_memory // bytes_per_fft))
    )


def _cuda_four_step_pack_four(plan: LeafPlan | None) -> bool:
    profile = current_profile()
    return (profile.backend == "cuda" and (os.getenv("FLAGFFT_CUDA_FOURSTEP_PACK4") == "1"
            or (os.getenv("FLAGFFT_CUDA_POWER2_REGISTER") == "1" and plan is not None
                and len(plan.factors)==2 and all(r in {16,32} for r in plan.factors)))
            and plan is not None and plan.length <= 1024 and plan.smem_size <= 1024
            and profile.max_dynamic_shared_memory is not None
            and profile.max_dynamic_shared_memory >= 128 * 1024)


def _musa_four_step_inner_pack_override(plan, axis: str) -> int | None:
    if (
        not _mthreads_backend_active()
        or not musa_1d_batch_default_enabled()
        or plan is None
        or plan.length < 64
        or plan.length > 1024
    ):
        return None
    specific_name = f"FLAGFFT_MUSA_1D_FOURSTEP_{axis.upper()}_PACK"
    override = os.getenv(specific_name, "auto")
    if override == "auto":
        override = os.getenv("FLAGFFT_MUSA_1D_FOURSTEP_PACK", "auto")
    if override == "auto":
        return None
    if override not in {"1", "2", "4", "8"}:
        raise ValueError(f"{specific_name} must be auto, 1, 2, 4 or 8")
    return _bounded_inner_pack(int(override), plan)


def _hcu_four_step_pack_for(pack, plan):
    profile = current_profile()
    if profile.backend != "hcu":
        return pack
    mode = os.getenv("FLAGFFT_HCU_FOURSTEP_PACKING", "resource")
    if mode not in {"resource", "fp32_wave", "fp32_wide"}:
        raise ValueError("FLAGFFT_HCU_FOURSTEP_PACKING must be resource, fp32_wave or fp32_wide")
    if (mode == "fp32_wide" and plan is not None and profile.policy != "legacy"
            and plan.dtype == "complex64" and hcu_four_step_single_buffer(plan)
            and max(cooperative_stage_lanes_for(plan)) <= profile.warp_size):
        # Widen the contiguous independent-row axis while keeping at most
        # eight physical waves. The caller applies the queried LDS bound.
        pack = max(pack, 8)
    if (mode == "fp32_wave" and plan is not None and profile.policy != "legacy"
            and plan.dtype == "complex64" and len(plan.factors) == 2
            and hcu_four_step_single_buffer(plan)):
        lanes = cooperative_stage_lanes_for(plan)
        if len(set(lanes)) == 1 and lanes[0] <= profile.warp_size:
            pack = min(pack, max(1, profile.warp_size // lane_block_for(lanes[0])))
    return pack


def four_step_col_inner_pack_for(n1, n2, dtype="complex64", plan=None):
    if _cuda_four_step_pack_four(plan):
        return 4
    if _ix_backend_active() and _maca_knob("TLE_INNER_PACK") and not _portable_leaf_backend_active():
        return _bounded_inner_pack(_positive_knob("TLE_INNER_PACK", _maca_knob("TLE_INNER_PACK")), plan)
    override = _musa_four_step_inner_pack_override(plan, "col")
    if override is not None:
        return override
    # HCU's FP32 1024-point leaves benefit from using the exact unpadded
    # four-step stride: pack four fills 64 KiB and improved the measured
    # large C2R cases.  Keep the generic 64-byte margin for FP64; pack two
    # also fills all 64 KiB there, but the measured n=524288 case regressed
    # from 0.32x to 0.26x as LDS residency fell.
    extra_smem_elements = (
        0
        if current_profile().backend == "hcu" and not _is_double_dtype(dtype)
        else 1
    )
    return _bounded_inner_pack(
        _hcu_four_step_pack_for(_four_step_col_inner_pack_for(n1, n2, dtype, plan), plan),
        plan,
        extra_smem_elements=extra_smem_elements,
    )


def four_step_row_inner_pack_for(n1, n2, dtype="complex64", plan=None):
    if _cuda_four_step_pack_four(plan):
        return 4
    if _ix_backend_active() and _maca_knob("TLE_INNER_PACK") and not _portable_leaf_backend_active():
        return _bounded_inner_pack(_positive_knob("TLE_INNER_PACK", _maca_knob("TLE_INNER_PACK")), plan)
    override = _musa_four_step_inner_pack_override(plan, "row")
    if override is not None:
        return override
    extra_smem_elements = (
        0
        if current_profile().backend == "hcu" and not _is_double_dtype(dtype)
        else 1
    )
    return _bounded_inner_pack(
        _hcu_four_step_pack_for(_four_step_row_inner_pack_for(n1, n2, dtype, plan), plan),
        plan,
        extra_smem_elements=extra_smem_elements,
    )


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
    if (current_profile().backend == "cuda" and os.getenv("FLAGFFT_CUDA_POWER2_REGISTER") == "1"
            and n1 in {256,512,1024} and n2 in {256,512,1024}):
        return True
    if _portable_leaf_backend_active():
        return False
    if _hcu_backend_active():
        setting = os.getenv("FLAGFFT_HCU_FP32_ROW_TWIDDLE", "0")
        if setting not in {"0", "1"}:
            raise ValueError("FLAGFFT_HCU_FP32_ROW_TWIDDLE must be 0 or 1")
        shared = current_profile().max_dynamic_shared_memory or 65536
        if (setting == "1" and dtype == "complex64" and 1 < n1 <= 1024
                and 1 < n2 <= 1024 and n1 * n2 * 8 >= shared):
            return True
    return use_tle_fused_twiddle(n1, n2, dtype) or (
        _is_double_dtype(dtype) and n1 * n2 >= _TLE_FUSED_TWIDDLE_MIN_LENGTH
    )


def _leaf_single_smem_buffer_eligible(
    plan: LeafPlan,
    *,
    io_mode: str,
    four_step_n1: int,
    four_step_n2: int,
) -> bool:
    """Whether one shared buffer can safely replace the stage ping-pong.

    In-place stages swap the roles of the two buffers, so a stage's stores
    may only overwrite data the same stage has already read.  Stages that
    iterate more than one register group need a barrier per iteration; the
    measured four-stage 1024 case still produced wrong results with that
    barrier, so multi-group stages stay out of this path.  The buffer
    reduction only pays off when the two-buffer footprint would limit
    residency; small leaves keep their existing layout.
    """
    profile = current_profile()
    if profile.policy == "legacy" or profile.max_dynamic_shared_memory is None:
        return False
    if plan.dtype != "complex64" or len(plan.factors) < 2:
        return False
    if io_mode in {"permuted_store", "strided_permuted_store", "permuted_r2c", "permuted_store_r2c"}:
        return _ix_permuted_single_smem(plan)
    ix_boundary = (
        profile.backend == "ix" and profile.device_arch == "71"
        and _maca_knob("BLUESTEIN_SINGLE_SMEM", "0") == "1"
        and io_mode in {"bluestein_prepare_leaf", "bluestein_finish_leaf"}
    )
    ix_cooperative = (
        profile.backend == "ix" and profile.device_arch == "71"
        and os.getenv("FLAGFFT_IX_RADIX8", "0") == "1"
        and io_mode in {"contiguous", "strided"}
        and 64 <= plan.length <= 4096 and plan.length & (plan.length - 1) == 0
        and all(radix in {2,4,8} for radix in plan.factors)
    )
    if io_mode not in {"contiguous", "strided"} and not ix_boundary:
        return False
    stage_lanes = cooperative_stage_lanes_for(plan)
    if not (ix_boundary or ix_cooperative) and any(lanes != plan.lanes for lanes in stage_lanes):
        return False
    for index in range(len(plan.factors) - 1):
        if plan.length // (stage_lanes[index] * plan.factors[index]) != 1:
            return False
    batch_pack = contiguous_batch_pack_for(plan)
    smem_pack = max(batch_pack, 1)
    smem_slot_stride = plan.smem_size + 1 if batch_pack >= 4 else plan.smem_size
    smem_n = lane_block_for(smem_slot_stride * smem_pack)
    smem_bytes = 4 * smem_n * _real_element_bytes(plan.dtype)
    return smem_bytes >= profile.max_dynamic_shared_memory // 4


def _use_single_smem_buffer(
    plan: LeafPlan,
    *,
    io_mode: str = "contiguous",
    four_step_n1: int = 0,
    four_step_n2: int = 0,
) -> bool:
    """Reuse one shared buffer between generated mixed-radix stages."""
    if hcu_convolution_single_buffer(plan, io_mode):
        return True
    if hcu_leaf_single_buffer(plan, io_mode):
        return True
    if (io_mode.startswith(("four_step_", "bluestein_four_step_")) and not io_mode.endswith("_strided")
            and hcu_four_step_single_buffer(plan)):
        return True
    if (_hcu_wide_permuted_store(plan) and io_mode in {
            "permuted_store", "strided_permuted_store", "permuted_r2c",
            "permuted_store_r2c"}):
        return True
    if (current_profile().backend == "cuda" and
            os.getenv("FLAGFFT_CUDA_LEAF_SINGLE_BUFFER") == "1"):
        allowed_modes = {
            "contiguous", "strided", "contiguous_r2c", "contiguous_c2r", "packed_r2c", "packed_c2r",
            "permuted_store", "strided_permuted_store", "permuted_r2c", "permuted_store_r2c",
            "bluestein_prepare_leaf", "bluestein_finish_leaf",
        }
        # Every overwriting stage must consume its entire shared input before
        # the barrier and its first write. Multi-group stages cannot do that.
        stage_lanes = cooperative_stage_lanes_for(plan)
        if (io_mode in allowed_modes and len(plan.factors) >= 2 and
                (all(lanes == plan.lanes for lanes in stage_lanes) or
                 os.getenv("FLAGFFT_CUDA_WIDE_STAGE") == "1") and
                all(plan.length == lanes * radix
                    for lanes,radix in zip(stage_lanes,plan.factors[:-1]))):
            return True
    if _leaf_single_smem_buffer_eligible(
        plan,
        io_mode=io_mode,
        four_step_n1=four_step_n1,
        four_step_n2=four_step_n2,
    ):
        return True
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


def _declared_backend() -> str:
    """Backend identity from the queried device profile, else the codegen target."""
    profile = current_profile()
    if profile.device_arch != "unspecified":
        return profile.backend
    from .target import backend_name

    return backend_name()


def _mthreads_backend_active() -> bool:
    """Whether the installed Triton targets Moore Threads (MUSA/mtgpu)."""
    backend = _declared_backend()
    if backend:
        return backend in {"musa", "mthreads", "mtgpu"}
    return _triton_plugin_present("mthreads")


def _ppu_backend_active() -> bool:
    """Whether the installed Triton targets the T-Head PPU (XuanTie GPU).

    The thread-local mixed-radix four-step kernels and the vectorized 3D
    transpose variants rely on PTX inline-asm register patterns that the
    PPU compiler toolchain does not support, so they are disabled there.
    """
    backend = _declared_backend()
    if backend:
        return backend == "ppu"
    return _triton_plugin_present("ppu")


def _maca_backend_active() -> bool:
    backend = _declared_backend()
    if backend:
        return backend in {"maca", "metax"}
    return _triton_plugin_present("metax")


def _portable_leaf_backend_active() -> bool:
    """Select tensor exchange when the backend's TLE shared memory is unavailable."""
    return (current_profile().backend == "cuda" and
            os.getenv("FLAGFFT_CUDA_PORTABLE_LEAF") == "1") or _maca_backend_active() or (
        _npu_backend_active() and os.environ.get("FLAGFFT_NPU_FOURSTEP_LEAF") == "1"
    ) or (
        _ix_backend_active() and _maca_knob("PORTABLE_LEAF") == "1"
    )


def _npu_backend_active() -> bool:
    """Whether the installed Triton targets the Ascend NPU (CANN/torch_npu).

    The standalone generator has no queried profile, so the legacy environment
    markers stay as a fallback for direct ``python -m`` invocation.
    """
    backend = _declared_backend()
    if backend:
        return backend in {"npu", "ascend"}
    return (
        os.environ.get("TRITON_JIT_BACKEND") == "NPU"
        or os.environ.get("FLAGTREE_BACKEND") == "ascend"
        or os.environ.get("TRITON_BACKEND") in {"npu", "torch_npu"}
    )


def direct_dft_batch_tile(n: int, *, strided: bool = False, dtype: str = "complex64") -> int:
    """Batches per program for tiny direct DFTs on the NPU.

    Small transforms are launch-bound: at batch 64 the per-program dispatch
    dominates the arithmetic. Folding a fixed group of batches into one
    program's vector lanes keeps the grid small without lengthening the
    dependency chain.
    """
    if strided or dtype != "complex64" or not _npu_backend_active() or n > 32:
        return 1
    override = os.environ.get("FLAGFFT_NPU_DIRECT_DFT_BT")
    if override:
        return max(1, int(override))
    return 8


def _hcu_backend_active() -> bool:
    """Whether the installed Triton targets Hygon BW1000/HCU."""
    backend = _declared_backend()
    if backend:
        return backend in {"hcu", "hygon"}
    return (
        os.environ.get("TRITON_JIT_BACKEND", "").lower() in {"hcu", "hygon"}
        or os.environ.get("FLAGTREE_BACKEND", "").lower() in {"hcu", "hygon"}
        or _triton_plugin_present("hcu")
    )


def _ix_backend_active() -> bool:
    """Whether the installed Triton targets Iluvatar (Tianshu/CoreX)."""
    backend = _declared_backend()
    if backend:
        return backend in {"ix", "corex", "iluvatar"}
    return _triton_plugin_present("iluvatar")


def _non_nvidia_backend_active() -> bool:
    """Whether the installed Triton is a non-NVIDIA port (MThreads/PPU/IX/NPU/MACA/HCU).

    The thread-local mixed-radix four-step kernels and the vectorized 3D
    transpose variants rely on register/asm patterns that the MThreads
    MTGPU LLVM backend cannot compile (llc register allocation failure)
    and that the PPU/IX/NPU toolchains do not support, so they are disabled on
    these backends. HCU also uses a non-PTX HIP code-generation path, so it
    must not receive the NVIDIA inline-assembly variants by default.
    """
    return (
        _mthreads_backend_active()
        or _ppu_backend_active()
        or _ix_backend_active()
        or _maca_backend_active()
        or _npu_backend_active()
        or _hcu_backend_active()
    )


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
    "_PORTABLE_EXCHANGE_MAX_PACK",
    "_PORTABLE_EXCHANGE_MIN_ELEMENTS",
    "_ix_backend_active",
    "_hcu_backend_active",
    "_maca_backend_active",
    "_maca_four_step_inner_pack",
    "_maca_knob",
    "_mthreads_backend_active",
    "_next_power_of_two",
    "_non_nvidia_backend_active",
    "_npu_backend_active",
    "_portable_exchange_pack_floor",
    "_ppu_backend_active",
    "_real_element_bytes",
    "_tl_real_dtype",
    "_triton_plugin_present",
    "_use_single_smem_buffer",
    "_zero_other",
    "codelet_radices_for",
    "contiguous_batch_pack_for",
    "cooperative_stage_lanes_for",
    "emitted_leaf_factors",
    "four_step_col_inner_pack_for",
    "four_step_row_inner_pack_for",
    "lane_block_for",
    "use_four_step_row_fused_twiddle",
    "use_tle_fused_twiddle",
]
