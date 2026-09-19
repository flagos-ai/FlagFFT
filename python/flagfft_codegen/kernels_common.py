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
# The MetaX plugin cannot lower maca.shfl.sync, so the portable exchange must
# gather from a tensor wider than one 64-thread warp.
_PORTABLE_EXCHANGE_MIN_ELEMENTS = 128
# Hard ceiling for the exchange pack.  Past four the exchange tensor outgrows
# MetaX's 64 KiB of shared memory per SM: measured at eight the kernel drops to
# a tenth of its bandwidth, and at sixteen it fails to compile.
_PORTABLE_EXCHANGE_MAX_PACK = 4
# Derived packs stop here.  Measured on the 1D ct four-step leaves, two beats
# four on every batch shape (1.28x at n=8192, 1.59x at n=46189) because the
# wider gather costs more than the extra inner columns buy back.
_PORTABLE_EXCHANGE_PREFERRED_PACK = 2
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


def emitted_leaf_factors(
    plan: LeafPlan, io_mode: str = "contiguous"
) -> tuple[int, ...]:
    if (
        _maca_backend_active()
        and io_mode != "bluestein_full_leaf"
        and plan.length
        in _NATURAL_ORDER_CODELET_RADICES | _THREAD_LOCAL_MIXED_RADICES | {16}
    ):
        # Keep small FFTs in registers: this SDK cannot parse the plugin's
        # maca.shfl.sync op generated for small tensor layout conversions.
        return (plan.length,)
    return plan.factors


def cooperative_stage_lanes_for(plan: LeafPlan) -> tuple[int, ...]:
    if _maca_backend_active():
        # The portable exchange evaluates every butterfly in one tensor.
        return tuple(plan.length // radix for radix in plan.factors)
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


def _maca_knob(name: str, default: str = "") -> str:
    """Read a MACA code-generation override (``FLAGFFT_MACA_<NAME>``).

    Every caller's default reproduces the shipped constant, so an environment
    without these variables keeps today's behaviour exactly.
    """
    return os.environ.get(f"FLAGFFT_MACA_{name}", default).strip().lower()


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

    def pack_for(target_threads):
        thread_pack = max(1, target_threads // lane_block)
        tiny_single_stage = plan.length <= 8 and len(plan.factors) == 1
        if not tiny_single_stage and plan.length <= 128:
            thread_pack = min(thread_pack, 4)
        if len(plan.factors) <= 1:
            return thread_pack
        bytes_per_fft = 4 * (plan.smem_size + 1) * _real_element_bytes(plan.dtype)
        smem_pack = max(
            1, profile.shared_budget(_LEAF_PACK_SMEM_BUDGET_BYTES) // bytes_per_fft
        )
        return _floor_power_of_two(max(1, min(thread_pack, smem_pack)))

    return _register_bounded_batch_pack(
        plan, pack_for(profile.leaf_target_threads), pack_for(32)
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
    return max(pack, min(_PORTABLE_EXCHANGE_MAX_PACK, _next_power_of_two(needed)))


def contiguous_batch_pack_for(plan: LeafPlan) -> int:
    if _maca_backend_active():
        override = _maca_knob("BATCH_PACK")
        if override == "auto":
            return _profile_batch_pack_for(plan)
        if override:
            return _positive_knob("BATCH_PACK", override)
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
    bytes_per_fft = 4 * plan.smem_size * _real_element_bytes(plan.dtype)
    smem_pack = max(
        1,
        current_profile().shared_budget(_FOUR_STEP_PACK_SMEM_BUDGET_BYTES)
        // bytes_per_fft,
    )
    max_pack = _FOUR_STEP_LARGE_INNER_PACK
    if _mthreads_small_mixed_leaf(plan):
        max_pack = 8
    return _floor_power_of_two(max(1, min(max_pack, thread_pack, smem_pack)))


def _maca_four_step_inner_pack(plan: LeafPlan | None) -> int:
    """Inner transforms packed per four-step row/column block on MACA.

    The bring-up pinned this to one before any C550 measurement.  The override
    lets the profile-aware derivation (``auto``) and explicit packs be compared
    against that baseline without rebuilding.
    """
    override = _maca_knob("INNER_PACK", "")
    # An explicit number is an experiment setting and wins outright; only the
    # derived packs go through the tensor-width floor.
    if override not in {"", "auto"}:
        return min(_positive_knob("INNER_PACK", override), _PORTABLE_EXCHANGE_MAX_PACK)
    if plan is None:
        return 1
    pack = (
        _four_step_resource_inner_pack_for(plan)
        if override == "auto"
        else 1
    )
    # The exchange tensor is O(lane_block * pack) and MetaX has 64 KiB of
    # shared memory per SM: measured at 8 the kernel collapses to a tenth of
    # its bandwidth and at 16 it fails to compile ("memory size or pointer
    # value too large to fit in 32 bit").  Clamp so no setting can reach it.
    return min(
        _portable_exchange_pack_floor(plan, pack), _PORTABLE_EXCHANGE_PREFERRED_PACK
    )


def _four_step_col_inner_pack_for(
    n1: int,
    n2: int,
    dtype: str = "complex64",
    plan: LeafPlan | None = None,
) -> int:
    if _maca_backend_active():
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
    if _maca_backend_active():
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


def _bounded_inner_pack(pack: int, plan: LeafPlan | None) -> int:
    profile = current_profile()
    if (
        plan is None
        or profile.policy == "legacy"
        or profile.max_dynamic_shared_memory is None
    ):
        return pack
    bytes_per_fft = 4 * (plan.smem_size + 1) * _real_element_bytes(plan.dtype)
    return _floor_power_of_two(
        max(1, min(pack, profile.max_dynamic_shared_memory // bytes_per_fft))
    )


def four_step_col_inner_pack_for(n1, n2, dtype="complex64", plan=None):
    return _bounded_inner_pack(_four_step_col_inner_pack_for(n1, n2, dtype, plan), plan)


def four_step_row_inner_pack_for(n1, n2, dtype="complex64", plan=None):
    return _bounded_inner_pack(_four_step_row_inner_pack_for(n1, n2, dtype, plan), plan)


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
    if _maca_backend_active():
        return False
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
    if io_mode not in {"contiguous", "strided"}:
        return False
    stage_lanes = cooperative_stage_lanes_for(plan)
    if any(lanes != plan.lanes for lanes in stage_lanes):
        return False
    for index in range(len(plan.factors) - 1):
        if plan.length // (plan.lanes * plan.factors[index]) != 1:
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


def _ix_backend_active() -> bool:
    """Whether the installed Triton targets Iluvatar (Tianshu/CoreX)."""
    backend = _declared_backend()
    if backend:
        return backend in {"ix", "corex", "iluvatar"}
    return _triton_plugin_present("iluvatar")


def _non_nvidia_backend_active() -> bool:
    """Whether the installed Triton is a non-NVIDIA port (MThreads/PPU/IX/NPU/MACA).

    The thread-local mixed-radix four-step kernels and the vectorized 3D
    transpose variants rely on register/asm patterns that the MThreads
    MTGPU LLVM backend cannot compile (llc register allocation failure)
    and that the PPU/IX/NPU toolchains do not support, so they are disabled on
    these backends.
    """
    return (
        _mthreads_backend_active()
        or _ppu_backend_active()
        or _ix_backend_active()
        or _maca_backend_active()
        or _npu_backend_active()
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
    "_PORTABLE_EXCHANGE_PREFERRED_PACK",
    "_ix_backend_active",
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
