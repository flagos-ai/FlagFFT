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

"""Generated-module assembly: signatures, module source wrapping and JIT metadata."""

from pathlib import Path
from typing import Any

from .kernels_common import (
    _CODELET_DIR,
    LeafPlan,
    contiguous_batch_pack_for,
    cooperative_stage_lanes_for,
    four_step_col_inner_pack_for,
    four_step_row_inner_pack_for,
    lane_block_for,
    use_four_step_row_fused_twiddle,
)
from .registry import (
    CONTIGUOUS_BATCH_PACK_KERNELS,
    INNER_PACK_COL_KERNELS,
    INNER_PACK_ROW_KERNELS,
    STRIDED_FOUR_STEP_KERNELS,
    is_four_step_twiddle_eligible,
)

def _pointer_signature(dtype: str) -> str:
    if dtype == "complex128":
        return "*fp64:16"
    return "*fp32:16"


def _dtype_suffix(dtype: str) -> str:
    return "f64" if dtype == "complex128" else "f32"


def _zero_literal(dtype: str) -> str:
    # `tl.load(..., other=0.0)` is auto-promoted to the pointer's dtype by
    # Triton's masked-load lowering, so a single literal works for fp32 and
    # fp64. Indexing a constexpr (e.g. ``tl.zeros((1,), tl.float64)[0]``) is
    # rejected at IR-build time.
    del dtype
    return "0.0"


def _csv_ints(raw: str) -> tuple[int, ...]:
    if raw == "":
        return ()
    return tuple(int(part) for part in raw.split(",") if part)


def _module_source(kernel_source: str) -> str:
    helpers = (
        "import triton\n"
        "import triton.language as tl\n"
        "import triton.experimental.tle.language as tle\n\n"
    )
    utils_path = _CODELET_DIR / "utils.py"
    if utils_path.exists():
        helpers += utils_path.read_text() + "\n\n"

    for codelet_file in sorted(_CODELET_DIR.glob("*.py")):
        if codelet_file.name not in {utils_path.name, Path(__file__).name}:
            helpers += codelet_file.read_text() + "\n\n"

    return helpers + "\n\n" + kernel_source + "\n"


def _arg_signature(name: str, dtype: str) -> str:
    if name == "nbatch":
        return "i32"
    if name in {"n", "m", "input_distance", "output_distance", "outer_stride"}:
        return "i64"
    if name == "idx_ptr":
        return "*i32:16"
    return _pointer_signature(dtype)


def _signature(arg_names: list[str], dtype: str) -> str:
    return ",".join(_arg_signature(name, dtype) for name in arg_names)


def _metadata(
    *,
    module_path: Path,
    kernel_name: str,
    arg_names: list[str],
    plan: LeafPlan,
    kernel_type: str,
    n1: int,
    n2: int,
    dtype: str,
) -> dict[str, Any]:
    batch_per_block = (
        contiguous_batch_pack_for(plan)
        if kernel_type in CONTIGUOUS_BATCH_PACK_KERNELS
        else 1
    )
    stage_lanes = cooperative_stage_lanes_for(plan)
    tle_fused_twiddle = (
        is_four_step_twiddle_eligible(kernel_type)
        and use_four_step_row_fused_twiddle(n1, n2, dtype)
    )
    if kernel_type in STRIDED_FOUR_STEP_KERNELS:
        inner_pack = 1
    elif kernel_type in INNER_PACK_ROW_KERNELS:
        inner_pack = four_step_row_inner_pack_for(n1, n2, dtype, plan)
    elif kernel_type in INNER_PACK_COL_KERNELS:
        inner_pack = four_step_col_inner_pack_for(n1, n2, dtype, plan)
    else:
        inner_pack = 1
    num_warps = int(plan.num_warps)
    if kernel_type == "direct_dft" and dtype == "complex64":
        num_warps = 4
    if tle_fused_twiddle:
        num_warps = min(8, num_warps * inner_pack)
    work_pack = max(batch_per_block, inner_pack)
    if work_pack > 1 or any(lanes != plan.lanes for lanes in stage_lanes):
        cooperative_warps = 1
        required_warps = (lane_block_for(max(stage_lanes)) * work_pack + 31) // 32
        while cooperative_warps < required_warps and cooperative_warps < 8:
            cooperative_warps *= 2
        num_warps = max(num_warps, cooperative_warps)
    if "_thread_local_" in kernel_name:
        num_warps = 4
    return {
        "module_path": str(module_path),
        "kernel_name": kernel_name,
        "signature": _signature(arg_names, dtype),
        "num_warps": num_warps,
        "num_stages": 1,
        "batch_per_block": int(batch_per_block),
        "arg_names": arg_names,
        "kernel_type": kernel_type,
        "length": int(plan.length),
        "lanes": int(plan.lanes),
        "direction": plan.direction,
        "dtype": dtype,
        "n1": int(n1),
        "n2": int(n2),
        "inner_pack": inner_pack,
        "tle_fused_twiddle": tle_fused_twiddle,
    }


__all__ = [
    "_arg_signature",
    "_csv_ints",
    "_dtype_suffix",
    "_metadata",
    "_module_source",
    "_pointer_signature",
    "_signature",
    "_zero_literal",
]
