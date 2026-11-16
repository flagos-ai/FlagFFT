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
import os
from typing import Any

from .backend_profile import current_profile
from .kernels_common import (
    _CODELET_DIR,
    LeafPlan,
    _cooperative_warp_cap,
    _dtype_suffix,
    _hcu_backend_active,
    _maca_knob,
    _mthreads_backend_active,
    _npu_backend_active,
    _portable_leaf_backend_active,
    _ix_backend_active,
    _zero_other,
    contiguous_batch_pack_for,
    cooperative_stage_lanes_for,
    four_step_col_inner_pack_for,
    four_step_row_inner_pack_for,
    lane_block_for,
    permuted_store_batch_pack_for,
    use_four_step_row_fused_twiddle,
)
from .registry import (
    CONTIGUOUS_BATCH_PACK_KERNELS,
    INNER_PACK_COL_KERNELS,
    INNER_PACK_ROW_KERNELS,
    STRIDED_FOUR_STEP_KERNELS,
    is_four_step_twiddle_eligible,
)
from .target import musa_1d_batch_default_enabled


def _pointer_signature(dtype: str) -> str:
    if dtype == "complex128":
        return "*fp64:16"
    return "*fp32:16"


def _csv_ints(raw: str) -> tuple[int, ...]:
    if raw == "":
        return ()
    return tuple(int(part) for part in raw.split(",") if part)


def _module_source(kernel_source: str, radices: tuple[int, ...] = ()) -> str:
    if _npu_backend_active():
        if os.environ.get("FLAGFFT_NPU_VECTOR_REMAINDER") == "1":
            from .npu_source import vector_remainders
            kernel_source = vector_remainders(kernel_source)
        # CANN exposes tensor slice operations through the language extension.
        # Use that public surface instead of requiring the optional TLE module.
        for operation in ("insert_slice", "extract_slice"):
            kernel_source = kernel_source.replace(f"tle.dsa.{operation}", f"al.{operation}")
        if "al." in kernel_source and "import triton.language.extra.cann.extension as al" not in kernel_source:
            kernel_source = "import triton.language.extra.cann.extension as al\n" + kernel_source
    helpers = (
        "import os\n"
        "if (os.environ.get('TRITON_BACKEND') in {'torch_npu', 'npu'}\n"
        "        or os.environ.get('FLAGTREE_BACKEND') == 'ascend'\n"
        "        or os.environ.get('TRITON_JIT_BACKEND') == 'NPU'):\n"
        "    import torch\n"
        "    import torch_npu\n"
        "import triton\n"
        "import triton.language as tl\n"
    )
    # Keep the plain-Triton path independent of FlagTree TLE.  Leaf kernels
    # still request TLE explicitly through their generated ``tle.`` calls,
    # while direct/pointwise/transpose kernels only need the standard Triton
    # language and can therefore be bootstrapped on a backend before TLE
    # features are enabled.
    if "tle." in kernel_source:
        helpers += "import triton.experimental.tle.language as tle\n"
    helpers += "\n"
    utils_path = _CODELET_DIR / "utils.py"
    if utils_path.exists():
        helpers += utils_path.read_text() + "\n\n"

    for radix in radices:
        if (_npu_backend_active()
                and os.environ.get("FLAGFFT_NPU_SYMMETRIC_CODELETS", "1") != "0"):
            from .npu_codelets import codelet_source
            optimized = codelet_source(radix)
            if optimized is not None:
                helpers += optimized + "\n\n"
                continue
        codelet_path = _CODELET_DIR / f"radix{radix}.py"
        if codelet_path.exists():
            helpers += codelet_path.read_text() + "\n\n"

    source = helpers + "\n\n" + kernel_source + "\n"
    if _portable_leaf_backend_active():
        # Portable MACA kernels do not use TLE. Keep their modules independent
        # of the backend-specific TLE extensions during isolated compilation.
        source = source.replace("import triton.experimental.tle.language as tle\n", "")
    if "_ixdisjoint_" in kernel_source:
        import re
        names=re.findall(r"def (\w+_ixdisjoint_[0-9a-f]{12})\(",kernel_source)
        if len(names)!=1:
            raise ValueError("Expected one generated IX disjoint-stage entry")
        source += ("\nfrom flagfft_codegen.ix_disjoint_stage import register_kernel as _register_ix_stage\n"
                   + f"_register_ix_stage({names[0]})\n")
    return source


def _arg_signature(name: str, dtype: str) -> str:
    if name == "nbatch":
        return "i32"
    if name in {
        "n",
        "m",
        "input_distance",
        "output_distance",
        "outer_stride",
        "perm_span",
        "rows_per_transform",
        "span",
    }:
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
    hcu_full_smem: bool = False,
    permutation_span: int = 0,
    permutation_batch: int = 0,
) -> dict[str, Any]:
    if (
        current_profile().backend == "hcu"
        and kernel_type
        in {"leaf_permuted_store", "leaf_strided_permuted_store", "leaf_r2c_permuted_store"}
    ):
        pack_knob = None
        for tag, knob in (
            ("outer_first", "FLAGFFT_HCU_3D_FIRST_PACK"),
            ("inner_middle", "FLAGFFT_HCU_3D_MIDDLE_PACK"),
            ("outer_last", "FLAGFFT_HCU_3D_FINAL_PACK"),
        ):
            if f"permuted_store_{tag}_" in kernel_name:
                pack_knob = knob
                break
        requested_pack = os.environ.get(pack_knob, "auto") if pack_knob else "auto"
        final_pack = int(requested_pack) if requested_pack != "auto" else None
        if (
            final_pack is None
            and current_profile().backend == "hcu"
            and "permuted_store_inner_middle_c2r_cube_" in kernel_name
        ):
            final_pack = 8
        batch_per_block = permuted_store_batch_pack_for(
            plan, force_full_smem=hcu_full_smem, pack_override=final_pack,
            permutation_span=permutation_span, permutation_batch=permutation_batch,
            inner=kernel_name.startswith("permuted_store_inner"),
        )
    elif (kernel_type in {"leaf_permuted_store", "leaf_strided_permuted_store", "leaf_r2c_permuted_store"}
          or (current_profile().backend == "cuda" and kernel_type == "leaf_r2c"
              and kernel_name.startswith("r2c_permuted_store"))) and (
        current_profile().backend == "cuda"
        or _ix_backend_active()
        or _mthreads_backend_active()
        or (
            _npu_backend_active()
            and os.environ.get("FLAGFFT_NPU_2D_TRANSPOSE_STORE") == "1"
        )
    ):
        # The generated kernel advances batch_id by this pack.  Launch the
        # same number of rows per CTA that the source actually processes.
        batch_per_block = permuted_store_batch_pack_for(plan)
        if _npu_backend_active() and os.environ.get("FLAGFFT_NPU_2D_TRANSPOSE_STORE") == "1":
            override = os.environ.get("FLAGFFT_NPU_2D_TRANSPOSE_PACK")
            if override is not None:
                try:
                    batch_per_block = int(override)
                except ValueError as exc:
                    raise ValueError(
                        "FLAGFFT_NPU_2D_TRANSPOSE_PACK must be 1, 2, 4 or 8"
                    ) from exc
                if batch_per_block not in {1, 2, 4, 8}:
                    raise ValueError("FLAGFFT_NPU_2D_TRANSPOSE_PACK must be 1, 2, 4 or 8")
    elif (
        current_profile().backend == "maca"
        and kernel_type in {"leaf_permuted_store", "leaf_strided_permuted_store", "leaf_r2c_permuted_store"}
    ):
        batch_per_block = permuted_store_batch_pack_for(plan)
    elif kernel_type in CONTIGUOUS_BATCH_PACK_KERNELS:
        real_transform = (
            "r2c"
            if kernel_type in {"leaf_r2c", "leaf_packed_r2c", "leaf_bluestein_r2c"}
            else "c2r"
            if kernel_type in {"leaf_c2r", "leaf_packed_c2r", "leaf_bluestein_c2r"}
            else None
        )
        batch_per_block = contiguous_batch_pack_for(
            plan,
            real_boundary=real_transform is not None,
            real_transform=real_transform,
        )
    else:
        batch_per_block = 1
    stage_lanes = cooperative_stage_lanes_for(plan)
    tle_fused_twiddle = is_four_step_twiddle_eligible(
        kernel_type
    ) and use_four_step_row_fused_twiddle(n1, n2, dtype)
    if kernel_type in STRIDED_FOUR_STEP_KERNELS:
        inner_pack = 1
    elif kernel_type in INNER_PACK_ROW_KERNELS:
        inner_pack = four_step_row_inner_pack_for(n1, n2, dtype, plan)
    elif kernel_type in INNER_PACK_COL_KERNELS:
        inner_pack = four_step_col_inner_pack_for(n1, n2, dtype, plan)
    else:
        inner_pack = 1
    profile = current_profile()
    num_warps = profile.planner_warps(int(plan.num_warps))
    if kernel_type in {
        "direct_dft",
        "direct_dft_cube",
        "direct_dft_cube_transposed",
        "direct_dft_cube_strided",
        "direct_dft_r2c",
        "direct_dft_c2r",
    } and dtype == "complex64":
        num_warps = 4
    if profile.backend == "cuda" and kernel_type in {
        "direct_dft", "direct_dft_strided", "direct_dft_r2c", "direct_dft_c2r"
    }:
        override = os.getenv("FLAGFFT_CUDA_DIRECT_DFT_WARPS", "auto")
        if override != "auto":
            if override not in {"1", "2", "4"}:
                raise ValueError("FLAGFFT_CUDA_DIRECT_DFT_WARPS must be auto, 1, 2 or 4")
            num_warps = int(override)
    if tle_fused_twiddle and not (profile.backend == "hcu" and dtype == "complex64"):
        num_warps = min(8, num_warps * inner_pack)
    work_pack = max(batch_per_block, inner_pack)
    maca_backend = _portable_leaf_backend_active()
    if work_pack > 1 or any(lanes != plan.lanes for lanes in stage_lanes):
        cooperative_warps = 1
        # MACA is validated against its plugin's 64-thread warp; the other
        # backends derive the warp count from the queried hardware profile.
        if maca_backend:
            from .target import warp_size

            threads = warp_size()
            required_warps = (
                lane_block_for(max(stage_lanes)) * work_pack + threads - 1
            ) // threads
        else:
            required_warps = profile.warps_for(
                lane_block_for(max(stage_lanes)) * work_pack
            )
        warp_cap = _cooperative_warp_cap()
        while cooperative_warps < required_warps and cooperative_warps < warp_cap:
            cooperative_warps *= 2
        num_warps = max(num_warps, cooperative_warps)
    if "_thread_local_" in kernel_name:
        num_warps = 4
    if maca_backend and not _ix_backend_active():
        # One warp triggers unsupported shuffle lowering on MACA. IX shares
        # portable exchange code but validates its own warp override below.
        num_warps = max(2, num_warps)
        warp_override = _maca_knob("WARPS")
        # IX validates its override against the IX hardware profile below;
        # the portable path must not impose MetaX's two-warp minimum there.
        if warp_override and not _ix_backend_active():
            try:
                requested_warps = int(warp_override)
            except ValueError:
                raise ValueError(
                    f"FLAGFFT_MACA_WARPS must be 2, 4 or 8, got {warp_override!r}"
                ) from None
            if requested_warps not in {2, 4, 8}:
                raise ValueError(
                    f"FLAGFFT_MACA_WARPS must be 2, 4 or 8, got {requested_warps}"
                )
            profile.validate(requested_warps)
            num_warps = requested_warps
    if _ix_backend_active() and _maca_knob("WARPS"):
        num_warps = int(_maca_knob("WARPS"))
        profile.validate(num_warps)
    if _ix_backend_active():
        resource_warps = _maca_knob("RESOURCE_WARPS", "0")
        if resource_warps not in {"0", "1"}:
            raise ValueError("FLAGFFT_IX_RESOURCE_WARPS must be 0 or 1")
        if (resource_warps == "1" and maca_backend
                and kernel_type in INNER_PACK_ROW_KERNELS | INNER_PACK_COL_KERNELS
                and kernel_type not in STRIDED_FOUR_STEP_KERNELS
                and os.environ.get("FLAGFFT_IX_WARPS") is None):
            # Bound per-thread live data before codelet temporaries. For P8,
            # the 512-point row can use two warps while the 1024-point column
            # needs four to avoid the spills observed with narrower launches.
            threads = (plan.length * inner_pack + 31) // 32
            num_warps = max(2, profile.warps_for(threads))
            profile.validate(num_warps)
    from .target import ix_fourstep_preset
    if (_ix_backend_active() and kernel_type == "leaf_packed_r2c" and ix_fourstep_preset() not in {7,8}
            and os.environ.get("FLAGFFT_IX_WARPS") is None):
        # The half-length reconstruction has twice the register work of a
        # normal leaf. Four warps caused a large batch-64 regression on V150;
        # the measured two-warp launch retains the one-kernel benefit.
        num_warps = 2
    if (profile.backend == "musa" and musa_1d_batch_default_enabled()
            and 64 <= plan.length <= 1024):
        warp_override = os.getenv("FLAGFFT_MUSA_1D_BATCH_WARPS", "auto")
        if warp_override not in {"auto", "1", "2", "4", "8"}:
            raise ValueError("FLAGFFT_MUSA_1D_BATCH_WARPS must be auto, 1, 2, 4 or 8")
        if warp_override != "auto":
            num_warps = int(warp_override)
            profile.validate(num_warps)
    if profile.backend == "hcu" and kernel_type in {"leaf_permuted_store", "leaf_strided_permuted_store"}:
        is_final_axis = "permuted_store_outer_last_" in kernel_name
        is_middle_axis = "permuted_store_inner_" in kernel_name
        final_warps = os.environ.get("FLAGFFT_HCU_3D_FINAL_WARPS") if is_final_axis else None
        middle_warps = os.environ.get("FLAGFFT_HCU_3D_MIDDLE_WARPS") if is_middle_axis else None
        fused_warps = os.environ.get("FLAGFFT_HCU_3D_FUSED_WARPS")
        if final_warps not in (None, "auto"):
            fused_warps = final_warps
        if middle_warps not in (None, "auto"):
            fused_warps = middle_warps
        if fused_warps is not None:
            try:
                num_warps = int(fused_warps)
            except ValueError as exc:
                raise ValueError(
                    "FLAGFFT_HCU_3D_FUSED_WARPS, FINAL_WARPS and MIDDLE_WARPS must be 1, 2, 4 or 8"
                ) from exc
            profile.validate(num_warps)
    warp_pack = 0
    if kernel_name.startswith("hcu_small_wave_"):
        import re
        batch_per_block = int(re.search(r"_p(\d+)_", kernel_name).group(1))
        num_warps = 1
        profile.validate(num_warps)
        if batch_per_block * plan.length != profile.warp_size:
            raise ValueError("small HCU FFT rows must exactly fill one wave")
    if kernel_name.startswith("hcu_warp"):
        import re
        batch_per_block = int(re.search(r"_p(\d+)_", kernel_name).group(1))
        wave_lanes = int(kernel_name.split("_lanes", 1)[1].split("_", 1)[0]) if "_lanes" in kernel_name else 64
        num_warps = min(8, batch_per_block * wave_lanes // 64)
        profile.validate(num_warps)
        chunk = int(kernel_name.split("_chunk", 1)[1].split("_", 1)[0]) if "_chunk" in kernel_name else plan.length // wave_lanes
        element_bytes = 16 if dtype == "complex128" else 8
        if batch_per_block * wave_lanes * chunk * element_bytes > profile.max_dynamic_shared_memory:
            raise ValueError("HCU wave FFT output tile exceeds shared-memory capacity")
        if kernel_type in {"four_step_row", "four_step_col", "four_step_real_row",
                           "four_step_hermitian_row", "four_step_r2c_col", "four_step_c2r_col"}:
            inner_pack = batch_per_block
            batch_per_block = 1
            tle_fused_twiddle = True
    if kernel_name.startswith("ix_warp"):
        from .target import ix_warp_fft_pack
        warp_pack = ix_warp_fft_pack()
        physical_warps=max(1,(warp_pack*min(plan.length,64)+63)//64)
        profile.validate(physical_warps)
        exchange_points=plan.length
        if "_chunk" in kernel_name:
            exchange_points=64*int(kernel_name.split("_chunk",1)[1].split("_",1)[0])
        element_bytes=4 if "_c2r_" in kernel_name and "_packed_c2r_" not in kernel_name else 8
        if warp_pack * exchange_points * element_bytes > profile.max_dynamic_shared_memory:
            raise ValueError("IX warp FFT output tile exceeds shared-memory capacity")
        num_warps = physical_warps
        batch_per_block = warp_pack
        inner_pack = 1
        if kernel_type in {'four_step_row','four_step_col','four_step_real_row',
                           'four_step_hermitian_row','four_step_r2c_col','four_step_c2r_col'}:
            batch_per_block=1
            inner_pack=warp_pack
            tle_fused_twiddle=True
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
        "ix_warp_fft_pack": warp_pack,
    }


__all__ = [
    "_arg_signature",
    "_csv_ints",
    "_dtype_suffix",
    "_metadata",
    "_module_source",
    "_pointer_signature",
    "_signature",
    "_zero_other",
]
