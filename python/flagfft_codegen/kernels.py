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

"""Compatibility shim re-exporting the kernel generation submodules.

Direct imports of internal helpers are being migrated to the submodules;
this module is kept so existing callers keep working during the split.
"""

from .kernels_common import *  # noqa: F401,F403
from .kernels_legacy import *  # noqa: F401,F403
from .kernels_leaf import *  # noqa: F401,F403
from .kernels_special import *  # noqa: F401,F403
from .kernels_real import *  # noqa: F401,F403
from .kernels_layout import *  # noqa: F401,F403

__all__ = [
    "LeafIoMode",
    "LeafPlan",
    "SPECIALIZED_INLINE_CODELET_RADICES",
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
    "_bluestein_finalize_kernel",
    "_bluestein_pointwise_kernel",
    "_bluestein_prepare_kernel",
    "_build_compact_to_hermitian_full_kernel_source",
    "_build_complex_to_real_kernel_source",
    "_build_direct_dft_kernel_source",
    "_build_four_step_col_kernel_source",
    "_build_four_step_row_kernel_source",
    "_build_leaf_kernel_source",
    "_build_leaf_kernel_source_for_io",
    "_build_r2c_half_pack_kernel_source",
    "_build_rader_finalize_kernel_source",
    "_build_rader_pointwise_kernel_source",
    "_build_rader_prepare_kernel_source",
    "_build_real_to_complex_kernel_source",
    "_build_reshape_pack_kernel_source",
    "_build_thread_local_mixed_four_step_kernel_source",
    "_build_tiled_transpose3d_kernel_source",
    "_build_tiled_transpose3d_v2_kernel_source",
    "_build_tiled_transpose_kernel_source",
    "_build_twiddle_reshape_pack_kernel_source",
    "_cmul",
    "_direction_sign",
    "_distributed_join_tree",
    "_dtype_suffix",
    "_emit_distributed_split_tree",
    "_emit_inline_constant_codelet",
    "_emit_input_base",
    "_emit_input_index",
    "_emit_local_mixed_codelet_call",
    "_emit_local_radix32_codelet_call",
    "_emit_natural_order_codelet_call",
    "_emit_natural_order_radix32_codelet_call",
    "_emit_output_base",
    "_emit_output_index",
    "_emit_radix16_codelet_call",
    "_emit_route_base",
    "_emit_route_index",
    "_emit_stage_block",
    "_emit_table_codelet",
    "_floor_power_of_two",
    "_fmt_const",
    "_four_step_resource_inner_pack_for",
    "_is_double_dtype",
    "_leaf_kernel_params",
    "_leaf_kernel_params_for_io",
    "_mthreads_backend_active",
    "_next_pow2",
    "_non_nvidia_backend_active",
    "_packed_layout",
    "_ppu_backend_active",
    "_real_element_bytes",
    "_time_major_stride",
    "_tl_real_dtype",
    "_transpose_complex_kernel",
    "_triton_plugin_present",
    "_twiddle_transpose_complex_kernel",
    "_use_single_smem_buffer",
    "_use_thread_local_mixed_leaf",
    "_zero_other",
    "contiguous_batch_pack_for",
    "cooperative_stage_lanes_for",
    "four_step_col_inner_pack_for",
    "four_step_row_inner_pack_for",
    "lane_block_for",
    "use_four_step_row_fused_twiddle",
    "use_tle_fused_twiddle",
]
