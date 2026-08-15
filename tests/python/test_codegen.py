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

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))


@pytest.fixture(scope="module")
def kernels():
    pytest.importorskip("triton")
    try:
        return importlib.import_module("flagfft_codegen.kernels")
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.skip(f"Triton/TLE codegen dependencies are unavailable: {exc}")


@pytest.fixture(scope="module")
def jit_source():
    pytest.importorskip("triton")
    try:
        return importlib.import_module("flagfft_codegen.jit_source")
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.skip(f"Triton/TLE codegen dependencies are unavailable: {exc}")


def test_codelet_directory_lives_under_codegen(kernels) -> None:
    assert kernels._CODELET_DIR == ROOT / "python" / "flagfft_codegen" / "codelet"
    assert (kernels._CODELET_DIR / "utils.py").is_file()
    assert (kernels._CODELET_DIR / "radix16.py").is_file()


def test_leaf_kernel_source_generation_uses_plan_fields(kernels) -> None:
    plan = kernels.LeafPlan(
        length=16,
        factors=(4, 4),
        remainder=1,
        lanes=4,
        num_warps=1,
        generic_radices=(),
        smem_size=16,
        direction="forward",
    )

    kernel_name, source = kernels._build_leaf_kernel_source(plan)

    assert kernel_name == "fft_kernel_4_4_l4_b4"
    assert "@triton.jit" in source
    assert f"def {kernel_name}" in source
    assert "nbatch" in source


def test_inverse_leaf_kernel_source_is_directional(kernels) -> None:
    forward = kernels.LeafPlan(
        length=8,
        factors=(8,),
        remainder=1,
        lanes=1,
        num_warps=1,
        generic_radices=(),
        smem_size=0,
        direction="forward",
    )
    inverse = kernels.LeafPlan(
        length=8,
        factors=(8,),
        remainder=1,
        lanes=1,
        num_warps=1,
        generic_radices=(),
        smem_size=0,
        direction="inverse",
    )

    forward_name, forward_source = kernels._build_leaf_kernel_source(forward)
    inverse_name, inverse_source = kernels._build_leaf_kernel_source(inverse)

    assert forward_name != inverse_name
    assert forward_name == "fft_kernel_8_l1_b1"
    assert inverse_name == "ifft_kernel_8_l1_b1"
    assert f"def {forward_name}" in forward_source
    assert f"def {inverse_name}" in inverse_source
    assert forward_source != inverse_source


def test_four_step_inner_pack_threshold(kernels) -> None:
    assert kernels.four_step_col_inner_pack_for(64, 128) == 1
    assert kernels.four_step_col_inner_pack_for(128, 64) == 4
    assert kernels.four_step_col_inner_pack_for(128, 2048, "complex128") == 1
    assert kernels.four_step_col_inner_pack_for(1024, 1024, "complex64") == 4
    assert kernels.four_step_col_inner_pack_for(1024, 1024, "complex128") == 4
    assert kernels.four_step_col_inner_pack_for(512, 2048, "complex64") == 1
    assert kernels.four_step_col_inner_pack_for(495, 2023, "complex64") == 1
    assert kernels.four_step_row_inner_pack_for(1024, 1024, "complex64") == 4
    assert kernels.four_step_row_inner_pack_for(1024, 1024, "complex128") == 1
    assert kernels.four_step_row_inner_pack_for(495, 2023, "complex64") == 4
    assert kernels.four_step_row_inner_pack_for(513, 2023, "complex64") == 1
    assert kernels.four_step_row_inner_pack_for(495, 2023, "complex128") == 1
    assert kernels.use_tle_fused_twiddle(1024, 1024)
    assert kernels.use_tle_fused_twiddle(768, 1024)
    assert kernels.use_tle_fused_twiddle(640, 1024)
    assert kernels.use_tle_fused_twiddle(896, 1024)
    assert not kernels.use_tle_fused_twiddle(768, 1024, "complex128")
    assert not kernels.use_tle_fused_twiddle(512, 2048)


def test_four_step_resource_pack_uses_leaf_occupancy(kernels) -> None:
    row_plan = kernels.LeafPlan(
        length=459,
        factors=(17, 3, 3, 3),
        remainder=1,
        lanes=9,
        num_warps=1,
        generic_radices=(),
        smem_size=512,
        dtype="complex128",
    )
    col_plan = kernels.LeafPlan(
        length=1040,
        factors=(13, 10, 2, 2, 2),
        remainder=1,
        lanes=8,
        num_warps=1,
        generic_radices=(),
        smem_size=2048,
        dtype="complex128",
    )
    large_plan = kernels.LeafPlan(
        length=2375,
        factors=(19, 5, 5, 5),
        remainder=1,
        lanes=25,
        num_warps=1,
        generic_radices=(),
        smem_size=4096,
        dtype="complex128",
    )

    assert kernels.four_step_row_inner_pack_for(459, 1040, "complex128", row_plan) == 4
    assert kernels.four_step_col_inner_pack_for(459, 1040, "complex128", col_plan) == 2
    assert (
        kernels.four_step_col_inner_pack_for(405, 2375, "complex128", large_plan) == 1
    )


def test_low_lane_three_stage_leaf_uses_cooperative_stage_lanes(kernels) -> None:
    plan = kernels.LeafPlan(
        length=780,
        factors=(13, 10, 6),
        remainder=1,
        lanes=2,
        num_warps=1,
        generic_radices=(),
        smem_size=1024,
        direction="forward",
    )

    assert kernels.cooperative_stage_lanes_for(plan) == (60, 78, 65)
    _, source = kernels._build_four_step_row_kernel_source(plan, 780, 850)

    assert "tl.arange(0, 512)" in source
    assert "lane_mask = base_lane_mask & (lane < 60)" in source
    assert "lane_mask = base_lane_mask & (lane < 78)" in source
    assert "lane_mask = base_lane_mask & (lane < 65)" in source
    assert "for group_0 in tl.range(0, 1)" in source
    assert "for group_1 in tl.range(0, 1)" in source
    assert "for group_2 in tl.range(0, 2)" in source


def test_low_lane_four_stage_leaf_uses_cooperative_stage_lanes(kernels) -> None:
    plan = kernels.LeafPlan(
        length=1071,
        factors=(17, 7, 3, 3),
        remainder=1,
        lanes=3,
        num_warps=1,
        generic_radices=(),
        smem_size=2048,
        direction="forward",
    )

    assert kernels.cooperative_stage_lanes_for(plan) == (63, 51, 119, 119)


def test_imbalanced_stage_lanes_keep_fixed_layout(kernels) -> None:
    plan = kernels.LeafPlan(
        length=1573,
        factors=(13, 11, 11),
        remainder=1,
        lanes=11,
        num_warps=1,
        generic_radices=(),
        smem_size=2048,
        direction="forward",
    )

    assert kernels.cooperative_stage_lanes_for(plan) == (11, 11, 11)
    assert kernels.four_step_col_inner_pack_for(544, 1573, "complex64", plan) == 1


def test_double_imbalanced_stage_lanes_prefer_parallel_stages(kernels) -> None:
    plan = kernels.LeafPlan(
        length=1573,
        factors=(13, 11, 11),
        remainder=1,
        lanes=11,
        num_warps=1,
        generic_radices=(),
        smem_size=2048,
        direction="forward",
        dtype="complex128",
    )

    assert kernels.cooperative_stage_lanes_for(plan) == (121, 13, 13)
    assert kernels.four_step_col_inner_pack_for(544, 1573, "complex128", plan) == 2


def test_large_odd_four_step_packs_rows_but_not_columns(kernels) -> None:
    row_plan = kernels.LeafPlan(
        length=495,
        factors=(15, 11, 3),
        remainder=1,
        lanes=3,
        num_warps=1,
        generic_radices=(),
        smem_size=512,
        direction="forward",
    )
    col_plan = kernels.LeafPlan(
        length=2023,
        factors=(17, 17, 7),
        remainder=1,
        lanes=17,
        num_warps=1,
        generic_radices=(),
        smem_size=2048,
        direction="forward",
    )

    _, row_source = kernels._build_four_step_row_kernel_source(row_plan, 495, 2023)
    _, col_source = kernels._build_four_step_col_kernel_source(col_plan, 495, 2023)

    assert "four_step_inner_base = tl.program_id(0) * 4" in row_source
    assert "tl.arange(0, 256)" in row_source
    assert "(four_step_inner < 2023)" in row_source
    assert "four_step_inner = tl.program_id(0)" in col_source
    assert "four_step_inner_base" not in col_source
    assert "tl.arange(0, 32)" in col_source


def test_large_four_step_generates_twiddle_in_row_pipeline(kernels) -> None:
    plan = kernels.LeafPlan(
        length=1024,
        factors=(16, 16, 4),
        remainder=1,
        lanes=64,
        num_warps=2,
        generic_radices=(),
        smem_size=1024,
        direction="forward",
    )

    _, row_source = kernels._build_four_step_row_kernel_source(plan, 1024, 1024)
    _, col_source = kernels._build_four_step_col_kernel_source(plan, 1024, 1024)

    assert "twiddle_ptr" in row_source
    assert "tle.load(twiddle_ptr" not in row_source
    assert "sin.approx.f32" in row_source
    assert "tl.range(0," in row_source
    assert "num_stages=2" not in row_source
    assert "tl.arange(0, 256)" in row_source
    assert row_source.count("tl.debug_barrier()") == 3
    assert "smem_a_r = tle.gpu.alloc" not in row_source
    assert "smem_dst0 = dst0 ^ (dst0 >> 5)" in row_source
    assert "smem_phys0 = logical_phys0 ^ (logical_phys0 >> 5)" in row_source
    assert "twiddle_ptr" not in col_source
    assert "tl.load(in_ptr" in col_source
    assert "tl.arange(0, 256)" in col_source
    assert col_source.count("tl.debug_barrier()") == 3
    assert "smem_a_r = tle.gpu.alloc" not in col_source


def test_double_four_step_loads_twiddle_contiguously_in_row(kernels) -> None:
    plan = kernels.LeafPlan(
        length=459,
        factors=(17, 3, 3, 3),
        remainder=1,
        lanes=9,
        num_warps=1,
        generic_radices=(),
        smem_size=512,
        direction="forward",
        dtype="complex128",
    )

    _, row_source = kernels._build_four_step_row_kernel_source(plan, 459, 1040)

    assert kernels.use_four_step_row_fused_twiddle(459, 1040, "complex128")
    assert "tw_r0 = tl.load(twiddle_ptr + dst_idx0 * 2" in row_source
    assert "tw_i0 = tl.load(twiddle_ptr + dst_idx0 * 2 + 1" in row_source
    assert "sin.approx.f32" not in row_source


def test_rectangular_four_step_fused_twiddle_uses_fft_coordinates(kernels) -> None:
    row_plan = kernels.LeafPlan(
        length=486,
        factors=(9, 9, 6),
        remainder=1,
        lanes=27,
        num_warps=1,
        generic_radices=(),
        smem_size=512,
        direction="forward",
    )
    col_plan = kernels.LeafPlan(
        length=675,
        factors=(15, 15, 3),
        remainder=1,
        lanes=45,
        num_warps=2,
        generic_radices=(),
        smem_size=1024,
        direction="forward",
    )

    _, row_source = kernels._build_four_step_row_kernel_source(row_plan, 486, 675)
    _, col_source = kernels._build_four_step_col_kernel_source(col_plan, 486, 675)

    assert "outer_angle0 = four_step_inner * out_idx0 *" in row_source
    assert "outer_angle0 = dst_idx0 *" not in row_source
    assert "(four_step_inner < 675)" in row_source
    assert "(four_step_inner < 486)" in col_source


def test_2p20_thread_local_radix32_uses_high_register_leaf_and_one_exchange(
    kernels, jit_source, tmp_path
) -> None:
    plan = kernels.LeafPlan(
        length=1024,
        factors=(32, 32),
        remainder=1,
        lanes=32,
        num_warps=2,
        generic_radices=(32,),
        smem_size=1024,
        direction="forward",
    )

    row_name, row_source = kernels._build_four_step_row_kernel_source(plan, 1024, 1024)
    col_name, col_source = kernels._build_four_step_col_kernel_source(plan, 1024, 1024)

    assert "thread_local" in row_name
    assert "thread_local" in col_name
    assert row_source.count(") = _fwd_rad16_b1(") == 4
    assert col_source.count(") = _fwd_rad16_b1(") == 4
    assert "shfl.sync.bfly.b32" not in row_source
    assert "tl.arange(0, 128)" in row_source
    assert row_source.count("tl.debug_barrier()") == 1
    assert row_source.count("tle.gpu.alloc") == 2
    assert "tw1_r_ptr" in row_source
    assert "dft32_r_ptr" in row_source
    assert row_source.count("ld.global.v2.f32") == 32
    assert col_source.count("ld.global.v2.f32") == 32
    assert row_source.count("st.global.v2.f32") == 32
    assert col_source.count("st.global.v2.f32") == 32
    assert "outer_tw_r = tl.load(twiddle_ptr + outer_base_offset" in row_source
    assert "sin.approx.f32" not in row_source
    assert "tle.load(twiddle_ptr" not in row_source
    assert "twiddle_ptr" not in col_source

    metadata = jit_source._metadata(
        module_path=tmp_path / "unused.py",
        kernel_name=row_name,
        arg_names=["in_ptr", "twiddle_ptr", "out_ptr", "nbatch"],
        plan=plan,
        kernel_type="four_step_row",
        n1=1024,
        n2=1024,
        dtype="complex64",
    )
    assert metadata["inner_pack"] == 4
    assert metadata["num_warps"] == 4


def test_packed_real_radix32_uses_factorized_codelet(kernels) -> None:
    plan = kernels.LeafPlan(
        length=1024,
        factors=(32, 32),
        remainder=1,
        lanes=32,
        num_warps=2,
        generic_radices=(32,),
        smem_size=1024,
        direction="forward",
    )

    kernel_name, source = kernels._build_leaf_kernel_source_for_io(
        plan,
        io_mode="four_step_r2c_col",
        four_step_n1=576,
        four_step_n2=1024,
    )

    assert "thread_local" in kernel_name
    assert "lane_vec = tl.arange(0, 128)" in source
    assert source.count(") = _fwd_rad16_b1(") == 4
    assert "tl.load(dft32_r_ptr" not in source
    assert source.count("tl.debug_barrier()") == 1
    assert "compact_mask0 = output_lane_mask &" in source
    assert "four_step_batch * output_distance + dst_idx0" in source


def test_real_row_thread_local_mixed_radix_uses_factorized_codelets(kernels) -> None:
    plan = kernels.LeafPlan(
        length=576,
        factors=(18, 32),
        remainder=1,
        lanes=18,
        num_warps=4,
        generic_radices=(),
        smem_size=576,
        direction="forward",
    )

    kernel_name, source = kernels._build_leaf_kernel_source_for_io(
        plan,
        io_mode="four_step_real_row",
        four_step_n1=576,
        four_step_n2=1024,
    )

    assert "thread_local" in kernel_name
    assert ") = _fwd_rad6_b1(" in source
    assert ") = _fwd_rad3_b1(" in source
    assert ") = _fwd_rad16_b1(" in source
    assert "tl.load(dft18_r_ptr" not in source
    assert "tl.load(dft32_r_ptr" not in source
    assert "four_step_batch * input_distance + src_idx0" in source
    assert "i0 = r0 * 0.0" in source
    assert "outer_base_idx = fft_thread" in source
    assert "outer_tw_r = tl.load(twiddle_ptr + outer_base_offset" in source


def test_thread_local_inverse_real_modes_preserve_compact_io(kernels) -> None:
    row_plan = kernels.LeafPlan(
        length=640,
        factors=(20, 32),
        remainder=1,
        lanes=20,
        num_warps=1,
        generic_radices=(),
        smem_size=1024,
        direction="inverse",
    )
    col_plan = kernels.LeafPlan(
        length=1024,
        factors=(32, 32),
        remainder=1,
        lanes=32,
        num_warps=2,
        generic_radices=(32,),
        smem_size=1024,
        direction="inverse",
    )

    row_name, row_source = kernels._build_leaf_kernel_source_for_io(
        row_plan,
        io_mode="four_step_hermitian_row",
        four_step_n1=640,
        four_step_n2=1024,
    )
    col_name, col_source = kernels._build_leaf_kernel_source_for_io(
        col_plan,
        io_mode="four_step_c2r_col",
        four_step_n1=640,
        four_step_n2=1024,
    )

    assert "thread_local" in row_name
    assert "thread_local" in col_name
    assert "compact_idx0 = tl.where(src_idx0 < 327681" in row_source
    assert "four_step_batch * input_distance + compact_idx0" in row_source
    assert "i0 = tl.where(src_idx0 < 327681, i0, -i0)" in row_source
    assert "outer_base_idx = fft_thread" in row_source
    assert "outer_tw_r = tl.load(twiddle_ptr + outer_base_offset" in row_source
    assert "four_step_batch * output_distance + dst_idx0" in col_source
    assert "tl.store(out_ptr + output_offset0, r0" in col_source
    assert "output_offset0) * 2" not in col_source


def test_two_factor_cooperative_leaf_uses_per_stage_lane_counts(kernels) -> None:
    plan = kernels.LeafPlan(
        length=576,
        factors=(18, 32),
        remainder=1,
        lanes=18,
        num_warps=4,
        generic_radices=(),
        smem_size=576,
        direction="forward",
    )

    assert kernels.cooperative_stage_lanes_for(plan) == (32, 18)

    _, source = kernels._build_leaf_kernel_source_for_io(
        plan,
        io_mode="four_step_real_row",
        four_step_n1=576,
        four_step_n2=2048,
    )
    assert "base_lane_mask = lane_mask" in source
    assert "lane_mask = base_lane_mask & (lane < 32)" in source
    assert "lane_mask = base_lane_mask & (lane < 18)" in source


def test_small_double_mixed_leaf_uses_cooperative_stage_lanes(kernels) -> None:
    plan = kernels.LeafPlan(
        length=323,
        factors=(19, 17),
        remainder=1,
        lanes=1,
        num_warps=1,
        generic_radices=(),
        smem_size=512,
        direction="forward",
        dtype="complex128",
    )

    assert kernels.cooperative_stage_lanes_for(plan) == (17, 19)

    _, source = kernels._build_leaf_kernel_source_for_io(
        plan,
        io_mode="four_step_row",
        four_step_n1=323,
        four_step_n2=330,
    )
    assert "four_step_inner_base = tl.program_id(0) * 4" in source
    assert "lane_vec = tl.arange(0, 128)" in source
    assert "lane_mask = base_lane_mask & (lane < 17)" in source
    assert "lane_mask = base_lane_mask & (lane < 19)" in source


def test_large_double_mixed_leaf_uses_cooperative_stage_lanes(kernels) -> None:
    plan = kernels.LeafPlan(
        length=2375,
        factors=(19, 5, 5, 5),
        remainder=1,
        lanes=25,
        num_warps=1,
        generic_radices=(),
        smem_size=4096,
        direction="forward",
        dtype="complex128",
    )

    assert kernels.cooperative_stage_lanes_for(plan) == (125, 95, 95, 95)


def test_table_codelet_accumulators_follow_register_shape(kernels) -> None:
    source = "\n".join(kernels._emit_table_codelet("    ", 31, 32))

    assert "acc_r_0 = tl.zeros_like(r0)" in source
    assert "acc_i_0 = tl.zeros_like(i0)" in source
    assert "acc_r_0 = tl.zeros((32,)" not in source


def test_double_direct_dft_uses_compensated_accumulation(kernels) -> None:
    _, double_source, _ = kernels._build_direct_dft_kernel_source(
        23, "forward", "complex128"
    )
    _, float_source, _ = kernels._build_direct_dft_kernel_source(
        23, "forward", "complex64"
    )

    assert "comp_r = tl.zeros((32,), dtype=tl.float64)" in double_source
    assert "corrected_r = term_r - comp_r" in double_source
    assert "comp_r = (next_r - acc_r) - corrected_r" in double_source
    assert "comp_r =" not in float_source


def test_strided_leaf_kernel_source_generation(kernels) -> None:
    plan = kernels.LeafPlan(
        length=32,
        factors=(2, 16),
        remainder=1,
        lanes=16,
        num_warps=1,
        generic_radices=(),
        smem_size=32,
        direction="forward",
    )

    kernel_name, source = kernels._build_leaf_kernel_source_for_io(
        plan, io_mode="strided"
    )

    assert kernel_name.startswith("fft_strided_kernel_")
    assert "outer_stride" in source
    assert "batch_index = current_batch // outer_stride" in source
    assert "in0 * outer_stride" in source
    assert "out_idx0 * outer_stride" in source


def test_strided_direct_dft_kernel_source_generation(kernels) -> None:
    kernel_name, source, arg_names = kernels._build_direct_dft_kernel_source(
        23, "forward", "complex64", strided=True
    )

    assert "strided" in kernel_name
    assert arg_names == [
        "in_ptr",
        "out_ptr",
        "dft_r_ptr",
        "dft_i_ptr",
        "outer_stride",
        "nbatch",
    ]
    assert "base + j * outer_stride" in source
    assert "batch_index = pid_batch // outer_stride" in source
    assert "base + k * outer_stride" in source


def test_tiled_transpose_uses_register_transpose(kernels) -> None:
    _, source, _ = kernels._build_tiled_transpose_kernel_source(64, 32, "complex64")

    assert "tl.trans(src_real)" in source
    assert "tl.trans(src_imag)" in source
    assert "safe_col[:, None] * 64 + safe_row[None, :]" in source


def test_strided_four_step_row_kernel_source_generation(kernels) -> None:
    plan = kernels.LeafPlan(
        length=128,
        factors=(8, 4, 4),
        remainder=1,
        lanes=32,
        num_warps=1,
        generic_radices=(),
        smem_size=128,
        direction="forward",
    )

    _, source = kernels._build_leaf_kernel_source_for_io(
        plan,
        io_mode="four_step_row_strided",
        four_step_n1=128,
        four_step_n2=128,
    )

    assert "four_step_batch_index = four_step_batch // outer_stride" in source
    assert "src_idx0 * outer_stride" in source
    assert "dst_idx0 * outer_stride" in source
    assert "twiddle_ptr" not in source


def test_strided_four_step_col_kernel_source_generation(kernels) -> None:
    plan = kernels.LeafPlan(
        length=128,
        factors=(8, 4, 4),
        remainder=1,
        lanes=32,
        num_warps=1,
        generic_radices=(),
        smem_size=128,
        direction="forward",
    )

    _, source = kernels._build_leaf_kernel_source_for_io(
        plan,
        io_mode="four_step_col_strided",
        four_step_n1=128,
        four_step_n2=128,
    )

    assert "four_step_batch_index = four_step_batch // outer_stride" in source
    assert "src_idx0 * outer_stride" in source
    assert "dst_idx0 * outer_stride" in source
    assert "twiddle_ptr + src_idx0 * 2" in source


@pytest.mark.parametrize(
    ("register_radix", "n1", "inner_codelet"),
    [
        (18, 576, "_fwd_rad6_b1"),
        (20, 640, "_fwd_rad5_b1"),
        (24, 768, "_fwd_rad3_b1"),
        (25, 800, "_fwd_rad5_b1"),
        (27, 864, "_fwd_rad9_b1"),
        (28, 896, "_fwd_rad7_b1"),
        (30, 960, "_fwd_rad10_b1"),
    ],
)
def test_large_mixed_thread_local_leaf_is_generated(
    kernels, register_radix, n1, inner_codelet
) -> None:
    plan = kernels.LeafPlan(
        length=n1,
        factors=(register_radix, 32),
        remainder=1,
        lanes=register_radix,
        num_warps=1,
        generic_radices=(),
        smem_size=1024,
        direction="forward",
    )

    row_name, row_source = kernels._build_four_step_row_kernel_source(plan, n1, 1024)

    assert f"kernel_{register_radix}_32_thread_local" in row_name
    assert inner_codelet in row_source
    assert row_source.count(") = _fwd_rad16_b1(") == 2
    assert row_source.count("tl.debug_barrier()") == 1
    assert row_source.count("tle.gpu.alloc") == 2
    assert row_source.count("ld.global.v2.f32") == register_radix
    assert f"inner_tw_idx1 = 1 + {register_radix} * fft_thread" in row_source
    assert "tl.load(tw1_r_ptr + inner_tw_idx1)" in row_source
    assert "outer_tw_r = tl.load(twiddle_ptr + outer_base_offset" in row_source
    assert "sin.approx.f32" not in row_source
    assert "mask=output_lane_mask" in row_source


def test_16384_four_step_keeps_measured_kernel_contract(kernels) -> None:
    row_plan = kernels.LeafPlan(
        length=256,
        factors=(8, 8, 4),
        remainder=1,
        lanes=32,
        num_warps=1,
        generic_radices=(),
        smem_size=256,
        direction="forward",
    )
    col_plan = kernels.LeafPlan(
        length=64,
        factors=(4, 4, 4),
        remainder=1,
        lanes=16,
        num_warps=1,
        generic_radices=(),
        smem_size=64,
        direction="forward",
    )

    _, row_source = kernels._build_four_step_row_kernel_source(row_plan, 256, 64)
    _, col_source = kernels._build_four_step_col_kernel_source(col_plan, 256, 64)

    assert "twiddle_ptr" not in row_source
    assert "num_stages=2" not in row_source
    assert "twiddle_ptr" in col_source


def test_2p20_tle_argument_contract_covers_real_four_step_modes(kernels) -> None:
    plan = kernels.LeafPlan(
        length=1024,
        factors=(16, 16, 4),
        remainder=1,
        lanes=64,
        num_warps=2,
        generic_radices=(),
        smem_size=1024,
        direction="forward",
    )

    row_modes = {
        "four_step_row": (),
        "four_step_real_row": ("input_distance",),
        "four_step_hermitian_row": ("input_distance",),
    }
    col_modes = {
        "four_step_col": (),
        "four_step_r2c_col": ("output_distance",),
        "four_step_c2r_col": ("output_distance",),
    }
    for mode, distance_args in row_modes.items():
        _, source = kernels._build_leaf_kernel_source_for_io(
            plan, io_mode=mode, four_step_n1=1024, four_step_n2=1024
        )
        signature = source.split("):", 1)[0]
        assert signature.index("in_ptr") < signature.index("twiddle_ptr")
        assert signature.index("twiddle_ptr") < signature.index("out_ptr")
        assert all(arg in signature for arg in distance_args)

    for mode, distance_args in col_modes.items():
        _, source = kernels._build_leaf_kernel_source_for_io(
            plan, io_mode=mode, four_step_n1=1024, four_step_n2=1024
        )
        signature = source.split("):", 1)[0]
        assert signature.index("in_ptr") < signature.index("out_ptr")
        assert "twiddle_ptr" not in signature
        assert all(arg in signature for arg in distance_args)


def test_2p20_col_metadata_uses_tle_pipeline_and_eight_warps(
    kernels, jit_source, tmp_path
) -> None:
    plan = kernels.LeafPlan(
        length=1024,
        factors=(16, 16, 4),
        remainder=1,
        lanes=64,
        num_warps=2,
        generic_radices=(),
        smem_size=1024,
        direction="forward",
    )

    metadata = jit_source._metadata(
        module_path=tmp_path / "unused.py",
        kernel_name="unused",
        arg_names=["in_ptr", "out_ptr", "nbatch"],
        plan=plan,
        kernel_type="four_step_col",
        n1=1024,
        n2=1024,
        dtype="complex64",
    )

    assert metadata["num_stages"] == 1
    assert metadata["num_warps"] == 8
    assert metadata["tle_fused_twiddle"] is True


def test_2p20_row_metadata_packs_four_ffts_into_eight_warps(
    kernels, jit_source, tmp_path
) -> None:
    plan = kernels.LeafPlan(
        length=1024,
        factors=(16, 16, 4),
        remainder=1,
        lanes=64,
        num_warps=2,
        generic_radices=(),
        smem_size=1024,
        direction="forward",
    )

    metadata = jit_source._metadata(
        module_path=tmp_path / "unused.py",
        kernel_name="unused",
        arg_names=["in_ptr", "twiddle_ptr", "out_ptr", "nbatch"],
        plan=plan,
        kernel_type="four_step_row",
        n1=1024,
        n2=1024,
        dtype="complex64",
    )

    assert metadata["inner_pack"] == 4
    assert metadata["num_warps"] == 8


def test_jit_csv_parsing_accepts_empty_and_populated_lists(jit_source) -> None:
    assert jit_source._csv_ints("") == ()
    assert jit_source._csv_ints("16,8,4") == (16, 8, 4)


def test_kernel_registry_is_complete_and_consistent() -> None:
    import flagfft_codegen.registry as registry

    assert len(registry.KERNEL_NAMES) == 34
    assert len(set(registry.KERNEL_NAMES)) == 34
    assert set(registry.KERNEL_SPECS) == set(registry.KERNEL_NAMES)

    for name, spec in registry.KERNEL_SPECS.items():
        assert spec.name == name
        if spec.is_leaf_like:
            assert spec.io_mode is not None
            assert {"length", "factors", "lanes", "num_warps", "smem_size"} <= set(
                spec.requires
            )
        if spec.is_four_step:
            assert {"four_step_n1", "four_step_n2"} <= set(spec.requires)


def test_jit_bluestein_source_metadata(jit_source, tmp_path) -> None:
    metadata = jit_source._emit_bluestein_jit_kernel(
        kernel="bluestein_prepare",
        n=331,
        m=1024,
        out_dir=tmp_path,
    )

    assert metadata["kernel_type"] == "bluestein_prepare"
    assert metadata["arg_names"] == [
        "in_ptr",
        "chirp_ptr",
        "out_ptr",
        "n",
        "m",
        "nbatch",
    ]
    assert metadata["signature"] == "*fp32:16,*fp32:16,*fp32:16,i64,i64,i32"
    assert metadata["bluestein_n"] == 331
    assert metadata["bluestein_m"] == 1024
    assert (tmp_path / "flagfft_jit_bluestein_prepare_n331_m1024_f32.py").is_file()


def test_jit_reshape_pack_source_metadata(jit_source, tmp_path) -> None:
    reshape = jit_source._emit_reshape_jit_kernel(
        kernel="reshape_pack",
        n1=64,
        n2=128,
        dtype="complex128",
        out_dir=tmp_path,
    )
    twiddle = jit_source._emit_reshape_jit_kernel(
        kernel="twiddle_reshape_pack",
        n1=128,
        n2=64,
        out_dir=tmp_path,
    )

    assert reshape["kernel_type"] == "reshape_pack"
    assert reshape["signature"] == "*fp64:16,*fp64:16,i32"
    assert reshape["reshape_n1"] == 64
    assert reshape["reshape_n2"] == 128
    assert (tmp_path / "flagfft_jit_reshape_pack_n64_128_f64.py").is_file()

    assert twiddle["kernel_type"] == "twiddle_reshape_pack"
    assert twiddle["arg_names"] == ["in_ptr", "twiddle_ptr", "out_ptr", "nbatch"]
    assert twiddle["signature"] == "*fp32:16,*fp32:16,*fp32:16,i32"
    assert (tmp_path / "flagfft_jit_twiddle_reshape_pack_n128_64_f32.py").is_file()


def test_jit_r2c_pointwise_source_metadata(jit_source, tmp_path) -> None:
    expand = jit_source._emit_r2c_pointwise_jit_kernel(
        kernel="real_to_complex",
        n=17,
        dtype="complex64",
        out_dir=tmp_path,
    )
    pack = jit_source._emit_r2c_pointwise_jit_kernel(
        kernel="r2c_half_pack",
        n=17,
        dtype="complex64",
        out_dir=tmp_path,
    )
    expand_inverse = jit_source._emit_r2c_pointwise_jit_kernel(
        kernel="compact_to_hermitian_full",
        n=17,
        dtype="complex128",
        out_dir=tmp_path,
    )
    pack_inverse = jit_source._emit_r2c_pointwise_jit_kernel(
        kernel="complex_to_real",
        n=17,
        dtype="complex128",
        out_dir=tmp_path,
    )

    assert expand["kernel_type"] == "real_to_complex"
    assert expand["arg_names"] == ["in_ptr", "out_ptr", "input_distance", "nbatch"]
    assert expand["signature"] == "*fp32:16,*fp32:16,i64,i32"
    assert (tmp_path / "flagfft_jit_real_to_complex_n17_f32.py").is_file()

    assert pack["kernel_type"] == "r2c_half_pack"
    assert pack["length"] == 17
    assert pack["arg_names"] == ["in_ptr", "out_ptr", "output_distance", "nbatch"]
    assert pack["signature"] == "*fp32:16,*fp32:16,i64,i32"
    assert (tmp_path / "flagfft_jit_r2c_half_pack_n17_f32.py").is_file()

    assert expand_inverse["kernel_type"] == "compact_to_hermitian_full"
    assert expand_inverse["signature"] == "*fp64:16,*fp64:16,i64,i32"
    assert (tmp_path / "flagfft_jit_compact_to_hermitian_full_n17_f64.py").is_file()

    assert pack_inverse["kernel_type"] == "complex_to_real"
    assert pack_inverse["signature"] == "*fp64:16,*fp64:16,i64,i32"
    assert (tmp_path / "flagfft_jit_complex_to_real_n17_f64.py").is_file()
