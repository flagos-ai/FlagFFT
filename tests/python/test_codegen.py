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
    assert kernels.four_step_col_inner_pack_for(128, 64) == 2
    assert kernels.four_step_col_inner_pack_for(128, 2048, "complex128") == 1
    assert kernels.four_step_col_inner_pack_for(1024, 1024, "complex64") == 4
    assert kernels.four_step_col_inner_pack_for(1024, 1024, "complex128") == 2
    assert kernels.four_step_col_inner_pack_for(512, 2048, "complex64") == 2
    assert kernels.four_step_row_inner_pack_for(1024, 1024, "complex64") == 4
    assert kernels.four_step_row_inner_pack_for(1024, 1024, "complex128") == 1
    assert kernels.use_tle_fused_twiddle(1024, 1024)
    assert kernels.use_tle_fused_twiddle(768, 1024)
    assert kernels.use_tle_fused_twiddle(640, 1024)
    assert kernels.use_tle_fused_twiddle(896, 1024)
    assert not kernels.use_tle_fused_twiddle(768, 1024, "complex128")
    assert not kernels.use_tle_fused_twiddle(512, 2048)


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
    assert "sin.approx.f32" in row_source
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


@pytest.mark.parametrize(
    ("register_radix", "n1", "inner_codelet"),
    [
        (20, 640, "_fwd_rad5_b1"),
        (24, 768, "_fwd_rad3_b1"),
        (28, 896, "_fwd_rad7_b1"),
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
    assert "tw1_r_ptr +" not in row_source
    assert "sin.approx.f32" in row_source
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
