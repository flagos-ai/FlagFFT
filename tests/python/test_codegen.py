from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def kernels():
    pytest.importorskip("triton")
    for name in list(sys.modules):
        if name == "src" or name.startswith("src."):
            del sys.modules[name]
    src_spec = importlib.util.spec_from_file_location(
        "src",
        ROOT / "src" / "__init__.py",
        submodule_search_locations=[str(ROOT / "src")],
    )
    assert src_spec is not None and src_spec.loader is not None
    src_module = importlib.util.module_from_spec(src_spec)
    sys.modules["src"] = src_module
    src_spec.loader.exec_module(src_module)

    codegen_spec = importlib.util.spec_from_file_location(
        "src.codegen",
        ROOT / "src" / "codegen" / "__init__.py",
        submodule_search_locations=[str(ROOT / "src" / "codegen")],
    )
    assert codegen_spec is not None
    codegen_module = importlib.util.module_from_spec(codegen_spec)
    sys.modules["src.codegen"] = codegen_module

    kernels_spec = importlib.util.spec_from_file_location(
        "src.codegen.kernels", ROOT / "src" / "codegen" / "kernels.py"
    )
    assert kernels_spec is not None and kernels_spec.loader is not None
    module = importlib.util.module_from_spec(kernels_spec)
    sys.modules["src.codegen.kernels"] = module
    try:
        kernels_spec.loader.exec_module(module)
        return module
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.skip(f"Triton/TLE codegen dependencies are unavailable: {exc}")


def test_codelet_directory_lives_under_codegen(kernels) -> None:
    assert kernels._CODELET_DIR == ROOT / "src" / "codegen" / "codelet"
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


def test_jit_csv_parsing_accepts_empty_and_populated_lists(kernels) -> None:
    pytest.importorskip("triton")
    jit_source_spec = importlib.util.spec_from_file_location(
        "src.codegen.jit_source", ROOT / "src" / "codegen" / "jit_source.py"
    )
    assert jit_source_spec is not None and jit_source_spec.loader is not None
    module = importlib.util.module_from_spec(jit_source_spec)
    sys.modules["src.codegen.jit_source"] = module
    jit_source_spec.loader.exec_module(module)

    assert module._csv_ints("") == ()
    assert module._csv_ints("16,8,4") == (16, 8, 4)


def test_jit_bluestein_source_metadata(tmp_path) -> None:
    pytest.importorskip("triton")
    jit_source_spec = importlib.util.spec_from_file_location(
        "src.codegen.jit_source", ROOT / "src" / "codegen" / "jit_source.py"
    )
    assert jit_source_spec is not None and jit_source_spec.loader is not None
    module = importlib.util.module_from_spec(jit_source_spec)
    sys.modules["src.codegen.jit_source"] = module
    jit_source_spec.loader.exec_module(module)

    metadata = module._emit_bluestein_jit_kernel(
        kernel="bluestein_prepare",
        n=331,
        m=1024,
        out_dir=tmp_path,
    )

    assert metadata["kernel_type"] == "bluestein_prepare"
    assert metadata["arg_names"] == ["in_ptr", "chirp_ptr", "out_ptr", "n", "m", "nbatch"]
    assert metadata["signature"] == "*fp32:16,*fp32:16,*fp32:16,i64,i64,i32"
    assert metadata["bluestein_n"] == 331
    assert metadata["bluestein_m"] == 1024
    assert (tmp_path / "flagfft_jit_bluestein_prepare_n331_m1024_f32.py").is_file()


def test_jit_reshape_pack_source_metadata(tmp_path) -> None:
    pytest.importorskip("triton")
    jit_source_spec = importlib.util.spec_from_file_location(
        "src.codegen.jit_source", ROOT / "src" / "codegen" / "jit_source.py"
    )
    assert jit_source_spec is not None and jit_source_spec.loader is not None
    module = importlib.util.module_from_spec(jit_source_spec)
    sys.modules["src.codegen.jit_source"] = module
    jit_source_spec.loader.exec_module(module)

    reshape = module._emit_reshape_jit_kernel(
        kernel="reshape_pack",
        n1=64,
        n2=128,
        dtype="complex128",
        out_dir=tmp_path,
    )
    twiddle = module._emit_reshape_jit_kernel(
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


def test_jit_r2c_pointwise_source_metadata(tmp_path) -> None:
    pytest.importorskip("triton")
    jit_source_spec = importlib.util.spec_from_file_location(
        "src.codegen.jit_source", ROOT / "src" / "codegen" / "jit_source.py"
    )
    assert jit_source_spec is not None and jit_source_spec.loader is not None
    module = importlib.util.module_from_spec(jit_source_spec)
    sys.modules["src.codegen.jit_source"] = module
    jit_source_spec.loader.exec_module(module)

    expand = module._emit_r2c_pointwise_jit_kernel(
        kernel="real_to_complex",
        n=17,
        dtype="complex64",
        out_dir=tmp_path,
    )
    pack = module._emit_r2c_pointwise_jit_kernel(
        kernel="r2c_half_pack",
        n=17,
        dtype="complex64",
        out_dir=tmp_path,
    )
    expand_inverse = module._emit_r2c_pointwise_jit_kernel(
        kernel="compact_to_hermitian_full",
        n=17,
        dtype="complex128",
        out_dir=tmp_path,
    )
    pack_inverse = module._emit_r2c_pointwise_jit_kernel(
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
