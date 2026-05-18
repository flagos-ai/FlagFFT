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


def test_aot_csv_parsing_accepts_empty_and_populated_lists(kernels) -> None:
    pytest.importorskip("triton")
    triton_aot_spec = importlib.util.spec_from_file_location(
        "src.codegen.triton_aot", ROOT / "src" / "codegen" / "triton_aot.py"
    )
    assert triton_aot_spec is not None and triton_aot_spec.loader is not None
    module = importlib.util.module_from_spec(triton_aot_spec)
    sys.modules["src.codegen.triton_aot"] = module
    triton_aot_spec.loader.exec_module(module)

    assert module._csv_ints("") == ()
    assert module._csv_ints("16,8,4") == (16, 8, 4)
