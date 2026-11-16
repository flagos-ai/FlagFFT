# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Batched permutation addresses must agree with an independent ND transpose."""

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from flagfft_codegen.backend_profile import BackendProfile, reset_profile, set_profile
from flagfft_codegen.emit import emit_jit_kernel
from flagfft_codegen.kernels_common import LeafPlan
from flagfft_codegen.kernels_hcu_warp import hcu_warp_store_pack


@pytest.fixture(autouse=True)
def hcu_profile(monkeypatch):
    token = set_profile(BackendProfile(backend="hcu", device_arch="gfx936", warp_size=64,
                                     max_dynamic_shared_memory=65536, policy="native"))
    monkeypatch.setenv("FLAGFFT_HCU_3D_WARP_FFT", "1")
    monkeypatch.setenv("FLAGFFT_HCU_WAVE_ROOT_TABLE", "1")
    yield
    reset_profile(token)


@pytest.mark.parametrize("outer", [8, 16, 128])
@pytest.mark.parametrize("batch", [2, 3, 5])
def test_generated_batched_store_matches_numpy_transpose(tmp_path, outer, batch):
    n, span = 64, 64
    rows_per_cube = outer * span
    rows = batch * rows_per_cube
    metadata = emit_jit_kernel(kernel="leaf_permuted_store", length=n, factors=(4, 4, 4),
        lanes=16, num_warps=1, generic_radices=(), smem_size=n, direction="forward",
        dtype="complex64", prime_n=0, four_step_n1=0, four_step_n2=0,
        perm_form="inner_batched", permutation_span=span, permutation_batch=rows, out_dir=tmp_path)
    source = Path(metadata["module_path"]).read_text()
    compile(source, "<hcu-batched-wave>", "exec")
    assert "rows_per_transform" in metadata["arg_names"]
    assert "_batched_leaf" in metadata["kernel_name"]
    pack = metadata["batch_per_block"]
    function = next(v for v in ast.parse(source).body
                    if isinstance(v, ast.FunctionDef) and v.name == metadata["kernel_name"])
    env = {"pid": np.arange(rows // pack, dtype=np.int64)[:, None],
           "slot": np.arange(pack, dtype=np.int64)[None, :],
           "span": span, "nbatch": rows, "rows_per_transform": rows_per_cube}
    # Evaluate the emitted arithmetic rather than reimplementing its index
    # formulas. Compare it with NumPy's independently described axis reorder.
    for node in function.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in {"row_base", "row", "cube", "local_row", "gbase", "stride"}:
                env[name] = eval(compile(ast.Expression(node.value), "<generated-address>", "eval"), {}, env)
    assert env["row"].size == rows
    expected = np.arange(batch * n * rows_per_cube, dtype=np.int64).reshape(batch, n, span, outer)
    expected = expected.transpose(0, 3, 2, 1).reshape(rows, n)
    row = env["row"].ravel()
    actual = env["gbase"].ravel()[:, None] + np.arange(n)[None, :] * env["stride"]
    np.testing.assert_array_equal(actual, expected[row])
    np.testing.assert_array_equal(np.sort(row), np.arange(rows))


@pytest.mark.parametrize("budget,expected", [("64", 8), ("128", 16)])
def test_large_output_tiles_respect_explicit_payload_budget(tmp_path, monkeypatch, budget, expected):
    monkeypatch.setenv("FLAGFFT_HCU_WAVE_SCALAR_BUDGET", budget)
    plan = LeafPlan(2048, (16, 16, 8), 1, 128, 4, (), 2048, dtype="complex64")
    assert hcu_warp_store_pack(plan, "permuted_store", "inner_batched", 64, 32768) == expected
    metadata = emit_jit_kernel(kernel="leaf_permuted_store", length=2048, factors=plan.factors,
        lanes=128, num_warps=4, generic_radices=(), smem_size=2048, direction="inverse",
        dtype="complex64", prime_n=0, four_step_n1=0, four_step_n2=0,
        perm_form="inner_batched", permutation_span=64, permutation_batch=32768, out_dir=tmp_path)
    assert metadata["batch_per_block"] == expected
    assert metadata["num_warps"] == 8
    assert 2048 * expected * 2 <= 8 * 64 * int(budget)


@pytest.mark.parametrize("span", [65, 129, 1025])
@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("dtype,n,factors", [("complex64", 2048, (16, 16, 8)),
                                            ("complex128", 256, (16, 16))])
def test_padded_outer_tiles_match_independent_matrix_transpose(tmp_path, monkeypatch, span, batch, dtype, n, factors):
    monkeypatch.setenv("FLAGFFT_HCU_3D_WARP_FP64", "1")
    rows = batch * span
    metadata = emit_jit_kernel(kernel="leaf_permuted_store", length=n, factors=factors,
        lanes=128 if n == 2048 else 16, num_warps=4, generic_radices=(), smem_size=n,
        direction="forward", dtype=dtype, prime_n=0, four_step_n1=0, four_step_n2=0,
        perm_form="outer", permutation_span=span, permutation_batch=rows, out_dir=tmp_path)
    source = Path(metadata["module_path"]).read_text()
    function = next(v for v in ast.parse(source).body
                    if isinstance(v, ast.FunctionDef) and v.name == metadata["kernel_name"])
    pack = metadata["batch_per_block"]
    assert pack == (8 if dtype == "complex64" else 16)
    groups = (span + pack - 1) // pack
    env = {"pid": np.arange(batch * groups, dtype=np.int64)[:, None],
           "slot": np.arange(pack, dtype=np.int64)[None, :], "span": span, "nbatch": rows}
    for node in function.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in {"groups", "row_base", "row", "gbase"} and not any(
                isinstance(v, ast.Call) for v in ast.walk(node.value)
            ):
                env[name] = eval(compile(ast.Expression(node.value), "<outer-address>", "eval"), {}, env)
    valid = (env["row"] < rows) & ((env["pid"] % groups) * pack + env["slot"] < span)
    row, base = env["row"][valid], env["gbase"][valid]
    np.testing.assert_array_equal(np.sort(row), np.arange(rows))
    assert np.all(env["gbase"][:, 0] % pack == 0)
    frequencies = np.array([0, 1, n // 2, n - 1])
    expected = np.arange(batch * n * span, dtype=np.int64).reshape(batch, n, span)
    actual = base[:, None] + frequencies[None, :] * span
    np.testing.assert_array_equal(actual,
        expected[row[:, None] // span, frequencies[None, :], row[:, None] % span])
