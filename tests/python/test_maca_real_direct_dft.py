# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU execution of generated real DirectDFT source; no GPU or JIT compilation.

The small NumPy TL model checks arithmetic and masked pointer accesses, not
Triton lowering, reassociation, GPU scheduling, or the native runtime gate.
"""

from __future__ import annotations

import ast
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

ABI = ["in_ptr", "out_ptr", "dft_r_ptr", "dft_i_ptr", "nbatch"]
# Include multiple FP32 input tiles in addition to the requested tiny/odd/even
# sizes, so a missing j_base or partial final tile cannot pass unnoticed.
LENGTHS = [1, 2, 22, 23, 24, 33, 128]
DTYPES = ["complex64", "complex128"]
TREE_LENGTHS = [1, 2, 22, 23, 24, 31, 32]
REDUCTION_ENV = "FLAGFFT_MACA_REAL_DFT_REDUCTION"


@pytest.fixture(autouse=True)
def clean_reduction_env(monkeypatch):
    monkeypatch.delenv(REDUCTION_ENV, raising=False)


class Pointer:
    def __init__(self, values, offset=0, writes=None):
        self.values = values
        self.offset = np.asarray(offset)
        self.writes = writes

    def __add__(self, offset):
        return Pointer(self.values, self.offset + offset, self.writes)


class TensorLanguage:
    float32 = np.float32
    float64 = np.float64
    arange = staticmethod(np.arange)
    zeros = staticmethod(np.zeros)
    where = staticmethod(np.where)
    sum = staticmethod(np.sum)
    range = staticmethod(range)
    static_range = staticmethod(range)

    def __init__(self):
        self.batch = 0

    def program_id(self, axis):
        assert axis == 0
        return self.batch

    @staticmethod
    def indices(pointer, mask):
        offsets, valid = np.broadcast_arrays(pointer.offset, np.asarray(mask))
        selected = offsets[valid]
        # Inactive addresses may be outside the allocation, including negative
        # reflected C2R columns; only active lanes may actually access memory.
        assert np.all((selected >= 0) & (selected < pointer.values.size))
        return offsets, valid, selected

    @classmethod
    def load(cls, pointer, mask=True, other=0.0):
        offsets, valid, selected = cls.indices(pointer, mask)
        result = np.full(offsets.shape, other, dtype=pointer.values.dtype)
        result[valid] = pointer.values[selected]
        return result

    @classmethod
    def store(cls, pointer, value, mask=True):
        offsets, valid, selected = cls.indices(pointer, mask)
        assert pointer.writes is not None, "input and tables must be read-only"
        pointer.values[selected] = np.broadcast_to(value, offsets.shape)[valid]
        np.add.at(pointer.writes, selected, 1)


def dft_tables(n, dtype, *, inverse=False):
    # Match tables.cpp: integer products precede the floating-point angle;
    # FP64 uses long-double pi/trig before the final cast to device FP64.
    indices = np.arange(n, dtype=np.int64)
    products = indices[:, None] * indices[None, :]
    sign = 1 if inverse else -1
    if dtype == "complex128":
        pi = np.longdouble("3.141592653589793238462643383279502884")
        angle = np.longdouble(sign) * 2 * pi * products.astype(np.longdouble) / n
        real_dtype = np.float64
    else:
        angle = sign * 2.0 * np.pi * products.astype(np.float64) / n
        real_dtype = np.float32
    return tuple(np.asarray(fn(angle), dtype=real_dtype) for fn in (np.cos, np.sin))


def execute_source(values, n, dtype, *, inverse=False, tables=None):
    from flagfft_codegen.kernels_real import _build_real_direct_dft_kernel_source

    name, source, args = _build_real_direct_dft_kernel_source(n, dtype, inverse=inverse)
    assert args == ABI
    function = ast.parse(source).body[0]
    assert [arg.arg for arg in function.args.args] == ABI
    language = TensorLanguage()
    scope = {"tl": language, "triton": SimpleNamespace(jit=lambda fn: fn)}
    exec(compile(source, "<real_direct_dft_cpu>", "exec"), scope)

    real_dtype = np.float64 if dtype == "complex128" else np.float32
    batch = values.shape[0]
    inputs = np.ascontiguousarray(values, dtype=dtype if inverse else real_dtype)
    inputs = inputs.view(real_dtype).reshape(-1)
    if tables is None:
        # Runtime uploads two full n*n real tables with the transform sign.
        tables = dft_tables(n, dtype, inverse=inverse)
    table_r, table_i = [np.asarray(table, dtype=real_dtype).reshape(-1) for table in tables]
    assert table_r.size == table_i.size == n * n
    size = batch * (n if inverse else 2 * (n // 2 + 1))
    result = np.full(size, np.nan, dtype=real_dtype)
    writes = np.zeros(size, dtype=np.int32)
    pointers = [Pointer(inputs), Pointer(result, writes=writes), Pointer(table_r), Pointer(table_i)]
    # Simulate a rounded-up launch; the extra program must return before access.
    for pid in range(batch + 1):
        language.batch = pid
        scope[name](*pointers, batch)
    np.testing.assert_array_equal(writes, np.ones(size, dtype=np.int32))
    if not inverse:
        result = result.view(dtype)
    return result.reshape(batch, -1)


def assert_fft_close(actual, expected, dtype):
    tolerance = 4e-6 if dtype == "complex64" else 3e-13
    # Relative to signal size, including for the small-amplitude input tests;
    # cancellation bins need an absolute allowance tied to the same scale.
    magnitude = np.max(np.abs(expected))
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance * magnitude)


@pytest.mark.parametrize("n", LENGTHS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
def test_r2c_matches_numpy_and_has_exact_real_endpoints(n, dtype, scale):
    real_dtype = np.float64 if dtype == "complex128" else np.float32
    rng = np.random.default_rng(713 + n)
    values = np.zeros((5, n), dtype=real_dtype)
    values[0] = rng.normal(size=n)
    values[1] = 2.5  # DC
    values[2, 0] = 1.0
    values[3, -1] = -2.0  # Phase/sign and final input position.
    values[4] = (-1.0) ** np.arange(n)
    values *= scale
    actual = execute_source(values, n, dtype)
    assert_fft_close(actual, np.fft.rfft(values, axis=-1), dtype)
    np.testing.assert_array_equal(actual[:, 0].imag, 0)
    if n % 2 == 0:
        np.testing.assert_array_equal(actual[:, -1].imag, 0)


@pytest.mark.parametrize("n", LENGTHS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
def test_c2r_matches_unnormalized_numpy_and_restores_hermitian(n, dtype, scale):
    half = n // 2 + 1
    rng = np.random.default_rng(911 + n)
    values = np.zeros((5, half), dtype=dtype)
    values[0] = rng.normal(size=half) + 1j * rng.normal(size=half)
    values[1, 0] = 2.5
    values[2, -1] = 3j  # Odd last bin is NOT a Nyquist endpoint.
    values[3, -1] = -2.0
    impulse = np.zeros(n)
    impulse[-1] = 1.0
    values[4] = np.fft.rfft(impulse)
    values *= scale
    actual = execute_source(values, n, dtype, inverse=True)
    expected = np.fft.irfft(values, n=n, axis=-1) * n
    assert_fft_close(actual, expected, dtype)


@pytest.mark.parametrize("n", LENGTHS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_c2r_ignores_endpoint_imaginary_even_when_nonfinite(n, dtype):
    rng = np.random.default_rng(n)
    values = (rng.normal(size=(3, n // 2 + 1)) + 1j * rng.normal(size=(3, n // 2 + 1))).astype(dtype)
    clean = values.copy()
    clean[:, 0].imag = 0
    values[:, 0].imag = [1e20, np.nan, np.inf]
    if n % 2 == 0:
        clean[:, -1].imag = 0
        values[:, -1].imag = [-np.inf, -1e20, np.nan]
    expected = execute_source(clean, n, dtype, inverse=True)
    actual = execute_source(values, n, dtype, inverse=True)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("n", [1, 2, 22, 23, 24])
@pytest.mark.parametrize("dtype", DTYPES)
def test_single_batch_roundtrip_has_no_inverse_normalization(n, dtype):
    values = np.random.default_rng(n).normal(size=(1, n))
    half = execute_source(values, n, dtype)
    actual = execute_source(half, n, dtype, inverse=True)
    assert_fft_close(actual, values * n, dtype)


@pytest.mark.parametrize("inverse", [False, True])
def test_fp64_compensation_retains_small_terms(inverse):
    # Exact representable coefficients isolate the accumulator from trig-table
    # approximation. Plain left-to-right summation loses all the unit terms.
    n = 24
    tables = (np.ones((n, n)), np.zeros((n, n)))
    if inverse:
        values = np.ones((1, n // 2 + 1), dtype=np.complex128)
    else:
        values = np.ones((1, n), dtype=np.float64)
    values[0, 0] = 1e16
    actual = execute_source(values, n, "complex128", inverse=inverse, tables=tables)
    np.testing.assert_array_equal(actual.real, np.float64(10**16 + n - 1))


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
def test_source_uses_expected_accumulation_and_table_abi(dtype, inverse):
    from flagfft_codegen.kernels_real import _build_real_direct_dft_kernel_source

    name, source, args = _build_real_direct_dft_kernel_source(23, dtype, inverse=inverse)
    ast.parse(source)
    assert ("direct_dft_c2r" if inverse else "direct_dft_r2c") in name
    assert args == ABI
    if dtype == "complex128":
        assert "tl.range(0, 23)" in source
        assert "comp_r = (next_r - acc_r) - corrected_r" in source
        assert "tl.sum" not in source
        if not inverse:
            assert "comp_i = (next_i - acc_i) - corrected_i" in source
    else:
        assert "tl.static_range(0, 23, 32)" in source
        assert "tl.sum" in source
        assert "comp_r" not in source


@pytest.fixture
def maca_profile():
    from flagfft_codegen.backend_profile import BackendProfile, reset_profile, set_profile
    from flagfft_codegen import target

    profile_token = set_profile(BackendProfile.from_device(
        {"backend": "maca", "warp_size": 64, "device_arch": "102"}, "legacy"
    ))
    target_token = target._target.set("maca:80:64")
    try:
        yield
    finally:
        target._target.reset(target_token)
        reset_profile(profile_token)


def emit_generated_kernel(tmp_path, kernel, direction, dtype, n=23):
    from flagfft_codegen.emit import emit_jit_kernel

    return emit_jit_kernel(
        kernel=kernel, length=n, factors=(), lanes=1, num_warps=2,
        generic_radices=(), smem_size=0, direction=direction, dtype=dtype,
        prime_n=0, four_step_n1=0, four_step_n2=0, out_dir=tmp_path,
    )


def test_codegen_metadata_table_abi_and_distinct_modules(tmp_path, maca_profile):
    entries = []
    for n in [1, 2, 22, 23, 24]:
        for dtype in DTYPES:
            for kind, direction in [
                ("direct_dft_r2c", "forward"), ("direct_dft_c2r", "inverse"),
                ("direct_dft", "forward"), ("direct_dft", "inverse"),
            ]:
                entry = emit_generated_kernel(tmp_path, kind, direction, dtype, n)
                entries.append(entry)
                assert entry["arg_names"] == ABI
                pointer = "*fp64:16" if dtype == "complex128" else "*fp32:16"
                assert entry["signature"] == ",".join([pointer] * 4 + ["i32"])
                assert entry["kernel_type"] == kind
                assert entry["direction"] == direction
                assert entry["dtype"] == dtype
                assert entry["length"] == n
                assert entry["batch_per_block"] == 1
                assert entry["num_warps"] == (2 if dtype == "complex128" else 4)
                path = Path(entry["module_path"])
                tree = ast.parse(path.read_text())
                function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == entry["kernel_name"])
                assert [arg.arg for arg in function.args.args] == ABI
                assert json.loads(path.with_suffix(".json").read_text()) == entry
    assert len({entry["module_path"] for entry in entries}) == len(entries)
    assert len({entry["kernel_name"] for entry in entries}) == len(entries)


@pytest.mark.parametrize("kind,direction", [("direct_dft_r2c", "inverse"), ("direct_dft_c2r", "forward")])
def test_codegen_rejects_wrong_real_direction(tmp_path, maca_profile, kind, direction):
    with pytest.raises(ValueError, match="direction"):
        emit_generated_kernel(tmp_path, kind, direction, "complex64")


@pytest.mark.parametrize("n,dtype", [(0, "complex64"), (129, "complex64"), (23, "float32")])
def test_real_direct_dft_rejects_unsupported_input(n, dtype):
    from flagfft_codegen.kernels_real import _build_real_direct_dft_kernel_source

    with pytest.raises(ValueError):
        _build_real_direct_dft_kernel_source(n, dtype)


@pytest.fixture
def tree_profile(maca_profile, clean_reduction_env, monkeypatch):
    monkeypatch.setenv(REDUCTION_ENV, "tree")


@pytest.mark.parametrize("n", TREE_LENGTHS)
@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
def test_tree_numerical_oracles(tree_profile, n, scale):
    # Reuse the same random/DC/impulse/Hermitian and exact endpoint checks as
    # the default path, including active-address and one-write-per-lane checks.
    test_r2c_matches_numpy_and_has_exact_real_endpoints(n, "complex128", scale)
    test_c2r_matches_unnormalized_numpy_and_restores_hermitian(n, "complex128", scale)


@pytest.mark.parametrize("n", TREE_LENGTHS)
def test_tree_endpoint_nonfinite_and_roundtrip(tree_profile, n):
    test_c2r_ignores_endpoint_imaginary_even_when_nonfinite(n, "complex128")
    test_single_batch_roundtrip_has_no_inverse_normalization(n, "complex128")


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("n", TREE_LENGTHS)
def test_tree_is_explicit_adjacent_balanced_addition(tree_profile, n, inverse):
    from flagfft_codegen.kernels_real import _build_real_direct_dft_kernel_source

    name, source, args = _build_real_direct_dft_kernel_source(n, "complex128", inverse=inverse)
    assert name.endswith("_tree")
    assert args == ABI
    assert "tl.sum" not in source and "tl.range" not in source
    assert "tl.float32" not in source and "comp_r" not in source
    function = ast.parse(source).body[0]
    assert not any(isinstance(node, (ast.For, ast.While, ast.AugAssign)) for node in ast.walk(function))
    assignments = {
        node.targets[0].id: node.value
        for node in function.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    for component in ("r",) if inverse else ("r", "i"):
        leaves = [f"tree_{component}_0_{j}" for j in range(n)]
        assert all(leaf in assignments for leaf in leaves)
        level = 0
        while len(leaves) > 1:
            level += 1
            parents = []
            for j in range(0, len(leaves) - 1, 2):
                parent = f"tree_{component}_{level}_{j // 2}"
                expression = assignments[parent]
                assert isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add)
                assert isinstance(expression.left, ast.Name) and expression.left.id == leaves[j]
                assert isinstance(expression.right, ast.Name) and expression.right.id == leaves[j + 1]
                parents.append(parent)
            leaves = parents + (leaves[-1:] if len(leaves) % 2 else [])
        assert level == (n - 1).bit_length()
        assert sum(key.startswith(f"tree_{component}_") for key in assignments) == 2 * n - 1
        # R2C acc_i is subsequently replaced by the endpoint-zeroing where.
        assert any(
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == f"acc_{component}"
            and isinstance(node.value, ast.Name) and node.value.id == leaves[0]
            for node in function.body
        )


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("dtype,n", [("complex128", 23), ("complex128", 33), ("complex64", 23)])
@pytest.mark.parametrize("backend", ["maca", "cuda"])
def test_reduction_opt_in_scope_and_unchanged_fallback(monkeypatch, maca_profile, inverse, dtype, n, backend):
    from flagfft_codegen import target
    from flagfft_codegen.backend_profile import BackendProfile, reset_profile, set_profile
    from flagfft_codegen.kernels_real import _build_real_direct_dft_kernel_source

    profile_token = set_profile(BackendProfile(backend=backend, device_arch="source-contract"))
    token = target._target.set(f"{backend}:80:{64 if backend == 'maca' else 32}")
    try:
        baseline = _build_real_direct_dft_kernel_source(n, dtype, inverse=inverse)
        for mode in ("", "kahan", "TREE", "unknown", "tree "):
            monkeypatch.setenv(REDUCTION_ENV, mode)
            assert _build_real_direct_dft_kernel_source(n, dtype, inverse=inverse) == baseline
        monkeypatch.setenv(REDUCTION_ENV, "tree")
        result = _build_real_direct_dft_kernel_source(n, dtype, inverse=inverse)
        if backend == "maca" and dtype == "complex128" and n <= 32:
            assert result[0] == baseline[0] + "_tree"
            assert result[1] != baseline[1]
            assert result[2] == baseline[2]
        else:
            assert result == baseline
    finally:
        target._target.reset(token)
        reset_profile(profile_token)


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("n", [22, 23, 31, 32])
def test_tree_fp64_cancellation_error_bound(tree_profile, monkeypatch, n, inverse):
    # Bound only reduction error: fsum receives the already rounded FP64
    # products/differences, independent of trig-table approximation accuracy.
    tables = dft_tables(n, "complex128", inverse=inverse)
    rng = np.random.default_rng(n)
    count = n // 2 + 1 if inverse else n
    real = rng.choice([-1., 1.], (2, count)) * np.exp2(rng.integers(-40, 41, (2, count)))
    real[0, :3] = [1e16, 1., -1e16]
    if inverse:
        values = real.astype(np.complex128)
        values.imag = rng.normal(size=(2, count))
        values[:, 0].imag = 0
        if n % 2 == 0:
            values[:, -1].imag = 0
        tail = values[:, 1:-1] if n % 2 == 0 else values[:, 1:]
        full = np.concatenate([values, tail[:, ::-1].conj()], axis=1)
        terms = [full.real[:, None, :] * tables[0] - full.imag[:, None, :] * tables[1]]
    else:
        values = real
        terms = [real[:, None, :] * table[:n // 2 + 1] for table in tables]
        terms[1][:, 0] = 0
        if n % 2 == 0:
            terms[1][:, -1] = 0

    original_store = TensorLanguage.store

    def fp64_store(pointer, value, mask=True):
        assert np.asarray(value).dtype == np.float64
        return original_store(pointer, value, mask)

    monkeypatch.setattr(TensorLanguage, "store", staticmethod(fp64_store))
    actual = execute_source(values, n, "complex128", inverse=inverse, tables=tables)
    components = [actual] if inverse else [actual.real, actual.imag]
    unit_roundoff = np.finfo(np.float64).eps / 2
    depth = (n - 1).bit_length()
    gamma = depth * unit_roundoff / (1 - depth * unit_roundoff)
    for component, rounded_terms in zip(components, terms):
        for row in range(component.shape[0]):
            for k in range(component.shape[1]):
                summands = rounded_terms[row, k]
                reference = math.fsum(summands)
                bound = gamma * math.fsum(abs(value) for value in summands) + abs(np.spacing(reference))
                assert abs(component[row, k] - reference) <= bound


@pytest.mark.parametrize("kind,direction", [("direct_dft_r2c", "forward"), ("direct_dft_c2r", "inverse")])
def test_tree_metadata_isolates_module_path_and_kernel_name(tmp_path, maca_profile, monkeypatch, kind, direction):
    baseline = emit_generated_kernel(tmp_path, kind, direction, "complex128")
    monkeypatch.setenv(REDUCTION_ENV, "tree")
    tree = emit_generated_kernel(tmp_path, kind, direction, "complex128")
    assert tree["module_path"] != baseline["module_path"]
    assert tree == {**baseline, "kernel_name": baseline["kernel_name"] + "_tree",
                    "module_path": tree["module_path"]}
    source = Path(tree["module_path"]).read_text()
    assert f'def {tree["kernel_name"]}(' in source
    assert json.loads(Path(tree["module_path"]).with_suffix(".json").read_text()) == tree
    assert "_tree(" not in Path(baseline["module_path"]).read_text()


@pytest.mark.parametrize("n", [23, 32])
@pytest.mark.parametrize("inverse", [False, True])
def test_tree_uses_contiguous_k_loads_of_bitwise_symmetric_fp64_tables(tree_profile, n, inverse):
    from flagfft_codegen.kernels_real import _build_real_direct_dft_kernel_source

    # Mirror tables.cpp's FP64 formula: integer k*j, long-double angle/trig,
    # then rounding to double. Transposing indices preserves even signed zero.
    tables = dft_tables(n, "complex128", inverse=inverse)
    for table in tables:
        np.testing.assert_array_equal(table.view(np.uint64), table.T.copy().view(np.uint64))
    _, source, _ = _build_real_direct_dft_kernel_source(n, "complex128", inverse=inverse)
    for component in ("r", "i"):
        assert source.count(f"tl.load(dft_{component}_ptr + j * {n} + k,") == n
        assert f"dft_{component}_ptr + k * {n} + j" not in source
    rng = np.random.default_rng(n)
    if inverse:
        values = rng.normal(size=(2, n // 2 + 1)) + 1j * rng.normal(size=(2, n // 2 + 1))
        expected = np.fft.irfft(values, n=n) * n
    else:
        values = rng.normal(size=(2, n))
        expected = np.fft.rfft(values)
    actual = execute_source(values, n, "complex128", inverse=inverse, tables=tables)
    assert_fft_close(actual, expected, "complex128")
