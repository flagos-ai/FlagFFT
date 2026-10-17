"""Check emitted middle-pass addresses against a batch-preserving transpose."""
import ast

import numpy as np
import pytest

from flagfft_codegen.backend_profile import BackendProfile, reset_profile, set_profile
from flagfft_codegen.kernels_common import LeafPlan
from flagfft_codegen.kernels_leaf import _build_leaf_kernel_source_for_io
from flagfft_codegen.metadata import _arg_signature


@pytest.mark.parametrize("backend", ("musa", "hcu", "ix"))
@pytest.mark.parametrize("io_mode", ("permuted_store", "strided_permuted_store"))
@pytest.mark.parametrize("form", ("inner", "inner_middle", "inner_middle_c2r_cube"))
def test_batched_middle_store_preserves_transform_axis(monkeypatch, backend, io_mode, form):
    monkeypatch.setenv("FLAGFFT_IX_3D_DIRECT_STORE", "1")
    token = set_profile(BackendProfile(
        backend=backend, device_arch="test", warp_size=32 if backend != "hcu" else 64,
        max_threads_per_block=1024, max_dynamic_shared_memory=65536, policy="native",
    ))
    try:
        plan = LeafPlan(length=64, factors=(4, 4, 4), remainder=1, lanes=16,
                        num_warps=1, generic_radices=(), smem_size=64)
        old_name, old_source = _build_leaf_kernel_source_for_io(plan, io_mode=io_mode, perm_form=form)
        name, source = _build_leaf_kernel_source_for_io(plan, io_mode=io_mode, perm_form=form + "_batched")
        assert name != old_name
        old_fn = ast.parse(old_source).body[-1]
        fn = ast.parse(source).body[-1]
        assert "rows_per_transform" not in [a.arg for a in old_fn.args.args]
        assert "rows_per_transform" in [a.arg for a in fn.args.args]
        # Evaluate the actual emitted address expressions with NumPy vectors;
        # the expected result is independently obtained by transposing axes.
        assignments = {
            node.targets[0].id: node.value for node in ast.walk(fn)
            if isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        }
        for batch, n0, span in ((1, 24, 48), (3, 64, 33), (8, 32, 48)):
            row = np.arange(batch * n0 * span)
            expected = np.arange(batch * n0 * span * 64).reshape(batch, 64, span, n0)
            b, i0, i2 = row // (n0 * span), (row // span) % n0, row % span
            for suffix in ("", "_scalar", "_vector"):
                key = "perm_gbase" + suffix
                if key not in assignments:
                    continue
                env = {"rows_per_transform": n0 * span, "perm_span": span,
                       "nbatch": len(row), "perm_outer": n0,
                       "perm_i0" + suffix: row // span, "perm_i1" + suffix: i2}
                base = eval(compile(ast.Expression(assignments[key]), "<emitted>", "eval"), {}, env)
                stride = eval(compile(ast.Expression(assignments["perm_k_stride"]), "<emitted>", "eval"), {}, env)
                for k in (0, 17, 63):
                    np.testing.assert_array_equal(base + k * stride, expected[b, k, i2, i0])
        assert _arg_signature("rows_per_transform", "complex64") == "i64"
    finally:
        reset_profile(token)
