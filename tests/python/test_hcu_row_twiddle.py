# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Precomputed outer twiddles move once, with the native launch ABI."""

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from flagfft_codegen.backend_profile import BackendProfile, reset_profile, set_profile
from flagfft_codegen.emit import emit_jit_kernel


@pytest.mark.parametrize("n1,n2,factors,lanes", [
    (64, 128, (8, 8), 8), (256, 512, (16, 16), 16), (512, 1024, (32, 16), 16),
])
@pytest.mark.parametrize("direction", ["forward", "inverse"])
def test_twiddle_moves_to_row_store_without_approximate_math(tmp_path, monkeypatch, n1, n2, factors, lanes, direction):
    token = set_profile(BackendProfile(backend="hcu", device_arch="gfx936", warp_size=64,
                                     max_dynamic_shared_memory=65536, policy="native"))
    try:
        monkeypatch.setenv("FLAGFFT_HCU_FP32_ROW_TWIDDLE", "1")
        for kind, n, fs, ls in (("four_step_row", n1, factors, lanes),
                                ("four_step_col", n2, (16, n2 // 16), 16)):
            metadata = emit_jit_kernel(kernel=kind, length=n, factors=fs, lanes=ls,
                num_warps=1, generic_radices=(32,) if 32 in fs else (), smem_size=n,
                direction=direction, dtype="complex64", prime_n=0,
                four_step_n1=n1, four_step_n2=n2, out_dir=tmp_path)
            source = Path(metadata["module_path"]).read_text()
            compile(source, "<hcu-row-twiddle>", "exec")
            assert "sin.approx" not in source and "cos.approx" not in source
            assert ("twiddle_ptr" in metadata["arg_names"]) == (kind == "four_step_row")
            if kind == "four_step_row":
                assignment = next(node for node in ast.walk(ast.parse(source))
                    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "dst_idx0")
                env = {"four_step_inner": np.arange(n2)[:, None], "out_idx0": np.arange(n1)[None, :]}
                indices = eval(compile(ast.Expression(assignment.value), "<emitted-row-index>", "eval"), {}, env)
                np.testing.assert_array_equal(indices, np.arange(n1*n2).reshape(n2, n1))
                assert "twiddle_ptr + dst_idx0 * 2" in source
    finally:
        reset_profile(token)
