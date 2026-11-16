# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""A final-stage lane remap must mask the remapped FFT rows."""

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from flagfft_codegen.backend_profile import BackendProfile, reset_profile, set_profile
from flagfft_codegen.emit import emit_jit_kernel


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("n,factors,lanes", [(256, (16, 16), 16),
    (1024, (16, 16, 4), 64), (2048, (16, 16, 8), 128)])
@pytest.mark.parametrize("rows", [1, 3, 7, 65, 1025, 4100])
def test_emitted_final_store_never_selects_rows_beyond_the_requested_batch(
        tmp_path, monkeypatch, dtype, n, factors, lanes, rows):
    token = set_profile(BackendProfile(backend="hcu", device_arch="gfx936", warp_size=64,
                                     max_dynamic_shared_memory=65536, policy="native"))
    monkeypatch.setenv("FLAGFFT_HCU_3D_STORE_MODE", "cyclic")
    monkeypatch.setenv("FLAGFFT_HCU_3D_DIRECT_STORE", "1")
    monkeypatch.setenv("FLAGFFT_HCU_DENSE_WAVE", "0")
    monkeypatch.setenv("FLAGFFT_HCU_3D_WARP_FFT", "0")
    try:
        metadata = emit_jit_kernel(kernel="leaf_permuted_store", length=n,
            factors=factors, lanes=lanes, num_warps=4, generic_radices=(), smem_size=n,
            direction="forward", dtype=dtype, prime_n=0, four_step_n1=0, four_step_n2=0,
            perm_form="outer", out_dir=tmp_path)
    finally:
        reset_profile(token)
    fn = next(node for node in ast.parse(Path(metadata["module_path"]).read_text()).body
              if isinstance(node, ast.FunctionDef) and node.name == metadata["kernel_name"])
    pack = metadata["batch_per_block"]
    assert pack > 1
    keys = {"pid", "batch_id", "lane_vec", "batch_slot", "lane", "current_batch",
            "base_lane_mask", "lane_mask", "perm_i0_vector", "perm_i1_vector", "perm_gbase_vector"}
    class TL:
        arange = staticmethod(np.arange)
    env = {"tl": TL, "nbatch": rows, "perm_span": rows}
    for node in fn.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if name not in keys:
            continue
        if name == "pid":
            env[name] = (rows - 1)//pack
        else:
            env[name] = eval(compile(ast.Expression(node.value), "<emitted-tail-mask>", "eval"), {}, env)
    # These are the row IDs selected by the generated output-store mask,
    # after the final-stage thread-to-FFT mapping has changed.
    actual = np.unique(env["current_batch"][env["lane_mask"]])
    expected = np.arange(((rows - 1)//pack)*pack, rows)
    np.testing.assert_array_equal(actual, expected)
