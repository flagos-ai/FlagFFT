# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Bitwise oracle for the two structural 3D transpose routes and partial tiles."""

import ast
from pathlib import Path

import numpy as np
import pytest

from flagfft_codegen.backend_profile import BackendProfile, reset_profile, set_profile
from flagfft_codegen.emit import _emit_tiled_transpose3d_jit_kernel
from test_hcu_convolution_buffer import Pointer, TL, Trace, tensor


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("order", ["021", "210", "201", "120"])
def test_resource_transpose_matches_numpy_bits_across_batches(
        tmp_path, monkeypatch, dtype, order):
    token = set_profile(BackendProfile(backend="hcu", device_arch="gfx936", warp_size=64,
                                     max_dynamic_shared_memory=65536, policy="native"))
    monkeypatch.setenv("FLAGFFT_HCU_3D_TRANSPOSE_MODE", "resource")
    monkeypatch.setenv("FLAGFFT_HCU_3D_TRANSPOSE_TILE", "resource")
    shape, batches = (9, 33, 65), 2
    try:
        meta = _emit_tiled_transpose3d_jit_kernel(n0=shape[0], n1=shape[1], n2=shape[2],
            order=order, dtype=dtype, out_dir=tmp_path)
    finally:
        reset_profile(token)
    scalar = np.float64 if dtype == "complex128" else np.float32
    bits = np.uint64 if dtype == "complex128" else np.uint32
    data_bits = np.random.default_rng(74).integers(0, np.iinfo(bits).max,
                        size=(batches, *shape, 2), dtype=bits)
    data = data_bits.view(scalar)
    axes = tuple(map(int, order))
    output = np.empty((batches, *(shape[axis] for axis in axes), 2), dtype=scalar)
    functions = [node for node in ast.parse(Path(meta["module_path"]).read_text()).body
                 if isinstance(node, ast.FunctionDef)]
    for function in functions:
        function.decorator_list = []
    code = compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])),
                   "<hcu-resource-transpose>", "exec")
    for batch in range(batches):
        for pid in range(meta["grid_x_override"]):
            runtime = type("Runtime", (TL,), {"load": staticmethod(Trace.load),
                "store": staticmethod(Trace.store),
                "program_id": staticmethod(lambda axis: tensor((pid, 0, batch)[axis]))})
            namespace = {"tl": runtime}
            exec(code, namespace)
            namespace[meta["kernel_name"]](Pointer(data), Pointer(output), batches)
    np.testing.assert_array_equal(output.view(bits), data_bits.transpose(0, *(a+1 for a in axes), 4))
