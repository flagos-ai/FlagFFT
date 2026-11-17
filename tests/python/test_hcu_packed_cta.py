# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Independent NumPy oracles for the generated packed boundary and its tails."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from flagfft_codegen.backend_profile import BackendProfile, reset_profile, set_profile
from flagfft_codegen.emit import emit_jit_kernel
from flagfft_codegen.emit import _emit_tiled_transpose_jit_kernel
from flagfft_codegen.kernels_common import LeafPlan, cooperative_stage_lanes_for
from test_hcu_convolution_buffer import Pointer, TL, Trace, tensor


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("n,factors,lanes,wave", [(128, (16, 8), 16, False),
    (1024, (16, 8, 8), 128, True)])
def test_resource_selected_transpose_preserves_the_common_leaf_abi(
        tmp_path, monkeypatch, dtype, n, factors, lanes, wave):
    token = set_profile(BackendProfile(backend="hcu", device_arch="gfx936", warp_size=64,
                                     max_dynamic_shared_memory=65536, policy="native"))
    monkeypatch.setenv("FLAGFFT_HCU_PACKED_REAL_CTA", "1")
    monkeypatch.setenv("FLAGFFT_HCU_PACKED_WAVE_LAYOUT", "resource")
    monkeypatch.setenv("FLAGFFT_HCU_PACKED_REAL_STORE", "1")
    monkeypatch.setenv("FLAGFFT_HCU_WAVE_ROOT_TABLE", "1")
    monkeypatch.setenv("FLAGFFT_HCU_LEAF_SINGLE_BUFFER", "1")
    try:
        meta = emit_jit_kernel(kernel="leaf_packed_r2c", length=n, factors=factors,
            lanes=lanes, num_warps=4, generic_radices=(), smem_size=n, direction="forward",
            dtype=dtype, prime_n=0, four_step_n1=0, four_step_n2=0,
            perm_form="outer_packed", out_dir=tmp_path)
    finally:
        reset_profile(token)
    assert meta["kernel_name"].startswith("hcu_warp") == wave
    assert meta["arg_names"][:3] == ["in_ptr", "out_ptr", "packed_twiddle_ptr"]
    assert meta["arg_names"][-3:] == ["input_distance", "output_distance", "nbatch"]


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("n,factors,lanes", [(32, (8, 4), 4),
    (64, (8, 8), 8), (128, (16, 8), 16), (128, (8, 4, 4), 16),
    (256, (16, 16), 16), (512, (8, 8, 8), 64), (1024, (16, 8, 8), 128)])
@pytest.mark.parametrize("span,maximum", [(0, "auto"), (0, "1"), (3, "auto"), (65, "auto")])
@pytest.mark.parametrize("vector", ["0", "fp32"])
def test_generated_packed_cta_matches_numpy_across_matrix_and_cta_tails(
        tmp_path, monkeypatch, dtype, n, factors, lanes, span, maximum, vector):
    token = set_profile(BackendProfile(backend="hcu", device_arch="gfx936", warp_size=64,
                                     max_dynamic_shared_memory=65536, policy="native"))
    monkeypatch.setenv("FLAGFFT_HCU_PACKED_REAL_CTA", "1")
    monkeypatch.setenv("FLAGFFT_HCU_PACKED_CTA_MAX_PACK", maximum)
    monkeypatch.setenv("FLAGFFT_HCU_LEAF_SINGLE_BUFFER", "1")
    monkeypatch.setenv("FLAGFFT_HCU_LEAF_PACKING", "wave")
    monkeypatch.setenv("FLAGFFT_HCU_VECTOR_IO", vector)
    monkeypatch.setenv("FLAGFFT_HCU_DENSE_WAVE", "packed")
    try:
        plan = LeafPlan(n, factors, 1, lanes, 4, (), n, dtype=dtype)
        widths = cooperative_stage_lanes_for(plan)
        meta = emit_jit_kernel(kernel="leaf_packed_r2c", length=n, factors=factors, lanes=lanes,
            num_warps=4, generic_radices=(), smem_size=n, direction="forward", dtype=dtype,
            prime_n=0, four_step_n1=0, four_step_n2=0,
            perm_form="outer_packed" if span else "outer", out_dir=tmp_path)
    finally:
        reset_profile(token)
    scalar = np.float64 if dtype == "complex128" else np.float32
    complex_dtype = np.complex128 if dtype == "complex128" else np.complex64
    rows = 2*span if span else 7
    data = np.random.default_rng(78).normal(size=(rows, 2*n)).astype(scalar)
    output = np.full((rows, n+1), np.nan, dtype=complex_dtype)
    boundary = np.exp(-1j*np.pi*np.arange(n)/n).astype(complex_dtype)
    tables = {}
    prefix = factors[0]
    for stage in range(1, len(factors)):
        width, radix = widths[stage], factors[stage]
        physical = np.arange(n)
        codelet = physical % width+width*((physical//width)//radix)
        angle = -2*np.pi*(codelet % prefix)*((physical//width) % radix)/(prefix*radix)
        tables[f"tw{stage}_r_ptr"] = np.cos(angle).astype(scalar)
        tables[f"tw{stage}_i_ptr"] = np.sin(angle).astype(scalar)
        prefix *= radix
    functions = [node for node in ast.parse(Path(meta["module_path"]).read_text()).body
                 if isinstance(node, ast.FunctionDef)]
    for function in functions:
        function.decorator_list = []
    tree = ast.Module(body=functions, type_ignores=[])
    code = compile(ast.fix_missing_locations(tree), "<hcu-native-packed-cta>", "exec")
    pack = meta["batch_per_block"]
    alias = "    packed_r = smem_b_r" in Path(meta["module_path"]).read_text()
    for pid in range((rows+pack-1)//pack):
        trace = Trace()
        runtime = type("Runtime", (TL,), {"load": staticmethod(trace.load),
            "store": staticmethod(trace.store), "debug_barrier": staticmethod(trace.barrier),
            "program_id": staticmethod(lambda axis: tensor(pid))})
        gpu = SimpleNamespace(alloc=trace.alloc, smem="shared", local_ptr=lambda p, i: p+i[0])
        namespace = {"tl": runtime, "tle": SimpleNamespace(gpu=gpu)}
        exec(code, namespace)
        values = {"in_ptr": data, "out_ptr": output.view(scalar),
                  "packed_twiddle_ptr": boundary.view(scalar), **tables}
        argv = [rows if name == "nbatch" else 2*n if name == "input_distance"
                else span or n+1 if name == "output_distance" else Pointer(values[name])
                for name in meta["arg_names"]]
        namespace[meta["kernel_name"]](*argv)
        if alias:
            assert len(trace.buffers) == 2
            assert all(buffer.overwrites == len(factors)-1 for buffer in trace.buffers)
    expected = np.fft.rfft(data)
    if span:
        expected = expected.reshape(2, span, n+1).transpose(0, 2, 1).reshape(rows, n+1)
    relative = np.linalg.norm(output-expected)/np.linalg.norm(expected)
    assert relative < (2e-14 if dtype == "complex128" else 5e-7)


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("shape", [(17, 33), (65, 129)])
def test_coalesced_transpose_preserves_payload_bits_in_partial_tiles(
        tmp_path, monkeypatch, dtype, shape):
    token = set_profile(BackendProfile(backend="hcu", device_arch="gfx936", warp_size=64,
                                     max_dynamic_shared_memory=65536, policy="native"))
    monkeypatch.setenv("FLAGFFT_HCU_2D_TRANSPOSE", "coalesced")
    try:
        meta = _emit_tiled_transpose_jit_kernel(n0=shape[0], n1=shape[1], dtype=dtype,
                                              tile_size=32, out_dir=tmp_path)
    finally:
        reset_profile(token)
    scalar = np.float64 if dtype == "complex128" else np.float32
    bits = np.uint64 if dtype == "complex128" else np.uint32
    data_bits = np.random.default_rng(409).integers(0, np.iinfo(bits).max,
                            size=(2, *shape, 2), dtype=bits)
    data = data_bits.view(scalar)
    output = np.empty((2, shape[1], shape[0], 2), dtype=scalar)
    functions = [node for node in ast.parse(Path(meta["module_path"]).read_text()).body
                 if isinstance(node, ast.FunctionDef)]
    for function in functions:
        function.decorator_list = []
    code = compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])),
                   "<hcu-coalesced-transpose>", "exec")
    for batch in range(2):
        for row in range((shape[0]+31)//32):
            for col in range((shape[1]+31)//32):
                runtime = type("Runtime", (TL,), {"load": staticmethod(Trace.load),
                    "store": staticmethod(Trace.store),
                    "program_id": staticmethod(lambda axis: tensor((col, row, batch)[axis]))})
                namespace = {"tl": runtime}
                exec(code, namespace)
                namespace[meta["kernel_name"]](Pointer(data), Pointer(output), 2)
    np.testing.assert_array_equal(output.view(bits), data_bits.transpose(0, 2, 1, 3))
