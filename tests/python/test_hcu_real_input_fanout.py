# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Execute reordered wide real-inverse stages against an independent FFT."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from flagfft_codegen.backend_profile import BackendProfile, reset_profile, set_profile
from flagfft_codegen.emit import emit_jit_kernel
from flagfft_codegen.kernels_common import LeafPlan, cooperative_stage_lanes_for
from test_hcu_convolution_buffer import Pointer, TL, Trace, tensor


@pytest.mark.parametrize("n,factors", [(512, (4, 16, 8)), (1024, (8, 16, 8))])
def test_hermitian_input_fanout_preserves_full_inverse_math(tmp_path, monkeypatch, n, factors):
    token = set_profile(BackendProfile(backend="hcu", device_arch="gfx936", warp_size=64,
                                     max_dynamic_shared_memory=65536, policy="native"))
    monkeypatch.setenv("FLAGFFT_HCU_DENSE_WAVE", "packed")
    monkeypatch.setenv("FLAGFFT_HCU_VECTOR_IO", "0")
    monkeypatch.setenv("FLAGFFT_HCU_LEAF_SINGLE_BUFFER", "1")
    monkeypatch.setenv("FLAGFFT_HCU_LEAF_PACKING", "wave")
    try:
        plan = LeafPlan(n, factors, 1, 128, 4, (), n, dtype="complex128", direction="inverse")
        widths = cooperative_stage_lanes_for(plan)
        meta = emit_jit_kernel(kernel="leaf_c2r", length=n, factors=factors, lanes=128,
            num_warps=4, generic_radices=(), smem_size=n, direction="inverse", dtype="complex128",
            prime_n=0, four_step_n1=0, four_step_n2=0, out_dir=tmp_path)
    finally:
        reset_profile(token)
    rows = 3
    real = np.random.default_rng(53).normal(size=(rows, n))
    data = np.fft.rfft(real)
    output = np.full((rows, n), np.nan)
    tables = {}
    prefix = factors[0]
    for stage in range(1, len(factors)):
        width, radix = widths[stage], factors[stage]
        physical = np.arange(n)
        codelet = physical % width + width*((physical//width)//radix)
        angle = 2*np.pi*(codelet % prefix)*((physical//width) % radix)/(prefix*radix)
        tables[f"tw{stage}_r_ptr"], tables[f"tw{stage}_i_ptr"] = np.cos(angle), np.sin(angle)
        prefix *= radix
    functions = [node for node in ast.parse(Path(meta["module_path"]).read_text()).body
                 if isinstance(node, ast.FunctionDef)]
    for function in functions:
        function.decorator_list = []
    code = compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])),
                   "<hcu-real-input-fanout>", "exec")
    pack = meta["batch_per_block"]
    for pid in range((rows+pack-1)//pack):
        trace = Trace()
        runtime = type("Runtime", (TL,), {"load": staticmethod(trace.load),
            "store": staticmethod(trace.store), "debug_barrier": staticmethod(trace.barrier),
            "program_id": staticmethod(lambda axis: tensor(pid))})
        gpu = SimpleNamespace(alloc=trace.alloc, smem="shared", local_ptr=lambda p, i: p+i[0])
        namespace = {"tl": runtime, "tle": SimpleNamespace(gpu=gpu)}
        exec(code, namespace)
        values = {"in_ptr": data.view(np.float64), "out_ptr": output, **tables}
        argv = [rows if name == "nbatch" else n//2+1 if name == "input_distance"
                else n if name == "output_distance" else Pointer(values[name])
                for name in meta["arg_names"]]
        namespace[meta["kernel_name"]](*argv)
    expected = np.fft.irfft(data, n=n)*n
    assert np.linalg.norm(output-expected)/np.linalg.norm(expected) < 2e-14
