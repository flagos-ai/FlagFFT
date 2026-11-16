# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Run emitted strided-input FFTs against independent NumPy axis transforms."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from flagfft_codegen.backend_profile import BackendProfile, reset_profile, set_profile
from flagfft_codegen.emit import emit_jit_kernel
from flagfft_codegen.kernels_common import LeafPlan
from flagfft_codegen.kernels_hcu_warp import hcu_warp_store_pack
from test_hcu_wide_lanes import NumpyTL, Pointer


@pytest.mark.parametrize("n,factors,lanes", [(64, (4, 4, 4), 16), (256, (16, 16), 16),
                                          (512, (32, 16), 16), (2048, (16, 16, 8), 128)])
@pytest.mark.parametrize("span", [17, 33, 65])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("tile", ["balanced", "output"])
def test_generated_strided_input_and_batched_store_match_numpy(
        tmp_path, monkeypatch, n, factors, lanes, span, batch, inverse, tile):
    monkeypatch.setenv("FLAGFFT_HCU_3D_WARP_FFT", "1")
    monkeypatch.setenv("FLAGFFT_HCU_WAVE_ROOT_TABLE", "1")
    monkeypatch.setenv("FLAGFFT_HCU_WAVE_STRIDED_INPUT", "1")
    monkeypatch.setenv("FLAGFFT_HCU_WAVE_STRIDED_TILE", tile)
    token = set_profile(BackendProfile(backend="hcu", device_arch="gfx936", warp_size=64,
                                     max_dynamic_shared_memory=65536, policy="native"))
    outer = 3  # A CTA can cross both a span and a cube boundary.
    rows = batch * outer * span
    try:
        metadata = emit_jit_kernel(kernel="leaf_strided_permuted_store", length=n,
            factors=factors, lanes=lanes, num_warps=1, generic_radices=(32,) if 32 in factors else (),
            smem_size=n, direction="inverse" if inverse else "forward", dtype="complex64",
            prime_n=0, four_step_n1=0, four_step_n2=0,
            perm_form="inner_batched" if batch > 1 else "inner",
            permutation_span=span, permutation_batch=rows, out_dir=tmp_path)
    finally:
        reset_profile(token)
    assert "_stridedinput" in metadata["kernel_name"]
    source = Path(metadata["module_path"]).read_text()
    # Run only this generated function; imports and JIT decorators belong to
    # the device compiler. NumpyTL evaluates its real loads, math and stores.
    begin = source.index("def " + metadata["kernel_name"] + "(")
    source = source[begin:].replace("tl.program_id(0).to(tl.int64)", "tl.program_id(0)")
    rng = np.random.default_rng(716)
    data = (rng.normal(size=(batch, outer, n, span)) +
            1j*rng.normal(size=(batch, outer, n, span))).astype(np.complex64)
    transformed = np.fft.ifft(data, axis=2)*n if inverse else np.fft.fft(data, axis=2)
    expected = transformed.transpose(0, 2, 3, 1)
    output = np.full(expected.shape, np.nan, dtype=np.complex64)
    roots = np.exp((1j if inverse else -1j)*2*np.pi*np.arange(n)/n).astype(np.complex64)
    real_table = Pointer(np.concatenate((np.zeros(n, np.float32), roots.real)))
    imag_table = Pointer(np.concatenate((np.zeros(n, np.float32), roots.imag)))
    namespace = {"tl": NumpyTL}
    exec(compile(source, "<emitted-strided-wave>", "exec"), namespace)
    args = {}
    for name in metadata["arg_names"]:
        args[name] = imag_table if "_i_ptr" in name else real_table
    values = dict(in_ptr=Pointer(data.view(np.float32)), out_ptr=Pointer(output.view(np.float32)),
                  outer_stride=span, perm_span=span, nbatch=rows)
    args.update({name: value for name, value in values.items() if name in args})
    if "rows_per_transform" in metadata["arg_names"]:
        args["rows_per_transform"] = outer*span
    pack = metadata["batch_per_block"]
    for pid in range(metadata["grid_x_override"]):
        monkeypatch.setattr(NumpyTL, "pid", pid)
        namespace[metadata["kernel_name"]](**args)
    np.testing.assert_allclose(output, expected, rtol=2e-5, atol=2e-4)


def test_strided_wave_requires_opt_in_and_preserves_other_io_fallbacks(monkeypatch):
    token = set_profile(BackendProfile(backend="hcu", device_arch="gfx936", warp_size=64,
                                     max_dynamic_shared_memory=65536, policy="native"))
    monkeypatch.setenv("FLAGFFT_HCU_3D_WARP_FFT", "1")
    monkeypatch.setenv("FLAGFFT_HCU_WAVE_ROOT_TABLE", "1")
    plan = LeafPlan(256, (16, 16), 1, 16, 1, (), 256)
    try:
        assert hcu_warp_store_pack(plan, "strided_permuted_store", "inner", 33, 99) == 0
        monkeypatch.setenv("FLAGFFT_HCU_WAVE_STRIDED_INPUT", "1")
        assert hcu_warp_store_pack(plan, "strided_permuted_store", "inner", 33, 99) > 0
        assert hcu_warp_store_pack(plan, "strided_permuted_store", "outer", 33, 99) == 0
    finally:
        reset_profile(token)
