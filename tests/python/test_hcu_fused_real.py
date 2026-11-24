# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from flagfft_codegen.backend_profile import BackendProfile, reset_profile, set_profile
from flagfft_codegen.emit import emit_jit_kernel


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("n,factors,lanes", [
    (128, (8, 4, 4), 16), (256, (16, 16), 16),
    (512, (32, 16), 16), (1024, (32, 32), 32),
])
def test_hcu_native_packed_boundary_retains_distances_and_nyquist(tmp_path, monkeypatch, dtype, n, factors, lanes):
    token = set_profile(BackendProfile(backend="hcu", device_arch="gfx936", warp_size=64,
                                     max_dynamic_shared_memory=65536, policy="native"))
    try:
        monkeypatch.setenv("FLAGFFT_HCU_FUSED_PACKED_REAL", "1")
        monkeypatch.setenv("FLAGFFT_HCU_LEAF_SINGLE_BUFFER", "1")
        monkeypatch.setenv("FLAGFFT_HCU_CYCLIC_SKEW", "wave")
        metadata = emit_jit_kernel(
            kernel="leaf_packed_r2c", length=n, factors=factors, lanes=lanes, num_warps=1,
            generic_radices=(32,) if 32 in factors else (), smem_size=n,
            direction="forward", dtype=dtype, prime_n=0, four_step_n1=0,
            four_step_n2=0, out_dir=tmp_path,
        )
        source = Path(metadata["module_path"]).read_text()
        compile(source, "<hcu-fused-packed-r2c>", "exec")
        assert metadata["arg_names"][2] == "packed_twiddle_ptr"
        assert {"input_distance", "output_distance", "nbatch"} <= set(metadata["arg_names"])
        assert "packed_partner" in source and "packed_nyquist" in source
        if metadata["batch_per_block"] > 1:
            assert "packed_index0 = out_idx0 + smem_offset" in source
        assert "packed_r = tle.gpu.alloc" in source
        assert "tl.inline_asm_elementwise" not in source
    finally:
        reset_profile(token)
