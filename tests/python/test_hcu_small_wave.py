# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from flagfft_codegen.backend_profile import BackendProfile, reset_profile, set_profile
from flagfft_codegen.emit import emit_jit_kernel
from flagfft_codegen.kernels_common import LeafPlan
from flagfft_codegen.kernels_hcu_small import small_wave_pack


@pytest.mark.parametrize("n", [16, 32, 64])
@pytest.mark.parametrize("kind,mode,direction", [
    ("leaf", "contiguous", "forward"), ("leaf", "contiguous", "inverse"),
    ("leaf_r2c", "contiguous_r2c", "forward"), ("leaf_c2r", "contiguous_c2r", "inverse"),
])
def test_small_wave_keeps_leaf_distances_and_fills_wave(tmp_path, monkeypatch, n, kind, mode, direction):
    token = set_profile(BackendProfile(backend="hcu", device_arch="gfx936", warp_size=64,
                                     max_dynamic_shared_memory=65536, policy="native"))
    try:
        monkeypatch.setenv("FLAGFFT_HCU_SMALL_WAVE", "1")
        plan = LeafPlan(n, (n,), 1, 1, 1, (), n, dtype="complex64", direction=direction)
        assert small_wave_pack(plan, mode) * n == 64
        metadata = emit_jit_kernel(kernel=kind, length=n, factors=(n,), lanes=1, num_warps=1,
            generic_radices=(n,) if n>16 else (), smem_size=n, direction=direction,
            dtype="complex64", prime_n=0, four_step_n1=0, four_step_n2=0, out_dir=tmp_path)
        source = Path(metadata["module_path"]).read_text()
        compile(source, "<hcu-small-wave>", "exec")
        assert metadata["num_warps"] == 1
        assert metadata["batch_per_block"] * n == 64
        assert "row[:,None]<nbatch" in source
        assert "input_distance" in source and "output_distance" in source
        if mode != "contiguous":
            assert {"input_distance", "output_distance"} <= set(metadata["arg_names"])
        assert "tle.gpu.alloc" not in source
        assert "pid=tl.program_id(0).to(tl.int64)" in source
        assert small_wave_pack(LeafPlan(n, (n,), 1, 1, 1, (), n, dtype="complex128"), mode) == 0
    finally:
        reset_profile(token)
