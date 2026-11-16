# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Packed native leaves must preserve tail coverage and storage bounds."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from flagfft_codegen.backend_profile import BackendProfile, reset_profile, set_profile
from flagfft_codegen.emit import emit_jit_kernel


@pytest.mark.parametrize("packing", ["resource", "wave", "two_waves"])
@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("n,factors,lanes", [
    (64, (4, 4, 4), 16), (128, (8, 4, 4), 16),
    (256, (16, 16), 16), (512, (32, 16), 16), (1024, (32, 32), 32),
])
@pytest.mark.parametrize("kind", ["leaf", "leaf_r2c", "leaf_c2r"])
def test_native_pack_covers_partial_batches_with_bounded_storage(
        tmp_path, monkeypatch, packing, dtype, n, factors, lanes, kind):
    token = set_profile(BackendProfile(backend="hcu", device_arch="gfx936", warp_size=64,
                                     max_dynamic_shared_memory=65536, policy="native"))
    try:
        monkeypatch.setenv("FLAGFFT_HCU_LEAF_PACKING", packing)
        metadata = emit_jit_kernel(kernel=kind, length=n, factors=factors, lanes=lanes,
            num_warps=1, generic_radices=(32,) if 32 in factors else (), smem_size=n,
            direction="inverse" if kind == "leaf_c2r" else "forward", dtype=dtype,
            prime_n=0, four_step_n1=0, four_step_n2=0, out_dir=tmp_path)
        source = Path(metadata["module_path"]).read_text()
        compile(source, "<native-leaf-packing>", "exec")
        pack = metadata["batch_per_block"]
        assert 4 * (n + (1 if pack >= 4 else 0)) * pack * (8 if dtype == "complex128" else 4) <= 65536
        assert f"batch_id = pid * {pack}" in source
        for batch in (1, 3, 7, 65):
            rows = [pid * pack + slot for pid in range((batch + pack - 1) // pack)
                    for slot in range(pack) if pid * pack + slot < batch]
            assert rows == list(range(batch))
    finally:
        reset_profile(token)
