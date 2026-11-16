# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Paired I/O must retain masks, dtype and the portable HCU instruction set."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from flagfft_codegen.backend_profile import BackendProfile, reset_profile, set_profile
from flagfft_codegen.kernels_common import LeafPlan
from flagfft_codegen.kernels_leaf import _build_leaf_kernel_source_for_io


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("mode", ["contiguous", "four_step_row", "four_step_col"])
@pytest.mark.parametrize("direction", ["forward", "inverse"])
@pytest.mark.parametrize("setting", ["1", "packed", "fp32"])
def test_pair_io_keeps_complex_components_and_tail_masks(monkeypatch, dtype, mode, direction, setting):
    token = set_profile(BackendProfile(
        backend="hcu", device_arch="gfx936", warp_size=64,
        max_dynamic_shared_memory=65536, policy="native",
    ))
    try:
        monkeypatch.setenv("FLAGFFT_HCU_VECTOR_IO", setting)
        plan = LeafPlan(128, (8, 4, 4), 1, 16, 1, (), 128,
                        dtype=dtype, direction=direction)
        _, source = _build_leaf_kernel_source_for_io(
            plan, io_mode=mode, four_step_n1=128, four_step_n2=96,
        )
        compile(source, "<hcu-paired-io>", "exec")
        if setting == "packed" and dtype == "complex64":
            assert "tl.pointer_type(tl.uint64)" in source
            assert "_packed_pair_" in source and "_packed_store_" in source
            assert "bitcast=True" in source and ">> 32" in source and "<< 32" in source
            assert "tl.split(_pair_" not in source
        elif setting != "1" and dtype == "complex128":
            monkeypatch.setenv("FLAGFFT_HCU_VECTOR_IO", "0")
            _, fallback = _build_leaf_kernel_source_for_io(
                plan, io_mode=mode, four_step_n1=128, four_step_n2=96,
            )
            assert source == fallback
        else:
            assert "_fft_pair_offsets = tl.arange(0, 2)[None, :]" in source
            assert "tl.split(_pair_" in source
            assert "tl.join(" in source
            assert "[:, None]" in source
        assert "mask=" in source
        assert "ld.global" not in source and "st.global" not in source
    finally:
        reset_profile(token)


def test_fp64_pair_layout_reserves_shared_memory_before_packing(tmp_path, monkeypatch):
    from flagfft_codegen.emit import emit_jit_kernel
    token = set_profile(BackendProfile(backend="hcu", device_arch="gfx936", warp_size=64,
                                      max_dynamic_shared_memory=65536, policy="native"))
    monkeypatch.setenv("FLAGFFT_HCU_FOURSTEP_SINGLE_BUFFER", "1")
    try:
        packs = []
        for setting in ("1", "0"):
            monkeypatch.setenv("FLAGFFT_HCU_VECTOR_IO", setting)
            metadata = emit_jit_kernel(kernel="four_step_col", length=1024,
                factors=(32, 32), lanes=32, num_warps=1, generic_radices=(32,),
                smem_size=1024, direction="forward", dtype="complex128", prime_n=0,
                four_step_n1=512, four_step_n2=1024, out_dir=tmp_path / setting)
            packs.append(metadata["inner_pack"])
        assert packs == [2, 4]
        # TLE's two scalar FFT buffers plus a complex layout-exchange tile.
        assert (2 * 1024 + 2 * 32) * packs[0] * 8 <= 65536
    finally:
        reset_profile(token)
