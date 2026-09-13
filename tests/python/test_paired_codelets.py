# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Check emitted arithmetic against an independent DFT and bound its use."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))


@pytest.mark.parametrize("radix", [11, 13, 17, 19])
@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_paired_codelet_matches_dft(radix, scale, dtype):
    from flagfft_codegen.paired_codelets import paired_codelet_source

    namespace = {}
    exec(paired_codelet_source(radix).replace("@triton.jit\n", ""), namespace)
    kernel = namespace[f"_fwd_rad{radix}_b1"]
    rng = np.random.default_rng(20260913)
    x = (
        np.concatenate(
            (
                np.eye(radix),
                1j * np.eye(radix),
                np.ones((1, radix)),
                rng.normal(size=(32, radix)) + 1j * rng.normal(size=(32, radix)),
            )
        )
        * scale
    )

    x = x.astype(dtype)
    tolerance = 3e-7 if dtype == np.complex64 else 2e-15

    def transform(values):
        result = np.asarray(kernel(*values.real.T, *values.imag.T))
        return (result[:radix] + 1j * result[radix:]).T

    expected = np.fft.fft(x, axis=1)
    actual = transform(x)
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < tolerance
    # Existing inverse codelets swap real and imaginary inputs/outputs.
    swapped = transform(actual.imag + 1j * actual.real)
    inverse = swapped.imag + 1j * swapped.real
    assert np.linalg.norm(inverse - radix * x) / np.linalg.norm(radix * x) < tolerance


@pytest.mark.parametrize("radix", [11, 13, 17, 19])
def test_bundled_codelets_use_paired_arithmetic(radix):
    from flagfft_codegen.paired_codelets import paired_codelet_source

    source = (
        Path(__file__).resolve().parents[2]
        / "python"
        / "flagfft_codegen"
        / "codelet"
        / f"radix{radix}.py"
    ).read_text()
    assert source[source.index("@triton.jit") :] == paired_codelet_source(radix)


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize(
    "kernel",
    [
        "leaf",
        "four_step_row",
        "four_step_row_strided",
        "bluestein_four_step_prepare_row",
    ],
)
def test_paired_codelets_are_used_for_all_kernel_kinds(tmp_path, dtype, kernel):
    from flagfft_codegen import emit

    metadata = emit.emit_jit_kernel(
        kernel=kernel,
        length=143,
        factors=(13, 11),
        lanes=1,
        num_warps=1,
        generic_radices=(),
        smem_size=256,
        direction="forward",
        dtype=dtype,
        prime_n=67,
        four_step_n1=143,
        four_step_n2=221,
        out_dir=tmp_path,
    )
    source = Path(metadata["module_path"]).read_text()
    assert "    c1r = r0" in source
