"""Stockham stage cache identity and specialization regression checks."""

import ast
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))


def test_repeated_radix_stages_have_distinct_modules(tmp_path):
    from flagfft_codegen.emit import emit_jit_kernel

    metadata = []
    for span in (1, 8, 64):
        metadata.append(emit_jit_kernel(kernel="stockham_stage", length=512,
            factors=(8, span), lanes=1, num_warps=4, generic_radices=(),
            smem_size=0, direction="forward", dtype="complex64", prime_n=0,
            four_step_n1=0, four_step_n2=0, out_dir=tmp_path))
    assert len({entry["module_path"] for entry in metadata}) == 3
    assert len({entry["kernel_name"] for entry in metadata}) == 3
    assert all(Path(entry["module_path"]).exists() for entry in metadata)
    # A span is not a codelet dependency (64 is not a natural-order radix).
    for entry in metadata:
        ast.parse(Path(entry["module_path"]).read_text())


@pytest.mark.parametrize("span", [-1, 3, 1024])
def test_reject_invalid_stage_span(span):
    from flagfft_codegen.kernels_stockham import build_stockham_stage

    with pytest.raises(ValueError, match="span"):
        build_stockham_stage(512, 8, "forward", "complex64", span)
