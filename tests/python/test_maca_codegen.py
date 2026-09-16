# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Opt-in numerical checks for the MetaX portable leaf implementation.

Run with FLAGFFT_TEST_MACA=1 and a single physical device filtered before
starting Python. These tests generate modules in memory and write no source.
"""
from __future__ import annotations

import linecache
import math
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
pytestmark = pytest.mark.skipif(os.environ.get("FLAGFFT_TEST_MACA") != "1",
                                reason="requires explicit MetaX GPU test opt-in")


@pytest.fixture(scope="module")
def runtime():
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    triton = pytest.importorskip("triton")
    assert torch.cuda.device_count() == 1, "filter a single physical device before running"
    torch.cuda.set_device(0)
    assert triton.runtime.driver.active.get_current_target().backend == "maca"
    from flagfft_codegen.target import set_codegen_target
    set_codegen_target("maca:80:64")
    yield np, torch
    set_codegen_target("")


@pytest.mark.parametrize("n,factors,lanes", [
    (8, (8,), 1), (15, (15,), 1), (16, (16,), 1), (32, (32,), 1),
    (64, (2, 2, 2, 2, 2, 2), 32),
    (256, (2, 2, 2, 2, 2, 2, 2, 2), 128),
    (780, (13, 10, 6), 2),
])
@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("direction", ["forward", "inverse"])
@pytest.mark.parametrize("batch", [1, 2, 7, 16, 256])
def test_leaf_matches_numpy(runtime, n, factors, lanes, dtype, direction, batch):
    np, torch = runtime
    from flagfft_codegen.kernels_common import (
        LeafPlan, codelet_radices_for, contiguous_batch_pack_for,
        emitted_leaf_factors, lane_block_for,
    )
    from flagfft_codegen.kernels_leaf import _build_leaf_kernel_source
    from flagfft_codegen.metadata import _module_source

    plan = LeafPlan(n, factors, 1, lanes, 4, (), lane_block_for(n), direction, dtype)
    name, source = _build_leaf_kernel_source(plan)
    radices = codelet_radices_for(factors) | codelet_radices_for(emitted_leaf_factors(plan))
    module = _module_source(source, tuple(sorted(radices)))
    filename = f"<maca_leaf_{n}_{dtype}_{direction}>"
    linecache.cache[filename] = (len(module), None, module.splitlines(True), filename)
    scope = {"__name__": __name__}
    exec(compile(module, filename, "exec"), scope)

    rng = np.random.default_rng(1234)
    x = (rng.normal(size=(batch, n)) + 1j * rng.normal(size=(batch, n))).astype(dtype)
    input_tensor = torch.from_numpy(x).cuda()
    output = torch.empty_like(input_tensor)
    args = [torch.view_as_real(input_tensor), torch.view_as_real(output)]
    real_dtype = torch.float64 if dtype == "complex128" else torch.float32
    for stage in range(1, len(factors)):
        prefix = math.prod(factors[:stage])
        count = n // factors[stage]
        angle = ((1 if direction == "inverse" else -1) * 2 * np.pi *
                 np.arange(factors[stage])[:, None] * (np.arange(count)[None, :] % prefix) /
                 (prefix * factors[stage]))
        for table in (np.cos(angle), np.sin(angle)):
            args.append(torch.tensor(table.reshape(-1), dtype=real_dtype, device="cuda"))
    args.append(batch)
    scope[name][(math.ceil(batch / contiguous_batch_pack_for(plan)),)](*args, num_warps=4)
    expected = np.fft.fft(x, axis=-1) if direction == "forward" else np.fft.ifft(x, axis=-1) * n
    # A global norm avoids treating cancellation near zero as relative error.
    relative_error = np.linalg.norm((output.cpu().numpy() - expected).ravel()) / np.linalg.norm(expected.ravel())
    assert relative_error < (2e-6 if dtype == "complex64" else 2e-12)
