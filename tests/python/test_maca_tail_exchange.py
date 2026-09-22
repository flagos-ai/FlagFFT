# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for FLAGFFT_MACA_MIXED_EXCHANGE=direct (default off).

Run with the worktree's python directory on PYTHONPATH.  The NumPy interpreter
executes the generated exchanges, including strict gather bounds, without a
GPU.  MACA compilation and performance remain separate acceptance checks.
"""

from __future__ import annotations

import ast
import math
import re
import textwrap

import numpy as np
import pytest

from test_maca_exchange import TensorLanguage
from flagfft_codegen.backend_profile import BackendProfile, reset_profile, set_profile
from flagfft_codegen.kernels_common import LeafPlan
from flagfft_codegen import kernels_leaf as leaf


FACTORS = [(17, 13, 8), (9, 9, 9), (10, 9, 5), (7, 6, 6, 4)]
BOUNDARIES = [(f, stage) for f in FACTORS for stage in range(len(f) - 1)]


@pytest.fixture
def maca(monkeypatch):
    token = set_profile(
        BackendProfile.from_device(
            {"backend": "maca", "warp_size": 64, "device_arch": "102"}, "legacy"
        )
    )
    monkeypatch.setenv("FLAGFFT_MACA_EXCHANGE", "direct_all")
    monkeypatch.delenv("FLAGFFT_MACA_MIXED_EXCHANGE", raising=False)
    try:
        yield
    finally:
        reset_profile(token)


def _execute(lines, registers):
    scope = {"tl": TensorLanguage, **registers}
    exec(textwrap.dedent("\n".join(lines)), scope)
    return scope


def _layout(factors, pack, inner, padded):
    n = math.prod(factors)
    lane_block = max(128, 1 << (max(n // r for r in factors) - 1).bit_length())
    slot_stride = (1 << (n - 1).bit_length()) + int(padded)
    return dict(
        lane_block=lane_block,
        size=1 << (slot_stride * pack - 1).bit_length(),
        slot_stride=slot_stride,
        pack=pack,
        register_lane_stride=pack if inner else 1,
        register_slot_stride=1 if inner else lane_block,
    )


@pytest.mark.parametrize("factors,stage", BOUNDARIES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("pack", [1, 2, 4])
@pytest.mark.parametrize("inner", [False, True])
@pytest.mark.parametrize("padded", [False, True])
def test_direct_registers_match_two_gathers(
    maca, monkeypatch, factors, stage, dtype, pack, inner, padded
):
    layout = _layout(factors, pack, inner, padded)
    block = layout["lane_block"]
    n = math.prod(factors)
    rng = np.random.default_rng(918 + stage)
    registers = {
        f"exchange_{component}{digit}": rng.normal(size=block * pack).astype(dtype)
        for component in ("r", "i")
        for digit in range(factors[stage])
    }
    # Deliberately leave invalid source lanes nonzero, so a bad source index
    # cannot be hidden by pre-zeroing the input register bank.
    old = _execute(
        leaf._emit_portable_exchange("smem", stage, factors, **layout), registers
    )
    monkeypatch.setenv("FLAGFFT_MACA_MIXED_EXCHANGE", "direct")
    lines = leaf._emit_portable_exchange("smem", stage, factors, **layout)
    new = _execute(lines, registers)
    assert "smem_r" not in new and "smem_i" not in new
    assert sum("tl.gather" in line for line in lines) == 2 * factors[stage + 1]
    assert sum(" = tl.reshape(" in line for line in lines) == 2
    # One lane decode, one joined bank, and three lines per destination digit.
    assert len(lines) <= 16 + 2 * len(factors) + 3 * factors[stage + 1]
    assert len("\n".join(lines)) < 14000
    target = np.arange(block * pack)
    lane = target // pack if inner else target % block
    slot = target % pack if inner else target // block
    valid = lane < n // factors[stage + 1]
    # Simulate an incomplete last batch/inner tile as well as padded lanes.
    load_mask = valid & (slot < max(1, pack - 1))
    for digit in range(factors[stage + 1]):
        index = slot * layout["slot_stride"] + lane + digit * (n // factors[stage + 1])
        index = np.where(valid, index, 0)
        for component in ("r", "i"):
            expected = np.where(valid, old[f"smem_{component}"][index], 0)
            result = new[f"smem_register_{component}{digit}"]
            assert result.dtype == dtype
            np.testing.assert_array_equal(result, expected)
        loaded = _execute(
            leaf._emit_exchange_load(
                "    ", "smem", "unused_index", digit, True, mixed_direct=True
            ),
            {**new, "lane_mask": load_mask},
        )
        for component in ("r", "i"):
            np.testing.assert_array_equal(
                loaded[f"{component}{digit}"],
                np.where(load_mask, old[f"smem_{component}"][index], 0),
            )


@pytest.mark.parametrize("factors", FACTORS + [(16, 8, 8), (17,)])
@pytest.mark.parametrize("pack", [1, 2, 4])
@pytest.mark.parametrize("inner", [False, True])
@pytest.mark.parametrize("padded", [False, True])
def test_natural_order_and_power_two_unchanged(
    maca, monkeypatch, factors, pack, inner, padded
):
    layout = _layout(factors, pack, inner, padded)
    for stage in range(len(factors)):
        natural = stage == len(factors) - 1 or factors in FACTORS
        monkeypatch.delenv("FLAGFFT_MACA_MIXED_EXCHANGE", raising=False)
        old = leaf._emit_portable_exchange(
            "smem", stage, factors, natural_order=natural, **layout
        )
        monkeypatch.setenv("FLAGFFT_MACA_MIXED_EXCHANGE", "direct")
        new = leaf._emit_portable_exchange(
            "smem", stage, factors, natural_order=natural, **layout
        )
        assert old == new


@pytest.mark.parametrize(
    "override",
    [
        {"lane_block": 32},
        {"lane_block": 129},
        {"pack": 3},
        {"slot_stride": 200},
        {"size": 256},
        {"size": 3000},
        {"register_lane_stride": 2, "register_slot_stride": 2},
        {"allow_mixed_direct": False},
    ],
)
def test_unsupported_layout_falls_back(maca, monkeypatch, override):
    layout = {**_layout((9, 9, 9), 4, False, True), **override}
    old = leaf._emit_portable_exchange("smem", 0, (9, 9, 9), **layout)
    monkeypatch.setenv("FLAGFFT_MACA_MIXED_EXCHANGE", "direct")
    assert old == leaf._emit_portable_exchange("smem", 0, (9, 9, 9), **layout)


@pytest.mark.parametrize("value", [None, "", "0", "off", "unknown"])
def test_default_and_unknown_values_off(maca, monkeypatch, value):
    layout = _layout((9, 9, 9), 1, False, False)
    old = leaf._emit_portable_exchange("smem", 0, (9, 9, 9), **layout)
    if value is not None:
        monkeypatch.setenv("FLAGFFT_MACA_MIXED_EXCHANGE", value)
    assert old == leaf._emit_portable_exchange("smem", 0, (9, 9, 9), **layout)


@pytest.mark.parametrize(
    "backend,warp", [("cuda", 32), ("musa", 32), ("hcu", 64), ("npu", 32), ("ppu", 32)]
)
def test_other_backends_unchanged(monkeypatch, backend, warp):
    token = set_profile(
        BackendProfile.from_device(
            {"backend": backend, "warp_size": warp, "device_arch": "80"}, "legacy"
        )
    )
    try:
        plan = LeafPlan(729, (9, 9, 9), 1, 81, 4, (), 1024)
        monkeypatch.delenv("FLAGFFT_MACA_MIXED_EXCHANGE", raising=False)
        old = leaf._build_leaf_kernel_source(plan)
        monkeypatch.setenv("FLAGFFT_MACA_MIXED_EXCHANGE", "direct")
        assert old == leaf._build_leaf_kernel_source(plan)
    finally:
        reset_profile(token)


def _assert_buffers_defined(source):
    tree = ast.parse(source)
    names = sorted(
        (node for node in ast.walk(tree) if isinstance(node, ast.Name)),
        # Assignments read RHS before defining LHS (including same-line aliases).
        key=lambda node: (node.lineno, isinstance(node.ctx, ast.Store)),
    )
    defined = set()
    for node in names:
        if not re.fullmatch(r"smem_[ab]_(?:register_[ri]\d+|[ri])", node.id):
            continue
        if isinstance(node.ctx, ast.Load):
            assert node.id in defined, (node.id, node.lineno)
        else:
            defined.add(node.id)


@pytest.mark.parametrize("exchange", ["", "join", "transpose", "direct", "direct_all"])
def test_opt_in_independent_of_existing_exchange(maca, monkeypatch, exchange):
    monkeypatch.setenv("FLAGFFT_MACA_EXCHANGE", exchange)
    monkeypatch.setenv("FLAGFFT_MACA_MIXED_EXCHANGE", "direct")
    plan = LeafPlan(729, (9, 9, 9), 1, 81, 4, (), 1024)
    _, source = leaf._build_leaf_kernel_source(plan)
    _assert_buffers_defined(source)
    assert "exchange_joined_base" in source
    assert not re.search(r"smem_[ab]_[ri] =", source)


@pytest.mark.parametrize("factors", FACTORS)
@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("pack", [1, 2, 4])
@pytest.mark.parametrize("direction", ["forward", "inverse"])
@pytest.mark.parametrize(
    "io_mode",
    [
        "contiguous",
        "strided",
        "permuted_store",
        "contiguous_r2c",
        "contiguous_c2r",
        "four_step_row",
        "four_step_col",
        "four_step_r2c_col",
        "four_step_c2r_col",
        "bluestein_full_leaf",
    ],
)
def test_leaf_producer_consumer_and_source_size(
    maca, monkeypatch, factors, dtype, pack, direction, io_mode
):
    # Keep resource policy out of this prototype; force each supported packing
    # through the real builder independently of current plan heuristics.
    monkeypatch.setattr(leaf, "contiguous_batch_pack_for", lambda plan: pack)
    monkeypatch.setattr(leaf, "permuted_store_batch_pack_for", lambda plan: pack)
    monkeypatch.setattr(leaf, "four_step_row_inner_pack_for", lambda *args: pack)
    monkeypatch.setattr(leaf, "four_step_col_inner_pack_for", lambda *args: pack)
    monkeypatch.setattr(
        leaf, "_use_thread_local_mixed_leaf", lambda *args, **kwargs: False
    )
    monkeypatch.setattr(leaf, "use_four_step_row_fused_twiddle", lambda *args: False)
    n = math.prod(factors)
    plan = LeafPlan(
        n,
        factors,
        1,
        n // factors[0],
        4,
        (),
        1 << (n - 1).bit_length(),
        direction,
        dtype,
    )
    args = dict(io_mode=io_mode, prime_n=n - 2, four_step_n1=n, four_step_n2=n)
    _, baseline = leaf._build_leaf_kernel_source_for_io(plan, **args)
    monkeypatch.setenv("FLAGFFT_MACA_MIXED_EXCHANGE", "direct")
    _, source = leaf._build_leaf_kernel_source_for_io(plan, **args)
    _assert_buffers_defined(source)
    assert "exchange_joined_base" in source
    assert len(source) < len(baseline) * 1.35
    if io_mode != "bluestein_full_leaf":
        assert not re.search(r"smem_[ab]_[ri] =", source)
    else:
        # The first FFT's natural-order output must remain materialized for
        # pass two, even though every internal mixed boundary is direct.
        assert re.search(r"smem_[ab]_[ri] =", source)


@pytest.mark.parametrize(
    "fused,stage_lanes", [(True, (81, 81, 81)), (False, (27, 27, 27))]
)
def test_stage_layout_fallback_has_no_register_reads(
    maca, monkeypatch, fused, stage_lanes
):
    monkeypatch.setenv("FLAGFFT_MACA_MIXED_EXCHANGE", "direct")
    # Fixed lanes cannot be portable unless each codelet covers a whole stage;
    # use the nonportable path to exercise that explicit boundary fallback.
    portable = not (stage_lanes == (27, 27, 27))
    source = "\n".join(
        leaf._emit_stage_block(
            1,
            (9, 9, 9),
            729,
            27,
            128,
            stage_lanes=stage_lanes,
            portable_exchange=portable,
            exchange_size=1024,
            exchange_slot_stride=1024,
            fuse_twiddle_into_row=fused,
        )
    )
    assert "_register_" not in source
    assert "exchange_joined_base" not in source
