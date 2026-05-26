"""Smoke benchmark tests: quick validation on a small subset of sizes."""

from __future__ import annotations

import pytest

from benchmark.suites import DEFAULT_APIS, DEFAULT_DIRECTIONS, SMOKE_SIZES


@pytest.mark.parametrize("size", SMOKE_SIZES)
@pytest.mark.parametrize("api", DEFAULT_APIS)
@pytest.mark.parametrize("direction", DEFAULT_DIRECTIONS)
def test_bench_smoke(
    invoke_cli,
    bench_warmup,
    bench_iters,
    bench_launches_per_sample,
    size,
    api,
    direction,
):
    """Run a quick benchmark on a small set of sizes to verify the pipeline."""
    result, report = invoke_cli(
        "bench",
        "--api",
        api,
        "--direction",
        direction,
        "--shape",
        str(size),
        "--batch",
        "1",
        "--warmup",
        str(bench_warmup),
        "--iters",
        str(bench_iters),
        "--launches-per-sample",
        str(bench_launches_per_sample),
        "--print-path",
    )
    assert result.returncode == 0, f"Benchmark failed: {report}"
    case = report["cases"][0]
    assert case["correctness"][
        "passed"
    ], f"Correctness check failed: {case['correctness']}"
    assert "speedup" in case["timing"], "Missing speedup in timing"
    assert case["timing"]["flagfft_median_ms"] > 0, "Median time should be positive"
