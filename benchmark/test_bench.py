"""Real benchmark cases covering ctest parameter ranges.

Run serially:  pytest benchmark/test_bench.py --flagfft-cli ./build/flagfft-cli
If pytest-xdist is installed, add -p no:xdist to force serial execution.
Suite selection: --bench-suite=smoke|typical|full (default: typical)
"""

from __future__ import annotations

import pytest

from benchmark.utils.defaults import DEFAULTS
from benchmark.utils.suites import get_suite


def pytest_generate_tests(metafunc):
    """Parametrize at collection time, filtered by --bench-suite."""
    suite_name = metafunc.config.getoption("bench_suite")
    if suite_name is None:
        suite_name = DEFAULTS["suite"]
    suite = get_suite(suite_name)

    from benchmark.utils.suites import expand_params

    params = list(expand_params(suite))

    ids = [
        f"{api}-{direction}-{size}-b{batch}" for size, batch, api, direction in params
    ]
    metafunc.parametrize(
        "size,batch,api,direction",
        params,
        ids=ids,
    )


@pytest.mark.bench
def test_bench(size, batch, api, direction, run_benchmark, record_result):
    """Run a single benchmark configuration, record the result."""
    result, report = run_benchmark(size, api, direction, batch)

    case = report["cases"][0]
    case["size"] = size
    case["batch"] = batch
    record_result(case)

    if result.returncode != 0:
        pytest.fail(
            f"Benchmark process failed for {api}/{direction} size={size} batch={batch}"
        )

    timing = case.get("timing", {})
    assert timing.get("flagfft_median_ms", 0) > 0, (
        f"Missing positive FlagFFT timing for {api}/{direction} "
        f"size={size} batch={batch}"
    )
    assert timing.get("ref_median_ms", 0) > 0, (
        f"Missing positive reference timing for {api}/{direction} "
        f"size={size} batch={batch}"
    )
    assert (
        timing.get("speedup", 0) > 0
    ), f"Missing positive speedup for {api}/{direction} size={size} batch={batch}"
