"""Quick verification of the benchmark CLI tool.

Runs ALL size/batch/API/direction combinations with warm=1 iter=1.
Full assertions on exit code, correctness, and timing fields.
Generates console report + optional CSV on session finish.
"""

from __future__ import annotations

import pytest

from benchmark.utils.suites import ALL_BATCHES, ALL_SIZES, API_DIRECTIONS

ALL_APIS = list(API_DIRECTIONS.keys())


def pytest_generate_tests(metafunc):
    """Parametrize over the full ctest grid (no suite filtering)."""
    params = []
    for size in ALL_SIZES:
        for batch in ALL_BATCHES:
            for api in ALL_APIS:
                for direction in API_DIRECTIONS[api]:
                    params.append((size, batch, api, direction))

    ids = [
        f"{api}-{direction}-{size}-b{batch}" for size, batch, api, direction in params
    ]
    metafunc.parametrize("size,batch,api,direction", params, ids=ids)


@pytest.mark.bench
def test_bench_cli(size, batch, api, direction, invoke_cli, record_result):
    """Verify the bench subcommand works correctly (warm=1, iter=1)."""
    result, report = invoke_cli(
        "bench",
        "--api",
        api,
        "--direction",
        direction,
        "--shape",
        str(size),
        "--batch",
        str(batch),
        "--warmup",
        "1",
        "--iters",
        "1",
        "--print-path",
    )

    case = report["cases"][0]
    case["size"] = size
    case["batch"] = batch
    record_result(case)

    assert (
        result.returncode == 0
    ), f"Bench CLI failed for {api}/{direction} size={size} batch={batch}"
    assert case["correctness"]["passed"], (
        f"Correctness failed for {api}/{direction} size={size} batch={batch}: "
        f"max_abs={case['correctness'].get('max_abs', 'N/A')}, "
        f"rms={case['correctness'].get('rms', 'N/A')}"
    )
    assert (
        "speedup" in case["timing"]
    ), f"Missing speedup in timing for {api}/{direction} size={size} batch={batch}"
    assert (
        case["timing"]["flagfft_median_ms"] > 0
    ), f"Median time should be positive for {api}/{direction} size={size} batch={batch}"
