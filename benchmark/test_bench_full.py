"""Full benchmark suite: comprehensive performance testing across all sizes."""

from __future__ import annotations

import json

import pytest

from benchmark.report import generate_json_report, generate_markdown
from benchmark.suites import DEFAULT_APIS, DEFAULT_DIRECTIONS, FULL_SIZES


@pytest.mark.parametrize("size", FULL_SIZES)
@pytest.mark.parametrize("api", DEFAULT_APIS)
@pytest.mark.parametrize("direction", DEFAULT_DIRECTIONS)
def test_bench_full(run_benchmark, size, api, direction):
    """Run full benchmark across all sizes and verify correctness."""
    result, report = run_benchmark(size, api, direction)
    assert (
        result.returncode == 0
    ), f"Benchmark failed for size={size}, api={api}, dir={direction}: {report}"
    case = report["cases"][0]
    assert case["correctness"]["passed"], (
        f"Correctness failed for size={size}: max_abs={case['correctness'].get('max_abs', 'N/A')}, "
        f"rms={case['correctness'].get('rms', 'N/A')}"
    )
    timing = case["timing"]
    assert (
        timing["flagfft_median_ms"] > 0
    ), f"Median time should be positive for size={size}"


@pytest.mark.skip(reason="run separately for reporting")
def test_bench_full_report(invoke_cli, bench_warmup, bench_iters, tmp_path):
    """Aggregate all benchmark results and generate Markdown + JSON reports."""
    results = {"cases": []}

    for size in FULL_SIZES:
        for api in DEFAULT_APIS:
            for direction in DEFAULT_DIRECTIONS:
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
                    "--print-path",
                )
                assert result.returncode == 0
                case = report["cases"][0]
                case["size"] = size
                results["cases"].append(case)

    md_report = generate_markdown(results)
    json_report = generate_json_report(results)

    # Write reports to tmp_path
    md_path = tmp_path / "benchmark_report.md"
    json_path = tmp_path / "benchmark_report.json"
    md_path.write_text(md_report)
    json_path.write_text(json_report)

    # Print reports for CI/terminal output
    print(f"\nMarkdown report: {md_path}")
    print(md_report)
    print(f"\nJSON report: {json_path}")
    print(json_report)

    # Write to stdout for capture
    result_json = json.dumps(results)
    print(f"\nBENCH_JSON:{result_json}")
