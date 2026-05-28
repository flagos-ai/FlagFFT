"""Benchmark report generation: console tables, CSV, JSON, and Markdown output."""

from __future__ import annotations

import csv
import io
import json
import math
from typing import Any


def _build_table(records: list[dict[str, Any]]) -> str:
    """Build a markdown table from benchmark records."""
    if not records:
        return "_No benchmark results._\n"

    headers = [
        "Size",
        "API",
        "Direction",
        "Batch",
        "FlagFFT (ms)",
        "cuFFT (ms)",
        "Speedup",
        "Correctness",
    ]
    rows = [headers]

    for r in records:
        correctness = r.get("correctness", {})
        passed = correctness.get("passed", False)
        timing = r.get("timing", {})
        rows.append(
            [
                str(r.get("size", "")),
                str(r.get("api", "")),
                str(r.get("direction", "")),
                str(r.get("batch", "")),
                f"{timing.get('flagfft_median_ms', 0):.4f}",
                f"{timing.get('cufft_median_ms', 0):.4f}",
                f"{timing.get('speedup', 0):.2f}x",
                "PASS" if passed else "FAIL",
            ]
        )

    widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]
    lines = []
    for ri, row in enumerate(rows):
        line = (
            "| "
            + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
            + " |"
        )
        lines.append(line)
        if ri == 0:
            lines.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")

    return "\n".join(lines) + "\n"


def _geometric_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    log_sum = sum(math.log(max(v, 1e-12)) for v in values)
    return math.exp(log_sum / len(values))


def _compute_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract summary statistics from benchmark records."""
    passed = sum(1 for r in records if r.get("correctness", {}).get("passed", False))
    total = len(records)
    timings = [r.get("timing", {}) for r in records]
    flagfft_ms = [
        t.get("flagfft_median_ms", 0)
        for t in timings
        if t.get("flagfft_median_ms", 0) > 0
    ]
    speedups = [t.get("speedup", 0) for t in timings if t.get("speedup", 0) > 0]
    return {
        "passed": passed,
        "total": total,
        "flagfft_ms": flagfft_ms,
        "speedups": speedups,
    }


def generate_console_table(
    records: list[dict[str, Any]],
    suite: str,
    warmup: int,
    iters: int,
) -> str:
    """Generate a console-printable benchmark report with table and summary."""
    stats = _compute_stats(records)
    passed, total = stats["passed"], stats["total"]

    lines = [
        "=" * 84,
        (
            f"FlagFFT Benchmark Report  |  Suite: {suite}  |  Warmup: {warmup}"
            f"  |  Iters: {iters}  |  Passed: {passed}/{total}"
        ),
        "",
        _build_table(records),
    ]

    if records:
        lines.append(f"Correctness: {passed}/{total} passed")
        if stats["flagfft_ms"]:
            lines.append(
                f"FlagFFT median range: {min(stats['flagfft_ms']):.4f}"
                f" - {max(stats['flagfft_ms']):.4f} ms"
            )
            lines.append(
                f"Geometric mean: {_geometric_mean(stats['flagfft_ms']):.4f} ms"
            )
        if stats["speedups"]:
            lines.append(
                f"Overall speedup (geomean):"
                f" {_geometric_mean(stats['speedups']):.2f}x"
            )

    return "\n".join(lines) + "\n"


CSV_COLUMNS = [
    "size",
    "api",
    "direction",
    "batch",
    "backend",
    "flagfft_median_ms",
    "cufft_median_ms",
    "speedup",
    "correctness_passed",
    "max_abs",
    "rms",
    "warmup",
    "iters",
]


def generate_csv(records: list[dict[str, Any]]) -> str:
    """Generate a CSV string from benchmark records."""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for r in records:
        correctness = r.get("correctness", {})
        timing = r.get("timing", {})
        writer.writerow(
            {
                "size": r.get("size", ""),
                "api": r.get("api", ""),
                "direction": r.get("direction", ""),
                "batch": r.get("batch", ""),
                "backend": r.get("backend", ""),
                "flagfft_median_ms": timing.get("flagfft_median_ms", 0),
                "cufft_median_ms": timing.get("cufft_median_ms", 0),
                "speedup": timing.get("speedup", 0),
                "correctness_passed": correctness.get("passed", False),
                "max_abs": correctness.get("max_abs", ""),
                "rms": correctness.get("rms", ""),
                "warmup": r.get("warmup", ""),
                "iters": r.get("iters", ""),
            }
        )
    return output.getvalue()


def generate_markdown(report: dict[str, Any]) -> str:
    """Generate a Markdown benchmark report from a CLI JSON report."""
    cases = report.get("cases", [])
    records = []
    for case in cases:
        records.append(
            {
                "size": case.get("shape", ""),
                "api": case.get("api", ""),
                "direction": case.get("direction", ""),
                "batch": case.get("batch", 1),
                "correctness": case.get("correctness", {}),
                "timing": case.get("timing", {}),
            }
        )

    lines = ["# FlagFFT Benchmark Report\n"]
    lines.append(_build_table(records))

    stats = _compute_stats(records)
    lines.append(f"**Correctness:** {stats['passed']}/{stats['total']} passed")
    if stats["flagfft_ms"]:
        lines.append(
            f"**FlagFFT median range:** {min(stats['flagfft_ms']):.4f}"
            f" - {max(stats['flagfft_ms']):.4f} ms"
        )
        lines.append(
            f"**Geometric mean:** {_geometric_mean(stats['flagfft_ms']):.4f} ms"
        )
    if stats["speedups"]:
        lines.append(
            f"**Overall speedup (geomean):**"
            f" {_geometric_mean(stats['speedups']):.2f}x"
        )

    return "\n".join(lines) + "\n"


def generate_json_report(report: dict[str, Any]) -> str:
    """Generate a JSON benchmark report string (pretty-printed)."""
    return json.dumps(report, indent=2)
