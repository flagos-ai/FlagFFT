"""Benchmark report generation: JSON terminal output and Markdown tables."""

from __future__ import annotations

import json
from typing import Any


def _build_table(records: list[dict[str, Any]]) -> str:
    """Build a markdown table from benchmark records."""
    if not records:
        return "_No benchmark results._\n"

    headers = [
        "Size",
        "API",
        "Direction",
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
        if ri == 0:  # separator after header
            lines.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")

    return "\n".join(lines) + "\n"


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
                "correctness": case.get("correctness", {}),
                "timing": case.get("timing", {}),
            }
        )

    lines = ["# FlagFFT Benchmark Report\n"]
    lines.append(_build_table(records))

    # Summary statistics
    passed = sum(1 for r in records if r["correctness"].get("passed", False))
    total = len(records)
    lines.append(f"**Correctness:** {passed}/{total} passed")
    if records:
        medians = [
            r["timing"].get("flagfft_median_ms", 0)
            for r in records
            if r["timing"].get("flagfft_median_ms", 0) > 0
        ]
        if medians:
            lines.append(
                f"**FlagFFT median range:** {min(medians):.4f} – {max(medians):.4f} ms"
            )
            lines.append(f"**Geometric mean:** {_geometric_mean(medians):.4f} ms")

    return "\n".join(lines) + "\n"


def generate_json_report(report: dict[str, Any]) -> str:
    """Generate a JSON benchmark report string (pretty-printed)."""
    return json.dumps(report, indent=2)


def _geometric_mean(values: list[float]) -> float:
    import math

    if not values:
        return 0.0
    log_sum = sum(math.log(max(v, 1e-12)) for v in values)
    return math.exp(log_sum / len(values))
