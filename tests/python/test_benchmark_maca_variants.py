from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from benchmark_maca_variants import summarize_records  # noqa: E402


def test_summary_uses_aggregated_repeats_record_per_variant() -> None:
    case = "3d_c2c__fp32__n128x2048x64__b1__inverse__outofplace__s1"
    variants = [("A", {}), ("B", {"FLAGFFT_MACA_TRANSPOSE3D": "pair16sg2rg2"})]
    records = [
        {
            "case": case,
            "variant": "A",
            "status": "passed",
            "flagfft_ms": 1.662208,
            "platform_ms": 0.856320,
            "plan": "default",
        },
        {
            "case": case,
            "variant": "B",
            "status": "passed",
            "flagfft_ms": 1.651456,
            "platform_ms": 0.856320,
            "plan": "group2",
        },
    ]

    summary = summarize_records(records, variants)[case]

    assert summary["variants"]["A"]["median_ms"] == 1.662208
    assert summary["variants"]["B"]["median_ms"] == 1.651456
    assert abs(summary["speedup_vs_baseline"]["B"] - 1.006512) < 1e-5
    assert "group2" == summary["variants"]["B"]["plan"]
