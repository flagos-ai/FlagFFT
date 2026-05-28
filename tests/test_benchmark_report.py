from __future__ import annotations

import csv
import io

from benchmark.utils.report import generate_console_table, generate_csv


def test_generate_csv_writes_ref_median_ms_from_cli_timing() -> None:
    csv_text = generate_csv(
        [
            {
                "size": 16,
                "api": "c2c",
                "direction": "forward",
                "batch": 1,
                "backend": "cuda",
                "timing": {
                    "flagfft_median_ms": 0.125,
                    "ref_median_ms": 0.25,
                    "speedup": 2.0,
                },
            }
        ]
    )

    row = next(csv.DictReader(io.StringIO(csv_text)))

    assert "ref_median_ms" in row
    assert "cufft_median_ms" not in row
    assert row["ref_median_ms"] == "0.25"


def test_generate_console_table_uses_ref_median_ms_without_correctness() -> None:
    table = generate_console_table(
        [
            {
                "size": 16,
                "api": "c2c",
                "direction": "forward",
                "batch": 1,
                "timing": {
                    "flagfft_median_ms": 0.125,
                    "ref_median_ms": 0.25,
                    "speedup": 2.0,
                },
            }
        ],
        suite="smoke",
        warmup=1,
        iters=1,
    )

    assert "0.2500" in table
    assert "0.0000" not in table
    assert "Correctness:" not in table
