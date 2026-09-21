#!/usr/bin/env python3
"""Audit a complete acceptance result against a per-case speedup threshold.

The runner's performance Passed verdict means execution succeeded, not that a
performance target was met. This audit also checks each reference accuracy case
independently, including on backends where baseline_valid does not summarize it.
"""

import argparse
import csv
import json
import math
from pathlib import Path


def load_cases(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text())["cases"]


def audit(result_dir, threshold, expected_count=None, ascend_ct_single=False):
    manifest = json.loads((result_dir / "manifest.json").read_text())
    expected = manifest["performance_cases"]
    rows = []
    for case in expected:
        op_dir = result_dir / case["op_id"]
        perf = load_cases(op_dir / "performance_result.json").get(case["case_id"], {})
        accuracy = load_cases(op_dir / "accuracy_result.json")
        platform = load_cases(op_dir / "platform_accuracy_result.json")
        accuracy_ids = [
            c["case_id"]
            for c in manifest["cases"]
            if c["case_id"].rsplit("__s", 1)[0] == case["case_id"]
        ]
        reasons = []
        if not accuracy_ids:
            reasons.append("missing expected accuracy cases")
        for label, records in (("accuracy", accuracy), ("platform_accuracy", platform)):
            if any(
                records.get(case_id, {}).get("status") != "Passed"
                for case_id in accuracy_ids
            ):
                reasons.append(label + " incomplete or failed")
        if perf.get("status") != "Passed":
            reasons.append("performance incomplete or failed")
        if perf.get("baseline_valid") is not True:
            reasons.append("baseline_valid is not true")
        if perf.get("reference_available") is not True:
            reasons.append("reference unavailable")
        values = [
            perf.get(key) for key in ("flagfft_median_ms", "ref_median_ms", "speedup")
        ]
        valid_numbers = all(
            isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in values
        )
        if not valid_numbers:
            reasons.append("invalid timing or speedup")
        elif not math.isclose(values[2], values[1] / values[0], rel_tol=1e-5):
            reasons.append("speedup inconsistent with timings")
        elif values[2] < threshold:
            reasons.append("speedup below threshold")
        rows.append(
            {
                "case_id": case["case_id"],
                "flagfft_median_ms": values[0],
                "ref_median_ms": values[1],
                "speedup": values[2],
                "threshold": threshold,
                "passed": not reasons,
                "reason": "; ".join(reasons),
            }
        )
    complete = expected_count is None or len(rows) == expected_count
    ids = [row["case_id"] for row in rows]
    unique = len(set(ids)) == len(ids)
    matrix_matches = True
    if ascend_ct_single:
        lengths = (
            16,
            1024,
            2048,
            8192,
            16384,
            46189,
            185640,
            328050,
            340200,
            663000,
            1048576,
        )
        matrix = {
            f"1d_ct_single_{api}__n{n}__b1__{direction}"
            for api, directions in (
                ("c2c", ("forward", "inverse")),
                ("r2c", ("forward",)),
                ("c2r", ("inverse",)),
            )
            for n in lengths
            for direction in directions
        }
        matrix_matches = set(ids) == matrix
    return {
        "passed": bool(rows)
        and complete
        and unique
        and matrix_matches
        and all(row["passed"] for row in rows),
        "expected_count": expected_count,
        "case_count": len(rows),
        "unique_case_ids": unique,
        "matrix_matches": matrix_matches,
        "count_matches": complete,
        "passed_count": sum(row["passed"] for row in rows),
        "threshold": threshold,
        "cases": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument(
        "--ascend-ct-single",
        action="store_true",
        help="Require the exact 44-case Ascend CT single matrix",
    )
    args = parser.parse_args()
    if not math.isfinite(args.threshold) or args.threshold <= 0:
        parser.error("threshold must be finite and positive")
    if args.expected_count is not None and args.expected_count <= 0:
        parser.error("expected-count must be positive")
    report = audit(
        args.result_dir, args.threshold, args.expected_count, args.ascend_ct_single
    )
    (args.result_dir / "performance_gate.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    with (args.result_dir / "performance_gate.csv").open("w", newline="") as stream:
        fields = (
            "case_id",
            "flagfft_median_ms",
            "ref_median_ms",
            "speedup",
            "threshold",
            "passed",
            "reason",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report["cases"])
    print(
        f"{'PASS' if report['passed'] else 'FAIL'}: {report['passed_count']}/{report['case_count']} cases at >= {args.threshold:g}"
    )
    if not report["count_matches"]:
        print(f"Expected {args.expected_count} cases, found {report['case_count']}")
    if not report["unique_case_ids"] or not report["matrix_matches"]:
        print("Duplicate performance case IDs or unexpected Ascend CT single matrix")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
