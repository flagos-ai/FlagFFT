#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Gate a complete accuracy + performance run, using case-weighted speedups."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def evaluate(root, expected_cases=400, expected_ops=30, target=1.0, floor=0.7, matrix_hash=None):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    expected = {case["case_id"] for case in manifest["performance_cases"]}
    operators = manifest["operators"]
    accuracy_counts = Counter(case["op_id"] for case in manifest["cases"])
    problems, records, groups = [], {}, {}
    if matrix_hash is not None and manifest["config"].get("matrix_sha256") != matrix_hash:
        problems.append("matrix identity is missing or differs from the requested matrix")
    if len(expected) != expected_cases or len(operators) != expected_ops:
        problems.append("manifest does not cover the requested case/operator counts")
    if manifest["config"].get("performance_only") or manifest["config"].get("accuracy_only"):
        problems.append("a combined accuracy + performance run is required")
    for op in operators:
        op_id = op["id"]
        for phase in ("accuracy", "platform_accuracy", "performance"):
            path = root / op_id / (phase + "_result.json")
            if not path.exists():
                problems.append(f"{op_id}: missing {phase} results")
                continue
            result = json.loads(path.read_text())
            count = sum(case["op_id"] == op_id for case in manifest["performance_cases"]) if phase == "performance" else accuracy_counts[op_id]
            # Accuracy status uses Passed in per-phase files, unlike summary.json.
            if (result.get("status") not in ("Passed", "PASS") or
                    result.get("total") != count or result.get("passed") != count or
                    any(result.get(k, 0) for k in ("failed", "errors", "missing", "skipped"))):
                problems.append(f"{op_id}: {phase} is not complete and passing")
            if phase != "performance":
                continue
            op_speeds = []
            for key, case in result.get("cases", {}).items():
                if key in records:
                    problems.append(f"duplicate performance case: {key}")
                records[key] = case
                values = [case.get(k) for k in ("speedup", "flagfft_median_ms", "ref_median_ms")]
                if (case.get("status") != "Passed" or case.get("baseline_valid") is not True or
                        any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values)):
                    problems.append(f"{key}: invalid timing or unvalidated reference")
                    continue
                speedup, fft, ref = values
                if not math.isclose(speedup, ref / fft, rel_tol=1e-5):
                    problems.append(f"{key}: speedup disagrees with recorded timings")
                    continue
                op_speeds.append(speedup)
            if op_speeds:
                groups[op_id] = {"cases": len(op_speeds), "geomean": math.exp(sum(map(math.log, op_speeds)) / len(op_speeds)), "minimum": min(op_speeds)}
    if set(records) != expected:
        problems.append(f"performance coverage mismatch: {len(expected - set(records))} missing, {len(set(records) - expected)} extra")
    valid = {key: case["speedup"] for key, case in records.items()
             if type(case.get("speedup")) in (int, float) and math.isfinite(case["speedup"]) and case["speedup"] > 0}
    gm = math.exp(sum(map(math.log, valid.values())) / len(valid)) if valid else None
    below = [{"case_id": key, "speedup": speedup} for key, speedup in sorted(valid.items(), key=lambda item: item[1]) if speedup < floor]
    if gm is None or gm < target:
        problems.append("case-weighted geometric mean is below target")
    if below:
        problems.append(f"{len(below)} cases are below the per-case floor")
    operator_gm = math.exp(sum(math.log(group["geomean"]) for group in groups.values()) / len(groups)) if groups else None
    return {"passed": not problems, "source_commit": manifest["env"].get("git_commit"),
            "expected_matrix_sha256": matrix_hash,
            "expected_cases": expected_cases, "recorded_cases": len(records),
            "case_weighted_geomean": gm, "minimum_speedup": min(valid.values()) if valid else None,
            "operator_weighted_geomean": operator_gm,
            "target_geomean": target, "per_case_floor": floor,
            "operators": groups, "below_floor": below, "problems": problems}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--matrix-file", type=Path, default=Path(__file__).resolve().parents[1] / "conf/test_matrix.yaml")
    parser.add_argument("--expected-cases", type=int, default=400)
    parser.add_argument("--expected-ops", type=int, default=30)
    parser.add_argument("--target", type=float, default=1.0)
    parser.add_argument("--floor", type=float, default=0.7)
    args = parser.parse_args()
    if min(args.expected_cases, args.expected_ops, args.target, args.floor) <= 0 or not all(map(math.isfinite, (args.target, args.floor))):
        parser.error("counts and thresholds must be finite and positive")
    try:
        matrix_hash = hashlib.sha256(args.matrix_file.read_bytes()).hexdigest()
        report = evaluate(args.result_dir, args.expected_cases, args.expected_ops, args.target, args.floor, matrix_hash)
    except (OSError, ValueError, KeyError, TypeError) as error:
        report = {"passed": False, "problems": [str(error)]}
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
