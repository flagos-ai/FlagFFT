#!/usr/bin/env python3
"""Check per-operator geometric means against a frozen acceptance manifest.

Accuracy and timing can come from separate runs. Their case identities and
actual FlagFFT plans must agree. Reference exclusions come only from the
supplied manifest, not from a result file's choice to skip a case.
"""

import argparse
import json
import math
from pathlib import Path
import re


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=unique_object)


def read_cases(path):
    return read_json(path)["cases"] if Path(path).exists() else {}


def performance_id(case_id):
    return case_id.rsplit("__s", 1)[0]


def matches_identity(actual, expected, *, accuracy):
    fields = ("op_id", "api", "dtype", "rank", "shape", "batch", "placement", "direction")
    if accuracy:
        fields += ("scale",)
    return all(actual.get(key) == expected.get(key) for key in fields)


def matches_plan(plan, expected):
    if not isinstance(plan, str):
        return False
    match = re.search(r"^rank=(\d+) n=\[([^\]]+)\] batch=(\d+) type=\d+", plan, re.M)
    return bool(match and int(match[1]) == expected["rank"]
                and [int(n.strip()) for n in match[2].split(",")] == expected["shape"]
                and int(match[3]) == expected["batch"])


def audit(manifest, accuracy_dir, performance_dir, threshold=0.815):
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be positive and finite")
    expected = manifest["cases"]
    if not expected or len({c["case_id"] for c in expected}) != len(expected):
        raise ValueError("expected accuracy manifest must be nonempty and unique")
    exclusions = manifest.get("reference_skips", {})
    if set(exclusions) - {c["case_id"] for c in expected}:
        raise ValueError("reference exclusions contain unexpected case IDs")
    if any(not isinstance(reason, str) or not reason for reason in exclusions.values()):
        raise ValueError("reference exclusions require explicit reasons")
    operators = {}
    issues = []
    for op in sorted({c["op_id"] for c in expected}):
        requests = {c["case_id"]: c for c in expected if c["op_id"] == op}
        accuracy = read_cases(Path(accuracy_dir) / op / "accuracy_result.json")
        platform = read_cases(Path(accuracy_dir) / op / "platform_accuracy_result.json")
        timings = read_cases(Path(performance_dir) / op / "performance_result.json")
        perf_requests = {}
        for case in requests.values():
            perf_requests.setdefault(performance_id(case["case_id"]), []).append(case)
        for label, records, ids in (("accuracy", accuracy, requests), ("platform", platform, requests),
                                     ("performance", timings, perf_requests)):
            extra = set(records) - set(ids)
            if extra:
                issues.append(f"{op}: unexpected {label} case IDs: {sorted(extra)}")
            if any(row.get("case_id") != key for key, row in records.items()):
                issues.append(f"{op}: inconsistent {label} case IDs")
        rows, speedups = [], []
        for case_id, related in perf_requests.items():
            reasons = []
            skip_flags = [c["case_id"] in exclusions for c in related]
            if len(set(skip_flags)) != 1:
                raise ValueError(f"inconsistent reference policy across scales: {case_id}")
            skip = skip_flags[0]
            timing = timings.get(case_id, {})
            exemplar = related[0]
            for request in related:
                key = request["case_id"]
                measured = accuracy.get(key, {})
                reference = platform.get(key, {})
                if measured.get("status") != "Passed" or not matches_identity(measured, request, accuracy=True):
                    reasons.append(f"{key}: accuracy missing, failed or mismatched")
                if (measured.get("metric") or {}).get("passed") is not True or (measured.get("metric") or {}).get("finite") is not True:
                    reasons.append(f"{key}: accuracy metric is not validated")
                if not matches_plan(measured.get("plan", ""), request):
                    reasons.append(f"{key}: actual FlagFFT dimensions/batch do not match")
                if not measured.get("input_sha256") or not measured.get("output_sha256"):
                    reasons.append(f"{key}: input/output identity missing")
                if skip:
                    if reference.get("status") != "Skipped":
                        reasons.append(f"{key}: expected reference skip is missing")
                else:
                    if reference.get("status") != "Passed" or not matches_identity(reference, request, accuracy=True):
                        reasons.append(f"{key}: platform accuracy missing, failed or mismatched")
                    if (reference.get("metric") or {}).get("passed") is not True or (reference.get("metric") or {}).get("finite") is not True:
                        reasons.append(f"{key}: platform metric is not validated")
                    if reference.get("input_sha256") != measured.get("input_sha256"):
                        reasons.append(f"{key}: FlagFFT and reference input hashes differ")
                    if (measured.get("plan") or "").rstrip("\r\n") != (timing.get("plan") or "").rstrip("\r\n"):
                        reasons.append(f"{key}: accuracy and timing plans differ")
            if not matches_identity(timing, exemplar, accuracy=False):
                reasons.append("performance identity missing or mismatched")
            speedup = None
            if skip:
                if timing.get("status") != "Skipped":
                    reasons.append("expected performance skip is missing")
                if any(timing.get(key) is not None for key in ("flagfft_median_ms", "ref_median_ms", "speedup")):
                    reasons.append("reference-skipped case contains timings or speedup")
            else:
                if timing.get("status") != "Passed":
                    reasons.append("performance missing or failed")
                # performance-only runs deliberately leave this aggregate flag
                # null. The supplied accuracy run is checked case by case above;
                # an explicit invalid verdict or a malformed/missing flag must
                # still fail. Never require a redundant combined runner pass.
                validity = timing.get("baseline_valid")
                if (timing.get("reference_available") is not True
                        or "baseline_valid" not in timing
                        or (validity is not True and validity is not None)):
                    reasons.append("reference timing is not validated")
                values = [timing.get(k) for k in ("flagfft_median_ms", "ref_median_ms", "speedup")]
                if not all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in values):
                    reasons.append("invalid timing or speedup")
                elif not math.isclose(values[2], values[1] / values[0], rel_tol=1e-5):
                    reasons.append("speedup inconsistent with timings")
                else:
                    speedup = values[2]
            if speedup is not None and not reasons:
                speedups.append(speedup)
            rows.append(dict(case_id=case_id, reference_skipped=skip, speedup=speedup,
                             valid=not reasons, reasons=reasons))
        complete = all(row["valid"] for row in rows)
        mean = math.exp(sum(math.log(s) for s in speedups) / len(speedups)) if speedups else None
        qualified_mean = mean if complete else None
        operators[op] = dict(expected_accuracy=len(requests), expected_performance=len(rows),
            reference_skips=sum(row["reference_skipped"] for row in rows), valid_performance=len(speedups),
            complete=complete, geometric_mean_speedup=qualified_mean,
            diagnostic_valid_case_mean=mean, threshold=threshold,
            passed=complete and qualified_mean is not None and qualified_mean >= threshold,
            cases=rows)
    return dict(passed=not issues and all(op["passed"] for op in operators.values()),
                operator_count=len(operators), expected_accuracy_count=len(expected),
                threshold=threshold, scope_source_commit=manifest.get("source_commit"),
                issues=issues, operators=operators)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--accuracy-dir", type=Path, required=True)
    parser.add_argument("--performance-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.815)
    args = parser.parse_args()
    result = audit(read_json(args.manifest), args.accuracy_dir, args.performance_dir, args.threshold)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for op, row in result["operators"].items():
        mean = row["geometric_mean_speedup"]
        value = "incomplete" if mean is None else f"{mean:.6f}x"
        print(f"{op}: {value} {'PASS' if row['passed'] else 'FAIL'}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
