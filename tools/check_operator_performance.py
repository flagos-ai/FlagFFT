#!/usr/bin/env python3
"""Audit per-operator geometric means against an independently fixed matrix.

Use an archived manifest chosen before optimization as --matrix-manifest.
Only the reference-policy exclusions in that manifest may be skipped. A
reference exclusion still requires the FlagFFT versus NumPy accuracy result.
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path


IDENTITY_FIELDS = ("case_id", "op_id", "api", "dtype", "rank", "shape", "batch",
                   "placement", "direction", "scale")


def _identity(case):
    return json.dumps({key: case[key] for key in IDENTITY_FIELDS if key in case},
                      sort_keys=True)


def _records(root, op, phase):
    path = root / op / (phase + "_result.json")
    return json.loads(path.read_text()).get("cases", {}) if path.exists() else {}


def _policy_skip(record, reason):
    return record.get("status") == "Skipped" and record.get("skip_reason") == reason


def audit_operators(result_dir, matrix_manifest, *, accuracy_dir=None, threshold=0.815,
                    ops=None):
    result_dir, matrix_manifest = Path(result_dir), Path(matrix_manifest)
    accuracy_dir = Path(accuracy_dir) if accuracy_dir is not None else result_dir
    fixed = json.loads(matrix_manifest.read_text())
    performance_manifest = json.loads((result_dir / "manifest.json").read_text())
    accuracy_manifest = json.loads((accuracy_dir / "manifest.json").read_text())
    all_ops = {case["op_id"] for case in fixed["performance_cases"]}
    selected = set(ops) if ops is not None else all_ops
    errors = []
    if not selected or not selected <= all_ops:
        errors.append("requested operators are empty or absent from the fixed matrix")
    for label, manifest in (("performance", performance_manifest), ("accuracy", accuracy_manifest)):
        for key in ("backend", "architecture", "reference_library"):
            expected_value = fixed.get("env", {}).get(key)
            if expected_value is not None and manifest.get("env", {}).get(key) != expected_value:
                errors.append(label + " environment differs from fixed " + key)
    accuracy_commit = accuracy_manifest.get("env", {}).get("git_commit")
    performance_commit = performance_manifest.get("env", {}).get("git_commit")
    if accuracy_commit != performance_commit:
        errors.append("accuracy and performance source commits differ")
    for phase, manifest in (("performance_cases", performance_manifest),
                            ("cases", accuracy_manifest)):
        expected = [_identity(c) for c in fixed[phase] if c["op_id"] in selected]
        actual = [_identity(c) for c in manifest[phase] if c["op_id"] in selected]
        if len(set(expected)) != len(expected):
            errors.append("duplicate cases in fixed " + phase)
        ids = [c["case_id"] for c in fixed[phase] if c["op_id"] in selected]
        if len(set(ids)) != len(ids):
            errors.append("duplicate case IDs in fixed " + phase)
        if Counter(actual) != Counter(expected):
            errors.append("candidate " + phase + " differs from the fixed matrix")
    rows, operators = [], []
    for op in sorted(selected & all_ops):
        performance = _records(result_dir, op, "performance")
        accuracy = _records(accuracy_dir, op, "accuracy")
        reference = _records(accuracy_dir, op, "platform_accuracy")
        expected_perf = [c for c in fixed["performance_cases"] if c["op_id"] == op]
        expected_acc = [c for c in fixed["cases"] if c["op_id"] == op]
        allowed_perf = {c["case_id"] for c in expected_perf}
        # Some runner versions include policy-skipped accuracy IDs in the
        # performance record map as well. They are not extra timing samples.
        allowed_perf.update(c["case_id"] for c in expected_acc if c.get("skip_reason"))
        unexpected = set(performance) - allowed_perf
        if unexpected:
            errors.append(op + ": unexpected performance records: " + ", ".join(sorted(unexpected)))
        op_rows = []
        for case in expected_perf:
            case_id = case["case_id"]
            perf = performance.get(case_id, {})
            related = [c for c in expected_acc if c["case_id"].rsplit("__s", 1)[0] == case_id]
            reason = case.get("skip_reason")
            failures = []
            if not related:
                failures.append("no expected accuracy samples")
            for sample in related:
                sample_id = sample["case_id"]
                if accuracy.get(sample_id, {}).get("status") != "Passed":
                    failures.append("FlagFFT accuracy incomplete or failed: " + sample_id)
                ref = reference.get(sample_id, {})
                if reason:
                    if not _policy_skip(ref, reason):
                        failures.append("reference skip does not match fixed policy: " + sample_id)
                elif ref.get("status") != "Passed":
                    failures.append("reference accuracy incomplete or failed: " + sample_id)
            speedup = None
            if reason:
                if not _policy_skip(perf, reason):
                    failures.append("performance skip does not match fixed policy")
            else:
                if perf.get("status") != "Passed":
                    failures.append("performance incomplete or failed")
                if perf.get("baseline_valid") is not True or perf.get("reference_available") is not True:
                    failures.append("reference baseline is not valid")
                times = [perf.get(key) for key in ("flagfft_median_ms", "ref_median_ms", "speedup")]
                if not all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in times):
                    failures.append("invalid timing or speedup")
                elif not math.isclose(times[2], times[1] / times[0], rel_tol=1e-5):
                    failures.append("speedup inconsistent with timings")
                else:
                    speedup = times[1] / times[0]
            row = dict(op_id=op, case_id=case_id, reference_policy_skip=reason,
                       flagfft_median_ms=perf.get("flagfft_median_ms"),
                       ref_median_ms=perf.get("ref_median_ms"), speedup=speedup,
                       evidence_valid=not failures, failures=failures)
            rows.append(row)
            op_rows.append(row)
        comparable = [r for r in op_rows if not r["reference_policy_skip"]]
        speeds = [r["speedup"] for r in comparable if r["evidence_valid"]]
        complete = bool(op_rows) and all(r["evidence_valid"] for r in op_rows)
        mean = math.exp(math.fsum(map(math.log, speeds)) / len(speeds)) if speeds else None
        operators.append(dict(op_id=op, expected_count=len(op_rows),
            policy_skips=len(op_rows) - len(comparable), comparable_count=len(comparable),
            valid_count=len(speeds), geometric_mean_speedup=mean,
            evidence_complete=complete,
            passed=complete and bool(comparable) and len(speeds) == len(comparable)
                and mean >= threshold))
    return dict(passed=not errors and bool(operators) and all(op["passed"] for op in operators),
                threshold=threshold, matrix_manifest=str(matrix_manifest),
                result_dir=str(result_dir), accuracy_dir=str(accuracy_dir),
                coverage_errors=errors, operators=operators, cases=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--matrix-manifest", type=Path, required=True)
    parser.add_argument("--accuracy-dir", type=Path)
    parser.add_argument("--threshold", type=float, default=0.815)
    parser.add_argument("--ops", nargs="+")
    args = parser.parse_args()
    if not math.isfinite(args.threshold) or args.threshold <= 0:
        parser.error("threshold must be finite and positive")
    report = audit_operators(args.result_dir, args.matrix_manifest,
        accuracy_dir=args.accuracy_dir, threshold=args.threshold, ops=args.ops)
    (args.result_dir / "operator_performance_gate.json").write_text(json.dumps(report, indent=2) + "\n")
    for op in report["operators"]:
        mean = op["geometric_mean_speedup"]
        value = f"{mean:.6f}x" if mean is not None else "unavailable"
        print(f"{'PASS' if op['passed'] else 'FAIL'} {op['op_id']}: {value}; "
              f"{op['valid_count']}/{op['comparable_count']} valid, "
              f"{op['policy_skips']} reference-policy skips")
    for error in report["coverage_errors"]:
        print("FAIL coverage: " + error)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
