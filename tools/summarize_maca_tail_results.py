#!/usr/bin/env python3
"""Read synced suite directories and print JSON; never modify source results.

Usage: python tools/summarize_maca_tail_results.py results/suite1 results/suite2
meets_0_8 checks usable timing against ref/FlagFFT >= 0.8, independently of
accuracy. Performance status Passed alone does not imply this target is met.
Missing/unusable timing produces null, not a pass. build-* trees are skipped.
"""

import argparse
import json
import math
import os
from pathlib import Path
import re
import sys


def number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def read_json(path, warnings):
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or not isinstance(data.get("cases", {}), dict):
            raise ValueError("expected an object with a cases mapping")
        return data
    except (OSError, ValueError) as error:
        warnings.append(f"{path}: {error}")
        return {}


def nearest_environment(directory, suite):
    while True:
        path = directory / "environment.txt"
        if path.is_file():
            return path
        if directory == suite:
            return None
        directory = directory.parent


def source_commit(environment, warnings):
    if environment is None:
        return None
    try:
        # Accept git rev-parse output or an explicitly labelled commit. Do not
        # mistake binary SHA256 lines for the source revision.
        for line in environment.read_text().splitlines():
            match = re.fullmatch(
                r"\s*(?:(?:source_?commit|git_commit|commit)\s*[=:]\s*)?"
                r"([0-9a-fA-F]{40})\s*", line, re.IGNORECASE
            )
            if match:
                return match[1].lower()
        warnings.append(f"{environment}: no source commit found")
    except OSError as error:
        warnings.append(f"{environment}: {error}")
    return None


def summarize(suites):
    rows, warnings, suite_summaries, seen = [], [], [], set()
    for suite in suites:
        suite = suite.resolve()
        if not suite.is_dir() or any(p.startswith("build-") for p in suite.parts):
            warnings.append(f"{suite}: not a suite directory, or inside build-*")
            continue
        start = len(rows)
        files = 0
        for directory, children, names in os.walk(suite):
            children[:] = sorted(c for c in children if not c.startswith("build-"))
            if "performance_result.json" not in names:
                continue
            path = Path(directory) / "performance_result.json"
            if path.resolve() in seen:
                continue
            seen.add(path.resolve())
            files += 1
            performance = read_json(path, warnings)
            accuracy_path = path.with_name("accuracy_result.json")
            accuracy = read_json(accuracy_path, warnings)
            environment = nearest_environment(path.parent, suite)
            commit = source_commit(environment, warnings)
            if environment is None:
                warnings.append(f"{path}: no environment.txt within suite")
            parts = path.relative_to(suite).parts
            variant_parts = parts[:parts.index("acceptance")] if "acceptance" in parts else parts[:-2]
            variant = "/".join(variant_parts) or "default"
            for case_id, case in sorted(performance.get("cases", {}).items()):
                if not isinstance(case, dict):
                    warnings.append(f"{path}: invalid case {case_id}")
                    continue
                ff = number(case.get("flagfft_median_ms"))
                ref = number(case.get("ref_median_ms"))
                speedup = number(case.get("speedup"))
                usable = (
                    case.get("status") == "Passed"
                    and case.get("reference_available") is True
                    and case.get("baseline_valid") is not False
                    and all(v is not None and v > 0 for v in (ff, ref, speedup))
                    and math.isclose(speedup, ref / ff, rel_tol=1e-5)
                )
                matching_accuracy = [
                    {"case_id": aid, "status": a.get("status")}
                    for aid, a in sorted(accuracy.get("cases", {}).items())
                    if isinstance(a, dict) and (aid == case_id or aid.rsplit("__s", 1)[0] == case_id)
                ]
                rows.append({
                    "suite": suite.name,
                    "variant": variant,
                    "case_id": case_id,
                    "flagfft_us": ff * 1000 if ff is not None else None,
                    "ref_us": ref * 1000 if ref is not None else None,
                    "speedup": speedup,
                    "performance_status": case.get("status"),
                    "meets_0_8": speedup >= 0.8 if usable else None,
                    "accuracy": {
                        "status": accuracy.get("status", "Missing"),
                        **{k: accuracy.get(k) for k in (
                            "reference", "total", "completed", "passed", "failed",
                            "errors", "skipped", "missing"
                        )},
                        "matching_cases": matching_accuracy,
                    },
                    "source_commit": commit,
                    "environment_file": str(environment) if environment else None,
                    "performance_file": str(path),
                    "accuracy_file": str(accuracy_path),
                })
        suite_summaries.append({
            "suite": suite.name, "path": str(suite),
            "performance_files": files, "cases": len(rows) - start,
        })
    return {
        "threshold": 0.8,
        "threshold_semantics": "usable ref/FlagFFT timing >= 0.8; accuracy reported separately; null = unknown",
        "summary": {
            "cases": len(rows),
            "meets_0_8": sum(r["meets_0_8"] is True for r in rows),
            "below_0_8": sum(r["meets_0_8"] is False for r in rows),
            "unknown": sum(r["meets_0_8"] is None for r in rows),
        },
        "suites": suite_summaries,
        "cases": rows,
        "warnings": warnings,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suites", nargs="+", type=Path)
    args = parser.parse_args()
    report = summarize(args.suites)
    json.dump(report, sys.stdout, ensure_ascii=False, indent=2, allow_nan=False)
    print()
    return 1 if report["warnings"] else 0


if __name__ == "__main__":
    sys.exit(main())
