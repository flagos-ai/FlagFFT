#!/usr/bin/env python3
"""A/B arbitrary environment variants on a chosen case subset.

Generalises ``benchmark_hardware_profile.py``: instead of comparing execution
policies, each variant is a named set of environment overrides.  Every variant
is gated on NumPy correctness before its timing is recorded, so a variant that
changes results cannot masquerade as a speedup.

Variants are written ``name:KEY=VAL,KEY=VAL`` separated by ``;``; an empty
override list (``name:``) is the shipped baseline.
"""
import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import run_tests as acceptance


def parse_variants(raw: str) -> list[tuple[str, dict[str, str]]]:
    variants: list[tuple[str, dict[str, str]]] = []
    for entry in raw.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        name, _, assignments = entry.partition(":")
        name = name.strip()
        if not name:
            raise ValueError(f"variant without a name: {entry!r}")
        overrides: dict[str, str] = {}
        for assignment in filter(None, (a.strip() for a in assignments.split(","))):
            key, sep, value = assignment.partition("=")
            if not sep or not key.strip():
                raise ValueError(f"variant {name!r}: expected KEY=VALUE, got {assignment!r}")
            overrides[key.strip()] = value.strip()
        variants.append((name, overrides))
    if not variants:
        raise ValueError("no variants given")
    if len({name for name, _ in variants}) != len(variants):
        raise ValueError("variant names must be unique")
    return variants


def select_cases(args) -> list[dict]:
    ops = acceptance.load_operators(args.operators)
    matrix = acceptance.load_test_matrix(args.test_matrix)
    cases = acceptance.expand_all_test_cases(ops, matrix)
    if args.ops:
        wanted = set(filter(None, (o.strip() for o in args.ops.split(","))))
        unknown = wanted - {op["id"] for op in ops}
        if unknown:
            raise SystemExit(f"unknown operator ids: {sorted(unknown)}")
        cases = [c for c in cases if c["op_id"] in wanted]
    if args.shapes:
        shapes = {
            tuple(int(v) for v in spec.split("x"))
            for spec in filter(None, (s.strip() for s in args.shapes.split(",")))
        }
        cases = [c for c in cases if tuple(c["shape"]) in shapes]
    if args.batches:
        batches = {int(b) for b in args.batches.split(",")}
        cases = [c for c in cases if c["batch"] in batches]
    if args.directions:
        directions = set(filter(None, (d.strip() for d in args.directions.split(","))))
        cases = [c for c in cases if c["direction"] in directions]
    if not cases:
        raise SystemExit("no cases selected")
    return cases


def measure(case, build, root, variant, overrides, args, env_base):
    case_dir = root / variant / acceptance.case_name(case)
    case_dir.mkdir(parents=True, exist_ok=True)
    env = dict(env_base, FLAGFFT_TUNE_DISABLE="1", **overrides)
    record = {"case": acceptance.case_name(case), "variant": variant, "status": "failed"}

    value, _ = acceptance.make_input(case["api"], tuple(case["shape"]), case["batch"], 1.0)
    expected = acceptance.numpy_reference(
        value, case["api"], tuple(case["shape"]), case["direction"]
    )
    value.tofile(case_dir / "input.bin")
    proc = subprocess.run(
        acceptance.build_accuracy_cmd(
            case, build / "ctest/numpy_fft_capture", case_dir, "flagfft"
        ),
        env=env,
        capture_output=True,
        text=True,
        timeout=args.timeout,
    )
    (case_dir / "accuracy.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode:
        record["reason"] = f"capture exit {proc.returncode}"
        return record
    actual = acceptance.load_raw(case_dir / "flagfft.bin", case["api"], tuple(case["shape"]), case["batch"])
    stats = acceptance.judged_stats(
        acceptance.error_stats(actual, expected, expected.size // case["batch"], case["batch"]),
        acceptance.accuracy_limit(case["api"], acceptance.product(case["shape"])),
    )
    acceptance.write_json(case_dir / "accuracy.json", stats)
    record["correct"] = stats["passed"]
    (case_dir / "input.bin").unlink(missing_ok=True)
    (case_dir / "flagfft.bin").unlink(missing_ok=True)
    if not stats["passed"]:
        record["reason"] = "NumPy correctness failed"
        return record

    times = []
    plan = ""
    for repeat in range(args.repeats):
        proc = subprocess.run(
            acceptance.build_perf_cmd(case, build, args.warmup, args.iters),
            env=env,
            capture_output=True,
            text=True,
            timeout=args.timeout,
        )
        (case_dir / f"performance.{repeat}.log").write_text(proc.stdout + proc.stderr)
        if proc.returncode:
            record["reason"] = f"benchmark exit {proc.returncode}"
            return record
        result = json.loads(proc.stdout)["cases"][0]
        times.append(result["timing"]["flagfft_median_ms"])
        plan = result.get("plan_description", plan) or plan
        record["platform_ms"] = result["timing"]["ref_median_ms"]
    record.update(
        status="passed",
        flagfft_ms=float(np.median(times)),
        spread=float(max(times) - min(times)),
        plan=plan,
    )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--operators", type=Path, default=Path("conf/operators.yaml"))
    parser.add_argument("--test-matrix", type=Path, default=Path("conf/test_matrix.yaml"))
    parser.add_argument("--ops", help="Comma-separated operator ids")
    parser.add_argument("--shapes", help="Comma-separated shapes, e.g. 8192,64x64,16x997x64")
    parser.add_argument("--batches", help="Comma-separated batch sizes")
    parser.add_argument("--directions", help="forward,inverse")
    parser.add_argument("--variants", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--pythonpath",
        help="Codegen tree to put first on PYTHONPATH (defaults to this checkout)",
    )
    args = parser.parse_args()

    build = args.build_dir.resolve()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    variants = parse_variants(args.variants)
    cases = select_cases(args)

    env_base = dict(os.environ)
    codegen = Path(args.pythonpath).resolve() if args.pythonpath else Path(__file__).resolve().parents[1] / "python"
    env_base["PYTHONPATH"] = str(codegen) + os.pathsep + env_base.get("PYTHONPATH", "")

    device = subprocess.run(
        [str(build / "flagfft-cli"), "device-info", "--json"],
        capture_output=True, text=True, check=True, timeout=30, env=env_base,
    )
    acceptance.write_json(
        root / "environment.json",
        {
            "device": json.loads(device.stdout),
            "variants": {name: overrides for name, overrides in variants},
            "warmup": args.warmup,
            "iters": args.iters,
            "repeats": args.repeats,
            "cases": len(cases),
            "pythonpath": str(codegen),
            "git_commit": acceptance.git_commit(Path(__file__).resolve().parents[1]),
        },
    )

    records = []
    fields = ["case", "variant", "status", "correct", "flagfft_ms", "platform_ms", "spread", "plan", "reason"]
    with (root / "incremental.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for index, case in enumerate(cases, 1):
            for name, overrides in variants:
                try:
                    record = measure(case, build, root, name, overrides, args, env_base)
                except (subprocess.SubprocessError, OSError, ValueError, RuntimeError, KeyError) as exc:
                    record = {
                        "case": acceptance.case_name(case),
                        "variant": name,
                        "status": "failed",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                records.append(record)
                writer.writerow(record)
                out.flush()
                print(
                    json.dumps({k: v for k, v in record.items() if k != "plan"}),
                    flush=True,
                )
            print(f"[{index}/{len(cases)}] {acceptance.case_name(case)}", flush=True)
    acceptance.write_json(root / "records.json", records)

    summary = {}
    for case in sorted({r["case"] for r in records}):
        per_variant = {}
        for name, _ in variants:
            trials = [r for r in records if r["case"] == case and r["variant"] == name]
            if len(trials) == args.repeats and all(r["status"] == "passed" for r in trials):
                per_variant[name] = {
                    "median_ms": float(np.median([r["flagfft_ms"] for r in trials])),
                    "platform_ms": trials[0]["platform_ms"],
                    "plan": trials[0]["plan"],
                }
        baseline = per_variant.get(variants[0][0], {}).get("median_ms")
        summary[case] = {
            "variants": per_variant,
            "speedup_vs_baseline": {
                name: baseline / data["median_ms"]
                for name, data in per_variant.items()
                if baseline and name != variants[0][0]
            },
        }
    acceptance.write_json(root / "summary.json", summary)
    print(f"\nwrote {root/'summary.json'}")


if __name__ == "__main__":
    main()
