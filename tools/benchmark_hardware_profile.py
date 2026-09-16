#!/usr/bin/env python3
"""Paired end-to-end policy experiments with NumPy correctness and incremental CSV."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import run_tests as acceptance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--baseline-build-dir", type=Path,
                        help="Optional unmodified main build used for the legacy measurements")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policies", default="legacy,native,packed")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--apis", default="c2c,r2c,c2r")
    parser.add_argument("--shapes", help="Optional exact shape filter, comma-separated")
    parser.add_argument("--in-process-repeats", action="store_true",
                        help="Reuse one benchmark process per case/policy for repeated timing series")
    args = parser.parse_args()
    build, root = args.build_dir.resolve(), args.output_dir.resolve()
    baseline = args.baseline_build_dir.resolve() if args.baseline_build_dir else build
    root.mkdir(parents=True, exist_ok=False)
    policies = args.policies.split(",")
    if (args.repeats < 1 or args.timeout <= 0 or any(p not in {"legacy", "native", "packed", "balanced"} for p in policies)
            or any(api not in {"c2c", "r2c", "c2r"} for api in args.apis.split(","))):
        parser.error("positive repeats and valid execution policies required")
    device = subprocess.run([str(build / "flagfft-cli"), "device-info", "--json"],
                            capture_output=True, text=True, check=True, timeout=30)
    (root / "environment.json").write_text(json.dumps({"device": json.loads(device.stdout),
        "warmup": 5, "iters": 20, "repeats": args.repeats, "policies": policies,
        "in_process_repeats": args.in_process_repeats,
        "baseline_build_dir": str(baseline),
        "baseline_commit": acceptance.git_commit(baseline.parent),
        "git_commit": acceptance.git_commit(Path(__file__).resolve().parents[1])}, indent=2))
    shapes = [((16,), 1), ((256,), 257), ((1024,), 256), ((65536,), 1),
              ((23,), 256), ((997,), 1), ((8191,), 16),
              ((64, 64), 1), ((1024, 1024), 1), ((32, 32, 32), 1), ((128, 128, 128), 1)]
    if args.quick:
        shapes = [((256,), 257), ((64, 64), 1), ((32, 32, 32), 1)]
    if args.shapes:
        selected = set(args.shapes.split(","))
        shapes = [(shape, batch) for shape, batch in shapes if "x".join(map(str, shape)) in selected]
        if not shapes:
            parser.error("no matching shapes")
    records = []
    with (root / "incremental.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=["case", "policy", "repeat", "status", "correct",
                                                "flagfft_ms", "platform_ms", "plan", "reason"])
        writer.writeheader()
        for shape, batch in shapes:
            for api in args.apis.split(","):
                direction = "inverse" if api == "c2r" else "forward"
                case_id = f"{api}_{'x'.join(map(str, shape))}_b{batch}"
                case = dict(api=api, shape=shape, rank=len(shape), batch=batch, direction=direction)
                value, _ = acceptance.make_input(api, shape, batch, 1.0)
                expected = acceptance.numpy_reference(value, api, shape, direction)
                correct = {}
                series = {}
                for repeat in range(args.repeats):
                    # Rotate policy order to reduce systematic clock/thermal bias.
                    order = policies[repeat % len(policies):] + policies[:repeat % len(policies)]
                    for policy in order:
                        directory = root / case_id / policy / str(repeat)
                        directory.mkdir(parents=True)
                        env = dict(os.environ, FLAGFFT_EXECUTION_POLICY=policy, FLAGFFT_TUNE_DISABLE="1")
                        active_build = baseline if policy == "legacy" else build
                        env["PYTHONPATH"] = str(active_build.parent / "python") + os.pathsep + env.get("PYTHONPATH", "")
                        record = dict(case=case_id, policy=policy, repeat=repeat, status="failed")
                        try:
                            if policy not in correct:
                                value.tofile(directory / "input.bin")
                                proc = subprocess.run(acceptance.build_accuracy_cmd(case, active_build / "ctest/numpy_fft_capture",
                                                          directory, "flagfft"), env=env, capture_output=True,
                                                      text=True, timeout=args.timeout)
                                (directory / "accuracy.log").write_text(proc.stdout + proc.stderr)
                                if proc.returncode:
                                    raise RuntimeError(f"capture exit {proc.returncode}")
                                actual = acceptance.load_raw(directory / "flagfft.bin", api, shape, batch)
                                stats = acceptance.judged_stats(acceptance.error_stats(actual, expected,
                                                 expected.size // batch, batch), acceptance.accuracy_limit(api, acceptance.product(shape)))
                                (directory / "accuracy.json").write_text(json.dumps(stats, indent=2))
                                correct[policy] = stats["passed"]
                                # Successful raw buffers are not needed for performance analysis.
                                if stats["passed"]:
                                    (directory / "input.bin").unlink()
                                    (directory / "flagfft.bin").unlink()
                            record["correct"] = correct[policy]
                            if not correct[policy]:
                                raise RuntimeError("NumPy correctness failed")
                            cmd = acceptance.build_perf_cmd(case, active_build, 5, 20)
                            if args.in_process_repeats:
                                shape_index = cmd.index("--shape") + 1
                                cmd[shape_index] = ",".join([cmd[shape_index]] * args.repeats)
                            if not args.in_process_repeats or policy not in series:
                                proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=args.timeout)
                                (directory / "performance.log").write_text(proc.stdout + proc.stderr)
                                if proc.returncode:
                                    raise RuntimeError(f"benchmark exit {proc.returncode}")
                                series[policy] = json.loads(proc.stdout)["cases"]
                            result = series[policy][repeat if args.in_process_repeats else 0]
                            (directory / "performance.json").write_text(json.dumps(result, indent=2))
                            record.update(status="passed", flagfft_ms=result["timing"]["flagfft_median_ms"],
                                          platform_ms=result["timing"]["ref_median_ms"], plan=result.get("plan_description", ""))
                        except (subprocess.SubprocessError, OSError, ValueError, RuntimeError, KeyError) as exc:
                            record["reason"] = str(exc)
                        records.append(record)
                        writer.writerow(record)
                        out.flush()
                        print(json.dumps({k: v for k, v in record.items() if k != "plan"}), flush=True)
    (root / "records.json").write_text(json.dumps(records, indent=2))
    comparison = {}
    for case_id in sorted({r["case"] for r in records}):
        medians = {}
        for policy in policies:
            trials = [r for r in records if r["case"] == case_id and r["policy"] == policy]
            if len(trials) == args.repeats and all(r["status"] == "passed" for r in trials):
                medians[policy] = float(np.median([r["flagfft_ms"] for r in trials]))
        comparison[case_id] = {"median_ms": medians, "speedup_vs_legacy": {
            p: medians["legacy"] / v for p, v in medians.items() if "legacy" in medians and p != "legacy"}}
    (root / "summary.json").write_text(json.dumps({"cases": comparison,
        "failed_records": sum(r["status"] != "passed" for r in records)}, indent=2))


if __name__ == "__main__":
    main()
