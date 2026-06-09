#!/usr/bin/env python3
"""FlagFFT unified test runner."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import queue
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_operators(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("ops", [])


def load_test_matrix(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_sizes(matrix: dict, ref) -> list:
    if isinstance(ref, str):
        return matrix.get(ref, [])
    return ref if isinstance(ref, list) else []


def expand_test_cases(
    ops: list[dict], matrix: dict, combination: str
) -> list[dict[str, Any]]:
    combo = matrix.get("combinations", {}).get(combination)
    if combo is None:
        print(f"ERROR: unknown combination '{combination}'")
        sys.exit(1)

    sizes = resolve_sizes(matrix, combo["sizes"])
    batches = resolve_sizes(matrix, combo.get("batches", [1]))
    scales = resolve_sizes(matrix, combo.get("scales", [1.0]))

    cases = []
    for op in ops:
        for algo in op["algorithms"]:
            # Skip if algorithm doesn't match combination
            if combination in ("ct", "bs") and algo == "2d":
                continue
            if combination in ("2d", "2d_full") and algo != "2d":
                continue

            # For bs algorithm, use bs sizes when combination is ct or full
            op_sizes = sizes
            if algo == "bs":
                if combination in ("ct", "full"):
                    op_sizes = resolve_sizes(matrix, "sizes_bs")
            elif algo == "ct":
                if combination == "bs":
                    op_sizes = resolve_sizes(matrix, "sizes_ct")

            for size in op_sizes:
                for batch in batches:
                    for scale in scales:
                        for direction in op["directions"]:
                            nx = size[0] if isinstance(size, list) else size
                            ny = size[1] if isinstance(size, list) else 0
                            cases.append(
                                {
                                    "op_id": op["id"],
                                    "algo": algo,
                                    "nx": nx,
                                    "ny": ny,
                                    "batch": batch,
                                    "scale": scale,
                                    "direction": direction,
                                    "ctest": op["ctest"],
                                    "cli_type": op["cli_type"],
                                    "rank": op["rank"],
                                }
                            )
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FlagFFT unified test runner")
    parser.add_argument("--ops", default=None, help="Comma-separated operator IDs")
    parser.add_argument("--stages", default="stable", help="Comma-separated stages")
    parser.add_argument(
        "--combination",
        default="ct",
        choices=["ct", "bs", "full", "2d", "2d_full"],
        help="Test combination (ct/bs for quick, full for all sizes/batches/scales)",
    )
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU IDs or 'all'")
    parser.add_argument("--accuracy-only", action="store_true")
    parser.add_argument("--performance-only", action="store_true")
    parser.add_argument("--build-dir", default=str(ROOT / "build"))
    parser.add_argument("--output", default="summary.json")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--warmup", type=int, default=10, help="Benchmark warmup iterations"
    )
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


# Test execution functions


def build_accuracy_cmd(case: dict, build_dir: Path) -> tuple[list[str], str]:
    binary = build_dir / "ctest" / case["ctest"]
    cmd = [str(binary), f"--nx={case['nx']}"]
    if case["rank"] == 2:
        cmd.append(f"--ny={case['ny']}")
    cmd.append(f"--batch={case['batch']}")
    cmd.append(f"--direction={case['direction']}")
    cmd.append(f"--scale={case['scale']}")
    json_file = f"/tmp/flagfft_acc_{os.getpid()}_{case['op_id']}_{case['nx']}_{case['batch']}.json"
    cmd.append(f"--json-file={json_file}")
    return cmd, json_file


def build_perf_cmd(case: dict, build_dir: Path, warmup: int, iters: int) -> list[str]:
    binary = build_dir / "flagfft-cli"
    cmd = [
        str(binary),
        "bench",
        "--api",
        case["cli_type"],
        "--direction",
        case["direction"],
        "--json",
    ]
    if case["rank"] == 1:
        cmd += ["--shape", str(case["nx"])]
    else:
        cmd += ["--rank", "2", "--shape", f"{case['nx']}x{case['ny']}"]
    cmd += [
        "--batch",
        str(case["batch"]),
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
    ]
    return cmd


def run_subprocess(
    cmd: list[str], timeout: int, gpu_id: int
) -> tuple[int, str, str, float]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    start = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr, time.monotonic() - start
    except subprocess.TimeoutExpired:
        return -100, "", "TIMEOUT", time.monotonic() - start


def parse_accuracy_result(json_file: str) -> dict[str, Any]:
    try:
        with open(json_file) as f:
            data = json.load(f)
        total = data.get("total", 0)
        passed = data.get("passed", 0)
        failed = data.get("failed", 0)
        skipped = data.get("skipped", 0)
        status = "Failed" if failed > 0 else ("Passed" if passed > 0 else "Skipped")
        return {
            "status": status,
            "total": total,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "duration_ms": data.get("duration_ms", 0),
            "failures": data.get("failures", []),
        }
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return {"status": "Error", "error": str(e)}


def parse_perf_result(output: str) -> dict[str, Any]:
    try:
        data = json.loads(output)
        cases = data.get("cases", [])
        if not cases:
            return {"status": "Error", "error": "no cases in output"}
        case = cases[0]
        timing = case.get("timing", {})
        return {
            "status": "Passed"
            if (
                timing.get("speedup", 0) > 0
                and timing.get("flagfft_median_ms", 0) > 0
                and timing.get("ref_median_ms", 0) > 0
            )
            else "Failed",
            "latency_flagfft_ms": timing.get("flagfft_median_ms", 0),
            "latency_ref_ms": timing.get("ref_median_ms", 0),
            "speedup": timing.get("speedup", 0),
            "plan_description": case.get("plan_description", ""),
        }
    except (json.JSONDecodeError, IndexError, KeyError) as e:
        return {"status": "Error", "error": str(e)}


# Worker process


def worker_proc(
    gpu_id,
    work_queue,
    display_queue,
    build_dir,
    timeout,
    accuracy_only,
    performance_only,
    warmup,
    iters,
):
    while True:
        try:
            case = work_queue.get_nowait()
        except queue.Empty:
            break

        case_id = f"{case['op_id']} nx={case['nx']} batch={case['batch']}"

        if not performance_only:
            display_queue.put(
                {
                    "gpu": gpu_id,
                    "phase": "accuracy",
                    "case": case_id,
                    "status": "running",
                }
            )
            cmd, jf = build_accuracy_cmd(case, build_dir)
            rc, stdout, stderr, elapsed = run_subprocess(cmd, timeout, gpu_id)
            if rc == -100:
                acc_result = {"status": "Timeout", "duration_s": elapsed}
            elif rc in (0, 77):
                acc_result = parse_accuracy_result(jf)
                acc_result["duration_s"] = elapsed
            else:
                acc_result = {"status": "Error", "rc": rc, "stderr": stderr[:500]}
            display_queue.put(
                {
                    "gpu": gpu_id,
                    "phase": "accuracy",
                    "case": case_id,
                    "status": acc_result["status"],
                    "duration": elapsed,
                    "result": acc_result,
                    "op_id": case["op_id"],
                    "nx": case["nx"],
                    "batch": case["batch"],
                }
            )
            try:
                os.unlink(jf)
            except OSError:
                pass

        if not accuracy_only:
            display_queue.put(
                {
                    "gpu": gpu_id,
                    "phase": "performance",
                    "case": case_id,
                    "status": "running",
                }
            )
            cmd = build_perf_cmd(case, build_dir, warmup, iters)
            rc, stdout, stderr, elapsed = run_subprocess(cmd, timeout, gpu_id)
            if rc == -100:
                perf_result = {"status": "Timeout", "duration_s": elapsed}
            elif rc in (0, 77):
                perf_result = parse_perf_result(stdout)
                perf_result["duration_s"] = elapsed
            else:
                perf_result = {"status": "Error", "rc": rc, "stderr": stderr[:500]}
            display_queue.put(
                {
                    "gpu": gpu_id,
                    "phase": "performance",
                    "case": case_id,
                    "status": perf_result["status"],
                    "duration": elapsed,
                    "result": perf_result,
                    "op_id": case["op_id"],
                    "nx": case["nx"],
                    "batch": case["batch"],
                }
            )


# LiveDisplay class


class LiveDisplay:
    def __init__(self, n_gpus: int):
        self.n_gpus = n_gpus
        self.gpu_status: dict[int, str] = {i: "idle" for i in range(n_gpus)}
        self.completed = 0
        self.total = 0
        self.is_tty = sys.stderr.isatty()
        self._last_lines = 0
        self.completed_cases: set[str] = set()

    def set_total(self, total: int):
        self.total = total

    def update(self, msg: dict):
        gpu = msg["gpu"]
        phase = msg["phase"]
        case = msg["case"]
        status = msg["status"]
        duration = msg.get("duration", 0)

        if status == "running":
            self.gpu_status[gpu] = f"{phase:12s} {case}"
        else:
            self.gpu_status[gpu] = f"{phase:12s} {case} [{status:>8s} {duration:.1f}s]"
            if case not in self.completed_cases:
                self.completed_cases.add(case)
                self.completed = len(self.completed_cases)
        self._render()

    def _render(self):
        if not self.is_tty:
            return
        for _ in range(self._last_lines):
            sys.stderr.write("\033[1A\033[2K")
        lines = []
        for gpu_id in range(self.n_gpus):
            lines.append(f"  [GPU {gpu_id}] {self.gpu_status[gpu_id]}")
        if self.total > 0:
            pct = self.completed / self.total
            bar_width = 40
            filled = int(bar_width * pct)
            bar = "━" * filled + "╸" + " " * max(0, bar_width - filled - 1)
            lines.append(f"  {bar} {self.completed}/{self.total} ({pct * 100:.1f}%)")
        for line in lines:
            sys.stderr.write(line + "\n")
        self._last_lines = len(lines)

    def finish(self):
        if self.is_tty and self._last_lines > 0:
            for _ in range(self._last_lines):
                sys.stderr.write("\033[1A\033[2K")


# Result aggregation


def aggregate_results(raw_results: list[dict], ops: list[dict]) -> dict[str, Any]:
    op_results: dict[str, dict] = {}
    for op in ops:
        op_results[op["id"]] = {
            "accuracy": {
                "status": "NotFound",
                "total": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "duration_s": 0,
            },
            "performance": {"status": "NotFound", "cases": [], "duration_s": 0},
        }

    for result in raw_results:
        op_id = result.get("op_id")
        if op_id not in op_results:
            continue
        phase = result.get("phase")
        r = result.get("result", {})

        if phase == "accuracy":
            acc = op_results[op_id]["accuracy"]
            acc["total"] += r.get("total", 0)
            acc["passed"] += r.get("passed", 0)
            acc["failed"] += r.get("failed", 0)
            acc["skipped"] += r.get("skipped", 0)
            acc["duration_s"] += result.get("duration", 0)
            if acc["failed"] > 0:
                acc["status"] = "Failed"
            elif acc["passed"] > 0:
                acc["status"] = "Passed"
            elif acc["skipped"] > 0:
                acc["status"] = "Skipped"

        elif phase == "performance":
            perf = op_results[op_id]["performance"]
            perf["cases"].append(
                {
                    "nx": result.get("nx", 0),
                    "batch": result.get("batch", 0),
                    "speedup": r.get("speedup", 0),
                    "latency_flagfft_ms": r.get("latency_flagfft_ms", 0),
                    "latency_ref_ms": r.get("latency_ref_ms", 0),
                }
            )
            perf["duration_s"] += result.get("duration", 0)
            if r.get("status") == "Passed":
                perf["status"] = "Passed"

    return op_results


def compute_speedup_stats(op_results: dict) -> dict:
    """Compute aggregate speedup statistics across all operators."""
    all_speedups = []
    for op_result in op_results.values():
        for case in op_result.get("performance", {}).get("cases", []):
            if case.get("speedup", 0) > 0:
                all_speedups.append(case["speedup"])

    if not all_speedups:
        return {"count": 0}

    log_sum = sum(math.log(s) for s in all_speedups)
    geo_mean = math.exp(log_sum / len(all_speedups))

    return {
        "count": len(all_speedups),
        "geometric_mean_speedup": round(geo_mean, 4),
        "min_speedup": round(min(all_speedups), 4),
        "max_speedup": round(max(all_speedups), 4),
    }


def write_summary(
    output_path: Path, op_results: dict, config: dict, total_duration: float
):
    summary = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "config": config,
        "result": op_results,
        "summary": {
            "total_ops": len(op_results),
            "accuracy_passed": sum(
                1 for r in op_results.values() if r["accuracy"]["status"] == "Passed"
            ),
            "accuracy_failed": sum(
                1 for r in op_results.values() if r["accuracy"]["status"] == "Failed"
            ),
            "performance_passed": sum(
                1 for r in op_results.values() if r["performance"]["status"] == "Passed"
            ),
            "performance_failed": sum(
                1 for r in op_results.values() if r["performance"]["status"] == "Failed"
            ),
            "total_duration_s": round(total_duration, 1),
            "speedup_stats": compute_speedup_stats(op_results),
        },
    }
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {output_path}")


def main() -> int:
    args = parse_args()

    ops = load_operators(ROOT / "conf" / "operators.yaml")
    matrix = load_test_matrix(ROOT / "conf" / "test_matrix.yaml")

    stages = set(args.stages.split(","))
    ops = [
        op
        for op in ops
        if any(list(s.keys())[0] in stages for s in (op.get("stages") or []))
    ]

    if args.ops:
        op_ids = set(args.ops.split(","))
        ops = [op for op in ops if op["id"] in op_ids]

    if not ops:
        print("ERROR: no operators match the filter criteria")
        return 1

    cases = expand_test_cases(ops, matrix, args.combination)
    print(f"Expanded {len(cases)} test cases from {len(ops)} operators")

    if args.gpus == "all":
        gpu_ids = list(range(os.cpu_count() or 1))
    else:
        gpu_ids = [int(g) for g in args.gpus.split(",")]

    build_dir = Path(args.build_dir)

    print(f"Using GPUs: {gpu_ids}")
    print(f"Build dir: {build_dir}")
    print(f"Combination: {args.combination}")
    print(f"Accuracy: {not args.performance_only}")
    print(f"Performance: {not args.accuracy_only}")
    print()

    work_queue: multiprocessing.Queue = multiprocessing.Queue()
    display_queue: multiprocessing.Queue = multiprocessing.Queue()

    for case in cases:
        work_queue.put(case)

    workers = []
    for gpu_id in gpu_ids:
        p = multiprocessing.Process(
            target=worker_proc,
            args=(
                gpu_id,
                work_queue,
                display_queue,
                build_dir,
                args.timeout,
                args.accuracy_only,
                args.performance_only,
                args.warmup,
                args.iters,
            ),
        )
        p.start()
        workers.append(p)

    display = LiveDisplay(len(gpu_ids))
    display.set_total(len(cases))

    raw_results = []
    start_time = time.monotonic()

    while any(w.is_alive() for w in workers):
        try:
            msg = display_queue.get(timeout=0.5)
            display.update(msg)
            if "result" in msg:
                raw_results.append(msg)
        except queue.Empty:
            pass

    # Drain remaining messages
    while not display_queue.empty():
        try:
            msg = display_queue.get_nowait()
            display.update(msg)
            if "result" in msg:
                raw_results.append(msg)
        except queue.Empty:
            break

    for p in workers:
        p.join(timeout=30)

    display.finish()
    total_duration = time.monotonic() - start_time

    op_results = aggregate_results(raw_results, ops)
    config = {
        "combination": args.combination,
        "stages": list(stages),
        "gpus": gpu_ids,
        "accuracy_only": args.accuracy_only,
        "performance_only": args.performance_only,
    }
    write_summary(Path(args.output), op_results, config, total_duration)

    acc_passed = sum(
        1 for r in op_results.values() if r["accuracy"]["status"] == "Passed"
    )
    acc_failed = sum(
        1 for r in op_results.values() if r["accuracy"]["status"] == "Failed"
    )
    perf_passed = sum(
        1 for r in op_results.values() if r["performance"]["status"] == "Passed"
    )
    perf_failed = sum(
        1 for r in op_results.values() if r["performance"]["status"] == "Failed"
    )

    print(f"\n{'=' * 60}")
    print(f"Accuracy:    {acc_passed} passed, {acc_failed} failed")
    print(f"Performance: {perf_passed} passed, {perf_failed} failed")
    print(f"Duration:    {total_duration:.1f}s")
    print(f"{'=' * 60}")

    return 1 if acc_failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
