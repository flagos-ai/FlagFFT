#!/usr/bin/env python3
"""FlagFFT unified test runner."""

from __future__ import annotations

import argparse
import glob
import json
import math
import multiprocessing
import os
import platform
import queue
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]

# Global state for environment info, worker tracking, and signal handling
ENV_INFO: dict[str, Any] = {}
WORKER_PROCESSES: list[multiprocessing.Process] = []
INTERRUPTED = False

# ANSI color variables -- overridden by init_colors()
RED = ""
GREEN = ""
YELLOW = ""
CYAN = ""
DIM = ""
NC = ""


def init_colors(mode: str) -> None:
    """Set ANSI color codes based on --color mode."""
    global RED, GREEN, YELLOW, CYAN, DIM, NC
    if mode == "never":
        RED = GREEN = YELLOW = CYAN = DIM = NC = ""
    elif mode == "always" or (mode == "auto" and sys.stderr.isatty()):
        RED = "\033[31m"
        GREEN = "\033[32m"
        YELLOW = "\033[33m"
        CYAN = "\033[36m"
        DIM = "\033[2m"
        NC = "\033[0m"


def pinfo(msg: str) -> None:
    print(f"{GREEN}[INFO]{NC} {msg}")


def pwarn(msg: str) -> None:
    print(f"{YELLOW}[WARN]{NC} {msg}", file=sys.stderr)


def perror(msg: str) -> None:
    print(f"{RED}[ERROR]{NC} {msg}", file=sys.stderr)


def probe_env() -> None:
    """Probe runtime environment and store results in ENV_INFO."""
    ENV_INFO["architecture"] = platform.machine()
    ENV_INFO["python"] = platform.python_version()

    # PyTorch
    try:
        import torch

        ENV_INFO["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_name": (
                torch.cuda.get_device_name() if torch.cuda.is_available() else "N/A"
            ),
            "device_count": (
                torch.cuda.device_count() if torch.cuda.is_available() else 0
            ),
        }
    except ImportError:
        ENV_INFO["torch"] = {
            "version": "N/A",
            "cuda_available": False,
            "device_name": "N/A",
            "device_count": 0,
        }

    # Triton
    try:
        import triton

        ENV_INFO["triton"] = {"version": triton.__version__}
    except ImportError:
        ENV_INFO["triton"] = {"version": "N/A"}


def terminate_workers() -> None:
    """Send SIGTERM to all tracked worker processes, then force-kill survivors."""
    for p in WORKER_PROCESSES:
        if p.is_alive():
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass
    for p in WORKER_PROCESSES:
        p.join(timeout=5)
        if p.is_alive():
            p.kill()


def cleanup_intermediate_files() -> None:
    """Remove temporary accuracy JSON files left behind by workers."""
    for pattern in ["/tmp/flagfft_acc_*.json"]:
        for f in glob.glob(pattern):
            try:
                os.unlink(f)
            except OSError:
                pass


def handle_interrupt(signum, frame) -> None:
    """Handle SIGINT/SIGTERM: terminate workers, clean up, and exit."""
    global INTERRUPTED
    if INTERRUPTED:
        return
    INTERRUPTED = True
    pwarn("Interrupted. Cleaning up ...")
    terminate_workers()
    cleanup_intermediate_files()
    pwarn("Cleanup done.")
    sys.exit(1)


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
    parser.add_argument(
        "--op-list-file",
        default=None,
        help="Path to operator list file (one ID per line, # for comments)",
    )
    parser.add_argument(
        "--start", default=None, help="ID of the first operator to test"
    )
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
    parser.add_argument(
        "--output-dir", default="results", help="Relative path to root for test data"
    )
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--warmup", type=int, default=10, help="Benchmark warmup iterations"
    )
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--dump-output",
        action="store_true",
        help="Dump stdout/stderr of each test to log files",
    )
    parser.add_argument(
        "--color",
        choices=["auto", "always", "never"],
        default="auto",
        help="Color mode for terminal output",
    )
    return parser.parse_args()


# Test execution functions

# Operators that include direction (fwd/inv) in their binary name
_OPS_WITH_DIRECTION = {"c2c", "z2z"}


def build_accuracy_cmd(case: dict, build_dir: Path) -> tuple[list[str], str]:
    # Build binary name based on algorithm:
    # - 2D tests use a single binary: test_2d_correctness
    # - 1D tests: test_exec_{type}_{direction}_{algo}_{batch_mode}
    #   For types with direction (c2c, z2z): include fwd/inv
    #   For types without direction (r2c, c2r, d2z, z2d, r2c_c2r, d2z_z2d): no direction
    #   batch_mode: batch=1 -> _s, batch>1 -> _b
    ctest_base = case["ctest"]
    algo = case["algo"]

    if algo == "2d":
        # 2D tests use a single binary for all types
        binary_name = ctest_base
    else:
        batch_mode = "s" if case["batch"] == 1 else "b"
        # Check if this operator has direction in binary name
        # Extract the type prefix from op_id (e.g., "c2c_1d" -> "c2c")
        type_prefix = case["op_id"].split("_")[0]
        has_direction = type_prefix in _OPS_WITH_DIRECTION

        if has_direction:
            dir_str = "fwd" if case["direction"] == "forward" else "inv"
            binary_name = f"{ctest_base}_{dir_str}_{algo}_{batch_mode}"
        else:
            binary_name = f"{ctest_base}_{algo}_{batch_mode}"

    binary = build_dir / "ctest" / binary_name
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
    # Vendor-aware GPU selection: FlagFFT targets CUDA only.
    # For multi-vendor support (ROCm, etc.), extend this based on ENV_INFO.
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
            start_new_session=True,
        )
        return result.returncode, result.stdout, result.stderr, time.monotonic() - start
    except subprocess.TimeoutExpired:
        return -100, "", "TIMEOUT", time.monotonic() - start


def parse_accuracy_result(json_file: str, data_file: str = "") -> dict[str, Any]:
    try:
        with open(json_file) as f:
            data = json.load(f)
        total = data.get("total", 0)
        passed = data.get("passed", 0)
        failed = data.get("failed", 0)
        skipped = data.get("skipped", 0)
        status = "Failed" if failed > 0 else ("Passed" if passed > 0 else "Skipped")
        result: dict[str, Any] = {
            "status": status,
            "total": total,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "duration_ms": data.get("duration_ms", 0),
            "details": data.get("failures", []),
        }
        if data_file:
            result["data_file"] = data_file
        return result
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return {"status": "Error", "error": str(e)}


def parse_perf_result(output: str, case: dict | None = None) -> dict[str, Any]:
    try:
        data = json.loads(output)
        cases = data.get("cases", [])
        if not cases:
            return {"status": "Error", "error": "no cases in output"}
        raw_case = cases[0]
        timing = raw_case.get("timing", {})
        speedup = timing.get("speedup", 0)
        passed = (
            speedup > 0
            and timing.get("flagfft_median_ms", 0) > 0
            and timing.get("ref_median_ms", 0) > 0
        )

        # Build shape key from the test case (e.g., "[1024]" or "[1024]batch=2")
        if case is not None:
            nx = case.get("nx", 0)
            batch = case.get("batch", 1)
            ny = case.get("ny", 0)
            if ny > 0:
                shape_key = f"[{nx},{ny}]"
            else:
                shape_key = f"[{nx}]"
            if batch > 1:
                shape_key += f"batch={batch}"
        else:
            shape_key = "[unknown]"

        details_entry = {
            "base": timing.get("ref_median_ms", 0),
            "gems": timing.get("flagfft_median_ms", 0),
            "speedup": speedup,
        }

        return {
            "status": "Passed" if passed else "Failed",
            "speedup": speedup,
            "data": {
                "default": {
                    "result": "OK" if passed else "FAIL",
                    "details": {shape_key: details_entry},
                    "speedup": speedup,
                }
            },
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
    output_dir,
):
    while True:
        try:
            case = work_queue.get_nowait()
        except queue.Empty:
            break

        case_id = f"{case['op_id']} nx={case['nx']} batch={case['batch']}"
        op_dir = output_dir / case["op_id"]
        op_dir.mkdir(parents=True, exist_ok=True)

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
            data_file = f"{case['op_id']}/accuracy_result.json"
            if rc == -100:
                acc_result = {"status": "Timeout", "duration": elapsed}
            elif rc in (0, 77):
                acc_result = parse_accuracy_result(jf, data_file=data_file)
                acc_result["duration"] = elapsed
            else:
                acc_result = {"status": "Error", "rc": rc, "stderr": stderr[:500]}
            # Save per-op accuracy result file
            try:
                with open(op_dir / "accuracy_result.json", "w") as f:
                    json.dump(acc_result, f, indent=2)
            except OSError:
                pass
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
            data_file = f"{case['op_id']}/performance_result.json"
            if rc == -100:
                perf_result = {"status": "Timeout", "duration": elapsed}
            elif rc in (0, 77):
                perf_result = parse_perf_result(stdout, case=case)
                perf_result["duration"] = elapsed
                perf_result["data_file"] = data_file
            else:
                perf_result = {"status": "Error", "rc": rc, "stderr": stderr[:500]}
            # Save per-op performance result file
            try:
                with open(op_dir / "performance_result.json", "w") as f:
                    json.dump(perf_result, f, indent=2)
            except OSError:
                pass
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
                "duration": 0,
                "data_file": "",
                "details": [],
            },
            "performance": {
                "status": "NotFound",
                "duration": 0,
                "data_file": "",
                "data": {},
            },
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
            acc["duration"] += result.get("duration", 0)
            if r.get("data_file"):
                acc["data_file"] = r["data_file"]
            # Merge details from each case result
            case_details = r.get("details", [])
            if case_details:
                acc["details"].extend(case_details)
            if acc["failed"] > 0:
                acc["status"] = "Failed"
            elif acc["passed"] > 0:
                acc["status"] = "Passed"
            elif acc["skipped"] > 0:
                acc["status"] = "Skipped"

        elif phase == "performance":
            perf = op_results[op_id]["performance"]
            perf["duration"] += result.get("duration", 0)
            if r.get("data_file"):
                perf["data_file"] = r["data_file"]
            # Merge data (dtype -> details) from each case result
            perf_data = r.get("data", {})
            for dtype_key, dtype_info in perf_data.items():
                if dtype_key not in perf["data"]:
                    perf["data"][dtype_key] = {
                        "result": dtype_info.get("result", "OK"),
                        "details": {},
                        "speedup": dtype_info.get("speedup", 0),
                    }
                perf["data"][dtype_key]["details"].update(dtype_info.get("details", {}))
                # Update overall speedup to latest
                perf["data"][dtype_key]["speedup"] = dtype_info.get("speedup", 0)
            if r.get("status") == "Passed":
                perf["status"] = "Passed"

    return op_results


def compute_speedup_stats(op_results: dict) -> dict:
    """Compute aggregate speedup statistics across all operators."""
    all_speedups = []
    for op_result in op_results.values():
        for dtype_info in op_result.get("performance", {}).get("data", {}).values():
            for detail in dtype_info.get("details", {}).values():
                if detail.get("speedup", 0) > 0:
                    all_speedups.append(detail["speedup"])

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
        "env": ENV_INFO,
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
            "total_duration": round(total_duration, 1),
            "speedup_stats": compute_speedup_stats(op_results),
        },
    }
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {output_path}")


def read_op_list_file(path: str) -> list[str]:
    """Read operator IDs from a file, one per line. Lines starting with # are comments."""
    op_ids = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            op_ids.append(line)
    return op_ids


def main() -> int:
    args = parse_args()

    # Probe environment info before anything else
    probe_env()

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)

    init_colors(args.color)

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

    if args.op_list_file:
        file_op_ids = set(read_op_list_file(args.op_list_file))
        ops = [op for op in ops if op["id"] in file_op_ids]

    if args.start:
        ops = [op for op in ops if op["id"] >= args.start]

    if not ops:
        perror("no operators match the filter criteria")
        return 1

    cases = expand_test_cases(ops, matrix, args.combination)
    print(f"Expanded {len(cases)} test cases from {len(ops)} operators")

    if args.gpus == "all":
        gpu_ids = list(range(os.cpu_count() or 1))
    else:
        gpu_ids = [int(g) for g in args.gpus.split(",")]

    build_dir = Path(args.build_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
                output_dir,
            ),
        )
        p.start()
        workers.append(p)
        WORKER_PROCESSES.append(p)

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
    write_summary(output_dir / "summary.json", op_results, config, total_duration)

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
