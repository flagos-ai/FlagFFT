#!/usr/bin/env python3
"""Bounded, isolated FP64 diagnostics; does not change acceptance policy."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def arithmetic_worker(layer):
    import numpy as np
    x = np.array([1 + 2.0**-40, 1 - 2.0**-40, -1 + 2.0**-40], dtype=np.float64)
    expected = (x - 1.0) * 3.0
    if layer == "runtime":
        with tempfile.TemporaryDirectory(prefix="flagfft-fp64-") as directory:
            binary = str(Path(directory) / "native_probe")
            subprocess.run(["nvcc", str(Path(__file__).with_name("fp64_runtime_probe.cu")),
                            "-o", binary], check=True, stdout=sys.stderr, stderr=sys.stderr)
            output = subprocess.run([binary], capture_output=True, text=True, check=True)
            actual = np.array([float(v) for v in output.stdout.split()], dtype=np.float64)
    else:
        import torch
        import triton
        import triton.language as tl

        @triton.jit
        def kernel(X, Y):
            i = tl.arange(0, 64)
            v = tl.load(X + i, i < 3, 0)
            tl.store(Y + i, (v - 1.0) * 3.0, i < 3)

        dx = torch.as_tensor(x, device="cuda")
        dy = torch.empty_like(dx)
        kernel[(1,)](dx, dy, num_warps=1)
        actual = dy.cpu().numpy()
    passed = bool(np.array_equal(actual, expected))
    return {"status": "passed" if passed else "failed", "actual": actual.tolist(),
            "expected": expected.tolist(), "scope": "FP64 subtraction/multiplication; 3 inputs"}


def run(cmd, directory, timeout):
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        (directory / "stdout.txt").write_text(proc.stdout)
        (directory / "stderr.txt").write_text(proc.stderr)
        if proc.returncode:
            return {"status": "failed", "returncode": proc.returncode,
                    "reason": "see stderr.txt; failure does not prove hardware lacks FP64"}
        return {"status": "passed", "stdout": proc.stdout}
    except subprocess.TimeoutExpired:
        return {"status": "unknown", "reason": "probe timed out"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--worker", choices=("runtime", "triton"))
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(arithmetic_worker(args.worker)))
        return
    if args.output_dir is None:
        parser.error("--output-dir is required")
    import numpy as np
    import run_tests as acceptance
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    build = args.build_dir.resolve()
    info = subprocess.run([str(build / "flagfft-cli"), "device-info", "--json"],
                          capture_output=True, text=True, timeout=30, check=True)
    report = {"device": json.loads(info.stdout), "fp64": {},
              "acceptance_policy": "unchanged; IX FP64 remains skipped"}
    for layer in ("runtime", "triton"):
        directory = root / layer
        directory.mkdir()
        result = run([sys.executable, str(Path(__file__).resolve()), "--worker", layer], directory, args.timeout)
        if result["status"] == "passed":
            try:
                result = json.loads(result["stdout"])
            except ValueError:
                result = {"status": "unknown", "reason": "invalid worker output"}
        report["fp64"][layer] = result
    for layer in ("flagfft", "platform"):
        results = []
        for api in ("z2z", "d2z", "z2d"):
            for direction in acceptance.DIRECTIONS[api]:
                directory = root / f"{layer}_{api}_{direction}"
                directory.mkdir()
                shape = (16,)
                value, _ = acceptance.make_input(api, shape, 1, 1.0)
                value.tofile(directory / "input.bin")
                case = {"api": api, "shape": shape, "batch": 1, "direction": direction}
                result = run(acceptance.build_accuracy_cmd(case, build / "ctest/numpy_fft_capture",
                                                           directory, layer), directory, args.timeout)
                if result["status"] == "passed":
                    dtype = np.float64 if api == "z2d" else np.complex128
                    actual = np.fromfile(directory / f"{layer}.bin", dtype=dtype)
                    expected = acceptance.numpy_reference(value, api, shape, direction).ravel()
                    passed = actual.shape == expected.shape and np.allclose(actual, expected, rtol=1e-12, atol=1e-12)
                    result = {"status": "passed" if passed else "failed",
                              "max_abs_error": float(np.max(np.abs(actual - expected))) if actual.shape == expected.shape else None}
                result.update(api=api, direction=direction, shape=shape)
                results.append(result)
        report["fp64"][layer] = results
    (root / "capabilities.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
