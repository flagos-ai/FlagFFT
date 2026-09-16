#!/usr/bin/env python3
"""Bounded, isolated FP64 diagnostics; does not change acceptance policy."""
import argparse
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import tempfile


def arithmetic_worker(layer, native_compiler=None, native_arch="ivcore11"):
    import numpy as np
    x = np.array([1 + 2.0**-40, 1 - 2.0**-40, -1 + 2.0**-40], dtype=np.float64)
    expected = (x - 1.0) * 3.0
    details = {}
    if layer == "runtime":
        with tempfile.TemporaryDirectory(prefix="flagfft-fp64-") as directory:
            binary = str(Path(directory) / "native_probe")
            corex = Path(os.environ.get("COREX_HOME", "/usr/local/corex"))
            compiler = native_compiler or (str(corex / "bin/clang++") if (corex / "bin/clang++").is_file() else "nvcc")
            command = [compiler, str(Path(__file__).with_name("fp64_runtime_probe.cu")), "-o", binary]
            if Path(compiler).name.startswith("clang"):
                sdk = Path(compiler).resolve().parent.parent
                command[1:1] = ["-x", "ivcore", f"--cuda-gpu-arch={native_arch}"]
                command += [f"-L{sdk / 'lib64'}", "-lcudart"]
            details["compile_command"] = command
            compilation = subprocess.run(command, stdout=sys.stderr, stderr=sys.stderr)
            if compilation.returncode:
                return {"status": "failed", "stage": "compile", **details}
            if not Path(binary).is_file():
                return {"status": "unknown", "stage": "compile",
                        "reason": "compiler returned success without creating a binary", **details}
            output = subprocess.run([binary], capture_output=True, text=True)
            if output.returncode:
                print(output.stderr, file=sys.stderr)
                return {"status": "failed", "stage": "execute", **details}
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
            "expected": expected.tolist(), "scope": "FP64 subtraction/multiplication; 3 inputs", **details}


def run(cmd, directory, timeout):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, start_new_session=True)
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = proc.communicate()
    (directory / "stdout.txt").write_text(stdout)
    (directory / "stderr.txt").write_text(stderr)
    if timed_out:
        return {"status": "unknown", "reason": "probe timed out"}
    if proc.returncode:
        return {"status": "failed", "returncode": proc.returncode,
                "reason": "see stderr.txt; failure does not prove hardware lacks FP64"}
    return {"status": "passed", "stdout": stdout}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--worker", choices=("runtime", "triton"))
    parser.add_argument("--native-compiler", help="SDK compiler override; CoreX clang++ is preferred when installed")
    parser.add_argument("--native-arch", default="ivcore11", help="CoreX native-probe target")
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(arithmetic_worker(args.worker, args.native_compiler, args.native_arch)))
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
              "environment": {"python": sys.version, "git_commit": acceptance.git_commit(Path(__file__).resolve().parents[1]),
                              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                              "ix_visible_devices": os.environ.get("IX_VISIBLE_DEVICES")},
              "acceptance_policy": "unchanged; IX FP64 remains skipped"}
    for layer in ("runtime", "triton"):
        directory = root / layer
        directory.mkdir()
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", layer, "--native-arch", args.native_arch]
        if args.native_compiler:
            command += ["--native-compiler", args.native_compiler]
        result = run(command, directory, args.timeout)
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
