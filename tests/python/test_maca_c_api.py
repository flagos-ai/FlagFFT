# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Opt-in, same-process MACA production C API checks against the runner's oracle.

Required: FLAGFFT_TEST_MACA=1, FLAGFFT_TEST_LIBRARY, FLAGFFT_TEST_MACA_EXE_DIR,
FLAGFFT_TEST_MACA_OUTPUT_DIR and an isolated TRITON_CACHE_DIR below EXE_DIR.
Copy the Python executable into EXE_DIR (a symlink does not isolate /proc/self/exe)
and run with PYTHONHOME=/opt/conda on the MACA validation container.

Optional filters: FLAGFFT_TEST_MACA_{SHAPES,APIS,DIRECTIONS,SCALES}, all comma
separated, with scales also accepting "all". Defaults are the configured 1D
single CT/prime cases, all six APIs, their legal directions and matrix scales.
FLAGFFT_TEST_MACA_RESULTS_ROOT defaults to the workspace results/ directory;
set it to /workspace/FlagFFT-results for the remote validation mount.

This is correctness only. The validation coordinator must hold the GPU slot;
do not run pytest-xdist or change a variant's environment within the process.
"""
from __future__ import annotations

import ctypes
import fcntl
import hashlib
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "flagfft_maca_accuracy_oracle", ROOT / "tools" / "run_tests.py"
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)
pytestmark = pytest.mark.skipif(
    os.environ.get("FLAGFFT_TEST_MACA") != "1",
    reason="requires explicit FLAGFFT_TEST_MACA=1",
)


def selected_cases():
    def selection(name, allowed):
        raw = os.environ.get(name)
        if raw is None:
            return set(allowed)
        values = {part.strip() for part in raw.split(",") if part.strip()}
        if not values or values - set(allowed):
            raise pytest.UsageError(f"{name}: expected values from {sorted(allowed)}")
        return values

    apis = selection("FLAGFFT_TEST_MACA_APIS", RUNNER.DIRECTIONS)
    directions = selection("FLAGFFT_TEST_MACA_DIRECTIONS", {"forward", "inverse"})
    ops = [op for op in RUNNER.load_operators(ROOT / "conf/operators.yaml")
           if op["api"] in apis]
    cases = RUNNER.expand_test_cases(
        ops, RUNNER.load_test_matrix(ROOT / "conf/test_matrix.yaml"),
        "1d_ct_single,1d_prime_single", os.environ.get("FLAGFFT_TEST_MACA_SCALES"),
        RUNNER.parse_shape_filter(os.environ.get("FLAGFFT_TEST_MACA_SHAPES")),
    )
    cases = [case for case in cases if case["direction"] in directions]
    if not cases:
        raise pytest.UsageError("MACA C API filters selected no configured legal cases")
    return cases


CASES = selected_cases()


def required_path(name):
    value = os.environ.get(name)
    assert value and Path(value).is_absolute(), f"set {name} to an absolute path"
    return Path(value).resolve()


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def variant_environment():
    return {key: value for key, value in sorted(os.environ.items())
            if (key.startswith("FLAGFFT_MACA_")
                or key.startswith("TRITON_")
                or key in {"CUDA_VISIBLE_DEVICES", "MACA_VISIBLE_DEVICES",
                           "MC_VISIBLE_DEVICES", "PYTHONHOME", "PYTHONPATH",
                           "FLAGFFT_EXECUTION_POLICY", "FLAGFFT_PYTHON",
                           "FLAGFFT_PACKED_REAL", "FLAGFFT_TUNE_DISABLE", "FLAGFFT_TUNE_DB"})}


def bind_api(library):
    api = ctypes.CDLL(str(library))
    handle, integer, pointer = ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p
    api.flagfftPlan1d.argtypes = [ctypes.POINTER(handle), integer, integer, integer]
    api.flagfftPlan1d.restype = integer
    api.flagfftSetStream.argtypes = [handle, pointer]
    api.flagfftSetStream.restype = integer
    api.flagfftDestroy.argtypes = [handle]
    api.flagfftDestroy.restype = integer
    api.flagfftGetPlanDescription.argtypes = [handle]
    api.flagfftGetPlanDescription.restype = ctypes.c_char_p
    for operation in RUNNER.TYPE_CODES:
        call = getattr(api, "flagfftExec" + operation.upper())
        call.argtypes = [handle, pointer, pointer] + (
            [integer] if operation in {"c2c", "z2z"} else [])
        call.restype = integer
    return api


@pytest.fixture(scope="module")
def runtime():
    assert "PYTEST_XDIST_WORKER" not in os.environ, "use one process and one GPU slot"
    exe_dir = required_path("FLAGFFT_TEST_MACA_EXE_DIR")
    executable = Path("/proc/self/exe").resolve()
    assert executable.parent == exe_dir, (
        f"/proc/self/exe is {executable}; copy Python into {exe_dir}, do not symlink it")
    cache = required_path("TRITON_CACHE_DIR")
    assert cache != exe_dir and cache.is_relative_to(exe_dir), (
        "TRITON_CACHE_DIR must be a dedicated child of the isolated executable directory")
    assert not cache.is_relative_to(exe_dir / ".flagfft"), "keep the two caches separate"
    assert not (exe_dir / ".flagfft").is_symlink(), "the C++ cache must not redirect to another variant"
    library = required_path("FLAGFFT_TEST_LIBRARY")
    assert library.is_file(), f"missing library: {library}"
    output = required_path("FLAGFFT_TEST_MACA_OUTPUT_DIR")
    result_root = Path(os.environ.get("FLAGFFT_TEST_MACA_RESULTS_ROOT", ROOT.parent / "results"))
    assert result_root.is_absolute(), "results root must be absolute"
    assert output != result_root.resolve() and output.is_relative_to(result_root.resolve()), (
        "output must be a suite directory under the workspace results root")
    assert any(os.environ.get(name) for name in (
        "CUDA_VISIBLE_DEVICES", "MACA_VISIBLE_DEVICES", "MC_VISIBLE_DEVICES")), (
        "explicitly filter one physical device before starting Python")

    # Serialize this executable's cache ownership independently of the external
    # GPU slot lock. Never adopt unlabelled nonempty caches from another runner.
    with (exe_dir / ".flagfft-maca-test.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        codegen_spec = importlib.util.find_spec("flagfft_codegen")
        assert codegen_spec and codegen_spec.origin, "make the intended codegen package importable"
        codegen_root = Path(codegen_spec.origin).resolve().parent
        # Resolve the interpreter used by the production C++ JIT as well. The
        # test/oracle worktree need not be the tested codegen worktree.
        jit_source = json.loads(subprocess.check_output(
            [os.environ.get("FLAGFFT_PYTHON") or "python3", "-c",
             "import importlib.util,json,sys; "
             "print(json.dumps({'python':sys.executable,"
             "'module':importlib.util.find_spec('flagfft_codegen').origin}))"],
            text=True, timeout=30,
        ))
        assert Path(jit_source["module"]).resolve().parent == codegen_root, (
            "test process and production JIT resolve different codegen packages")
        jit_python = Path(jit_source["python"]).resolve()
        codegen_hash = hashlib.sha256()
        for path in sorted(codegen_root.rglob("*.py")):
            codegen_hash.update(str(path.relative_to(codegen_root)).encode())
            codegen_hash.update(path.read_bytes())
        identity = {
            "environment": variant_environment(), "library": str(library),
            "library_sha256": file_hash(library), "executable": str(executable),
            "executable_sha256": file_hash(executable), "codegen_root": str(codegen_root),
            "codegen_sha256": codegen_hash.hexdigest(),
            "jit_python": str(jit_python), "jit_python_sha256": file_hash(jit_python),
        }
        tune_disabled = os.environ.get("FLAGFFT_TUNE_DISABLE", "").lower() not in {
            "", "0", "false", "off", "no"}
        tune_db = Path(os.environ.get("FLAGFFT_TUNE_DB") or exe_dir / ".flagfft/tuned_plans.sqlite")
        identity["tune_db_sha256"] = (
            file_hash(tune_db) if not tune_disabled and tune_db.is_file() else None)
        owner = exe_dir / ".flagfft-maca-test-variant.json"
        if owner.exists():
            assert json.loads(owner.read_text()) == identity, (
                "variant identity changed; use a fresh isolated executable/cache directory")
        else:
            for directory in (exe_dir / ".flagfft", cache):
                assert not directory.exists() or not any(directory.iterdir()), (
                    f"refusing unlabelled nonempty cache: {directory}")
            RUNNER.write_json(owner, identity)
        cache.mkdir(parents=True, exist_ok=True)
        output.mkdir(parents=True, exist_ok=True)
        assert not (output / "manifest.json").exists(), "use a fresh results suite directory"
        RUNNER.write_json(output / "manifest.json", {
            "purpose": "same-process production C API correctness; no performance measurements",
            "variant": identity, "test_source_root": str(ROOT),
            "test_sha256": file_hash(Path(__file__).resolve()),
            "runner_path": str(ROOT / "tools/run_tests.py"),
            "runner_sha256": file_hash(ROOT / "tools/run_tests.py"),
            "config_sha256": {name: file_hash(ROOT / "conf" / name)
                              for name in ("operators.yaml", "test_matrix.yaml")},
            "cases": CASES,
        })

        # No device-dependent import or initialization occurs during collection.
        import torch
        import triton

        assert torch.cuda.device_count() == 1, "filter exactly one physical MACA GPU"
        torch.cuda.set_device(0)
        assert triton.runtime.driver.active.get_current_target().backend == "maca"
        stream = torch.cuda.Stream()
        state = SimpleNamespace(api=bind_api(library), torch=torch, stream=stream,
                                output=output, environment=variant_environment())
        yield state
        stream.synchronize()
        assert variant_environment() == state.environment, "variant environment changed during tests"


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["case_id"])
def test_production_c_api(runtime, case):
    assert variant_environment() == runtime.environment, "variant environment changed during tests"
    operation, shape = case["api"], tuple(case["shape"])
    source, seed = RUNNER.make_input(operation, shape, case["batch"], case["scale"])
    expected = RUNNER.numpy_reference(source, operation, shape, case["direction"])
    limits = RUNNER.accuracy_limit(operation, RUNNER.product(shape))
    raw = np.ascontiguousarray(source).view(np.uint8).reshape(-1)
    record = dict(case, seed=seed, status="Running", plan=None, metric=None, limits=limits,
                  input_dtype=str(source.dtype), numpy_dtype=str(expected.dtype),
                  input_sha256=hashlib.sha256(raw).hexdigest())
    path = runtime.output / (case["case_id"] + ".json")
    assert not path.exists(), f"refusing to overwrite {path}"
    RUNNER.write_json(path, record)
    handle = ctypes.c_void_p()
    started = time.monotonic()
    try:
        assert runtime.api.flagfftPlan1d(
            ctypes.byref(handle), shape[0], RUNNER.TYPE_CODES[operation], case["batch"]
        ) == 0, "flagfftPlan1d failed"
        assert handle.value, "flagfftPlan1d returned a null handle"
        record["plan"] = runtime.api.flagfftGetPlanDescription(handle).decode()
        RUNNER.write_json(path, record)
        assert runtime.api.flagfftSetStream(handle, runtime.stream.cuda_stream) == 0
        torch = runtime.torch
        source_device = torch.from_numpy(raw.copy()).cuda()
        dtype = RUNNER.raw_dtype(operation, is_input=False)
        output_bytes = expected.size * dtype.itemsize
        storage = torch.full((output_bytes + 128,), 0xA5, dtype=torch.uint8, device="cuda")
        output_device = storage[64:-64]
        output_device.fill_(0xFF)  # NaNs detect missing output stores.
        torch.cuda.synchronize()
        args = [handle, source_device.data_ptr(), output_device.data_ptr()]
        if operation in {"c2c", "z2z"}:
            args.append(-1 if case["direction"] == "forward" else 1)
        assert getattr(runtime.api, "flagfftExec" + operation.upper())(*args) == 0
        runtime.stream.synchronize()
        actual = output_device.cpu().numpy().copy().view(dtype).reshape(expected.shape)
        guard = storage.cpu().numpy()
        record["guards_ok"] = bool(np.all(guard[:64] == 0xA5) and np.all(guard[-64:] == 0xA5))
        record["plan"] = runtime.api.flagfftGetPlanDescription(handle).decode()
        record["output_sha256"] = hashlib.sha256(actual.view(np.uint8)).hexdigest()
        record["metric"] = RUNNER.judged_stats(
            RUNNER.error_stats(actual, expected, expected.size, case["batch"]), limits)
        record["status"] = "Passed" if record["metric"]["passed"] and record["guards_ok"] else "Failed"
        assert record["guards_ok"], "output guard overwritten"
        assert record["metric"]["passed"], record["metric"]
    except BaseException as error:
        if record["status"] != "Failed":
            record["status"] = "Error"
        record["error"] = repr(error)
        raise
    finally:
        try:
            if handle.value:
                runtime.stream.synchronize()
                result = runtime.api.flagfftDestroy(handle)
                assert result == 0, "flagfftDestroy failed"
        except BaseException as error:
            record.update(status="Error", cleanup_error=repr(error))
            raise
        finally:
            record["duration_seconds"] = time.monotonic() - started
            RUNNER.write_json(path, record)
