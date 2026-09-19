# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "flagfft_run_tests", ROOT / "tools" / "run_tests.py"
)
assert SPEC is not None and SPEC.loader is not None
RUN_TESTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN_TESTS)


@pytest.fixture
def operators():
    return RUN_TESTS.load_operators(ROOT / "conf" / "operators.yaml")


@pytest.fixture
def matrix():
    return RUN_TESTS.load_test_matrix(ROOT / "conf" / "test_matrix.yaml")


# A CPU stand-in for the native capture.  It speaks the same stdin/stdout
# protocol as capture.cpp, so the runner's pipe path, plan parsing and
# comparison are exercised for real without a GPU.
FAKE_CAPTURE = """#!{python}
import os
import sys

import numpy as np

sys.path.insert(0, {tools!r})
import run_tests

args = dict(arg[2:].split("=", 1) for arg in sys.argv[1:])
if args.get("input") != "-" or args.get("output-dir") != "-":
    print("fake capture requires stream mode", file=sys.stderr)
    sys.exit(2)
implementation = args["implementation"]
api, direction = args["api"], args["direction"]
shape = tuple(int(n) for n in args["shape"].split(","))
batch = int(args["batch"])
mode = os.environ.get("FAKE_CAPTURE_PLATFORM_MODE", "ok")
if implementation == "platform" and mode == "error":
    print("platform capture failed", file=sys.stderr)
    sys.exit(1)

in_shape = run_tests.input_shape(api, shape, batch)
in_dtype = run_tests.raw_dtype(api, is_input=True)
wanted = run_tests.product(in_shape) * in_dtype.itemsize
raw = sys.stdin.buffer.read(wanted)
if len(raw) != wanted:
    print(f"short input: {{len(raw)}} of {{wanted}}", file=sys.stderr)
    sys.exit(3)
value = np.frombuffer(raw, dtype=in_dtype).reshape(in_shape)

hang = os.environ.get("FAKE_CAPTURE_HANG", "")
hang_direction = os.environ.get("FAKE_CAPTURE_HANG_DIRECTION", "inverse")
if hang == implementation and direction == hang_direction:
    pid_file = os.environ.get("FAKE_CAPTURE_PID_FILE")
    if pid_file:
        with open(pid_file, "w") as stream:
            stream.write(str(os.getpid()))
    import time

    time.sleep(60)

reference = run_tests.numpy_reference(value, api, shape, direction)
out_dtype = run_tests.raw_dtype(api, is_input=False)
output = np.ascontiguousarray(reference.astype(out_dtype))
if implementation == "platform" and mode == "corrupt":
    output.reshape(-1)[0] += 100

sys.stderr.write(
    "===== FLAGFFT PLAN BEGIN =====\\n" + {plan!r} + "===== FLAGFFT PLAN END =====\\n"
)
sys.stderr.flush()
sys.stdout.buffer.write(output.tobytes())
sys.stdout.buffer.flush()
"""

PLAN = 'LeafPlan(n=256, factors=[4,4,4,4])\nCompiledRawLeaf(kernel="fft")\n'


def fake_capture(tmp_path):
    """Write the CPU capture stand-in and return its executable path."""
    script = tmp_path / "fake_capture"
    script.write_text(
        FAKE_CAPTURE.format(
            python=sys.executable,
            tools=str(ROOT / "tools"),
            plan=PLAN,
        )
    )
    script.chmod(0o755)
    return script


def summary_op_entry(summary, op_id):
    """Look one operator up in the flat, FlagGems-shaped summary array."""
    assert isinstance(summary, list), "summary.json must be a flat array"
    return next(entry for entry in summary if entry["operator"] == op_id)


def test_acceptance_has_36_ordered_operators_and_distinct_cases(operators, matrix):
    apis = ("c2c", "c2r", "r2c", "z2z", "z2d", "d2z")
    expected_ids = [f"{group}_{api}" for group in RUN_TESTS.GROUPS for api in apis]
    assert [op["id"] for op in operators] == expected_ids
    assert "combinations" not in matrix
    cases = RUN_TESTS.expand_all_test_cases(operators, matrix)
    ct_ref = next(op["sizes"] for op in operators if op.get("algorithm") == "ct")
    prime_ref = next(op["sizes"] for op in operators if op.get("algorithm") == "prime")
    expected_count = (
        8
        * (
            len(matrix[ct_ref])
            * (len(matrix["batches"]["single"]) + len(matrix["batches"]["batch"]))
            + len(matrix[prime_ref])
            * (len(matrix["batches"]["single"]) + len(matrix["batches"]["batch"]))
            + len(matrix["sizes_2d"]) * len(matrix["batches"]["2d"])
            + len(matrix["sizes_3d"]) * len(matrix["batches"]["3d"])
        )
        * len(matrix.get("scales", [1.0]))
    )
    assert len(cases) == expected_count
    assert len({case["case_id"] for case in cases}) == len(cases)
    assert {case["op_id"] for case in cases} == set(expected_ids)
    for op in operators:
        op_cases = [case for case in cases if case["op_id"] == op["id"]]
        batch_key = op["batch"] if op["rank"] == 1 else f"{op['rank']}d"
        assert {case["batch"] for case in op_cases} == set(matrix["batches"][batch_key])
        assert {case["direction"] for case in op_cases} == set(
            RUN_TESTS.DIRECTIONS[op["api"]]
        )
        assert len(op_cases) == len(matrix[op["sizes"]]) * len(
            matrix["batches"][batch_key]
        ) * (len(RUN_TESTS.DIRECTIONS[op["api"]])) * len(matrix.get("scales", [1.0]))


def test_group_selection_and_alias_do_not_duplicate_cases(operators, matrix):
    cases = RUN_TESTS.expand_test_cases(operators, matrix, "1d_prime_batch,1d_bs_batch")
    prime_ref = next(op["sizes"] for op in operators if op.get("algorithm") == "prime")
    assert len(cases) == len(matrix[prime_ref]) * len(matrix["batches"]["batch"]) * 8
    assert {case["algorithm"] for case in cases} == {"prime"}
    assert {case["batch"] for case in cases} == set(matrix["batches"]["batch"])
    assert {case["rank"] for case in cases} == {1}


def test_ix_backend_detection_and_fp64_policy(tmp_path, operators, matrix):
    build_dir = tmp_path / "ix-build"
    build_dir.mkdir()
    (build_dir / "CMakeCache.txt").write_text("BACKEND:STRING=IX\n")
    assert RUN_TESTS.detect_backend(build_dir) == "ix"

    assert RUN_TESTS.UNSUPPORTED_APIS_BY_BACKEND["ix"] == {
        "z2z",
        "z2d",
        "d2z",
    }
    c2c = next(op for op in operators if op["api"] == "c2c")
    z2z = next(op for op in operators if op["api"] == "z2z")
    assert RUN_TESTS.operator_skip_reason(c2c, "ix") is None
    reason = RUN_TESTS.operator_skip_reason(z2z, "ix")
    assert reason and "FP64" in reason

    cases = RUN_TESTS.expand_test_cases([z2z], matrix, "1d_ct_single")[:1]
    result = RUN_TESTS.aggregate_results(
        [],
        [z2z],
        cases,
        True,
        True,
        {z2z["id"]},
        {z2z["id"]: reason},
    )
    op_result = result[z2z["id"]]
    assert op_result["accuracy"]["status"] == "Skipped"
    assert op_result["performance"]["status"] == "Skipped"
    assert op_result["accuracy"]["policy_skipped"]
    assert all(
        case["status"] == "Skipped" for case in op_result["accuracy"]["cases"].values()
    )
    assert RUN_TESTS.requested_phases_passed(result, True, True)


def test_npu_backend_detection_and_fp64_policy(tmp_path, operators, matrix):
    build_dir = tmp_path / "npu-build"
    build_dir.mkdir()
    (build_dir / "CMakeCache.txt").write_text("BACKEND:STRING=NPU\n")
    assert RUN_TESTS.detect_backend(build_dir) == "npu"
    assert RUN_TESTS.UNSUPPORTED_APIS_BY_BACKEND["npu"] == {"z2z", "z2d", "d2z"}
    c2c = next(op for op in operators if op["api"] == "c2c")
    z2z = next(op for op in operators if op["api"] == "z2z")
    assert RUN_TESTS.operator_skip_reason(c2c, "npu") is None
    reason = RUN_TESTS.operator_skip_reason(z2z, "npu")
    assert reason and "FP64" in reason and "Ascend 910B" in reason

    cases = RUN_TESTS.expand_test_cases([z2z], matrix, "1d_ct_single")[:1]
    result = RUN_TESTS.aggregate_results(
        [], [z2z], cases, True, True, {z2z["id"]}, {z2z["id"]: reason}
    )
    assert result[z2z["id"]]["accuracy"]["status"] == "Skipped"
    assert result[z2z["id"]]["performance"]["status"] == "Skipped"
    assert RUN_TESTS.requested_phases_passed(result, True, True)


def test_npu_benchmark_parser_accepts_ops_fft_reference_timing():
    payload = {
        "cases": [
            {
                "timing": {
                    "flagfft_median_ms": 1.25,
                    "ref_median_ms": 2.5,
                    "speedup": 2.0,
                },
                "plan_description": PLAN,
            }
        ]
    }
    result = RUN_TESTS.parse_perf_result(json.dumps(payload), backend="npu")
    assert result["status"] == "Passed"
    assert result["flagfft_median_ms"] == 1.25
    assert result["ref_median_ms"] == 2.5
    assert result["speedup"] == 2.0
    assert result["reference_available"] is True


def test_npu_ops_fft_case_policy_limits(operators, matrix):
    npu_cases = RUN_TESTS.expand_test_cases(
        [op for op in operators if op["api"] in ("c2c", "r2c", "c2r")],
        matrix,
        "full",
    )
    reasons = {
        case["case_id"]: RUN_TESTS.case_skip_reason(case, "npu") for case in npu_cases
    }
    assert any(reason and "3D" in reason for reason in reasons.values())
    assert any(reason and "2D" in reason for reason in reasons.values())
    assert any(reason and "prime factor" in reason for reason in reasons.values())
    supported = [
        case
        for case, reason in ((case, reasons[case["case_id"]]) for case in npu_cases)
        if reason is None
    ]
    assert supported
    assert all(case["api"] in ("c2c", "r2c", "c2r") for case in supported)
    assert all(
        case["rank"] == 1
        or (case["api"] == "c2c" and all(n in (32, 64, 128) for n in case["shape"]))
        for case in supported
    )


def test_npu_reference_skip_keeps_flagfft_accuracy(operators):
    op = next(op for op in operators if op["id"] == "2d_c2c")
    case = {
        "case_id": "npu-2d-reference-skip",
        "op_id": op["id"],
        "api": "c2c",
        "rank": 2,
        "algorithm": "ct",
        "batch_mode": "single",
        "shape": [2048, 2048],
        "batch": 1,
        "direction": "forward",
        "scale": 1.0,
        "skip_reason": "ops-fft 2D C2C size is unsupported",
    }
    passed = {"status": "Passed", "metric": {"passed": True}, "plan": "plan"}
    skipped = {
        "status": "Skipped",
        "skip_reason": case["skip_reason"],
        "error": case["skip_reason"],
        "plan": None,
    }
    result = RUN_TESTS.aggregate_results(
        [{**case, "phase": "accuracy", "result": passed, "platform_result": skipped}],
        [op],
        [case],
        True,
        True,
    )
    op_result = result[op["id"]]
    assert op_result["accuracy"]["status"] == "Passed"
    assert op_result["platform_accuracy"]["status"] == "Skipped"
    assert op_result["platform_accuracy"]["policy_skipped"]
    assert op_result["performance"]["status"] == "Skipped"


def test_ix_policy_skip_is_visible_in_incremental_csv(operators):
    op = next(op for op in operators if op["api"] == "d2z")
    case = {
        "case_id": "ix-skip",
        "op_id": op["id"],
        "api": op["api"],
        "rank": 1,
        "algorithm": "ct",
        "batch_mode": "single",
        "shape": [16],
        "batch": 1,
        "direction": "forward",
        "scale": 1.0,
        "skip_reason": "IX/CoreX does not support FP64",
    }
    row = RUN_TESTS.incremental_row(RUN_TESTS.policy_skip_message(case, "accuracy"))
    assert row["status"] == "Skipped"
    assert row["skip_reason"] == case["skip_reason"]


@pytest.mark.parametrize("flag", ["--artifacts", "--analyze-only", "--dump-output"])
def test_removed_result_tree_flags_are_rejected(flag):
    # Raw arrays and per-case evidence no longer reach the result tree, so the
    # flags that controlled them are gone rather than silently ignored.
    with pytest.raises(SystemExit):
        RUN_TESTS.parse_args([flag, "x"])


def test_source_commit_override_is_recorded_for_archive_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("FLAGFFT_SOURCE_COMMIT", "c3ca9d2")
    monkeypatch.setattr(RUN_TESTS, "git_commit", lambda _source: "unknown")
    monkeypatch.setattr(RUN_TESTS, "detect_backend", lambda _build: "npu")
    RUN_TESTS.ENV_INFO.clear()
    RUN_TESTS.probe_env(tmp_path)
    assert RUN_TESTS.ENV_INFO["git_commit"] == "c3ca9d2"


def test_ix_dry_run_keeps_six_operator_group_and_skips_fp64(tmp_path, capsys):
    build_dir = tmp_path / "ix-build"
    build_dir.mkdir()
    (build_dir / "CMakeCache.txt").write_text("BACKEND:STRING=IX\n")

    assert (
        RUN_TESTS.main(
            ["--dry-run", "--build-dir", str(build_dir), "--combination", "2d"]
        )
        == 0
    )
    output = capsys.readouterr().out
    payload = json.loads(output[output.index("{") :])
    assert payload["backend"] == "ix"
    assert len(payload["operators"]) == 6
    assert len(payload["skipped_operators"]) == 3
    assert {case["api"] for case in payload["cases"] if "skip_reason" in case} == {
        "z2z",
        "z2d",
        "d2z",
    }


def test_npu_dry_run_keeps_six_operator_group_and_skips_fp64(tmp_path, capsys):
    build_dir = tmp_path / "npu-build"
    build_dir.mkdir()
    (build_dir / "CMakeCache.txt").write_text("BACKEND:STRING=NPU\n")

    assert (
        RUN_TESTS.main(
            ["--dry-run", "--build-dir", str(build_dir), "--combination", "2d"]
        )
        == 0
    )
    output = capsys.readouterr().out
    payload = json.loads(output[output.index("{") :])
    assert payload["backend"] == "npu"
    assert len(payload["operators"]) == 6
    assert len(payload["skipped_operators"]) == 3


def test_missing_operator_is_not_accepted_as_a_complete_suite(tmp_path, operators):
    path = tmp_path / "operators.yaml"
    path.write_text(yaml.safe_dump({"ops": operators[:-1]}))
    with pytest.raises(ValueError, match="36 acceptance operators"):
        RUN_TESTS.load_operators(path)


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group signals")
def test_interrupt_saves_completed_cases_and_stops_native_process(
    tmp_path, operators, matrix
):
    # A CPU-only capture fixture exercises the real CLI/worker/signal path.
    capture = fake_capture(tmp_path)
    size = min(matrix[operators[0]["sizes"]])
    cases = RUN_TESTS.expand_test_cases(
        [operators[0]], matrix, shapes={(size,)}, scales="1"
    )
    inverse = next(case for case in cases if case["direction"] == "inverse")
    output = tmp_path / "output"
    pid_file = tmp_path / "native.pid"
    process = subprocess.Popen(
        [
            sys.executable,
            str(ROOT / "tools/run_tests.py"),
            "--accuracy-only",
            "--ops",
            operators[0]["id"],
            "--shapes",
            str(size),
            "--scales",
            "1",
            "--capture-bin",
            str(capture),
            "--output-dir",
            str(output),
        ],
        env={
            **os.environ,
            "FAKE_CAPTURE_HANG": "flagfft",
            "FAKE_CAPTURE_HANG_DIRECTION": "inverse",
            "FAKE_CAPTURE_PID_FILE": str(pid_file),
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 60
        while (
            not pid_file.is_file()
            and process.poll() is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.05)
        assert pid_file.is_file(), "inverse fixture did not start"
        native_pid = int(pid_file.read_text())
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 130, stdout + stderr
        summary = json.loads((output / "summary.json").read_text())
        entry = summary_op_entry(summary, operators[0]["id"])
        assert entry["config"]["interrupted"]
        assert entry["accuracy"]["passed"] == 1
        accuracy = json.loads(
            (output / operators[0]["id"] / "accuracy_result.json").read_text()
        )
        assert accuracy["status"] == "Incomplete"
        assert accuracy["passed"] == 1 and accuracy["missing"] == 1
        with pytest.raises(ProcessLookupError):
            os.kill(native_pid, 0)
        assert (output / "manifest.json").is_file()
        assert (output / "incremental.csv").is_file()
        assert not any(output.rglob("*.bin"))
        assert not (output / inverse["op_id"] / inverse["case_id"]).exists()
    finally:
        if pid_file.is_file():
            try:
                os.killpg(int(pid_file.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=5)


@pytest.mark.parametrize("mode,expected", [("timeout", "Timeout"), ("error", "Error")])
def test_native_process_timeout_and_error_retain_logs(tmp_path, mode, expected):
    script = (
        "import time; print('started', flush=True); time.sleep(30)"
        if mode == "timeout"
        else "import sys; print('native failure', file=sys.stderr); sys.exit(1)"
    )
    log_path = tmp_path / "bench.log"
    stage = RUN_TESTS.run_logged_command(
        [sys.executable, "-c", script], 1, 0, tmp_path, log_path
    )
    assert stage["status"] == expected
    assert log_path.is_file()
    if mode == "timeout":
        assert "started" in log_path.read_text()
    else:
        assert "native failure" in log_path.read_text()


@pytest.mark.parametrize(
    "value", ["full,1d_ct_single", "1d_ct_single,all", "full,all", "2d_bs", ""]
)
def test_invalid_group_filters_are_rejected(value):
    with pytest.raises(ValueError):
        RUN_TESTS.resolve_combination_names(value)


def test_scale_expansion_keeps_performance_unique(operators, matrix):
    cases = RUN_TESTS.expand_test_cases(operators, matrix, scales="all")
    baseline_count = len(RUN_TESTS.expand_test_cases(operators, matrix, scales="1"))
    assert len(cases) == baseline_count * 3
    perf_cases = RUN_TESTS.performance_cases(cases)
    assert len(perf_cases) == baseline_count
    assert all("scale" not in case for case in perf_cases)
    matrix.pop("scales")
    assert {
        case["scale"] for case in RUN_TESTS.expand_all_test_cases(operators, matrix)
    } == {1.0}


@pytest.mark.parametrize(
    "mode,values",
    [
        ("single", [2]),
        ("batch", [1, 16]),
        ("3d", [16]),
        ("batch", []),
        ("batch", [16, 16]),
    ],
)
def test_invalid_batch_matrix_is_rejected(tmp_path, matrix, mode, values):
    matrix["batches"][mode] = values
    path = tmp_path / "test_matrix.yaml"
    path.write_text(yaml.safe_dump(matrix))
    with pytest.raises(ValueError):
        RUN_TESTS.load_test_matrix(path)


@pytest.mark.parametrize("scales", ["nan", "inf", "0", "-1", "1,1", ""])
def test_invalid_scales_are_rejected(scales):
    with pytest.raises(ValueError):
        RUN_TESTS.parse_scales(scales, [1.0])


def test_missing_or_wrong_rank_size_set_is_rejected(operators, matrix):
    matrix[operators[0]["sizes"]] = [[16, 16]]
    with pytest.raises(ValueError):
        RUN_TESTS.expand_all_test_cases(operators, matrix)
    del matrix[operators[0]["sizes"]]
    with pytest.raises(ValueError):
        RUN_TESTS.expand_all_test_cases(operators, matrix)


def test_capture_command_uses_api_instead_of_id_prefix(tmp_path, operators, matrix):
    case = RUN_TESTS.expand_test_cases(operators, matrix, "3d")[0]
    cmd = RUN_TESTS.build_accuracy_cmd(
        case, tmp_path / "numpy_fft_capture", tmp_path, "flagfft"
    )
    assert cmd[0] == str(tmp_path / "numpy_fft_capture")
    assert "--api=c2c" in cmd
    assert f"--shape={'x'.join(str(n) for n in case['shape'])}" in cmd
    assert "--batch=1" in cmd
    assert "--direction=forward" in cmd
    assert "--implementation=flagfft" in cmd
    perf_cmd = RUN_TESTS.build_perf_cmd(case, tmp_path, 2, 5)
    assert "--print-path" in perf_cmd
    assert "--scale" not in perf_cmd


@pytest.mark.parametrize("api", ["c2r", "z2d"])
@pytest.mark.parametrize("shape", [(23,), (8, 10), (8, 7), (4, 5, 6)])
def test_real_inverse_input_is_a_valid_multidimensional_half_spectrum(api, shape):
    value, seed = RUN_TESTS.make_input(api, shape, 2, 1.0)
    same, same_seed = RUN_TESTS.make_input(api, shape, 2, 1.0)
    np.testing.assert_array_equal(value, same)
    assert seed == same_seed
    axes = tuple(range(1, len(shape) + 1))
    real = np.fft.irfftn(value.astype(np.complex128), s=shape, axes=axes)
    restored = np.fft.rfftn(real, s=shape, axes=axes)
    np.testing.assert_allclose(restored, value, rtol=1e-13, atol=1e-13)


def test_input_generation_fills_splitmix_stream_in_bounded_chunks(monkeypatch):
    seed = 0x123456789ABCDEF0
    expected = RUN_TESTS.splitmix_signed_unit(37, seed).astype(np.float32)
    actual = np.empty(expected.size, dtype=np.float32)
    monkeypatch.setattr(RUN_TESTS, "INPUT_GENERATION_CHUNK_ELEMENTS", 5)
    RUN_TESTS.fill_splitmix_signed_unit(actual, seed)
    np.testing.assert_array_equal(actual, expected)


def test_make_input_preserves_chunked_splitmix_values(monkeypatch):
    monkeypatch.setattr(RUN_TESTS, "INPUT_GENERATION_CHUNK_ELEMENTS", 3)
    value, seed = RUN_TESTS.make_input("c2c", (7,), 2, 1.0)
    expected = RUN_TESTS.as_complex_from_interleaved(
        RUN_TESTS.splitmix_signed_unit(28, seed), "c2c", (2, 7)
    )
    np.testing.assert_array_equal(value, expected)


def test_numpy_reference_uses_double_precision_and_unnormalized_inverse():
    value, _ = RUN_TESTS.make_input("c2c", (23,), 1, 1.0)
    forward = RUN_TESTS.numpy_reference(value, "c2c", (23,), "forward")
    inverse = RUN_TESTS.numpy_reference(forward, "c2c", (23,), "inverse")
    assert forward.dtype == np.complex128
    np.testing.assert_allclose(
        inverse, value.astype(np.complex128) * 23, rtol=1e-13, atol=1e-13
    )


def test_numpy_reference_chunks_batched_transforms(monkeypatch):
    value, _ = RUN_TESTS.make_input("c2c", (8,), 3, 1.0)
    expected = np.fft.fftn(value.astype(np.complex128), s=(8,), axes=(1,))
    monkeypatch.setattr(RUN_TESTS, "REFERENCE_BATCH_CHUNK", 1)
    actual = RUN_TESTS.numpy_reference(value, "c2c", (8,), "forward")
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("api", "direction"),
    [
        ("c2c", "forward"),
        ("c2r", "inverse"),
        ("r2c", "forward"),
        ("z2z", "forward"),
        ("z2d", "inverse"),
        ("d2z", "forward"),
    ],
)
def test_compare_stream_matches_materialized_reference(monkeypatch, api, direction):
    shape = (8,)
    batch = 3
    value, _ = RUN_TESTS.make_input(api, shape, batch, 1.0)
    reference = RUN_TESTS.numpy_reference(value, api, shape, direction)
    output_dtype = RUN_TESTS.raw_dtype(api, is_input=False)
    payload = np.ascontiguousarray(reference.astype(output_dtype)).tobytes()

    monkeypatch.setattr(RUN_TESTS, "REFERENCE_BATCH_CHUNK", 1)
    streaming, digest = RUN_TESTS.compare_stream(
        io.BytesIO(payload), value, api, shape, direction, batch
    )
    elements = RUN_TESTS.product(RUN_TESTS.output_shape(api, shape, batch)[1:])
    materialized = RUN_TESTS.error_stats(
        reference.astype(output_dtype), reference, elements, batch
    )
    assert streaming == materialized
    assert digest == hashlib.sha256(payload).hexdigest()


def test_compare_stream_reports_a_truncated_pipe(monkeypatch):
    shape = (8,)
    value, _ = RUN_TESTS.make_input("c2c", shape, 2, 1.0)
    monkeypatch.setattr(RUN_TESTS, "REFERENCE_BATCH_CHUNK", 1)
    with pytest.raises(ValueError, match="of"):
        RUN_TESTS.compare_stream(
            io.BytesIO(b"\0" * 8), value, "c2c", shape, "forward", 2
        )


def test_error_metric_detects_worst_batch_and_nonfinite_values():
    reference = np.ones((2, 8), dtype=np.float64)
    value = reference.copy()
    value[1, 0] += 0.1
    stats = RUN_TESTS.error_stats(value, reference, 8, 2)
    assert stats["worst_l2_batch"] == stats["worst_linf_batch"] == 1
    assert stats["rel_l2"] > 0
    value[1, 0] = np.nan
    stats = RUN_TESTS.error_stats(value, reference, 8, 2)
    assert stats["rel_l2"] == float("inf")
    assert stats["worst_l2_batch"] == 1
    assert not RUN_TESTS.judged_stats(stats, RUN_TESTS.accuracy_limit("z2z", 8))[
        "passed"
    ]


def test_error_stats_reduces_in_bounded_chunks(monkeypatch):
    reference = np.arange(24, dtype=np.float64).reshape(2, 12)
    value = reference.copy()
    value[1, 7] += 0.1
    monkeypatch.setattr(RUN_TESTS, "ERROR_STATS_CHUNK_ELEMENTS", 3)
    chunked = RUN_TESTS.error_stats(value, reference, 12, 2)
    monkeypatch.setattr(RUN_TESTS, "ERROR_STATS_CHUNK_ELEMENTS", 100)
    whole = RUN_TESTS.error_stats(value, reference, 12, 2)
    assert chunked == whole


def test_accuracy_case_streams_without_writing_arrays_into_the_result_tree(
    tmp_path, operators, matrix
):
    case = RUN_TESTS.expand_all_test_cases(operators, matrix)[0]
    output = tmp_path / "output"
    scratch = tmp_path / "scratch"
    record = RUN_TESTS.run_accuracy_case(
        case, fake_capture(tmp_path), output, 0, 60, scratch
    )
    assert record["accuracy"]["status"] == "Passed"
    assert record["platform_accuracy"]["status"] == "Passed"
    assert record["accuracy"]["plan"] == PLAN.strip()
    assert record["accuracy"]["metric"]["passed"]
    # The streamed bytes are hashed as they arrive, so the digest still covers
    # the whole output even though no output file exists.
    assert len(record["accuracy"]["output_sha256"]) == 64
    op_dir = output / case["op_id"]
    assert list(op_dir.iterdir()) == []
    assert not (op_dir / case["case_id"]).exists()
    assert not any(path.suffix in (".bin", ".npy") for path in output.rglob("*"))
    # The transient input is deleted as soon as the case finishes.
    assert not any(scratch.rglob("input.bin"))


def test_operator_logs_collect_every_case(tmp_path, operators, matrix):
    case = RUN_TESTS.expand_all_test_cases(operators, matrix)[0]
    scratch = tmp_path / "scratch"
    record = RUN_TESTS.run_accuracy_case(
        case, fake_capture(tmp_path), tmp_path / "output", 0, 60, scratch
    )
    message = accuracy_message(case, record)
    op_dir = tmp_path / "output" / case["op_id"]
    for field, label in (("result", "flagfft"), ("platform_result", "platform")):
        RUN_TESTS.append_operator_log(
            op_dir,
            RUN_TESTS.ACCURACY_LOG,
            case["case_id"],
            label,
            (message[field].get("capture") or {}).get("log_file"),
        )
    text = (op_dir / RUN_TESTS.ACCURACY_LOG).read_text()
    assert f"----- {case['case_id']} flagfft -----" in text
    assert f"----- {case['case_id']} platform -----" in text
    assert RUN_TESTS.PLAN_BEGIN in text and RUN_TESTS.PLAN_END in text
    assert PLAN.strip() in text
    # The scratch copy is consumed, so nothing is left outside the result tree.
    assert not any(scratch.rglob("*.log"))


def accuracy_message(case, record):
    return {
        **case,
        "phase": "accuracy",
        "duration": record["duration"],
        "result": record["accuracy"],
        "platform_result": record["platform_accuracy"],
    }


@pytest.mark.parametrize(
    "hang,platform_mode,platform_status",
    [(True, "ok", "Timeout"), (False, "error", "Error"), (False, "corrupt", "Failed")],
)
def test_platform_failures_do_not_fail_flagfft_acceptance(
    tmp_path, monkeypatch, operators, matrix, hang, platform_mode, platform_status
):
    case = RUN_TESTS.expand_all_test_cases(operators, matrix)[0]
    monkeypatch.setenv("FAKE_CAPTURE_PLATFORM_MODE", platform_mode)
    if hang:
        monkeypatch.setenv("FAKE_CAPTURE_HANG", "platform")
        monkeypatch.setenv("FAKE_CAPTURE_HANG_DIRECTION", case["direction"])
    record = RUN_TESTS.run_accuracy_case(
        case, fake_capture(tmp_path), tmp_path / "output", 0, 5, tmp_path / "scratch"
    )
    assert record["accuracy"]["status"] == "Passed"
    assert record["accuracy"]["plan"] == PLAN.strip()
    assert record["platform_accuracy"]["status"] == platform_status
    results = RUN_TESTS.aggregate_results(
        [accuracy_message(case, record)], [operators[0]], [case], True, False
    )
    assert results[case["op_id"]]["accuracy"]["status"] == "Passed"
    assert results[case["op_id"]]["platform_accuracy"]["status"] == "Failed"
    assert RUN_TESTS.requested_phases_passed(results, True, False)


def test_both_scales_survive_in_the_operator_result(tmp_path, operators, matrix):
    case1 = RUN_TESTS.expand_test_cases([operators[0]], matrix, scales="1,2")[0]
    case2 = {**case1, "scale": 2.0}
    case2["case_id"] = RUN_TESTS.case_name(case2)
    assert case1["case_id"] != case2["case_id"]
    capture = fake_capture(tmp_path)
    output = tmp_path / "output"
    records = [
        RUN_TESTS.run_accuracy_case(case, capture, output, 0, 60, tmp_path / "scratch")
        for case in (case1, case2)
    ]
    assert records[0]["input_sha256"] != records[1]["input_sha256"]
    results = RUN_TESTS.aggregate_results(
        [
            accuracy_message(case, record)
            for case, record in zip((case1, case2), records)
        ],
        [operators[0]],
        [case1, case2],
        True,
        False,
    )
    block = results[operators[0]["id"]]["accuracy"]
    assert set(block["cases"]) == {case1["case_id"], case2["case_id"]}
    assert block["status"] == "Passed" and block["passed"] == 2
    RUN_TESTS.write_summary(output / "summary.json", results, {}, 0.0)
    saved = json.loads(
        (output / operators[0]["id"] / "accuracy_result.json").read_text()
    )
    assert {entry["scale"] for entry in saved["cases"].values()} == {1.0, 2.0}
    assert all(entry["plan"] == PLAN.strip() for entry in saved["cases"].values())


def test_flaggems_status_follows_the_platform_rule():
    def shaped(**overrides):
        block = {
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "missing": 0,
            "data_file": "op/accuracy_result.json",
            "cases": {},
        }
        block.update(overrides)
        return RUN_TESTS.flaggems_accuracy_block(block, Path("/run/op/accuracy.log"))

    assert shaped(passed=3)["status"] == "PASS"
    assert shaped(passed=2, failed=1)["status"] == "FAIL"
    # A pure execution failure has no cases to count, which the platform reads
    # as a failure rather than as a vacuous pass.
    assert shaped(errors=1)["status"] == "FAIL"
    assert shaped(passed=0, skipped=2)["status"] == "FAIL"
    assert shaped(passed=1, skipped=1)["status"] == "PASS"
    assert shaped(passed=1, missing=2)["errors"] == 2
    for overrides in ({"passed": 3}, {"passed": 2, "failed": 1}, {"errors": 1}):
        entry = shaped(**overrides)
        assert entry["total"] == entry["passed"] + entry["failed"] + entry["skipped"]
        assert entry["exit_code"] == (0 if entry["status"] == "PASS" else 1)


def test_summary_json_is_a_flat_flaggems_array(tmp_path, operators, matrix):
    case = RUN_TESTS.expand_all_test_cases(operators, matrix)[0]
    record = RUN_TESTS.run_accuracy_case(
        case, fake_capture(tmp_path), tmp_path, 0, 60, tmp_path / "scratch"
    )
    results = RUN_TESTS.aggregate_results(
        [accuracy_message(case, record)], [operators[0]], [case], True, False
    )
    RUN_TESTS.write_summary(tmp_path / "summary.json", results, {"ops": []}, 1.5)
    summary = json.loads((tmp_path / "summary.json").read_text())
    entry = summary_op_entry(summary, operators[0]["id"])
    assert entry["accuracy"]["status"] == "PASS"
    assert entry["accuracy"]["passed"] == 1
    assert Path(entry["accuracy"]["log_path"]).name == RUN_TESTS.ACCURACY_LOG
    assert Path(entry["perf_log_path"]).name == RUN_TESTS.PERF_LOG
    assert entry["format_version"] == RUN_TESTS.FORMAT_VERSION
    assert entry["performance"] == []
    # The per-case detail lives in the operator file, not in the platform view.
    assert "result" not in entry and "cases" not in entry["accuracy"]


def test_accuracy_log_carries_a_verdict_the_platform_parser_reads(
    tmp_path, operators, matrix
):
    """A reader that only sees the log must reach the same verdict as summary.json.

    This mirrors parse_pytest_summary_from_text in ref/run_flaggems_test_new.py,
    including its PASS/FAIL rule and its plural "errors" key.
    """
    case = RUN_TESTS.expand_all_test_cases(operators, matrix)[0]
    record = RUN_TESTS.run_accuracy_case(
        case, fake_capture(tmp_path), tmp_path, 0, 60, tmp_path / "scratch"
    )
    results = RUN_TESTS.aggregate_results(
        [accuracy_message(case, record)], [operators[0]], [case], True, False
    )
    RUN_TESTS.write_summary(tmp_path / "summary.json", results, {"ops": []}, 1.5)
    entry = summary_op_entry(
        json.loads((tmp_path / "summary.json").read_text()), operators[0]["id"]
    )

    text = (tmp_path / operators[0]["id"] / RUN_TESTS.ACCURACY_LOG).read_text()
    counters = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    for match in re.finditer(r"(\d+)\s+([A-Za-z_]+)", text):
        key = match.group(2).lower()
        if key in counters:
            counters[key] = int(match.group(1))
    total = counters["passed"] + counters["failed"] + counters["skipped"]
    if (
        counters["failed"] > 0
        or (counters["errors"] > 0 and total == 0)
        or counters["passed"] == 0
    ):
        status = "FAIL"
    else:
        status = "PASS"

    assert counters["passed"] == entry["accuracy"]["passed"]
    assert counters["failed"] == entry["accuracy"]["failed"]
    assert counters["errors"] == entry["accuracy"]["errors"]
    assert total == entry["accuracy"]["total"]
    assert status == entry["accuracy"]["status"]


def test_performance_rows_put_speedup_in_the_complex_column():
    block = {
        "cases": {
            "op_n256": {"case_id": "op_n256", "speedup": 3.4123},
            "op_n512": {"case_id": "op_n512", "speedup": 1.5},
        }
    }
    rows = RUN_TESTS.flaggems_performance_rows(block, 2.2627)
    assert [row["func_name"] for row in rows] == ["op_n256", "op_n512"]
    assert rows[0]["cfloat"] == "3.4123"
    for row in rows:
        assert row["avg_speedup"] == "2.2627"
        assert row["float32"] == "" and row["cfloat"] != ""


def test_operator_speedup_stats_is_the_geometric_mean():
    block = {
        "cases": {
            "a": {"status": "Passed", "speedup": 2.0, "baseline_valid": True},
            "b": {"status": "Passed", "speedup": 8.0, "baseline_valid": True},
            "c": {"status": "Failed", "speedup": 100.0, "baseline_valid": True},
            "d": {"status": "Passed", "speedup": 100.0, "baseline_valid": False},
        }
    }
    assert RUN_TESTS.operator_speedup_stats(block) == {
        "count": 2,
        "geometric_mean_speedup": 4.0,
        "min_speedup": 2.0,
        "max_speedup": 8.0,
    }
    assert RUN_TESTS.operator_speedup_stats({"cases": {}}) == {"count": 0}


def test_missing_case_prevents_operator_pass(operators, matrix):
    cases = RUN_TESTS.expand_test_cases([operators[0]], matrix)[:2]
    message = {
        **cases[0],
        "phase": "accuracy",
        "result": {"status": "Passed", "plan": PLAN},
        "platform_result": {"status": "Passed"},
    }
    result = RUN_TESTS.aggregate_results([message], [operators[0]], cases, True, False)
    assert result[operators[0]["id"]]["accuracy"]["status"] == "Incomplete"
    assert result[operators[0]["id"]]["accuracy"]["missing"] == 1
    assert not RUN_TESTS.requested_phases_passed(result, True, False)


def test_csv_roundtrip_preserves_plan_and_both_accuracy_results(operators, matrix):
    case = RUN_TESTS.expand_all_test_cases(operators, matrix)[0]
    message = {
        **case,
        "phase": "accuracy",
        "duration": 0.1,
        "result": {
            "status": "Passed",
            "plan": PLAN,
            "metric": {"rel_l2": 1e-8},
            "limits": {"rel_l2": 1e-5},
            "data_file": "case.json",
        },
        "platform_result": {"status": "Failed", "metric": {"rel_l2": 1e-3}},
    }
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=RUN_TESTS.INC_COLUMNS)
    writer.writeheader()
    writer.writerow(RUN_TESTS.incremental_row(message))
    stream.seek(0)
    row = next(csv.DictReader(stream))
    assert row["plan"] == PLAN
    assert row["op_id"] == case["op_id"]
    assert row["flagfft_status"] == "Passed"
    assert row["platform_status"] == "Failed"
    assert float(row["flagfft_rel_l2"]) == 1e-8


def test_benchmark_parser_keeps_actual_plan_and_ref_timing():
    payload = {
        "cases": [
            {
                "timing": {
                    "flagfft_median_ms": 1.0,
                    "ref_median_ms": 2.0,
                    "speedup": 2.0,
                },
                "plan_description": PLAN,
            }
        ]
    }
    result = RUN_TESTS.parse_perf_result(
        "[device caps] {not JSON}\n" + json.dumps(payload)
    )
    assert result["status"] == "Passed"
    assert result["ref_median_ms"] == 2
    assert result["plan"] == PLAN
    assert "cufft_median_ms" not in result


def test_existing_report_structure_keeps_both_benchmark_directions(operators, matrix):
    cases = RUN_TESTS.expand_test_cases([operators[0]], matrix)[:2]
    messages = [
        {
            **case,
            "case_id": RUN_TESTS.case_name(case, performance=True),
            "phase": "performance",
            "result": {
                "status": "Passed",
                "flagfft_median_ms": 1.0,
                "ref_median_ms": 2.0,
                "speedup": 2.0,
                "plan": PLAN,
            },
        }
        for case in cases
    ]
    results = RUN_TESTS.aggregate_results(messages, [operators[0]], cases, False, True)
    op = results[operators[0]["id"]]
    assert op["accuracy"]["details"] == []
    report = op["performance"]["data"]["default"]
    assert report["result"] == "OK" and report["speedup"] == 2.0
    assert len(report["details"]) == 2
    assert any("direction=forward" in key for key in report["details"])
    assert any("direction=inverse" in key for key in report["details"])
    assert all(
        value["base"] == 2.0 and value["gems"] == 1.0
        for value in report["details"].values()
    )


def test_speedup_summary_excludes_incorrect_baseline():
    result = {
        "op": {
            "performance": {
                "cases": {
                    "valid": {
                        "status": "Passed",
                        "speedup": 2.0,
                        "baseline_valid": True,
                    },
                    "invalid": {
                        "status": "Passed",
                        "speedup": 100.0,
                        "baseline_valid": False,
                    },
                }
            }
        }
    }
    assert RUN_TESTS.compute_speedup_stats(result)["geometric_mean_speedup"] == 2.0


def test_speedup_summary_excludes_incorrect_flagfft_output():
    results = {
        "failed_op": {
            "accuracy": {"status": "Failed"},
            "performance": {
                "cases": {
                    "fast_but_wrong": {
                        "status": "Passed",
                        "speedup": 100.0,
                        "baseline_valid": True,
                    }
                }
            },
        }
    }
    assert RUN_TESTS.compute_speedup_stats(results)["count"] == 0


def test_failed_numeric_metrics_are_written_as_strict_json(tmp_path):
    path = tmp_path / "result.json"
    RUN_TESTS.write_json(
        path, {"finite": False, "rel_l2": np.float64(np.inf), "rel_linf": np.nan}
    )
    data = json.loads(path.read_text(), parse_constant=lambda value: pytest.fail(value))
    assert data == {"finite": False, "rel_l2": None, "rel_linf": None}
