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
import importlib.util
import io
import json
import os
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
        case["status"] == "Skipped"
        for case in op_result["accuracy"]["cases"].values()
    )
    assert RUN_TESTS.requested_phases_passed(result, True, True)


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
    row = RUN_TESTS.incremental_row(
        RUN_TESTS.policy_skip_message(case, "accuracy")
    )
    assert row["status"] == "Skipped"
    assert row["skip_reason"] == case["skip_reason"]


def test_artifact_policy_cli_defaults_to_failed():
    assert RUN_TESTS.parse_args([]).artifact_policy == "failed"
    assert RUN_TESTS.parse_args(["--artifacts", "none"]).artifact_policy == "none"
    assert RUN_TESTS.parse_args(["--artifacts", "all"]).artifact_policy == "all"


def test_ix_dry_run_keeps_six_operator_group_and_skips_fp64(tmp_path, capsys):
    build_dir = tmp_path / "ix-build"
    build_dir.mkdir()
    (build_dir / "CMakeCache.txt").write_text("BACKEND:STRING=IX\n")

    assert RUN_TESTS.main(
        ["--dry-run", "--build-dir", str(build_dir), "--combination", "2d"]
    ) == 0
    output = capsys.readouterr().out
    payload = json.loads(output[output.index("{") :])
    assert payload["backend"] == "ix"
    assert len(payload["operators"]) == 6
    assert len(payload["skipped_operators"]) == 3
    assert {
        case["api"] for case in payload["cases"] if "skip_reason" in case
    } == {"z2z", "z2d", "d2z"}


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
    capture = tmp_path / "capture"
    capture.write_text(
        f"#!{sys.executable}\n"
        + """
import os
import sys
import time
from pathlib import Path
import numpy as np
args = dict(arg[2:].split('=', 1) for arg in sys.argv[1:])
directory = Path(args['output-dir'])
if args['implementation'] == 'flagfft':
    (directory / 'flagfft_plan.txt').write_text('CPU capture fixture plan\\n')
    if args['direction'] == 'inverse':
        (directory / 'native.pid').write_text(str(os.getpid()))
        time.sleep(30)
value = np.fromfile(args['input'], dtype=np.complex64)
value = value.reshape(int(args['batch']), int(args['shape']))
np.fft.fft(value.astype(np.complex128), axis=1).astype(np.complex64).tofile(
    directory / (args['implementation'] + '.bin')
)
"""
    )
    capture.chmod(0o755)
    size = min(matrix[operators[0]["sizes"]])
    cases = RUN_TESTS.expand_test_cases(
        [operators[0]], matrix, shapes={(size,)}, scales="1"
    )
    inverse = next(case for case in cases if case["direction"] == "inverse")
    output = tmp_path / "output"
    pid_file = output / inverse["op_id"] / inverse["case_id"] / "native.pid"
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
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 30
        while (
            not pid_file.is_file()
            and process.poll() is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.05)
        assert pid_file.is_file(), "inverse fixture did not start"
        native_pid = int(pid_file.read_text())
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 130, stdout + stderr
        summary = json.loads((output / "summary.json").read_text())
        assert summary["config"]["interrupted"]
        accuracy = summary["result"][operators[0]["id"]]["accuracy"]
        assert accuracy["status"] == "Incomplete"
        assert accuracy["passed"] == 1 and accuracy["missing"] == 1
        with pytest.raises(ProcessLookupError):
            os.kill(native_pid, 0)
        assert (output / "manifest.json").is_file()
        assert (output / "incremental.csv").is_file()
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
    stage = RUN_TESTS.run_subprocess(
        [sys.executable, "-c", script], 1, 0, tmp_path, "native"
    )
    assert stage["status"] == expected
    if mode == "timeout":
        assert "started" in (tmp_path / stage["stdout_file"]).read_text()
    else:
        assert "native failure" in stage["error"]


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
    expected = np.fft.fftn(
        value.astype(np.complex128), s=(8,), axes=(1,)
    )
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
def test_streaming_error_stats_matches_materialized_reference(
    tmp_path, monkeypatch, api, direction
):
    shape = (8,)
    batch = 3
    value, _ = RUN_TESTS.make_input(api, shape, batch, 1.0)
    reference = RUN_TESTS.numpy_reference(value, api, shape, direction)
    output_dtype = (
        RUN_TESTS.complex_dtype(api)
        if np.iscomplexobj(reference)
        else RUN_TESTS.real_dtype(api)
    )
    input_path = tmp_path / "input.bin"
    output_path = tmp_path / "flagfft.bin"
    value.tofile(input_path)
    reference.astype(output_dtype).tofile(output_path)

    monkeypatch.setattr(RUN_TESTS, "REFERENCE_BATCH_CHUNK", 1)
    input_memmap = RUN_TESTS.load_raw_memmap(input_path, api, shape, batch)
    try:
        streaming = RUN_TESTS.streaming_error_stats(
            output_path, input_memmap, api, shape, direction, batch
        )
    finally:
        del input_memmap
    elements = RUN_TESTS.product(
        RUN_TESTS.output_shape(api, shape, batch)[1:]
    )
    materialized = RUN_TESTS.error_stats(
        reference.astype(output_dtype), reference, elements, batch
    )
    assert streaming == materialized


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


PLAN = 'LeafPlan(n=256, factors=[4,4,4,4])\nCompiledRawLeaf(kernel="fft")\n'


def mock_capture(monkeypatch, platform_status="Completed", platform_corrupt=False):
    def run_subprocess(cmd, timeout, gpu_id, case_dir, implementation):
        api = next(arg.split("=", 1)[1] for arg in cmd if arg.startswith("--api="))
        shape = tuple(
            int(n)
            for n in next(
                arg.split("=", 1)[1] for arg in cmd if arg.startswith("--shape=")
            ).split("x")
        )
        batch = int(
            next(arg.split("=", 1)[1] for arg in cmd if arg.startswith("--batch="))
        )
        direction = next(
            arg.split("=", 1)[1] for arg in cmd if arg.startswith("--direction=")
        )
        value = RUN_TESTS.load_raw(case_dir / "input.bin", api, shape, batch)
        output = RUN_TESTS.numpy_reference(value, api, shape, direction)
        dtype = (
            RUN_TESTS.complex_dtype(api)
            if np.iscomplexobj(output)
            else RUN_TESTS.real_dtype(api)
        )
        output = output.astype(dtype)
        if implementation == "platform" and platform_corrupt:
            output.reshape(-1)[0] += 100
        output.tofile(case_dir / f"{implementation}.bin")
        if implementation == "flagfft":
            (case_dir / "flagfft_plan.txt").write_text(PLAN)
        (case_dir / f"{implementation}.stdout").write_text("capture complete\n")
        (case_dir / f"{implementation}.stderr").write_text("")
        return {
            "status": "Completed" if implementation == "flagfft" else platform_status,
            "duration": 0.01,
            "command": cmd,
        }

    monkeypatch.setattr(RUN_TESTS, "run_subprocess", run_subprocess)


def test_accuracy_captures_finish_before_numpy_reference(
    tmp_path, monkeypatch, operators, matrix
):
    case = RUN_TESTS.expand_all_test_cases(operators, matrix)[0]
    events = []
    original_reference_chunks = RUN_TESTS.numpy_reference_chunks

    def delayed_reference_chunks(value, api, shape, direction):
        events.append("reference")
        yield from original_reference_chunks(value, api, shape, direction)

    def capture(cmd, timeout, gpu_id, case_dir, implementation):
        events.append(f"capture:{implementation}")
        value = RUN_TESTS.load_raw(
            case_dir / "input.bin", case["api"], tuple(case["shape"]), case["batch"]
        )
        np.zeros_like(value).tofile(case_dir / f"{implementation}.bin")
        if implementation == "flagfft":
            (case_dir / "flagfft_plan.txt").write_text(PLAN)
        return {"status": "Completed", "duration": 0.01, "command": cmd}

    monkeypatch.setattr(
        RUN_TESTS, "numpy_reference_chunks", delayed_reference_chunks
    )
    monkeypatch.setattr(RUN_TESTS, "run_subprocess", capture)

    record = RUN_TESTS.run_accuracy_case(
        case, tmp_path / "capture", tmp_path, 0, 10, "none"
    )

    assert events[:2] == ["capture:flagfft", "capture:platform"]
    assert events[2:] == ["reference", "reference"]
    assert record["capture_stages"]["flagfft"]["status"] == "Completed"
    assert record["capture_stages"]["platform"]["status"] == "Completed"


def accuracy_message(case, record):
    return {
        **case,
        "phase": "accuracy",
        "duration": record["duration"],
        "result": record["accuracy"],
        "platform_result": record["platform_accuracy"],
    }


@pytest.mark.parametrize(
    "platform_status,corrupt",
    [("Timeout", False), ("Error", False), ("Completed", True)],
)
def test_platform_failures_do_not_fail_flagfft_acceptance(
    tmp_path,
    monkeypatch,
    operators,
    matrix,
    platform_status,
    corrupt,
):
    case = RUN_TESTS.expand_all_test_cases(operators, matrix)[0]
    mock_capture(monkeypatch, platform_status, corrupt)
    record = RUN_TESTS.run_accuracy_case(case, tmp_path / "capture", tmp_path, 0, 10)
    assert record["accuracy"]["status"] == "Passed"
    assert record["accuracy"]["plan"] == PLAN
    assert record["platform_accuracy"]["status"] != "Passed"
    results = RUN_TESTS.aggregate_results(
        [accuracy_message(case, record)],
        [operators[0]],
        [case],
        True,
        False,
    )
    assert results[case["op_id"]]["accuracy"]["status"] == "Passed"
    assert results[case["op_id"]]["platform_accuracy"]["status"] == "Failed"
    assert RUN_TESTS.requested_phases_passed(results, True, False)


def test_case_artifacts_are_not_overwritten_for_multiple_scales(
    tmp_path, monkeypatch, operators, matrix
):
    cases = RUN_TESTS.expand_test_cases([operators[0]], matrix, scales="1,2")[:1]
    case1 = cases[0]
    case2 = {**case1, "scale": 2.0}
    case2["case_id"] = RUN_TESTS.case_name(case2)
    mock_capture(monkeypatch)
    first = RUN_TESTS.run_accuracy_case(
        case1, tmp_path / "capture", tmp_path, 0, 10, "all"
    )
    second = RUN_TESTS.run_accuracy_case(
        case2, tmp_path / "capture", tmp_path, 0, 10, "all"
    )
    assert first["data_file"] != second["data_file"]
    assert first["input_sha256"] != second["input_sha256"]
    for record in (first, second):
        saved = json.loads((tmp_path / record["data_file"]).read_text())
        assert saved["scale"] == record["scale"]
        assert saved["accuracy"]["plan"] == PLAN


def test_none_artifact_policy_keeps_only_results_and_logs(
    tmp_path, monkeypatch, operators, matrix
):
    case = RUN_TESTS.expand_all_test_cases(operators, matrix)[0]
    mock_capture(monkeypatch)
    record = RUN_TESTS.run_accuracy_case(
        case, tmp_path / "capture", tmp_path, 0, 10, "none"
    )
    case_dir = tmp_path / case["op_id"] / case["case_id"]
    assert record["accuracy"]["status"] == "Passed"
    assert record["raw_artifacts_retained"] is False
    assert "numpy_sha256" not in record
    assert not any(
        (case_dir / filename).exists()
        for filename in RUN_TESTS.RAW_ARTIFACT_FILENAMES
    )
    assert (case_dir / "case.json").is_file()
    assert (case_dir / "flagfft.stdout").is_file()


def test_analyze_only_requires_all_artifacts(tmp_path):
    RUN_TESTS.write_json(
        tmp_path / "manifest.json",
        {"config": {"artifact_policy": "failed"}},
    )
    with pytest.raises(ValueError, match="--artifacts all"):
        RUN_TESTS.analyze_only(tmp_path)


def test_failed_artifact_policy_retains_failed_case_without_npy(
    tmp_path, monkeypatch, operators, matrix
):
    case = RUN_TESTS.expand_all_test_cases(operators, matrix)[0]
    mock_capture(monkeypatch, platform_corrupt=True)
    record = RUN_TESTS.run_accuracy_case(
        case, tmp_path / "capture", tmp_path, 0, 10, "failed"
    )
    case_dir = tmp_path / case["op_id"] / case["case_id"]
    assert record["accuracy"]["status"] == "Passed"
    assert record["platform_accuracy"]["status"] == "Failed"
    assert record["raw_artifacts_retained"] is True
    assert (case_dir / "input.bin").is_file()
    assert (case_dir / "flagfft.bin").is_file()
    assert (case_dir / "platform.bin").is_file()
    assert not (case_dir / "input.npy").exists()
    assert not (case_dir / "numpy.npy").exists()


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


def test_reanalysis_uses_saved_data_without_gpu_execution(
    tmp_path, monkeypatch, operators, matrix
):
    case = RUN_TESTS.expand_all_test_cases(operators, matrix)[0]
    mock_capture(monkeypatch)
    record = RUN_TESTS.run_accuracy_case(
        case, tmp_path / "capture", tmp_path, 0, 10, "all"
    )
    RUN_TESTS.write_json(
        tmp_path / "manifest.json",
        {
            "operators": [operators[0]],
            "cases": [case],
            "performance_cases": RUN_TESTS.performance_cases([case]),
            "env": {},
            "config": {
                "accuracy_only": True,
                "performance_only": False,
                "artifact_policy": "all",
            },
        },
    )
    case_dir = tmp_path / case["op_id"] / case["case_id"]
    output = np.fromfile(case_dir / "flagfft.bin", dtype=np.complex64)
    output[0] += 100
    output.tofile(case_dir / "flagfft.bin")
    monkeypatch.setattr(
        RUN_TESTS,
        "run_subprocess",
        lambda *args: pytest.fail("GPU execution in analyze-only"),
    )
    assert RUN_TESTS.analyze_only(tmp_path) == 1
    summary = json.loads((tmp_path / "summary.json").read_text())
    result = summary["result"][case["op_id"]]
    assert result["accuracy"]["status"] == "Failed"
    assert result["platform_accuracy"]["status"] == "Passed"
    assert result["accuracy"]["cases"][case["case_id"]]["plan"] == PLAN
    assert (tmp_path / record["data_file"]).is_file()


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
