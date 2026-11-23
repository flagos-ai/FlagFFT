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

"""Pytest interface used by flagos-optest for both FFT test suites."""

import os
from pathlib import Path

import pytest
import yaml

from tools import run_tests

ROOT = run_tests.ROOT
OPERATOR_FILE = ROOT / "conf" / "operators.yaml"
MATRIX_FILE = ROOT / "conf" / "test_matrix.yaml"
RECORDS = pytest.StashKey[dict]()
CASE_RECORD = pytest.StashKey[dict]()


def pytest_addoption(parser):
    group = parser.getgroup("FlagFFT operator tests")
    group.addoption("--collect-marks", type=Path)
    group.addoption("--record", choices=("none", "json"), default="none")
    group.addoption("--output", type=Path)
    group.addoption("--ref", choices=("cpu",), default="cpu")
    group.addoption("--quick", action="store_true")
    group.addoption("--level", choices=("core", "comprehensive"), default="comprehensive")
    group.addoption(
        "--fft-case-timeout", type=int, default=os.environ.get("FLAGFFT_CASE_TIMEOUT", "200")
    )
    group.addoption(
        "--fft-warmup", type=int, default=os.environ.get("FLAGFFT_BENCH_WARMUP", "10")
    )
    group.addoption(
        "--fft-iters", type=int, default=os.environ.get("FLAGFFT_BENCH_ITERS", "100")
    )


def pytest_configure(config):
    for op in run_tests.load_operators(OPERATOR_FILE):
        config.addinivalue_line("markers", f"{op['id']}: {op['description']}")
    config.stash[RECORDS] = {}
    if config.getoption("--record") == "json" and config.getoption("--output") is None:
        raise pytest.UsageError("--record json requires --output")
    if (
        config.getoption("--fft-case-timeout") <= 0
        or config.getoption("--fft-warmup") < 0
        or config.getoption("--fft-iters") <= 0
    ):
        raise pytest.UsageError("FFT case timeout and iterations must be positive and warmup nonnegative")


def _build_dir():
    return (ROOT / os.environ.get("FLAGFFT_BUILD_DIR", "build")).resolve()


def _is_benchmark(path):
    return path.is_relative_to(ROOT / "benchmark")


def _representative_cases(cases):
    first_shape = {}
    for case in cases:
        if case["shape"] == first_shape.setdefault(case["op_id"], case["shape"]):
            yield case


def pytest_generate_tests(metafunc):
    if "fft_case" not in metafunc.fixturenames:
        return
    config = metafunc.config
    is_benchmark = _is_benchmark(metafunc.definition.path)
    backend = run_tests.detect_backend(_build_dir())
    ops = run_tests.load_operators(OPERATOR_FILE)
    matrix = run_tests.load_test_matrix(MATRIX_FILE)
    cases, _ = run_tests.filter_unsupported_dtypes(
        ops, run_tests.expand_all_test_cases(ops, matrix), backend
    )
    reduced = config.getoption("--quick")
    if is_benchmark:
        reduced = (
            config.getoption("--level") == "core" and os.environ.get("FLAGFFT_OPTEST_FULL") != "1"
        )
    if reduced:
        cases = list(_representative_cases(cases))

    if is_benchmark:
        reference_cases = {}
        for case in cases:
            key = run_tests.case_name(case, performance=True)
            reference_cases.setdefault(key, []).append(case)
        cases = run_tests.performance_cases(cases)
        for case in cases:
            case["reference_cases"] = reference_cases[case["case_id"]]

    params = []
    for case in cases:
        if reason := run_tests.case_skip_reason(case, backend):
            case["skip_reason"] = reason
        params.append(
            pytest.param(case, id=case["case_id"], marks=getattr(pytest.mark, case["op_id"]))
        )
    metafunc.parametrize("fft_case", params)


@pytest.fixture(scope="session")
def fft_build_dir():
    return _build_dir()


@pytest.fixture(scope="session")
def fft_backend(fft_build_dir):
    backend = run_tests.detect_backend(fft_build_dir)
    if backend == "unknown":
        pytest.fail(f"Cannot identify the FlagFFT backend in {fft_build_dir}", pytrace=False)

    # The native runtimes may also read the older aliases set by run_tests.py.
    # Copy the scheduler's assignment without renumbering the device.
    aliases = {
        "ix": ("IX_VISIBLE_DEVICES", ("ILUVATAR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")),
        "ppu": ("PPU_VISIBLE_DEVICES", ("CUDA_VISIBLE_DEVICES",)),
        "maca": ("MC_VISIBLE_DEVICES", ("MACA_VISIBLE_DEVICES",)),
        "npu": ("ASCEND_VISIBLE_DEVICES", ("ASCEND_RT_VISIBLE_DEVICES",)),
    }
    with pytest.MonkeyPatch.context() as env:
        if backend in aliases:
            target, sources = aliases[backend]
            for source in sources:
                if source in os.environ:
                    env.setenv(target, os.environ[source])
                    break
        if backend in {"ix", "hcu"} and "FLAGFFT_NUMPY_CAPTURE_FULL_BATCH" not in os.environ:
            env.setenv("FLAGFFT_NUMPY_CAPTURE_FULL_BATCH", "1")
        yield backend


@pytest.fixture
def optest_record(request):
    record = {}
    request.node.stash[CASE_RECORD] = record
    return record


def pytest_collection_finish(session):
    path = session.config.getoption("--collect-marks")
    if path is None:
        return
    operator_ids = {op["id"] for op in run_tests.load_operators(OPERATOR_FILE)}
    marks = [
        {"test_case": item.nodeid, "marks": sorted({m.name for m in item.iter_markers()} & operator_ids)}
        for item in session.items
        if "fft_case" in item.fixturenames
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(marks), encoding="utf-8")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    if item.config.getoption("--record") != "json" or "fft_case" not in item.fixturenames:
        return
    report = outcome.get_result()
    if report.when != "call" and report.passed:
        return

    case = item.callspec.params["fft_case"]
    record = {
        "case_id": case["case_id"],
        "params": {
            name: case[name]
            for name in ("op_id", "dtype", "shape", "batch", "direction", "placement", "scale")
            if name in case
        },
        **item.stash.get(CASE_RECORD, {}),
        "result": report.outcome,
    }
    if report.skipped:
        record["reason"] = report.longrepr[2].removeprefix("Skipped: ")
    elif report.failed:
        crash = getattr(report.longrepr, "reprcrash", None)
        record["reason"] = crash.message if crash else report.longreprtext
    item.config.stash[RECORDS][item.nodeid] = (_is_benchmark(item.path), record)


def _performance_report(results):
    operators = {}
    for nodeid, record in results.items():
        op = record["params"]["op_id"]
        operators.setdefault(op, {})[nodeid] = record

    output = {}
    for op, cases in operators.items():
        details = {}
        status = "skipped"
        reason = next(iter(cases.values())).get("reason")
        for record in cases.values():
            outcome = record["result"]
            if outcome == "failed":
                status, reason = outcome, record.get("reason")
            elif outcome == "passed" and status != "failed":
                status, reason = outcome, None
            if "measurement" not in record:
                continue
            dtype = record["params"]["dtype"]
            row = details.setdefault(dtype, {"dtype": dtype, "result": []})
            row["result"].append(record["measurement"])
            if skip_reason := record.get("baseline_skip_reason"):
                row["native_baseline_skip_reason"] = skip_reason
        output[op] = {
            "test_case": next(iter(cases)),
            "result": status,
            "reason": reason,
            "details": list(details.values()),
            "cases": cases,
        }
    return output


def pytest_sessionfinish(session, exitstatus):
    config = session.config
    if config.getoption("--record") != "json":
        return
    records = config.stash[RECORDS]
    results = {nodeid: record for nodeid, (_, record) in records.items()}
    phases = {is_benchmark for is_benchmark, _ in records.values()}
    if phases == {False, True}:
        pytest.exit("Record FlagFFT accuracy and performance in separate pytest commands", returncode=4)
    if phases == {True}:
        results = _performance_report(results)

    path = config.getoption("--output")
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        run_tests.write_json(staging, results)
        staging.replace(path)
    finally:
        staging.unlink(missing_ok=True)
