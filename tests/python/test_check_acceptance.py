# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "check_acceptance", Path(__file__).resolve().parents[2] / "tools/check_acceptance.py"
)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


@pytest.fixture
def run(tmp_path):
    cases = [{"case_id": str(i), "op_id": "a" if i == 0 else "b", "batch": 3} for i in range(4)]
    manifest = {"env": {"git_commit": "frozen-source", "capture_full_batch": True}, "config": {},
                "operators": [{"id": "a"}, {"id": "b"}],
                "cases": cases, "performance_cases": cases}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    for op in manifest["operators"]:
        op_id = op["id"]
        selected = [case for case in cases if case["op_id"] == op_id]
        directory = tmp_path / op_id
        directory.mkdir()
        result = {"status": "Passed", "total": len(selected), "passed": len(selected)}
        result["cases"] = {case["case_id"]: {**case, "status": "Passed",
                           "capture_batch": {"requested": 3, "execution": 3}} for case in selected}
        for phase in ("accuracy", "platform_accuracy"):
            (directory / (phase + "_result.json")).write_text(json.dumps(result))
        speedup = 0.8 if op_id == "a" else 1.2
        result["cases"] = {case["case_id"]: {**case, "status": "Passed", "baseline_valid": True,
                            "speedup": speedup, "ref_median_ms": speedup, "flagfft_median_ms": 1.0}
                           for case in selected}
        (directory / "performance_result.json").write_text(json.dumps(result))
    return tmp_path


def edit(path, change):
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))


def test_gate_weights_cases_and_checks_floor(run):
    report = gate.evaluate(run, 4, 2)
    assert report["passed"]
    assert report["case_weighted_geomean"] == pytest.approx((0.8 * 1.2**3)**0.25)
    assert not gate.evaluate(run, 400, 30)["passed"]
    assert not gate.evaluate(run, 4, 2, floor=0.9)["passed"]


@pytest.mark.parametrize("change", [
    lambda data: data["cases"]["0"].update(baseline_valid=None),
    lambda data: data["cases"]["0"].update(speedup=float("nan")),
    lambda data: data["cases"]["0"].update(flagfft_median_ms=0),
    lambda data: data["cases"]["0"].update(ref_median_ms=10),
    lambda data: data["cases"].pop("0"),
])
def test_gate_rejects_unvalidated_missing_or_inconsistent_timings(run, change):
    edit(run / "a/performance_result.json", change)
    assert not gate.evaluate(run, 4, 2)["passed"]


def test_accuracy_and_combined_run_are_required(run):
    edit(run / "a/accuracy_result.json", lambda data: data.update(passed=0))
    assert not gate.evaluate(run, 4, 2)["passed"]
    edit(run / "manifest.json", lambda data: data["config"].update(performance_only=True))
    assert "a combined accuracy + performance run is required" in gate.evaluate(run, 4, 2)["problems"]


def test_matrix_identity_cannot_be_replaced_by_equal_case_counts(run):
    assert not gate.evaluate(run, 4, 2, matrix_hash="frozen")["passed"]
    edit(run / "manifest.json", lambda data: data["config"].update(matrix_sha256="different"))
    assert not gate.evaluate(run, 4, 2, matrix_hash="frozen")["passed"]
    edit(run / "manifest.json", lambda data: data["config"].update(matrix_sha256="frozen"))
    assert gate.evaluate(run, 4, 2, matrix_hash="frozen")["passed"]


@pytest.mark.parametrize("phase", ["accuracy", "platform_accuracy"])
@pytest.mark.parametrize("change", [
    lambda data: data["cases"]["0"].pop("capture_batch"),
    lambda data: data["cases"]["0"]["capture_batch"].update(execution=1),
    lambda data: data["cases"]["0"].update(status="Failed"),
    lambda data: data["cases"].update(other=data["cases"].pop("0")),
])
def test_gate_rejects_reduced_batches_and_wrong_accuracy_cases(run, phase, change):
    edit(run / f"a/{phase}_result.json", change)
    assert not gate.evaluate(run, 4, 2)["passed"]


def test_gate_requires_full_batch_manifest(run):
    edit(run / "manifest.json", lambda data: data["env"].update(capture_full_batch=False))
    assert not gate.evaluate(run, 4, 2)["passed"]
