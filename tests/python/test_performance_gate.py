"""Device-independent checks that invalid evidence cannot pass the gate."""

import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "performance_gate", Path(__file__).parents[2] / "tools/check_performance_gate.py"
)
gate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gate)


@pytest.fixture
def evidence(tmp_path):
    op = "1d_ct_single_r2c"
    case_id = op + "__n1024__b1__forward"
    accuracy_id = case_id + "__s1"
    (tmp_path / op).mkdir()
    documents = {
        "manifest.json": {
            "performance_cases": [{"op_id": op, "case_id": case_id}],
            "cases": [{"op_id": op, "case_id": accuracy_id}],
        },
        f"{op}/accuracy_result.json": {"cases": {accuracy_id: {"status": "Passed"}}},
        f"{op}/platform_accuracy_result.json": {
            "cases": {accuracy_id: {"status": "Passed"}}
        },
        f"{op}/performance_result.json": {
            "cases": {
                case_id: {
                    "status": "Passed",
                    "flagfft_median_ms": 1.0,
                    "ref_median_ms": 0.9,
                    "speedup": 0.9,
                    "reference_available": True,
                    "baseline_valid": True,
                }
            }
        },
    }

    def save():
        for name, data in documents.items():
            (tmp_path / name).write_text(json.dumps(data))
        return tmp_path

    return documents, save, op, case_id, accuracy_id


def test_valid_and_threshold(evidence):
    docs, save, op, case_id, accuracy_id = evidence
    root = save()
    assert gate.audit(root, 0.8, 1)["passed"]
    assert not gate.audit(root, 1.0, 1)["passed"]
    assert not gate.audit(root, 0.8, 44)["passed"]
    assert not gate.audit(root, 0.8, 1, ascend_ct_single=True)["passed"]


@pytest.mark.parametrize(
    "failure",
    [
        "missing_accuracy",
        "platform_failure",
        "fake_ratio",
        "missing_performance",
        "duplicate",
        "skipped",
        "unknown_baseline",
    ],
)
def test_invalid_evidence(evidence, failure):
    docs, save, op, case_id, accuracy_id = evidence
    performance = docs[f"{op}/performance_result.json"]["cases"][case_id]
    count = 1
    if failure == "missing_accuracy":
        docs[f"{op}/accuracy_result.json"]["cases"].clear()
    elif failure == "platform_failure":
        docs[f"{op}/platform_accuracy_result.json"]["cases"][accuracy_id][
            "status"
        ] = "Failed"
    elif failure == "fake_ratio":
        performance["speedup"] = 5.0
    elif failure == "missing_performance":
        docs[f"{op}/performance_result.json"]["cases"].clear()
    elif failure == "duplicate":
        docs["manifest.json"]["performance_cases"] *= 2
        count = 2
    elif failure == "skipped":
        performance["status"] = "Skipped"
    elif failure == "unknown_baseline":
        performance["baseline_valid"] = None
    assert not gate.audit(save(), 0.8, count)["passed"]
