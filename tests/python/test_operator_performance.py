"""Operator means require complete, independent accuracy and timing evidence."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location("operator_gate",
    Path(__file__).parents[2] / "tools/check_operator_performance.py")
gate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gate)


@pytest.fixture
def evidence(tmp_path):
    op = "3d_c2c"
    performance, accuracy, reference, cases, timings = {}, {}, {}, [], []
    for i, speedup in enumerate((0.5, 2.0, None)):
        case_id = f"{op}__n{i + 2}__b1__forward"
        case = dict(case_id=case_id, op_id=op, shape=[i + 2], batch=1)
        if speedup is None:
            case["skip_reason"] = "fixed reference limitation"
        timings.append(case)
        sample = dict(case, case_id=case_id + "__s1", scale=1.0)
        cases.append(sample)
        accuracy[sample["case_id"]] = dict(status="Passed")
        if speedup is None:
            reference[sample["case_id"]] = dict(status="Skipped", skip_reason=case["skip_reason"])
            performance[case_id] = dict(status="Skipped", skip_reason=case["skip_reason"])
        else:
            reference[sample["case_id"]] = dict(status="Passed")
            performance[case_id] = dict(status="Passed", baseline_valid=True,
                reference_available=True, flagfft_median_ms=1.0,
                ref_median_ms=speedup, speedup=speedup)
    manifest = dict(cases=cases, performance_cases=timings)
    fixed = tmp_path / "fixed.json"
    fixed.write_text(json.dumps(manifest))
    docs = {"manifest.json": copy.deepcopy(manifest),
        f"{op}/performance_result.json": dict(cases=performance),
        f"{op}/accuracy_result.json": dict(cases=accuracy),
        f"{op}/platform_accuracy_result.json": dict(cases=reference)}

    def save():
        for name, value in docs.items():
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
        return tmp_path

    return docs, save, fixed, op


def test_geometric_mean_and_fixed_policy(evidence):
    docs, save, fixed, op = evidence
    root = save()
    result = gate.audit_operators(root, fixed)
    assert result["passed"]  # .5 is a valid case; the operator mean is 1x.
    assert result["operators"][0]["geometric_mean_speedup"] == 1.0
    assert result["operators"][0]["comparable_count"] == 2
    assert result["operators"][0]["policy_skips"] == 1
    assert not gate.audit_operators(root, fixed, threshold=1.01)["passed"]
    assert not gate.audit_operators(root, fixed, ops=["missing"])["passed"]


@pytest.mark.parametrize("fault", ["missing_accuracy", "missing_reference", "missing_performance",
    "timeout", "nan", "bool", "fake_ratio", "invalid_baseline", "new_skip", "policy_expansion",
    "skip_without_numpy", "manifest_omission", "manifest_duplicate", "manifest_changed_shape",
    "unexpected_result"])
def test_incomplete_evidence_fails(evidence, fault):
    docs, save, fixed, op = evidence
    performance = docs[f"{op}/performance_result.json"]["cases"]
    accuracy = docs[f"{op}/accuracy_result.json"]["cases"]
    reference = docs[f"{op}/platform_accuracy_result.json"]["cases"]
    first, second, skipped = list(performance)
    sample = next(iter(accuracy))
    if fault == "missing_accuracy":
        del accuracy[sample]
    elif fault == "missing_reference":
        del reference[sample]
    elif fault == "missing_performance":
        del performance[first]
    elif fault == "timeout":
        performance[first]["status"] = "Timeout"
    elif fault in ("nan", "bool", "fake_ratio"):
        performance[first]["speedup"] = {"nan": float("nan"), "bool": True, "fake_ratio": 5}[fault]
    elif fault == "invalid_baseline":
        performance[first]["baseline_valid"] = None
    elif fault == "new_skip":
        performance[first].update(status="Skipped", skip_reason="new reference limitation")
    elif fault == "policy_expansion":
        performance[skipped]["skip_reason"] = "new reference limitation"
    elif fault == "skip_without_numpy":
        accuracy[skipped + "__s1"]["status"] = "Skipped"
    elif fault == "manifest_omission":
        docs["manifest.json"]["performance_cases"].pop(0)
    elif fault == "manifest_duplicate":
        docs["manifest.json"]["performance_cases"].append(docs["manifest.json"]["performance_cases"][0])
    elif fault == "manifest_changed_shape":
        docs["manifest.json"]["performance_cases"][0]["shape"] = [123]
    elif fault == "unexpected_result":
        performance["unexpected"] = dict(performance[first])
    result = gate.audit_operators(save(), fixed)
    assert not result["passed"]


def test_separate_accuracy_directory(evidence, tmp_path):
    docs, save, fixed, op = evidence
    root = save()
    separate = tmp_path / "accuracy"
    (separate / op).mkdir(parents=True)
    for filename in ("manifest.json", f"{op}/accuracy_result.json", f"{op}/platform_accuracy_result.json"):
        (root / filename).rename(separate / filename)
    (root / "manifest.json").write_text(json.dumps(docs["manifest.json"]))
    assert gate.audit_operators(root, fixed, accuracy_dir=separate)["passed"]
    assert not gate.audit_operators(root, fixed)["passed"]


@pytest.mark.parametrize("key", ["backend", "architecture", "reference_library", "git_commit"])
def test_incompatible_evidence_environment(evidence, key):
    docs, save, fixed, op = evidence
    matrix = json.loads(fixed.read_text())
    matrix["env"] = {key: "original"}
    fixed.write_text(json.dumps(matrix))
    docs["manifest.json"]["env"] = {key: "changed"}
    if key == "git_commit":
        # A candidate may differ from the baseline commit. Accuracy and
        # performance must still describe the same candidate implementation.
        root = save()
        other = root / "other"
        (other / op).mkdir(parents=True)
        for name in ("manifest.json", f"{op}/accuracy_result.json", f"{op}/platform_accuracy_result.json"):
            (other / name).write_text((root / name).read_text())
        acc_manifest = dict(docs["manifest.json"], env={key: "older_candidate"})
        (other / "manifest.json").write_text(json.dumps(acc_manifest))
        assert not gate.audit_operators(root, fixed, accuracy_dir=other)["passed"]
    else:
        assert not gate.audit_operators(save(), fixed)["passed"]
