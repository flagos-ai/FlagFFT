"""An operator mean must retain failed/missing cases and the requested batch."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("operator_gate", ROOT / "tools/check_operator_gate.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def fixture(tmp_path, speedups=(2.0, 0.5)):
    op = "2d_c2c"
    folder = tmp_path / op
    folder.mkdir()
    requests, accuracy, platform, timing = [], {}, {}, {}
    for size, speedup in zip((4, 8), speedups):
        key = f"{op}__fp32__n{size}x{size}__b1__forward__outofplace"
        request = dict(case_id=key + "__s1", op_id=op, api="c2c", dtype="torch.float32", rank=2,
                       shape=[size, size], batch=1, placement="out-of-place", direction="forward", scale=1.0)
        requests.append(request)
        measured = dict(request, status="Passed", plan=f"rank=2 n=[{size},{size}] batch=1 type=41\nbody",
                        input_sha256="source", output_sha256="output", metric=dict(passed=True, finite=True))
        accuracy[request["case_id"]] = measured
        platform[request["case_id"]] = copy.deepcopy(measured)
        perf = dict(measured, case_id=key, flagfft_median_ms=1.0, ref_median_ms=speedup, speedup=speedup,
                    reference_available=True, baseline_valid=True)
        perf.pop("scale")
        timing[key] = perf
    return dict(cases=requests, reference_skips={}), accuracy, platform, timing


def write(tmp_path, accuracy, platform, timing):
    for name, records in (("accuracy", accuracy), ("platform_accuracy", platform), ("performance", timing)):
        (tmp_path / "2d_c2c" / f"{name}_result.json").write_text(json.dumps(dict(cases=records)))


def test_gate_uses_each_operator_mean_instead_of_per_case_threshold(tmp_path):
    manifest, accuracy, platform, timing = fixture(tmp_path)
    write(tmp_path, accuracy, platform, timing)
    result = gate.audit(manifest, tmp_path, tmp_path)
    assert result["passed"] and result["operators"]["2d_c2c"]["geometric_mean_speedup"] == 1.0
    assert result["operator_count"] == 1 and result["expected_accuracy_count"] == 2


def test_low_operator_mean_fails(tmp_path):
    manifest, accuracy, platform, timing = fixture(tmp_path, (0.8, 0.8))
    write(tmp_path, accuracy, platform, timing)
    assert not gate.audit(manifest, tmp_path, tmp_path)["passed"]


@pytest.mark.parametrize("phase", ["accuracy", "platform", "timing"])
@pytest.mark.parametrize("failure", ["missing", "failed"])
def test_missing_or_failed_rows_cannot_be_dropped_from_mean(tmp_path, phase, failure):
    manifest, accuracy, platform, timing = fixture(tmp_path, (4.0, 4.0))
    records = dict(accuracy=accuracy, platform=platform, timing=timing)[phase]
    key = next(iter(records))
    if failure == "missing":
        del records[key]
    else:
        records[key]["status"] = "Failed"
    write(tmp_path, accuracy, platform, timing)
    result = gate.audit(manifest, tmp_path, tmp_path)
    assert not result["passed"]
    assert result["operators"]["2d_c2c"]["geometric_mean_speedup"] is None


def test_requested_batch_requires_matching_actual_plan(tmp_path):
    manifest, accuracy, platform, timing = fixture(tmp_path)
    for measured in accuracy.values():
        measured["plan"] = measured["plan"].replace("batch=1", "batch=2")
    for measured in timing.values():
        measured["plan"] = measured["plan"].replace("batch=1", "batch=2")
    write(tmp_path, accuracy, platform, timing)
    assert not gate.audit(manifest, tmp_path, tmp_path)["passed"]


def test_reference_exclusion_keeps_flagfft_accuracy_required(tmp_path):
    manifest, accuracy, platform, timing = fixture(tmp_path)
    request = manifest["cases"][1]
    manifest["reference_skips"][request["case_id"]] = "fixed reference support policy"
    platform[request["case_id"]]["status"] = "Skipped"
    perf = timing[gate.performance_id(request["case_id"])]
    perf["status"] = "Skipped"
    for key in ("flagfft_median_ms", "ref_median_ms", "speedup"):
        perf.pop(key)
    write(tmp_path, accuracy, platform, timing)
    assert gate.audit(manifest, tmp_path, tmp_path)["passed"]
    del accuracy[request["case_id"]]
    write(tmp_path, accuracy, platform, timing)
    assert not gate.audit(manifest, tmp_path, tmp_path)["passed"]


def test_unlisted_reference_skip_fails(tmp_path):
    manifest, accuracy, platform, timing = fixture(tmp_path)
    next(iter(platform.values()))["status"] = "Skipped"
    next(iter(timing.values()))["status"] = "Skipped"
    write(tmp_path, accuracy, platform, timing)
    assert not gate.audit(manifest, tmp_path, tmp_path)["passed"]


@pytest.mark.parametrize("field,value", [("speedup", float("nan")), ("flagfft_median_ms", 0),
    ("ref_median_ms", -1), ("speedup", True), ("speedup", 7), ("baseline_valid", None),
    ("reference_available", False), ("batch", 7), ("case_id", "wrong"), ("plan", "different plan")])
def test_invalid_or_inconsistent_timing_fails(tmp_path, field, value):
    manifest, accuracy, platform, timing = fixture(tmp_path)
    next(iter(timing.values()))[field] = value
    write(tmp_path, accuracy, platform, timing)
    assert not gate.audit(manifest, tmp_path, tmp_path)["passed"]


def test_reference_input_hash_is_checked(tmp_path):
    manifest, accuracy, platform, timing = fixture(tmp_path)
    next(iter(platform.values()))["input_sha256"] = "different"
    write(tmp_path, accuracy, platform, timing)
    assert not gate.audit(manifest, tmp_path, tmp_path)["passed"]


@pytest.mark.parametrize("field", ["metric", "plan"])
def test_error_rows_with_null_diagnostics_are_reported_as_incomplete(tmp_path, field):
    manifest, accuracy, platform, timing = fixture(tmp_path)
    next(iter(accuracy.values()))[field] = None
    write(tmp_path, accuracy, platform, timing)
    result = gate.audit(manifest, tmp_path, tmp_path)
    assert not result["passed"]
    assert result["operators"]["2d_c2c"]["geometric_mean_speedup"] is None


def test_duplicate_manifest_and_json_keys_fail(tmp_path):
    manifest, accuracy, platform, timing = fixture(tmp_path)
    manifest["cases"].append(manifest["cases"][0])
    with pytest.raises(ValueError, match="unique"):
        gate.audit(manifest, tmp_path, tmp_path)
    path = tmp_path / "duplicate.json"
    path.write_text('{"cases": {}, "cases": {}}')
    with pytest.raises(ValueError, match="duplicate"):
        gate.read_json(path)


def test_unexpected_result_row_is_rejected(tmp_path):
    manifest, accuracy, platform, timing = fixture(tmp_path)
    timing["unexpected"] = dict(next(iter(timing.values())), case_id="unexpected")
    write(tmp_path, accuracy, platform, timing)
    assert not gate.audit(manifest, tmp_path, tmp_path)["passed"]


def test_operators_are_not_averaged_together(tmp_path):
    manifest, accuracy, platform, timing = fixture(tmp_path, (2.0, 0.5))
    second = manifest["cases"].pop()
    second_key = second["case_id"]
    second["op_id"] = "another_op"
    manifest["cases"].append(second)
    # Split records into their declared operators, preserving individual means.
    other = tmp_path / "another_op"
    other.mkdir()
    groups = {}
    for label, records in (("accuracy", accuracy), ("platform_accuracy", platform), ("performance", timing)):
        key = gate.performance_id(second_key) if label == "performance" else second_key
        record = records.pop(key)
        record["op_id"] = "another_op"
        (other / f"{label}_result.json").write_text(json.dumps(dict(cases={key:record})))
    write(tmp_path, accuracy, platform, timing)
    result = gate.audit(manifest, tmp_path, tmp_path)
    assert not result["passed"]
    assert result["operators"]["2d_c2c"]["passed"]
    assert not result["operators"]["another_op"]["passed"]
