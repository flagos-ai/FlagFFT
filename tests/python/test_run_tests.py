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

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "flagfft_run_tests", ROOT / "tools" / "run_tests.py"
)
assert SPEC is not None and SPEC.loader is not None
RUN_TESTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN_TESTS)


def test_combinations_expand_to_expected_scope_and_count() -> None:
    operators = RUN_TESTS.load_operators(ROOT / "conf" / "operators.yaml")
    matrix = RUN_TESTS.load_test_matrix(ROOT / "conf" / "test_matrix.yaml")

    expected_scope = {
        "1d_ct_single": (1, {"ct"}),
        "1d_ct_batch": (1, {"ct"}),
        "1d_bs_single": (1, {"bs"}),
        "1d_bs_batch": (1, {"bs"}),
        "2d_ct": (2, {"2d"}),
        "2d_bs": (2, {"2d"}),
        "3d": (3, {"3d"}),
    }

    for name, (rank, algorithms) in expected_scope.items():
        combo = matrix["combinations"][name]
        sizes = matrix[combo["sizes"]]
        batches = combo.get("batches", [1])
        scales = combo.get("scales", [1.0])
        cases = RUN_TESTS.expand_test_cases(operators, matrix, name)

        assert cases
        assert {case["rank"] for case in cases} == {rank}
        assert {case["algo"] for case in cases} == algorithms
        # 18 operators (6 per rank) expose 8 direction entries in total.
        assert len(cases) == len(sizes) * len(batches) * len(scales) * 8


def test_full_expands_every_combination_once() -> None:
    operators = RUN_TESTS.load_operators(ROOT / "conf" / "operators.yaml")
    matrix = RUN_TESTS.load_test_matrix(ROOT / "conf" / "test_matrix.yaml")

    full_cases = RUN_TESTS.expand_all_test_cases(operators, matrix)
    manual_cases = []
    for name in matrix["combinations"]:
        manual_cases.extend(RUN_TESTS.expand_test_cases(operators, matrix, name))

    assert full_cases == manual_cases
    assert len(full_cases) == len(manual_cases)

    case_keys = [
        (
            case["op_id"],
            case["algo"],
            case["nx"],
            case["ny"],
            case["nz"],
            case["batch"],
            case["direction"],
            case["scale"],
        )
        for case in full_cases
    ]
    assert len(set(case_keys)) == len(full_cases)

    assert RUN_TESTS.resolve_combination_names("full", matrix) == list(
        matrix["combinations"]
    )
    assert RUN_TESTS.resolve_combination_names("all", matrix) == list(
        matrix["combinations"]
    )


def test_full_and_all_must_be_used_alone() -> None:
    matrix = RUN_TESTS.load_test_matrix(ROOT / "conf" / "test_matrix.yaml")

    with pytest.raises(SystemExit):
        RUN_TESTS.resolve_combination_names("full,1d_ct_single", matrix)
    with pytest.raises(SystemExit):
        RUN_TESTS.resolve_combination_names("1d_ct_single,all", matrix)
    with pytest.raises(SystemExit):
        RUN_TESTS.resolve_combination_names("full,all", matrix)


def test_3d_accuracy_command_uses_rank_binary_and_api_filter(tmp_path: Path) -> None:
    case = {
        "op_id": "c2c_3d",
        "algo": "3d",
        "nx": 23,
        "ny": 30,
        "nz": 67,
        "batch": 1,
        "scale": 1.0,
        "direction": "forward",
        "ctest": "test_3d_correctness",
        "cli_type": "c2c",
        "rank": 3,
    }

    command, _ = RUN_TESTS.build_accuracy_cmd(case, tmp_path)

    assert command[0] == str(tmp_path / "ctest" / "test_3d_correctness")
    assert "--nx=23" in command
    assert "--ny=30" in command
    assert "--nz=67" in command
    assert "--batch=1" in command
    assert "--api=c2c" in command
    assert "--direction=forward" in command


def test_aggregate_results_preserves_case_failures() -> None:
    raw_results = [
        {
            "op_id": "c2c_2d",
            "phase": "accuracy",
            "case": "accuracy-pass",
            "result": {
                "status": "Passed",
                "total": 1,
                "passed": 1,
                "failed": 0,
                "skipped": 0,
            },
        },
        {
            "op_id": "c2c_2d",
            "phase": "accuracy",
            "case": "accuracy-error",
            "result": {"status": "Error", "stderr": "launch failed"},
        },
        {
            "op_id": "c2c_2d",
            "phase": "performance",
            "case": "performance-pass",
            "result": {
                "status": "Passed",
                "flagfft_median_ms": 1.0,
                "ref_median_ms": 2.0,
                "speedup": 2.0,
            },
        },
        {
            "op_id": "c2c_2d",
            "phase": "performance",
            "case": "performance-error",
            "result": {"status": "Error", "stderr": "launch failed"},
        },
    ]

    result = RUN_TESTS.aggregate_results(raw_results, [{"id": "c2c_2d"}])["c2c_2d"]

    assert result["accuracy"]["status"] == "Failed"
    assert result["accuracy"]["total"] == 2
    assert result["accuracy"]["failed"] == 1
    assert result["performance"]["status"] == "Failed"
