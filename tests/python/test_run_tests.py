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

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "flagfft_run_tests", ROOT / "tools" / "run_tests.py"
)
assert SPEC is not None and SPEC.loader is not None
RUN_TESTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN_TESTS)


def test_multidimensional_full_case_counts_and_batch() -> None:
    operators = RUN_TESTS.load_operators(ROOT / "conf" / "operators.yaml")
    matrix = RUN_TESTS.load_test_matrix(ROOT / "conf" / "test_matrix.yaml")

    cases_2d = RUN_TESTS.expand_test_cases(operators, matrix, "2d_full")
    cases_3d = RUN_TESTS.expand_test_cases(operators, matrix, "3d_full")

    assert len(cases_2d) == len(matrix["sizes_2d_full"]) * 8
    assert len(cases_3d) == len(matrix["sizes_3d_full"]) * 8
    assert {case["rank"] for case in cases_2d} == {2}
    assert {case["rank"] for case in cases_3d} == {3}
    assert {case["batch"] for case in cases_2d + cases_3d} == {1}


def test_bs_combination_only_expands_prime_cases() -> None:
    operators = RUN_TESTS.load_operators(ROOT / "conf" / "operators.yaml")
    matrix = RUN_TESTS.load_test_matrix(ROOT / "conf" / "test_matrix.yaml")

    cases = RUN_TESTS.expand_test_cases(operators, matrix, "bs")

    assert len(cases) == len(matrix["sizes_bs"]) * 10
    assert {case["algo"] for case in cases} == {"bs"}
    assert {case["nx"] for case in cases} == set(matrix["sizes_bs"])
    assert {case["rank"] for case in cases} == {1}


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
