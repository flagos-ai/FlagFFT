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

import pytest

from tools import run_tests


def test_fft_performance(
    fft_case, fft_build_dir, fft_backend, optest_record, pytestconfig, tmp_path
):
    if reason := fft_case.get("skip_reason"):
        pytest.skip(reason)

    capture = fft_build_dir / "ctest" / "numpy_fft_capture"
    cli = fft_build_dir / "flagfft-cli"
    assert capture.is_file() and cli.is_file(), f"Build both FlagFFT test targets in {fft_build_dir}"
    timeout = pytestconfig.getoption("--fft-case-timeout")

    # OpTest launches this suite in a separate process. Check the native
    # reference on the same FFT input before publishing a speedup.
    reference_checks = []
    for case in fft_case["reference_cases"]:
        reference = run_tests.run_accuracy_capture(
            case,
            capture,
            tmp_path / case["case_id"],
            "platform",
            gpu_id=None,
            timeout=timeout,
            preserve_batch=fft_backend == "npu",
        )
        reference_checks.append(reference)
        if reference["status"] != "Passed":
            break
    optest_record["platform_accuracy"] = reference_checks
    baseline_valid = all(reference["status"] == "Passed" for reference in reference_checks)
    result = run_tests.run_performance_case(
        fft_case,
        fft_build_dir,
        tmp_path,
        gpu_id=None,
        timeout=timeout,
        warmup=pytestconfig.getoption("--fft-warmup"),
        iters=pytestconfig.getoption("--fft-iters"),
        baseline_valid=baseline_valid,
        scratch_dir=tmp_path / "scratch",
        backend=fft_backend,
    )["performance"]
    optest_record["performance"] = result
    assert result["status"] == "Passed", (
        f"{fft_case['case_id']}: {result['status']}: {result.get('error', result)}"
    )
    optest_record["measurement"] = {
        "shape_detail": [
            fft_case["shape"],
            fft_case["batch"],
            fft_case["direction"],
            fft_case["placement"],
        ],
        "latency_base": result["ref_median_ms"] if baseline_valid else None,
        "latency": result["flagfft_median_ms"],
        "speedup": result["speedup"] if baseline_valid else None,
    }
    if not baseline_valid:
        optest_record["baseline_skip_reason"] = "Platform FFT reference did not pass NumPy validation"
