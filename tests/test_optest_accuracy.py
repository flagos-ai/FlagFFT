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

from tools import run_tests


def test_fft_accuracy(fft_case, fft_build_dir, fft_backend, optest_record, pytestconfig, tmp_path):
    capture = fft_build_dir / "ctest" / "numpy_fft_capture"
    assert capture.is_file(), f"Build the NumPy capture target first: {capture}"

    result = run_tests.run_accuracy_case(
        fft_case,
        capture,
        tmp_path,
        gpu_id=None,
        timeout=pytestconfig.getoption("--fft-case-timeout"),
        scratch_dir=tmp_path / "scratch",
        backend=fft_backend,
    )
    accuracy = result["accuracy"]
    optest_record.update(accuracy=accuracy, platform_accuracy=result["platform_accuracy"])
    assert accuracy["status"] == "Passed", (
        f"{fft_case['case_id']}: {accuracy['status']}: "
        f"{accuracy.get('error', accuracy.get('metric'))}"
    )
