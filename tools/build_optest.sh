#!/usr/bin/env bash
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

set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
    echo "Usage: tools/build_optest.sh [vendor] [CMake options...]"
    echo "Set FLAGOS_GPU_VENDOR, FLAGFFT_PYTHON and FLAGFFT_BUILD_DIR to use the active runtime."
    echo "Set FLAGFFT_CMAKE and FLAGFFT_CMAKE_GENERATOR when required by the SDK."
    exit 0
fi

fft_source="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fft_vendor="${1:-${FLAGOS_GPU_VENDOR:-}}"
if (( $# > 0 )); then
    shift
fi
fft_python="$(command -v "${FLAGFFT_PYTHON:-python3}")"
fft_cmake="${FLAGFFT_CMAKE:-cmake}"
fft_build="${FLAGFFT_BUILD_DIR:-build}"
if [[ "$fft_build" != /* ]]; then
    fft_build="$fft_source/$fft_build"
fi
fft_backend="$(
    PYTHONPATH="$fft_source/python${PYTHONPATH:+:$PYTHONPATH}" \
        "$fft_python" -m flagfft_codegen.optest_metadata "$fft_vendor"
)"

if [[ ! -f "$fft_source/deps/libtriton_jit/CMakeLists.txt" ]]; then
    git -C "$fft_source" submodule update --init --recursive -- deps/libtriton_jit
fi

"$fft_cmake" -S "$fft_source" -B "$fft_build" -G "${FLAGFFT_CMAKE_GENERATOR:-Ninja}" \
    -DCMAKE_BUILD_TYPE=Release "$@" \
    -DBACKEND="$fft_backend" -DPython_EXECUTABLE="$fft_python" \
    -DFLAGFFT_BUILD_TESTS=ON -DFLAGFFT_BUILD_CLI=ON
"$fft_cmake" --build "$fft_build" --target flagfft-cli numpy_fft_capture
test -x "$fft_build/flagfft-cli" && test -x "$fft_build/ctest/numpy_fft_capture"
