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

"""Expose the native build's metadata to flagos-optest."""

import argparse
import os
from pathlib import Path

BACKENDS = {
    "nvidia": ("CUDA", "cuda"),
    "mthreads": ("MUSA", "musa"),
    "thead": ("PPU", "cuda"),
    "iluvatar": ("IX", "cuda"),
    "metax": ("MACA", "cuda"),
    "ascend": ("NPU", "npu"),
    "hygon": ("HCU", "cuda"),
}


def _build_metadata():
    # An editable install can find the source tree. Packaged installs can
    # locate the build by setting FLAGFFT_BUILD_DIR to an absolute path.
    source = Path(__file__).resolve().parents[2]
    build_dir = source / os.environ.get("FLAGFFT_BUILD_DIR", "build")
    try:
        lines = (build_dir / "CMakeCache.txt").read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return None, None, None

    fields = {}
    for line in lines:
        key, _, value = line.partition("=")
        name = key.partition(":")[0]
        if name in {"BACKEND", "CMAKE_PROJECT_VERSION"}:
            fields[name] = value
    for vendor, (backend, runtime_device) in BACKENDS.items():
        if backend == fields.get("BACKEND"):
            return fields.get("CMAKE_PROJECT_VERSION"), vendor, runtime_device
    return None, None, None


__version__, vendor_name, device = _build_metadata()


def main():
    parser = argparse.ArgumentParser(description="Print the FlagFFT CMake backend for an OpTest vendor")
    parser.add_argument("vendor", choices=BACKENDS)
    vendor = parser.parse_args().vendor
    print(BACKENDS[vendor][0])


if __name__ == "__main__":
    main()
