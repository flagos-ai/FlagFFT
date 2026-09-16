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

"""Smoke test for the plain-Triton Ascend FlagFFT MVP.

The input/output tensors are contiguous float32 buffers with interleaved
real/imaginary values.  The script calls the public FlagFFT C API through
ctypes, so it validates the ACL adaptor, embedded libtriton_jit path, NPU
kernel launch, and C API direction/batch handling in one process.
"""

from __future__ import annotations

import argparse
import ctypes
from pathlib import Path

import numpy as np
import torch
import torch_npu  # noqa: F401  # activates the Ascend torch backend


FLAGFFT_SUCCESS = 0
FLAGFFT_NOT_SUPPORTED = 14
FLAGFFT_C2C = 0x29
FLAGFFT_FORWARD = -1
FLAGFFT_INVERSE = 1


def _load_api(library: Path):
    api = ctypes.CDLL(str(library))
    handle = ctypes.c_void_p
    api.flagfftPlan1d.argtypes = [
        ctypes.POINTER(handle),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    api.flagfftPlan1d.restype = ctypes.c_int
    api.flagfftPlan2d.argtypes = [
        ctypes.POINTER(handle),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    api.flagfftPlan2d.restype = ctypes.c_int
    api.flagfftExecC2C.argtypes = [
        handle,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    api.flagfftExecC2C.restype = ctypes.c_int
    api.flagfftDestroy.argtypes = [handle]
    api.flagfftDestroy.restype = ctypes.c_int
    return api, handle


def _signal(batch: int, n: int) -> np.ndarray:
    values = np.empty((batch, n, 2), dtype=np.float32)
    index = np.arange(batch * n, dtype=np.float32)
    values[..., 0] = np.sin(0.13 * (index + 1)).reshape(batch, n)
    values[..., 1] = np.cos(0.07 * (index + 3)).reshape(batch, n)
    return values


def _run_case(api, handle_type, n: int, batch: int) -> None:
    signal = _signal(batch, n)
    signal_complex = signal[..., 0] + 1j * signal[..., 1]
    spectrum = np.fft.fft(signal_complex, axis=1)
    sources = (
        (FLAGFFT_FORWARD, signal, spectrum),
        (
            FLAGFFT_INVERSE,
            np.ascontiguousarray(
                np.stack((spectrum.real, spectrum.imag), axis=-1).astype(np.float32)
            ),
            n * signal_complex,
        ),
    )

    plan = handle_type()
    result = api.flagfftPlan1d(ctypes.byref(plan), n, FLAGFFT_C2C, batch)
    if result != FLAGFFT_SUCCESS:
        raise RuntimeError(f"flagfftPlan1d(n={n}, batch={batch}) returned {result}")
    try:
        for direction, source, expected in sources:
            source_tensor = torch.from_numpy(np.ascontiguousarray(source)).to("npu")
            output_tensor = torch.empty_like(source_tensor)
            result = api.flagfftExecC2C(
                plan,
                ctypes.c_void_p(source_tensor.data_ptr()),
                ctypes.c_void_p(output_tensor.data_ptr()),
                direction,
            )
            if result != FLAGFFT_SUCCESS:
                raise RuntimeError(
                    f"flagfftExecC2C(n={n}, batch={batch}, direction={direction}) "
                    f"returned {result}"
                )
            torch.npu.synchronize()
            actual = output_tensor.cpu().numpy()
            actual_complex = actual[..., 0] + 1j * actual[..., 1]
            max_error = float(np.max(np.abs(actual_complex - expected)))
            print(
                f"n={n} batch={batch} direction={direction} "
                f"max_abs_error={max_error:.6g}"
            )
            tolerance = 1.0e-2 if n > 8 else 3.0e-3
            if max_error > tolerance:
                raise RuntimeError(
                    f"accuracy check failed for n={n}, direction={direction}: "
                    f"{max_error} > {tolerance}"
                )
    finally:
        result = api.flagfftDestroy(plan)
        if result != FLAGFFT_SUCCESS:
            raise RuntimeError(f"flagfftDestroy(n={n}) returned {result}")


def _check_rejected_shapes(api, handle_type) -> None:
    plan = handle_type()
    result = api.flagfftPlan1d(ctypes.byref(plan), 256, FLAGFFT_C2C, 1)
    if result != FLAGFFT_NOT_SUPPORTED:
        raise RuntimeError(f"n=256 expected {FLAGFFT_NOT_SUPPORTED}, got {result}")
    result = api.flagfftPlan2d(ctypes.byref(plan), 8, 8, FLAGFFT_C2C)
    if result != FLAGFFT_NOT_SUPPORTED:
        raise RuntimeError(f"2D expected {FLAGFFT_NOT_SUPPORTED}, got {result}")
    print("unsupported-shapes=PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--library",
        type=Path,
        required=True,
        help="path to the BACKEND=NPU libflagfft.so",
    )
    args = parser.parse_args()
    api, handle_type = _load_api(args.library)
    _run_case(api, handle_type, n=8, batch=3)
    _run_case(api, handle_type, n=128, batch=2)
    _check_rejected_shapes(api, handle_type)
    print("flagfft_plain_ascend_mvp=PASS")

if __name__ == "__main__":
    main()
