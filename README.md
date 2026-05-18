# FlagFFT

FlagFFT is an experimental C++ FFT library with a cuFFT-style API and
Triton/TLE-generated CUDA kernels. The public runtime interface is C/C++; Python
is retained only for Triton AOT code generation and the offline tune entrypoint.

## Current API

The public header is `include/flagfft/flagfft.h` and exposes:

- `flagfftPlan1d`, `flagfftPlan2d`, `flagfftPlan3d`, `flagfftPlanMany`
- `flagfftExecC2C`, `flagfftExecZ2Z`, `flagfftExecR2C`, `flagfftExecD2Z`,
  `flagfftExecC2R`, `flagfftExecZ2D`
- `flagfftSetStream`, `flagfftDestroy`

The first native runtime slice supports out-of-place, contiguous, rank-1,
batched `FLAGFFT_C2C` transforms on `complex64` device pointers. Forward and
inverse C2C kernels are compiled during plan creation and selected at exec time.
Other declared plan/exec combinations return `FLAGFFT_NOT_SUPPORTED` until
their raw execution paths are implemented.

## Architecture

- `src/cpp/common/`: FFT request and key types reused by planning/codegen.
- `src/cpp/plan/`: plan tree construction for `ct_leaf` and fused `four_step`
  routes.
- `src/cpp/codegen/`: invokes the Python Triton AOT compiler and loads cubins.
- `src/cpp/runtime/`: CUDA Driver helpers and plan-owned device allocations.
- `src/cpp/exec/`: cuFFT-style C API, raw pointer execution, and legacy tensor
  execution code used only when the optional Python debug module is enabled.
- `src/codegen/` and `src/codelet/`: Python kernel source generation.

The C API uses an opaque `flagfftHandle`. Internally it owns the immutable plan
description, stream/lifecycle state, compiled forward and inverse raw execution
nodes, and device buffers for twiddle/table/stage data. Exec calls do not import
Python, compile kernels, rebuild plans, or allocate large buffers.

## Build

Configure and build the C++ library:

```sh
cmake -S . -B build/cpp -GNinja
cmake --build build/cpp
```

The legacy nanobind debug module is optional and disabled by default:

```sh
cmake -S . -B build/cpp-python -GNinja -DFLAGFFT_BUILD_PYTHON=ON
cmake --build build/cpp-python
```

When plan creation invokes Triton AOT, it uses `FLAGFFT_PYTHON` if set, otherwise
`python3`. Run C++ tests from the repository root or keep the CTest working
directory unchanged so the local `src/triton_aot.py` entrypoint is visible.

## Tuning

The Python runtime FFT API has been removed. The remaining console script is:

```sh
flagfft-tune --api fft --lengths 4096 --batch 256 --retune
```

The tune path still depends on the optional legacy `_flagfft_core` debug module
for candidate enumeration and forced-plan benchmarking. Build with
`FLAGFFT_BUILD_PYTHON=ON` when using that workflow.

## Validation

C++ correctness tests are registered with CTest when cuFFT is available:

```sh
cmake -S . -B build/cpp-tests -GNinja -DFLAGFFT_BUILD_TESTS=ON
cmake --build build/cpp-tests
ctest --test-dir build/cpp-tests --output-on-failure
```

The gtest suite compares `flagfftExecC2C` against `cufftExecC2C` for multiple
batch sizes and both leaf and four-step native routes.

## License

FlagFFT is licensed under the [Apache 2.0 license](./LICENSE).
