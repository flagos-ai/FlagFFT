# FlagFFT Architecture

FlagFFT now treats the C/C++ API as the runtime boundary. Python remains in the
repository for Triton/TLE AOT generation and offline tuning orchestration, but
Python no longer exposes a runtime FFT API.

## C++ Runtime

- `include/flagfft/flagfft.h` declares the cuFFT-style opaque handle API.
- `src/cpp/exec/c_api.cpp` owns `flagfftHandle` lifecycle, plan creation,
  stream state, and raw pointer exec dispatch.
- `src/cpp/plan/` maps a validated `FFTRequest` to a `PlanNode` tree.
- `src/cpp/codegen/` invokes the Python AOT compiler during plan creation and
  loads the resulting cubins as `AotKernel` objects.
- `src/cpp/runtime/` owns CUDA Driver helpers and RAII device allocations used
  by raw plans.

The current native C API slice is intentionally narrow: out-of-place,
contiguous, rank-1, batched `FLAGFFT_C2C` with `complex64` pointers. Plan
creation compiles both forward and inverse kernels so `flagfftExecC2C` only
validates inputs, selects the compiled direction, and launches kernels on the
handle stream. Unsupported ranks, layouts, precisions, and real transforms
return `FLAGFFT_NOT_SUPPORTED`.

## Raw Execution Nodes

Raw nodes mirror the existing plan tree but do not depend on PyTorch tensors:

- `CompiledRawLeafNode` launches a contiguous leaf kernel with plan-owned
  twiddle and DFT table allocations.
- `CompiledRawFourStepFusedNode` supports four-step routes whose row and column
  children are both leaves. It owns the four-step twiddle and intermediate
  stage buffer.

The old `CompiledNode` tensor path remains only to support the optional
`FLAGFFT_BUILD_PYTHON=ON` debug/tune module. It is not part of the default
runtime build surface.

## Python Boundary

Deleted runtime wrappers: top-level `flagfft.py`, `src/api.py`, and
`src/flagfft.py`.

Retained Python files:

- `src/triton_aot.py`
- `src/codegen/`
- `src/codelet/`
- `src/tuning.py`
- `src/flagfft_tune.py`

`flagfft-tune` is still the reserved tune entrypoint. Candidate enumeration and
forced-plan timing currently require the optional legacy `_flagfft_core` module,
so tuning workflows should build with `FLAGFFT_BUILD_PYTHON=ON`.

## Tests

C++ tests live in `ctests/` and are registered by CMake/CTest when cuFFT is
available. They compare `flagfftExecC2C` against `cufftExecC2C` across multiple
batch sizes, directions, and native route shapes.
