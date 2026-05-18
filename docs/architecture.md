# FlagFFT Architecture

FlagFFT now treats the C/C++ API as the runtime boundary. Python remains in the
repository for Triton/TLE AOT generation and offline tuning orchestration, but
Python no longer exposes a runtime FFT API.

## C++ Runtime

- `include/flagfft/flagfft.h` declares the cuFFT-style opaque handle API.
- `src/exec/` owns `flagfftHandle` lifecycle, plan creation, stream state,
  plan cache, raw pointer exec dispatch, and optional legacy tensor execution.
- `src/plan/` maps a validated `FFTRequest` to a `PlanNode` tree and is split
  into node, factorization, cost, auto-candidate, and tune-candidate units.
- `src/codegen/` invokes the Python AOT compiler during plan creation, loads
  cubins as `AotKernel` objects, and contains Python Triton/TLE source
  generation plus `src/codegen/codelet/`.
- `src/runtime/` owns CUDA Driver helpers and RAII device allocations used by
  raw plans.
- `src/utils/` owns shared request/key utilities, nanobind glue, and internal
  headers under `src/utils/include/flagfft/`.

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

## Build Options

The default CMake build produces only the C++ shared library `flagfft`. Optional
targets are controlled in the same build directory with `-D` switches:

- `FLAGFFT_BUILD_TESTS=ON` builds C++ gtest targets.
- `FLAGFFT_BUILD_BENCHMARKS=ON` builds `bench_vs_cufft`.
- `FLAGFFT_BUILD_PYTHON=ON` builds the legacy `_flagfft_core` debug/tune module.

## Python Boundary

Deleted runtime wrappers: top-level `flagfft.py`, `src/api.py`, and
`src/flagfft.py`.

Retained Python files:

- `src/triton_aot.py`
- `src/codegen/`
- `src/codegen/codelet/`
- `src/tune/`

`flagfft-tune` is still the reserved tune entrypoint. Candidate enumeration and
forced-plan timing currently require the optional legacy `_flagfft_core` module,
so tuning workflows should build with `FLAGFFT_BUILD_PYTHON=ON`.

Default generated artifacts and tuned-plan SQLite files live in `.flagfft`
beside the running executable unless an explicit override such as
`FLAGFFT_TUNE_DB` is provided.

The benchmark path is a C++ executable, `bench_vs_cufft`, built with
`FLAGFFT_BUILD_BENCHMARKS=ON`. It exercises the public C API, optionally invokes
`flagfft-tune` with `--tune` or `--retune`, and points both tuning and execution
at the executable-local `.flagfft/tuned_plans.sqlite`.
Timing uses a shared explicit CUDA stream for FlagFFT and cuFFT, alternates the
measurement order per sample, and divides grouped launch time by
`--launches-per-sample` before taking the median. Benchmark output labels the
FlagFFT plan mode separately from cuFFT's default contiguous `cufftPlan1d`.

## Tests

C++ tests live in `tests/ctest/` and are registered by CMake/CTest. Plan tests
cover route selection and key structure; cuFFT comparison tests cover
`flagfftExecC2C` across multiple batch sizes, directions, and native route
shapes when cuFFT is available. Python tests live in `tests/python/` and cover
codegen/tune behavior only.

Benchmark pytest cases live in `benchmark/` and shell out to `bench_vs_cufft`;
they are smoke/registration tests for the tool, not Python FFT runtime tests.
