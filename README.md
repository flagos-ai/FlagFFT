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

- `src/utils/`: shared C++ utilities, request/key types, nanobind module glue,
  and internal headers under `src/utils/include/flagfft/`.
- `src/plan/`: plan node definitions, factorization, cost model, automatic route
  selection, forced-plan parsing, and tune candidate enumeration.
- `src/codegen/`: C++ Triton AOT invocation/cache logic plus Python kernel
  source generation. Codelets live in `src/codegen/codelet/`.
- `src/runtime/`: CUDA Driver helpers and plan-owned device allocations.
- `src/exec/`: cuFFT-style C API, raw pointer execution nodes, plan cache, and
  legacy tensor execution used only by the optional Python debug/tune module.
- `src/tune/`: offline tune CLI and SQLite measurement orchestration.

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
`python3`. Generated AOT artifacts and tuned-plan SQLite defaults are stored in
`.flagfft` next to the running executable.

## Tuning

The Python runtime FFT API has been removed. The remaining console script is:

```sh
flagfft-tune --api fft --lengths 4096 --batch 256 --retune
```

The tune path still depends on the optional legacy `_flagfft_core` debug module
for candidate enumeration and forced-plan benchmarking. Build with
`FLAGFFT_BUILD_PYTHON=ON` when using that workflow.

## Benchmark

The supported benchmark entrypoint is the C++ C API tool:

```sh
cmake -S . -B build/cpp-bench -GNinja -DFLAGFFT_BUILD_BENCHMARKS=ON
cmake --build build/cpp-bench --target bench_vs_cufft
build/cpp-bench/bench_vs_cufft --lengths 256,1024 --batch 64
build/cpp-bench/bench_vs_cufft --lengths 4096 --batch 256 --retune
```

`--tune` keeps an existing SQLite winner, while `--retune` supersedes it. The
tool passes `--db` to `flagfft-tune` so tuning records are written to the same
`.flagfft/tuned_plans.sqlite` directory that the benchmark executable reads.
Use `--tune-static-limit` and `--tune-finalists` to reduce tune work for smoke
runs.

The benchmark binds both libraries to one explicit CUDA stream, alternates the
timed order per sample, and reports the median per-launch time from grouped
launches. Use `--launches-per-sample` to control the group size. Output labels
whether FlagFFT used auto planning, an existing SQLite winner, or a per-shape
tune/retune; cuFFT is reported as default contiguous `cufftPlan1d`.

## Validation

C++ plan tests and cuFFT comparison tests are registered with CTest:

```sh
cmake -S . -B build/cpp-tests -GNinja -DFLAGFFT_BUILD_TESTS=ON
cmake --build build/cpp-tests
ctest --test-dir build/cpp-tests --output-on-failure
```

The gtest suite compares `flagfftExecC2C` against `cufftExecC2C` for multiple
batch sizes and both leaf and four-step native routes. Python tests live under
`tests/python/` and cover codegen/tune behavior only.

Benchmark pytest wrappers live under `benchmark/` and invoke `bench_vs_cufft`
directly; they do not call Python FFT runtime APIs.

## License

FlagFFT is licensed under the [Apache 2.0 license](./LICENSE).
