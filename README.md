# FlagFFT

FlagFFT is an experimental C++ FFT library with a cuFFT-style API and
Triton/TLE-generated CUDA kernels. The public runtime interface is C; Python
is retained only for Triton/TLE JIT source generation (internal codegen).

## Current API

The public header is `include/flagfft/flagfft.h` and exposes:

- `flagfftPlan1d`, `flagfftPlan2d`, `flagfftPlan3d`, `flagfftPlanMany`
- `flagfftExecC2C`, `flagfftExecZ2Z`, `flagfftExecR2C`, `flagfftExecD2Z`,
  `flagfftExecC2R`, `flagfftExecZ2D`
- `flagfftSetStream`, `flagfftDestroy`
- `flagfftGetPlanDescription`

The first native runtime slice supports arbitrary-length out-of-place,
contiguous, rank-1, batched `FLAGFFT_C2C` transforms on `complex64` device
pointers. Forward and inverse C2C kernels are compiled during plan creation and
selected at exec time.
Other declared plan/exec combinations return `FLAGFFT_NOT_SUPPORTED` until
their raw execution paths are implemented.

`flagfftGetPlanDescription(plan)` returns a human-readable string describing
the plan node tree, kernel names, module paths, and compilation details.
Useful for understanding which execution path was selected and for performance
debugging. The returned pointer is valid for the lifetime of the plan.

## Architecture

- `src/utils/`: shared C++ utilities, request/key types, JSON serialization,
  SQLite wrapper, and internal headers under `src/utils/include/flagfft/`.
- `src/plan/`: plan node definitions, factorization, cost model, automatic route
  selection, JSON deserialization, and tune candidate enumeration.
- `src/codegen/`: C++ libtriton_jit invocation/cache logic plus Python kernel
  source generation. Codelets live in `src/codegen/codelet/`.
- `src/runtime/`: CUDA Driver helpers.
- `src/exec/`: cuFFT-style C API, raw pointer execution nodes, and tuned plan
  lookup.
- `src/tune_cpp/`: C++ offline tuning CLI and SQLite measurement orchestration.

The C API uses an opaque `flagfftHandle`. Internally it owns the immutable plan
description, stream/lifecycle state, compiled forward and inverse raw execution
nodes, and device buffers for twiddle/table/stage data. Exec calls do not import
Python, compile kernels, rebuild plans, or allocate large buffers.

## Kernel Backend

FlagFFT is JIT-only. It requires the `deps/libtriton_jit` submodule and targets
CUDA through `FLAGFFT_LIBTRITON_JIT_BACKEND=CUDA`.

```sh
cmake -S . -B build -GNinja
cmake --build build
```

Plan creation emits Triton source and calls libtriton_jit compile APIs so the
first `flagfftExecC2C` does not pay Python compilation latency. The raw C API
supports leaf, fused leaf/leaf four-step, and Bluestein fallback routes for
arbitrary 1D C2C lengths. Non-C2C, rank>1, in-place, and non-contiguous C API
requests keep returning `FLAGFFT_NOT_SUPPORTED`.

## Build

Configure and build the C++ library. Optional targets are controlled by CMake
`-D` switches in the same build directory; the default build only produces
`flagfft`.

```sh
cmake -S . -B build -GNinja
cmake --build build
```

Enable optional targets by reconfiguring the same directory:

```sh
cmake -S . -B build -GNinja \
  -DFLAGFFT_BUILD_TESTS=ON \
  -DFLAGFFT_BUILD_BENCHMARKS=ON
cmake --build build
```

`FLAGFFT_BUILD_TESTS` and `FLAGFFT_BUILD_BENCHMARKS` are disabled by default.

When plan creation emits Triton JIT source, it uses `FLAGFFT_PYTHON` if set,
otherwise `python3`. Generated source/metadata and tuned-plan SQLite defaults
are stored in `.flagfft` next to the running executable. Tuned plan lookups
read from `FLAGFFT_TUNE_DB` if set, or the default `.flagfft/tuned_plans.sqlite`
next to the executable.

## Tuning

The C++ tuning tool `flagfft_tune` benchmarks candidate plans directly via the
raw C API, verifies the winner against cuFFT, and persists results to a SQLite
database that the runtime reads at plan creation time.

```sh
cmake -S . -B build -GNinja -DFLAGFFT_BUILD_BENCHMARKS=ON
cmake --build build
./build/flagfft_tune --db /path/to/tuned_plans.sqlite --lengths 256 1024 4096
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--lengths N...` | (required) | FFT lengths to tune |
| `--batch B` | `1` | Batch size |
| `--direction` | `forward` | `forward` or `inverse` |
| `--api` | `fft` | `fft` (forward) or `ifft` (inverse) |
| `--warmup W` | `10` | Number of warmup kernel launches |
| `--iters K` | `100` | Number of timed iterations |
| `--db PATH` | (required) | SQLite database path |
| `--retune` | off | Re-benchmark prior winner alongside candidates |
| `--static-limit N` | `32` | Max candidate plans to benchmark |
| `--finalists N` | `3` | Number of top candidates verified against cuFFT |

The tune database is compatible with the C API runtime: set `FLAGFFT_TUNE_DB`
to the same path and `flagfftPlan1d` will use the tuned winner automatically.

## Benchmark

The supported benchmark entrypoint is the C++ C API tool:

```sh
cmake -S . -B build -GNinja -DFLAGFFT_BUILD_BENCHMARKS=ON
cmake --build build --target bench_vs_cufft
./build/bench_vs_cufft --lengths 256,1024 --batch 64
./build/bench_vs_cufft --lengths 4096 --batch 256 --retune
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--lengths N1,N2,...` | `256,1024,4096,...` | Comma-separated FFT lengths |
| `--batch B` | `64` | Batch size |
| `--direction` | `forward` | `forward` or `inverse` |
| `--warmup W` | `10` | Number of warmup iterations |
| `--iters K` | `100` | Number of timed samples |
| `--launches-per-sample K` | `10` | Group size per timed sample |
| `--tune` | off | Tune before benchmark if winner absent |
| `--retune` | off | Tune before benchmark (supersedes winner) |
| `--tune-db PATH` | `.flagfft/tuned_plans.sqlite` | Tune database path |
| `--tune-command CMD` | `flagfft_tune` | Tune executable |
| `--tune-static-limit N` | `32` | Max tune candidates |
| `--tune-finalists N` | `3` | Top candidates to verify |
| `--print-path` | off | Print execution plan and kernel details |

`--tune` keeps an existing SQLite winner, while `--retune` supersedes it. The
tool passes `--db` to `flagfft_tune` so tuning records are written to the same
`.flagfft/tuned_plans.sqlite` directory that the benchmark executable reads.

The benchmark binds both libraries to one explicit CUDA stream, alternates the
timed order per sample, and reports the median per-launch time from grouped
launches. Use `--launches-per-sample` to control the group size. Output labels
the FlagFFT kernel backend and whether FlagFFT used auto planning, an existing
SQLite winner, or a per-shape tune/retune; cuFFT is reported as default
contiguous `cufftPlan1d`.

### Path printing

`--print-path` calls `flagfftGetPlanDescription` after plan creation and prints
the plan node tree and compiled kernel details before the timing table:

```
=== FlagFFT Plan ===
rank=1 n=[10007] batch=1 type=41

-- Plan tree --
Bluestein(n=10007, conv_length=20020)
  FourStep(n=20020, n1=13, n2=1540)
    LeafPlan(n=13, factors=[13], lanes=1, num_warps=1, smem_size=0)
    LeafPlan(n=1540, factors=[11,10,7,2], lanes=2, num_warps=1, smem_size=2048)

-- Forward execution --
CompiledRawBluestein(n=10007, conv_length=20020,
  prepare_kernel=_bluestein_prepare_kernel,
  pointwise_kernel=_bluestein_pointwise_kernel,
  finalize_kernel=_bluestein_finalize_kernel,
  fft=CompiledRawFourStepFused(n=20020, n1=13, n2=1540,
    row_kernel=four_step_row_fft_kernel_13_n13_1540_l1_b1,
    col_kernel=four_step_col_fft_kernel_11_10_7_2_n13_1540_l2_b2))
```

This shows which plan strategy was selected (Leaf, FourStep, Bluestein), the
factorization, kernel names, and module paths — useful for performance
debugging and understanding how the planner routes a given FFT size.

## Validation

C++ plan tests and cuFFT comparison tests are registered with CTest:

```sh
cmake -S . -B build -GNinja -DFLAGFFT_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

The gtest suite compares `flagfftExecC2C` against `cufftExecC2C` for multiple
batch sizes and both leaf and four-step native routes. Python tests live under
`tests/python/` and cover codegen behavior only.

Benchmark pytest wrappers live under `benchmark/` and invoke `bench_vs_cufft`
directly; they do not call Python FFT runtime APIs.

## License

FlagFFT is licensed under the [Apache 2.0 license](./LICENSE).
