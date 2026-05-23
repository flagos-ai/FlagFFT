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

The native runtime supports arbitrary-length contiguous rank-1 batched
`FLAGFFT_C2C` (`complex64`) and `FLAGFFT_Z2Z` (`complex128`) transforms on
device pointers, plus `FLAGFFT_R2C`, `FLAGFFT_D2Z`, `FLAGFFT_C2R`, and
`FLAGFFT_Z2D` compact real transforms. Real forward outputs and real inverse
inputs use `n / 2 + 1` complex values per batch. In-place real transforms use
the cuFFT padded physical row layout, `2 * (n / 2 + 1)` real scalars per
batch. Forward and inverse kernels are compiled during plan creation and
selected at exec time for complex transforms. Both single-layer and nested
fused four-step routes are supported, so very large composite lengths
(e.g. `n = 2^23`) plan as multi-level four-step trees instead of falling back
to Bluestein.
Rank>1 and non-contiguous/custom stride or embed C API requests still return
`FLAGFFT_NOT_SUPPORTED`.

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
- `src/cli_tools/`: unified native CLI, shared execution/timing code, and
  SQLite tuning orchestration.

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
first exec call does not pay Python compilation latency. The raw C API
supports leaf, fused leaf/leaf four-step, generic nested four-step (FourStep of
arbitrary supported children), and Bluestein fallback routes for arbitrary 1D
complex lengths. Real transforms use pointwise staging around the complex FFT
route: real-to-complex, half-spectrum pack, compact Hermitian expansion, and
complex-to-real pack. Rank>1 and non-contiguous C API requests keep returning
`FLAGFFT_NOT_SUPPORTED`.

## Build

Configure and build the C++ library. The optional native validation,
benchmarking, and tuning entrypoint is built with `FLAGFFT_BUILD_CLI=ON`.

```sh
cmake -S . -B build -GNinja \
  -DFLAGFFT_BUILD_CLI=ON
cmake --build build --target flagfft-cli
```

The removed `FLAGFFT_BUILD_TESTS` and `FLAGFFT_BUILD_BENCHMARKS` switches no
longer create gtest, `bench_vs_cufft`, or `flagfft-tuner` executables.

When plan creation emits Triton JIT source, it uses `FLAGFFT_PYTHON` if set,
otherwise `python3`. Generated source/metadata and tuned-plan SQLite defaults
are stored in `.flagfft` next to the running executable. Tuned plan lookups
read from `FLAGFFT_TUNE_DB` if set, or the default `.flagfft/tuned_plans.sqlite`
next to the executable.

## Native CLI

`flagfft-cli` is the only native executable interface for validation,
measurement, and offline tuning:

```sh
./build/flagfft-cli test --suite correctness --api c2c --shape 256 --batch 4 --json
./build/flagfft-cli bench --api r2c --shape 4096 --batch 64 --warmup 10 --iters 100 \
  --launches-per-sample 10 --json
./build/flagfft-cli tune --api c2c --shape 256 --db /tmp/plans.sqlite \
  --warmup 10 --iters 100 --static-limit 32 --finalists 3 --json
```

Common case options are `--api c2c|z2z|r2c|d2z|c2r|z2d`,
repeatable `--shape N|NxM|NxMxK`, `--batch`, `--direction forward|inverse`,
`--placement out-of-place|in-place`, `--plan-api
plan1d|plan2d|plan3d|planmany`, and `--stream`. Repeated shapes share the
other case options.
The legacy comma-separated `--lengths` option is not part of this unified
interface; pass one or more complete `--shape` options instead.

`test --suite plan` runs route/key assertions, `test --suite api-errors`
checks C API error contracts, and `test --suite correctness` executes cases
against cuFFT. `bench` runs the same correctness comparison before reporting
CUDA event `median`/`p90` timings and speedup; `--print-path` adds the plan
description. `tune` benchmarks candidates, verifies finalists against cuFFT,
and writes the winning plan to SQLite; `--retune` supersedes an earlier winner.
Set `FLAGFFT_TUNE_DB` to that database when running `test` or `bench` to use it.
Integer option tokens must be fully numeric; for example, `--shape 16suffix`
and `--batch 2suffix` are rejected as invalid arguments. Correctness reports
include `non_finite_values`; any nonzero count fails both `test` and the
pre-benchmark correctness gate.

### Capability Matrix

| Command | Supported now | Reported `unsupported` |
|---|---|---|
| `test correctness`, `bench` | Six 1D APIs with `plan1d`, both complex directions, valid real direction, in/out-of-place; padded real in-place `planmany` | Rank 2/3 and other `planmany` layouts |
| `tune` | 1D `c2c` complex64, out-of-place `plan1d`, either direction | Other APIs, ranks, or layouts |

The JSON `status` and process code contract is stable: `passed`=`0`,
`failed` or invalid arguments=`1`, setup/runtime `error`=`2`, and
`skipped` (only after a successful CUDA query reports no device) or
`unsupported`=`77`. CUDA runtime initialization/query failures are `error`=`2`.

### Path printing

`bench --print-path --json` calls `flagfftGetPlanDescription` after plan
creation and returns the plan node tree and compiled kernel details in
`plan_description`:

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

Pytest invokes `flagfft-cli --json` for all native planner/API/correctness,
benchmark, and tune coverage:

```sh
cmake -S . -B build -GNinja -DFLAGFFT_BUILD_CLI=ON
cmake --build build --target flagfft-cli
pytest
```

Python codegen tests remain under `tests/python/`. No standalone benchmark,
tuner, or gtest native entrypoint is built.

## License

FlagFFT is licensed under the [Apache 2.0 license](./LICENSE).
