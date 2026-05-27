# FlagFFT

FlagFFT is an experimental C++ FFT library with a cuFFT-style API and
Triton/TLE-generated CUDA kernels. The public runtime interface is C; Python
is retained only for Triton/TLE JIT source generation (internal codegen).

## Current API

The public header is `include/flagfft.h` and exposes:

- `flagfftPlan1d`, `flagfftPlan2d`, `flagfftPlan3d`, `flagfftPlanMany`
- `flagfftExecC2C`, `flagfftExecZ2Z`, `flagfftExecR2C`, `flagfftExecD2Z`,
  `flagfftExecC2R`, `flagfftExecZ2D`
- `flagfftSetStream`, `flagfftDestroy`
- `flagfftGetPlanDescription`

`flagfftSetStream` accepts the backend-neutral opaque `flagfftStream_t`
declared in `include/flagfft.h`. With the current CUDA backend, callers may
pass a `cudaStream_t` directly without changing stream behavior.

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
- `src/codegen/`: C++ libtriton_jit invocation/cache logic.
- `python/flagfft_codegen/`: installable Python kernel source generator and
  its codelets.
- `src/adaptor/`: backend abstraction and the current CUDA Driver implementation.
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
CUDA through `BACKEND=CUDA`. The same option selects the FlagFFT adaptor and
the `libtriton_jit` backend; CUDA is the only adaptor backend currently
provided.

```sh
cmake -S . -B build -GNinja -DBACKEND=CUDA
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
Python code generation is distributed as the pure Python `flagfft-codegen`
package in this repository; CMake builds and installs the native library
separately.

The build environment must provide CMake, Ninja, a CUDA toolkit, SQLite3,
Python 3.10+ development files, PyTorch, and pybind11. Runtime JIT generation
and Python tests additionally require a preconfigured Triton/TLE-enabled
Python environment and `pytest`. Install the codegen package into the Python
environment that will execute JIT generation:

```sh
python3 -m pip install .
cmake -S . -B build -GNinja -DBACKEND=CUDA -DFLAGFFT_BUILD_CLI=ON
cmake --build build --target flagfft-cli
cmake --install build --prefix /path/to/flagfft-install
```

`FLAGFFT_BUILD_TESTS=ON` enables the C++ test suite (see [C++ Tests](#c-tests)).

When plan creation emits Triton JIT source, it uses `FLAGFFT_PYTHON` if set,
otherwise `python3`, to run `python -m flagfft_codegen.jit_source`. The
selected interpreter must have `flagfft-codegen` installed and provide the
compatible Triton/TLE environment. Generated source/metadata and tuned-plan
SQLite defaults are stored in `.flagfft` next to the running executable.
Tuned plan lookups read from `FLAGFFT_TUNE_DB` if set, or the default
`.flagfft/tuned_plans.sqlite` next to the executable.

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
backend event `median`/`p90` timings and speedup; `--print-path` adds the plan
description. `tune` benchmarks candidates, verifies finalists against cuFFT,
and writes the winning plan to SQLite; `--retune` supersedes an earlier winner.
The cuFFT use in this CLI is a CUDA-only correctness and performance oracle;
the FlagFFT library API and its stream handle do not expose CUDA types.
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
python3 -m pip install .
cmake -S . -B build -GNinja -DBACKEND=CUDA -DFLAGFFT_BUILD_CLI=ON
cmake --build build --target flagfft-cli
pytest -q
```

Python codegen tests remain under `tests/python/` and exercise the
`flagfft_codegen` package. C++ Google Test coverage is enabled separately with
`FLAGFFT_BUILD_TESTS=ON`; no standalone benchmark or tuner executable is built.

## C++ Tests

The C++ test suite lives under `ctest/` and uses Google Test with a unified
`flagfft::test_adaptor` interface to support multiple GPU platforms from a
single set of test sources.

```sh
cmake -S . -B build -GNinja -DBACKEND=CUDA -DFLAGFFT_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build -R "1D"   # 1D-only (skip pre-existing NOT_SUPPORTED for 2D/3D)
```

Google Test is fetched automatically by `deps/libtriton_jit` via FetchContent;
no additional installation is required.

### Architecture

The test adaptor (`src/adaptor/test_adaptor.h`, backend impl in
`src/adaptor/backend/<name>/test_adaptor.cpp`) is a shared layer used by both
ctest and the CLI bench command. It builds as a CMake OBJECT library
(`flagfft_test_adaptor`) and provides:

- **`RefPlanHandle`** — type-erased RAII wrapper for the platform-specific
  reference FFT plan handle (`cufftHandle` on CUDA). Uses `replace()` to set
  the handle from a locally-constructed backend handle, avoiding
  strict-aliasing UB.
- **Reference FFT interface** — `ref_plan_1d/2d/3d`, `ref_exec_c2c/z2z/r2c/d2z/c2r/z2d`
  mirroring the public FlagFFT API. The benchmark CLI uses `ref_plan_1d` +
  `ref_exec_c2c` as a correctness oracle.
- **Device memory** — `adaptor::Memory` (RAII device allocation with
  `copy_from_host`/`copy_to_host`), `adaptor::Stream`, `adaptor::EventTimer`
- **Comparison helpers** — `compute_error` returns `{max_abs, rms}` for
  `float*`, `double*`, `flagfftComplex*`, and `flagfftDoubleComplex*`
- **Random input generation** — `random_complex`, `random_real` for test data

FlagFFT C API convenience wrappers (`Plan1d`, `ExecC2C`, etc.) remain in
`ctest/flagfft_test.h` and assert `FLAGFFT_SUCCESS` via Google Test macros.

### Backends

| Backend | Reference | Behaviour |
|---|---|---|
| `BACKEND=CUDA` | cuFFT | Plan lifecycle + elementwise comparison against cuFFT for all transform types |

Adding a new GPU platform requires implementing the functions declared in
`src/adaptor/test_adaptor.h` under `src/adaptor/backend/<name>/test_adaptor.cpp`;
no test source changes are needed.

### Test files

| File | Coverage |
|------|----------|
| `test_plan.cpp` | Plan creation/destruction/description, invalid-parameter error codes |
| `test_exec_c2c.cpp` | `FLAGFFT_C2C` forward, inverse, roundtrip, batch, 2D/3D |
| `test_exec_z2z.cpp` | `FLAGFFT_Z2Z` double-precision complex |
| `test_exec_r2c_c2r.cpp` | `FLAGFFT_R2C`/`FLAGFFT_C2R` roundtrip and reference comparison |
| `test_exec_d2z_z2d.cpp` | `FLAGFFT_D2Z`/`FLAGFFT_Z2D` double-precision real |

## Python Benchmark Suite

The `benchmark/` directory provides parametrized pytest-based performance
benchmarking that invokes `flagfft-cli bench` and validates correctness,
timings, and speedup against cuFFT.

### Quick start

```sh
# Build the CLI first
cmake -S . -B build -GNinja -DBACKEND=CUDA -DFLAGFFT_BUILD_CLI=ON
cmake --build build --target flagfft-cli

# Smoke test (2 sizes, fast — verify the pipeline works)
pytest benchmark/test_bench_smoke.py -v

# Full benchmark (13 sizes)
pytest benchmark/test_bench_full.py -v

# Generate Markdown + JSON report (one-shot, skipped by default)
pytest benchmark/test_bench_full.py::test_bench_full_report -v -s
```

### Test sizes

| Suite | Sizes | Count |
|-------|-------|-------|
| Smoke | 256, 512 | 2 |
| Full | 16, 23, 64, 81, 243, 256, 361, 512, 997, 2048, 4096, 8192, 16384 | 13 |

The full suite covers powers of two (`16`–`16384`), primes (`23`, `997`),
composite non-powers-of-two (`81`, `243`, `361`), and mixed factors.

Each size is tested against `c2c` and `z2z` APIs in both `forward` and
`inverse` directions (4 combinations per size). Every case verifies:
- CLI exit code 0
- `correctness.passed == true` (against cuFFT reference)
- `timing.flagfft_median_ms > 0`

### CLI options

Customise warmup, iterations, and CLI path via pytest flags:

```sh
pytest benchmark/test_bench_smoke.py -v \
  --bench-warmup 20 --bench-iters 50 --bench-launches-per-sample 3

pytest benchmark/test_bench_full.py -v \
  --flagfft-cli ./build/flagfft-cli
```

| Option | Default | Description |
|--------|---------|-------------|
| `--flagfft-cli` | `build/flagfft-cli` or `$FLAGFFT_CLI_EXE` | Path to the CLI binary |
| `--bench-warmup` | 5 | Warmup iterations before timing |
| `--bench-iters` | 10 | Timed benchmark iterations |
| `--bench-launches-per-sample` | 1 | Kernel launches per sample |

If no CUDA device is available, tests are skipped automatically.

### Fixtures

`conftest.py` exposes these fixtures for custom benchmark scripts:

| Fixture | Scope | Description |
|---------|-------|-------------|
| `flagfft_cli` | session | Resolved path to `flagfft-cli` |
| `invoke_cli` | function | Call `flagfft-cli` with args, returns `(result, report)` |
| `run_benchmark` | function | Shortcut: `run_benchmark(size, api, direction)` with all bench options preset |
| `bench_warmup` | session | `--bench-warmup` value |
| `bench_iters` | session | `--bench-iters` value |
| `bench_launches_per_sample` | session | `--bench-launches-per-sample` value |

### Report generation

`report.py` produces Markdown tables and pretty-printed JSON from aggregated
benchmark results:

```python
from benchmark.report import generate_markdown, generate_json_report

results = {"cases": [...]}   # collected CLI JSON outputs
print(generate_markdown(results))
print(generate_json_report(results))
```

The Markdown report includes a per-case table (size, API, direction, FlagFFT
ms, cuFFT ms, speedup, correctness) and summary statistics (pass rate, median
range, geometric mean). `test_bench_full_report` demonstrates end-to-end
collection + report writing to `tmp_path`.

## Code Style

This project uses [pre-commit](https://pre-commit.com/) to enforce consistent
formatting and linting. The configuration covers:

- **clang-format** (v13, Google style with custom overrides in `.clang-format`)
- **isort** and **black** for Python
- **flake8** for Python linting
- Generic checks: YAML syntax, trailing whitespace, end-of-file newlines

### Installing clang-format

The pre-commit hook pulls `clang-format` v13 automatically, but if you need it
locally (e.g. for editor integration):

```sh
# Ubuntu / Debian
apt install clang-format-13

# Or via pip (requires a working Python environment)
pip install clang-format==13.0.0
```

### Installing and using pre-commit

```sh
# Install pre-commit itself
pip install pre-commit

# Install the git hook scripts (runs checks on every commit)
pre-commit install

# Run all hooks manually against all files (useful for first setup)
pre-commit run --all-files
```

After `pre-commit install`, every `git commit` will automatically run the
configured hooks. Rejected commits must be re-staged and re-committed after the
hooks fix the files in-place.

## License

FlagFFT is licensed under the [Apache 2.0 license](./LICENSE).
