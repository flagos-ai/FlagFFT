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
Rank>1 requests still return `FLAGFFT_NOT_SUPPORTED`. `flagfftPlanMany`
accepts the contiguous rank-1 layouts produced by `flagfftPlan1d`, plus the
cuFFT-compatible padded real in-place layout described below; other custom
stride, distance, or embed layouts return `FLAGFFT_NOT_SUPPORTED`.

`flagfftGetPlanDescription(plan)` returns a human-readable string describing
the plan node tree, kernel names, module paths, and compilation details.
Useful for understanding which execution path was selected and for performance
debugging. The returned pointer is valid for the lifetime of the plan.

## C API Usage

Callers provide device buffers and may attach a CUDA stream through the opaque
stream type. Complex forward and inverse transforms are unnormalized, matching
cuFFT: applying forward followed by inverse produces `n * input`.

```cpp
#include <cuda_runtime_api.h>
#include <flagfft.h>

int main() {
  constexpr int n = 256;
  constexpr int batch = 4;
  flagfftComplex* d_input = nullptr;
  flagfftComplex* d_output = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&d_input), n * batch * sizeof(flagfftComplex));
  cudaMalloc(reinterpret_cast<void**>(&d_output), n * batch * sizeof(flagfftComplex));

  flagfftHandle plan = nullptr;
  cudaStream_t stream = nullptr;
  cudaStreamCreate(&stream);

  flagfftResult status = flagfftPlan1d(&plan, n, FLAGFFT_C2C, batch);
  if (status == FLAGFFT_SUCCESS) {
    status = flagfftSetStream(plan, stream);
  }
  if (status == FLAGFFT_SUCCESS) {
    status = flagfftExecC2C(plan, d_input, d_output, FLAGFFT_FORWARD);
    cudaStreamSynchronize(stream);
  }
  if (plan != nullptr) {
    flagfftDestroy(plan);
  }

  cudaStreamDestroy(stream);
  cudaFree(d_output);
  cudaFree(d_input);
  return status == FLAGFFT_SUCCESS ? 0 : 1;
}
```

For an in-place rank-1 real forward transform, allocate
`2 * (n / 2 + 1)` real scalars per batch and describe the padded input row
with `flagfftPlanMany`:

```cpp
int dims[1] = {n};
int padded[1] = {2 * (n / 2 + 1)};
int compact[1] = {n / 2 + 1};
flagfftHandle plan = nullptr;
flagfftPlanMany(&plan, 1, dims, padded, 1, padded[0], compact, 1,
                compact[0], FLAGFFT_R2C, batch);
flagfftExecR2C(plan, d_real_in_place,
               reinterpret_cast<flagfftComplex*>(d_real_in_place));
```

Use the reversed compact/padded distances with `FLAGFFT_C2R` for an in-place
real inverse transform. `flagfftPlan2d` and `flagfftPlan3d` are present for API
compatibility but currently return `FLAGFFT_NOT_SUPPORTED`.

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
description, stream/lifecycle state, compiled raw execution node(s), and device
buffers for twiddle/table/stage data. Complex plans compile both directions;
real plans compile only the applicable forward or inverse operation. Exec calls
do not import Python, compile kernels, rebuild plans, or allocate large buffers.

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
complex-to-real pack. Rank>1 and unsupported custom rank-1 layouts keep
returning `FLAGFFT_NOT_SUPPORTED`.

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
`.flagfft/tuned_plans.sqlite` next to the executable. Set
`FLAGFFT_TUNE_DISABLE=1` to disable tuned-plan lookup and use automatic
planning only.

## Native CLI

`flagfft-cli` is the native executable interface for benchmark measurement:

```sh
./build/flagfft-cli bench --rank 1 --api r2c --shape 4096 --batch 64 \
  --warmup 10 --iters 100 --json
./build/flagfft-cli tune
```

Common case options are `--api c2c|z2z|r2c|d2z|c2r|z2d`,
`--rank 1|2|3`, `--shape N|NxM|NxMxK` (comma-separated for multiple cases),
`--batch`, `--direction forward|inverse`, and `--placement
out-of-place|in-place`. `--print-path` adds the plan description. `tune` is
currently a placeholder and exits with an unsupported status.
Real-to-complex APIs (`r2c`, `d2z`) only accept `forward`; complex-to-real
APIs (`c2r`, `z2d`) only accept `inverse`.
The cuFFT use in this CLI is a CUDA-only correctness and performance oracle;
the FlagFFT library API and its stream handle do not expose CUDA types.
Integer option tokens must be fully numeric; for example, `--shape 16suffix`
and `--batch 2suffix` are rejected as invalid arguments.

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

`test --suite correctness --print-path --json` and `bench --print-path --json`
call `flagfftGetPlanDescription` after plan creation and return the plan node
tree and compiled kernel details in `plan_description`:

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

The repository keeps two canonical runtime validation paths:

1. `tests/ctest/` pytest wrappers execute the Google Test binaries under
   `ctest/`, which then call the FlagFFT C API and compare against the backend
   reference implementation.
2. `benchmark/` pytest cases execute `flagfft-cli bench`, collect per-case
   FlagFFT/reference timings, and print speedup ratios.

Python codegen tests remain under `tests/python/` and exercise the
`flagfft_codegen` package. Lightweight CLI contract tests live under
`tests/cli/`; they are not the performance benchmark entrypoint.

Build the native test and benchmark targets before running the wrappers:

```sh
python3 -m pip install .
cmake -S . -B build -GNinja -DBACKEND=CUDA \
  -DFLAGFFT_BUILD_TESTS=ON -DFLAGFFT_BUILD_CLI=ON
cmake --build build

# Correctness chain: pytest -> Google Test executable -> FlagFFT
pytest tests/ctest/ -v --ctest-build-dir build/ctest

# Performance chain: pytest -> flagfft-cli bench -> FlagFFT/reference timing
pytest benchmark/test_bench.py -v --bench-suite=smoke \
  --flagfft-cli ./build/flagfft-cli
# Add -p no:xdist if pytest-xdist is installed
```

The benchmark report includes `FlagFFT (ms)`, `Reference (ms)`, per-case
`Speedup`, and an overall geometric-mean speedup. `speedup` is computed as
`ref_median_ms / flagfft_median_ms`, so values above `1.00x` mean FlagFFT is
faster than the reference for that case.

## C++ Tests

The C++ test suite lives under `ctest/` and uses Google Test with a unified
`flagfft::test_adaptor` interface to support multiple GPU platforms from a
single set of test sources.

```sh
cmake -S . -B build -GNinja -DBACKEND=CUDA -DFLAGFFT_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --verbose                              # full suite
FLAGFFT_TEST_PROFILE=smoke ctest --test-dir build --verbose  # quick validation
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
- **Reference FFT interface** — `ref_plan_1d/2d/3d`, `ref_set_stream`,
  `ref_exec_c2c/z2z/r2c/d2z/c2r/z2d` mirroring the public FlagFFT API. The
  benchmark CLI binds FlagFFT and reference plans to the same `adaptor::Stream`
  before timing so CUDA events enclose the measured kernels.
- **Device memory** — `adaptor::Memory` (RAII device allocation with
  `copy_from_host`/`copy_to_host`), `adaptor::Stream`, `adaptor::EventTimer`
- **Comparison helpers** — `compute_error` returns `{max_abs, rms}` for
  `float*`, `double*`, `flagfftComplex*`, and `flagfftDoubleComplex*`
- **Random input generation** — `random_complex`, `random_real` for test data

FlagFFT C API convenience wrappers (`Plan1d`, `ExecC2C`, etc.) remain in
`ctest/flagfft_test.h` and assert `FLAGFFT_SUCCESS` via Google Test macros.
The C++ tests additionally provide per-batch normwise `rel_l2`/`rel_linf`
acceptance checks and stable deterministic inputs in `ctest/flagfft_test.h`.

### Backends

| Backend | Reference | Behaviour |
|---|---|---|
| `BACKEND=CUDA` | cuFFT | Plan lifecycle + same-precision normwise comparison against cuFFT for all transform types |

Adding a new GPU platform requires implementing the functions declared in
`src/adaptor/test_adaptor.h` under `src/adaptor/backend/<name>/test_adaptor.cpp`;
no test source changes are needed.

### Test files

| File pattern | Coverage |
|------|----------|
| `test_plan.cpp` | 1D plan lifecycle/error codes and current unsupported 2D/3D contract |
| `test_exec_c2c_{fwd,inv}_{ct,bs}_{s,b}.cpp` | `FLAGFFT_C2C` forward/inverse, Cooley-Tukey/Bluestein routes, single/multi-batch |
| `test_exec_z2z_{fwd,inv}_{ct,bs}_{s,b}.cpp` | `FLAGFFT_Z2Z` double-precision complex coverage |
| `test_exec_r2c_{ct,bs}_{s,b}.cpp`, `test_exec_c2r_{ct,bs}_{s,b}.cpp` | Float real forward/inverse reference comparison |
| `test_exec_d2z_{ct,bs}_{s,b}.cpp`, `test_exec_z2d_{ct,bs}_{s,b}.cpp` | Double real forward/inverse reference comparison |
| `test_exec_r2c_c2r_{ct,bs}_{s,b}.cpp`, `test_exec_d2z_z2d_{ct,bs}_{s,b}.cpp` | Real-transform roundtrip validation |

### Numerical Acceptance

The reference gate is output alignment with cuFFT at the same precision.
Each batch is evaluated independently and the maximum batch statistic is
checked:

```text
rel_l2   = ||flagfft - cufft||_2   / ||cufft||_2
rel_linf = ||flagfft - cufft||_inf / ||cufft||_inf
```

Float and double use the same formulas and transform-class constants, scaled
by unit roundoff and a documented length-based work factor. Set
`FLAGFFT_TEST_REPORT_ACCURACY=1` when executing a test binary to print the
normalized statistics used to re-characterize those constants. Reference
cases use deterministic inputs at scales `2^-20`, `1`, and `2^20`.
Pointwise relative error with a denominator floor is diagnostic output only;
it is not the pass/fail metric.

CTest registers one process per test executable so the process can reuse its
JIT initialization across parameter cases. `FLAGFFT_TEST_PROFILE=smoke`
applies a runtime Google Test filter; Extended cases remain compiled into the
same binaries.

## Python Benchmark Suite

The `benchmark/` directory provides parametrized pytest-based performance
benchmarking that invokes `flagfft-cli bench` and records timing and speedup
against the reference implementation.

Pytest loads the shared benchmark fixtures via `pyproject.toml`:
`addopts = "-p benchmark.utils.pytest_plugin"`. Keep plugin registration out of
subdirectory `conftest.py` files; newer pytest versions reject that pattern, and
early manual imports can trigger assertion-rewrite warnings.

### Quick start

```sh
# Build the CLI first
cmake -S . -B build -GNinja -DBACKEND=CUDA -DFLAGFFT_BUILD_CLI=ON
cmake --build build --target flagfft-cli

# Smoke benchmark suite
pytest benchmark/test_bench.py -v --bench-suite=smoke \
  --flagfft-cli ./build/flagfft-cli
# Add -p no:xdist if pytest-xdist is installed
```

### Test sizes

| Suite | Sizes | Count |
|-------|-------|-------|
| Smoke | 16, 256, 2048 | 3 |
| Typical | All sizes at batch 1, plus selected multi-batch cases | 13 sizes |
| Full | 16, 23, 64, 81, 243, 256, 361, 512, 997, 2048, 4096, 8192, 16384 | 13 sizes |

The full suite covers powers of two (`16`–`16384`), primes (`23`, `997`),
composite non-powers-of-two (`81`, `243`, `361`), and mixed factors.

Each suite expands all six APIs with their valid directions: `c2c` and `z2z`
use both `forward` and `inverse`; `r2c` and `d2z` use `forward`; `c2r` and
`z2d` use `inverse`. Quick CLI verification asserts:
- CLI exit code 0
- `timing.flagfft_median_ms > 0`
- `timing.ref_median_ms > 0`
- `timing.speedup` is present

### CLI options

Customise warmup, iterations, and CLI path via pytest flags:

```sh
pytest benchmark/test_bench.py -v --bench-suite=smoke \
  --bench-warmup 20 --bench-iters 50

pytest benchmark/test_bench.py -v \
  --flagfft-cli ./build/flagfft-cli
```

| Option | Default | Description |
|--------|---------|-------------|
| `--flagfft-cli` | `build/flagfft-cli` or `$FLAGFFT_CLI_EXE` | Path to the CLI binary |
| `--bench-warmup` | 10 | Warmup iterations before timing |
| `--bench-iters` | 100 | Timed benchmark iterations |
| `--bench-suite` | `typical` | Suite: `smoke`, `typical`, or `full` |
| `--bench-csv` | auto path | CSV output path; empty string disables CSV |

If no CUDA device is available, tests are skipped automatically.

### Fixtures

The shared pytest plugin exposes these fixtures for custom benchmark scripts:

| Fixture | Scope | Description |
|---------|-------|-------------|
| `flagfft_cli` | session | Resolved path to `flagfft-cli` |
| `invoke_cli` | function | Call `flagfft-cli` with args, returns `(result, report)` |
| `run_benchmark` | function | Shortcut: `run_benchmark(size, api, direction)` with all bench options preset |
| `bench_warmup` | session | `--bench-warmup` value |
| `bench_iters` | session | `--bench-iters` value |
| `bench_suite` | session | `--bench-suite` value |
| `bench_csv` | session | `--bench-csv` value |

### Report generation

`benchmark.utils.report` produces console tables, CSV, Markdown, and
pretty-printed JSON from aggregated benchmark results:

```python
from benchmark.utils.report import generate_csv, generate_markdown, generate_json_report

results = {"cases": [...]}   # collected CLI JSON outputs
print(generate_markdown(results))
print(generate_csv(results["cases"]))
print(generate_json_report(results))
```

The Markdown report includes a per-case table (size, API, direction, FlagFFT
ms, reference ms, speedup, and optional correctness) and summary statistics.
CSV uses `ref_median_ms` for the reference timing column.

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

The GitHub Actions linter workflow runs the same `pre-commit run --all-files`
checks for pushes and pull requests targeting `main`. It can also be launched
manually from `.github/workflows/linter.yml`.

After `pre-commit install`, every `git commit` will automatically run the
configured hooks. Rejected commits must be re-staged and re-committed after the
hooks fix the files in-place.

## License

FlagFFT is licensed under the [Apache 2.0 license](./LICENSE).
