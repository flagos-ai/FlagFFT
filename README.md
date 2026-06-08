# FlagFFT

FlagFFT is an experimental C++ FFT library with a cuFFT-style API and
Triton/TLE-generated CUDA kernels. The public runtime interface is C; Python
is retained only for Triton/TLE JIT source generation (internal codegen).

## Quick Start

Build and run the native CLI in an environment with CMake, Ninja, CUDA,
Python, PyTorch, and the Triton/TLE dependencies available:

```sh
python3 -m pip install .
cmake -S . -B build -GNinja -DBACKEND=CUDA -DFLAGFFT_BUILD_CLI=ON
cmake --build build --target flagfft-cli
./build/flagfft-cli bench --rank 1 --api c2c --shape 4096 --batch 64 \
  --warmup 10 --iters 100 --json
```

The repository also provides Dockerfiles for preparing a build environment.
They do not compile FlagFFT into the image; after starting the container, run
the same build commands inside the mounted source tree:

```sh
docker build -f docker/Dockerfile -t flagfft-ubuntu2404:latest .
docker run --rm -it --gpus all \
  -v "$PWD":/workspace/FlagFFT-dev \
  -w /workspace/FlagFFT-dev \
  flagfft-ubuntu2404:latest
```

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
Contiguous row-major rank-2 `FLAGFFT_C2C` and `FLAGFFT_Z2Z` are supported
through `flagfftPlan2d` and batched `flagfftPlanMany`. The current 2D complex
path is an RTRT decomposition: row FFT, tiled transpose, row FFT over the
transposed columns, and tiled transpose back. Rank-2 real transforms, rank 3,
and custom rank-2 stride, distance, or embed layouts return
`FLAGFFT_NOT_SUPPORTED`.

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
real inverse transform. `flagfftPlan2d` currently supports only complex
`C2C`/`Z2Z`; `flagfftPlan3d` is present for API compatibility but currently
returns `FLAGFFT_NOT_SUPPORTED`.

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
Python 3.10+ development files, PyTorch, pybind11, and `pytest`. Runtime JIT
generation and Python tests additionally require a preconfigured
Triton/TLE-enabled Python environment. Install the codegen package into the
Python environment that will execute JIT generation:

```sh
python3 -m pip install .
cmake -S . -B build -GNinja -DBACKEND=CUDA -DFLAGFFT_BUILD_CLI=ON
cmake --build build --target flagfft-cli
cmake --install build --prefix /path/to/flagfft-install
```

`FLAGFFT_BUILD_TESTS=ON` enables the C++ test suite (see [C++ Tests](#c-tests)).

### Docker Build Environment

`docker/Dockerfile` builds a base Ubuntu 24.04 environment with Python 3.12,
PyTorch CUDA wheels, CUDA toolkit 13.2, FlagTree, CMake, Ninja, pybind11, and
`pytest`, plus other native build dependencies. The image only contains the
environment; it does not compile this repository during `docker build`.

```sh
docker build -f docker/Dockerfile -t flagfft-ubuntu2404:latest .
```

If GitHub access needs to be rewritten to a mirror, pass the optional build
argument:

```sh
docker build -f docker/Dockerfile \
  -t flagfft-ubuntu2404:latest \
  --build-arg GIT_URL_REWRITE_TO=https://example.com/github-mirror/ \
  .
```

Run the container with GPU access and mount the source tree, then build
FlagFFT inside the container:

```sh
docker run --rm -it --gpus all \
  -v "$PWD":/workspace/FlagFFT-dev \
  -w /workspace/FlagFFT-dev \
  flagfft-ubuntu2404:latest

python3 -m pip install .
cmake -S . -B build -GNinja -DBACKEND=CUDA -DFLAGFFT_BUILD_CLI=ON
cmake --build build --target flagfft-cli
```

`docker/Dockerfile.dev` is an optional developer shell image based on the base
image. It adds zsh, Oh My Zsh, Node.js, and Codex tooling:

```sh
docker build -f docker/Dockerfile.dev \
  --build-arg BASE_IMAGE=flagfft-ubuntu2404:latest \
  -t flagfft-dev:latest .
```

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
`--batch` is accepted for rank 1 and for rank-2 complex `c2c`/`z2z` cases.
Real-to-complex APIs (`r2c`, `d2z`) only accept `forward`; complex-to-real
APIs (`c2r`, `z2d`) only accept `inverse`.
The cuFFT use in this CLI is a CUDA-only correctness and performance oracle;
the FlagFFT library API and its stream handle do not expose CUDA types.
Integer option tokens must be fully numeric; for example, `--shape 16suffix`
and `--batch 2suffix` are rejected as invalid arguments.

### Capability Matrix

| Command | Supported now | Reported `unsupported` |
|---|---|---|
| `test correctness`, `bench` | Six 1D APIs with `plan1d`, both complex directions, valid real direction, in/out-of-place; padded real in-place `planmany`; contiguous rank-2 `c2c`/`z2z` including batch | Rank-2 real APIs, rank 3, and other `planmany` layouts |
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

## Testing

FlagFFT uses `tools/run_tests.py` for both accuracy and performance testing.
Python codegen unit tests remain under `tests/python/` and are run with
`pytest tests/python/`.

### Prerequisites

```sh
# Build with tests and CLI
cmake -B build -DCMAKE_BUILD_TYPE=Release -DFLAGFFT_BUILD_TESTS=ON -DFLAGFFT_BUILD_CLI=ON
cmake --build build -j$(nproc)

# Install Python dependencies
pip install pyyaml
```

### Quick Start

```sh
# CT smoke test (single GPU)
python tools/run_tests.py --combination ct

# Specific operators
python tools/run_tests.py --ops c2c_1d,r2c_1d --combination ct

# Full test matrix
python tools/run_tests.py --combination full

# Multi-GPU
python tools/run_tests.py --combination ct --gpus 0,1,2,3

# Accuracy only
python tools/run_tests.py --combination ct --accuracy-only

# Performance only
python tools/run_tests.py --combination ct --performance-only
```

### Configuration

- `conf/operators.yaml` — Operator definitions (transform type x dimension)
- `conf/test_matrix.yaml` — Test parameter matrix (sizes, batches, scales)

### Output

- Console: Real-time progress with per-GPU status
- JSON: Summary file (`summary.json`) with per-operator results

## C++ Tests

The C++ test suite lives under `ctest/` and uses Google Test with a unified
`flagfft::test_adaptor` interface to support multiple GPU platforms from a
single set of test sources. Tests are parametrized via command-line arguments
(`--nx`, `--batch`, `--direction`, `--scale`, `--json-file`) and driven by
`tools/run_tests.py`.

```sh
cmake -S . -B build -GNinja -DBACKEND=CUDA -DFLAGFFT_BUILD_TESTS=ON -DFLAGFFT_BUILD_CLI=ON
cmake --build build -j$(nproc)

# Run via the unified test runner (recommended)
python tools/run_tests.py --combination ct

# Or run individual test binaries directly
./build/ctest/test_exec_c2c_fwd_ct_s --nx 256 --batch 64 --direction forward --scale 1 --json-file result.json
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
  `ref_exec_c2c/z2z/r2c/d2z/c2r/z2d` mirroring the public FlagFFT API.
- **Device memory** — `adaptor::Memory` (RAII device allocation with
  `copy_from_host`/`copy_to_host`), `adaptor::Stream`, `adaptor::EventTimer`
- **Comparison helpers** — `compute_error` returns `{max_abs, rms}` for
  `float*`, `double*`, `flagfftComplex*`, and `flagfftDoubleComplex*`
- **Random input generation** — `random_complex`, `random_real` for test data

`ctest/flagfft_test.h` defines `TestParams` (nx, batch, direction, scale) and
`override_params()` which reads command-line arguments (`--nx`, `--batch`,
`--direction`, `--scale`, `--json-file`) to override defaults.
`JsonTestListener` writes per-test JSON results for collection by the unified
test runner. FlagFFT C API convenience wrappers (`Plan1d`, `ExecC2C`, etc.)
assert `FLAGFFT_SUCCESS` via Google Test macros. The C++ tests additionally
provide per-batch normwise `rel_l2`/`rel_linf` acceptance checks and stable
deterministic inputs.

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
JIT initialization across parameter cases. The unified test runner
(`tools/run_tests.py`) handles test selection, multi-GPU distribution, and
result collection.

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
