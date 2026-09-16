# FlagFFT

FlagFFT is a JIT-compiled GPU FFT library. It generates backend-targeted GPU
kernels at runtime via [Triton/TLE](https://github.com/FlagTree/flagtree) and
[libtriton_jit](https://github.com/Artlesbol/libtriton_jit), targeting
arbitrary-length transforms that vendor FFT libraries may not optimally
support. The current CMake build supports CUDA, MUSA, and PPU backends.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Dependencies](#dependencies)
- [Building](#building)
- [API](#api)
- [CLI Tool](#cli-tool)
- [Testing](#testing)
- [License](#license)

---

## Quick Start

Build the library, install the Python codegen package, and run the full test
suite:

```bash
# 1. Clone
git clone https://github.com/flagos-ai/FlagFFT.git
cd FlagFFT

# 2. Initialize submodule
git submodule update --init --recursive

# 3. Build the library, CLI, and test binaries
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DFLAGFFT_BUILD_CLI=ON \
      -DFLAGFFT_BUILD_TESTS=ON
cmake --build build -j$(nproc)

# 4. Install the Python codegen package (required for JIT kernel generation)
pip install .

# 5. Run the full accuracy + performance test suite
python tools/run_tests.py --combination full --gpus 0
```

The runner prints a live progress table and writes `summary.json` with
per-operator accuracy (pass/fail) and performance (geometric mean speedup vs
the selected backend's reference FFT library) results.

### Docker

A CUDA development environment with the default dependencies is available:

```bash
docker build -t flagfft-dev -f docker/Dockerfile .
docker run --gpus all -v $(pwd):/workspace/FlagFFT-dev -it flagfft-dev
# Inside the container, run steps 3-5 from above.
```

The Docker image and CI configuration use Python 3.12. MUSA and PPU builds
require their corresponding vendor SDK environment and should be configured
with `-DBACKEND=MUSA` or `-DBACKEND=PPU`.

---

## Dependencies

### Required

| Dependency | Minimum Version | Notes |
|---|---|---|
| CMake | 3.18 | Build system |
| C++ compiler | C++20 support | GCC 11+, Clang 14+ |
| Python | 3.10 | JIT codegen + test runner; the provided CUDA Docker/CI environments use 3.12 |
| flagtree | 0.5.0 | triton TLE support |
| SQLite3 | — | Tuning database |
| Backend SDK | — | CUDA Toolkit for CUDA, MUSA SDK for MUSA, or PPU SDK for PPU |
| libtriton_jit | submodule | Triton JIT compiler (`deps/libtriton_jit`) |
| PyYAML | — | Test runner (`pip install pyyaml`) |

### Optional

| Dependency | Purpose |
|---|---|
| Google Test | C++ unit tests (auto-fetched via FetchContent when `FLAGFFT_BUILD_TESTS=ON`) |
| Ninja | Faster build backend (`cmake -G Ninja`) |
| pytest | Python codegen tests |

### Submodule

Initialize the required submodule before building:

```bash
git submodule update --init --recursive
```

This pulls in `deps/libtriton_jit`, which provides the Triton JIT compiler and
`nlohmann_json`.

---

## Building

### Basic Build (library only)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

This produces `build/libflagfft.so`.

### Build Options

| Option | Default | Description |
|---|---|---|
| `FLAGFFT_BUILD_CLI` | `OFF` | Build the `flagfft-cli` benchmark/verification tool |
| `FLAGFFT_BUILD_TESTS` | `OFF` | Build the C++ test suite (requires Google Test + the selected backend's reference FFT library) |
| `BACKEND` | `CUDA` | GPU backend selector: `CUDA`, `MUSA`, or `PPU` |
| `CMAKE_BUILD_TYPE` | — | `Release`, `Debug`, `RelWithDebInfo` |

### Full Build (library + CLI + tests)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DFLAGFFT_BUILD_CLI=ON \
      -DFLAGFFT_BUILD_TESTS=ON
cmake --build build -j$(nproc)
```

The default backend is CUDA. Select another supported backend at configure
time, for example `-DBACKEND=MUSA` or `-DBACKEND=PPU`; the corresponding SDK
and runtime libraries must be installed.

### Environment Variables

| Variable | Description |
|---|---|
| `FLAGFFT_PYTHON` | Path to the Python interpreter used by JIT codegen (default: `python3` from PATH); keep its Python minor version aligned with the CMake build interpreter |
| `FLAGFFT_TUNE_DB` | Path to the SQLite tuning database (default: `~/.flagfft/tune.db`) |
| `FLAGFFT_TUNE_DISABLE` | Set to `1` to disable tuned plan lookup and always use auto-selected plans |

### Install

```bash
cmake --install build --prefix /usr/local
```

Installs `libflagfft.so`, the public header (`flagfft.h`), and `flagfft-cli` (if
built).

---

## API

FlagFFT exposes a cuFFT-compatible C API in `include/flagfft.h`.

### Plan Creation

```c
flagfftPlan1d(plan, nx, type, batch)
flagfftPlan2d(plan, nx, ny, type)
flagfftPlan3d(plan, nx, ny, nz, type)        // contiguous row-major rank-3 plan
flagfftPlanMany(plan, rank, n, inembed, istride, idist,
                onembed, ostride, odist, type, batch)
```

### Execution

```c
// Complex-to-Complex (single & double precision)
flagfftExecC2C(plan, idata, odata, direction)
flagfftExecZ2Z(plan, idata, odata, direction)

// Real-to-Complex (forward)
flagfftExecR2C(plan, idata, odata)
flagfftExecD2Z(plan, idata, odata)

// Complex-to-Real (inverse)
flagfftExecC2R(plan, idata, odata)
flagfftExecZ2D(plan, idata, odata)
```

### Management

```c
flagfftSetStream(plan, stream)    // Attach a backend stream
flagfftDestroy(plan)              // Free plan resources
flagfftGetPlanDescription(plan)   // Human-readable plan summary
```

### Data Types

| FlagFFT Type | C Type | Description |
|---|---|---|
| `flagfftComplex` | `float2` | Single-precision complex |
| `flagfftDoubleComplex` | `double2` | Double-precision complex |
| `flagfftReal` | `float` | Single-precision real |
| `flagfftDoubleReal` | `double` | Double-precision real |

### Transform Types

| Type Constant | Transform |
|---|---|
| `FLAGFFT_C2C` | Complex → Complex |
| `FLAGFFT_Z2Z` | Double Complex → Double Complex |
| `FLAGFFT_R2C` | Real → Complex |
| `FLAGFFT_D2Z` | Double Real → Double Complex |
| `FLAGFFT_C2R` | Complex → Real |
| `FLAGFFT_Z2D` | Double Complex → Double Real |

`flagfftPlan3d` supports contiguous row-major rank-3 plans for all six
transform types. The corresponding contiguous rank-3 forms are also available
through `flagfftPlanMany`; arbitrary custom strides and layouts beyond the
supported forms remain unsupported.

### Currently Supported

| Feature | Status |
|---|---|
| Rank-1 arbitrary-length C2C, Z2Z | ✅ Cooley-Tukey + Bluestein/Rader |
| Rank-1 arbitrary-length R2C, D2Z (forward) | ✅ |
| Rank-1 arbitrary-length C2R, Z2D (inverse) | ✅ |
| Rank-1 roundtrip (R2C→C2R, D2Z→Z2D) | ✅ |
| Rank-2 contiguous row-major C2C, Z2Z | ✅ RTRT decomposition |
| Rank-2 contiguous row-major R2C, D2Z, C2R, Z2D | ✅ |
| Rank-3 contiguous row-major C2C, Z2Z | ✅ RTRT decomposition (n2 → n1 → n0 + 3D axis permutations) |
| Rank-3 contiguous row-major R2C, D2Z, C2R, Z2D | ✅ half-packed on the innermost axis |
| Batched transforms | ✅ |
| In-place and out-of-place | ✅ |
| Backend adaptors | ✅ CUDA, MUSA, PPU (selected at build time) |
| Backend stream attachment | ✅ |

For the CUDA backend, `2^20` rank-1 transforms on `sm_80` select a `1024 x 1024`
Four-Step decomposition. The TLE kernels apply contiguous, asynchronously
loaded twiddles in the row pass, pack two adjacent row FFTs and four adjacent
column FFTs per block for single precision, and XOR-swizzle shared-memory
indices to reduce bank conflicts.

### Planned / Not Yet Supported

| Feature | Status |
|---|---|
| Rank-2 more exec algos | RTRT only now |

---

## CLI Tool

`flagfft-cli` is a native benchmark and verification tool. Build it with
`-DFLAGFFT_BUILD_CLI=ON`.

### Subcommands

#### `bench` — Benchmark FFT performance

```bash
flagfft-cli bench [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--rank` | `1` | Transform rank: `1`, `2`, or `3` |
| `--api` | `c2c` | Transform type: `c2c`, `z2z`, `r2c`, `d2z`, `c2r`, `z2d` |
| `--shape` | required | Transform size(s), comma-separated: `1024`, `256x256`, `16x16x16`, `1024,2048,4096` |
| `--batch` | `1` | Batch size |
| `--direction` | `forward` | `forward` or `inverse` |
| `--placement` | `out-of-place` | `out-of-place` or `in-place` |
| `--warmup` | `10` | Warmup iterations |
| `--iters` | `100` | Measurement iterations |
| `--json` | — | Output results as JSON |
| `--print-path` | — | Print the execution plan decomposition path (use with `--json`) |

**Examples:**

```bash
# Benchmark 1D C2C FFT of size 4096, batch 256
flagfft-cli bench --api c2c --shape 4096 --batch 256

# Benchmark 2D Z2Z FFT
flagfft-cli bench --rank 2 --api z2z --shape 256x256

# Benchmark 3D C2C FFT
flagfft-cli bench --rank 3 --api c2c --shape 16x16x16

# Compare multiple sizes with JSON output
flagfft-cli bench --api r2c --shape 1024,2048,4096,8192 --json

# Print the kernel execution plan
flagfft-cli bench --api c2c --shape 997 --print-path --json
```

#### `tune` — Auto-tuning (planned)

```bash
flagfft-cli tune [OPTIONS]
```

Currently a placeholder; exits with `FLAGFFT_NOT_SUPPORTED`.

### Exit Codes

| Code | Meaning |
|---|---|
| `0` | Passed |
| `1` | Failed / invalid arguments |
| `2` | Runtime error |
| `77` | Skipped / unsupported |

---

## Testing

FlagFFT has three layers of testing: a unified Python test runner, C++ unit
tests (Google Test), and Python codegen tests (pytest).

### Unified Test Runner

`tools/run_tests.py` is the single entry point for the 36 acceptance operators.
Correctness compares FlagFFT and the platform FFT library independently against
a float64/complex128 NumPy reference. An operator passes correctness only when
all selected FlagFFT-vs-NumPy cases pass; platform-library failures are reported
under `platform_accuracy` and do not fail FlagFFT correctness. Performance
continues to use `flagfft-cli bench`.

Install test dependencies with `pip install -e '.[test]'` and build with
`-DFLAGFFT_BUILD_TESTS=ON -DFLAGFFT_BUILD_CLI=ON`. The test build includes
`build/ctest/numpy_fft_capture`; no separate validation build is required.

`conf/operators.yaml` defines exactly 36 operators: six APIs
(C2C, C2R, R2C, Z2Z, Z2D, D2Z) for each of the following groups, in this order:

| Group | Operator ID example |
|---|---|
| 1D Cooley-Tukey single | `1d_ct_single_c2c` |
| 1D Prime single | `1d_prime_single_c2c` |
| 1D Cooley-Tukey batch | `1d_ct_batch_c2c` |
| 1D Prime batch | `1d_prime_batch_c2c` |
| 2D | `2d_c2c` |
| 3D | `3d_c2c` |

`batch: single/batch` in an operator is a category, not a numeric batch
count. `conf/test_matrix.yaml` owns size sets, numeric batches and scales.
The matrix controls the case count and numeric batch sizes; inspect the current
expansion with `--dry-run`. Single and current 3D cases require
batch 1. CT/Prime are acceptance size categories; the recorded runtime plan
shows the actual selected algorithm, including DirectDFT, Rader or Bluestein.

Complex APIs test both directions; real-to-complex APIs test forward and
complex-to-real APIs test inverse. Real-inverse inputs have valid
multidimensional Hermitian half spectra. NumPy inverse results are multiplied
by the transform size to match the unnormalized device APIs. Both comparisons
use the existing size/precision-aware worst-batch `rel_l2` and `rel_linf`
limits and reject nonfinite values.

#### Usage

| Flag | Default | Description |
|---|---|---|
| `--ops` | All 36 | Comma-separated acceptance operator IDs |
| `--op-list-file` | — | One operator ID per line; `#` starts a comment |
| `--start` | — | Start at this operator in configuration order |
| `--combination` | `full` | Group filter: the six groups above; comma-separated, or `full/all` alone |
| `--gpus` | `0` | Comma-separated device IDs or `all` |
| `--build-dir` | `build` beside the runner | CMake build directory |
| `--capture-bin` | `<build-dir>/ctest/numpy_fft_capture` | Optional native capture override |
| `--output-dir` | Workspace `results/<timestamp>_acceptance36` | Fresh result directory |
| `--incremental-csv` | `<output-dir>/incremental.csv` | One flushed row per completed case/phase |
| `--accuracy-only` / `--performance-only` | Both phases | Mutually exclusive phase filters |
| `--scales` | Matrix scales, or `[1.0]` if omitted | Comma-separated positive input amplitudes, or `all` for `2^-20,1,2^20` |
| `--shapes` | All configured sizes | Exact shapes, e.g. `256,64x64` |
| `--max-cases` | — | Select the first N cases for a partial run |
| `--timeout` | `600` | Independent timeout for each FlagFFT/platform/benchmark process |
| `--warmup` / `--iters` | `10 / 100` | Benchmark warmup and measurement iterations |
| `--dry-run` | — | Print selected cases without execution or result files |
| `--analyze-only RESULT_DIR` | — | Recompute NumPy comparisons from captured inputs and outputs |
| `--color` | `auto` | `auto/always/never` |

The old 18 API/rank IDs are replaced by the acceptance IDs above. The old
`1d_bs_single/batch` group filters are accepted as aliases for
`1d_prime_single/batch`; 2D is now one group containing both CT and Prime
sizes. Stages are no longer configured or filtered.

```bash
# Full acceptance suite; results are outside the repository/worktree.
python tools/run_tests.py --gpus 0

# Inspect the entire 36-operator execution matrix without using a GPU.
python tools/run_tests.py --dry-run

# NumPy correctness for selected operators.
python tools/run_tests.py --accuracy-only \
  --ops 1d_ct_single_c2c,1d_prime_batch_z2d

# Prime batch group.
python tools/run_tests.py --combination 1d_prime_batch --gpus 0

# All three correctness scales. Benchmark runs once per shape/batch/direction.
python tools/run_tests.py --scales all --ops 1d_ct_single_c2c

# Reanalyze saved data without executing device kernels.
python tools/run_tests.py --analyze-only ../results/20260916_120000_acceptance36
```

#### Output

All outputs use format version 2. A full run has 36 keys under
`summary.json.result`, in acceptance order. Filtered runs contain the
selected operators, and the manifest records the exact partial selection.

- `manifest.json`: selected operators, complete expected case lists, parameter matrix and runtime environment.
- `summary.json`: each operator's `accuracy`, `platform_accuracy` and `performance` results.
- `incremental.csv`: case/phase, operator ID, shape, numeric batch, direction, scale, both correctness statuses and errors, limits, timings and actual plan. CSV quoting preserves multiline plan text.
- `{op_id}/{case_id}/case.json`: correctness metrics, input seed/hashes, independent capture statuses and actual `accuracy.plan`.
- `{op_id}/{case_id}/`: exact inputs, NumPy and device outputs, per-library logs and `flagfft_plan.txt`.
- `{op_id}/performance/{case_id}/result.json`: timings and the benchmark's own `performance.plan`.
- `{op_id}/{accuracy,platform_accuracy,performance}_result.json`: aggregate per-operator results.
- `reanalyzed.csv`: refreshed comparisons produced by `--analyze-only`.

Correctness and performance have separate case IDs; scale is omitted from
performance IDs. A missing result cannot make an operator pass. Numeric
failures serialize nonfinite metrics as JSON null while retaining failure
status. A plan is saved as soon as its creation succeeds and refreshed after
successful execution; creation failures have `plan: null`.

Performance rows retain the raw speedup and record `baseline_valid`.
An incorrect platform baseline is excluded from the aggregate speedup
statistics; operators failing FlagFFT correctness are excluded as well.
Performance-only runs have an unknown baseline validity.

Exit code is 0 when all requested FlagFFT correctness and/or performance phases
pass, 1 for failed/incomplete/skipped requested phases, 2 for configuration
errors, and 130 for interruption. Platform correctness is reported independently.
Results are preserved on interruption; reanalysis uses the original manifest.

### C++ Tests (ctest/)

Built with `-DFLAGFFT_BUILD_TESTS=ON`. Each test binary compares FlagFFT
output against the selected backend's reference FFT library using normwise
relative error metrics (`rel_l2`, `rel_linf`).

#### Structure

| Test Pattern | Coverage |
|---|---|
| `test_plan` | Plan lifecycle, error codes, unsupported API contracts |
| `test_2d_correctness` | Rank-2 all-API correctness and complex/real roundtrips |
| `test_3d_correctness` | Rank-3 all-API correctness and complex/real roundtrips |
| `test_exec_c2c_{fwd,inv}_{ct,bs}_{s,b}` | C2C forward/inverse, Cooley-Tukey/Bluestein, single/multi-batch |
| `test_exec_z2z_{fwd,inv}_{ct,bs}_{s,b}` | Double-precision complex |
| `test_exec_r2c_{ct,bs}_{s,b}` | Float real → complex |
| `test_exec_d2z_{ct,bs}_{s,b}` | Double real → complex |
| `test_exec_c2r_{ct,bs}_{s,b}` | Complex → float real |
| `test_exec_z2d_{ct,bs}_{s,b}` | Double complex → double real |
| `test_exec_r2c_c2r_{ct,bs}_{s,b}` | Real roundtrip validation |
| `test_exec_d2z_z2d_{ct,bs}_{s,b}` | Double real roundtrip |

Suffix key: `s` = single-batch, `b` = multi-batch; `ct` = Cooley-Tukey, `bs`
= Bluestein/Rader.

#### Running Individual Tests

```bash
# Run a specific test
./build/ctest/test_exec_c2c_fwd_ct_s

# With custom parameters
./build/ctest/test_exec_c2c_fwd_ct_s --nx 4096 --batch 64 --direction forward

# Run all ctest tests
cd build && ctest --output-on-failure
```

Test binaries accept the applicable subset of: `--nx`, `--ny`, `--nz`,
`--batch`, `--direction`, `--api`, `--scale`, `--json-file`.

### Python Tests

Tests for the `flagfft_codegen` Python package. Requires the package installed
(`pip install .`).

```bash
# Run all Python tests
pytest tests/python/ -v

# Run only codegen-marked tests
pytest tests/python/ -v -m codegen
```

Tests cover codelet structure, kernel source generation, JIT CSV parsing, and
Bluestein/reshape/R2C metadata. Tests that require Triton/TLE are
automatically skipped when dependencies are unavailable.

### Test Configuration

The test parameter space is defined in `conf/`:

- `conf/operators.yaml` — 18 operator definitions (1D/2D/3D × C2C/Z2Z/R2C/D2Z/C2R/Z2D)
- `conf/test_matrix.yaml` — Parameter space for 1D CT/BS and 2D/3D sizes, with 7 combination rules

---

## License

Apache License, Version 2.0. See [LICENSE](LICENSE) for the full text.
