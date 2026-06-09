# FlagFFT

FlagFFT is a JIT-compiled GPU FFT library. It generates CUDA kernels at runtime
via [Triton/TLE](https://github.com/FlagTree/flagtree) and
[libtriton_jit](https://github.com/Artlesbol/libtriton_jit), targeting
arbitrary-length transforms that cuFFT does not optimally support.

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
# 1. Clone with submodules
git clone --recurse-submodules https://github.com/Artlesbol/FlagFFT-dev.git
cd FlagFFT-dev

# 2. Build the library, CLI, and test binaries
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DFLAGFFT_BUILD_CLI=ON \
      -DFLAGFFT_BUILD_TESTS=ON
cmake --build build -j$(nproc)

# 3. Install the Python codegen package (required for JIT kernel generation)
pip install .

# 4. Run the full accuracy + performance test suite
python tools/run_tests.py --combination full --gpus 0
```

The runner prints a live progress table and writes `summary.json` with
per-operator accuracy (pass/fail) and performance (geometric mean speedup vs
cuFFT) results.

### Docker

A pre-built environment with all dependencies is available:

```bash
docker build -t flagfft-dev -f docker/Dockerfile .
docker run --gpus all -v $(pwd):/workspace/FlagFFT-dev -it flagfft-dev
# Inside the container, run steps 2-4 from above.
```

---

## Dependencies

### Required

| Dependency | Minimum Version | Notes |
|---|---|---|
| CMake | 3.18 | Build system |
| C++ compiler | C++20 support | GCC 11+, Clang 14+ |
| Python | 3.10 | JIT codegen + test runner |
| SQLite3 | — | Tuning database |
| CUDA Toolkit | 12.x | cudart, cuFFT (for test adaptor/benchmarks) |
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
| `FLAGFFT_BUILD_TESTS` | `OFF` | Build the C++ test suite (requires Google Test + CUDA) |
| `BACKEND` | `CUDA` | GPU backend selector (only `CUDA` is currently supported) |
| `CMAKE_BUILD_TYPE` | — | `Release`, `Debug`, `RelWithDebInfo` |

### Full Build (library + CLI + tests)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DFLAGFFT_BUILD_CLI=ON \
      -DFLAGFFT_BUILD_TESTS=ON
cmake --build build -j$(nproc)
```

### Environment Variables

| Variable | Description |
|---|---|
| `FLAGFFT_PYTHON` | Path to the Python interpreter used by JIT codegen (default: `python3` from PATH) |
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
flagfftPlan3d(plan, nx, ny, nz, type)        // NOT_SUPPORTED
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
flagfftSetStream(plan, stream)    // Attach a CUDA stream
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

### Currently Supported

| Feature | Status |
|---|---|
| Rank-1 arbitrary-length C2C, Z2Z | ✅ Cooley-Tukey + Bluestein/Rader |
| Rank-1 arbitrary-length R2C, D2Z (forward) | ✅ |
| Rank-1 arbitrary-length C2R, Z2D (inverse) | ✅ |
| Rank-1 roundtrip (R2C→C2R, D2Z→Z2D) | ✅ |
| Rank-2 contiguous row-major C2C, Z2Z | ✅ RTRT decomposition |
| Rank-2 contiguous row-major R2C, D2Z, C2R, Z2D | ✅ |
| Batched transforms | ✅ |
| In-place and out-of-place | ✅ |
| CUDA stream attachment | ✅ |

### Planned / Not Yet Supported

| Feature | Status |
|---|---|
| Rank-3 transforms (`flagfftPlan3d`) | ❌ Returns `FLAGFFT_NOT_SUPPORTED` |
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
| `--rank` | `1` | Transform rank: `1` or `2` |
| `--api` | `c2c` | Transform type: `c2c`, `z2z`, `r2c`, `d2z`, `c2r`, `z2d` |
| `--shape` | required | Transform size(s), comma-separated: `1024`, `256x256`, `1024,2048,4096` |
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

`tools/run_tests.py` is the primary entry point for running the full test
suite. It orchestrates both accuracy tests (C++ ctest binaries comparing
FlagFFT output against cuFFT) and performance benchmarks (flagfft-cli bench).

#### Usage

```bash
python tools/run_tests.py [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--combination` | `ct` | Test combination preset (see below) |
| `--ops` | all | Comma-separated operator IDs to test (e.g. `c2c_1d,z2z_1d`) |
| `--stages` | `stable` | Comma-separated stage filter |
| `--gpus` | `0` | Comma-separated GPU IDs or `all` for multi-GPU parallel execution |
| `--accuracy-only` | — | Run only accuracy tests (skip benchmarks) |
| `--performance-only` | — | Run only benchmarks (skip accuracy tests) |
| `--build-dir` | `build` | Path to CMake build directory |
| `--output` | `summary.json` | Output file for results |
| `--timeout` | `600` | Per-test timeout in seconds |
| `--warmup` | `10` | Benchmark warmup iterations |
| `--iters` | `100` | Benchmark measurement iterations |
| `-v, --verbose` | — | Verbose output |

#### Combination Presets

| Preset | Description |
|---|---|
| `ct` | Quick smoke test — Cooley-Tukey sizes, batch 1, scale 1.0 |
| `bs` | Quick smoke test — Bluestein/Rader sizes, batch 1, scale 1.0 |
| `full` | Full 1D — all CT sizes × all batches × all scales |
| `2d` | Quick 2D — selected 2D sizes, batch {1,4}, scale 1.0 |
| `2d_full` | Full 2D — selected 2D sizes × all batches × all scales |

#### Examples

```bash
# Quick smoke test (default)
python tools/run_tests.py

# Full test suite on GPU 0
python tools/run_tests.py --combination full --gpus 0

# Full suite across 4 GPUs
python tools/run_tests.py --combination full --gpus 0,1,2,3

# Accuracy only, specific operators
python tools/run_tests.py --combination full --ops c2c_1d,r2c_1d --accuracy-only

# Performance benchmarks only
python tools/run_tests.py --combination full --performance-only
```

#### Output

The runner writes `summary.json` with:

- Per-operator accuracy results (pass/fail, error metrics)
- Per-operator performance results (median latency, speedup vs cuFFT)
- Aggregate geometric mean speedup across all operators

Exit code is `0` if all accuracy tests passed, `1` if any failed.

### C++ Tests (ctest/)

Built with `-DFLAGFFT_BUILD_TESTS=ON`. Each test binary compares FlagFFT
output against cuFFT using normwise relative error metrics (`rel_l2`,
`rel_linf`).

#### Structure

| Test Pattern | Coverage |
|---|---|
| `test_plan` | Plan lifecycle, error codes, unsupported API contracts |
| `test_2d_correctness` | Rank-2 C2C/Z2Z correctness |
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

Each test binary accepts: `--nx`, `--batch`, `--direction`, `--scale`,
`--json-file`.

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

- `conf/operators.yaml` — 14 operator definitions (1D/2D × C2C/Z2Z/R2C/D2Z/C2R/Z2D, plus roundtrip)
- `conf/test_matrix.yaml` — Parameter space: 11 smooth sizes (CT), 4 prime/composite sizes (Bluestein), 3 batch sizes, 3 scale factors, 6 combination rules

---

## License

Apache License, Version 2.0. See [LICENSE](LICENSE) for the full text.
