# FlagFFT

FlagFFT is a JIT-compiled GPU FFT library. It generates backend-targeted GPU
kernels at runtime via [Triton/TLE](https://github.com/FlagTree/flagtree) and
[libtriton_jit](https://github.com/Artlesbol/libtriton_jit), targeting
arbitrary-length transforms that vendor FFT libraries may not optimally
support. The current CMake build supports CUDA, MUSA, PPU, IX
(Iluvatar/Tianshu), MACA (MetaX), NPU (Ascend), and HCU (Hygon BW1000)
backends.

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

The runner prints a live progress table and writes `summary.json` as a flat
array with one element per operator: accuracy (pass/fail), performance
(geometric mean speedup vs the selected backend's reference FFT library) and
the path of the operator's console log. The result directory holds nothing but
JSON and logs.

### Docker

A CUDA development environment with the default dependencies is available:

```bash
docker build -t flagfft-dev -f docker/Dockerfile .
docker run --gpus all -v $(pwd):/workspace/FlagFFT-dev -it flagfft-dev
# Inside the container, run steps 3-5 from above.
```

The Docker image and CI configuration use Python 3.12. MUSA, PPU, IX, MACA,
NPU, and HCU builds require their corresponding vendor SDK/runtime environment
and should be configured with `-DBACKEND=MUSA`, `-DBACKEND=PPU`,
`-DBACKEND=IX`, `-DBACKEND=MACA`, `-DBACKEND=NPU`, or `-DBACKEND=HCU`.

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
| Backend SDK | — | CUDA Toolkit, MUSA SDK, PPU SDK, CoreX CUDA-compatible SDK for IX, MACA SDK, CANN/Ascend runtime for NPU, or DTK 26.04/HIP + hipFFT for HCU |
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
| `FLAGFFT_BUILD_TESTS` | `OFF` | Build the C++ test suite (requires Google Test; NPU also builds the NumPy capture target, other backends also require a reference FFT library) |
| `BACKEND` | `CUDA` | Backend selector: `CUDA`, `MUSA`, `PPU`, `IX`, `MACA`, `NPU`, or `HCU` |
| `CMAKE_BUILD_TYPE` | — | `Release`, `Debug`, `RelWithDebInfo` |

### Full Build (library + CLI + tests)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DFLAGFFT_BUILD_CLI=ON \
      -DFLAGFFT_BUILD_TESTS=ON
cmake --build build -j$(nproc)
```

The default backend is CUDA. Select another supported backend at configure
time, for example `-DBACKEND=MUSA`, `-DBACKEND=PPU`, `-DBACKEND=IX`,
`-DBACKEND=MACA`, `-DBACKEND=NPU`, or `-DBACKEND=HCU`; the
corresponding SDK and runtime libraries must be installed.

### Iluvatar/Tianshu (IX) Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DBACKEND=IX \
      -DCUDAToolkit_ROOT=/usr/local/corex-4.4.0 \
      -DFLAGFFT_BUILD_CLI=ON \
      -DFLAGFFT_BUILD_TESTS=ON
cmake --build build -j$(nproc)
```

The IX backend uses the CoreX CUDA-compatible driver/runtime and ixfft
reference library. It also requires an Iluvatar-enabled FlagTree/Triton
runtime (for example `flagtree===0.5.1+iluvatar3.1`).

### MetaX C550 (MACA) Build

Use the tested MetaX container and the SDK's `cmake_maca` / `make_maca`
wrappers. Set `-DBACKEND=MACA -DMACA_PATH=/opt/maca`; set all three device
filters to the physical card being tested. The card-4 build and validation
procedure is recorded with the MACA validation results in the workspace
`results/` tree (`20260917_165501_maca_hw_profile/REPORT.md`).

### Ascend (NPU) Build

Use the CANN 9 toolchain and the Ascend-enabled Triton/libtriton_jit checkout:

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_OPS_FFT_ROOT=/path/to/ops-fft
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DBACKEND=NPU \
      -DFLAGFFT_TRITON_JIT_SOURCE_DIR=/path/to/libtriton_jit \
      -DASCEND_OPS_FFT_ROOT="$ASCEND_OPS_FFT_ROOT" \
      -DFLAGFFT_BUILD_CLI=ON \
      -DFLAGFFT_BUILD_TESTS=ON
cmake --build build -j$(nproc)
```

The FlagFFT Ascend 910B profile is FP32 `C2C`, `R2C`, and `C2R` for
contiguous 1D, 2D, and 3D plans. Native FP64 is unavailable on Ascend 910B:
`Z2Z`, `Z2D`, and `D2Z` plans return `FLAGFFT_NOT_SUPPORTED`; they are never
silently downcast or moved to a CPU fallback. The unified acceptance runner
keeps those three APIs in its 36-operator manifest and records them as policy
skips on NPU.

For the native comparison and performance baseline, set
`ASCEND_OPS_FFT_ROOT` to a built [CANN ops-fft](https://gitcode.com/cann/ops-fft)
tree. Its current 910B reference is FP32-only: horizontal 1D
`C2C`/`R2C`/`C2R` subject to the documented length limits, and 2D `C2C` only
when each dimension is 32, 64, or 128. It has no 2D real or 3D plans. The
runner still checks FlagFFT against NumPy for those cases and marks only the
unavailable platform/performance rows as `Skipped` with the ops-fft reason.

### Hygon BW1000 (HCU) Build

Use the HCU 3.6 DTK 26.04 image from the
[FlagTree HCU user manual](https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-hcu).
The FlagFFT HCU adaptor uses HIP for device memory, streams, events, and
graphs, and hipFFT as the correctness/performance reference. BW1000 reports
`gfx936` with 64-thread wavefronts, so the generated target is
`hcu:gfx936:64`. Set `HIP_VISIBLE_DEVICES` to the physical BW1000 IDs under
test; the acceptance runner sets it per worker when `--gpus` is used:

```bash
export FLAGTREE_BACKEND=hcu
export HIP_VISIBLE_DEVICES=6,7
cmake -S . -B build-hcu -DCMAKE_BUILD_TYPE=Release \
      -DBACKEND=HCU -DFLAGFFT_BUILD_CLI=ON -DFLAGFFT_BUILD_TESTS=ON
cmake --build build-hcu -j$(nproc)
```

### Environment Variables

| Variable | Description |
|---|---|
| `FLAGFFT_PYTHON` | Path to the Python interpreter used by JIT codegen (default: `python3` from PATH); keep its Python minor version aligned with the CMake build interpreter |
| `FLAGFFT_TUNE_DB` | Path to the SQLite tuning database (default: `.flagfft/tuned_plans.sqlite` beside the executable) |
| `FLAGFFT_TUNE_DISABLE` | Set to `1` to disable tuned plan lookup and always use auto-selected plans |
| `FLAGFFT_EXECUTION_POLICY` | Hardware execution policy: `balanced` (IX default), `native` (HCU default; device warp with queried resource caps), `legacy` (other backends' default/comparison), or `packed` (experimental wider leaf packing) |

### Hardware profiles and IX experiments

On IX arch `71`, contiguous FP32 1D single requests select scoped policies:

| Length | Complex FFT policy | Real FFT policy |
|---|---|---|
| 1024 | `[16,8,8]`, tensor exchange, one physical warp | Same leaf with real input/output |
| 2048 | Tensor exchange, two physical warps | Same leaf with real input/output |
| 16384 | Tensor exchange, inner pack 4, two physical warps | Same Four-Step with real input/output |
| 328050, 340200 | Interleaved shared exchange and twiddle recurrence | Half-length complex FFT plus pre/postprocess |
| 663000 | Unchanged | Half-length complex FFT plus pre/postprocess |
| 1048576 | Interleaved/swizzled exchange, inner pack 8, four physical warps | Half-length complex FFT plus pre/postprocess |

Other lengths, batches, ranks, architectures and FP64 retain their existing
paths. Set `FLAGFFT_IX_CT_SINGLE=0` **before starting the process** to compare
against the original implementation. `FLAGFFT_PACKED_REAL=0` can separately
disable the half-length real path. The policies select matching stage-twiddle
tables and have separate in-process and filesystem kernel cache entries.

Use `tools/ix_ct_single_sweep.py --binary <build>/flagfft-cli --output-dir
<workspace>/results/<timestamp>_ix_ct_single` for serial, alternating
baseline/default measurements on physical GPU 2 (`--gpu` selects another card).
Accuracy validation remains
the responsibility of `tools/run_tests.py`; screening timings alone are not
acceptance results.

`flagfft-cli device-info --json` reports the current device's driver-queried
warp size, thread-block limit and shared-memory limits. Code generation receives
these facts explicitly. `balanced` uses the device warp width for leaf launch
heuristics and bounds packing by queried shared-memory and live-value limits.
`native` uses the device warp width without the live-value packing guard.
Algorithmic
radices and transpose tile dimensions retain their existing meaning.
`packed` additionally targets one device warp when packing small leaf FFTs;
it is an experiment, not a claim of better performance for every shape.
Generated modules use profile-specific directories, and compiled plan text
records warp size, block threads, packing and the profile identifier.

Run isolated FP64 diagnostics and correctness-gated paired experiments in the
backend's container, with `PYTHONPATH` pointing at this checkout's `python/`:

```bash
./build/flagfft-cli device-info --json
python tools/probe_capabilities.py --build-dir build \
  --output-dir ../results/20260917_000000_ix_fp64_probe
python tools/benchmark_hardware_profile.py --build-dir build --policies legacy,balanced --repeats 3 \
  --output-dir ../results/20260917_001000_ix_hardware_policy
```

Use a fresh timestamped output directory for each run. The paired experiment
uses warmup 5 / iterations 20, rotates policy order across repetitions, checks
FlagFFT against NumPy, and writes incremental CSV plus per-case logs. Times
cover the complete FFT execution, excluding plan creation/JIT and host copies.
For acceptance use `tools/run_tests.py`; the experiment is a representative
matrix, not a replacement for the complete 36-operator report.

FP64 diagnostics test native-SDK arithmetic (CoreX `clang++`, otherwise `nvcc`), Triton arithmetic and small
FlagFFT/platform FFTs in separate bounded processes. A failed compiler or
library probe does not prove missing hardware support; passing a small probe
does not certify every transform. Diagnostics do not enable IX FP64 acceptance.
Pass `--capability-report <probe-dir>/capabilities.json` to `tools/run_tests.py`
to attach the probe to the acceptance JSON. The runner checks the device identity
and launch limits before accepting this report; its recorded environment and
tested scope remain relevant when interpreting the results.

### Install

```bash
cmake --install build --prefix /usr/local
```

Installs `libflagfft.so`, the public header (`flagfft.h`), and `flagfft-cli` (if
built), plus the private `libflagfft_triton_jit.so` runtime and the Python
helper scripts it needs, so an installed build does not depend on the source
checkout or on a separately installed `libtriton_jit`. The `flagfft_codegen`
Python package (`pip install .`), PyTorch, and the CUDA runtime remain external
prerequisites.

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
| Ascend 910B FP32 profile | ✅ FlagFFT C2C, R2C, C2R for contiguous 1D/2D/3D; FP64 Z2Z/Z2D/D2Z return `FLAGFFT_NOT_SUPPORTED`; ops-fft reference has narrower 1D/2D coverage |
| Batched transforms | ✅ |
| In-place and out-of-place | ✅ |
| Backend adaptors | ✅ CUDA, MUSA, PPU, IX, NPU (selected at build time) |
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

#### `tune` — Decomposition auto-tuning

```bash
flagfft-cli tune --api c2c --shape 1048576 --batch 256 --db .flagfft/tuned_plans.sqlite
```

`tune` builds the rank-1 decomposition candidates for one length, screens
`--max-candidates` of them, re-times the best `--finalists`, validates each
candidate's output against the reference, and persists one validated winner
per request (device architecture, length, batch bucket, dtype and direction)
into the SQLite tuning database. `--api` accepts `c2c` or `z2z` today, and
`--no-save` runs the same search without writing to the database.

| Option | Default | Description |
|---|---|---|
| `--shape` | `1048576` | Single 1D FFT length |
| `--batch` | `1` | Batch size |
| `--api` | `c2c` | Complex precision: `c2c` or `z2z` |
| `--max-candidates` | `5` | Candidate plans to screen |
| `--finalists` | `2` | Candidates that get the long benchmark |
| `--screen-warmup` / `--screen-iters` | `10` / `50` | Screening timings |
| `--final-warmup` / `--final-iters` | `50` / `1000` | Finalist timings |
| `--db` | `.flagfft/tuned_plans.sqlite` beside the executable | Tuning database |
| `--no-save` | — | Do not persist trials or the winning plan |
| `--json` | — | Output results as JSON |

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
Correctness compares FlagFFT and, when available, the platform FFT library
independently against a float64/complex128 NumPy reference. An operator passes
correctness only when all selected FlagFFT-vs-NumPy cases pass; platform-library
failures are reported under `platform_accuracy` and do not fail FlagFFT
correctness. Performance continues to use `flagfft-cli bench`.

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

The current IX acceptance policy disables FP64. The three FP64 APIs (`Z2Z`, `Z2D`, and
`D2Z`) remain present in the 36-operator manifest for a stable acceptance
surface, but the runner records their accuracy and performance cases as
policy `Skipped` and never dispatches them. The other 18 operators are run
normally. The limitation is recorded in `skip_reason` in JSON and CSV.

On NPU, Ascend 910B has the same three FP64 APIs in the manifest, but the
runner records their accuracy and performance cases as policy `Skipped` before
launch. The remaining FP32 cases run FlagFFT against NumPy. The ops-fft
reference is used for supported horizontal 1D FP32 C2C/R2C/C2R cases and
supported 2D C2C sizes; its unsupported 2D real, 3D, out-of-matrix 2D, and
out-of-range 1D cases retain the FlagFFT NumPy result while their platform and
performance rows are marked `Skipped`. NPU performance reports ops-fft
reference timing and speedup wherever that reference exists.

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
| `--timeout` | `200` | Independent timeout for each FlagFFT/platform/benchmark process |
| `--warmup` / `--iters` | `10 / 100` | Benchmark warmup and measurement iterations |
| `--dry-run` | — | Print selected cases without execution or result files |
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
```

#### Output

All outputs use format version 3. `summary.json` is a flat array with one
element per operator, matching the shape the acceptance platform parses.
Filtered runs contain the selected operators, and the manifest records the
exact partial selection.

- `manifest.json`: selected operators, complete expected case lists, parameter matrix and runtime environment.
- `summary.json`: one element per operator, in acceptance order, carrying `accuracy`, `platform_accuracy`, `performance`, `perf_log_path` and the run metadata.
- `incremental.csv`: case/phase, operator ID, shape, numeric batch, direction, scale, both correctness statuses and errors, policy skip reason, limits, timings and actual plan. CSV quoting preserves multiline plan text.
- `{op_id}/{accuracy,platform_accuracy,performance}_result.json`: aggregate per-operator results, including the per-case metrics, seeds, hashes and plans. The `accuracy.details` and `performance.data.default` report fields are retained here.
- `{op_id}/accuracy.log`: the FlagFFT and platform capture console output for every accuracy case of that operator, with each plan between `FLAGFFT PLAN BEGIN/END` delimiters and each case introduced by a `----- <case_id> <impl> -----` header. The file closes with a pytest-shaped verdict line, so a platform log parser derives the same counts and status as `summary.json`.
- `{op_id}/perf.log`: the benchmark console output for every performance case.

Neither the generated input nor the captured output ever touches the disk. The
input is regenerated from its seed one batch group at a time and pushed into the
capture's stdin as the capture consumes it, and the output is streamed back
through a pipe and folded into the NumPy comparison batch by batch. The scratch
directory holds nothing but the capture's own console logs. No `.bin`, `.npy`,
`case.json`, `flagfft_plan.txt` or per-case directory is produced, so a full run
writes exactly five files per operator.

In `summary.json`, `accuracy` and `platform_accuracy` carry the counts, the
`PASS`/`FAIL` status and the log path the platform reads; `performance` is one
row per case with the measured speedup in the complex column and the operator's
geometric mean in `avg_speedup`.

Correctness and performance have separate case IDs; scale is omitted from
performance IDs. A missing result cannot make an operator pass. Numeric
failures serialize nonfinite metrics as JSON null while retaining failure
status. A plan is saved as soon as its creation succeeds and refreshed after
successful execution; creation failures have `plan: null`.

Performance rows retain the raw speedup and record `baseline_valid`. NPU rows
for ops-fft-supported cases include reference timing and speedup; policy-skipped
reference cases serialize those fields as `null`.
An incorrect platform baseline is excluded from the aggregate speedup
statistics; operators failing FlagFFT correctness are excluded as well.
Performance-only runs have an unknown baseline validity.

Exit code is 0 when all requested FlagFFT correctness and/or performance phases
pass; an explicit backend-policy skip such as IX FP64 is allowed. Exit code 1
indicates a failure, incomplete phase, or unexpected skip; 2 is a configuration
error and 130 is interruption. Platform correctness is reported independently.

### C++ Tests (ctest/)

Built with `-DFLAGFFT_BUILD_TESTS=ON`. Each test binary compares FlagFFT
output against the selected backend's reference FFT library using normwise
relative error metrics (`rel_l2`, `rel_linf`). On NPU, the reference-based
GTest binaries are skipped because they are not wired to the host-pointer
ops-fft adaptor; `ctest/numpy_fft_capture` is used by the unified runner. It
runs FlagFFT for every selected FP32 case, compares against NumPy, and invokes
ops-fft for cases inside its support matrix.

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

The unified acceptance runner is the supported entry point for IX and NPU. It
keeps the three FP64 APIs in the report but filters them before native
execution. On NPU it also applies the ops-fft 1D/2D/3D support matrix per case:
FlagFFT accuracy continues to run, while an unavailable platform comparison or
benchmark is recorded as `Skipped`. Direct invocation of bundled FP64 C++ test
binaries is not an IX or NPU acceptance test and may call unsupported vendor
functionality. NPU `FLAGFFT_BUILD_TESTS` builds the capture executable and
links ops-fft when `ASCEND_OPS_FFT_ROOT` is set.

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

- `conf/operators.yaml` — the fixed 36-operator acceptance list (six APIs across six groups)
- `conf/test_matrix.yaml` — 1D CT/Prime and 2D/3D sizes, numeric batch counts, and scales

---

## License

Apache License, Version 2.0. See [LICENSE](LICENSE) for the full text.
