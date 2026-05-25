# FlagFFT Architecture

FlagFFT treats the C/C++ API as the runtime boundary. Python remains for
Triton/TLE JIT source generation and pytest orchestration; native validation,
timing, and tuning enter through `flagfft-cli`.

## C++ Runtime

- `include/flagfft.h` declares the cuFFT-style opaque handle API, including
  the backend-neutral `flagfftStream_t` public stream handle.
- `src/exec/` owns `flagfftHandle` lifecycle, plan creation, stream state,
  plan cache, raw pointer exec dispatch, and optional legacy tensor execution.
- `src/plan/` maps a validated `FFTRequest` to a `PlanNode` tree and is split
  into node, factorization, cost, auto-candidate, and tune-candidate units.
- `src/codegen/` invokes installed Python Triton/TLE source generation during
  plan creation and compiles the result through libtriton_jit.
- `python/flagfft_codegen/` provides the pip-installable source generator and
  its bundled codelets.
- `src/adaptor/` owns device allocation, stream/event operations, target
  identity, and device capability queries. Its current backend implementation
  is CUDA Driver based; the common plan/codegen/exec code does not depend on
  CUDA device types.
- `src/utils/` owns shared request/key utilities, JSON/SQLite tuning support, and internal
  headers under `src/utils/include/flagfft/`.

The native C API supports arbitrary-length contiguous rank-1 batched C2C,
Z2Z, R2C, D2Z, C2R, and Z2D plans. Real in-place operation uses padded rows;
that verified padded rank-1 form is also supported through `PlanMany`. Rank
2/3 and other custom layouts remain unsupported.

## Raw Execution Nodes

Raw nodes mirror the existing plan tree:

- `CompiledRawLeafNode` launches a contiguous leaf kernel with plan-owned
  twiddle and DFT table allocations.
- `CompiledRawFourStepFusedNode` supports four-step routes whose row and column
  children are both leaves. It owns the four-step twiddle and intermediate
  stage buffer.
- `CompiledRawBluesteinNode` handles prime and awkward composite lengths through
  JIT prepare, pointwise, finalize, and convolution FFT child kernels.

## CLI Tools

`src/cli_tools/common/` owns `CaseSpec`, deterministic buffer generation,
FlagFFT/cuFFT dispatch, and comparison. Device memory, stream, synchronization,
timer, and query operations use `src/adaptor/`; cuFFT remains in the CLI only
as the CUDA validation/performance oracle.
Each subcommand queries that capability layer before plan creation:

- `test` embeds route/key and error-contract suites, or runs shared correctness cases.
- `bench` requires shared correctness to pass before returning timing values.
- `tune` uses the shared case contract and `src/cli_tools/tune/` SQLite/candidate
  code; only 1D out-of-place C2C is enabled in this tuning backend.

The unified interface accepts repeatable `--shape` values and does not retain
the removed legacy `--lengths` CSV parsing path.

The JSON status boundary is `passed`/`0`, `failed`/`1`, runtime `error`/`2`,
and `skipped` or `unsupported`/`77`. A failed CUDA device query is a runtime
error; `skipped` applies only when a successful query reports zero devices.
Correctness comparison counts non-finite values and fails validation when any
FlagFFT or cuFFT output, or their difference, is non-finite.

## Build Options

The default CMake build produces only `flagfft`. `FLAGFFT_BUILD_CLI=ON` adds
`flagfft-cli` and its cuFFT dependency. `FLAGFFT_BUILD_TESTS=ON` adds the
Google Test targets under `ctest/`; CLI behavior remains covered by pytest.
The standalone `bench_vs_cufft` and `flagfft-tuner` targets were removed.

`BACKEND=CUDA` selects both the FlagFFT adaptor implementation and
`libtriton_jit` backend. CUDA is the only backend delivered in this version;
other values fail configuration until an adaptor implementation exists.

CMake is the native build/install entrypoint. The pure Python
`flagfft-codegen` package is installed separately with `pip install .` into
the Triton/TLE-enabled interpreter selected at runtime by `FLAGFFT_PYTHON` or
`python3`.

## Python Boundary

Deleted runtime wrappers: top-level `flagfft.py`, `src/api.py`, and
`src/flagfft.py`.

Retained Python package:

- `python/flagfft_codegen/`
- `python/flagfft_codegen/codelet/`

The native runtime invokes `python -m flagfft_codegen.jit_source`; the chosen
Python environment must already supply compatible Triton/TLE dependencies.
Generated JIT source/metadata live in `.flagfft` beside the executable.
`flagfft-cli tune --db PATH` writes measurements and one validated rank-zero
winner with the detected CUDA architecture. Runtime plan lookup consumes that
database when `FLAGFFT_TUNE_DB=PATH` is set.

## Tests

`tests/cli/` shells out to `flagfft-cli --json` for plan/error contracts,
cuFFT comparisons, benchmark schema, unsupported boundaries, and tuning
database reuse. `tests/python/` continues to cover code generation.
