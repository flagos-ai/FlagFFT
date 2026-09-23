<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

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
  into node, factorization, cost-model, auto/tune-candidate, plan-deserialization
  and builder-context units.
- `src/codegen/` invokes installed Python Triton/TLE source generation during
  plan creation and compiles the result through libtriton_jit.
- `python/flagfft_codegen/` provides the pip-installable source generator and
  its bundled codelets.
- `src/adaptor/` owns device allocation, stream/event operations, target
  identity, and device capability queries. CUDA and IX use the CUDA-compatible
  driver interface, while MUSA, PPU, and MACA use their vendor runtime
  interfaces and NPU uses ACL; the common plan/codegen/exec code does not
  depend on device types.
- `src/utils/` owns shared request/key utilities, JSON/SQLite tuning support, and internal
  headers under `src/utils/include/flagfft/`.

## Hardware Profile and Execution Policy

The adaptor queries the current device and passes its limits to the code
generation subprocess as JSON (`--device-profile`). `BackendProfile`
(`python/flagfft_codegen/backend_profile.py`) turns those facts — backend
identity, warp width, max threads per block and dynamic shared memory — into
the launch decisions: warp count, cooperative stage lanes, leaf and batch
packing, and the shared-memory budget. When the driver omits a fact, the
backend's static defaults apply and the profile records the fallback.

`FLAGFFT_EXECUTION_POLICY` selects how aggressively the device warp is used:

| Policy | Warp heuristic | Leaf packing | Shared-memory budget | Default |
|---|---|---|---|---|
| `legacy` | 32-lane hint | 32 logical lanes | existing heuristic budgets | CUDA, MUSA, PPU, MACA |
| `native` | queried device width | 32 logical lanes | capped by the queried limit | HCU |
| `packed` | queried device width | one device warp | capped by the queried limit | — |
| `balanced` | queried device width | one device warp, bounded by a live-value budget | capped by the queried limit | IX |

Device facts and policy form a profile fingerprint that participates in the
generated-module path, the tuned-plan fingerprints and the in-process kernel
cache key (`KernelKey::repr()` plus the device profile and policy);
`ctest/test_kernel_key.cpp` guards that every field reaching the generator is
part of the key.

Platform notes:

- MACA compiles kernels in the code-generation process and loads the resulting
  `.mcfatbin` from C++ instead of compiling at run time. Its leaves use a
  portable register exchange rather than TLE shared pointers and require at
  least two warps.
- The 3D axis-permutation kernel has three variants: `v1` (correctness
  baseline), `v2` (inline-asm `ld/st.global.v2`, NVIDIA only) and `tile`
  (portable register transpose). Only backends validated for `tile`
  (`ix`, `maca`, `musa`) select it.
- Large complex64 leaves on profile-aware backends can stage their radix
  passes through one shared buffer instead of the two-buffer ping-pong when
  the two-buffer footprint would otherwise limit residency.

The native C API supports arbitrary-length contiguous rank-1 batched C2C,
Z2Z, R2C, D2Z, C2R, and Z2D plans. Real in-place operation uses padded rows;
that verified padded rank-1 form is also supported through `PlanMany`.
Contiguous row-major rank-2 C2C/Z2Z plans are supported through `Plan2d` and
batched `PlanMany`; the current execution strategy is row FFT, tiled
transpose, row FFT, tiled transpose back. Rank-2 real transforms are also
supported. Contiguous row-major rank-3 plans (all six types) are supported
through `Plan3d` and batched `PlanMany` with an RTRT route: FFT along the
innermost axis n2, 3D axis permutation, FFT along n1, permutation, FFT along
n0, and a final permutation back to the natural (n0, n1, n2) layout. Real
rank-3 forward transforms half-pack only the innermost axis and keep the
remaining two axes complex. Custom strided/layouts beyond the supported forms
remain unsupported.

## Raw Execution Nodes

Raw nodes mirror the existing plan tree:

- `CompiledRawLeafNode` / `CompiledRawStridedLeafNode` launch a contiguous or
  strided leaf kernel with plan-owned twiddle and DFT table allocations; the
  real forms are `CompiledRawR2CLeafNode` / `CompiledRawC2RLeafNode`, and
  `CompiledRawDirectDftNode` (with a strided sibling) covers the small
  direct-DFT fallback.
- `CompiledRawFourStepFusedNode` supports four-step routes whose row and column
  children are both leaves. It owns the four-step twiddle and intermediate
  stage buffer. `CompiledRawFourStepGenericNode` and
  `CompiledRawFourStepStridedNode` extend the route to non-leaf children and
  strided I/O, and the `R2CFourStep*` / `C2RFourStep*` / `PackedR2C` /
  `PackedC2R` nodes cover the real-transform forms.
- `CompiledRawBluesteinNode` and its siblings
  (`CompiledRawBluesteinLeafNode`, `CompiledRawBluesteinFullLeafNode`,
  `CompiledRawBluesteinFourStepNode`), together with `CompiledRawRaderNode`,
  handle prime and awkward composite lengths through JIT prepare, pointwise,
  finalize, and convolution FFT child kernels.
- `CompiledRaw2DNode` handles contiguous complex 2D plans with an RTRT route;
  `CompiledRaw2DR2CNode` / `CompiledRaw2DC2RNode` (and their `...RC` forms)
  cover the real variants, and `CompiledRaw1DAs2DNode` runs a batched 1D
  transform through the 2D path.
  This is the correctness baseline for future rocFFT-style `2D_SINGLE` and
  row-plus-block-column strategies.
- `CompiledRaw3DNode` executes contiguous complex 3D plans: per-axis C2C leaf
  chains (with Rader/Bluestein fallback per axis) interleaved with generic
  tiled 3D axis-permutation kernels (`_tiled_transpose3d_kernel_{order}`,
  orders 021/210/201/120). Forward and inverse directions each get a compiled
  node with the permutation order baked in.
- `CompiledRaw3DR2CNode` / `CompiledRaw3DC2RNode` handle 3D real transforms:
  the innermost axis runs expand → C2C → half-pack (or the reverse on the way
  back) while the two outer axes run plain C2C after the axis permutations,
  producing or consuming the (n0, n1, n2/2+1) half-packed layout.

## CLI Tools

`src/cli_tools/common/` owns `CaseSpec`, deterministic buffer generation,
FlagFFT/platform-FFT dispatch, and comparison. Device memory, stream,
synchronization, timer, and query operations use `src/adaptor/`; the selected
backend's reference library is the validation/performance oracle. On Ascend,
the `ops-fft` reference API accepts host pointers and performs its own H2D/D2H
transfers, so the reference path keeps aligned host buffers.
The bench subcommand queries that capability layer before plan creation:

- `bench` binds FlagFFT and the platform reference plan to one adaptor stream before
  warmup and timing so reported event durations cover the actual kernel work.
- `tune` screens candidate plans, re-times the finalists and persists one
  validated winner per request into the SQLite tuning database (`--db PATH`,
  `--no-save` to skip persistence); runtime plan lookup consumes that database
  when `FLAGFFT_TUNE_DB` is set.

The unified interface accepts comma-separated `--shape` values and does not
retain the removed legacy `--lengths` CSV parsing path.

The JSON status boundary is `passed`/`0`, `failed`/`1`, runtime `error`/`2`,
and `skipped` or `unsupported`/`77`. A failed CUDA device query is a runtime
error; `skipped` applies only when a successful query reports zero devices.
Correctness comparison counts non-finite values and fails validation when any
FlagFFT or platform output, or their difference, is non-finite.

## Build Options

The default CMake build produces only `flagfft`. `FLAGFFT_BUILD_CLI=ON` adds
`flagfft-cli` and the selected backend's reference FFT dependency. `FLAGFFT_BUILD_TESTS=ON` adds the
Google Test targets under `ctest/`; CLI behavior remains covered by pytest.
The standalone `bench_vs_cufft` and `flagfft-tuner` targets were removed.

`BACKEND=CUDA`, `BACKEND=MUSA`, `BACKEND=PPU`, `BACKEND=IX`, `BACKEND=MACA`, or
`BACKEND=NPU` selects both the FlagFFT adaptor implementation and the
`libtriton_jit` backend. The IX backend targets Iluvatar/Tianshu GPUs through
the CoreX CUDA-compatible driver and uses the CoreX `libcufft` (ixfft)
implementation as the reference oracle in tests and benchmarks; the MACA
backend targets MetaX GPUs through the native `mcruntime` interface and uses
mcFFT as its reference.

The NPU adaptor uses ACL for FlagFFT device memory and streams and links the
CANN `ops-fft` library for native reference calls. The unified runner uses that
reference only for cases it implements. Reference-unsupported cases still run
the FlagFFT-versus-NumPy accuracy check, while their platform accuracy and
performance entries are recorded as policy skips.

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

The generator is split by responsibility and algorithm family:

- `registry.py` is the single source of truth for the 36 `--kernel` kinds:
  family, leaf I/O mode, required CLI flags, and module-name pattern.
- `cli.py`, `emit.py`, and `metadata.py` own the CLI entry point, per-family
  kernel emission, and generated-module assembly/signatures respectively.
- `kernels_common.py`, `kernels_leaf.py`, `kernels_special.py`,
  `kernels_real.py`, and `kernels_layout.py` contain the shared plan model,
  mixed-radix leaf generation, direct DFT, real-transform pointwise kernels,
  and layout/transpose kernels.
- `backend_profile.py` holds the queried device facts and the execution
  policy; `target.py` carries the explicit codegen target and its warp size;
  `paired_codelets.py` emits the paired odd-radix butterfly forms.
- `codelet/` remains the bundled radix codelet data; generated modules now
  include only the codelet files actually referenced by a kernel.

The native runtime invokes `python -m flagfft_codegen.jit_source` (a thin
facade over `cli.py`); the chosen Python environment must already supply
compatible Triton/TLE dependencies.
Generated JIT source/metadata live in `.flagfft` beside the executable.
`flagfft-cli tune --db PATH` writes measurements and one validated winner per
request (device architecture, length, batch bucket, dtype, direction) into the
SQLite tuning database. Runtime plan lookup consumes that database when
`FLAGFFT_TUNE_DB=PATH` is set.

## Tests

`ctest/` contains Google Test based accuracy tests for all operators.
`tools/run_tests.py` expands 30 acceptance operators from `conf/operators.yaml`
using the dimensions, numeric batches and scales in `conf/test_matrix.yaml`.
Its native capture target compares FlagFFT and the platform library independently
against NumPy; FlagFFT correctness alone decides acceptance. Performance uses
`flagfft-cli bench` once per shape/batch/direction, regardless of input scales.
JSON and incremental CSV retain per-case runtime plans and both correctness
results. The generated input is regenerated from its seed a batch group at a
time and piped into the capture, whose output is streamed back through a pipe,
so no raw array is ever written to disk at all: the result directory holds only
the per-operator JSON files and the aggregated `accuracy.log`/`perf.log`, and
the scratch directory holds only the capture's console logs.
Workspace policy since 2026-09-17 is not to retain raw `.npy`/`.bin` dumps at
all. `summary.json` is a flat array of operator elements in the shape the
acceptance platform parses.
`tests/python/` covers runner behavior and code generation.
`tools/probe_capabilities.py` records device facts, FP64 support and prototype
runs per device; `tools/benchmark_hardware_profile.py` runs the paired
policy A/B comparisons.

Each logical acceptance operator has `torch.float32` and `torch.float64` cases.
On IX, unsupported FP64 cases are omitted before aggregation, so they do not
contribute to accuracy or performance totals. The 2D/3D batch groups use
batch 4 and test out-of-place execution through contiguous `flagfftPlanMany`
layouts.

On Ascend 910B, FP64 is unavailable and `ops-fft` is FP32-only. FlagFFT's
FP32 path covers contiguous 1D, 2D, and 3D C2C/R2C/C2R plans. The current
`ops-fft` reference covers horizontal 1D C2C/R2C/C2R subject to its length
rules and 2D C2C only when each dimension is 32, 64, or 128; it has no 2D real
or 3D plans. These reference limits are represented per case in the runner;
they do not disable FlagFFT's NumPy-backed accuracy check.
