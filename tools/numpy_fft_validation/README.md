# Native output capture for NumPy acceptance

`capture.cpp` executes FlagFFT and the platform FFT library on identical
input bytes. It is built as `build/ctest/numpy_fft_capture` by
`-DFLAGFFT_BUILD_TESTS=ON`. Python input generation, NumPy comparisons and
matrix expansion are integrated into `tools/run_tests.py`; the separate
`validate.py` entry point has been removed.

Install dependencies with `pip install -e '.[test]'`. Run from the repository
or its sibling worktree:

```bash
python tools/run_tests.py --accuracy-only \
  --ops 1d_ct_single_c2c,1d_prime_single_c2r
python tools/run_tests.py --accuracy-only --combination 2d
python tools/run_tests.py --scales all --accuracy-only --ops 1d_ct_single_z2d
```

FlagFFT and platform capture run in separate processes with independent
timeouts. FlagFFT acceptance depends on its NumPy comparison.
`platform_accuracy` reports platform-vs-NumPy metrics without affecting that
conclusion. Inputs retain native float32/complex64 or float64/complex128 dtype;
the NumPy oracle explicitly computes in float64/complex128. Real-inverse inputs
satisfy multidimensional Hermitian constraints, and inverse normalization
matches the unnormalized device APIs.

The runner passes `--input=-` and `--output-dir=-`, so the capture reads the
generated input from stdin and writes its result to stdout. The input is not
read back from a file either: its SplitMix64 stream is addressable, so the
runner regenerates one batch group at a time and pushes it into stdin as the
capture consumes it. Nothing large is materialized on disk, and the result
directory therefore contains only the per-operator JSON and the aggregated
`accuracy.log`/`perf.log`. In that mode the plan description goes to stderr
between `FLAGFFT PLAN BEGIN/END` delimiters: the runner lifts it into the JSON,
and it also reaches the operator log, where it stays readable. The plan is
written before execution and refreshed with compiled details afterwards.

## Optional standalone native build

The CMake file in this directory remains available to capture outputs against
an already-built FlagFFT tree. Compile and run inside the development container
with the target backend:

```bash
cmake -S tools/numpy_fft_validation -B build/numpy-capture \
  -DFLAGFFT_SOURCE_DIR="$PWD" -DFLAGFFT_BUILD_DIR="$PWD/build" -DBACKEND=CUDA
cmake --build build/numpy-capture -j
python tools/run_tests.py --accuracy-only \
  --capture-bin build/numpy-capture/numpy_fft_capture \
  --ops 1d_ct_single_c2c
```

For MUSA/PPU/IX use the matching backend and SDK paths. For MACA, point
`-DMACA_PATH` at the same SDK used for the FlagFFT build and run the capture
inside the MetaX environment with `CUDA_VISIBLE_DEVICES=4`,
`MACA_VISIBLE_DEVICES=4`, and `MC_VISIBLE_DEVICES=4` when validating card 4.
The capture's platform stage uses mcFFT independently from FlagFFT.

For NPU, use CANN 9 and
set `ASCEND_OPS_FFT_ROOT` to a built [ops-fft](https://gitcode.com/cann/ops-fft)
tree; ops-fft uses host pointers, which the NPU adaptor handles explicitly.
The reference library is FP32-only and currently covers horizontal 1D
`C2C`/`R2C`/`C2R` plus 2D `C2C` for dimensions 32, 64, and 128. Unsupported
reference cases are reported as platform/performance `Skipped` rows while
FlagFFT still runs against NumPy.

For IX, point
`CUDAToolkit_ROOT` at the CoreX SDK and use an Iluvatar-enabled Triton runtime;
the unified runner omits FP64 cases under the current IX acceptance policy,
without adding them to test totals. Use `tools/probe_capabilities.py` for
device/toolchain-specific FP64 evidence. The native executable
supports `--implementation=flagfft|platform|both`; the runner uses the two
single-library modes, because stdin holds a single copy of the input and
cannot be replayed for a second library. Rank-2 and rank-3 batched capture uses
PlanMany with contiguous layouts; batch cases test in-place and out-of-place
execution.

The named-file form (`--input FILE --output-dir DIR`, which also writes
`flagfft_plan.txt` into the output directory) remains available and is what
`tools/benchmark_hardware_profile.py` and `tools/probe_capabilities.py` use.
