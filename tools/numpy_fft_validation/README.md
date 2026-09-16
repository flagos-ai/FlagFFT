# Native output capture for NumPy acceptance

`capture.cpp` executes FlagFFT and the platform FFT library on identical
input bytes. It is built as `build/ctest/numpy_fft_capture` by
`-DFLAGFFT_BUILD_TESTS=ON`. Python input generation, NumPy comparisons,
matrix expansion and reanalysis are integrated into `tools/run_tests.py`;
the separate `validate.py` entry point has been removed.

Install dependencies with `pip install -e '.[test]'`. Run from the repository
or its sibling worktree:

```bash
python tools/run_tests.py --accuracy-only \
  --ops 1d_ct_single_c2c,1d_prime_single_c2r
python tools/run_tests.py --accuracy-only --combination 2d
python tools/run_tests.py --scales all --accuracy-only --ops 1d_ct_single_z2d
python tools/run_tests.py --analyze-only ../results/20260916_120000_acceptance36
```

FlagFFT and platform capture run in separate processes with independent
timeouts. FlagFFT acceptance depends on its NumPy comparison.
`platform_accuracy` reports platform-vs-NumPy metrics without affecting that
conclusion. Inputs retain native float32/complex64 or float64/complex128 dtype;
the NumPy oracle explicitly computes in float64/complex128. Real-inverse inputs
satisfy multidimensional Hermitian constraints, and inverse normalization
matches the unnormalized device APIs.

Each correctness case always retains `case.json`, library-specific
stdout/stderr, and `flagfft_plan.txt`. The native `input.bin`, `flagfft.bin`,
and `platform.bin` files are temporary by default and are retained according to
`--artifacts failed|all`; no `npy` files are generated. Use `--artifacts all`
when `--analyze-only` or raw-data reproduction is needed. The actual plan text
is also embedded in the final JSON and incremental CSV. A plan is saved before
execution and updated with compiled details after successful execution.

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

For MUSA/PPU/IX use the matching backend and SDK paths. For NPU, use CANN 9 and
set `ASCEND_OPS_FFT_ROOT` to a built [ops-fft](https://gitcode.com/cann/ops-fft)
tree; ops-fft uses host pointers, which the NPU adaptor handles explicitly.
The reference library is FP32-only and currently covers horizontal 1D
`C2C`/`R2C`/`C2R` plus 2D `C2C` for dimensions 32, 64, and 128. Unsupported
reference cases are reported as platform/performance `Skipped` rows while
FlagFFT still runs against NumPy. For IX, point
`CUDAToolkit_ROOT` at the CoreX SDK and use an Iluvatar-enabled Triton runtime;
the unified runner keeps `Z2Z`, `Z2D`, and `D2Z` in the manifest but skips them
because IX does not support FP64. The native executable
supports `--implementation=flagfft|platform|both`; the runner uses the two
single-library modes. Current rank-3 capture and benchmark require batch 1.
