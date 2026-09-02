# FlagFFT / platform FFT / NumPy correctness validation tool

This directory contains an out-of-tree validation harness.  It does not add a source file
to `libflagfft`, change the public API, or change the existing ctest suite.

The native executable links against an existing `libflagfft.so` and compiles
the already-existing platform test adaptor (`cuFFT`, `muFFT`, or the PPU
CUDA-compatible FFT library).  It accepts the optional
`--implementation=both|flagfft|platform` switch; omitting it keeps the legacy
`both` behavior.  The Python driver uses the two single-implementation modes
so FlagFFT and the platform library each have an independent process and
timeout, while still using the same input and storing the same output files.
It then computes a NumPy reference and applies the current
`ctest/flagfft_test.h` metric:

* worst-batch relative L2 error (`rel_l2`);
* worst-batch relative L-infinity error (`rel_linf`);
* `max_abs` and the mixed pointwise diagnostic;
* the existing transform-class constants and `unit_roundoff * work_factor(N)`
  limits.

The device inverse APIs are unnormalized.  NumPy's `ifftn`/`irfftn` are
normalized, so the driver multiplies their result by `prod(shape)` before
comparison.  C2R/Z2D inputs are generated with real DC/Nyquist coefficients,
matching the current device tests.

## Build the capture executable

Run this from the host or from the development container.  `FLAGFFT_BUILD_DIR`
must point to the already-built FlagFFT tree for the target platform.

```bash
cmake -S tools/numpy_fft_validation \
      -B /tmp/flagfft-numpy-capture-cuda \
      -DFLAGFFT_SOURCE_DIR="$PWD" \
      -DFLAGFFT_BUILD_DIR="$PWD/build" \
      -DBACKEND=CUDA
cmake --build /tmp/flagfft-numpy-capture-cuda -j
```

For MUSA or PPU, use `-DBACKEND=MUSA` or `-DBACKEND=PPU` and set
`-DMUSA_HOME=...` or `-DPPU_HOME=...` when the SDK is not in its default
location.  The build directory must have been built with the same backend.

## Run a smoke validation

The result directory is created under the workspace-level `results/` directory
with the required timestamp prefix.  A full matrix can contain very large
batch-256 arrays; start with a small combination and then expand it.

```bash
python3 tools/numpy_fft_validation/validate.py \
    --capture-bin /tmp/flagfft-numpy-capture-cuda/numpy_fft_capture \
    --backend CUDA \
    --combination 1d_ct_single \
    --ops c2c_1d,r2c_1d,c2r_1d \
    --max-cases 6 \
    --gpu 0
```

Use `--scales all` to run the same three input scales as direct ctest
(`2^-20`, `1`, and `2^20`).  The default follows the selected matrix
combination, which is normally scale `1.0`.

## Result layout

Each case directory contains:

* `input.bin` and `input.npy`: exact device input;
* `flagfft.bin` and `platform.bin`: exact host output bytes from each library;
* `numpy.npy`: the NumPy reference output;
* `case.json`: dimensions, seed, hashes, metric values, limits, and statuses;
* `capture.stdout`, `capture.stderr`, `flagfft.stdout`, `flagfft.stderr`,
  `platform.stdout`, `platform.stderr`, and `flagfft_plan.txt` for diagnosis.

The suite-level `summary.json` and `summary.csv` report both independent
comparisons (`FlagFFT vs NumPy`, `platform vs NumPy`) and the diagnostic
`FlagFFT vs platform` comparison.  A run returns zero only when every captured
case passes for both libraries.  `case.json` retains the legacy `status.capture`
field and adds independent `status.flagfft` / `status.platform` values.  A
timeout in one implementation is therefore recorded as, for example,
`flagfft=passed, platform=timeout`; `capture_stages` also records each stage's
command, duration, return code, and stage-specific stdout/stderr files.

The existing `--timeout` option is applied independently to each
implementation.  The legacy `capture.stdout` and `capture.stderr` files are
still written as a combined view; the separate logs are `flagfft.stdout`,
`flagfft.stderr`, `platform.stdout`, and `platform.stderr`.

If the metric implementation or NumPy environment changes, recompute results
without touching the GPU:

```bash
python3 tools/numpy_fft_validation/validate.py \
    --analyze-only /path/to/results/20260901_120000_numpy_fft_correctness
```

## Is NumPy a suitable gold standard?

Yes, as a common CPU-side oracle and a useful complement to the vendor
library.  It makes platform differences visible and avoids treating one
vendor implementation as automatically correct.  It is still an
implementation, not an exact-arithmetic proof: pin and record Python/NumPy
versions, keep the normalization contract explicit, and retain the current
size-aware tolerance instead of requiring bitwise equality.  For a few small
or suspicious cases, an independent high-precision/direct-DFT check is a good
secondary audit, but it is not needed for the main matrix.
