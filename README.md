# FlagFFT

FlagFFT is an experimental FFT library that routes Python API calls through a
nanobind C++ core and Triton/TLE-generated CUDA kernels.

## Architecture

The public Python package only exposes the torch.fft-compatible API surface.
Planning, runtime cache lookup, AOT artifact loading, and execution live in the
C++ extension. The runtime cache is split into `ProblemKey`, `PlanKey`, and
`KernelKey` layers so repeated user input maps directly to an executable entry,
while equivalent execution routes and generated Triton kernels are reused across
compatible problems. Optional offline tuning stores measured plan winners in a
SQLite database; C++ reads only fingerprint-matched valid winners and falls back
to the automatic planner when no tuned route is available. Python code outside
the public API is limited to Triton/TLE code generation and offline benchmark
orchestration for kernels invoked by the C++ backend.

Current source layout:

- `src/cpp/plan/`: FFT route selection and plan tree construction.
- `src/cpp/exec/`: executable plan cache and CUDA execution flow.
- `src/cpp/codegen/`: C++ wrapper around the Python Triton AOT compiler.
- `src/cpp/runtime/`: CUDA Driver and tensor runtime helpers.
- `src/codegen/`: Python Triton/TLE kernel source emission and AOT CLI.
- `src/codelet/`: reusable generated-kernel codelets.

Forward complex leaf kernels currently have specialized codelets for radix
2/3/4/5/6/7/8/9/10/11/12/13/15/16/17/19. These radices are emitted inline in
generated Triton kernels, so the C++ planner does not pass DFT table parameters
for those stages.

Contiguous leaf kernels with fewer than one warp of active lanes may pack
multiple batch rows into one CTA. The generated AOT artifact records
`batch_per_block`, and C++ uses it to launch `ceil(batch / batch_per_block)`
blocks. Packed leaves keep per-batch shared-memory regions separate and use
limited padding only where Nsight Compute showed the bank-conflict reduction was
worth the additional shared memory.

Four-step plans whose row and column children are both `ct_leaf` nodes execute
through fused AOT kernels. The row kernel reads the original four-step strided
columns directly, and the column kernel applies the four-step twiddle while
writing final natural order. This removes the separate transpose and
twiddle-transpose launches from the hot path. The fused column kernel shares one
inner-pack rule between Python codegen and C++ launch sizing: smaller `n1`
plans keep one inner column per CTA, while `n1 >= 128` packs two adjacent
`inner` columns and interleaves vector lanes to reduce natural-order store
stride pressure.

Unsupported large-prime or large-prime-factor lengths fall back to a C++
`BluesteinPlanNode`. Bluestein maps an `N`-point FFT to a convolution length
`M >= 2N - 1` chosen from lengths the existing planner can execute, then reuses
the normal `M`-point FFT child plan. The `ProblemKey` cache still maps repeated
user inputs directly to an executable entry, while `KernelKey` caching avoids
recompiling the shared `FFT_M` kernels when nearby Bluestein lengths choose the
same convolution size.

The implemented compute path is currently `flagfft.fft` for CUDA tensors on the
last dimension. The remaining torch.fft-compatible entrypoints are present as API
stubs and raise `NotImplementedError` until their C++ plan/exec paths are added.

## Installation

Install the package into the current Python environment:

```sh
python -m pip install .
```

## Development

For local development, use an editable install:

```sh
python -m pip install -e . --no-build-isolation
```

This reuses build dependencies already installed in the current environment.
Re-run it after changing the C++ extension or build configuration.

Useful C++ debug helpers exposed by `_flagfft_core` include `debug_keys()` for
inspecting problem/plan/kernel keys, `cache_info()` for the three cache layers,
`cache_keys()` for currently cached entries, `enumerate_plan_candidates()` for
offline tuning, and `fft_with_plan()` for benchmarking an explicit plan.

Offline tuning is exposed as the `flagfft-tune` console script and as
`flagfft.tune(...)`. Both routes benchmark explicit C++ plan candidates and
store measurement history in SQLite:

```sh
flagfft-tune --api fft --lengths 4096 --batch 256 --retune
```

```py
import flagfft

flagfft.tune(flagfft.fft, lengths=[4096], batch=256, retune=True)
```

The tuner writes `.flagfft/tuned_plans.sqlite` unless `--db` or the Python
`db=` argument is provided. Normal `flagfft.fft` calls read that default
database automatically on cold plan-cache misses. `FLAGFFT_TUNE_DB` overrides
the path, and `FLAGFFT_TUNE_DISABLE=1` disables tuned-plan lookup. Use
`--explain-cache` to inspect why a current problem does or does not hit the
tuned database.

Problem lists can be supplied as a JSON string or JSON file. The JSON payload
may be a single object or a list of objects:

```sh
flagfft-tune --json '{"api":"fft","lengths":[4096,8192],"batch":256,"retune":true}'
flagfft-tune --json-file problems.json
```

The JSON fields are `api`, `lengths`, `batch`, `dtype`, `device`, `dim`, `norm`,
`db`, `retune`, `dry_run`, `static_limit`, `finalists`, `warmup`, and `iters`.
Only `fft` is currently benchmarkable; the other torch.fft-compatible names are
reserved and raise `NotImplementedError` in the tune layer.

For the focused 16K single-batch four-step case:

```sh
flagfft-tune --api fft --lengths 16384 --batch 1 --retune
```

## Validation

The test suite requires CUDA for the active correctness cases:

```sh
python -m pytest test
```

For a focused correctness sweep without pytest collection, run:

```sh
python test/test_fft_mixed_radix.py --lengths 10 12 15 17 19 60 120 190 255 1020
```

Benchmarks compare FlagFFT against `torch.fft` through the C++ backend:

```sh
python benchmark/benchmark_fft_mixed_radix.py --lengths 10 12 15 17 19 60 120 190 255 1020
```

Add `--tune` to run normal tuning for each length before timing; existing valid
winners are reused. Use `--tune retune` to supersede existing tuned winners
before the benchmark:

```sh
python benchmark/benchmark_fft_mixed_radix.py --lengths 16384 --batch 1 --tune
python benchmark/benchmark_fft_mixed_radix.py --lengths 16384 --batch 1 --tune retune
```

Set `FLAGFFT_TUNE_DB=/path/to/tuned_plans.sqlite` only when benchmarking against
a non-default tuned-plan database.

## License

FlagFFT is licensed under the [Apache (Version 2.0) license](./LICENSE).
