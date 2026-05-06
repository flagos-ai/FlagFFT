# FlagFFT

FlagFFT is an experimental FFT library that routes Python API calls through a
nanobind C++ core and Triton/TLE-generated CUDA kernels.

## Architecture

The public Python package only exposes the torch.fft-compatible API surface.
Planning, runtime cache lookup, AOT artifact loading, and execution live in the
C++ extension. The runtime cache is split into `ProblemKey`, `PlanKey`, and
`KernelKey` layers so repeated user input maps directly to an executable entry,
while equivalent execution routes and generated Triton kernels are reused across
compatible problems. Python code outside the public API is limited to Triton/TLE
code generation for kernels invoked by the C++ backend.

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
and `cache_keys()` for currently cached entries.

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

## License

FlagFFT is licensed under the [Apache (Version 2.0) license](./LICENSE).
