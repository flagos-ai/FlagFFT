# FlagFFT

FlagFFT is an experimental FFT library that routes Python calls through a
nanobind C++ core and Triton/TLE-generated CUDA kernels.

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

## Validation

The test suite requires CUDA for the active correctness cases:

```sh
python -m pytest test
```

For a focused correctness sweep without pytest collection, run:

```sh
python test/test_fft_mixed_radix.py --lengths 16 105 4096
```

Benchmarks compare FlagFFT against `torch.fft`:

```sh
python benchmark/benchmark_fft_mixed_radix.py --backend cpp --lengths 4096 8192
```

## License

FlagFFT is licensed under the [Apache (Version 2.0) license](./LICENSE).
