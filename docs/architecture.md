# FlagFFT Architecture

FlagFFT keeps execution ownership in C++ and leaves Python responsible only for
the public API wrappers and Triton/TLE code generation.

## C++ Backend

- `src/cpp/common/` builds and validates FFT requests, plan keys, and debug dicts.
- `src/cpp/plan/` maps FFT lengths to plan trees such as `ct_leaf` and `four_step`.
- `src/cpp/codegen/` invokes the Python AOT compiler, parses kernel artifacts, and
  builds runtime tables required by compiled nodes.
- `src/cpp/runtime/` owns CUDA Driver loading, launch helpers, current stream lookup,
  and PyTorch tensor allocation helpers.
- `src/cpp/exec/` owns executable plans, cache hits/misses, normalization, and plan
  node execution. It can optionally read a SQLite tuned-plan database named by
  `FLAGFFT_TUNE_DB`; only valid rows with matching problem fields and tune
  fingerprints are used.
- `src/cpp/bindings/` exposes the nanobind module `_flagfft_core`.

## Python Boundary

`src/api.py` and `src/flagfft.py` expose only the torch.fft-compatible FlagFFT API.
No Python plan cache, tensor table cache, or FFT execution fallback is allowed.

`src/codegen/` is the only Python implementation area below the API layer. It emits
Triton/TLE kernel source and compiles AOT artifacts for C++ to load and execute.
`benchmark/tune_fft_plans.py` is an offline development tool that benchmarks
explicit C++ plans and records measurement history in SQLite. It does not provide
a Python FFT execution fallback or runtime plan cache.

## Current Status

`flagfft.fft` is implemented for CUDA tensors on the last dimension. Other torch.fft
entrypoints exist as API stubs until their C++ plan and exec paths are implemented.
