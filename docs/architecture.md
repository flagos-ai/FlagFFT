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
  node execution. On cold `ProblemKey` misses it can read a SQLite tuned-plan
  database, using `.flagfft/tuned_plans.sqlite` by default, `FLAGFFT_TUNE_DB` as
  an override, and `FLAGFFT_TUNE_DISABLE` as a kill switch. Only valid rows with
  matching problem fields and tune fingerprints are used; warm `ProblemKey`
  hits return the cached executable directly.
- `src/cpp/bindings/` exposes the nanobind module `_flagfft_core`.

Four-step execution has two runtime forms. Nested or non-leaf children use the
generic transpose, row exec, twiddle-transpose, column exec, final transpose
sequence. Four-step nodes whose row and column children are both `ct_leaf` use
specialized fused row/column AOT kernels keyed as `four_step_row` and
`four_step_col`. The fused row kernel reads the original strided input columns
directly, while the fused column kernel loads the four-step twiddle and writes
the final natural output order, reducing the hot path to two launches. The fused
column kernel uses a shared codegen/runtime inner-pack rule: it keeps one inner
column per CTA for smaller `n1` values and packs two adjacent inner columns once
`n1 >= 128`, interleaving those slots in the vector layout to reduce
natural-order store stride pressure visible in Nsight Compute.

## Python Boundary

`src/api.py` and `src/flagfft.py` expose only the torch.fft-compatible FlagFFT API.
No Python plan cache, tensor table cache, or FFT execution fallback is allowed.

`src/codegen/` is the only Python implementation area below the API layer. It emits
Triton/TLE kernel source and compiles AOT artifacts for C++ to load and execute.
This includes the fused four-step row/column kernel variants; Python still does
not own plan caching, tensor caching, or FFT execution fallback.

`src/tuning.py` provides the official offline tuning orchestration used by
`flagfft.tune(...)` and the thin `flagfft-tune` CLI. It benchmarks explicit C++
plans through `_flagfft_core.enumerate_plan_candidates()` and
`_flagfft_core.fft_with_plan()`, records measurement history in SQLite, and can
load problem lists from CLI flags, a JSON string, or a JSON file. The focused
mixed-radix benchmark can call the same tuner before timing with `--tune`, or
force a new winner with `--tune retune`. Only `flagfft.fft` is currently
benchmarkable; all other API names are reserved in the tune dispatcher and raise
`NotImplementedError` without invoking any Python FFT fallback.

For contiguous `ct_leaf` kernels whose selected lane count is smaller than a
warp, codegen can pack multiple independent batch rows into one CTA. The pack
factor is derived from the lane block and a conservative shared-memory budget,
rounded to powers of two for Triton/TLE shape constraints, and capped for very
small leaves to avoid launching fewer blocks than the GPU can occupy. Packed
artifacts expose `batch_per_block`; `src/cpp/exec/` uses that value only for
contiguous leaf grid sizing. Four-step fused row kernels keep their existing
`(inner, batch)` grid; fused column kernels use the dynamic inner-pack rule
described above.

Planner `ct_leaf` eligibility is request-aware. For `float32` and `complex64`
inputs, the planner records `smem_size` as float32 scratch elements and converts
that to bytes only when checking whether the current CUDA device can launch the
candidate. It queries `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`,
falls back to `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK`, and uses 48 KiB
only if the Driver query fails. This changes candidate enumeration without
changing the plan schema: large single-CTA leaves such as 4096-point FFTs can be
tuned on devices with sufficient opt-in dynamic shared memory, while unsupported
double-precision scratch rules remain outside the leaf planner.

## Current Status

`flagfft.fft` is implemented for CUDA tensors on the last dimension. Other torch.fft
entrypoints exist as API stubs until their C++ plan and exec paths are implemented.
