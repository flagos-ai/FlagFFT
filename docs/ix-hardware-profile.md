# IX hardware-profile implementation

The adaptor queries the current device and passes its limits as JSON to the
code-generation subprocess. `BackendProfile` holds the target identity, warp
width, block-thread limit, shared-memory limit and a versioned execution policy.
Explicit backend identity also controls backend-specific source selection;
the installed Triton plugin is only a fallback for standalone source generation.

The existing serialized planner `num_warps` is a hint in 32-thread units.
Codegen converts that hint to device warps once, then adjusts it for cooperative
stage lanes and packing. The launch backend checks the resulting warp width.
Radices, DFT chunks and transpose tile dimensions are not warp-width aliases.

Policies:

| Policy | Warp heuristic | Leaf packing target | Shared-memory packing budget |
|---|---|---|---|
| legacy | Existing 32-lane hint | 32 logical lanes | Existing heuristic budgets |
| native | Queried device width | 32 logical lanes | Capped by queried device limit |
| packed | Queried device width | One device warp | Capped by queried device limit |
| balanced | Queried device width | One device warp, bounded by live-value budget | Capped by queried device limit |

These budgets prune packing choices; they do not predict registers or replace
the compiler/driver's final resource checks. The initial candidate set uses
1/2/4/8 warps. The 4096-thread hardware limit is not a recommended block size.

Four-Step row and column kernels use the same profile model. Their legacy
packing thresholds (`n1 <= 512`, `n1 >= 128`) encode 32-lane device tradeoffs;
under a queried profile both kernels derive the inner pack from the device warp
width, cooperative stage lanes and shared-memory budget instead. This keeps
the four-step data passes coalesced on a 64-lane device: on the BI-V150,
1D C2C 8192 batch 256 dropped from 0.79 ms to 0.37 ms and 1048576 batch 1 from
0.57 ms to 0.33 ms. Measured packing above the budget (8) regressed, so 4
remains the cap for complex64. Algorithmic radices, DFT chunk lengths and the
2D transpose tile keep their existing values.

The 3D axis permutation has three generator variants: `v1` (correctness-first,
one contiguous destination block per program, uncoalesced source reads), `v2`
(tiled, `ld/st.global.v2` inline asm, NVIDIA only) and `tile` (tiled register
transpose with `tl.trans`, no inline asm). Targets where the Triton runtime
exposes a `mthreads`, `ppu` or `iluvatar` module use `tile`, which keeps both
sides of the permutation coalesced and cut 3D C2C 128x2048x64 from 20.9 ms to
6.8 ms on the BI-V150. Tile 32 with four warps measured best; tile 64 was
about 40% slower and is not used.

Large complex64 leaves stage their radix passes through one shared buffer
instead of the two-buffer ping-pong when the queried device's shared memory
would otherwise cap residency. The eligibility rule is conservative: the
legacy policy is excluded, cooperative stage lanes are excluded, every
non-final stage must cover the transform in a single register group, and the
two-buffer footprint must reach a quarter of the device's per-SM shared
memory. In-place stages depend on barriers between the read and write phases,
and a measured four-stage 1024-point leaf still produced wrong results with
per-iteration barriers, so multi-group stage layouts keep the ping-pong. On
the BI-V150 the 2048-point leaf drops from 32 KiB to 16 KiB per block, which
doubles resident blocks and cuts 2D C2C 2048x2048 from 1.30 ms to 0.96 ms,
1D C2C 2048 batch 256 from 58 us to 38 us, 2D R2C 2048x2048 from 1.12 ms to
0.84 ms and 3D C2C 128x2048x64 from 6.9 ms to 6.1 ms.

`balanced` is the default IX policy. It estimates the live FFT value bytes per physical thread from the
transform length, precision, cooperative stage lanes and candidate packing.
It permits packing growth only within the native physical-warp budget and
while the estimate is at most 128 bytes per thread. Otherwise it falls back
toward the native packing. This is
a tunable policy budget rather than a measured register allocation. The first
paired experiment motivated this constraint: packing four 256-point transforms
helped, while packing two 1024-point transforms into one 64-thread block hurt.
The 128-cubed experiment additionally showed that packing based only on the
initial stage lanes could increase the actual launch from one warp to two.
The constraints express those pressure and exchange tradeoffs without a
shape-specific rule. Register usage and barrier costs remain hypotheses until
verified with compiler/profiler counters; measured end-to-end time decides.

Generated module paths contain a fingerprint of the profile, policy version,
Triton version and code-generator sources. The in-process kernel cache includes
device facts and policy; tuned-plan codegen fingerprints distinguish policies.
The cache key is built from `KernelKey::repr()`, so every field that reaches
the generator must appear in it; `ctest/test_kernel_key.cpp` guards that
invariant. Compiled plan descriptions expose actual warp count, block-thread
count, batch/inner packing and profile identifier. Acceptance records device
facts and policy in `env`, preserving the existing 36 operators and JSON/CSV
plan fields.

## Reproduction

Use matching committed checkouts for the candidate and unmodified main, with
the same CoreX SDK, compiler, build flags and dependency revision. Build the
CLI and `numpy_fft_capture` in the IX container. Set both visibility variables:

```bash
export CUDA_VISIBLE_DEVICES=2 IX_VISIBLE_DEVICES=2
export PYTHONPATH="$PWD/python:$PYTHONPATH"
python tools/probe_capabilities.py --build-dir build-ix \
  --output-dir ../results/<timestamp>_ix_fp64
python tools/benchmark_hardware_profile.py --build-dir build-ix \
  --baseline-build-dir ../FlagFFT-dev-ix-hardware-baseline/build-ix \
  --policies legacy,balanced --repeats 3 \
  --output-dir ../results/<timestamp>_ix_profile
```

The paired script gates timing on NumPy correctness, uses warmup 5 / iterations
20 for every timing series and retains incremental CSV and per-case logs.
`--in-process-repeats` reuses a CLI process for repeated series; those repetitions
are grouped by policy rather than interleaved. `--apis` and `--shapes` select
focused follow-up cases. For a final acceptance run use `tools/run_tests.py`.

Times cover complete FFT execution, excluding JIT, plan creation and host/device
copies. Report median-of-series per shape and the baseline/candidate ratio.
Keep platform FFT times as a separate reference and inspect drift before
attributing small differences to the implementation.

## FP64 interpretation

The diagnostic reports native SDK arithmetic, Triton arithmetic, FlagFFT FFTs
and platform FFTs separately. It includes values requiring more than FP32
precision and checks small Z2Z/D2Z/Z2D transforms against NumPy. Compiler failure,
incorrect arithmetic and a timeout are different outcomes. Passing these small
tests certifies only the tested paths. IX FP64 acceptance remains disabled.

On the tested BI-V150/CoreX 4.4 environment, driver queries report a 64-thread
warp, 4096 threads per block and 131072 bytes shared memory per block. Small
platform FP64 FFTs pass while the tested FlagFFT/Triton FP64 paths produce
incorrect results. This is why the acceptance restriction is phrased as a
current implementation policy, not a universal statement about IX hardware.
