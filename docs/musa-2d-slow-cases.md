# MUSA 2D slow cases: launch packing and batched prime FFTs

This experiment starts from `efc88a6`, after the S5000 batch-1 complex 2D
graph replay change. Results are stored outside the worktree at
`../results/20260913_020010_musa_2d_slow/`. The optimized implementation
is `a8ea518`.

## Changes

* **Bluestein launch metadata:** the generated prepare-row, pointwise-row,
  and finish-column kernels report `inner_pack`, but the C++ loader did
  not read it for those kernel kinds. Their runtime value stayed at one,
  inflating the launch grid when the generated kernel packed four inner
  transforms. The loader now reads both packing and fused-twiddle metadata
  consistently. This fix applies to all backends.
* **MUSA small mixed leaves:** 128–256-point leaves with base `lanes=1`
  can pack up to eight transforms per CTA, subject to the existing thread
  and shared-memory limits. This improves the 209/221-point leaves of
  46189 = 209 × 221. The bounded policy uses at most 64 KiB of shared
  memory for eight FP64 transforms. Power-of-two layouts are unchanged.
* **S5000 batched FP64 8191:** at batch sizes starting from the measured
  lower bound of 16, prefer Bluestein's 16384-point convolution to the
  Rader 8190 = 9 × 910 convolution. Keep smaller batches and the 1009-point
  leaf Rader route on their existing policy. This is a measured tuning
  choice for `device_type=musa, device_arch=31`, not a general assertion
  that Bluestein beats Rader.
* **S5000 FP64 Bluestein fusion:** for this 8191-point route, fuse chirp
  and pointwise operations into the four-step FFT when the complete batch
  fits the existing chunk budget. Larger batches retain chunked generic
  execution. Arithmetic precision and tolerances are unchanged.

## Experiments that were not retained

For 46189 × 48 Z2Z, increasing packing from four to eight reduced the
initial measurement from approximately 7.8 to 5.0 ms; sixteen took 6.76 ms.
For C2C, eight took 1.13 ms, whereas sixteen took 2.81 ms. A 143 × 323
decomposition took 5.85 ms for Z2Z and did not beat the original split
with packing eight.

For 8191 × 1009 Z2Z, switching only the 8191 axis to Bluestein took
11.50 ms, versus approximately 45.6 ms for the original Rader route.
Switching both axes took 13.94 ms, so the 1009 Rader was retained.
Extending packing eight to regular power-of-two FP64 leaves also regressed
this experiment (12.05 ms), and was excluded from the final policy.
Enabling the bounded four-step FP64 fusion reduced the exploratory result
to 9.51 ms. These exploratory measurements use ten warmups and 50 timed
iterations; use the final paired matrix for reported speedups.

## Validation and reproduction

The MUSA environment is `baai-mthread`, MTT S5000 GPU 0, capability 3.1,
driver/runtime 4.3, Release build, and the existing mthreads FlagTree site.
`bench.sh` and `verify.sh` in the result directory preserve the commands.
Final performance runs use 20 warmups and 100 timed iterations, batch one,
out of place, with an interleaved muFFT reference.

The baseline uses both a saved shared library and a snapshot of the Python
generator at `efc88a6`; using only the old library would silently include
the new packing policy in the baseline. Both variants otherwise share the
CLI, GPU, Triton environment, and reference library.

Validation passed 72 2D correctness tests covering all six APIs and
roundtrips at 128 × 128, 46189 × 48, 32 × 46189, 8191 × 1009,
32 × 8191, and 8191 × 32. Five additional 1D tests exercise mixed CT and
batched FP64 Bluestein reuse. The original three input scales and error
limits are retained. Two C++ regression tests and three Python tests pass.
The new boundary-metadata regression test fails against the old library
(`inner_pack=1` versus the plain column kernel's four), and passes with the
fix; both logs are retained.

`comparison.csv` and `REPORT.md` contain the final paired per-case numbers.
The unchanged 128 × 128 control cases can vary by several microseconds in
event timing; such variation is not attributed to these optimizations.

The independent Luna A100 validation uses
`../results/20260913_020649_a100_2d_opt/`. CUDA graph replay should retain
its existing policy; MUSA launch-policy timings must not be transferred
to A100 without measurements.

## Final paired results

Selected forward cases (milliseconds), relative to `efc88a6`:

| API | Shape | Baseline | Optimized | Speedup |
|---|---|---:|---:|---:|
| C2C | 46189 × 48 | 1.64064 | 1.12412 | 1.459× |
| C2C | 32 × 46189 | 1.03608 | 0.68712 | 1.508× |
| Z2Z | 46189 × 48 | 7.49400 | 5.04216 | 1.486× |
| Z2Z | 8191 × 1009 | 45.52448 | 9.49932 | 4.792× |
| D2Z | 8191 × 1009 | 25.11944 | 6.89856 | 3.641× |

All 24 measured cases outside the unchanged 128 × 128 control improved.
Several large FP64 transforms still trail muFFT; the complete report
includes reference timings rather than implying parity from these gains.

Luna also validated the common metadata fix on A100: C2C forward
8191 × 1009 fell from 3.123200 to 2.521088 ms (1.239×), with 20 warmups
and 100 iterations. All three C2C correctness tests and the new metadata
regression passed. This follow-up uses `a8ea518` cherry-picked as `08e2161`
in `FlagFFT-dev-a100-2d-opt`; it is separate from the earlier CUDA graph
comparison, and retains CUDA graph execution.

The 128 × 128 Z2Z forward control was rechecked with 200 warmups,
1000 iterations, and reversed baseline/candidate order. Two old/new
pairs were 0.034480/0.034120 and 0.034400/0.034440 ms. Differences
were within approximately 1.1%; the initial apparent ~20% slowdown did
not reproduce. Raw initial measurements remain in the paired matrix.
