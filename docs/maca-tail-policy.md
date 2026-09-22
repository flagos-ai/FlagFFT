# MACA tail candidate policy

This closed candidate policy is **on by default** for a genuine rank-1
batch-1 MACA request. Set `FLAGFFT_MACA_TAIL_POLICY=0` to opt out.
It does not promote an unmeasured size/API/dtype. Eligibility requires MACA,
public rank 1, and batch 1. Both directions are
covered for complex transforms. The native request retains `origin_rank` and
`real_transform`, because C API kernel requests normalize real dtypes to complex.

| Public transform | Length | Candidate |
| --- | ---: | --- |
| Z2Z | 997 | Bluestein 2048 |
| C2C | 1009 | Bluestein 2048 |
| Z2Z | 1048576 | 1024×1024, plain row/col FP64 kernels use bounded P4W4 |
| D2Z/Z2D | 997 | Bluestein 2048 |
| R2C/C2R | 1009 | Bluestein 2048 |
| R2C/C2R/D2Z/Z2D | 2≤N≤23 | Fuse boundaries only when the root is DirectDFT; FP64 tree only at N=23 |
| R2C/C2R | 24≤N≤37 | Fuse DirectDFT boundary only; D2Z/Z2D remain on the production path |

The general rule is therefore: choose a candidate only when the measured
algorithm family, precision, public API and length all match; never infer a
candidate from length alone. All other combinations retain their existing path.
In particular no automatic real 997/1009 combinations outside the API pairs
listed above, no real 1M, 328050, mixed-exchange, batch or multidimensional
changes are included. A 2D/3D axis cannot qualify
merely by having rank-1/batch-1 shape.
The P4W4 resource scope additionally requires plain `FourStepRow/Col`, complex128,
1024-point leaf and 1024×1024 dimensions; real, strided and Bluestein kernels
cannot inherit it. Existing shared-memory bounds remain enforced.

## Strategy classes

- `prime-convolution`: only the measured FP64 997 and FP32 1009 roots use
  Bluestein 2048. The public API is part of the predicate: 997 is D2Z/Z2D,
  while 1009 is R2C/C2R.
- `balanced-fp64-four-step`: only FP64 1048576 uses 1024x1024, and P4W4 is
  attached to its plain row/column 1024 leaves.
- The measured 328050 405x810 candidate is intentionally not automatic: it
  improves relative to baseline but remains below the 0.8 acceptance gate.
- `real-direct`: a small DirectDFT root with `2≤N≤23` accepts all four real
  APIs; `24≤N≤37` accepts only R2C/C2R. The FP64 balanced reduction is
  selected only for N=23; other lengths use the Kahan fallback. Public API and
  the actual DirectDFT root are both part of the predicate.

These are strategy predicates, not a global “fast mode”: a request receives at
most one matching class, and an unmatched request receives `off`.

## Precedence and rollback

1. Explicit old experiment settings win: `FLAGFFT_MACA_TAIL_PLAN` (including
   `default`) replaces automatic plan selection; its original prerequisites
   still apply. Explicit `REAL_DIRECT_DFT=0`, `REAL_DFT_REDUCTION=kahan`,
   `INNER_PACK`, `MAX_WARPS`, `FP64_REGISTER_PACK`, etc. remain honored.
2. The new tail policy supplies narrowly scoped defaults.
3. Existing MACA single-transform defaults and planner remain the fallback.

An automatic whitelisted **complex** root bypasses tuned-plan DB lookup, so an
old DB entry cannot silently defeat the experiment. Nonwhitelisted requests
retain DB behavior. `FLAGFFT_MACA_TAIL_POLICY=0` restores that behavior and is
the supported rollback; any other non-empty value is rejected.
The policy does not set process environment variables. `FLAGFFT_MACA_1D_SINGLE=0`
still disables the older single defaults; do not combine that rollback switch
with the candidate policy when comparing the measured configuration.

## Cache and validation

Native cache identity includes the resolved per-kernel mode plus legacy tree
and resource/codegen overrides. CLI profile directories and generated module
names also separate those variants. This covers `REAL_DFT_REDUCTION=tree` even
with the tail policy off. Switching settings then **creating a new plan** in
the same process cannot hit a previous variant's kernel; an already constructed
plan intentionally retains its original kernels. Concurrent mutation of the
process environment during plan creation is not supported.

CPU tests cover exact guards, original rank/API preservation, repeated planner
toggle, legacy tree/cache identity, CLI same-process emission, scoped source and
metadata equivalence to the manual P4W4 experiment. Device validation of this
automatic integration is still required; CPU/source checks are not MACA runs.

Validation of code `f50973b`: 1299 Python tests and 8 subtests passed; independent
CPU planner suite 13/13 passed; full local CUDA-backend native library compiled
and linked with no visible devices. Evidence is under
`results/20260922_162559_maca_tail_policy_cpu/` in the workspace root.
Rebuild the native library and CLI before device testing; do not combine the
new request/compiler definitions with an older experiment executable.

## GPU verification on the merged dev base (2026-09-22)

The merged branch `perf/maca-tail-on-dev` (`a4ad593`, current `dev` plus the
tail policy) was built for MACA and verified with paired same-card A/B runs:
baseline leaves `FLAGFFT_MACA_TAIL_POLICY` unset, candidate sets it to `1`;
everything else (source, binary, cache isolation, 5 warmup / 30 iterations)
is identical. C550, one physical card per pair.

Accuracy passed for every case in both variants (10/10 performance cases,
`Passed`).

| Case | Direction | baseline | candidate | selected plan change |
| --- | --- | ---: | ---: | --- |
| Z2Z 1048576 (FP64) | forward | 0.260 | 0.811 | `inner_pack` 1/2 -> 4 |
| Z2Z 1048576 (FP64) | inverse | 0.317 | 0.800 | `inner_pack` 1/2 -> 4 |
| Z2Z 997 (FP64) | forward | 0.437 | 0.893 | BS `20x10x10` m2000 -> `16x16x8` m2048 |
| Z2Z 997 (FP64) | inverse | 0.439 | 0.792 | same |
| D2Z 997 (FP64) | forward | 0.468 | 0.899 | same |
| C2C 1009 (FP32) | forward | 0.562 | 0.797 | Rader `n=1008 [7,6,6,4]` -> BS2048 |
| C2C 1009 (FP32) | inverse | 0.572 | 0.800 | same |
| C2R 1009 (FP32) | inverse | 0.636 | 0.854 | same |
| R2C 23 (FP32) | forward | 0.684 | 1.071 | `real_to_complex`+`direct_dft_kernel`+`half_pack` -> single `direct_dft_r2c_kernel` |

The selected plan was read back from each run rather than inferred from the
speedup, so the gains are attributable to the intended algorithm family.

Two points sit marginally below the 0.8 gate (`Z2Z 997` inverse 0.792,
`C2C 1009` forward 0.797). Both measured above 0.8 in the earlier manual
experiments; the 5 warmup / 30 iteration protocol is known to show an
in-run timing step change on this device, so these are not treated as
steady-state regressions or as proof of passing.

Results: `results/20260922_213500_maca_tail_policy_ab/`.
