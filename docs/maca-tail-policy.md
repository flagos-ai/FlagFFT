# MACA tail candidate policy

`FLAGFFT_MACA_TAIL_POLICY=1` enables a **default-off** closed candidate policy.
It does not promote an unmeasured size/API/dtype and does not change dev.
Eligibility requires MACA, public rank 1, and batch 1. Both directions are
covered for complex transforms. The native request retains `origin_rank` and
`real_transform`, because C API kernel requests normalize real dtypes to complex.

| Public transform | Length | Candidate |
| --- | ---: | --- |
| Z2Z | 997 | Bluestein 2048 |
| C2C | 1009 | Bluestein 2048 |
| Z2Z | 1048576 | 1024×1024, plain row/col FP64 kernels use bounded P4W4 |
| C2C | 328050 | 405×810 |
| R2C/C2R/D2Z/Z2D | 23 | Fuse boundaries **only if the existing root is DirectDFT**; FP64 uses tree |
| R2C/C2R | 29/31/37 | Fuse DirectDFT boundary only; D2Z/Z2D remain on the production path |

The general rule is therefore: choose a candidate only when the measured
algorithm family, precision, public API and length all match; never infer a
candidate from length alone. All other combinations retain their existing path.
In particular no automatic real 997/1009/1M, FP64 328050, mixed-exchange,
batch or multidimensional changes are included. A 2D/3D axis cannot qualify
merely by having rank-1/batch-1 shape.
The P4W4 resource scope additionally requires plain `FourStepRow/Col`, complex128,
1024-point leaf and 1024×1024 dimensions; real, strided and Bluestein kernels
cannot inherit it. Existing shared-memory bounds remain enforced.

## Strategy classes

- `prime-convolution`: only the measured FP64 997 and FP32 1009 roots use
  Bluestein 2048.
- `balanced-fp64-four-step`: only FP64 1048576 uses 1024x1024, and P4W4 is
  attached to its plain row/column 1024 leaves.
- `real-direct`: only measured DirectDFT real roots use boundary fusion. The
  FP64 balanced reduction is selected only for N=23; N=29/31/37 use the tested
  Kahan fallback. Public API is part of the predicate.

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
retain DB behavior. `FLAGFFT_MACA_TAIL_POLICY=0` (or unset) restores that behavior.
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
