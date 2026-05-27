# 1D FFT Test Refactor Design

**Date:** 2026-05-26
**Status:** Amended 2026-05-27 (cuFFT conformance and runtime tiers)

## Goal

Refactor C++ test suite (`ctest/`) for 1D FFT: one test file per transform type, comprehensive size coverage (small/medium/large), batch combined with size as test parameters, and value-parameterized tests (`TEST_P`) to eliminate boilerplate.

## File Structure

```
ctest/
├── flagfft_test.h             # Shared header + test registry config
├── test_exec_c2c.cpp          # C2C: Forward, Inverse, Roundtrip  (NEW)
├── test_exec_z2z.cpp          # Z2Z: Forward, Inverse, Roundtrip  (NEW)
├── test_exec_r2c.cpp          # R2C: Forward vs Ref               (NEW)
├── test_exec_c2r.cpp          # C2R: Inverse vs Ref               (NEW)
├── test_exec_d2z.cpp          # D2Z: Forward vs Ref               (NEW)
├── test_exec_z2d.cpp          # Z2D: Inverse vs Ref               (NEW)
├── test_exec_r2c_c2r.cpp      # R2C<->C2R Roundtrip                (existing, refactored)
├── test_exec_d2z_z2d.cpp      # D2Z<->Z2D Roundtrip                (existing, refactored)
├── test_plan.cpp              # Plan and unsupported-rank contract tests
├── main.cpp                   # Test runner                        (unchanged)
└── CMakeLists.txt             # Updated target list
```

Original execution files are rewritten as 1D-only. `Plan2D` and `Plan3D`
remain present as contract tests: the current API must return
`FLAGFFT_NOT_SUPPORTED` for all transform types until multidimensional plans
are implemented.

## Test Registry (shared in flagfft_test.h)

```cpp
constexpr int k1DSizesSmall[]   = {16, 23, 64, 81};
constexpr int k1DSizesMedium[]  = {243, 256, 361, 512, 997};
constexpr int k1DSizesLarge[]   = {2048, 4096, 8192, 16384};
constexpr int k1DBatchValues[]  = {1, 4, 256};
```

Sizes and batch values are defined in one place. Adding/removing entries is a single-line change.

## Size Coverage Rationale

| Size | Factorization | Code Path | Rationale |
|------|--------------|-----------|-----------|
| 16   | 2^4          | DirectDFT / Leaf | Smallest pure butterfly |
| 23   | prime        | **DirectDFT** | Small prime >19, no codelet |
| 64   | 2^6          | DirectDFT boundary | Upper bound of DirectDFT |
| 81   | 3^4          | Leaf | Non-power-of-2, codelet-3 only |
| 243  | 3^5          | Leaf | Non-power-of-2 (existing test) |
| 256  | 2^8          | Leaf | Classic pow2 baseline |
| 361  | 19^2         | Leaf | Largest codelet (19 compromise) |
| 512  | 2^9          | Leaf | Mid-large pow2 |
| 997  | prime        | **Bluestein** | Large prime, exercises Bluestein fallback |
| 2048 | 2^11         | Leaf | Large leaf |
| 4096 | 2^12         | Leaf boundary | kLeafMaxN limit |
| 8192 | 2^13         | **FourStep** | First size exceeding leaf range |
| 16384| 2^14         | Specialized FourStep | Hardcoded 64x256 decomposition |

## Batch Values

`{1, 4, 256}` — single, small batch, and large batch. Combined with all 13 sizes via cartesian product.

## Test Dimensions per File

### C2C / Z2Z (3 TEST_P suites x 39 params = 117 cases)

| Suite | Validation |
|-------|-----------|
| `ForwardVsReference` | Compare with same-precision cuFFT using normwise `rel_l2` and `rel_linf` |
| `InverseVsReference` | Compare with same-precision cuFFT using normwise `rel_l2` and `rel_linf` |
| `Roundtrip` | F -> I, verify output = N * input using the same normwise metric |

### R2C / D2Z (1 TEST_P suite x 39 params = 39 cases)

| Suite | Validation |
|-------|-----------|
| `ForwardVsReference` | Compare with same-precision cuFFT using normwise metrics |

### C2R / Z2D (1 TEST_P suite x 39 params = 39 cases)

| Suite | Validation |
|-------|-----------|
| `InverseVsReference` | Compare with same-precision cuFFT using normwise metrics |

### Roundtrip files (1 TEST_P suite x 39 params = 39 cases)

| Suite | Validation |
|-------|-----------|
| `Roundtrip1D` | F -> I, verify output = N * input using normwise metrics |

## TEST_P Fixture Design

Each file defines one fixture class inheriting `TestWithParam<Test1DParam>`:

```cpp
struct Test1DParam { int N; int batch; };

class C2C_1D_Test : public ::testing::TestWithParam<Test1DParam> {
protected:
  void SetUp() override;    // create plan, alloc, generate data, copy to device
  void TearDown() override; // free, destroy
  // shared state: N, batch, total, plan, device pointers, host vectors
};
```

All parameter cases are compiled and registered. Instantiation separates a
representative quick subset from the remainder solely for runtime filtering:

```cpp
INSTANTIATE_TEST_SUITE_P(Smoke,          C2C_1D_Test, Generate1DParamsSmoke());
INSTANTIATE_TEST_SUITE_P(ExtendedSmall,  C2C_1D_Test, Generate1DParamsExtendedSmall());
INSTANTIATE_TEST_SUITE_P(ExtendedMedium, C2C_1D_Test, Generate1DParamsExtendedMedium());
INSTANTIATE_TEST_SUITE_P(ExtendedLarge,  C2C_1D_Test, Generate1DParamsExtendedLarge());
```

`Smoke` consists of representative Direct, Bluestein, and FourStep cases.
The Extended groups contain the rest of the original cartesian product.
CTest registers one invocation per test executable rather than one invocation
per discovered Google Test case, because the GPU/JIT initialization must be
shared across parameters. A plain `ctest --test-dir build` executes the
complete matrix once per executable. Quick local validation applies a runtime
filter without removing any compiled case:

```sh
FLAGFFT_TEST_PROFILE=smoke ctest --test-dir build
```

## Numerical Acceptance Standard

Product conformance is alignment with cuFFT at the same input and output
precision: float FlagFFT is compared with cuFFT float and double FlagFFT with
cuFFT double. A float-to-double high-precision diagnostic may be used for
investigation, but is not the pass/fail oracle. Both promotions in that
diagnostic are bit-exact; its remaining reference uncertainty is at double
arithmetic scale.

For each batch independently:

```text
e_i       = flagfft_i - cufft_i
rel_l2    = sqrt(sum |e_i|^2) / sqrt(sum |cufft_i|^2)
rel_linf  = max |e_i| / max |cufft_i|
u(dtype)  = epsilon(dtype) / 2
M(N)      = ceil_power_of_two(2*N - 1)
work(N)   = N                         , N <= 64
          = 3 * ceil(log2(M(N))) + 3 , N > 64
limit     = C(class, metric) * u(dtype) * work(N)
```

Complex norms use complex magnitudes, not independent component ratios.
Batch maxima are gated so that a large batch does not hide a failing
transform. A pointwise denominator floor of `max(|ref|, 1)` is retained only
for diagnostic messages; zero input is tested as an exact-zero behavior.

The work-factor rationale is:

- For `N <= 64`, the implementation can use direct DFT summation, whose
  rounding accumulation is modeled as `O(N*u)` rather than `O(log(N)*u)`.
- For `N > 64`, the length-only budget deliberately envelopes a worst-case
  Bluestein route. Bluestein requires a convolution of at least `2*N-1`;
  FlagFFT's supported-length selection does not exceed `M(N)`, and the raw
  fallback path uses `M(N)` directly.
- Bluestein runs three child FFT contributions visible in the output
  (precomputed chirp FFT, input FFT, inverse convolution FFT) and three
  pointwise complex-multiplication stages (prepare, convolution product,
  finalize), producing the `3*log2(M)+3` model.
- The `O(u log N)` FFT-stage basis follows Higham, *Accuracy and Stability of
  Numerical Algorithms*, 2nd ed., Chapter 24. The coefficients `3` and `+3`
  are FlagFFT execution-graph accounting, not constants quoted from Higham.

Inputs use a deterministic generator and each reference case is evaluated at
scales `2^-20`, `1`, and `2^20`; power-of-two scaling leaves the represented
signal exact while checking that the acceptance metric is not magnitude
dependent. Constants are shared by float/double within each transform class
and were frozen with margin `K=2` from this 936-measurement reference matrix
on NVIDIA A100-SXM4-40GB with CUDA/cuFFT 13.2:

| Transform class | APIs | Max normalized `rel_l2` | `C_l2` | Max normalized `rel_linf` | `C_linf` |
|---|---|---:|---:|---:|---:|
| Complex | C2C, Z2Z | 0.6209693273 | 1.2419386546 | 0.9671984544 | 1.9343969088 |
| Real forward | R2C, D2Z | 0.6173405002 | 1.2346810004 | 0.9130279098 | 1.8260558196 |
| Real inverse | C2R, Z2D | 0.4886148521 | 0.9772297042 | 0.6860913487 | 1.3721826973 |

The class peaks currently occur on double paths, so sharing constants remains
conservative for float. A future split by precision requires repeated
characterization showing a normalized ratio greater than two and a documented
precision-specific implementation reason.

## CMakeLists.txt Changes

Replace the 4 original exec targets with 8:

```cmake
set(TEST_TARGETS
    test_plan
    test_exec_c2c
    test_exec_z2z
    test_exec_r2c
    test_exec_c2r
    test_exec_d2z
    test_exec_z2d
    test_exec_r2c_c2r
    test_exec_d2z_z2d
)
```

## Out of Scope

- Implementing 2D/3D transforms; tests cover their current unsupported contract
- Python tests (`tests/`)
- CLI tolerance policy; the CLI is scheduled for removal
- CLI tests
