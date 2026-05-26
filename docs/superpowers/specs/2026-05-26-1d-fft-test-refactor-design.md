# 1D FFT Test Refactor Design

**Date:** 2026-05-26
**Status:** Approved

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
├── test_plan.cpp              # Plan tests                         (unchanged)
├── main.cpp                   # Test runner                        (unchanged)
└── CMakeLists.txt             # Updated target list
```

Original files (`test_exec_c2c.cpp`, `test_exec_z2z.cpp`, `test_exec_r2c_c2r.cpp`, `test_exec_d2z_z2d.cpp`) are rewritten as 1D-only. 2D/3D tests are removed for now (to be refactored separately in the same pattern later).

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
| `ForwardVsReference` | Compare with cuFFT, check max relative error < tolerance |
| `InverseVsReference` | Compare with cuFFT, check max relative error < tolerance |
| `Roundtrip` | F -> I, verify output = N * input within tolerance |

### R2C / D2Z (1 TEST_P suite x 39 params = 39 cases)

| Suite | Validation |
|-------|-----------|
| `ForwardVsReference` | Compare with cuFFT, check max relative error < tolerance |

### C2R / Z2D (1 TEST_P suite x 39 params = 39 cases)

| Suite | Validation |
|-------|-----------|
| `InverseVsReference` | Compare with cuFFT, check max relative error < tolerance |

### Roundtrip files (1 TEST_P suite x 39 params = 39 cases)

| Suite | Validation |
|-------|-----------|
| `Roundtrip1D` | F -> I, verify output = N * input within tolerance |

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

Instantiation groups by Small/Medium/Large for clear failure attribution:

```cpp
INSTANTIATE_TEST_SUITE_P(Small,  C2C_1D_Test, Generate1DParamsSmall());
INSTANTIATE_TEST_SUITE_P(Medium, C2C_1D_Test, Generate1DParamsMedium());
INSTANTIATE_TEST_SUITE_P(Large,  C2C_1D_Test, Generate1DParamsLarge());
```

`Generate1DParams*()` are shared helper functions in `flagfft_test.h` that compute the cartesian product of the corresponding size array with `k1DBatchValues`.

## Tolerances

| Precision | Types | Tolerance |
|-----------|-------|-----------|
| Single (float) | C2C, R2C, C2R | 1e-4 |
| Double | Z2Z, D2Z, Z2D | 1e-10 |

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

- 2D/3D tests (removed from existing files, to be refactored later)
- Python tests (`tests/`)
- `test_plan.cpp` (unchanged)
- CLI tests
