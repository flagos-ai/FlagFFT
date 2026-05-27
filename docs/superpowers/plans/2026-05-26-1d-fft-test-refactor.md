# 1D FFT Test Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor ctest/ 1D FFT tests: one file per transform type, TEST_P parameterization over (size, batch), shared registry in flagfft_test.h.

**Architecture:** Shared header `flagfft_test.h` holds `Test1DParam` struct, size/batch registry arrays, and `Generate1DParams*()` cartesian-product helpers. Each of 8 exec test files defines a fixture subclass of `TestWithParam<Test1DParam>` with 1-3 TEST_P suites. Fixture manages plan creation, device memory, random data in SetUp/TearDown. Tests are ~5-line comparisons against cuFFT reference.

**Tech Stack:** C++20, Google Test (TEST_P), CUDA cuFFT (reference), flagfft C API.

---

### Task 1: Update flagfft_test.h — Add Test1DParam, registry, and param generators

**Files:**
- Modify: `ctest/flagfft_test.h`

- [ ] **Step 1: Add Test1DParam struct, registry, and Generate1DParams helpers**

Add the following after the `random_double_real` function (before the closing `}  // namespace flagfft_test::adaptor`):

```cpp
// =========================================================================
// 1D Test parameterization
// =========================================================================

struct Test1DParam {
  int N;
  int batch;
};

// Registry — add/remove sizes and batch values here
constexpr int k1DSizesSmall[]   = {16, 23, 64, 81};
constexpr int k1DSizesMedium[]  = {243, 256, 361, 512, 997};
constexpr int k1DSizesLarge[]   = {2048, 4096, 8192, 16384};
constexpr int k1DBatchValues[]  = {1, 4, 256};

inline std::vector<Test1DParam> Generate1DParams(const int* sizes, int numSizes) {
  std::vector<Test1DParam> params;
  for (int i = 0; i < numSizes; ++i)
    for (int b : k1DBatchValues)
      params.push_back({sizes[i], b});
  return params;
}

inline std::vector<Test1DParam> Generate1DParamsSmall() {
  return Generate1DParams(k1DSizesSmall, sizeof(k1DSizesSmall) / sizeof(k1DSizesSmall[0]));
}
inline std::vector<Test1DParam> Generate1DParamsMedium() {
  return Generate1DParams(k1DSizesMedium, sizeof(k1DSizesMedium) / sizeof(k1DSizesMedium[0]));
}
inline std::vector<Test1DParam> Generate1DParamsLarge() {
  return Generate1DParams(k1DSizesLarge, sizeof(k1DSizesLarge) / sizeof(k1DSizesLarge[0]));
}
```

**No commit yet** — we'll commit after Task 2 when we have a compilable baseline.

---

### Task 2: Rewrite test_exec_c2c.cpp — C2C 1D with TEST_P

**Files:**
- Rewrite: `ctest/test_exec_c2c.cpp`

- [ ] **Step 1: Write the new test_exec_c2c.cpp**

```cpp
#include "flagfft_test.h"

using namespace flagfft_test::adaptor;

constexpr double kRelTol = 1e-4;

class C2C_1D_Test : public ::testing::TestWithParam<Test1DParam> {
protected:
  void SetUp() override {
    auto p = GetParam();
    N = p.N;
    batch = p.batch;
    total = N * batch;

    plan = nullptr;
    Plan1d(&plan, N, FLAGFFT_C2C, batch);

    h_in = random_complex(total);

    auto bytes = total * sizeof(flagfftComplex);
    d_in  = static_cast<flagfftComplex*>(allocate_device(bytes));
    d_out = static_cast<flagfftComplex*>(allocate_device(bytes));
    d_aux = static_cast<flagfftComplex*>(allocate_device(bytes));
    ASSERT_NE(d_in, nullptr);
    ASSERT_NE(d_out, nullptr);
    ASSERT_NE(d_aux, nullptr);

    copy_host_to_device(h_in.data(), d_in, bytes);
  }

  void TearDown() override {
    if (d_in)  free_device(d_in);
    if (d_out) free_device(d_out);
    if (d_aux) free_device(d_aux);
    if (plan)  flagfftDestroy(plan);
  }

  int N = 0;
  int batch = 0;
  int total = 0;
  flagfftHandle plan = nullptr;
  std::vector<flagfftComplex> h_in;
  flagfftComplex* d_in = nullptr;
  flagfftComplex* d_out = nullptr;
  flagfftComplex* d_aux = nullptr;
};

TEST_P(C2C_1D_Test, ForwardVsReference) {
  ExecC2C(plan, d_in, d_out, FLAGFFT_FORWARD);

  RefHandle ref;
  ref_plan_1d(ref, N, FLAGFFT_C2C, batch);
  ref_exec_c2c(ref, d_in, d_aux, FLAGFFT_FORWARD);

  std::vector<flagfftComplex> h_out(total);
  std::vector<flagfftComplex> h_ref(total);
  copy_device_to_host(d_out, h_out.data(), total * sizeof(flagfftComplex));
  copy_device_to_host(d_aux, h_ref.data(), total * sizeof(flagfftComplex));

  double max_err = max_relative_error(h_out.data(), h_ref.data(), total);
  EXPECT_LT(max_err, kRelTol) << "N=" << N << " batch=" << batch << " max relative error: " << max_err;
}

TEST_P(C2C_1D_Test, InverseVsReference) {
  ExecC2C(plan, d_in, d_out, FLAGFFT_INVERSE);

  RefHandle ref;
  ref_plan_1d(ref, N, FLAGFFT_C2C, batch);
  ref_exec_c2c(ref, d_in, d_aux, FLAGFFT_INVERSE);

  std::vector<flagfftComplex> h_out(total);
  std::vector<flagfftComplex> h_ref(total);
  copy_device_to_host(d_out, h_out.data(), total * sizeof(flagfftComplex));
  copy_device_to_host(d_aux, h_ref.data(), total * sizeof(flagfftComplex));

  double max_err = max_relative_error(h_out.data(), h_ref.data(), total);
  EXPECT_LT(max_err, kRelTol) << "N=" << N << " batch=" << batch << " max relative error: " << max_err;
}

TEST_P(C2C_1D_Test, Roundtrip) {
  ExecC2C(plan, d_in, d_aux, FLAGFFT_FORWARD);
  ExecC2C(plan, d_aux, d_out, FLAGFFT_INVERSE);

  std::vector<flagfftComplex> h_out(total);
  copy_device_to_host(d_out, h_out.data(), total * sizeof(flagfftComplex));

  for (int i = 0; i < total; ++i) {
    double expected_x = h_in[i].x * N;
    double expected_y = h_in[i].y * N;
    EXPECT_NEAR(h_out[i].x, expected_x, N * kRelTol)
        << "N=" << N << " batch=" << batch << " i=" << i << " (real)";
    EXPECT_NEAR(h_out[i].y, expected_y, N * kRelTol)
        << "N=" << N << " batch=" << batch << " i=" << i << " (imag)";
  }
}

INSTANTIATE_TEST_SUITE_P(Small,  C2C_1D_Test, ::testing::ValuesIn(Generate1DParamsSmall()));
INSTANTIATE_TEST_SUITE_P(Medium, C2C_1D_Test, ::testing::ValuesIn(Generate1DParamsMedium()));
INSTANTIATE_TEST_SUITE_P(Large,  C2C_1D_Test, ::testing::ValuesIn(Generate1DParamsLarge()));
```

- [ ] **Step 2: Commit**

```bash
git add ctest/flagfft_test.h ctest/test_exec_c2c.cpp
git commit -m "refactor: add 1D test registry and rewrite C2C exec tests with TEST_P"
```

---

### Task 3: Rewrite test_exec_z2z.cpp — Z2Z 1D with TEST_P

**Files:**
- Rewrite: `ctest/test_exec_z2z.cpp`

- [ ] **Step 1: Write the new test_exec_z2z.cpp**

```cpp
#include "flagfft_test.h"

using namespace flagfft_test::adaptor;

constexpr double kRelTol = 1e-10;

class Z2Z_1D_Test : public ::testing::TestWithParam<Test1DParam> {
protected:
  void SetUp() override {
    auto p = GetParam();
    N = p.N;
    batch = p.batch;
    total = N * batch;

    plan = nullptr;
    Plan1d(&plan, N, FLAGFFT_Z2Z, batch);

    h_in = random_double_complex(total);

    auto bytes = total * sizeof(flagfftDoubleComplex);
    d_in  = static_cast<flagfftDoubleComplex*>(allocate_device(bytes));
    d_out = static_cast<flagfftDoubleComplex*>(allocate_device(bytes));
    d_aux = static_cast<flagfftDoubleComplex*>(allocate_device(bytes));
    ASSERT_NE(d_in, nullptr);
    ASSERT_NE(d_out, nullptr);
    ASSERT_NE(d_aux, nullptr);

    copy_host_to_device(h_in.data(), d_in, bytes);
  }

  void TearDown() override {
    if (d_in)  free_device(d_in);
    if (d_out) free_device(d_out);
    if (d_aux) free_device(d_aux);
    if (plan)  flagfftDestroy(plan);
  }

  int N = 0;
  int batch = 0;
  int total = 0;
  flagfftHandle plan = nullptr;
  std::vector<flagfftDoubleComplex> h_in;
  flagfftDoubleComplex* d_in = nullptr;
  flagfftDoubleComplex* d_out = nullptr;
  flagfftDoubleComplex* d_aux = nullptr;
};

TEST_P(Z2Z_1D_Test, ForwardVsReference) {
  ExecZ2Z(plan, d_in, d_out, FLAGFFT_FORWARD);

  RefHandle ref;
  ref_plan_1d(ref, N, FLAGFFT_Z2Z, batch);
  ref_exec_z2z(ref, d_in, d_aux, FLAGFFT_FORWARD);

  std::vector<flagfftDoubleComplex> h_out(total);
  std::vector<flagfftDoubleComplex> h_ref(total);
  copy_device_to_host(d_out, h_out.data(), total * sizeof(flagfftDoubleComplex));
  copy_device_to_host(d_aux, h_ref.data(), total * sizeof(flagfftDoubleComplex));

  double max_err = max_relative_error(h_out.data(), h_ref.data(), total);
  EXPECT_LT(max_err, kRelTol) << "N=" << N << " batch=" << batch << " max relative error: " << max_err;
}

TEST_P(Z2Z_1D_Test, InverseVsReference) {
  ExecZ2Z(plan, d_in, d_out, FLAGFFT_INVERSE);

  RefHandle ref;
  ref_plan_1d(ref, N, FLAGFFT_Z2Z, batch);
  ref_exec_z2z(ref, d_in, d_aux, FLAGFFT_INVERSE);

  std::vector<flagfftDoubleComplex> h_out(total);
  std::vector<flagfftDoubleComplex> h_ref(total);
  copy_device_to_host(d_out, h_out.data(), total * sizeof(flagfftDoubleComplex));
  copy_device_to_host(d_aux, h_ref.data(), total * sizeof(flagfftDoubleComplex));

  double max_err = max_relative_error(h_out.data(), h_ref.data(), total);
  EXPECT_LT(max_err, kRelTol) << "N=" << N << " batch=" << batch << " max relative error: " << max_err;
}

TEST_P(Z2Z_1D_Test, Roundtrip) {
  ExecZ2Z(plan, d_in, d_aux, FLAGFFT_FORWARD);
  ExecZ2Z(plan, d_aux, d_out, FLAGFFT_INVERSE);

  std::vector<flagfftDoubleComplex> h_out(total);
  copy_device_to_host(d_out, h_out.data(), total * sizeof(flagfftDoubleComplex));

  for (int i = 0; i < total; ++i) {
    double expected_x = h_in[i].x * N;
    double expected_y = h_in[i].y * N;
    EXPECT_NEAR(h_out[i].x, expected_x, N * kRelTol)
        << "N=" << N << " batch=" << batch << " i=" << i << " (real)";
    EXPECT_NEAR(h_out[i].y, expected_y, N * kRelTol)
        << "N=" << N << " batch=" << batch << " i=" << i << " (imag)";
  }
}

INSTANTIATE_TEST_SUITE_P(Small,  Z2Z_1D_Test, ::testing::ValuesIn(Generate1DParamsSmall()));
INSTANTIATE_TEST_SUITE_P(Medium, Z2Z_1D_Test, ::testing::ValuesIn(Generate1DParamsMedium()));
INSTANTIATE_TEST_SUITE_P(Large,  Z2Z_1D_Test, ::testing::ValuesIn(Generate1DParamsLarge()));
```

- [ ] **Step 2: Commit**

```bash
git add ctest/test_exec_z2z.cpp
git commit -m "refactor: rewrite Z2Z exec tests with TEST_P"
```

---

### Task 4: Create test_exec_r2c.cpp — R2C Forward vs Reference

**Files:**
- Create: `ctest/test_exec_r2c.cpp`

- [ ] **Step 1: Write test_exec_r2c.cpp**

```cpp
#include "flagfft_test.h"

using namespace flagfft_test::adaptor;

constexpr double kRelTol = 1e-4;

class R2C_1D_Test : public ::testing::TestWithParam<Test1DParam> {
protected:
  void SetUp() override {
    auto p = GetParam();
    N = p.N;
    batch = p.batch;
    total_in = N * batch;
    total_out = (N / 2 + 1) * batch;

    plan = nullptr;
    Plan1d(&plan, N, FLAGFFT_R2C, batch);

    h_in = random_real(total_in);

    d_in  = static_cast<flagfftReal*>(allocate_device(total_in * sizeof(flagfftReal)));
    d_out = static_cast<flagfftComplex*>(allocate_device(total_out * sizeof(flagfftComplex)));
    d_ref = static_cast<flagfftComplex*>(allocate_device(total_out * sizeof(flagfftComplex)));
    ASSERT_NE(d_in, nullptr);
    ASSERT_NE(d_out, nullptr);
    ASSERT_NE(d_ref, nullptr);

    copy_host_to_device(h_in.data(), d_in, total_in * sizeof(flagfftReal));
  }

  void TearDown() override {
    if (d_in)  free_device(d_in);
    if (d_out) free_device(d_out);
    if (d_ref) free_device(d_ref);
    if (plan)  flagfftDestroy(plan);
  }

  int N = 0;
  int batch = 0;
  int total_in = 0;
  int total_out = 0;
  flagfftHandle plan = nullptr;
  std::vector<flagfftReal> h_in;
  flagfftReal* d_in = nullptr;
  flagfftComplex* d_out = nullptr;
  flagfftComplex* d_ref = nullptr;
};

TEST_P(R2C_1D_Test, ForwardVsReference) {
  ExecR2C(plan, d_in, d_out);

  RefHandle ref;
  ref_plan_1d(ref, N, FLAGFFT_R2C, batch);
  ref_exec_r2c(ref, d_in, d_ref);

  std::vector<flagfftComplex> h_out(total_out);
  std::vector<flagfftComplex> h_ref_out(total_out);
  copy_device_to_host(d_out, h_out.data(), total_out * sizeof(flagfftComplex));
  copy_device_to_host(d_ref, h_ref_out.data(), total_out * sizeof(flagfftComplex));

  double max_err = max_relative_error(h_out.data(), h_ref_out.data(), total_out);
  EXPECT_LT(max_err, kRelTol) << "N=" << N << " batch=" << batch << " max relative error: " << max_err;
}

INSTANTIATE_TEST_SUITE_P(Small,  R2C_1D_Test, ::testing::ValuesIn(Generate1DParamsSmall()));
INSTANTIATE_TEST_SUITE_P(Medium, R2C_1D_Test, ::testing::ValuesIn(Generate1DParamsMedium()));
INSTANTIATE_TEST_SUITE_P(Large,  R2C_1D_Test, ::testing::ValuesIn(Generate1DParamsLarge()));
```

- [ ] **Step 2: Commit**

```bash
git add ctest/test_exec_r2c.cpp
git commit -m "refactor: add R2C forward-vs-reference 1D tests with TEST_P"
```

---

### Task 5: Create test_exec_c2r.cpp — C2R Inverse vs Reference

**Files:**
- Create: `ctest/test_exec_c2r.cpp`

- [ ] **Step 1: Write test_exec_c2r.cpp**

```cpp
#include "flagfft_test.h"

using namespace flagfft_test::adaptor;

constexpr double kRelTol = 1e-4;

class C2R_1D_Test : public ::testing::TestWithParam<Test1DParam> {
protected:
  void SetUp() override {
    auto p = GetParam();
    N = p.N;
    batch = p.batch;
    total_in = (N / 2 + 1) * batch;
    total_out = N * batch;

    plan = nullptr;
    Plan1d(&plan, N, FLAGFFT_C2R, batch);

    h_in = random_complex(total_in);

    d_in  = static_cast<flagfftComplex*>(allocate_device(total_in * sizeof(flagfftComplex)));
    d_out = static_cast<flagfftReal*>(allocate_device(total_out * sizeof(flagfftReal)));
    d_ref = static_cast<flagfftReal*>(allocate_device(total_out * sizeof(flagfftReal)));
    ASSERT_NE(d_in, nullptr);
    ASSERT_NE(d_out, nullptr);
    ASSERT_NE(d_ref, nullptr);

    copy_host_to_device(h_in.data(), d_in, total_in * sizeof(flagfftComplex));
  }

  void TearDown() override {
    if (d_in)  free_device(d_in);
    if (d_out) free_device(d_out);
    if (d_ref) free_device(d_ref);
    if (plan)  flagfftDestroy(plan);
  }

  int N = 0;
  int batch = 0;
  int total_in = 0;
  int total_out = 0;
  flagfftHandle plan = nullptr;
  std::vector<flagfftComplex> h_in;
  flagfftComplex* d_in = nullptr;
  flagfftReal* d_out = nullptr;
  flagfftReal* d_ref = nullptr;
};

TEST_P(C2R_1D_Test, InverseVsReference) {
  ExecC2R(plan, d_in, d_out);

  RefHandle ref;
  ref_plan_1d(ref, N, FLAGFFT_C2R, batch);
  ref_exec_c2r(ref, d_in, d_ref);

  std::vector<flagfftReal> h_out(total_out);
  std::vector<flagfftReal> h_ref_out(total_out);
  copy_device_to_host(d_out, h_out.data(), total_out * sizeof(flagfftReal));
  copy_device_to_host(d_ref, h_ref_out.data(), total_out * sizeof(flagfftReal));

  double max_err = max_relative_error_real(h_out.data(), h_ref_out.data(), total_out);
  EXPECT_LT(max_err, kRelTol) << "N=" << N << " batch=" << batch << " max relative error: " << max_err;
}

INSTANTIATE_TEST_SUITE_P(Small,  C2R_1D_Test, ::testing::ValuesIn(Generate1DParamsSmall()));
INSTANTIATE_TEST_SUITE_P(Medium, C2R_1D_Test, ::testing::ValuesIn(Generate1DParamsMedium()));
INSTANTIATE_TEST_SUITE_P(Large,  C2R_1D_Test, ::testing::ValuesIn(Generate1DParamsLarge()));
```

- [ ] **Step 2: Commit**

```bash
git add ctest/test_exec_c2r.cpp
git commit -m "refactor: add C2R inverse-vs-reference 1D tests with TEST_P"
```

---

### Task 6: Create test_exec_d2z.cpp — D2Z Forward vs Reference

**Files:**
- Create: `ctest/test_exec_d2z.cpp`

- [ ] **Step 1: Write test_exec_d2z.cpp**

```cpp
#include "flagfft_test.h"

using namespace flagfft_test::adaptor;

constexpr double kRelTol = 1e-10;

class D2Z_1D_Test : public ::testing::TestWithParam<Test1DParam> {
protected:
  void SetUp() override {
    auto p = GetParam();
    N = p.N;
    batch = p.batch;
    total_in = N * batch;
    total_out = (N / 2 + 1) * batch;

    plan = nullptr;
    Plan1d(&plan, N, FLAGFFT_D2Z, batch);

    h_in = random_double_real(total_in);

    d_in  = static_cast<flagfftDoubleReal*>(allocate_device(total_in * sizeof(flagfftDoubleReal)));
    d_out = static_cast<flagfftDoubleComplex*>(allocate_device(total_out * sizeof(flagfftDoubleComplex)));
    d_ref = static_cast<flagfftDoubleComplex*>(allocate_device(total_out * sizeof(flagfftDoubleComplex)));
    ASSERT_NE(d_in, nullptr);
    ASSERT_NE(d_out, nullptr);
    ASSERT_NE(d_ref, nullptr);

    copy_host_to_device(h_in.data(), d_in, total_in * sizeof(flagfftDoubleReal));
  }

  void TearDown() override {
    if (d_in)  free_device(d_in);
    if (d_out) free_device(d_out);
    if (d_ref) free_device(d_ref);
    if (plan)  flagfftDestroy(plan);
  }

  int N = 0;
  int batch = 0;
  int total_in = 0;
  int total_out = 0;
  flagfftHandle plan = nullptr;
  std::vector<flagfftDoubleReal> h_in;
  flagfftDoubleReal* d_in = nullptr;
  flagfftDoubleComplex* d_out = nullptr;
  flagfftDoubleComplex* d_ref = nullptr;
};

TEST_P(D2Z_1D_Test, ForwardVsReference) {
  ExecD2Z(plan, d_in, d_out);

  RefHandle ref;
  ref_plan_1d(ref, N, FLAGFFT_D2Z, batch);
  ref_exec_d2z(ref, d_in, d_ref);

  std::vector<flagfftDoubleComplex> h_out(total_out);
  std::vector<flagfftDoubleComplex> h_ref_out(total_out);
  copy_device_to_host(d_out, h_out.data(), total_out * sizeof(flagfftDoubleComplex));
  copy_device_to_host(d_ref, h_ref_out.data(), total_out * sizeof(flagfftDoubleComplex));

  double max_err = max_relative_error(h_out.data(), h_ref_out.data(), total_out);
  EXPECT_LT(max_err, kRelTol) << "N=" << N << " batch=" << batch << " max relative error: " << max_err;
}

INSTANTIATE_TEST_SUITE_P(Small,  D2Z_1D_Test, ::testing::ValuesIn(Generate1DParamsSmall()));
INSTANTIATE_TEST_SUITE_P(Medium, D2Z_1D_Test, ::testing::ValuesIn(Generate1DParamsMedium()));
INSTANTIATE_TEST_SUITE_P(Large,  D2Z_1D_Test, ::testing::ValuesIn(Generate1DParamsLarge()));
```

- [ ] **Step 2: Commit**

```bash
git add ctest/test_exec_d2z.cpp
git commit -m "refactor: add D2Z forward-vs-reference 1D tests with TEST_P"
```

---

### Task 7: Create test_exec_z2d.cpp — Z2D Inverse vs Reference

**Files:**
- Create: `ctest/test_exec_z2d.cpp`

- [ ] **Step 1: Write test_exec_z2d.cpp**

```cpp
#include "flagfft_test.h"

using namespace flagfft_test::adaptor;

constexpr double kRelTol = 1e-10;

class Z2D_1D_Test : public ::testing::TestWithParam<Test1DParam> {
protected:
  void SetUp() override {
    auto p = GetParam();
    N = p.N;
    batch = p.batch;
    total_in = (N / 2 + 1) * batch;
    total_out = N * batch;

    plan = nullptr;
    Plan1d(&plan, N, FLAGFFT_Z2D, batch);

    h_in = random_double_complex(total_in);

    d_in  = static_cast<flagfftDoubleComplex*>(allocate_device(total_in * sizeof(flagfftDoubleComplex)));
    d_out = static_cast<flagfftDoubleReal*>(allocate_device(total_out * sizeof(flagfftDoubleReal)));
    d_ref = static_cast<flagfftDoubleReal*>(allocate_device(total_out * sizeof(flagfftDoubleReal)));
    ASSERT_NE(d_in, nullptr);
    ASSERT_NE(d_out, nullptr);
    ASSERT_NE(d_ref, nullptr);

    copy_host_to_device(h_in.data(), d_in, total_in * sizeof(flagfftDoubleComplex));
  }

  void TearDown() override {
    if (d_in)  free_device(d_in);
    if (d_out) free_device(d_out);
    if (d_ref) free_device(d_ref);
    if (plan)  flagfftDestroy(plan);
  }

  int N = 0;
  int batch = 0;
  int total_in = 0;
  int total_out = 0;
  flagfftHandle plan = nullptr;
  std::vector<flagfftDoubleComplex> h_in;
  flagfftDoubleComplex* d_in = nullptr;
  flagfftDoubleReal* d_out = nullptr;
  flagfftDoubleReal* d_ref = nullptr;
};

TEST_P(Z2D_1D_Test, InverseVsReference) {
  ExecZ2D(plan, d_in, d_out);

  RefHandle ref;
  ref_plan_1d(ref, N, FLAGFFT_Z2D, batch);
  ref_exec_z2d(ref, d_in, d_ref);

  std::vector<flagfftDoubleReal> h_out(total_out);
  std::vector<flagfftDoubleReal> h_ref_out(total_out);
  copy_device_to_host(d_out, h_out.data(), total_out * sizeof(flagfftDoubleReal));
  copy_device_to_host(d_ref, h_ref_out.data(), total_out * sizeof(flagfftDoubleReal));

  double max_err = max_relative_error_real(h_out.data(), h_ref_out.data(), total_out);
  EXPECT_LT(max_err, kRelTol) << "N=" << N << " batch=" << batch << " max relative error: " << max_err;
}

INSTANTIATE_TEST_SUITE_P(Small,  Z2D_1D_Test, ::testing::ValuesIn(Generate1DParamsSmall()));
INSTANTIATE_TEST_SUITE_P(Medium, Z2D_1D_Test, ::testing::ValuesIn(Generate1DParamsMedium()));
INSTANTIATE_TEST_SUITE_P(Large,  Z2D_1D_Test, ::testing::ValuesIn(Generate1DParamsLarge()));
```

- [ ] **Step 2: Commit**

```bash
git add ctest/test_exec_z2d.cpp
git commit -m "refactor: add Z2D inverse-vs-reference 1D tests with TEST_P"
```

---

### Task 8: Rewrite test_exec_r2c_c2r.cpp — R2C<->C2R Roundtrip with TEST_P

**Files:**
- Rewrite: `ctest/test_exec_r2c_c2r.cpp`

- [ ] **Step 1: Write the new test_exec_r2c_c2r.cpp**

```cpp
#include "flagfft_test.h"

using namespace flagfft_test::adaptor;

constexpr double kRelTol = 1e-4;

class R2C_C2R_Roundtrip_Test : public ::testing::TestWithParam<Test1DParam> {
protected:
  void SetUp() override {
    auto p = GetParam();
    N = p.N;
    batch = p.batch;
    total_real = N * batch;
    total_complex = (N / 2 + 1) * batch;

    plan_fwd = nullptr;
    plan_inv = nullptr;
    Plan1d(&plan_fwd, N, FLAGFFT_R2C, batch);
    Plan1d(&plan_inv, N, FLAGFFT_C2R, batch);

    h_in = random_real(total_real);

    d_in  = static_cast<flagfftReal*>(allocate_device(total_real * sizeof(flagfftReal)));
    d_mid = static_cast<flagfftComplex*>(allocate_device(total_complex * sizeof(flagfftComplex)));
    d_out = static_cast<flagfftReal*>(allocate_device(total_real * sizeof(flagfftReal)));
    ASSERT_NE(d_in, nullptr);
    ASSERT_NE(d_mid, nullptr);
    ASSERT_NE(d_out, nullptr);

    copy_host_to_device(h_in.data(), d_in, total_real * sizeof(flagfftReal));
  }

  void TearDown() override {
    if (d_in)  free_device(d_in);
    if (d_mid) free_device(d_mid);
    if (d_out) free_device(d_out);
    if (plan_fwd) flagfftDestroy(plan_fwd);
    if (plan_inv) flagfftDestroy(plan_inv);
  }

  int N = 0;
  int batch = 0;
  int total_real = 0;
  int total_complex = 0;
  flagfftHandle plan_fwd = nullptr;
  flagfftHandle plan_inv = nullptr;
  std::vector<flagfftReal> h_in;
  flagfftReal* d_in = nullptr;
  flagfftComplex* d_mid = nullptr;
  flagfftReal* d_out = nullptr;
};

TEST_P(R2C_C2R_Roundtrip_Test, Roundtrip1D) {
  ExecR2C(plan_fwd, d_in, d_mid);
  ExecC2R(plan_inv, d_mid, d_out);

  std::vector<flagfftReal> h_out(total_real);
  copy_device_to_host(d_out, h_out.data(), total_real * sizeof(flagfftReal));

  for (int i = 0; i < total_real; ++i) {
    double expected = static_cast<double>(h_in[i]) * N;
    EXPECT_NEAR(static_cast<double>(h_out[i]), expected, N * kRelTol)
        << "N=" << N << " batch=" << batch << " i=" << i;
  }
}

INSTANTIATE_TEST_SUITE_P(Small,  R2C_C2R_Roundtrip_Test, ::testing::ValuesIn(Generate1DParamsSmall()));
INSTANTIATE_TEST_SUITE_P(Medium, R2C_C2R_Roundtrip_Test, ::testing::ValuesIn(Generate1DParamsMedium()));
INSTANTIATE_TEST_SUITE_P(Large,  R2C_C2R_Roundtrip_Test, ::testing::ValuesIn(Generate1DParamsLarge()));
```

- [ ] **Step 2: Commit**

```bash
git add ctest/test_exec_r2c_c2r.cpp
git commit -m "refactor: rewrite R2C<->C2R roundtrip 1D tests with TEST_P"
```

---

### Task 9: Rewrite test_exec_d2z_z2d.cpp — D2Z<->Z2D Roundtrip with TEST_P

**Files:**
- Rewrite: `ctest/test_exec_d2z_z2d.cpp`

- [ ] **Step 1: Write the new test_exec_d2z_z2d.cpp**

```cpp
#include "flagfft_test.h"

using namespace flagfft_test::adaptor;

constexpr double kRelTol = 1e-10;

class D2Z_Z2D_Roundtrip_Test : public ::testing::TestWithParam<Test1DParam> {
protected:
  void SetUp() override {
    auto p = GetParam();
    N = p.N;
    batch = p.batch;
    total_real = N * batch;
    total_complex = (N / 2 + 1) * batch;

    plan_fwd = nullptr;
    plan_inv = nullptr;
    Plan1d(&plan_fwd, N, FLAGFFT_D2Z, batch);
    Plan1d(&plan_inv, N, FLAGFFT_Z2D, batch);

    h_in = random_double_real(total_real);

    d_in  = static_cast<flagfftDoubleReal*>(allocate_device(total_real * sizeof(flagfftDoubleReal)));
    d_mid = static_cast<flagfftDoubleComplex*>(allocate_device(total_complex * sizeof(flagfftDoubleComplex)));
    d_out = static_cast<flagfftDoubleReal*>(allocate_device(total_real * sizeof(flagfftDoubleReal)));
    ASSERT_NE(d_in, nullptr);
    ASSERT_NE(d_mid, nullptr);
    ASSERT_NE(d_out, nullptr);

    copy_host_to_device(h_in.data(), d_in, total_real * sizeof(flagfftDoubleReal));
  }

  void TearDown() override {
    if (d_in)  free_device(d_in);
    if (d_mid) free_device(d_mid);
    if (d_out) free_device(d_out);
    if (plan_fwd) flagfftDestroy(plan_fwd);
    if (plan_inv) flagfftDestroy(plan_inv);
  }

  int N = 0;
  int batch = 0;
  int total_real = 0;
  int total_complex = 0;
  flagfftHandle plan_fwd = nullptr;
  flagfftHandle plan_inv = nullptr;
  std::vector<flagfftDoubleReal> h_in;
  flagfftDoubleReal* d_in = nullptr;
  flagfftDoubleComplex* d_mid = nullptr;
  flagfftDoubleReal* d_out = nullptr;
};

TEST_P(D2Z_Z2D_Roundtrip_Test, Roundtrip1D) {
  ExecD2Z(plan_fwd, d_in, d_mid);
  ExecZ2D(plan_inv, d_mid, d_out);

  std::vector<flagfftDoubleReal> h_out(total_real);
  copy_device_to_host(d_out, h_out.data(), total_real * sizeof(flagfftDoubleReal));

  for (int i = 0; i < total_real; ++i) {
    double expected = h_in[i] * N;
    EXPECT_NEAR(h_out[i], expected, N * kRelTol)
        << "N=" << N << " batch=" << batch << " i=" << i;
  }
}

INSTANTIATE_TEST_SUITE_P(Small,  D2Z_Z2D_Roundtrip_Test, ::testing::ValuesIn(Generate1DParamsSmall()));
INSTANTIATE_TEST_SUITE_P(Medium, D2Z_Z2D_Roundtrip_Test, ::testing::ValuesIn(Generate1DParamsMedium()));
INSTANTIATE_TEST_SUITE_P(Large,  D2Z_Z2D_Roundtrip_Test, ::testing::ValuesIn(Generate1DParamsLarge()));
```

- [ ] **Step 2: Commit**

```bash
git add ctest/test_exec_d2z_z2d.cpp
git commit -m "refactor: rewrite D2Z<->Z2D roundtrip 1D tests with TEST_P"
```

---

### Task 10: Update CMakeLists.txt — Add 4 new targets

**Files:**
- Modify: `ctest/CMakeLists.txt`

- [ ] **Step 1: Update the TEST_TARGETS list**

Change the `set(TEST_TARGETS ...)` block from:

```cmake
set(TEST_TARGETS
    test_plan
    test_exec_c2c
    test_exec_z2z
    test_exec_r2c_c2r
    test_exec_d2z_z2d
)
```

To:

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

- [ ] **Step 2: Commit**

```bash
git add ctest/CMakeLists.txt
git commit -m "build: add 4 new 1D exec test targets to CMakeLists.txt"
```

---

### Task 11: Build and verify compilation

**Files:**
- (No source changes — verification only)

- [ ] **Step 1: Configure and build the test targets**

```bash
cd /workspace/FlagFFT-dev-1d-test
cmake -B build -DFLAGFFT_BUILD_TESTS=ON -DFLAGFFT_BUILD_CLI=OFF
cmake --build build --target test_exec_c2c test_exec_z2z test_exec_r2c test_exec_c2r test_exec_d2z test_exec_z2d test_exec_r2c_c2r test_exec_d2z_z2d test_plan -j$(nproc)
```

Expected: Zero compile errors, all 9 targets built successfully.

- [ ] **Step 2: Commit if build succeeds (no source changes — skip)**

(Already committed in previous tasks.)

---

## Summary

| Task | File | Action | Commits |
|------|------|--------|---------|
| 1 | `flagfft_test.h` | Add registry + helpers | Combined with Task 2 |
| 2 | `test_exec_c2c.cpp` | Rewrite, 1D TEST_P | 1 commit |
| 3 | `test_exec_z2z.cpp` | Rewrite, 1D TEST_P | 1 commit |
| 4 | `test_exec_r2c.cpp` | Create | 1 commit |
| 5 | `test_exec_c2r.cpp` | Create | 1 commit |
| 6 | `test_exec_d2z.cpp` | Create | 1 commit |
| 7 | `test_exec_z2d.cpp` | Create | 1 commit |
| 8 | `test_exec_r2c_c2r.cpp` | Rewrite, 1D TEST_P | 1 commit |
| 9 | `test_exec_d2z_z2d.cpp` | Rewrite, 1D TEST_P | 1 commit |
| 10 | `CMakeLists.txt` | Add 4 targets | 1 commit |
| 11 | Build | Verify | (no commit) |

**Total: 9 commits** (Tasks 1+2 combined), 11 tasks.
# Adaptor Integration Note

This plan records the original 1D test refactor. In the integrated tree, its
test cases and numerical policy are preserved, while device storage and
reference plans use `flagfft::adaptor::Memory` and
`flagfft::test_adaptor::RefPlanHandle` from `src/adaptor/`; the legacy
`flagfft_test::adaptor` allocation/reference examples below are historical.
