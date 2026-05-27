# Benchmark Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract test adapter (reference FFT oracle) from ctest into `src/adaptor/test_adaptor.h`, refactor ctest and bench.cpp to use it, and build Python benchmark infrastructure.

**Architecture:** Two-layer adapter under `src/adaptor/`. Layer 1 (`adaptor.h`) = FlagFFT runtime backend (exists). Layer 2 (`test_adaptor.h`) = reference FFT oracle for testing/benchmarking. CUDA backend implements both in `backend/cuda/`. Python layer mirrors `tests/conftest.py` fixture pattern.

**Tech Stack:** C++20, CUDA/cuFFT, CMake, Google Test, Python pytest, nlohmann-json

---

### Task 1: Create test_adaptor.h interface

**Files:**
- Create: `src/adaptor/test_adaptor.h`

- [ ] **Step 1: Create the header file**

```cpp
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "flagfft.h"

namespace flagfft::test_adaptor {

// =========================================================================
// RefPlanHandle — RAII wrapper for reference FFT plan (cuFFT / rocFFT / ...)
// =========================================================================

class RefPlanHandle {
 public:
  RefPlanHandle();
  ~RefPlanHandle();
  RefPlanHandle(RefPlanHandle&&) noexcept;
  RefPlanHandle& operator=(RefPlanHandle&&) noexcept;
  RefPlanHandle(const RefPlanHandle&) = delete;
  RefPlanHandle& operator=(const RefPlanHandle&) = delete;

  std::uintptr_t get() const;
  std::uintptr_t* ptr();

 private:
  std::uintptr_t impl_;
};

// =========================================================================
// Plan creation
// =========================================================================

void ref_plan_1d(RefPlanHandle& plan, int nx, flagfftType type, int batch);
void ref_plan_2d(RefPlanHandle& plan, int nx, int ny, flagfftType type);
void ref_plan_3d(RefPlanHandle& plan, int nx, int ny, int nz, flagfftType type);

// =========================================================================
// Plan execution
// =========================================================================

void ref_exec_c2c(RefPlanHandle& plan, flagfftComplex* idata, flagfftComplex* odata, int direction);
void ref_exec_z2z(RefPlanHandle& plan, flagfftDoubleComplex* idata, flagfftDoubleComplex* odata, int direction);
void ref_exec_r2c(RefPlanHandle& plan, flagfftReal* idata, flagfftComplex* odata);
void ref_exec_d2z(RefPlanHandle& plan, flagfftDoubleReal* idata, flagfftDoubleComplex* odata);
void ref_exec_c2r(RefPlanHandle& plan, flagfftComplex* idata, flagfftReal* odata);
void ref_exec_z2d(RefPlanHandle& plan, flagfftDoubleComplex* idata, flagfftDoubleReal* odata);

// =========================================================================
// Backend metadata
// =========================================================================

void initialize();
std::string backend_name();

// =========================================================================
// Data generation
// =========================================================================

std::vector<flagfftComplex> random_complex(int n);
std::vector<flagfftDoubleComplex> random_double_complex(int n);
std::vector<flagfftReal> random_real(int n);
std::vector<flagfftDoubleReal> random_double_real(int n);

// =========================================================================
// Correctness comparison
// =========================================================================

struct ErrorMetric {
  double max_abs = 0.0;
  double rms = 0.0;
};

double max_relative_error(const flagfftComplex* a, const flagfftComplex* b, int n);
double max_relative_error(const flagfftDoubleComplex* a, const flagfftDoubleComplex* b, int n);
double max_relative_error_real(const flagfftReal* a, const flagfftReal* b, int n);
double max_relative_error_real(const flagfftDoubleReal* a, const flagfftDoubleReal* b, int n);

ErrorMetric compute_error(const float* a, const float* b, std::size_t n);
ErrorMetric compute_error(const double* a, const double* b, std::size_t n);

}  // namespace flagfft::test_adaptor
```

- [ ] **Step 2: Verify the header compiles in isolation**

Run: Check that `src/adaptor/test_adaptor.h` includes only `flagfft.h` and standard library headers — no CUDA, no cuFFT, no Google Test.

---

### Task 2: Create CUDA test_adaptor.cpp implementation

**Files:**
- Create: `src/adaptor/backend/cuda/test_adaptor.cpp`

- [ ] **Step 1: Migrate RefPlanHandle implementation from ctest**

The implementation is moved verbatim from `ctest/backend/cuda/adaptor.cpp`, changing the namespace from `flagfft_test::adaptor` to `flagfft::test_adaptor`.

```cpp
#include "adaptor/test_adaptor.h"

#include <cuda_runtime_api.h>
#include <cufft.h>

#include <cstdio>
#include <cstdlib>

namespace flagfft::test_adaptor {

// =========================================================================
// RefHandle — cuFFT implementation
// =========================================================================

static cufftHandle to_cufft(std::uintptr_t v) {
  return static_cast<cufftHandle>(static_cast<std::intptr_t>(v));
}
static std::uintptr_t from_cufft(cufftHandle h) {
  return static_cast<std::uintptr_t>(static_cast<std::intptr_t>(h));
}

RefPlanHandle::RefPlanHandle() : impl_(0) {
  cufftHandle h;
  if (cufftCreate(&h) != CUFFT_SUCCESS) {
    std::fprintf(stderr, "RefPlanHandle: cufftCreate failed\n");
    return;
  }
  impl_ = from_cufft(h);
}

RefPlanHandle::~RefPlanHandle() {
  if (impl_) cufftDestroy(to_cufft(impl_));
}

RefPlanHandle::RefPlanHandle(RefPlanHandle&& other) noexcept : impl_(other.impl_) {
  other.impl_ = 0;
}

RefPlanHandle& RefPlanHandle::operator=(RefPlanHandle&& other) noexcept {
  if (this != &other) {
    if (impl_) cufftDestroy(to_cufft(impl_));
    impl_ = other.impl_;
    other.impl_ = 0;
  }
  return *this;
}

std::uintptr_t RefPlanHandle::get() const { return impl_; }
std::uintptr_t* RefPlanHandle::ptr() { return &impl_; }

// =========================================================================
// Backend lifecycle
// =========================================================================

void initialize() {}
std::string backend_name() { return "cuda"; }

// =========================================================================
// Helpers
// =========================================================================

static cufftType to_cufft_type(flagfftType type) {
  return static_cast<cufftType>(static_cast<int>(type));
}

static void check_cufft(cufftResult r, const std::string& context) {
  if (r != CUFFT_SUCCESS) {
    std::fprintf(stderr, "%s failed with code %d\n", context.c_str(), static_cast<int>(r));
  }
}

// =========================================================================
// Plan creation
// =========================================================================

void ref_plan_1d(RefPlanHandle& plan, int nx, flagfftType type, int batch) {
  auto r = cufftPlan1d(reinterpret_cast<cufftHandle*>(plan.ptr()), nx, to_cufft_type(type), batch);
  check_cufft(r, "cufftPlan1d");
}

void ref_plan_2d(RefPlanHandle& plan, int nx, int ny, flagfftType type) {
  auto r = cufftPlan2d(reinterpret_cast<cufftHandle*>(plan.ptr()), nx, ny, to_cufft_type(type));
  check_cufft(r, "cufftPlan2d");
}

void ref_plan_3d(RefPlanHandle& plan, int nx, int ny, int nz, flagfftType type) {
  auto r = cufftPlan3d(reinterpret_cast<cufftHandle*>(plan.ptr()), nx, ny, nz, to_cufft_type(type));
  check_cufft(r, "cufftPlan3d");
}

// =========================================================================
// Plan execution
// =========================================================================

void ref_exec_c2c(RefPlanHandle& plan, flagfftComplex* idata, flagfftComplex* odata, int direction) {
  check_cufft(cufftExecC2C(to_cufft(plan.get()),
                           reinterpret_cast<cufftComplex*>(idata),
                           reinterpret_cast<cufftComplex*>(odata),
                           direction),
              "cufftExecC2C");
}

void ref_exec_z2z(RefPlanHandle& plan, flagfftDoubleComplex* idata, flagfftDoubleComplex* odata, int direction) {
  check_cufft(cufftExecZ2Z(to_cufft(plan.get()),
                           reinterpret_cast<cufftDoubleComplex*>(idata),
                           reinterpret_cast<cufftDoubleComplex*>(odata),
                           direction),
              "cufftExecZ2Z");
}

void ref_exec_r2c(RefPlanHandle& plan, flagfftReal* idata, flagfftComplex* odata) {
  check_cufft(cufftExecR2C(to_cufft(plan.get()),
                           reinterpret_cast<cufftReal*>(idata),
                           reinterpret_cast<cufftComplex*>(odata)),
              "cufftExecR2C");
}

void ref_exec_d2z(RefPlanHandle& plan, flagfftDoubleReal* idata, flagfftDoubleComplex* odata) {
  check_cufft(cufftExecD2Z(to_cufft(plan.get()),
                           reinterpret_cast<cufftDoubleReal*>(idata),
                           reinterpret_cast<cufftDoubleComplex*>(odata)),
              "cufftExecD2Z");
}

void ref_exec_c2r(RefPlanHandle& plan, flagfftComplex* idata, flagfftReal* odata) {
  check_cufft(cufftExecC2R(to_cufft(plan.get()),
                           reinterpret_cast<cufftComplex*>(idata),
                           reinterpret_cast<cufftReal*>(odata)),
              "cufftExecC2R");
}

void ref_exec_z2d(RefPlanHandle& plan, flagfftDoubleComplex* idata, flagfftDoubleReal* odata) {
  check_cufft(cufftExecZ2D(to_cufft(plan.get()),
                           reinterpret_cast<cufftDoubleComplex*>(idata),
                           reinterpret_cast<cufftDoubleReal*>(odata)),
              "cufftExecZ2D");
}

// =========================================================================
// Data generation
// =========================================================================

std::vector<flagfftComplex> random_complex(int n) {
  std::vector<flagfftComplex> v(n);
  for (int i = 0; i < n; ++i) {
    v[i].x = static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f;
    v[i].y = static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f;
  }
  return v;
}

std::vector<flagfftDoubleComplex> random_double_complex(int n) {
  std::vector<flagfftDoubleComplex> v(n);
  for (int i = 0; i < n; ++i) {
    v[i].x = static_cast<double>(std::rand()) / RAND_MAX * 2.0 - 1.0;
    v[i].y = static_cast<double>(std::rand()) / RAND_MAX * 2.0 - 1.0;
  }
  return v;
}

std::vector<flagfftReal> random_real(int n) {
  std::vector<flagfftReal> v(n);
  for (int i = 0; i < n; ++i) {
    v[i] = static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f;
  }
  return v;
}

std::vector<flagfftDoubleReal> random_double_real(int n) {
  std::vector<flagfftDoubleReal> v(n);
  for (int i = 0; i < n; ++i) {
    v[i] = static_cast<double>(std::rand()) / RAND_MAX * 2.0 - 1.0;
  }
  return v;
}

// =========================================================================
// Correctness comparison
// =========================================================================

inline float complex_abs(const flagfftComplex& c) {
  return std::sqrt(c.x * c.x + c.y * c.y);
}

inline double complex_abs(const flagfftDoubleComplex& c) {
  return std::sqrt(c.x * c.x + c.y * c.y);
}

double max_relative_error(const flagfftComplex* a, const flagfftComplex* b, int n) {
  double max_err = 0.0;
  for (int i = 0; i < n; ++i) {
    double diff = std::abs(complex_abs(a[i]) - complex_abs(b[i]));
    double denom = complex_abs(b[i]);
    if (denom > 0.0) {
      double rel = diff / denom;
      if (rel > max_err) max_err = rel;
    }
  }
  return max_err;
}

double max_relative_error(const flagfftDoubleComplex* a, const flagfftDoubleComplex* b, int n) {
  double max_err = 0.0;
  for (int i = 0; i < n; ++i) {
    double diff = std::abs(complex_abs(a[i]) - complex_abs(b[i]));
    double denom = complex_abs(b[i]);
    if (denom > 0.0) {
      double rel = diff / denom;
      if (rel > max_err) max_err = rel;
    }
  }
  return max_err;
}

double max_relative_error_real(const flagfftReal* a, const flagfftReal* b, int n) {
  double max_err = 0.0;
  for (int i = 0; i < n; ++i) {
    double diff = std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
    double denom = std::abs(static_cast<double>(b[i]));
    if (denom > 0.0) {
      double rel = diff / denom;
      if (rel > max_err) max_err = rel;
    }
  }
  return max_err;
}

double max_relative_error_real(const flagfftDoubleReal* a, const flagfftDoubleReal* b, int n) {
  double max_err = 0.0;
  for (int i = 0; i < n; ++i) {
    double diff = std::abs(a[i] - b[i]);
    double denom = std::abs(b[i]);
    if (denom > 0.0) {
      double rel = diff / denom;
      if (rel > max_err) max_err = rel;
    }
  }
  return max_err;
}

ErrorMetric compute_error(const float* a, const float* b, std::size_t n) {
  ErrorMetric err{};
  double sum_sq = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    double abs_diff = std::abs(diff);
    if (abs_diff > err.max_abs) err.max_abs = abs_diff;
    sum_sq += diff * diff;
  }
  err.rms = std::sqrt(sum_sq / static_cast<double>(n));
  return err;
}

ErrorMetric compute_error(const double* a, const double* b, std::size_t n) {
  ErrorMetric err{};
  double sum_sq = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    double diff = a[i] - b[i];
    double abs_diff = std::abs(diff);
    if (abs_diff > err.max_abs) err.max_abs = abs_diff;
    sum_sq += diff * diff;
  }
  err.rms = std::sqrt(sum_sq / static_cast<double>(n));
  return err;
}

}  // namespace flagfft::test_adaptor
```

---

### Task 3: Update CMakeLists.txt to build test_adaptor

**Files:**
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Add test_adaptor OBJECT library to the top-level CMakeLists.txt**

Add after the existing `flagfft` library definition and before `FLAGFFT_BUILD_CLI`. This OBJECT library depends on CUDA/cuFFT and is usable by both `flagfft-cli` and `ctest`.

Locate the line `if(FLAGFFT_BUILD_CLI)` in CMakeLists.txt. Insert before it:

```cmake
# Test adaptor — reference FFT oracle for correctness checks and benchmarking.
# Reused by flagfft-cli (bench/verify) and ctest (correctness tests).
# Only built when CLI or tests are enabled (both need cuFFT).
if(FLAGFFT_BUILD_CLI OR FLAGFFT_BUILD_TESTS)
    find_package(CUDAToolkit REQUIRED)
    add_library(flagfft_test_adaptor OBJECT
        src/adaptor/backend/cuda/test_adaptor.cpp
    )
    target_compile_features(flagfft_test_adaptor PUBLIC cxx_std_20)
    target_include_directories(flagfft_test_adaptor PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        src/utils/include
        src
    )
    target_link_libraries(flagfft_test_adaptor PUBLIC
        CUDA::cufft
        CUDA::cudart
        CUDA::cuda_driver
    )
endif()
```

- [ ] **Step 2: Link flagfft-cli against flagfft_test_adaptor**

Inside the `if(FLAGFFT_BUILD_CLI)` block, in the `target_link_libraries(flagfft-cli PRIVATE ...)` call, add `flagfft_test_adaptor`:

```cmake
target_link_libraries(flagfft-cli
    PRIVATE
        flagfft
        flagfft_cli_common
        flagfft_test_adaptor
        CUDA::cufft
        SQLite3::SQLite3
        nlohmann_json::nlohmann_json
)
```

---

### Task 4: Update ctest CMakeLists.txt to use test_adaptor from src/

**Files:**
- Modify: `ctest/CMakeLists.txt`
- Modify: `ctest/backend/CMakeLists.txt` → delete file
- Delete: `ctest/backend/cuda/adaptor.cpp`

- [ ] **Step 1: Rewrite ctest/CMakeLists.txt**

Replace the entire contents:

```cmake
enable_testing()

if(NOT TARGET gtest)
    find_package(GTest REQUIRED)
endif()
include(GoogleTest)

set(TEST_TARGETS
    test_plan
    test_exec_c2c
    test_exec_z2z
    test_exec_r2c_c2r
    test_exec_d2z_z2d
)

foreach(test_name ${TEST_TARGETS})
    add_executable(${test_name} ${test_name}.cpp main.cpp)
    target_link_libraries(${test_name} PRIVATE flagfft flagfft_test_adaptor gtest)
    target_include_directories(${test_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
    )
    target_compile_features(${test_name} PRIVATE cxx_std_20)
    gtest_discover_tests(${test_name})
endforeach()
```

- [ ] **Step 2: Delete ctest/backend directory**

```bash
rm -rf ctest/backend
```

---

### Task 5: Rewrite ctest/flagfft_test.h to use test_adaptor.h

**Files:**
- Modify: `ctest/flagfft_test.h`

- [ ] **Step 1: Replace contents — keep only Google Test wrappers for FlagFFT API**

The refactored file includes `adaptor/test_adaptor.h` for the reference FFT oracle, uses `adaptor::Memory` for device memory, and keeps only the FlagFFT convenience wrappers that are Google Test specific.

```cpp
#pragma once

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <vector>

#include "adaptor/adaptor.h"
#include "adaptor/test_adaptor.h"
#include "flagfft.h"

// =========================================================================
// FlagFFT plan convenience wrappers (assert success, otherwise GTEST_FAIL)
// =========================================================================

inline void Plan1d(flagfftHandle* plan, int nx, flagfftType type, int batch = 1) {
  flagfftResult r = flagfftPlan1d(plan, nx, type, batch);
  if (r != FLAGFFT_SUCCESS) {
    FAIL() << "flagfftPlan1d(nx=" << nx << ", type=" << type << ", batch=" << batch << ") failed with code " << r;
  }
}

inline void Plan2d(flagfftHandle* plan, int nx, int ny, flagfftType type) {
  flagfftResult r = flagfftPlan2d(plan, nx, ny, type);
  if (r != FLAGFFT_SUCCESS) {
    FAIL() << "flagfftPlan2d(nx=" << nx << ", ny=" << ny << ", type=" << type << ") failed with code " << r;
  }
}

inline void Plan3d(flagfftHandle* plan, int nx, int ny, int nz, flagfftType type) {
  flagfftResult r = flagfftPlan3d(plan, nx, ny, nz, type);
  if (r != FLAGFFT_SUCCESS) {
    FAIL() << "flagfftPlan3d(nx=" << nx << ", ny=" << ny << ", nz=" << nz << ", type=" << type
           << ") failed with code " << r;
  }
}

// =========================================================================
// FlagFFT exec convenience wrappers (assert success, otherwise GTEST_FAIL)
// =========================================================================

inline void ExecC2C(flagfftHandle plan, flagfftComplex* idata, flagfftComplex* odata, int direction) {
  flagfftResult r = flagfftExecC2C(plan, idata, odata, direction);
  if (r != FLAGFFT_SUCCESS) FAIL() << "flagfftExecC2C failed with code " << r;
}

inline void ExecZ2Z(flagfftHandle plan, flagfftDoubleComplex* idata, flagfftDoubleComplex* odata, int direction) {
  flagfftResult r = flagfftExecZ2Z(plan, idata, odata, direction);
  if (r != FLAGFFT_SUCCESS) FAIL() << "flagfftExecZ2Z failed with code " << r;
}

inline void ExecR2C(flagfftHandle plan, flagfftReal* idata, flagfftComplex* odata) {
  flagfftResult r = flagfftExecR2C(plan, idata, odata);
  if (r != FLAGFFT_SUCCESS) FAIL() << "flagfftExecR2C failed with code " << r;
}

inline void ExecD2Z(flagfftHandle plan, flagfftDoubleReal* idata, flagfftDoubleComplex* odata) {
  flagfftResult r = flagfftExecD2Z(plan, idata, odata);
  if (r != FLAGFFT_SUCCESS) FAIL() << "flagfftExecD2Z failed with code " << r;
}

inline void ExecC2R(flagfftHandle plan, flagfftComplex* idata, flagfftReal* odata) {
  flagfftResult r = flagfftExecC2R(plan, idata, odata);
  if (r != FLAGFFT_SUCCESS) FAIL() << "flagfftExecC2R failed with code " << r;
}

inline void ExecZ2D(flagfftHandle plan, flagfftDoubleComplex* idata, flagfftDoubleReal* odata) {
  flagfftResult r = flagfftExecZ2D(plan, idata, odata);
  if (r != FLAGFFT_SUCCESS) FAIL() << "flagfftExecZ2D failed with code " << r;
}
```

---

### Task 6: Update ctest test files to use new namespace and adaptor::Memory

**Files:**
- Modify: `ctest/test_exec_c2c.cpp`
- Modify: `ctest/test_exec_z2z.cpp`
- Modify: `ctest/test_exec_r2c_c2r.cpp`
- Modify: `ctest/test_exec_d2z_z2d.cpp`

- [ ] **Step 1: Rewrite test_exec_c2c.cpp**

Replace `using namespace flagfft_test::adaptor;` with `using namespace flagfft::test_adaptor;`.
Replace all `allocate_device/free_device/copy_host_to_device/copy_device_to_host` calls with `adaptor::Memory`.

The pattern change is:
```cpp
// Old:
auto* d_in = static_cast<flagfftComplex*>(allocate_device(N * sizeof(flagfftComplex)));
copy_host_to_device(h_in.data(), d_in, N * sizeof(flagfftComplex));
// ...
free_device(d_in);

// New:
adaptor::Memory d_in(N * sizeof(flagfftComplex));
d_in.copy_from_host(h_in.data(), N * sizeof(flagfftComplex));
// No explicit free — RAII
```

Full rewritten `test_exec_c2c.cpp`:

```cpp
#include "flagfft_test.h"

using namespace flagfft::test_adaptor;

constexpr double kRelTol = 1e-4;

// =========================================================================
// 1D C2C
// =========================================================================

TEST(C2C_1D, ForwardPowerOfTwo) {
  constexpr int N = 256;
  flagfftHandle plan = nullptr;
  Plan1d(&plan, N, FLAGFFT_C2C, 1);

  auto h_in = random_complex(N);
  adaptor::Memory d_in(N * sizeof(flagfftComplex));
  adaptor::Memory d_out(N * sizeof(flagfftComplex));
  adaptor::Memory d_ref(N * sizeof(flagfftComplex));
  d_in.copy_from_host(h_in.data(), N * sizeof(flagfftComplex));

  ExecC2C(plan, static_cast<flagfftComplex*>(d_in.data()),
          static_cast<flagfftComplex*>(d_out.data()), FLAGFFT_FORWARD);

  RefPlanHandle ref;
  ref_plan_1d(ref, N, FLAGFFT_C2C, 1);
  ref_exec_c2c(ref, static_cast<flagfftComplex*>(d_in.data()),
               static_cast<flagfftComplex*>(d_ref.data()), FLAGFFT_FORWARD);

  std::vector<flagfftComplex> h_out(N);
  std::vector<flagfftComplex> h_ref_out(N);
  d_out.copy_to_host(h_out.data(), N * sizeof(flagfftComplex));
  d_ref.copy_to_host(h_ref_out.data(), N * sizeof(flagfftComplex));

  double max_err = max_relative_error(h_out.data(), h_ref_out.data(), N);
  EXPECT_LT(max_err, kRelTol) << "Max relative error: " << max_err;

  flagfftDestroy(plan);
}

TEST(C2C_1D, InversePowerOfTwo) {
  constexpr int N = 256;
  flagfftHandle plan = nullptr;
  Plan1d(&plan, N, FLAGFFT_C2C, 1);

  auto h_in = random_complex(N);
  adaptor::Memory d_in(N * sizeof(flagfftComplex));
  adaptor::Memory d_out(N * sizeof(flagfftComplex));
  adaptor::Memory d_ref(N * sizeof(flagfftComplex));
  d_in.copy_from_host(h_in.data(), N * sizeof(flagfftComplex));

  ExecC2C(plan, static_cast<flagfftComplex*>(d_in.data()),
          static_cast<flagfftComplex*>(d_out.data()), FLAGFFT_INVERSE);

  RefPlanHandle ref;
  ref_plan_1d(ref, N, FLAGFFT_C2C, 1);
  ref_exec_c2c(ref, static_cast<flagfftComplex*>(d_in.data()),
               static_cast<flagfftComplex*>(d_ref.data()), FLAGFFT_INVERSE);

  std::vector<flagfftComplex> h_out(N);
  std::vector<flagfftComplex> h_ref_out(N);
  d_out.copy_to_host(h_out.data(), N * sizeof(flagfftComplex));
  d_ref.copy_to_host(h_ref_out.data(), N * sizeof(flagfftComplex));

  double max_err = max_relative_error(h_out.data(), h_ref_out.data(), N);
  EXPECT_LT(max_err, kRelTol) << "Max relative error: " << max_err;

  flagfftDestroy(plan);
}

TEST(C2C_1D, RoundtripForwardInverse) {
  constexpr int N = 256;
  flagfftHandle plan = nullptr;
  Plan1d(&plan, N, FLAGFFT_C2C, 1);

  auto h_in = random_complex(N);
  adaptor::Memory d_in(N * sizeof(flagfftComplex));
  adaptor::Memory d_mid(N * sizeof(flagfftComplex));
  adaptor::Memory d_out(N * sizeof(flagfftComplex));
  d_in.copy_from_host(h_in.data(), N * sizeof(flagfftComplex));

  ExecC2C(plan, static_cast<flagfftComplex*>(d_in.data()),
          static_cast<flagfftComplex*>(d_mid.data()), FLAGFFT_FORWARD);
  ExecC2C(plan, static_cast<flagfftComplex*>(d_mid.data()),
          static_cast<flagfftComplex*>(d_out.data()), FLAGFFT_INVERSE);

  std::vector<flagfftComplex> h_out(N);
  d_out.copy_to_host(h_out.data(), N * sizeof(flagfftComplex));

  for (int i = 0; i < N; ++i) {
    double expected_x = h_in[i].x * N;
    double expected_y = h_in[i].y * N;
    EXPECT_NEAR(h_out[i].x, expected_x, N * kRelTol) << "Mismatch at index " << i << " (real)";
    EXPECT_NEAR(h_out[i].y, expected_y, N * kRelTol) << "Mismatch at index " << i << " (imag)";
  }

  flagfftDestroy(plan);
}

TEST(C2C_1D, NonPowerOfTwo) {
  constexpr int N = 243;
  flagfftHandle plan = nullptr;
  Plan1d(&plan, N, FLAGFFT_C2C, 1);

  auto h_in = random_complex(N);
  adaptor::Memory d_in(N * sizeof(flagfftComplex));
  adaptor::Memory d_out(N * sizeof(flagfftComplex));
  adaptor::Memory d_ref(N * sizeof(flagfftComplex));
  d_in.copy_from_host(h_in.data(), N * sizeof(flagfftComplex));

  ExecC2C(plan, static_cast<flagfftComplex*>(d_in.data()),
          static_cast<flagfftComplex*>(d_out.data()), FLAGFFT_FORWARD);

  RefPlanHandle ref;
  ref_plan_1d(ref, N, FLAGFFT_C2C, 1);
  ref_exec_c2c(ref, static_cast<flagfftComplex*>(d_in.data()),
               static_cast<flagfftComplex*>(d_ref.data()), FLAGFFT_FORWARD);

  std::vector<flagfftComplex> h_out(N);
  std::vector<flagfftComplex> h_ref_out(N);
  d_out.copy_to_host(h_out.data(), N * sizeof(flagfftComplex));
  d_ref.copy_to_host(h_ref_out.data(), N * sizeof(flagfftComplex));

  double max_err = max_relative_error(h_out.data(), h_ref_out.data(), N);
  EXPECT_LT(max_err, kRelTol) << "Max relative error: " << max_err;

  flagfftDestroy(plan);
}

TEST(C2C_1D, Batch) {
  constexpr int N = 128;
  constexpr int B = 4;
  int total = N * B;
  flagfftHandle plan = nullptr;
  Plan1d(&plan, N, FLAGFFT_C2C, B);

  auto h_in = random_complex(total);
  adaptor::Memory d_in(total * sizeof(flagfftComplex));
  adaptor::Memory d_out(total * sizeof(flagfftComplex));
  adaptor::Memory d_ref(total * sizeof(flagfftComplex));
  d_in.copy_from_host(h_in.data(), total * sizeof(flagfftComplex));

  ExecC2C(plan, static_cast<flagfftComplex*>(d_in.data()),
          static_cast<flagfftComplex*>(d_out.data()), FLAGFFT_FORWARD);

  RefPlanHandle ref;
  ref_plan_1d(ref, N, FLAGFFT_C2C, B);
  ref_exec_c2c(ref, static_cast<flagfftComplex*>(d_in.data()),
               static_cast<flagfftComplex*>(d_ref.data()), FLAGFFT_FORWARD);

  std::vector<flagfftComplex> h_out(total);
  std::vector<flagfftComplex> h_ref_out(total);
  d_out.copy_to_host(h_out.data(), total * sizeof(flagfftComplex));
  d_ref.copy_to_host(h_ref_out.data(), total * sizeof(flagfftComplex));

  double max_err = max_relative_error(h_out.data(), h_ref_out.data(), total);
  EXPECT_LT(max_err, kRelTol) << "Max relative error: " << max_err;

  flagfftDestroy(plan);
}

// =========================================================================
// 2D C2C
// =========================================================================

TEST(C2C_2D, ForwardSmall) {
  constexpr int NX = 32;
  constexpr int NY = 16;
  constexpr int N = NX * NY;
  flagfftHandle plan = nullptr;
  Plan2d(&plan, NX, NY, FLAGFFT_C2C);

  auto h_in = random_complex(N);
  adaptor::Memory d_in(N * sizeof(flagfftComplex));
  adaptor::Memory d_out(N * sizeof(flagfftComplex));
  adaptor::Memory d_ref(N * sizeof(flagfftComplex));
  d_in.copy_from_host(h_in.data(), N * sizeof(flagfftComplex));

  ExecC2C(plan, static_cast<flagfftComplex*>(d_in.data()),
          static_cast<flagfftComplex*>(d_out.data()), FLAGFFT_FORWARD);

  RefPlanHandle ref;
  ref_plan_2d(ref, NX, NY, FLAGFFT_C2C);
  ref_exec_c2c(ref, static_cast<flagfftComplex*>(d_in.data()),
               static_cast<flagfftComplex*>(d_ref.data()), FLAGFFT_FORWARD);

  std::vector<flagfftComplex> h_out(N);
  std::vector<flagfftComplex> h_ref_out(N);
  d_out.copy_to_host(h_out.data(), N * sizeof(flagfftComplex));
  d_ref.copy_to_host(h_ref_out.data(), N * sizeof(flagfftComplex));

  double max_err = max_relative_error(h_out.data(), h_ref_out.data(), N);
  EXPECT_LT(max_err, kRelTol) << "Max relative error: " << max_err;

  flagfftDestroy(plan);
}

// =========================================================================
// 3D C2C
// =========================================================================

TEST(C2C_3D, ForwardSmall) {
  constexpr int NX = 16;
  constexpr int NY = 8;
  constexpr int NZ = 4;
  constexpr int N = NX * NY * NZ;
  flagfftHandle plan = nullptr;
  Plan3d(&plan, NX, NY, NZ, FLAGFFT_C2C);

  auto h_in = random_complex(N);
  adaptor::Memory d_in(N * sizeof(flagfftComplex));
  adaptor::Memory d_out(N * sizeof(flagfftComplex));
  adaptor::Memory d_ref(N * sizeof(flagfftComplex));
  d_in.copy_from_host(h_in.data(), N * sizeof(flagfftComplex));

  ExecC2C(plan, static_cast<flagfftComplex*>(d_in.data()),
          static_cast<flagfftComplex*>(d_out.data()), FLAGFFT_FORWARD);

  RefPlanHandle ref;
  ref_plan_3d(ref, NX, NY, NZ, FLAGFFT_C2C);
  ref_exec_c2c(ref, static_cast<flagfftComplex*>(d_in.data()),
               static_cast<flagfftComplex*>(d_ref.data()), FLAGFFT_FORWARD);

  std::vector<flagfftComplex> h_out(N);
  std::vector<flagfftComplex> h_ref_out(N);
  d_out.copy_to_host(h_out.data(), N * sizeof(flagfftComplex));
  d_ref.copy_to_host(h_ref_out.data(), N * sizeof(flagfftComplex));

  double max_err = max_relative_error(h_out.data(), h_ref_out.data(), N);
  EXPECT_LT(max_err, kRelTol) << "Max relative error: " << max_err;

  flagfftDestroy(plan);
}
```

- [ ] **Step 2: Apply the same namespace + adaptor::Memory pattern to the other 3 test files**

For `test_exec_z2z.cpp`: Replace `using namespace flagfft_test::adaptor` → `using namespace flagfft::test_adaptor`, and replace `allocate_device/free_device/copy_*` with `adaptor::Memory`. The data types become `flagfftDoubleComplex`, `flagfftDoubleReal` etc.

For `test_exec_r2c_c2r.cpp`: Same namespace + Memory migration.

For `test_exec_d2z_z2d.cpp`: Same namespace + Memory migration.

The pattern is identical to what was shown for `test_exec_c2c.cpp` — replace raw pointer device memory with `adaptor::Memory` RAII.

---

### Task 7: Build ctest and verify all tests pass

**Files:** (none — verification only)

- [ ] **Step 1: Build with C++ tests enabled in Docker container**

```bash
docker exec flagtree-dev3 bash -c "cd /workspace/FlagFFT-dev/.claude/worktrees/bench-rebuild && mkdir -p build && cd build && cmake .. -DFLAGFFT_BUILD_TESTS=ON -DFLAGFFT_BUILD_CLI=ON && make -j$(nproc)"
```

Expected: Build succeeds with no errors.

- [ ] **Step 2: Run ctest**

```bash
docker exec flagtree-dev3 bash -c "cd /workspace/FlagFFT-dev/.claude/worktrees/bench-rebuild/build && ctest --output-on-failure"
```

Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/adaptor/test_adaptor.h src/adaptor/backend/cuda/test_adaptor.cpp CMakeLists.txt
git add ctest/flagfft_test.h ctest/CMakeLists.txt ctest/test_exec_*.cpp
git rm ctest/backend/cuda/adaptor.cpp ctest/backend/CMakeLists.txt
git commit -m "refactor: extract test_adaptor from ctest into src/adaptor/

Move reference FFT oracle (RefPlanHandle, ref_plan_*, ref_exec_*,
data generation, error comparison) from ctest/ to src/adaptor/
as a shared test adapter layer. ctest now uses adaptor::Memory
for device allocations instead of raw CUDA malloc wrappers."
```

---

### Task 8: Refactor bench.cpp to use test_adaptor

**Files:**
- Modify: `src/cli_tools/tune/bench.cpp`

- [ ] **Step 1: Update the verify_against_cufft function**

Replace the raw cuFFT calls in `verify_against_cufft()` with `test_adaptor` interface:

In `verify_against_cufft()`, replace:
```cpp
// Old code:
flagfft::cli::CufftPlanHandle cf_plan;
check_cufft(cufftPlan1d(cf_plan.put(), static_cast<int>(n), CUFFT_C2C, static_cast<int>(batch)),
            "cufftPlan1d");
check_cufft(cufftSetStream(cf_plan.get(), reinterpret_cast<cudaStream_t>(stream.get())),
            "cufftSetStream");
```

With:
```cpp
// New code:
test_adaptor::RefPlanHandle ref_plan;
test_adaptor::ref_plan_1d(ref_plan, static_cast<int>(n), FLAGFFT_C2C, static_cast<int>(batch));
```

And replace the cuFFT exec:
```cpp
// Old code:
check_cufft(cufftExecC2C(cf_plan.get(),
                         reinterpret_cast<cufftComplex *>(d_in.get()),
                         reinterpret_cast<cufftComplex *>(d_out_cf.get()),
                         cf_dir),
            "cufftExecC2C");
```

With:
```cpp
// New code:
test_adaptor::ref_exec_c2c(ref_plan,
                           reinterpret_cast<flagfftComplex*>(d_in.get()),
                           reinterpret_cast<flagfftComplex*>(d_out_cf.get()),
                           cf_dir);
```

Then replace the manual error computation with:
```cpp
// Old code:
BenchError err {};
double sum_sq = 0.0;
for (std::size_t i = 0; i < out_ff.size(); ++i) {
  double diff = static_cast<double>(out_ff[i]) - static_cast<double>(out_cf[i]);
  double abs_diff = std::abs(diff);
  if (abs_diff > err.max_abs) err.max_abs = abs_diff;
  sum_sq += diff * diff;
}
err.rms = std::sqrt(sum_sq / static_cast<double>(out_ff.size()));

// New code:
auto metric = test_adaptor::compute_error(out_ff.data(), out_cf.data(), out_ff.size());
BenchError err{metric.max_abs, metric.rms};
```

- [ ] **Step 2: Replace generate_random_input with test_adaptor functions**

Remove the `generate_random_input()` function from `bench.cpp` (and its declaration from `bench.hpp`). Inline uses should switch to `test_adaptor::random_complex()`.

The existing `bench_candidate()` calls `generate_random_input(n, batch)` which returns `std::vector<float>` of interleaved real/imag parts. Replace with:
```cpp
auto h_complex = test_adaptor::random_complex(static_cast<int>(n * batch));
const float* host_data = reinterpret_cast<const float*>(h_complex.data());
std::size_t bytes = h_complex.size() * sizeof(flagfftComplex);
```

- [ ] **Step 3: Add include to bench.cpp**

```cpp
#include "adaptor/test_adaptor.h"
```

---

### Task 9: Build flagfft-cli and verify bench command works

**Files:** (none — verification only)

- [ ] **Step 1: Build with CLI enabled in Docker**

```bash
docker exec flagtree-dev3 bash -c "cd /workspace/FlagFFT-dev/.claude/worktrees/bench-rebuild/build && cmake .. -DFLAGFFT_BUILD_CLI=ON && make -j$(nproc) flagfft-cli"
```

- [ ] **Step 2: Smoke test the bench subcommand**

```bash
docker exec flagtree-dev3 bash -c "cd /workspace/FlagFFT-dev/.claude/worktrees/bench-rebuild && ./build/flagfft-cli bench --api c2c --shape 16 --batch 1 --warmup 0 --iters 1 --launches-per-sample 1 --print-path"
```

Expected: JSON output with `cases[0].correctness.passed == true` and valid `flagfft_ms` / `cufft_ms` timing values.

- [ ] **Step 3: Verify existing CLI tests still pass**

```bash
docker exec flagtree-dev3 bash -c "cd /workspace/FlagFFT-dev/.claude/worktrees/bench-rebuild && python -m pytest tests/cli/test_flagfft_cli.py -v"
```

- [ ] **Step 4: Commit**

```bash
git add src/cli_tools/tune/bench.cpp src/cli_tools/tune/bench.hpp
git commit -m "refactor: use test_adaptor in bench.cpp for cuFFT oracle calls

Replace raw cuFFT plan/exec calls and manual error computation
with test_adaptor interface. Replace generate_random_input with
test_adaptor::random_complex."
```

---

### Task 10: Create benchmark/conftest.py

**Files:**
- Create: `benchmark/__init__.py`
- Create: `benchmark/conftest.py`

- [ ] **Step 1: Create benchmark/__init__.py**

```python
```

- [ ] **Step 2: Create benchmark/conftest.py**

```python
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def pytest_addoption(parser):
    parser.addoption(
        "--flagfft-cli",
        default=None,
        help="Path to flagfft-cli. Defaults to FLAGFFT_CLI_EXE or build/flagfft-cli.",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "smoke: fast benchmark smoke tests")
    config.addinivalue_line("markers", "full: full benchmark suite")


@pytest.fixture(scope="session")
def flagfft_cli(request) -> Path:
    configured = request.config.getoption("--flagfft-cli")
    path = Path(
        configured or os.environ.get("FLAGFFT_CLI_EXE", ROOT / "build" / "flagfft-cli")
    )
    if not path.exists():
        pytest.skip(f"flagfft-cli is not built: {path}")
    return path


@pytest.fixture
def invoke_bench(flagfft_cli):
    def invoke(*arguments: str, env: dict[str, str] | None = None, timeout: int = 600):
        process_env = os.environ.copy()
        if env:
            process_env.update(env)
        result = subprocess.run(
            [str(flagfft_cli), *arguments],
            cwd=ROOT,
            env=process_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
        try:
            report = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            pytest.fail(f"invalid CLI JSON: {result.stdout}\n{result.stderr}\n{error}")
        if report.get("status") == "skipped":
            assert result.returncode == 77
            pytest.skip(report.get("reason", "CLI skipped"))
        if report.get("status") == "unsupported":
            assert result.returncode == 77
            pytest.skip(report.get("reason", "CLI unsupported"))
        return result, report

    return invoke
```

---

### Task 11: Create benchmark/suites.py

**Files:**
- Create: `benchmark/suites.py`

- [ ] **Step 1: Write the suites module**

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchCase:
    size: int
    factorization: str
    codepath: str
    rationale: str


SMOKE_SUITE: list[BenchCase] = [
    BenchCase(16, "2^4", "DirectDFT", "smallest pure butterfly"),
    BenchCase(256, "2^8", "Leaf", "classic pow2 benchmark"),
    BenchCase(997, "prime", "Bluestein", "large prime, test Bluestein"),
]

FULL_SUITE: list[BenchCase] = [
    BenchCase(16, "2^4", "DirectDFT", "smallest pure butterfly"),
    BenchCase(23, "prime", "DirectDFT", "small prime >19, no codelet"),
    BenchCase(64, "2^6", "DirectDFT boundary", "DirectDFT upper limit"),
    BenchCase(81, "3^4", "Leaf", "non-2 power, pure codelet-3"),
    BenchCase(243, "3^5", "Leaf", "non-2 power"),
    BenchCase(256, "2^8", "Leaf", "classic pow2 benchmark"),
    BenchCase(361, "19^2", "Leaf", "largest codelet (19)"),
    BenchCase(512, "2^9", "Leaf", "medium-large pow2"),
    BenchCase(997, "prime", "Bluestein", "large prime, test Bluestein"),
    BenchCase(2048, "2^11", "Leaf", "large leaf"),
    BenchCase(4096, "2^12", "Leaf boundary", "kLeafMaxN upper limit"),
    BenchCase(8192, "2^13", "FourStep", "first beyond leaf"),
    BenchCase(16384, "2^14", "Specialized FourStep", "hardcoded 64x256"),
]

SMOKE_API = ["c2c"]
FULL_API = ["c2c", "z2z", "r2c", "d2z", "c2r", "z2d"]
```

---

### Task 12: Create benchmark/test_bench_smoke.py

**Files:**
- Create: `benchmark/test_bench_smoke.py`

- [ ] **Step 1: Write the smoke test file**

```python
from __future__ import annotations

import pytest

from benchmark.suites import SMOKE_API, SMOKE_SUITE


@pytest.mark.smoke
@pytest.mark.parametrize("api", SMOKE_API)
@pytest.mark.parametrize("case", SMOKE_SUITE, ids=lambda c: f"N={c.size}")
def test_bench_smoke(invoke_bench, api, case):
    result, report = invoke_bench(
        "bench",
        "--api", api,
        "--direction", "forward",
        "--shape", str(case.size),
        "--batch", "1",
        "--warmup", "0",
        "--iters", "1",
        "--launches-per-sample", "1",
        "--print-path",
    )
    assert result.returncode == 0, report
    bench_case = report["cases"][0]
    assert bench_case["correctness"]["passed"]
    assert "speedup" in bench_case["timing"]
    assert "flagfft_ms" in bench_case["timing"]
    assert "cufft_ms" in bench_case["timing"]
```

---

### Task 13: Create benchmark/test_bench_full.py

**Files:**
- Create: `benchmark/test_bench_full.py`

- [ ] **Step 1: Write the full test file**

```python
from __future__ import annotations

import pytest

from benchmark.suites import FULL_API, FULL_SUITE


@pytest.mark.full
@pytest.mark.parametrize("api", FULL_API)
@pytest.mark.parametrize("case", FULL_SUITE, ids=lambda c: f"N={c.size}")
def test_bench_full(invoke_bench, api, case):
    result, report = invoke_bench(
        "bench",
        "--api", api,
        "--direction", "forward",
        "--shape", str(case.size),
        "--batch", "1",
        "--warmup", "5",
        "--iters", "20",
        "--launches-per-sample", "1",
        "--print-path",
    )
    assert result.returncode == 0, report
    bench_case = report["cases"][0]
    assert bench_case["correctness"]["passed"]
    timing = bench_case["timing"]
    assert timing["speedup"] > 0
    assert timing["flagfft_ms"] > 0
    assert timing["cufft_ms"] > 0
```

---

### Task 14: Create benchmark/report.py

**Files:**
- Create: `benchmark/report.py`

- [ ] **Step 1: Write the report generator**

```python
from __future__ import annotations

import sys
from pathlib import Path


def generate_markdown(results: list[dict], output_path: str | Path | None = None) -> str:
    """Generate a Markdown benchmark report from a list of bench result dicts.

    Each dict should have: api, size, codepath, flagfft_ms, cufft_ms, speedup, max_abs_error.
    """
    header = "| API | Size | Codepath | FlagFFT (ms) | cuFFT (ms) | Speedup | Max Abs Error |\n"
    header += "|-----|------|----------|-------------|-----------|---------|---------------|\n"

    rows = []
    for r in results:
        rows.append(
            f"| {r['api']} | {r['size']} | {r['codepath']} | "
            f"{r['flagfft_ms']:.4f} | {r['cufft_ms']:.4f} | {r['speedup']:.2f}x | "
            f"{r['max_abs_error']:.2e} |"
        )

    table = header + "\n".join(rows)

    if output_path:
        path = Path(output_path)
        path.write_text(f"# FlagFFT Benchmark Report\n\n{table}\n")

    return table


def extract_results(report: dict) -> list[dict]:
    """Extract flat result dicts from a flagfft-cli JSON report."""
    results = []
    for case in report.get("cases", []):
        results.append({
            "api": case.get("api", "?"),
            "size": case.get("shape", "?"),
            "codepath": case.get("plan_description", "?"),
            "flagfft_ms": case["timing"].get("flagfft_ms", 0),
            "cufft_ms": case["timing"].get("cufft_ms", 0),
            "speedup": case["timing"].get("speedup", 0),
            "max_abs_error": case["correctness"].get("max_abs_error", 0),
        })
    return results


if __name__ == "__main__":
    import json

    data = json.loads(sys.stdin.read())
    results = extract_results(data)
    md = generate_markdown(results, output_path=None)
    sys.stdout.write(md)
```

---

### Task 15: Final verification — full test suite

**Files:** (none — verification only)

- [ ] **Step 1: Run C++ ctest**

```bash
docker exec flagtree-dev3 bash -c "cd /workspace/FlagFFT-dev/.claude/worktrees/bench-rebuild/build && ctest --output-on-failure"
```
Expected: All tests pass.

- [ ] **Step 2: Run CLI tests**

```bash
docker exec flagtree-dev3 bash -c "cd /workspace/FlagFFT-dev/.claude/worktrees/bench-rebuild && python -m pytest tests/ -v"
```
Expected: All tests pass.

- [ ] **Step 3: Run smoke benchmark tests**

```bash
docker exec flagtree-dev3 bash -c "cd /workspace/FlagFFT-dev/.claude/worktrees/bench-rebuild && python -m pytest benchmark/test_bench_smoke.py -v -m smoke"
```
Expected: Smoke tests pass (3 APIs x 3 sizes = 9 test cases).

- [ ] **Step 4: Run full benchmark tests**

```bash
docker exec flagtree-dev3 bash -c "cd /workspace/FlagFFT-dev/.claude/worktrees/bench-rebuild && python -m pytest benchmark/test_bench_full.py -v -m full"
```
Expected: Full tests pass (6 APIs x 13 sizes = 78 test cases).

- [ ] **Step 5: Generate Markdown report**

```bash
docker exec flagtree-dev3 bash -c "cd /workspace/FlagFFT-dev/.claude/worktrees/bench-rebuild && ./build/flagfft-cli bench --api c2c --shape 16 --shape 256 --shape 997 --batch 1 --warmup 5 --iters 20 --launches-per-sample 1 | python benchmark/report.py"
```
Expected: Markdown table printed to stdout with correct values.

- [ ] **Step 6: Commit**

```bash
git add benchmark/
git commit -m "feat: add Python benchmark infrastructure

Mirror tests/conftest.py fixture pattern for benchmark CLI invocation.
Define smoke (3 sizes) and full (13 sizes) test suites in suites.py.
Add report.py for JSON-to-Markdown report generation."
```

---

### Task 16: Add .agents record

**Files:**
- Create: `.agents/2026-05-26-benchmark-refactor.md`

- [ ] **Step 1: Write the record**

Follow the project convention from CLAUDE.md.
