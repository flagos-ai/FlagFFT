#pragma once

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cstdint>

#include "flagfft.h"

namespace flagfft_test::adaptor {

// =========================================================================
// RefHandle — RAII wrapper for the platform-specific reference FFT plan
// =========================================================================

class RefHandle {
 public:
  RefHandle();
  ~RefHandle();
  RefHandle(RefHandle&&) noexcept;
  RefHandle& operator=(RefHandle&&) noexcept;
  RefHandle(const RefHandle&) = delete;

  uintptr_t get() const;
  uintptr_t* ptr();

 private:
  uintptr_t impl_;  // cufftHandle / rocfft_plan / 0
};

// =========================================================================
// Backend lifecycle
// =========================================================================

void initialize();
std::string backend_name();

// =========================================================================
// Device memory
// =========================================================================

void* allocate_device(std::size_t bytes);
void free_device(void* ptr);
void copy_host_to_device(const void* src, void* dst, std::size_t bytes);
void copy_device_to_host(const void* src, void* dst, std::size_t bytes);

// =========================================================================
// Convenience Plan wrappers (assert success, otherwise GTEST_FAIL)
// =========================================================================

inline void Plan1d(flagfftHandle* plan, int nx, flagfftType type, int batch = 1) {
  flagfftResult r = flagfftPlan1d(plan, nx, type, batch);
  if (r != FLAGFFT_SUCCESS) {
    FAIL() << "flagfftPlan1d(nx=" << nx << ", type=" << type << ", batch=" << batch << ") failed with code "
           << r;
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
// Convenience Exec wrappers (assert success, otherwise GTEST_FAIL)
// =========================================================================

inline void ExecC2C(flagfftHandle plan, flagfftComplex* idata, flagfftComplex* odata, int direction) {
  flagfftResult r = flagfftExecC2C(plan, idata, odata, direction);
  if (r != FLAGFFT_SUCCESS) {
    FAIL() << "flagfftExecC2C failed with code " << r;
  }
}

inline void ExecZ2Z(flagfftHandle plan,
                    flagfftDoubleComplex* idata,
                    flagfftDoubleComplex* odata,
                    int direction) {
  flagfftResult r = flagfftExecZ2Z(plan, idata, odata, direction);
  if (r != FLAGFFT_SUCCESS) {
    FAIL() << "flagfftExecZ2Z failed with code " << r;
  }
}

inline void ExecR2C(flagfftHandle plan, flagfftReal* idata, flagfftComplex* odata) {
  flagfftResult r = flagfftExecR2C(plan, idata, odata);
  if (r != FLAGFFT_SUCCESS) {
    FAIL() << "flagfftExecR2C failed with code " << r;
  }
}

inline void ExecD2Z(flagfftHandle plan, flagfftDoubleReal* idata, flagfftDoubleComplex* odata) {
  flagfftResult r = flagfftExecD2Z(plan, idata, odata);
  if (r != FLAGFFT_SUCCESS) {
    FAIL() << "flagfftExecD2Z failed with code " << r;
  }
}

inline void ExecC2R(flagfftHandle plan, flagfftComplex* idata, flagfftReal* odata) {
  flagfftResult r = flagfftExecC2R(plan, idata, odata);
  if (r != FLAGFFT_SUCCESS) {
    FAIL() << "flagfftExecC2R failed with code " << r;
  }
}

inline void ExecZ2D(flagfftHandle plan, flagfftDoubleComplex* idata, flagfftDoubleReal* odata) {
  flagfftResult r = flagfftExecZ2D(plan, idata, odata);
  if (r != FLAGFFT_SUCCESS) {
    FAIL() << "flagfftExecZ2D failed with code " << r;
  }
}

// =========================================================================
// Reference FFT interface (implemented per backend)
// =========================================================================

void ref_plan_1d(RefHandle& plan, int nx, flagfftType type, int batch);
void ref_plan_2d(RefHandle& plan, int nx, int ny, flagfftType type);
void ref_plan_3d(RefHandle& plan, int nx, int ny, int nz, flagfftType type);
void ref_exec_c2c(RefHandle& plan, flagfftComplex* idata, flagfftComplex* odata, int direction);
void ref_exec_z2z(RefHandle& plan, flagfftDoubleComplex* idata, flagfftDoubleComplex* odata, int direction);
void ref_exec_r2c(RefHandle& plan, flagfftReal* idata, flagfftComplex* odata);
void ref_exec_d2z(RefHandle& plan, flagfftDoubleReal* idata, flagfftDoubleComplex* odata);
void ref_exec_c2r(RefHandle& plan, flagfftComplex* idata, flagfftReal* odata);
void ref_exec_z2d(RefHandle& plan, flagfftDoubleComplex* idata, flagfftDoubleReal* odata);

// =========================================================================
// Comparison helpers
// =========================================================================

inline float complex_abs(const flagfftComplex& c) {
  return std::sqrt(c.x * c.x + c.y * c.y);
}

inline double complex_abs(const flagfftDoubleComplex& c) {
  return std::sqrt(c.x * c.x + c.y * c.y);
}

inline double max_relative_error(const flagfftComplex* a, const flagfftComplex* b, int n) {
  double max_err = 0.0;
  for (int i = 0; i < n; ++i) {
    double denom = std::max(static_cast<double>(complex_abs(b[i])), 1.0);
    double err_r = std::abs(static_cast<double>(a[i].x) - static_cast<double>(b[i].x)) / denom;
    double err_i = std::abs(static_cast<double>(a[i].y) - static_cast<double>(b[i].y)) / denom;
    if (err_r > max_err) max_err = err_r;
    if (err_i > max_err) max_err = err_i;
  }
  return max_err;
}

inline double max_relative_error(const flagfftDoubleComplex* a, const flagfftDoubleComplex* b, int n) {
  double max_err = 0.0;
  for (int i = 0; i < n; ++i) {
    double denom = std::max(complex_abs(b[i]), 1.0);
    double err_r = std::abs(a[i].x - b[i].x) / denom;
    double err_i = std::abs(a[i].y - b[i].y) / denom;
    if (err_r > max_err) max_err = err_r;
    if (err_i > max_err) max_err = err_i;
  }
  return max_err;
}

inline double max_relative_error_real(const flagfftReal* a, const flagfftReal* b, int n) {
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

inline double max_relative_error_real(const flagfftDoubleReal* a, const flagfftDoubleReal* b, int n) {
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

// =========================================================================
// Test data generators
// =========================================================================

inline std::vector<flagfftComplex> random_complex(int n) {
  std::vector<flagfftComplex> v(n);
  for (int i = 0; i < n; ++i) {
    v[i].x = static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f;
    v[i].y = static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f;
  }
  return v;
}

inline std::vector<flagfftDoubleComplex> random_double_complex(int n) {
  std::vector<flagfftDoubleComplex> v(n);
  for (int i = 0; i < n; ++i) {
    v[i].x = static_cast<double>(std::rand()) / RAND_MAX * 2.0 - 1.0;
    v[i].y = static_cast<double>(std::rand()) / RAND_MAX * 2.0 - 1.0;
  }
  return v;
}

inline std::vector<flagfftReal> random_real(int n) {
  std::vector<flagfftReal> v(n);
  for (int i = 0; i < n; ++i) {
    v[i] = static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f;
  }
  return v;
}

inline std::vector<flagfftDoubleReal> random_double_real(int n) {
  std::vector<flagfftDoubleReal> v(n);
  for (int i = 0; i < n; ++i) {
    v[i] = static_cast<double>(std::rand()) / RAND_MAX * 2.0 - 1.0;
  }
  return v;
}

// =========================================================================
// 1D Test parameterization
// =========================================================================

struct Test1DParam {
  int N;
  int batch;
};

// Registry — add/remove sizes and batch values here
constexpr int k1DSizesSmall[] = {16, 23, 64, 81};
constexpr int k1DSizesMedium[] = {243, 256, 361, 512, 997};
constexpr int k1DSizesLarge[] = {2048, 4096, 8192, 16384};
constexpr int k1DBatchValues[] = {1, 4, 256};

inline std::vector<Test1DParam> Generate1DParams(const int* sizes, int numSizes) {
  std::vector<Test1DParam> params;
  for (int i = 0; i < numSizes; ++i)
    for (int b : k1DBatchValues) params.push_back({sizes[i], b});
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

}  // namespace flagfft_test::adaptor
