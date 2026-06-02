#include "adaptor/test_adaptor.h"

#include <cuda_runtime_api.h>
#include <cufft.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

static_assert(sizeof(flagfftComplex) == 2 * sizeof(float), "flagfftComplex must have no padding");
static_assert(sizeof(flagfftDoubleComplex) == 2 * sizeof(double),
              "flagfftDoubleComplex must have no padding");

namespace flagfft::test_adaptor {

// =========================================================================
// RefPlanHandle - cuFFT implementation
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

std::uintptr_t RefPlanHandle::get() const {
  return impl_;
}
void RefPlanHandle::replace(std::uintptr_t new_handle) {
  if (impl_) cufftDestroy(to_cufft(impl_));
  impl_ = new_handle;
}

// =========================================================================
// Backend lifecycle
// =========================================================================

void initialize() {
}
std::string backend_name() {
  return "cuda";
}

// =========================================================================
// Helpers
// =========================================================================

static cufftType to_cufft_type(flagfftType type) {
  return static_cast<cufftType>(static_cast<int>(type));
}

static void check_cufft(cufftResult r, const std::string& context) {
  if (r != CUFFT_SUCCESS) {
    throw std::runtime_error(context + " failed with code " + std::to_string(static_cast<int>(r)));
  }
}

// =========================================================================
// Plan creation
// =========================================================================

void ref_plan_1d(RefPlanHandle& plan, int nx, flagfftType type, int batch) {
  cufftHandle h;
  auto r = cufftPlan1d(&h, nx, to_cufft_type(type), batch);
  check_cufft(r, "cufftPlan1d");
  plan.replace(from_cufft(h));
}

void ref_plan_2d(RefPlanHandle& plan, int nx, int ny, flagfftType type) {
  cufftHandle h;
  auto r = cufftPlan2d(&h, nx, ny, to_cufft_type(type));
  check_cufft(r, "cufftPlan2d");
  plan.replace(from_cufft(h));
}

void ref_plan_3d(RefPlanHandle& plan, int nx, int ny, int nz, flagfftType type) {
  cufftHandle h;
  auto r = cufftPlan3d(&h, nx, ny, nz, to_cufft_type(type));
  check_cufft(r, "cufftPlan3d");
  plan.replace(from_cufft(h));
}

void ref_set_stream(RefPlanHandle& plan, flagfftStream_t stream) {
  auto r = cufftSetStream(to_cufft(plan.get()), reinterpret_cast<cudaStream_t>(stream));
  if (r != CUFFT_SUCCESS) {
    throw std::runtime_error("cufftSetStream failed with code " + std::to_string(static_cast<int>(r)));
  }
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

void ref_exec_z2z(RefPlanHandle& plan,
                  flagfftDoubleComplex* idata,
                  flagfftDoubleComplex* odata,
                  int direction) {
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

static float complex_abs(const flagfftComplex& c) {
  return std::sqrt(c.x * c.x + c.y * c.y);
}

static double complex_abs(const flagfftDoubleComplex& c) {
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
  ErrorMetric err {};
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
  ErrorMetric err {};
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

ErrorMetric compute_error(const flagfftComplex* a, const flagfftComplex* b, std::size_t n) {
  return compute_error(reinterpret_cast<const float*>(a), reinterpret_cast<const float*>(b), n * 2);
}

ErrorMetric compute_error(const flagfftDoubleComplex* a, const flagfftDoubleComplex* b, std::size_t n) {
  return compute_error(reinterpret_cast<const double*>(a), reinterpret_cast<const double*>(b), n * 2);
}

}  // namespace flagfft::test_adaptor
