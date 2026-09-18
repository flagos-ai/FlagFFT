// Copyright 2026 FlagOS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "adaptor/test_adaptor.h"

#include <mcfft.h>
#include <mcr/mc_runtime.h>

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
// RefPlanHandle - mcFFT implementation
// =========================================================================

static mcfftHandle to_mcfft(std::uintptr_t v) {
  return static_cast<mcfftHandle>(static_cast<std::intptr_t>(v));
}
static std::uintptr_t from_mcfft(mcfftHandle h) {
  return static_cast<std::uintptr_t>(static_cast<std::intptr_t>(h));
}

RefPlanHandle::RefPlanHandle() : impl_(0) {
  mcfftHandle h;
  if (mcfftCreate(&h) != MCFFT_SUCCESS) {
    std::fprintf(stderr, "RefPlanHandle: mcfftCreate failed\n");
    return;
  }
  impl_ = from_mcfft(h);
}

RefPlanHandle::~RefPlanHandle() {
  if (impl_) mcfftDestroy(to_mcfft(impl_));
}

RefPlanHandle::RefPlanHandle(RefPlanHandle&& other) noexcept : impl_(other.impl_) {
  other.impl_ = 0;
}

RefPlanHandle& RefPlanHandle::operator=(RefPlanHandle&& other) noexcept {
  if (this != &other) {
    if (impl_) mcfftDestroy(to_mcfft(impl_));
    impl_ = other.impl_;
    other.impl_ = 0;
  }
  return *this;
}

std::uintptr_t RefPlanHandle::get() const {
  return impl_;
}
void RefPlanHandle::replace(std::uintptr_t new_handle) {
  if (impl_) mcfftDestroy(to_mcfft(impl_));
  impl_ = new_handle;
}

// =========================================================================
// Backend lifecycle
// =========================================================================

void initialize() {
}
std::string backend_name() {
  return "maca";
}
bool reference_available() {
  return true;
}
bool reference_uses_host_memory() {
  return false;
}

// =========================================================================
// Helpers
// =========================================================================

static mcfftType to_mcfft_type(flagfftType type) {
  return static_cast<mcfftType>(static_cast<int>(type));
}

static void check_mcfft(mcfftResult r, const std::string& context) {
  if (r != MCFFT_SUCCESS) {
    throw std::runtime_error(context + " failed with code " + std::to_string(static_cast<int>(r)));
  }
}

// =========================================================================
// Plan creation
// =========================================================================

void ref_plan_1d(RefPlanHandle& plan, int nx, flagfftType type, int batch) {
  mcfftHandle h;
  auto r = mcfftPlan1d(&h, nx, to_mcfft_type(type), batch);
  check_mcfft(r, "mcfftPlan1d");
  plan.replace(from_mcfft(h));
}

void ref_plan_2d(RefPlanHandle& plan, int nx, int ny, flagfftType type) {
  mcfftHandle h;
  auto r = mcfftPlan2d(&h, nx, ny, to_mcfft_type(type));
  check_mcfft(r, "mcfftPlan2d");
  plan.replace(from_mcfft(h));
}

void ref_plan_3d(RefPlanHandle& plan, int nx, int ny, int nz, flagfftType type) {
  mcfftHandle h;
  auto r = mcfftPlan3d(&h, nx, ny, nz, to_mcfft_type(type));
  check_mcfft(r, "mcfftPlan3d");
  plan.replace(from_mcfft(h));
}

void ref_set_stream(RefPlanHandle& plan, flagfftStream_t stream) {
  auto r = mcfftSetStream(to_mcfft(plan.get()), reinterpret_cast<mcStream_t>(stream));
  if (r != MCFFT_SUCCESS) {
    throw std::runtime_error("mcfftSetStream failed with code " + std::to_string(static_cast<int>(r)));
  }
}

// =========================================================================
// Plan execution
// =========================================================================

void ref_exec_c2c(RefPlanHandle& plan, flagfftComplex* idata, flagfftComplex* odata, int direction) {
  check_mcfft(mcfftExecC2C(to_mcfft(plan.get()),
                           reinterpret_cast<mcfftComplex*>(idata),
                           reinterpret_cast<mcfftComplex*>(odata),
                           direction),
              "mcfftExecC2C");
}

void ref_exec_z2z(RefPlanHandle& plan,
                  flagfftDoubleComplex* idata,
                  flagfftDoubleComplex* odata,
                  int direction) {
  check_mcfft(mcfftExecZ2Z(to_mcfft(plan.get()),
                           reinterpret_cast<mcfftDoubleComplex*>(idata),
                           reinterpret_cast<mcfftDoubleComplex*>(odata),
                           direction),
              "mcfftExecZ2Z");
}

void ref_exec_r2c(RefPlanHandle& plan, flagfftReal* idata, flagfftComplex* odata) {
  check_mcfft(mcfftExecR2C(to_mcfft(plan.get()),
                           reinterpret_cast<mcfftReal*>(idata),
                           reinterpret_cast<mcfftComplex*>(odata)),
              "mcfftExecR2C");
}

void ref_exec_d2z(RefPlanHandle& plan, flagfftDoubleReal* idata, flagfftDoubleComplex* odata) {
  check_mcfft(mcfftExecD2Z(to_mcfft(plan.get()),
                           reinterpret_cast<mcfftDoubleReal*>(idata),
                           reinterpret_cast<mcfftDoubleComplex*>(odata)),
              "mcfftExecD2Z");
}

void ref_exec_c2r(RefPlanHandle& plan, flagfftComplex* idata, flagfftReal* odata) {
  check_mcfft(mcfftExecC2R(to_mcfft(plan.get()),
                           reinterpret_cast<mcfftComplex*>(idata),
                           reinterpret_cast<mcfftReal*>(odata)),
              "mcfftExecC2R");
}

void ref_exec_z2d(RefPlanHandle& plan, flagfftDoubleComplex* idata, flagfftDoubleReal* odata) {
  check_mcfft(mcfftExecZ2D(to_mcfft(plan.get()),
                           reinterpret_cast<mcfftDoubleComplex*>(idata),
                           reinterpret_cast<mcfftDoubleReal*>(odata)),
              "mcfftExecZ2D");
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
