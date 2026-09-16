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

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

#include <cann_ops_fft.h>

namespace flagfft::test_adaptor {
namespace {

aclfftHandle as_ops_handle(std::uintptr_t value) {
  return reinterpret_cast<aclfftHandle>(value);
}

std::uintptr_t from_ops_handle(aclfftHandle value) {
  return reinterpret_cast<std::uintptr_t>(value);
}

void check_ops(aclfftResult result, const char* context) {
  if (result == ACLFFT_SUCCESS) {
    return;
  }
  const char* detail = aclfftGetErrorString(result);
  std::string message = std::string(context) + " failed with aclfftResult=" +
                        std::to_string(static_cast<int>(result));
  if (detail != nullptr && *detail != '\0') {
    message += ": ";
    message += detail;
  }
  throw std::runtime_error(message);
}

template <typename T>
ErrorMetric scalar_error(const T* a, const T* b, std::size_t n) {
  ErrorMetric error;
  if (n == 0) {
    return error;
  }
  double sum_sq = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    error.max_abs = std::max(error.max_abs, std::abs(diff));
    sum_sq += diff * diff;
  }
  error.rms = std::sqrt(sum_sq / static_cast<double>(n));
  return error;
}

template <typename T>
double relative_scalar_error(const T* a, const T* b, int n) {
  double max_error = 0.0;
  for (int i = 0; i < n; ++i) {
    const double numerator = std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
    const double denominator = std::abs(static_cast<double>(b[i]));
    if (denominator > 0.0) {
      max_error = std::max(max_error, numerator / denominator);
    }
  }
  return max_error;
}

template <typename T>
double complex_magnitude(const T& value) {
  return std::sqrt(static_cast<double>(value.x) * static_cast<double>(value.x) +
                   static_cast<double>(value.y) * static_cast<double>(value.y));
}

template <typename T>
double relative_complex_error(const T* a, const T* b, int n) {
  double max_error = 0.0;
  for (int i = 0; i < n; ++i) {
    const double numerator = std::abs(complex_magnitude(a[i]) - complex_magnitude(b[i]));
    const double denominator = complex_magnitude(b[i]);
    if (denominator > 0.0) {
      max_error = std::max(max_error, numerator / denominator);
    }
  }
  return max_error;
}

}  // namespace

RefPlanHandle::RefPlanHandle() : impl_(0) {}

RefPlanHandle::~RefPlanHandle() {
  if (impl_ != 0) {
    (void)aclfftDestroy(as_ops_handle(impl_));
  }
}

RefPlanHandle::RefPlanHandle(RefPlanHandle&& other) noexcept : impl_(other.impl_) {
  other.impl_ = 0;
}

RefPlanHandle& RefPlanHandle::operator=(RefPlanHandle&& other) noexcept {
  if (this != &other) {
    if (impl_ != 0) {
      (void)aclfftDestroy(as_ops_handle(impl_));
    }
    impl_ = other.impl_;
    other.impl_ = 0;
  }
  return *this;
}

std::uintptr_t RefPlanHandle::get() const {
  return impl_;
}

void RefPlanHandle::replace(std::uintptr_t new_handle) {
  if (impl_ != 0) {
    (void)aclfftDestroy(as_ops_handle(impl_));
  }
  impl_ = new_handle;
}

void ref_plan_1d(RefPlanHandle& plan, int nx, flagfftType type, int batch) {
  aclfftHandle handle = nullptr;
  check_ops(aclfftPlan1d(&handle,
                         nx,
                         static_cast<aclfftType>(type),
                         batch,
                         ACLFFT_HORIZONTAL),
            "aclfftPlan1d");
  plan.replace(from_ops_handle(handle));
}

void ref_plan_2d(RefPlanHandle& plan, int nx, int ny, flagfftType type) {
  aclfftHandle handle = nullptr;
  check_ops(aclfftPlan2d(&handle, 1, nx, ny, static_cast<aclfftType>(type)), "aclfftPlan2d");
  plan.replace(from_ops_handle(handle));
}

void ref_plan_3d(RefPlanHandle&, int, int, int, flagfftType) {
  throw std::runtime_error("ops-fft does not implement 3D plans on Ascend 910B");
}

void ref_set_stream(RefPlanHandle& plan, flagfftStream_t stream) {
  check_ops(aclfftSetStream(as_ops_handle(plan.get()), reinterpret_cast<aclrtStream>(stream)),
            "aclfftSetStream");
}

void ref_exec_c2c(RefPlanHandle& plan,
                  flagfftComplex* idata,
                  flagfftComplex* odata,
                  int direction) {
  check_ops(aclfftExecC2C(as_ops_handle(plan.get()),
                          reinterpret_cast<aclfftComplex*>(idata),
                          reinterpret_cast<aclfftComplex*>(odata),
                          direction),
            "aclfftExecC2C");
}

void ref_exec_z2z(RefPlanHandle&, flagfftDoubleComplex*, flagfftDoubleComplex*, int) {
  throw std::runtime_error("ops-fft does not implement FP64 Z2Z");
}

void ref_exec_r2c(RefPlanHandle& plan, flagfftReal* idata, flagfftComplex* odata) {
  check_ops(aclfftExecR2C(as_ops_handle(plan.get()),
                          reinterpret_cast<aclfftReal*>(idata),
                          reinterpret_cast<aclfftComplex*>(odata)),
            "aclfftExecR2C");
}

void ref_exec_d2z(RefPlanHandle&, flagfftDoubleReal*, flagfftDoubleComplex*) {
  throw std::runtime_error("ops-fft does not implement FP64 D2Z");
}

void ref_exec_c2r(RefPlanHandle& plan, flagfftComplex* idata, flagfftReal* odata) {
  check_ops(aclfftExecC2R(as_ops_handle(plan.get()),
                          reinterpret_cast<aclfftComplex*>(idata),
                          reinterpret_cast<aclfftReal*>(odata)),
            "aclfftExecC2R");
}

void ref_exec_z2d(RefPlanHandle&, flagfftDoubleComplex*, flagfftDoubleReal*) {
  throw std::runtime_error("ops-fft does not implement FP64 Z2D");
}

void initialize() {}

std::string backend_name() {
  return "npu-ops-fft";
}

bool reference_available() {
  return true;
}

bool reference_uses_host_memory() {
  // ops-fft's public aclfftExec* API accepts host pointers and performs the
  // H2D/D2H transfers internally.
  return true;
}

std::vector<flagfftComplex> random_complex(int n) {
  std::vector<flagfftComplex> values(static_cast<std::size_t>(n));
  for (auto& value : values) {
    value.x = static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f;
    value.y = static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f;
  }
  return values;
}

std::vector<flagfftDoubleComplex> random_double_complex(int n) {
  std::vector<flagfftDoubleComplex> values(static_cast<std::size_t>(n));
  for (auto& value : values) {
    value.x = static_cast<double>(std::rand()) / RAND_MAX * 2.0 - 1.0;
    value.y = static_cast<double>(std::rand()) / RAND_MAX * 2.0 - 1.0;
  }
  return values;
}

std::vector<flagfftReal> random_real(int n) {
  std::vector<flagfftReal> values(static_cast<std::size_t>(n));
  for (auto& value : values) {
    value = static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f;
  }
  return values;
}

std::vector<flagfftDoubleReal> random_double_real(int n) {
  std::vector<flagfftDoubleReal> values(static_cast<std::size_t>(n));
  for (auto& value : values) {
    value = static_cast<double>(std::rand()) / RAND_MAX * 2.0 - 1.0;
  }
  return values;
}

double max_relative_error(const flagfftComplex* a, const flagfftComplex* b, int n) {
  return relative_complex_error(a, b, n);
}

double max_relative_error(const flagfftDoubleComplex* a,
                          const flagfftDoubleComplex* b,
                          int n) {
  return relative_complex_error(a, b, n);
}

double max_relative_error_real(const flagfftReal* a, const flagfftReal* b, int n) {
  return relative_scalar_error(a, b, n);
}

double max_relative_error_real(const flagfftDoubleReal* a,
                               const flagfftDoubleReal* b,
                               int n) {
  return relative_scalar_error(a, b, n);
}

ErrorMetric compute_error(const float* a, const float* b, std::size_t n) {
  return scalar_error(a, b, n);
}

ErrorMetric compute_error(const double* a, const double* b, std::size_t n) {
  return scalar_error(a, b, n);
}

ErrorMetric compute_error(const flagfftComplex* a,
                          const flagfftComplex* b,
                          std::size_t n) {
  return scalar_error(reinterpret_cast<const float*>(a),
                      reinterpret_cast<const float*>(b), n * 2);
}

ErrorMetric compute_error(const flagfftDoubleComplex* a,
                          const flagfftDoubleComplex* b,
                          std::size_t n) {
  return scalar_error(reinterpret_cast<const double*>(a),
                      reinterpret_cast<const double*>(b), n * 2);
}

}  // namespace flagfft::test_adaptor
