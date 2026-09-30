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

#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <aclnn/acl_meta.h>
#include <fft_api.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace flagfft::test_adaptor {
namespace {

  void check_sip(AsdSip::AspbStatus status, const char* context) {
    if (status == AsdSip::ACL_SUCCESS) return;
    throw std::runtime_error(std::string(context) + " failed with SiP status=" +
                             std::to_string(status));
  }

  void check_acl(aclError status, const char* context) {
    if (status == ACL_SUCCESS) return;
    std::string message = std::string(context) + " failed with aclError=" +
                          std::to_string(static_cast<int>(status));
    if (const char* detail = aclGetRecentErrMsg(); detail != nullptr && *detail != '\0') {
      message += ": ";
      message += detail;
    }
    throw std::runtime_error(message);
  }

  AsdSip::asdFftType sip_type(flagfftType type) {
    switch (type) {
      case FLAGFFT_C2C: return AsdSip::ASCEND_FFT_C2C;
      case FLAGFFT_R2C: return AsdSip::ASCEND_FFT_R2C;
      case FLAGFFT_C2R: return AsdSip::ASCEND_FFT_C2R;
      default: throw std::runtime_error("SiP FFT reference supports FP32 C2C/R2C/C2R only");
    }
  }

  AsdSip::asdFftDirection sip_direction(int direction) {
    if (direction == FLAGFFT_FORWARD) return AsdSip::ASCEND_FFT_FORWARD;
    if (direction == FLAGFFT_INVERSE) return AsdSip::ASCEND_FFT_INVERSE;
    throw std::runtime_error("invalid SiP FFT direction");
  }

  struct SipPlan {
    AsdSip::asdFftHandle handle = nullptr;
    void* workspace = nullptr;
    AsdSip::asdFftHandle companion_handle = nullptr;
    void* companion_workspace = nullptr;
    aclTensor* input_tensor = nullptr;
    aclTensor* output_tensor = nullptr;
    void* input_ptr = nullptr;
    void* output_ptr = nullptr;
    std::vector<int64_t> input_shape;
    std::vector<int64_t> output_shape;
    flagfftType type = FLAGFFT_C2C;
    int direction = FLAGFFT_FORWARD;

    ~SipPlan() {
      if (input_tensor != nullptr) (void)aclDestroyTensor(input_tensor);
      if (output_tensor != nullptr) (void)aclDestroyTensor(output_tensor);
      if (companion_workspace != nullptr) (void)aclrtFree(companion_workspace);
      if (companion_handle != nullptr) (void)AsdSip::asdFftDestroy(companion_handle);
      if (workspace != nullptr) (void)aclrtFree(workspace);
      if (handle != nullptr) (void)AsdSip::asdFftDestroy(handle);
    }

    SipPlan() = default;
    SipPlan(const SipPlan&) = delete;
    SipPlan& operator=(const SipPlan&) = delete;
  };

  SipPlan& as_plan(RefPlanHandle& plan) {
    if (plan.get() == 0) throw std::runtime_error("SiP FFT reference plan is null");
    return *reinterpret_cast<SipPlan*>(plan.get());
  }

  aclTensor* make_tensor(const std::vector<int64_t>& shape, aclDataType dtype, void* data) {
    std::vector<int64_t> strides(shape.size());
    int64_t stride = 1;
    for (std::size_t i = shape.size(); i-- > 0;) {
      strides[i] = stride;
      stride *= shape[i];
    }
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0,
                                        ACL_FORMAT_ND, shape.data(), shape.size(), data);
    if (tensor == nullptr) throw std::runtime_error("aclCreateTensor failed for SiP FFT reference");
    return tensor;
  }

  void bind_tensors(SipPlan& plan, void* input, void* output) {
    if (input == output) throw std::runtime_error("SiP FFT reference requires out-of-place buffers");
    if (plan.input_tensor != nullptr && plan.output_tensor != nullptr &&
        plan.input_ptr == input && plan.output_ptr == output) {
      return;
    }
    if (plan.input_tensor != nullptr) (void)aclDestroyTensor(plan.input_tensor);
    if (plan.output_tensor != nullptr) (void)aclDestroyTensor(plan.output_tensor);
    plan.input_tensor = nullptr;
    plan.output_tensor = nullptr;
    plan.input_ptr = nullptr;
    plan.output_ptr = nullptr;

    const aclDataType input_type = plan.type == FLAGFFT_R2C ? ACL_FLOAT : ACL_COMPLEX64;
    const aclDataType output_type = plan.type == FLAGFFT_C2R ? ACL_FLOAT : ACL_COMPLEX64;
    plan.input_tensor = make_tensor(plan.input_shape, input_type, input);
    plan.output_tensor = make_tensor(plan.output_shape, output_type, output);
    plan.input_ptr = input;
    plan.output_ptr = output;
  }

void make_plan(RefPlanHandle& target, std::vector<int64_t> dimensions,
               flagfftType type, int batch) {
    if (batch <= 0 || dimensions.empty() || dimensions.size() > 3 ||
        std::any_of(dimensions.begin(), dimensions.end(), [](int64_t n) { return n <= 0; })) {
      throw std::runtime_error("invalid SiP FFT reference shape or batch");
    }
    auto plan = std::make_unique<SipPlan>();
    plan->type = type;
    plan->direction = target.direction();
    plan->input_shape = dimensions;
    plan->output_shape = dimensions;
    plan->input_shape.insert(plan->input_shape.begin(), batch);
    plan->output_shape.insert(plan->output_shape.begin(), batch);
    if (type == FLAGFFT_C2R) plan->input_shape.back() = dimensions.back() / 2 + 1;
    if (type == FLAGFFT_R2C) plan->output_shape.back() = dimensions.back() / 2 + 1;

    const auto fft_type = sip_type(type);
    auto create_handle = [&](AsdSip::asdFftHandle& handle, void*& workspace, int direction_value) {
      check_sip(AsdSip::asdFftCreate(handle), "asdFftCreate");
      const auto direction = sip_direction(direction_value);
      switch (dimensions.size()) {
        case 1:
          check_sip(AsdSip::asdFftMakePlan1D(handle, dimensions[0], fft_type,
                                              direction, batch, AsdSip::ASCEND_FFT_HORIZONTAL),
                    "asdFftMakePlan1D");
          break;
        case 2:
          check_sip(AsdSip::asdFftMakePlan2D(handle, dimensions[0], dimensions[1],
                                              fft_type, direction, batch), "asdFftMakePlan2D");
          break;
        case 3:
          check_sip(AsdSip::asdFftMakePlan3D(handle, dimensions[0], dimensions[1],
                                              dimensions[2], fft_type, direction, batch),
                    "asdFftMakePlan3D");
          break;
      }
      std::size_t workspace_size = 0;
      check_sip(AsdSip::asdFftGetWorkspaceSize(handle, workspace_size),
                "asdFftGetWorkspaceSize");
      if (workspace_size != 0) {
        check_acl(aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST),
                  "aclrtMalloc(SiP workspace)");
      }
      check_sip(AsdSip::asdFftSetWorkspace(handle, workspace), "asdFftSetWorkspace");
    };

    const bool paired_batched_c2c_16k = type == FLAGFFT_C2C && dimensions.size() == 1 &&
                                        dimensions[0] == 16384 && batch == 64;
    if (paired_batched_c2c_16k && target.direction() == FLAGFFT_INVERSE) {
      // Match FlagFFT's forward-then-inverse SiP plan allocation order. On
      // CANN 9, a lone 16384-point C2C batch-64 reference plan can leave
      // nondeterministic batch outputs; the paired plan layout is stable.
      create_handle(plan->companion_handle, plan->companion_workspace, FLAGFFT_FORWARD);
    }
    create_handle(plan->handle, plan->workspace, target.direction());
    if (paired_batched_c2c_16k && target.direction() == FLAGFFT_FORWARD) {
      create_handle(plan->companion_handle, plan->companion_workspace, FLAGFFT_INVERSE);
    }
    target.replace(reinterpret_cast<std::uintptr_t>(plan.release()));
  }

  template <typename T>
  ErrorMetric scalar_error(const T* a, const T* b, std::size_t n) {
    ErrorMetric error;
    if (n == 0) return error;
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
      if (denominator > 0.0) max_error = std::max(max_error, numerator / denominator);
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
      if (denominator > 0.0) max_error = std::max(max_error, numerator / denominator);
    }
    return max_error;
  }

}  // namespace

RefPlanHandle::RefPlanHandle() : impl_(0) {
}

RefPlanHandle::~RefPlanHandle() {
  delete reinterpret_cast<SipPlan*>(impl_);
}

RefPlanHandle::RefPlanHandle(RefPlanHandle&& other) noexcept
    : impl_(other.impl_), direction_(other.direction_), batch_(other.batch_) {
  other.impl_ = 0;
}

RefPlanHandle& RefPlanHandle::operator=(RefPlanHandle&& other) noexcept {
  if (this != &other) {
    delete reinterpret_cast<SipPlan*>(impl_);
    impl_ = other.impl_;
    direction_ = other.direction_;
    batch_ = other.batch_;
    other.impl_ = 0;
  }
  return *this;
}

std::uintptr_t RefPlanHandle::get() const {
  return impl_;
}

void RefPlanHandle::replace(std::uintptr_t new_handle) {
  delete reinterpret_cast<SipPlan*>(impl_);
  impl_ = new_handle;
}

void ref_plan_1d(RefPlanHandle& plan, int nx, flagfftType type, int batch) {
  make_plan(plan, {nx}, type, batch);
}

void ref_plan_2d(RefPlanHandle& plan, int nx, int ny, flagfftType type) {
  make_plan(plan, {nx, ny}, type, plan.batch());
}

void ref_plan_3d(RefPlanHandle& plan, int nx, int ny, int nz, flagfftType type) {
  make_plan(plan, {nx, ny, nz}, type, plan.batch());
}

void ref_set_stream(RefPlanHandle& plan, flagfftStream_t stream) {
  SipPlan& sip = as_plan(plan);
  check_sip(AsdSip::asdFftSetStream(sip.handle, stream), "asdFftSetStream");
  if (sip.companion_handle != nullptr) {
    check_sip(AsdSip::asdFftSetStream(sip.companion_handle, stream), "asdFftSetStream(companion)");
  }
}

void ref_exec_c2c(RefPlanHandle& plan, flagfftComplex* idata,
                  flagfftComplex* odata, int direction) {
  SipPlan& sip = as_plan(plan);
  if (sip.type != FLAGFFT_C2C || sip.direction != direction) {
    throw std::runtime_error("SiP C2C execution does not match its plan");
  }
  bind_tensors(sip, idata, odata);
  check_sip(AsdSip::asdFftExecC2C(sip.handle, sip.input_tensor, sip.output_tensor),
            "asdFftExecC2C");
}

void ref_exec_z2z(RefPlanHandle&, flagfftDoubleComplex*, flagfftDoubleComplex*, int) {
  throw std::runtime_error("SiP FFT reference does not implement FP64 Z2Z");
}

void ref_exec_r2c(RefPlanHandle& plan, flagfftReal* idata, flagfftComplex* odata) {
  SipPlan& sip = as_plan(plan);
  if (sip.type != FLAGFFT_R2C) throw std::runtime_error("SiP R2C execution does not match its plan");
  bind_tensors(sip, idata, odata);
  check_sip(AsdSip::asdFftExecR2C(sip.handle, sip.input_tensor, sip.output_tensor),
            "asdFftExecR2C");
}

void ref_exec_d2z(RefPlanHandle&, flagfftDoubleReal*, flagfftDoubleComplex*) {
  throw std::runtime_error("SiP FFT reference does not implement FP64 D2Z");
}

void ref_exec_c2r(RefPlanHandle& plan, flagfftComplex* idata, flagfftReal* odata) {
  SipPlan& sip = as_plan(plan);
  if (sip.type != FLAGFFT_C2R) throw std::runtime_error("SiP C2R execution does not match its plan");
  bind_tensors(sip, idata, odata);
  check_sip(AsdSip::asdFftExecC2R(sip.handle, sip.input_tensor, sip.output_tensor),
            "asdFftExecC2R");
}

void ref_exec_z2d(RefPlanHandle&, flagfftDoubleComplex*, flagfftDoubleReal*) {
  throw std::runtime_error("SiP FFT reference does not implement FP64 Z2D");
}

void initialize() {
}

std::string backend_name() {
  return "npu-sip";
}

bool reference_available() {
  return true;
}

bool reference_uses_host_memory() {
  return false;
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
  for (auto& value : values) value = static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f;
  return values;
}

std::vector<flagfftDoubleReal> random_double_real(int n) {
  std::vector<flagfftDoubleReal> values(static_cast<std::size_t>(n));
  for (auto& value : values) value = static_cast<double>(std::rand()) / RAND_MAX * 2.0 - 1.0;
  return values;
}

double max_relative_error(const flagfftComplex* a, const flagfftComplex* b, int n) {
  return relative_complex_error(a, b, n);
}

double max_relative_error(const flagfftDoubleComplex* a, const flagfftDoubleComplex* b, int n) {
  return relative_complex_error(a, b, n);
}

double max_relative_error_real(const flagfftReal* a, const flagfftReal* b, int n) {
  return relative_scalar_error(a, b, n);
}

double max_relative_error_real(const flagfftDoubleReal* a, const flagfftDoubleReal* b, int n) {
  return relative_scalar_error(a, b, n);
}

ErrorMetric compute_error(const float* a, const float* b, std::size_t n) {
  return scalar_error(a, b, n);
}

ErrorMetric compute_error(const double* a, const double* b, std::size_t n) {
  return scalar_error(a, b, n);
}

ErrorMetric compute_error(const flagfftComplex* a, const flagfftComplex* b, std::size_t n) {
  return scalar_error(reinterpret_cast<const float*>(a), reinterpret_cast<const float*>(b), n * 2);
}

ErrorMetric compute_error(const flagfftDoubleComplex* a, const flagfftDoubleComplex* b, std::size_t n) {
  return scalar_error(reinterpret_cast<const double*>(a), reinterpret_cast<const double*>(b), n * 2);
}

}  // namespace flagfft::test_adaptor
