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

#include "adaptor/backend/npu/sip_executor.hpp"

#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <aclnn/acl_meta.h>
#include <fft_api.h>

#include <algorithm>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace flagfft::adaptor::npu {
namespace {

  void check_sip(AsdSip::AspbStatus status, const char *context) {
    if (status == AsdSip::ACL_SUCCESS) return;
    throw std::runtime_error(std::string(context) + " failed with SiP status=" +
                             std::to_string(status));
  }

  void check_acl(aclError status, const char *context) {
    if (status == ACL_SUCCESS) return;
    std::string message = std::string(context) + " failed with aclError=" +
                          std::to_string(static_cast<int>(status));
    if (const char *detail = aclGetRecentErrMsg(); detail != nullptr && *detail != '\0') {
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
      default: throw std::runtime_error("SiP execution supports FP32 C2C/R2C/C2R only");
    }
  }

  AsdSip::asdFftDirection sip_direction(int direction) {
    if (direction == FLAGFFT_FORWARD) return AsdSip::ASCEND_FFT_FORWARD;
    if (direction == FLAGFFT_INVERSE) return AsdSip::ASCEND_FFT_INVERSE;
    throw std::runtime_error("invalid SiP FFT direction");
  }

  struct SipExecutor {
    AsdSip::asdFftHandle handle = nullptr;
    void *workspace = nullptr;
    aclTensor *input_tensor = nullptr;
    aclTensor *output_tensor = nullptr;
    void *input_ptr = nullptr;
    void *output_ptr = nullptr;
    std::vector<int64_t> input_shape;
    std::vector<int64_t> output_shape;
    flagfftType type = FLAGFFT_C2C;
    int direction = FLAGFFT_FORWARD;

    ~SipExecutor() {
      if (input_tensor != nullptr) (void)aclDestroyTensor(input_tensor);
      if (output_tensor != nullptr) (void)aclDestroyTensor(output_tensor);
      if (workspace != nullptr) (void)aclrtFree(workspace);
      if (handle != nullptr) (void)AsdSip::asdFftDestroy(handle);
    }
  };

  aclTensor *make_tensor(const std::vector<int64_t> &shape, aclDataType dtype, void *data) {
    std::vector<int64_t> strides(shape.size());
    int64_t stride = 1;
    for (std::size_t i = shape.size(); i-- > 0;) {
      strides[i] = stride;
      stride *= shape[i];
    }
    aclTensor *tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0,
                                        ACL_FORMAT_ND, shape.data(), shape.size(), data);
    if (tensor == nullptr) throw std::runtime_error("aclCreateTensor failed for SiP execution");
    return tensor;
  }

  void bind_tensors(SipExecutor &plan, void *input, void *output) {
    if (input == output) throw std::runtime_error("SiP execution requires out-of-place buffers");
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
    try {
      plan.output_tensor = make_tensor(plan.output_shape, output_type, output);
    } catch (...) {
      (void)aclDestroyTensor(plan.input_tensor);
      plan.input_tensor = nullptr;
      throw;
    }
    plan.input_ptr = input;
    plan.output_ptr = output;
  }

  SipExecutor &as_plan(void *opaque) {
    if (opaque == nullptr) throw std::runtime_error("SiP execution plan is null");
    return *static_cast<SipExecutor *>(opaque);
  }

}  // namespace

void *sip_plan_create(int64_t length, int64_t batch, flagfftType type, int direction) {
  if (length <= 0 || batch <= 0) throw std::runtime_error("invalid SiP execution shape or batch");
  auto plan = std::make_unique<SipExecutor>();
  plan->type = type;
  plan->direction = direction;
  const int64_t compact_length = length / 2 + 1;
  const int64_t input_length = type == FLAGFFT_C2R ? compact_length : length;
  const int64_t output_length = type == FLAGFFT_R2C ? compact_length : length;
  plan->input_shape = {batch, input_length};
  plan->output_shape = {batch, output_length};

  check_sip(AsdSip::asdFftCreate(plan->handle), "asdFftCreate");
  check_sip(AsdSip::asdFftMakePlan1D(plan->handle, length, sip_type(type), sip_direction(direction), batch,
                                    AsdSip::ASCEND_FFT_HORIZONTAL),
            "asdFftMakePlan1D");
  std::size_t workspace_size = 0;
  check_sip(AsdSip::asdFftGetWorkspaceSize(plan->handle, workspace_size), "asdFftGetWorkspaceSize");
  if (workspace_size != 0) {
    check_acl(aclrtMalloc(&plan->workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST),
              "aclrtMalloc(SiP workspace)");
  }
  check_sip(AsdSip::asdFftSetWorkspace(plan->handle, plan->workspace), "asdFftSetWorkspace");
  return plan.release();
}

void sip_plan_destroy(void *opaque) {
  delete static_cast<SipExecutor *>(opaque);
}

flagfftResult sip_plan_set_stream(void *opaque, StreamHandle stream) {
  if (opaque == nullptr) return FLAGFFT_SUCCESS;
  try {
    check_sip(AsdSip::asdFftSetStream(as_plan(opaque).handle, reinterpret_cast<aclrtStream>(stream)),
              "asdFftSetStream");
    return FLAGFFT_SUCCESS;
  } catch (...) {
    return FLAGFFT_EXEC_FAILED;
  }
}

flagfftResult sip_plan_execute(void *opaque, void *input, void *output) {
  try {
    SipExecutor &plan = as_plan(opaque);
    bind_tensors(plan, input, output);
    switch (plan.type) {
      case FLAGFFT_C2C:
        check_sip(AsdSip::asdFftExecC2C(plan.handle, plan.input_tensor, plan.output_tensor),
                  "asdFftExecC2C");
        break;
      case FLAGFFT_R2C:
        check_sip(AsdSip::asdFftExecR2C(plan.handle, plan.input_tensor, plan.output_tensor),
                  "asdFftExecR2C");
        break;
      case FLAGFFT_C2R:
        check_sip(AsdSip::asdFftExecC2R(plan.handle, plan.input_tensor, plan.output_tensor),
                  "asdFftExecC2R");
        break;
      default:
        throw std::runtime_error("unsupported SiP execution transform type");
    }
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] SiP execution failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

}  // namespace flagfft::adaptor::npu
