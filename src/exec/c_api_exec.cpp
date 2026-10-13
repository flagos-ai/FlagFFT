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

#include "c_api_internal.hpp"

#include <sstream>

#if defined(FLAGFFT_NPU_ENABLE_SIP_EXECUTION)
#include "adaptor/backend/npu/sip_executor.hpp"
#endif

extern "C" flagfftResult flagfftExecC2C(flagfftHandle handle,
                                        flagfftComplex *idata,
                                        flagfftComplex *odata,
                                        int direction) {
  flagfft::FlagFFTPlan *plan = flagfft::checked_plan(handle);
  if (plan == nullptr || plan->state.destroyed || !plan->state.initialized) {
    return FLAGFFT_INVALID_PLAN;
  }
  if (idata == nullptr || odata == nullptr) {
    return FLAGFFT_INVALID_VALUE;
  }
  if (plan->desc.type != FLAGFFT_C2C) {
    return FLAGFFT_INVALID_TYPE;
  }
  if (direction != FLAGFFT_FORWARD && direction != FLAGFFT_INVERSE) {
    return FLAGFFT_INVALID_VALUE;
  }

  std::lock_guard<std::mutex> lock(plan->mutex);
  const bool inverse = direction == FLAGFFT_INVERSE;
#if defined(FLAGFFT_NPU_ENABLE_SIP_EXECUTION)
  if (plan->npu_sip_enabled && static_cast<void *>(idata) != static_cast<void *>(odata)) {
    void *sip_plan = inverse ? plan->npu_sip_inverse : plan->npu_sip_forward;
    return flagfft::adaptor::npu::sip_plan_execute(sip_plan, idata, odata);
  }
#endif
  const flagfft::FFTRequest &request =
      inverse ? plan->executable.inverse_request : plan->executable.forward_request;
  const std::shared_ptr<flagfft::CompiledRawNode> &compiled =
      inverse ? plan->executable.inverse : plan->executable.forward;
  flagfft::RawExecutionContext context {request,
                                        plan->state.stream,
                                        plan->desc.batch,
                                        plan->desc.idist,
                                        plan->desc.odist};
  return compiled->execute(reinterpret_cast<flagfft::adaptor::DevicePtr>(idata),
                           reinterpret_cast<flagfft::adaptor::DevicePtr>(odata),
                           context);
}

extern "C" flagfftResult flagfftExecZ2Z(flagfftHandle handle,
                                        flagfftDoubleComplex *idata,
                                        flagfftDoubleComplex *odata,
                                        int direction) {
  flagfft::FlagFFTPlan *plan = flagfft::checked_plan(handle);
  if (plan == nullptr || plan->state.destroyed || !plan->state.initialized) {
    return FLAGFFT_INVALID_PLAN;
  }
  if (idata == nullptr || odata == nullptr) {
    return FLAGFFT_INVALID_VALUE;
  }
  if (plan->desc.type != FLAGFFT_Z2Z) {
    return FLAGFFT_INVALID_TYPE;
  }
  if (direction != FLAGFFT_FORWARD && direction != FLAGFFT_INVERSE) {
    return FLAGFFT_INVALID_VALUE;
  }

  std::lock_guard<std::mutex> lock(plan->mutex);
  const bool inverse = direction == FLAGFFT_INVERSE;
  const flagfft::FFTRequest &request =
      inverse ? plan->executable.inverse_request : plan->executable.forward_request;
  const std::shared_ptr<flagfft::CompiledRawNode> &compiled =
      inverse ? plan->executable.inverse : plan->executable.forward;
  flagfft::RawExecutionContext context {request,
                                        plan->state.stream,
                                        plan->desc.batch,
                                        plan->desc.idist,
                                        plan->desc.odist};
  return compiled->execute(reinterpret_cast<flagfft::adaptor::DevicePtr>(idata),
                           reinterpret_cast<flagfft::adaptor::DevicePtr>(odata),
                           context);
}

extern "C" flagfftResult flagfftExecR2C(flagfftHandle handle, flagfftReal *idata, flagfftComplex *odata) {
  flagfft::FlagFFTPlan *plan = flagfft::checked_plan(handle);
  if (plan == nullptr || plan->state.destroyed || !plan->state.initialized) {
    return FLAGFFT_INVALID_PLAN;
  }
  if (idata == nullptr || odata == nullptr) {
    return FLAGFFT_INVALID_VALUE;
  }
  if (plan->desc.type != FLAGFFT_R2C) {
    return FLAGFFT_INVALID_TYPE;
  }

  std::lock_guard<std::mutex> lock(plan->mutex);
#if defined(FLAGFFT_NPU_ENABLE_SIP_EXECUTION)
  if (plan->npu_sip_enabled && static_cast<void *>(idata) != static_cast<void *>(odata)) {
    return flagfft::adaptor::npu::sip_plan_execute(plan->npu_sip_forward, idata, odata);
  }
#endif
  flagfft::RawExecutionContext context {plan->executable.forward_request,
                                        plan->state.stream,
                                        plan->desc.batch,
                                        plan->desc.idist,
                                        plan->desc.odist};
  return plan->executable.forward->execute(reinterpret_cast<flagfft::adaptor::DevicePtr>(idata),
                                           reinterpret_cast<flagfft::adaptor::DevicePtr>(odata),
                                           context);
}

extern "C" flagfftResult flagfftExecD2Z(flagfftHandle handle,
                                        flagfftDoubleReal *idata,
                                        flagfftDoubleComplex *odata) {
  flagfft::FlagFFTPlan *plan = flagfft::checked_plan(handle);
  if (plan == nullptr || plan->state.destroyed || !plan->state.initialized) {
    return FLAGFFT_INVALID_PLAN;
  }
  if (idata == nullptr || odata == nullptr) {
    return FLAGFFT_INVALID_VALUE;
  }
  if (plan->desc.type != FLAGFFT_D2Z) {
    return FLAGFFT_INVALID_TYPE;
  }

  std::lock_guard<std::mutex> lock(plan->mutex);
  flagfft::RawExecutionContext context {plan->executable.forward_request,
                                        plan->state.stream,
                                        plan->desc.batch,
                                        plan->desc.idist,
                                        plan->desc.odist};
  return plan->executable.forward->execute(reinterpret_cast<flagfft::adaptor::DevicePtr>(idata),
                                           reinterpret_cast<flagfft::adaptor::DevicePtr>(odata),
                                           context);
}

extern "C" flagfftResult flagfftExecC2R(flagfftHandle handle, flagfftComplex *idata, flagfftReal *odata) {
  flagfft::FlagFFTPlan *plan = flagfft::checked_plan(handle);
  if (plan == nullptr || plan->state.destroyed || !plan->state.initialized) {
    return FLAGFFT_INVALID_PLAN;
  }
  if (idata == nullptr || odata == nullptr) {
    return FLAGFFT_INVALID_VALUE;
  }
  if (plan->desc.type != FLAGFFT_C2R) {
    return FLAGFFT_INVALID_TYPE;
  }

  std::lock_guard<std::mutex> lock(plan->mutex);
#if defined(FLAGFFT_NPU_ENABLE_SIP_EXECUTION)
  if (plan->npu_sip_enabled && static_cast<void *>(idata) != static_cast<void *>(odata)) {
    return flagfft::adaptor::npu::sip_plan_execute(plan->npu_sip_inverse, idata, odata);
  }
#endif
  flagfft::RawExecutionContext context {plan->executable.inverse_request,
                                        plan->state.stream,
                                        plan->desc.batch,
                                        plan->desc.idist,
                                        plan->desc.odist};
  return plan->executable.inverse->execute(reinterpret_cast<flagfft::adaptor::DevicePtr>(idata),
                                           reinterpret_cast<flagfft::adaptor::DevicePtr>(odata),
                                           context);
}

extern "C" flagfftResult flagfftExecZ2D(flagfftHandle handle,
                                        flagfftDoubleComplex *idata,
                                        flagfftDoubleReal *odata) {
  flagfft::FlagFFTPlan *plan = flagfft::checked_plan(handle);
  if (plan == nullptr || plan->state.destroyed || !plan->state.initialized) {
    return FLAGFFT_INVALID_PLAN;
  }
  if (idata == nullptr || odata == nullptr) {
    return FLAGFFT_INVALID_VALUE;
  }
  if (plan->desc.type != FLAGFFT_Z2D) {
    return FLAGFFT_INVALID_TYPE;
  }

  std::lock_guard<std::mutex> lock(plan->mutex);
  flagfft::RawExecutionContext context {plan->executable.inverse_request,
                                        plan->state.stream,
                                        plan->desc.batch,
                                        plan->desc.idist,
                                        plan->desc.odist};
  return plan->executable.inverse->execute(reinterpret_cast<flagfft::adaptor::DevicePtr>(idata),
                                           reinterpret_cast<flagfft::adaptor::DevicePtr>(odata),
                                           context);
}

extern "C" flagfftResult flagfftSetStream(flagfftHandle handle, flagfftStream_t stream) {
  flagfft::FlagFFTPlan *plan = flagfft::checked_plan(handle);
  if (plan == nullptr || plan->state.destroyed) {
    return FLAGFFT_INVALID_PLAN;
  }
  std::lock_guard<std::mutex> lock(plan->mutex);
#if defined(FLAGFFT_NPU_ENABLE_SIP_EXECUTION)
  if (plan->npu_sip_enabled) {
    const flagfftResult forward_status =
        flagfft::adaptor::npu::sip_plan_set_stream(plan->npu_sip_forward, stream);
    const flagfftResult inverse_status =
        flagfft::adaptor::npu::sip_plan_set_stream(plan->npu_sip_inverse, stream);
    if (forward_status != FLAGFFT_SUCCESS) return forward_status;
    if (inverse_status != FLAGFFT_SUCCESS) return inverse_status;
  }
#endif
  plan->state.stream = stream;
  return FLAGFFT_SUCCESS;
}

extern "C" flagfftResult flagfftDestroy(flagfftHandle handle) {
  if (handle == nullptr) {
    return FLAGFFT_INVALID_PLAN;
  }
  flagfft::FlagFFTPlan *plan = flagfft::checked_plan(handle);
  if (plan != nullptr) {
    {
      std::lock_guard<std::mutex> lock(plan->mutex);
      plan->state.destroyed = true;
    }
    delete plan;
    handle->impl = nullptr;
  }
  delete handle;
  return FLAGFFT_SUCCESS;
}

extern "C" const char *flagfftGetPlanDescription(flagfftHandle handle) {
  flagfft::FlagFFTPlan *plan = flagfft::checked_plan(handle);
  if (plan == nullptr || plan->state.destroyed || !plan->state.initialized) {
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(plan->mutex);
  if (!plan->description_cache.empty()) {
    return plan->description_cache.c_str();
  }

  std::ostringstream oss;
  oss << "=== FlagFFT Plan ===\n";
  oss << "rank=" << plan->desc.rank << " n=[" << plan->desc.n[0];
  for (std::size_t i = 1; i < plan->desc.n.size(); ++i) {
    oss << "," << plan->desc.n[i];
  }
  oss << "] batch=" << plan->desc.batch << " type=" << static_cast<int>(plan->desc.type) << "\n";

#if defined(FLAGFFT_NPU_ENABLE_SIP_EXECUTION)
  oss << "execution_backend="
      << (plan->npu_sip_enabled ? "SiP FFT (CANN; out-of-place; Triton in-place fallback)"
                                : "FlagFFT Triton")
      << "\n";
#endif

  oss << "\n-- Plan tree --\n";
  if (plan->executable.root) {
    oss << plan->executable.root->describe() << "\n";
  } else {
    oss << "(no plan tree)\n";
  }

  oss << "\n-- Forward execution --\n";
  if (plan->executable.forward) {
    oss << plan->executable.forward->describe() << "\n";
  } else {
    oss << "(not compiled)\n";
  }

  if (plan->executable.inverse) {
    oss << "\n-- Inverse execution --\n";
    oss << plan->executable.inverse->describe() << "\n";
  }

  plan->description_cache = oss.str();
  return plan->description_cache.c_str();
}
