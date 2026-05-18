#include "c_api_internal.hpp"

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
    if (idata == odata) {
        return FLAGFFT_NOT_SUPPORTED;
    }
    if (plan->desc.type != FLAGFFT_C2C) {
        return FLAGFFT_INVALID_TYPE;
    }
    if (direction != FLAGFFT_FORWARD && direction != FLAGFFT_INVERSE) {
        return FLAGFFT_INVALID_VALUE;
    }

    std::lock_guard<std::mutex> lock(plan->mutex);
    const bool inverse = direction == FLAGFFT_INVERSE;
    const flagfft::FFTRequest &request = inverse ? plan->executable.inverse_request
                                                 : plan->executable.forward_request;
    const std::shared_ptr<flagfft::CompiledRawNode> &compiled =
        inverse ? plan->executable.inverse : plan->executable.forward;
    flagfft::RawExecutionContext context{request, plan->state.stream, plan->desc.batch};
    return compiled->execute(reinterpret_cast<CUdeviceptr>(idata),
                             reinterpret_cast<CUdeviceptr>(odata),
                             context);
}

extern "C" flagfftResult flagfftExecZ2Z(flagfftHandle handle,
                                        flagfftDoubleComplex *idata,
                                        flagfftDoubleComplex *odata,
                                        int direction) {
    (void)idata;
    (void)odata;
    (void)direction;
    return flagfft::checked_plan(handle) == nullptr ? FLAGFFT_INVALID_PLAN : FLAGFFT_NOT_SUPPORTED;
}

extern "C" flagfftResult flagfftExecR2C(flagfftHandle handle,
                                        flagfftReal *idata,
                                        flagfftComplex *odata) {
    (void)idata;
    (void)odata;
    return flagfft::checked_plan(handle) == nullptr ? FLAGFFT_INVALID_PLAN : FLAGFFT_NOT_SUPPORTED;
}

extern "C" flagfftResult flagfftExecD2Z(flagfftHandle handle,
                                        flagfftDoubleReal *idata,
                                        flagfftDoubleComplex *odata) {
    (void)idata;
    (void)odata;
    return flagfft::checked_plan(handle) == nullptr ? FLAGFFT_INVALID_PLAN : FLAGFFT_NOT_SUPPORTED;
}

extern "C" flagfftResult flagfftExecC2R(flagfftHandle handle,
                                        flagfftComplex *idata,
                                        flagfftReal *odata) {
    (void)idata;
    (void)odata;
    return flagfft::checked_plan(handle) == nullptr ? FLAGFFT_INVALID_PLAN : FLAGFFT_NOT_SUPPORTED;
}

extern "C" flagfftResult flagfftExecZ2D(flagfftHandle handle,
                                        flagfftDoubleComplex *idata,
                                        flagfftDoubleReal *odata) {
    (void)idata;
    (void)odata;
    return flagfft::checked_plan(handle) == nullptr ? FLAGFFT_INVALID_PLAN : FLAGFFT_NOT_SUPPORTED;
}

extern "C" flagfftResult flagfftSetStream(flagfftHandle handle, cudaStream_t stream) {
    flagfft::FlagFFTPlan *plan = flagfft::checked_plan(handle);
    if (plan == nullptr || plan->state.destroyed) {
        return FLAGFFT_INVALID_PLAN;
    }
    std::lock_guard<std::mutex> lock(plan->mutex);
    plan->state.stream = reinterpret_cast<CUstream>(stream);
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
