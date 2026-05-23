#include "c_api_internal.hpp"

#include <sstream>

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
    const flagfft::FFTRequest &request = inverse ? plan->executable.inverse_request
                                                 : plan->executable.forward_request;
    const std::shared_ptr<flagfft::CompiledRawNode> &compiled =
        inverse ? plan->executable.inverse : plan->executable.forward;
    flagfft::RawExecutionContext context{
        request, plan->state.stream, plan->desc.batch, plan->desc.idist, plan->desc.odist};
    return compiled->execute(reinterpret_cast<flagfft::runtime::DevicePtr>(idata),
                             reinterpret_cast<flagfft::runtime::DevicePtr>(odata),
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
    const flagfft::FFTRequest &request = inverse ? plan->executable.inverse_request
                                                 : plan->executable.forward_request;
    const std::shared_ptr<flagfft::CompiledRawNode> &compiled =
        inverse ? plan->executable.inverse : plan->executable.forward;
    flagfft::RawExecutionContext context{
        request, plan->state.stream, plan->desc.batch, plan->desc.idist, plan->desc.odist};
    return compiled->execute(reinterpret_cast<flagfft::runtime::DevicePtr>(idata),
                             reinterpret_cast<flagfft::runtime::DevicePtr>(odata),
                             context);
}

extern "C" flagfftResult flagfftExecR2C(flagfftHandle handle,
                                        flagfftReal *idata,
                                        flagfftComplex *odata) {
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
    flagfft::RawExecutionContext context{
        plan->executable.forward_request, plan->state.stream, plan->desc.batch,
        plan->desc.idist, plan->desc.odist};
    return plan->executable.forward->execute(reinterpret_cast<flagfft::runtime::DevicePtr>(idata),
                                             reinterpret_cast<flagfft::runtime::DevicePtr>(odata),
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
    flagfft::RawExecutionContext context{
        plan->executable.forward_request, plan->state.stream, plan->desc.batch,
        plan->desc.idist, plan->desc.odist};
    return plan->executable.forward->execute(reinterpret_cast<flagfft::runtime::DevicePtr>(idata),
                                             reinterpret_cast<flagfft::runtime::DevicePtr>(odata),
                                             context);
}

extern "C" flagfftResult flagfftExecC2R(flagfftHandle handle,
                                        flagfftComplex *idata,
                                        flagfftReal *odata) {
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
    flagfft::RawExecutionContext context{
        plan->executable.inverse_request, plan->state.stream, plan->desc.batch,
        plan->desc.idist, plan->desc.odist};
    return plan->executable.inverse->execute(reinterpret_cast<flagfft::runtime::DevicePtr>(idata),
                                             reinterpret_cast<flagfft::runtime::DevicePtr>(odata),
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
    flagfft::RawExecutionContext context{
        plan->executable.inverse_request, plan->state.stream, plan->desc.batch,
        plan->desc.idist, plan->desc.odist};
    return plan->executable.inverse->execute(reinterpret_cast<flagfft::runtime::DevicePtr>(idata),
                                             reinterpret_cast<flagfft::runtime::DevicePtr>(odata),
                                             context);
}

extern "C" flagfftResult flagfftSetStream(flagfftHandle handle, cudaStream_t stream) {
    flagfft::FlagFFTPlan *plan = flagfft::checked_plan(handle);
    if (plan == nullptr || plan->state.destroyed) {
        return FLAGFFT_INVALID_PLAN;
    }
    std::lock_guard<std::mutex> lock(plan->mutex);
    plan->state.stream = reinterpret_cast<flagfft::runtime::StreamHandle>(stream);
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
    oss << "rank=" << plan->desc.rank
        << " n=[" << plan->desc.n[0];
    for (std::size_t i = 1; i < plan->desc.n.size(); ++i) {
        oss << "," << plan->desc.n[i];
    }
    oss << "] batch=" << plan->desc.batch
        << " type=" << static_cast<int>(plan->desc.type) << "\n";

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
