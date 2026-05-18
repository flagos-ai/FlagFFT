#include "flagfft/core.hpp"

struct flagfftPlan_t {
    void *impl = nullptr;
};

namespace flagfft {
namespace {

enum class FlagFFTPrecision { Float32, Float64 };
enum class FlagFFTTransformKind { C2C, Z2Z, R2C, D2Z, C2R, Z2D };

struct FlagFFTPlanDesc {
    int rank = 0;
    std::vector<int64_t> n;
    std::vector<int64_t> inembed;
    int64_t istride = 1;
    int64_t idist = 0;
    std::vector<int64_t> onembed;
    int64_t ostride = 1;
    int64_t odist = 0;
    int64_t batch = 1;
    flagfftType type = FLAGFFT_C2C;
    FlagFFTPrecision precision = FlagFFTPrecision::Float32;
    FlagFFTTransformKind kind = FlagFFTTransformKind::C2C;
    bool real_input = false;
    bool real_output = false;
    int device_index = 0;
    std::string device_arch;
};

struct FlagFFTPlanState {
    bool initialized = false;
    bool destroyed = false;
    CUstream stream = nullptr;
    flagfftResult last_error = FLAGFFT_SUCCESS;
};

struct FlagFFTExecutable {
    FFTRequest forward_request;
    FFTRequest inverse_request;
    ProblemKey forward_problem_key;
    ProblemKey inverse_problem_key;
    PlanKey plan_key;
    PlanNodePtr root;
    std::shared_ptr<CompiledRawNode> forward;
    std::shared_ptr<CompiledRawNode> inverse;
};

struct FlagFFTPlan {
    FlagFFTPlanDesc desc;
    FlagFFTPlanState state;
    FlagFFTExecutable executable;
    std::mutex mutex;
};

flagfftResult type_metadata(flagfftType type,
                            FlagFFTPrecision &precision,
                            FlagFFTTransformKind &kind,
                            bool &real_input,
                            bool &real_output) {
    switch (type) {
        case FLAGFFT_C2C:
            precision = FlagFFTPrecision::Float32;
            kind = FlagFFTTransformKind::C2C;
            real_input = false;
            real_output = false;
            return FLAGFFT_SUCCESS;
        case FLAGFFT_Z2Z:
            precision = FlagFFTPrecision::Float64;
            kind = FlagFFTTransformKind::Z2Z;
            real_input = false;
            real_output = false;
            return FLAGFFT_SUCCESS;
        case FLAGFFT_R2C:
            precision = FlagFFTPrecision::Float32;
            kind = FlagFFTTransformKind::R2C;
            real_input = true;
            real_output = false;
            return FLAGFFT_SUCCESS;
        case FLAGFFT_D2Z:
            precision = FlagFFTPrecision::Float64;
            kind = FlagFFTTransformKind::D2Z;
            real_input = true;
            real_output = false;
            return FLAGFFT_SUCCESS;
        case FLAGFFT_C2R:
            precision = FlagFFTPrecision::Float32;
            kind = FlagFFTTransformKind::C2R;
            real_input = false;
            real_output = true;
            return FLAGFFT_SUCCESS;
        case FLAGFFT_Z2D:
            precision = FlagFFTPrecision::Float64;
            kind = FlagFFTTransformKind::Z2D;
            real_input = false;
            real_output = true;
            return FLAGFFT_SUCCESS;
    }
    return FLAGFFT_INVALID_TYPE;
}

flagfftResult ensure_current_cuda_device(int &device_index, std::string &device_arch) {
    CUresult result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        return FLAGFFT_INVALID_DEVICE;
    }

    CUcontext context = nullptr;
    result = cuCtxGetCurrent(&context);
    if (result != CUDA_SUCCESS) {
        return FLAGFFT_INVALID_DEVICE;
    }

    CUdevice device = 0;
    if (context == nullptr) {
        result = cuDeviceGet(&device, 0);
        if (result != CUDA_SUCCESS) {
            return FLAGFFT_INVALID_DEVICE;
        }
        result = cuDevicePrimaryCtxRetain(&context, device);
        if (result != CUDA_SUCCESS) {
            return FLAGFFT_INVALID_DEVICE;
        }
        result = cuCtxSetCurrent(context);
        if (result != CUDA_SUCCESS) {
            return FLAGFFT_INVALID_DEVICE;
        }
    } else {
        result = cuCtxGetDevice(&device);
        if (result != CUDA_SUCCESS) {
            return FLAGFFT_INVALID_DEVICE;
        }
    }

    int major = 0;
    int minor = 0;
    result = cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    if (result != CUDA_SUCCESS) {
        return FLAGFFT_INVALID_DEVICE;
    }
    result = cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
    if (result != CUDA_SUCCESS) {
        return FLAGFFT_INVALID_DEVICE;
    }

    device_index = static_cast<int>(device);
    device_arch = "sm_" + std::to_string(major) + std::to_string(minor);
    return FLAGFFT_SUCCESS;
}

std::vector<int64_t> copy_dims(const int *values, int rank) {
    std::vector<int64_t> out;
    out.reserve(static_cast<std::size_t>(rank));
    for (int i = 0; i < rank; ++i) {
        out.push_back(static_cast<int64_t>(values[i]));
    }
    return out;
}

bool is_supported_minimal_desc(const FlagFFTPlanDesc &desc) {
    if (desc.rank != 1 || desc.type != FLAGFFT_C2C || desc.batch <= 0) {
        return false;
    }
    if (desc.n.size() != 1 || desc.n[0] <= 0) {
        return false;
    }
    return desc.istride == 1 && desc.ostride == 1 && desc.idist == desc.n[0] &&
           desc.odist == desc.n[0] && desc.inembed.size() == 1 &&
           desc.onembed.size() == 1 && desc.inembed[0] == desc.n[0] &&
           desc.onembed[0] == desc.n[0];
}

bool raw_supported_node(const PlanNodePtr &node) {
    if (std::dynamic_pointer_cast<LeafPlanNode>(node) != nullptr) {
        return true;
    }
    if (auto four_step = std::dynamic_pointer_cast<FourStepPlanNode>(node)) {
        return std::dynamic_pointer_cast<LeafPlanNode>(four_step->row_plan) != nullptr &&
               std::dynamic_pointer_cast<LeafPlanNode>(four_step->col_plan) != nullptr;
    }
    return false;
}

FFTRequest request_from_desc(const FlagFFTPlanDesc &desc, std::string direction) {
    FFTRequest request;
    request.fft_length = desc.n[0];
    request.input_shape = {desc.batch, desc.n[0]};
    request.input_strides = {desc.n[0], 1};
    request.n = std::nullopt;
    request.requested_n = desc.n[0];
    request.raw_dim = 1;
    request.normalized_dim = 1;
    request.norm = "backward";
    request.input_dtype = "complex64";
    request.output_dtype = "complex64";
    request.device_type = "cuda";
    request.device_index = desc.device_index;
    request.device_arch = desc.device_arch;
    request.input_layout = "contiguous";
    request.requires_contiguous_copy = false;
    request.direction = std::move(direction);
    request.batch = desc.batch;
    return request;
}

flagfftResult build_plan(flagfftHandle *out, FlagFFTPlanDesc desc) {
    if (out == nullptr) {
        return FLAGFFT_INVALID_VALUE;
    }
    *out = nullptr;
    if (!is_supported_minimal_desc(desc)) {
        return FLAGFFT_NOT_SUPPORTED;
    }

    flagfftResult device_result = ensure_current_cuda_device(desc.device_index, desc.device_arch);
    if (device_result != FLAGFFT_SUCCESS) {
        return device_result;
    }

    try {
        auto plan = std::make_unique<FlagFFTPlan>();
        plan->desc = std::move(desc);
        plan->executable.forward_request = request_from_desc(plan->desc, "forward");
        plan->executable.inverse_request = request_from_desc(plan->desc, "inverse");

        PlanBuilder builder;
        plan->executable.root =
            builder.build(plan->executable.forward_request.requested_n,
                          plan->executable.forward_request);
        if (!raw_supported_node(plan->executable.root)) {
            return FLAGFFT_NOT_SUPPORTED;
        }

        plan->executable.forward_problem_key =
            ProblemKey::from_request(plan->executable.forward_request);
        plan->executable.inverse_problem_key =
            ProblemKey::from_request(plan->executable.inverse_request);
        plan->executable.plan_key = PlanKey::from_node(plan->executable.root);

        TritonCompiler compiler;
        plan->executable.forward = compiler.compile_raw_node(
            plan->executable.root, plan->executable.forward_request, plan->desc.batch);
        plan->executable.inverse = compiler.compile_raw_node(
            plan->executable.root, plan->executable.inverse_request, plan->desc.batch);
        plan->state.initialized = true;

        std::unique_ptr<flagfftPlan_t> handle(new flagfftPlan_t());
        handle->impl = plan.release();
        *out = handle.release();
        return FLAGFFT_SUCCESS;
    } catch (const nb::python_error &) {
        if (Py_IsInitialized()) {
            PyErr_Clear();
        }
        return FLAGFFT_SETUP_FAILED;
    } catch (const std::bad_alloc &) {
        return FLAGFFT_ALLOC_FAILED;
    } catch (const std::exception &) {
        return FLAGFFT_SETUP_FAILED;
    }
}

FlagFFTPlan *checked_plan(flagfftHandle handle) {
    if (handle == nullptr || handle->impl == nullptr) {
        return nullptr;
    }
    return static_cast<FlagFFTPlan *>(handle->impl);
}

}  // namespace
}  // namespace flagfft

extern "C" flagfftResult flagfftPlan1d(flagfftHandle *plan,
                                       int nx,
                                       flagfftType type,
                                       int batch) {
    int n[1] = {nx};
    return flagfftPlanMany(plan, 1, n, nullptr, 1, nx, nullptr, 1, nx, type, batch);
}

extern "C" flagfftResult flagfftPlan2d(flagfftHandle *plan,
                                       int nx,
                                       int ny,
                                       flagfftType type) {
    int n[2] = {nx, ny};
    return flagfftPlanMany(plan, 2, n, nullptr, 1, nx * ny, nullptr, 1, nx * ny, type, 1);
}

extern "C" flagfftResult flagfftPlan3d(flagfftHandle *plan,
                                       int nx,
                                       int ny,
                                       int nz,
                                       flagfftType type) {
    int n[3] = {nx, ny, nz};
    return flagfftPlanMany(plan, 3, n, nullptr, 1, nx * ny * nz, nullptr, 1, nx * ny * nz, type, 1);
}

extern "C" flagfftResult flagfftPlanMany(flagfftHandle *plan,
                                         int rank,
                                         int *n,
                                         int *inembed,
                                         int istride,
                                         int idist,
                                         int *onembed,
                                         int ostride,
                                         int odist,
                                         flagfftType type,
                                         int batch) {
    if (plan == nullptr) {
        return FLAGFFT_INVALID_VALUE;
    }
    *plan = nullptr;
    if (rank <= 0 || n == nullptr || batch <= 0) {
        return FLAGFFT_INVALID_VALUE;
    }
    for (int i = 0; i < rank; ++i) {
        if (n[i] <= 0) {
            return FLAGFFT_INVALID_SIZE;
        }
    }

    flagfft::FlagFFTPlanDesc desc;
    desc.rank = rank;
    desc.n = flagfft::copy_dims(n, rank);
    desc.inembed = inembed == nullptr ? desc.n : flagfft::copy_dims(inembed, rank);
    desc.istride = istride;
    desc.idist = idist;
    desc.onembed = onembed == nullptr ? desc.n : flagfft::copy_dims(onembed, rank);
    desc.ostride = ostride;
    desc.odist = odist;
    desc.batch = batch;
    desc.type = type;

    flagfftResult type_result = flagfft::type_metadata(
        type, desc.precision, desc.kind, desc.real_input, desc.real_output);
    if (type_result != FLAGFFT_SUCCESS) {
        return type_result;
    }

    return flagfft::build_plan(plan, std::move(desc));
}

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
