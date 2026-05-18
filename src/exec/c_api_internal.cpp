#include "c_api_internal.hpp"

namespace flagfft {

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

FlagFFTPlan *checked_plan(flagfftHandle handle) {
    if (handle == nullptr || handle->impl == nullptr) {
        return nullptr;
    }
    return static_cast<FlagFFTPlan *>(handle->impl);
}

}  // namespace flagfft
