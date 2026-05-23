#include "runtime/device.hpp"

#include <cuda.h>

#include "runtime/types.hpp"
#include "triton_jit/backend_config.h"

namespace flagfft::runtime {
namespace {

inline constexpr int64_t kDynamicSmemFallbackBytes = 48 * 1024;

}  // namespace

int64_t max_dynamic_smem_bytes(int device_index) {
    if (device_index < 0 || cuInit(0) != CUDA_SUCCESS) {
        return kDynamicSmemFallbackBytes;
    }

    CUdevice device;
    if (cuDeviceGet(&device, static_cast<int>(device_index)) != CUDA_SUCCESS) {
        return kDynamicSmemFallbackBytes;
    }

    int value = 0;
    CUresult result = cuDeviceGetAttribute(
        &value, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device);
    if (result == CUDA_SUCCESS && value > 0) {
        return static_cast<int64_t>(value);
    }

    value = 0;
    result = cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, device);
    if (result == CUDA_SUCCESS && value > 0) {
        return static_cast<int64_t>(value);
    }
    return kDynamicSmemFallbackBytes;
}

flagfftResult ensure_device(int &device_index, std::string &device_arch) {
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

}  // namespace flagfft::runtime
