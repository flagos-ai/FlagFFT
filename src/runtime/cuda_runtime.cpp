#include "flagfft/core.hpp"

#include "triton_jit/triton_jit_function.h"

namespace flagfft {

DeviceAllocation::DeviceAllocation(CUdeviceptr ptr, std::size_t bytes) : ptr(ptr), bytes(bytes) {}

DeviceAllocation::~DeviceAllocation() {
    reset();
}

DeviceAllocation::DeviceAllocation(DeviceAllocation &&other) noexcept
    : ptr(other.ptr), bytes(other.bytes) {
    other.ptr = 0;
    other.bytes = 0;
}

DeviceAllocation &DeviceAllocation::operator=(DeviceAllocation &&other) noexcept {
    if (this != &other) {
        reset();
        ptr = other.ptr;
        bytes = other.bytes;
        other.ptr = 0;
        other.bytes = 0;
    }
    return *this;
}

void DeviceAllocation::reset() {
    if (ptr != 0) {
        cuMemFree(ptr);
        ptr = 0;
        bytes = 0;
    }
}

void cuda_check(CUresult result, const std::string &context) {
    if (result == CUDA_SUCCESS) {
        return;
    }
    const char *name = nullptr;
    const char *message = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &message);
    std::ostringstream out;
    out << context << " failed";
    if (name != nullptr) {
        out << " (" << name << ")";
    }
    if (message != nullptr) {
        out << ": " << message;
    }
    throw std::runtime_error(out.str());
}

int64_t ceil_div(int64_t numerator, int64_t denominator) {
    return (numerator + denominator - 1) / denominator;
}

int64_t cuda_device_max_dynamic_shared_memory_bytes(int64_t device_index) {
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

RuntimeKernelArg RuntimeKernelArg::device(CUdeviceptr value) {
    RuntimeKernelArg arg;
    arg.kind = RuntimeArgKind::DevicePtr;
    arg.device_ptr = value;
    return arg;
}

RuntimeKernelArg RuntimeKernelArg::i32(int32_t value) {
    RuntimeKernelArg arg;
    arg.kind = RuntimeArgKind::Int32;
    arg.int32_value = value;
    return arg;
}

RuntimeKernelArg RuntimeKernelArg::i64(int64_t value) {
    RuntimeKernelArg arg;
    arg.kind = RuntimeArgKind::Int64;
    arg.int64_value = value;
    return arg;
}

RuntimeKernel::~RuntimeKernel() = default;

void RuntimeKernel::compile() {
    std::lock_guard<std::mutex> lock(mutex);
    if (jit_function != nullptr) {
        return;
    }
    cuda_check(cuInit(0), "cuInit");
    CUdevice device = 0;
    cuda_check(cuCtxGetDevice(&device), "cuCtxGetDevice");
    jit_function =
        &triton_jit::TritonJITFunction::get_instance(module_path, kernel_name);
    auto *function = static_cast<triton_jit::TritonJITFunction *>(jit_function);
    function->compile(signature,
                      static_cast<unsigned int>(num_warps),
                      static_cast<unsigned int>(num_stages),
                      static_cast<int>(device));
}

void RuntimeKernel::launch(CUstream stream,
                       const std::vector<RuntimeKernelArg> &kernel_args,
                       int64_t grid_x,
                       int64_t grid_y,
                       int64_t grid_z) {
    compile();
    CUdeviceptr global_scratch = 0;
    CUdeviceptr profile_scratch = 0;
    std::vector<void *> args;
    args.reserve(kernel_args.size() + 2);
    for (const RuntimeKernelArg &arg : kernel_args) {
        switch (arg.kind) {
            case RuntimeArgKind::DevicePtr:
                args.push_back(const_cast<CUdeviceptr *>(&arg.device_ptr));
                break;
            case RuntimeArgKind::Int32:
                args.push_back(const_cast<int32_t *>(&arg.int32_value));
                break;
            case RuntimeArgKind::Int64:
                args.push_back(const_cast<int64_t *>(&arg.int64_value));
                break;
        }
    }
    args.push_back(&global_scratch);
    args.push_back(&profile_scratch);
    auto *function = static_cast<triton_jit::TritonJITFunction *>(jit_function);
    function->launch_with_raw_args(stream,
                                   static_cast<unsigned int>(grid_x),
                                   static_cast<unsigned int>(grid_y),
                                   static_cast<unsigned int>(grid_z),
                                   static_cast<unsigned int>(num_warps),
                                   static_cast<unsigned int>(num_stages),
                                   signature,
                                   args.data(),
                                   args.size());
}

}  // namespace flagfft
