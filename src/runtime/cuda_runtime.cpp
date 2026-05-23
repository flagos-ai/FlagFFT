#include "flagfft/core.hpp"

#include "triton_jit/triton_jit_function.h"

#include "runtime/check.hpp"
#include "runtime/types.hpp"

namespace flagfft {

int64_t ceil_div(int64_t numerator, int64_t denominator) {
    return (numerator + denominator - 1) / denominator;
}

RuntimeKernelArg RuntimeKernelArg::device(runtime::DevicePtr value) {
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
    runtime::check(cuInit(0), "cuInit");
    CUdevice device = 0;
    runtime::check(cuCtxGetDevice(&device), "cuCtxGetDevice");
    jit_function =
        &triton_jit::TritonJITFunction::get_instance(module_path, kernel_name);
    auto *function = static_cast<triton_jit::TritonJITFunction *>(jit_function);
    function->compile(signature,
                      static_cast<unsigned int>(num_warps),
                      static_cast<unsigned int>(num_stages),
                      static_cast<int>(device));
}

void RuntimeKernel::launch(runtime::StreamHandle stream,
                       const std::vector<RuntimeKernelArg> &kernel_args,
                       int64_t grid_x,
                       int64_t grid_y,
                       int64_t grid_z) {
    compile();
    runtime::DevicePtr global_scratch = 0;
    runtime::DevicePtr profile_scratch = 0;
    std::vector<void *> args;
    args.reserve(kernel_args.size() + 2);
    for (const RuntimeKernelArg &arg : kernel_args) {
        switch (arg.kind) {
            case RuntimeArgKind::DevicePtr:
                args.push_back(const_cast<runtime::DevicePtr *>(&arg.device_ptr));
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
