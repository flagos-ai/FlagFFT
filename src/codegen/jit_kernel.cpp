#include "flagfft/core.hpp"

#include "triton_jit/triton_jit_function.h"

namespace flagfft {

JitKernelArg JitKernelArg::device(adaptor::DevicePtr value) {
  JitKernelArg arg;
  arg.kind = JitArgKind::DevicePtr;
  arg.device_ptr = value;
  return arg;
}

JitKernelArg JitKernelArg::i32(int32_t value) {
  JitKernelArg arg;
  arg.kind = JitArgKind::Int32;
  arg.int32_value = value;
  return arg;
}

JitKernelArg JitKernelArg::i64(int64_t value) {
  JitKernelArg arg;
  arg.kind = JitArgKind::Int64;
  arg.int64_value = value;
  return arg;
}

JitKernel::~JitKernel() = default;

void JitKernel::compile() {
  std::lock_guard<std::mutex> lock(mutex);
  if (jit_function != nullptr) {
    return;
  }
  jit_function = &triton_jit::TritonJITFunction::get_instance(module_path, kernel_name);
  auto *function = static_cast<triton_jit::TritonJITFunction *>(jit_function);
  function->compile(signature,
                    static_cast<unsigned int>(num_warps),
                    static_cast<unsigned int>(num_stages),
                    triton_jit::DefaultBackend::get_device_index());
}

void JitKernel::launch(adaptor::StreamHandle stream,
                       const std::vector<JitKernelArg> &kernel_args,
                       int64_t grid_x,
                       int64_t grid_y,
                       int64_t grid_z) {
  compile();
  adaptor::DevicePtr global_scratch = 0;
  adaptor::DevicePtr profile_scratch = 0;
  std::vector<void *> args;
  args.reserve(kernel_args.size() + 2);
  for (const JitKernelArg &arg : kernel_args) {
    switch (arg.kind) {
      case JitArgKind::DevicePtr:
        args.push_back(const_cast<adaptor::DevicePtr *>(&arg.device_ptr));
        break;
      case JitArgKind::Int32:
        args.push_back(const_cast<int32_t *>(&arg.int32_value));
        break;
      case JitArgKind::Int64:
        args.push_back(const_cast<int64_t *>(&arg.int64_value));
        break;
    }
  }
  args.push_back(&global_scratch);
  args.push_back(&profile_scratch);
  auto *function = static_cast<triton_jit::TritonJITFunction *>(jit_function);
  function->launch_with_raw_args(reinterpret_cast<triton_jit::DefaultStreamType>(stream),
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
