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

#include "flagfft/core.hpp"
#include <cstdio>
#include <cstdlib>
#include <optional>

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

std::string JitKernel::execution_description() const {
  std::ostringstream out;
  out << kernel_name << "{warp_size=" << warp_size << ",num_warps=" << num_warps
      << ",block_threads=" << warp_size * num_warps << ",batch_per_block=" << batch_per_block
      << ",inner_pack=" << inner_pack << ",profile=" << profile_id << "}";
  return out.str();
}

void JitKernel::compile() {
  std::lock_guard<std::mutex> lock(mutex);
  if (jit_function != nullptr) {
    return;
  }
  if (warp_size != triton_jit::DefaultBackend::WARP_SIZE)
    throw std::runtime_error("codegen warp size disagrees with launch backend");
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
  // Diagnostic timings synchronise each launch and must not be used as
  // end-to-end benchmark results. Disabled unless explicitly requested.
  static const bool profile = env_flag_enabled(std::getenv("FLAGFFT_PROFILE_KERNELS"));
  std::optional<adaptor::EventTimer> timer;
  if (profile) {
    timer.emplace();
    timer->start(stream);
  }
  function->launch_with_raw_args(reinterpret_cast<triton_jit::DefaultStreamType>(stream),
                                 static_cast<unsigned int>(grid_x),
                                 static_cast<unsigned int>(grid_y),
                                 static_cast<unsigned int>(grid_z),
                                 static_cast<unsigned int>(num_warps),
                                 static_cast<unsigned int>(num_stages),
                                 signature,
                                 args.data(),
                                 args.size());
  if (timer) {
    timer->stop(stream);
    std::fprintf(stderr, "[kernel-profile],%s,%lld,%lld,%lld,%.6f\n", kernel_name.c_str(),
                 static_cast<long long>(grid_x), static_cast<long long>(grid_y),
                 static_cast<long long>(grid_z), timer->elapsed_ms());
  }
}

}  // namespace flagfft
