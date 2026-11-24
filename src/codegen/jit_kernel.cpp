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

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include "flagfft/core.hpp"

#include "triton_jit/triton_jit_function.h"

namespace flagfft {
#if defined(BACKEND_NPU)
namespace {
triton_jit::CompileOptions npu_compile_options(const JitKernel &kernel) {
  triton_jit::CompileOptions options;
  options.num_warps = static_cast<int>(kernel.num_warps);
  options.num_stages = static_cast<int>(kernel.num_stages);
  options.extra["structured_join_lowering"] = kernel.npu_join_lowering;
  options.extra["enable_auto_blockify"] = kernel.npu_auto_blockify ? "true" : "false";
  if (kernel.npu_single_buffer)
    options.extra["multibuffer"] = "false";
  return options;
}
}
#endif

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

JitKernel::~JitKernel() {
#if defined(BACKEND_MACA)
  delete static_cast<triton_jit::TritonKernelImpl<triton_jit::DefaultBackend> *>(jit_function);
#endif
}

std::string JitKernel::execution_description() const {
  std::ostringstream out;
  out << kernel_name << "{warp_size=" << warp_size << ",num_warps=" << num_warps
      << ",block_threads=" << warp_size * num_warps << ",batch_per_block=" << batch_per_block
      << ",inner_pack=" << inner_pack << ",profile=" << profile_id << "}";
  return out.str();
}

namespace {
// Pre-compile at plan creation when libtriton_jit offers the hook; releases
// without it compile the kernel on the first launch instead.
template <typename Function>
concept HasPrecompile = requires(Function &function, const std::string &signature) {
  function.compile(signature, 0u, 0u, 0);
};

template <typename Function>
void precompile(Function &function,
                const std::string &signature,
                unsigned int num_warps,
                unsigned int num_stages,
                int device_index) {
  if constexpr (HasPrecompile<Function>) {
    function.compile(signature, num_warps, num_stages, device_index);
  }
}
}  // namespace

void JitKernel::compile() {
  if (compiled.load(std::memory_order_acquire)) return;
  std::lock_guard<std::mutex> lock(mutex);
  if (compiled.load(std::memory_order_relaxed)) return;
  if (warp_size != triton_jit::DefaultBackend::WARP_SIZE)
    throw std::runtime_error("codegen warp size disagrees with launch backend");
#if defined(BACKEND_MACA)
  if (binary_dir.empty()) {
    throw std::runtime_error("MACA JIT binary directory is missing");
  }
  // Compilation happens in the codegen process. Embedding this SDK's Torch
  // in a native executable corrupts its FlashAttn teardown. Native loading
  // and dispatch continue to use libtriton_jit's MACA backend and hooks.
  triton_jit::DefaultBackend::ensure_context();
  triton_jit::DefaultBackend::load_kernel(binary_dir, kernel_name);
  jit_function = new triton_jit::TritonKernelImpl<triton_jit::DefaultBackend>(binary_dir, kernel_name);
#else
  jit_function = &triton_jit::TritonJITFunction::get_instance(module_path, kernel_name);
  auto *function = static_cast<triton_jit::TritonJITFunction *>(jit_function);
#if defined(BACKEND_NPU)
  function->compile(signature,npu_compile_options(*this),
                    triton_jit::DefaultBackend::get_device_index());
#else
  precompile(*function,
             signature,
             static_cast<unsigned int>(num_warps),
             static_cast<unsigned int>(num_stages),
             triton_jit::DefaultBackend::get_device_index());
#if defined(BACKEND_HCU)
  triton_jit::CompileOptions options;
  options.num_warps = static_cast<int>(num_warps);
  options.num_stages = static_cast<int>(num_stages);
  direct_launch = function->prepare_direct_launch(signature, options);
#endif
#endif
#endif
  compiled.store(true, std::memory_order_release);
}

void JitKernel::launch(adaptor::StreamHandle stream,
                       const std::vector<JitKernelArg> &kernel_args,
                       int64_t grid_x,
                       int64_t grid_y,
                       int64_t grid_z) {
  compile();
#if !defined(BACKEND_NPU)
  adaptor::DevicePtr global_scratch = 0;
  adaptor::DevicePtr profile_scratch = 0;
#endif
  // Most FFT launch signatures fit on the stack. Retain an unbounded
  // fallback for larger generated codelet tables.
  constexpr size_t inline_capacity = 32;
#if defined(BACKEND_NPU)
  const size_t num_args = kernel_args.size();
#else
  const size_t num_args = kernel_args.size() + 2;
#endif
  std::array<void *, inline_capacity> inline_args;
  std::vector<void *> overflow_args;
  if (num_args > inline_capacity) overflow_args.resize(num_args);
  void **args = num_args > inline_capacity ? overflow_args.data() : inline_args.data();
  size_t index = 0;
  for (const JitKernelArg &arg : kernel_args) {
    switch (arg.kind) {
      case JitArgKind::DevicePtr:
        args[index++] = const_cast<adaptor::DevicePtr *>(&arg.device_ptr);
        break;
      case JitArgKind::Int32:
        args[index++] = const_cast<int32_t *>(&arg.int32_value);
        break;
      case JitArgKind::Int64:
        args[index++] = const_cast<int64_t *>(&arg.int64_value);
        break;
    }
  }
#if !defined(BACKEND_NPU)
  args[index++] = &global_scratch;
  args[index++] = &profile_scratch;
#endif
#if !defined(BACKEND_MACA)
  auto *function = static_cast<triton_jit::TritonJITFunction *>(jit_function);
#endif
#if defined(BACKEND_NPU)
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (direct_launch == nullptr) {
      direct_launch = function->prepare_direct_launch(signature,npu_compile_options(*this));
    }
  }
#endif
  // Diagnostic timings synchronise each launch and must not be used as
  // end-to-end benchmark results. Disabled unless explicitly requested.
  static const bool profile = env_flag_enabled(std::getenv("FLAGFFT_PROFILE_KERNELS"));
  std::optional<adaptor::EventTimer> timer;
  if (profile) {
    timer.emplace();
    timer->start(stream);
  }
#if defined(BACKEND_NPU)
  auto *direct = static_cast<triton_jit::DefaultBackend::DirectLaunch *>(direct_launch.get());
  direct->launch(reinterpret_cast<triton_jit::DefaultStreamType>(stream),
                 static_cast<unsigned int>(grid_x),
                 static_cast<unsigned int>(grid_y),
                 static_cast<unsigned int>(grid_z),
                 args);
#elif defined(BACKEND_MACA)
  auto *kernel = static_cast<triton_jit::TritonKernelImpl<triton_jit::DefaultBackend> *>(jit_function);
  kernel->launch_with_signature(static_cast<unsigned int>(grid_x),
                                static_cast<unsigned int>(grid_y),
                                static_cast<unsigned int>(grid_z),
                                num_warps,
                                reinterpret_cast<triton_jit::DefaultStreamType>(stream),
                                args,
                                signature,
                                num_args);
#elif defined(BACKEND_HCU)
  const char *prepared_setting = std::getenv("FLAGFFT_HCU_PREPARED_LAUNCH");
  const bool prepared_enabled = prepared_setting == nullptr || std::strcmp(prepared_setting, "1") == 0;
  // Explicit streams and the compiled module identify the launch device.
  // Preserve the generic launcher when hooks are installed or the caller
  // uses a default stream, whose device can change independently.
  if (prepared_enabled && direct_launch != nullptr && stream != nullptr &&
      !triton_jit::detail::get_launch_hooks_snapshot()) {
    const auto &prepared = *static_cast<triton_jit::DefaultBackend::DirectLaunch *>(direct_launch.get());
    prepared.launch(reinterpret_cast<triton_jit::DefaultStreamType>(stream),
        static_cast<unsigned int>(grid_x), static_cast<unsigned int>(grid_y),
        static_cast<unsigned int>(grid_z), static_cast<unsigned int>(num_warps), args);
  } else {
    function->launch_with_raw_args(reinterpret_cast<triton_jit::DefaultStreamType>(stream),
        static_cast<unsigned int>(grid_x), static_cast<unsigned int>(grid_y),
        static_cast<unsigned int>(grid_z), static_cast<unsigned int>(num_warps),
        static_cast<unsigned int>(num_stages), signature, args, num_args);
  }
#else
  function->launch_with_raw_args(reinterpret_cast<triton_jit::DefaultStreamType>(stream),
                                 static_cast<unsigned int>(grid_x),
                                 static_cast<unsigned int>(grid_y),
                                 static_cast<unsigned int>(grid_z),
                                 static_cast<unsigned int>(num_warps),
                                 static_cast<unsigned int>(num_stages),
                                 signature,
                                 args,
                                 num_args);
#endif
  if (timer) {
    timer->stop(stream);
    std::fprintf(stderr,
                 "[kernel-profile],%s,%lld,%lld,%lld,%.6f\n",
                 kernel_name.c_str(),
                 static_cast<long long>(grid_x),
                 static_cast<long long>(grid_y),
                 static_cast<long long>(grid_z),
                 timer->elapsed_ms());
  }
}

}  // namespace flagfft
