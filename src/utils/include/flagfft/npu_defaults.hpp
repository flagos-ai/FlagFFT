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

#pragma once

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

namespace flagfft {

// Qualified FP32 scheduling defaults for Ascend 910B. Coefficients, kernel
// arithmetic and dot input precision retain their existing IEEE FP32 paths.
struct NpuPerformanceSetting {
  const char *name;
  const char *value;
};

inline constexpr NpuPerformanceSetting kNpuPerformanceDefaults[] = {
    {"FLAGFFT_NPU_2D_BATCH_PACK", "auto"},
    {"FLAGFFT_NPU_2D_EXCHANGE", "direct_all"},
    {"FLAGFFT_NPU_2D_FUSED_FFT_PACK", "16"},
    {"FLAGFFT_NPU_2D_FUSED_FFT_STORES", "1"},
    {"FLAGFFT_NPU_2D_LANE_MIN", "1"},
    {"FLAGFFT_NPU_AFFINE_MATRIX_IO", "1"},
    {"FLAGFFT_NPU_COMPACT_STAGES", "1"},
    {"FLAGFFT_NPU_DIF_BATCH_DENSE_PREFIX", "1"},
    {"FLAGFFT_NPU_DIF_BATCH_FINISH", "1"},
    {"FLAGFFT_NPU_DIF_COARSE_GT", "1"},
    {"FLAGFFT_NPU_DIF_CODELET_REUSE", "1"},
    {"FLAGFFT_NPU_DIF_CUBE_PREFIX", "1"},
    {"FLAGFFT_NPU_DIF_DENSE_SHORT_PREFIX", "0"},
    {"FLAGFFT_NPU_DIF_FUSED_FINISH", "1"},
    {"FLAGFFT_NPU_DIF_LATE_AUTO_BLOCK", "1"},
    {"FLAGFFT_NPU_DIF_PAD_PREFIX", "1"},
    {"FLAGFFT_NPU_DIF_PAD_SMALL_BATCH", "1"},
    {"FLAGFFT_NPU_DIF_PREFIX_PIPELINE", "1"},
    {"FLAGFFT_NPU_DIF_SINGLE_MATRIX_CHAIN", "1"},
    {"FLAGFFT_NPU_DIF_TRANSPOSE_TILE", "64"},
    {"FLAGFFT_NPU_DIF_VECTOR_BASIS", "1"},
    {"FLAGFFT_NPU_DIRECT_DFT_PAIR", "1"},
    {"FLAGFFT_NPU_DIRECT_DFT_PREFETCH", "1"},
    {"FLAGFFT_NPU_DIRECT_DFT_SINGLE_REDUCE", "1"},
    {"FLAGFFT_NPU_EXECUTION_BACKEND", "triton"},
    {"FLAGFFT_NPU_FLAT_POWER_DIF", "1"},
    {"FLAGFFT_NPU_FLAT_POWER_FIRST_PHASE", "1"},
    {"FLAGFFT_NPU_FLAT_POWER_FIRST_PIPELINE", "1"},
    {"FLAGFFT_NPU_FLAT_POWER_WORKSPACE_PAD", "1"},
    {"FLAGFFT_NPU_FOURSTEP_AUTO_BLOCK", "1"},
    {"FLAGFFT_NPU_FOURSTEP_DENSE_SECOND", "1"},
    {"FLAGFFT_NPU_FOURSTEP_PACK", "auto"},
    {"FLAGFFT_NPU_FREQUENCY_DFT", "1"},
    {"FLAGFFT_NPU_GRAPH", "auto"},
    {"FLAGFFT_NPU_MATRIX_AUTO_BLOCK", "1"},
    {"FLAGFFT_NPU_MATRIX_DIRECT_COMBINED", "1"},
    {"FLAGFFT_NPU_MATRIX_DIRECT_INPUT", "1"},
    {"FLAGFFT_NPU_MATRIX_FFT", "1"},
    {"FLAGFFT_NPU_MATRIX_REAL_FOURSTEP", "1"},
    {"FLAGFFT_NPU_MATRIX_REAL_POWER2", "1"},
    {"FLAGFFT_NPU_MATRIX_SINGLE_DIRECT", "1"},
    {"FLAGFFT_NPU_MATRIX_SINGLE_HALF_LAST", "0"},
    {"FLAGFFT_NPU_MATRIX_WIDE", "1"},
    {"FLAGFFT_NPU_MATRIX_WIDE_SINGLE", "0"},
    {"FLAGFFT_NPU_MIXED_FOURSTEP", "1"},
    {"FLAGFFT_NPU_PACKED_MIXED_AXIS", "1"},
    {"FLAGFFT_NPU_PACKED_MIXED_REAL", "1"},
    {"FLAGFFT_NPU_PACKED_REAL_TWO_AXIS", "1"},
    {"FLAGFFT_NPU_PAIRED_REAL_BOUNDARY", "1"},
    {"FLAGFFT_NPU_PORTABLE_LEAF_MAX", "2048"},
    {"FLAGFFT_NPU_PORTABLE_VECTOR_IO", "1"},
    {"FLAGFFT_NPU_POWER2_AUTO_BLOCK", "1"},
    {"FLAGFFT_NPU_REAL_BOUNDARY_TILE", "2048"},
    {"FLAGFFT_NPU_RECURSIVE_AXIS", "1"},
    {"FLAGFFT_NPU_RECURSIVE_POWER2", "1"},
    {"FLAGFFT_NPU_SHARED_POWER2", "1"},
    {"FLAGFFT_NPU_STRUCTURED_FOURSTEP", "1"},
    {"FLAGFFT_NPU_STRUCTURED_PACK", "32"},
    {"FLAGFFT_NPU_STRUCTURED_PACKED_FOURSTEP", "1"},
    {"FLAGFFT_NPU_STRUCTURED_REAL", "1"},
    {"FLAGFFT_NPU_TENSOR_FFT", "1"},
    {"FLAGFFT_NPU_TENSOR_PLANE", "1"},
    {"FLAGFFT_NPU_TRIG_TWIDDLES", "0"},
    {"FLAGFFT_NPU_VECTOR_DIF", "1"},
    {"FLAGFFT_NPU_VECTOR_REMAINDER", "1"},
    {"FLAGFFT_NPU_WIDE_SHORT_FFT", "1"},
};

inline bool npu_performance_qualified(std::string_view backend,
                                      std::string_view architecture,
                                      std::string_view dtype) {
  return backend == "npu" && architecture.rfind("Ascend910B", 0) == 0 &&
         (dtype == "complex64" || dtype == "float32");
}

inline bool npu_performance_defaults_requested() {
  const char *value = std::getenv("FLAGFFT_NPU_OPTIMIZED_DEFAULTS");
  if (value && std::string_view(value) != "0" && std::string_view(value) != "1") {
    throw std::runtime_error("FLAGFFT_NPU_OPTIMIZED_DEFAULTS must be 0 or 1");
  }
  return !value || std::string_view(value) == "1";
}

// Planning and compilation may nest or run concurrently for different targets.
// Resolve defaults locally; never mutate the application's environment.
inline thread_local bool npu_performance_defaults_active = false;

class NpuPerformanceScope {
 public:
  explicit NpuPerformanceScope(bool qualified)
      : previous_(npu_performance_defaults_active) {
    npu_performance_defaults_active = qualified && npu_performance_defaults_requested();
  }
  NpuPerformanceScope(std::string_view backend, std::string_view architecture,
                      std::string_view dtype)
      : NpuPerformanceScope(npu_performance_qualified(backend, architecture, dtype)) {}
  NpuPerformanceScope(std::string_view target, std::string_view dtype)
      : NpuPerformanceScope(target.rfind("npu:Ascend910B", 0) == 0 &&
                            (dtype == "complex64" || dtype == "float32")) {}
  NpuPerformanceScope(const NpuPerformanceScope &) = delete;
  NpuPerformanceScope &operator=(const NpuPerformanceScope &) = delete;
  ~NpuPerformanceScope() { npu_performance_defaults_active = previous_; }

 private:
  bool previous_;
};

inline const char *npu_performance_env(const char *name) {
  if (const char *override_value = std::getenv(name)) return override_value;
  if (npu_performance_defaults_active &&
      std::string_view(name).rfind("FLAGFFT_NPU_", 0) == 0) {
    for (const auto &setting : kNpuPerformanceDefaults) {
      if (std::string_view(name) == setting.name) return setting.value;
    }
  }
  return nullptr;
}

// Include resolved settings in both planner and JIT identities. Explicit
// rollback and custom tuning cannot reuse a plan or artifact with other options.
inline std::string npu_performance_identity() {
  std::string result = "npu-performance-v1";
  for (const auto &setting : kNpuPerformanceDefaults) {
    result += ";";
    result += setting.name;
    result += "=";
    const char *value = npu_performance_env(setting.name);
    result += value ? value : "<unset>";
  }
  return result;
}

}  // namespace flagfft
