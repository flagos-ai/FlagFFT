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

// Runtime settings qualified with conf/hcu_bw1000.sh at b11a09fb.
// Full-batch capture is a validation option, not a library runtime default.
struct HcuPerformanceSetting {
  const char *name;
  const char *value;
};

inline constexpr HcuPerformanceSetting kHcuPerformanceDefaults[] = {
    {"FLAGFFT_HCU_2D_REAL_COL_STORE", "1"},
    {"FLAGFFT_HCU_2D_REAL_ROWS", "1"},
    {"FLAGFFT_HCU_2D_REAL_ROW_STORE", "1"},
    {"FLAGFFT_HCU_2D_TRANSPOSE", "coalesced"},
    {"FLAGFFT_HCU_3D_C2R_FUSED_LOAD", "1"},
    {"FLAGFFT_HCU_3D_C2R_MIDDLE_TRANSPOSE", "0"},
    {"FLAGFFT_HCU_3D_DIRECT_STORE", "1"},
    {"FLAGFFT_HCU_3D_STATIC_BATCH", "1"},
    {"FLAGFFT_HCU_3D_STATIC_PERMUTE", "1"},
    {"FLAGFFT_HCU_3D_STORE_MODE", "cyclic"},
    {"FLAGFFT_HCU_3D_TRANSPOSE_MODE", "resource"},
    {"FLAGFFT_HCU_3D_TRANSPOSE_TILE", "auto"},
    {"FLAGFFT_HCU_3D_WARP_FFT", "1"},
    {"FLAGFFT_HCU_3D_WARP_FP64", "0"},
    {"FLAGFFT_HCU_3D_WARP_PACK", "resource"},
    {"FLAGFFT_HCU_CONV_BALANCE", "0"},
    {"FLAGFFT_HCU_CONV_SINGLE_BUFFER", "1"},
    {"FLAGFFT_HCU_CYCLIC_SKEW", "unit"},
    {"FLAGFFT_HCU_DENSE_WAVE", "packed"},
    {"FLAGFFT_HCU_DENSE_WAVE_FP64", "1"},
    {"FLAGFFT_HCU_FOURSTEP_PACKING", "fp32_wide"},
    {"FLAGFFT_HCU_FOURSTEP_PRESERVE_SPLIT", "irreducible"},
    {"FLAGFFT_HCU_FOURSTEP_RADIX_CAP", "fp64"},
    {"FLAGFFT_HCU_FOURSTEP_SINGLE_BUFFER", "1"},
    {"FLAGFFT_HCU_FP32_ROW_TWIDDLE", "0"},
    {"FLAGFFT_HCU_FP64_CONV_RADIX_CAP", "16"},
    {"FLAGFFT_HCU_FUSED_PACKED_REAL", "0"},
    {"FLAGFFT_HCU_LARGE_PACKED_REAL", "1"},
    {"FLAGFFT_HCU_LEAF_PACKING", "wave"},
    {"FLAGFFT_HCU_LEAF_SINGLE_BUFFER", "1"},
    {"FLAGFFT_HCU_PACKED_CTA_PUBLIC", "resource"},
    {"FLAGFFT_HCU_PACKED_REAL_CTA", "1"},
    {"FLAGFFT_HCU_PACKED_REAL_STORE", "1"},
    {"FLAGFFT_HCU_PACKED_WAVE_LAYOUT", "resource"},
    {"FLAGFFT_HCU_PERMUTED_FILL_WAVE", "1"},
    {"FLAGFFT_HCU_PHYSICAL_WAVE_COST", "0"},
    {"FLAGFFT_HCU_PLAN_GRAPH", "real_2d"},
    {"FLAGFFT_HCU_PRIME_COST_COMPARISON", "1"},
    {"FLAGFFT_HCU_STAGE_TRAFFIC_COST", "0"},
    {"FLAGFFT_HCU_VECTOR_IO", "fp32"},
    {"FLAGFFT_HCU_WAVE_CODELET", "1"},
    {"FLAGFFT_HCU_WAVE_ROOT_PERIOD", "0"},
    {"FLAGFFT_HCU_WAVE_ROOT_TABLE", "1"},
    {"FLAGFFT_HCU_WAVE_SCALAR_BUDGET", "64"},
    {"FLAGFFT_HCU_WAVE_STRIDED_INPUT", "1"},
    {"FLAGFFT_HCU_WAVE_STRIDED_TILE", "output"},
    {"FLAGFFT_TUNE_DISABLE", "1"},
};

inline bool hcu_performance_qualified(std::string_view backend,
                                      std::string_view architecture,
                                      std::string_view dtype) {
  return backend == "hcu" && architecture == "gfx936" &&
         (dtype == "complex64" || dtype == "float32" ||
          dtype == "complex128" || dtype == "float64");
}

inline bool hcu_performance_defaults_requested() {
  const char *value = std::getenv("FLAGFFT_HCU_OPTIMIZED_DEFAULTS");
  if (value && std::string_view(value) != "0" && std::string_view(value) != "1")
    throw std::runtime_error("FLAGFFT_HCU_OPTIMIZED_DEFAULTS must be 0 or 1");
  return !value || std::string_view(value) == "1";
}

inline thread_local bool hcu_performance_defaults_active = false;

// A nested request for another backend restores the caller's settings on exit.
// No setting is written into the application's process environment.
class HcuPerformanceScope {
 public:
  explicit HcuPerformanceScope(bool qualified)
      : previous_(hcu_performance_defaults_active) {
    hcu_performance_defaults_active = qualified && hcu_performance_defaults_requested();
  }
  HcuPerformanceScope(std::string_view backend, std::string_view architecture,
                      std::string_view dtype)
      : HcuPerformanceScope(hcu_performance_qualified(backend, architecture, dtype)) {}
  HcuPerformanceScope(std::string_view target, std::string_view dtype)
      : HcuPerformanceScope((target == "hcu:gfx936" || target.rfind("hcu:gfx936:", 0) == 0) &&
                            hcu_performance_qualified("hcu", "gfx936", dtype)) {}
  HcuPerformanceScope(const HcuPerformanceScope &) = delete;
  HcuPerformanceScope &operator=(const HcuPerformanceScope &) = delete;
  ~HcuPerformanceScope() { hcu_performance_defaults_active = previous_; }

 private:
  bool previous_;
};

inline const char *hcu_performance_env(const char *name) {
  if (const char *override_value = std::getenv(name)) return override_value;
  if (hcu_performance_defaults_active) {
    for (const auto &setting : kHcuPerformanceDefaults)
      if (std::string_view(name) == setting.name) return setting.value;
  }
  return nullptr;
}

// The same effective values identify explicit profiles and automatic defaults.
// Rollback and per-setting overrides cannot reuse incompatible cached plans/JIT.
inline std::string hcu_performance_identity() {
  std::string result = "hcu-performance-v1";
  for (const auto &setting : kHcuPerformanceDefaults) {
    result += ";";
    result += setting.name;
    result += "=";
    const char *value = hcu_performance_env(setting.name);
    result += value ? value : "<unset>";
  }
  return result;
}

}  // namespace flagfft
