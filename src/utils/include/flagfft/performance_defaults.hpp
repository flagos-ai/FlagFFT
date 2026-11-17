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

#include "flagfft/hcu_defaults.hpp"
#include "flagfft/npu_defaults.hpp"

namespace flagfft {

class BackendPerformanceScope {
 public:
  BackendPerformanceScope(std::string_view backend, std::string_view architecture,
                          std::string_view dtype)
      : npu_(backend, architecture, dtype), hcu_(backend, architecture, dtype) {}
  BackendPerformanceScope(std::string_view target, std::string_view dtype)
      : npu_(target, dtype), hcu_(target, dtype) {}

 private:
  NpuPerformanceScope npu_;
  HcuPerformanceScope hcu_;
};

inline const char *performance_env(const char *name) {
  if (const char *value = hcu_performance_env(name)) return value;
  return npu_performance_env(name);
}

}  // namespace flagfft
