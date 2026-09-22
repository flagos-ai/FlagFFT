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

// Request-level backend policy predicates.  These depend only on the request
// and the environment, so they live outside the code generator and remain
// linkable by CPU-only planner tests.

#include "flagfft/codegen.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace flagfft {

bool ix_ct_single_policy_enabled(const FFTRequest &request) {
  if (request.device_type != "ix" || request.device_arch != "71" || request.raw_dim != 1 ||
      request.batch != 1 || request.fft_length != request.requested_n ||
      (request.requested_n != 1024 && request.requested_n != 2048 && request.requested_n != 16384) ||
      request.input_dtype != "complex64" || request.output_dtype != "complex64" ||
      request.input_strides.empty() || request.input_strides.back() != 1) {
    return false;
  }
  const char *setting = std::getenv("FLAGFFT_IX_CT_SINGLE");
  if (!setting || std::string(setting) == "1") return true;
  if (std::string(setting) == "0") return false;
  throw std::runtime_error("FLAGFFT_IX_CT_SINGLE must be 0 or 1");
}

bool ix_packed_real_policy_enabled(const FFTRequest &request) {
  if (request.device_type != "ix" || request.device_arch != "71" || request.raw_dim != 1 ||
      request.batch != 1 || request.fft_length != request.requested_n ||
      request.input_dtype != "complex64" || request.output_dtype != "complex64" ||
      request.input_strides.empty() || request.input_strides.back() != 1) return false;
  switch (request.requested_n) {
    case 328050:
    case 340200:
    case 663000:
    case 1048576:
      break;
    default:
      return false;
  }
  const char *setting = std::getenv("FLAGFFT_IX_CT_SINGLE");
  if (!setting || std::string(setting) == "1") return true;
  if (std::string(setting) == "0") return false;
  throw std::runtime_error("FLAGFFT_IX_CT_SINGLE must be 0 or 1");
}

int ix_ct_single_tle_policy(const FFTRequest &request) {
  if (request.device_type != "ix" || request.device_arch != "71" || request.raw_dim != 1 ||
      request.batch != 1 || request.fft_length != request.requested_n ||
      request.input_dtype != "complex64" || request.output_dtype != "complex64" ||
      request.input_strides.empty() || request.input_strides.back() != 1) return 0;
  int policy = 0;
  if (request.requested_n == 328050 || request.requested_n == 340200) policy = 1;
  if (request.requested_n == 1048576) policy = 2;
  if (!policy) return 0;
  const char *setting = std::getenv("FLAGFFT_IX_CT_SINGLE");
  if (!setting || std::string(setting) == "1") return policy;
  if (std::string(setting) == "0") return 0;
  throw std::runtime_error("FLAGFFT_IX_CT_SINGLE must be 0 or 1");
}

}  // namespace flagfft
