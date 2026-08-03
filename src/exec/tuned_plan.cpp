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

namespace flagfft {

inline constexpr int64_t kFourStepColInnerPack = 4;
inline constexpr int64_t kFourStepLargeInnerPack = 4;
inline constexpr int64_t kFourStepColInnerPackMinN1 = 128;
inline constexpr int64_t kFourStepRowInnerPackMaxN1 = 512;
inline constexpr int64_t kFourStepPackedColLeafMaxN2 = 1024;
inline constexpr int64_t kTleFusedTwiddleMinLength = int64_t {1} << 18;
inline constexpr int64_t kTleFusedTwiddleMaxLeaf = 1024;

bool use_tle_fused_twiddle(int64_t n1, int64_t n2, const std::string &dtype) {
  const bool is_double = dtype == "complex128" || dtype == "float64";
  return !is_double && n1 * n2 >= kTleFusedTwiddleMinLength && n1 <= kTleFusedTwiddleMaxLeaf &&
         n2 <= kTleFusedTwiddleMaxLeaf;
}

int64_t four_step_col_inner_pack_for(int64_t n1, int64_t n2, const std::string &dtype) {
  if (n1 < kFourStepColInnerPackMinN1) {
    return 1;
  }
  if (use_tle_fused_twiddle(n1, n2, dtype)) {
    return kFourStepLargeInnerPack;
  }
  if (n2 > kFourStepPackedColLeafMaxN2) {
    // Large mixed-radix col leaves already consume enough registers and smem
    // to limit residency. Keep one FFT per CTA to avoid doubling that pressure.
    return 1;
  }
  return kFourStepColInnerPack;
}

int64_t four_step_row_inner_pack_for(int64_t n1, int64_t n2, const std::string &dtype) {
  if (use_tle_fused_twiddle(n1, n2, dtype)) {
    return kFourStepLargeInnerPack;
  }
  const bool is_double = dtype == "complex128" || dtype == "float64";
  if (!is_double && n1 <= kFourStepRowInnerPackMaxN1 && n2 > kFourStepPackedColLeafMaxN2) {
    // Small low-lane row leaves otherwise leave most of a warp idle. Pack four
    // independent rows while keeping their total smem below the A100 budget.
    return kFourStepLargeInnerPack;
  }
  return 1;
}

std::string batch_bucket(int64_t batch) {
  if (batch <= 1) {
    return "1";
  }
  if (batch <= 8) {
    return "2-8";
  }
  if (batch <= 64) {
    return "9-64";
  }
  if (batch <= 512) {
    return "65-512";
  }
  return "513+";
}

bool env_flag_enabled(const char *value) {
  if (value == nullptr) {
    return false;
  }
  std::string normalized(value);
  std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return !(normalized.empty() || normalized == "0" || normalized == "false" || normalized == "off" ||
           normalized == "no");
}

std::optional<std::filesystem::path> tuned_db_path() {
  if (env_flag_enabled(std::getenv("FLAGFFT_TUNE_DISABLE"))) {
    return std::nullopt;
  }
  const char *override_path = std::getenv("FLAGFFT_TUNE_DB");
  if (override_path != nullptr && std::string(override_path).size() > 0) {
    return std::filesystem::path(override_path);
  }
  return default_cache_dir() / "tuned_plans.sqlite";
}

}  // namespace flagfft
