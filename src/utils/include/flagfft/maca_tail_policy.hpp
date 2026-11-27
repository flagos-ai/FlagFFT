// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "flagfft/base.hpp"

#include <string_view>

namespace flagfft {

// On by default for a genuine rank-1 batch-1 MACA request.  `origin_rank`
// carries the public descriptor's rank, so the row/column sub-requests a 2D or
// 3D transform builds internally cannot inherit this: they keep the original
// rank and are rejected here.  Set FLAGFFT_MACA_TAIL_POLICY=0 to opt out.
inline bool maca_tail_policy_enabled(const FFTRequest& request) {
  const char* value = std::getenv("FLAGFFT_MACA_TAIL_POLICY");
  if (request.device_type != "maca" || request.raw_dim != 1 || request.batch != 1 ||
      (request.origin_rank != 0 && request.origin_rank != 1)) return false;
  if (value && *value) {
    if (std::string(value) == "0") return false;
    if (std::string(value) != "1")
      throw std::runtime_error("FLAGFFT_MACA_TAIL_POLICY must be 0 or 1");
  }
  return true;
}

inline bool maca_663000_c2c_scope(const FFTRequest& request) {
  return request.device_type == "maca" && request.device_arch == "102" &&
      request.raw_dim == 1 && request.origin_rank <= 1 && !request.real_transform &&
      !request.convolution_child && !request.packed_real_child &&
      request.requested_n == 663000 && request.fft_length == 663000 &&
      request.input_dtype == request.output_dtype &&
      !request.input_strides.empty() && request.input_strides.back() == 1;
}

inline bool maca_663000_c2c_enabled(const FFTRequest& request) {
  if (!maca_663000_c2c_scope(request) || request.input_dtype != "complex64") return false;
  const char* setting = std::getenv("FLAGFFT_MACA_663000_C2C");
  if (setting != nullptr && std::string(setting) != "1") return false;
  if (request.batch == 1) return maca_tail_policy_enabled(request);
  const char* batch_policy = std::getenv("FLAGFFT_MACA_1D_BATCH");
  return request.batch == 64 && (batch_policy == nullptr || std::string(batch_policy) != "0");
}

inline bool maca_663000_batch_split_column_enabled(const FFTRequest& request) {
  if (request.batch != 64) return false;
  const char* setting = std::getenv("FLAGFFT_MACA_663000_BATCH_SPLIT_COLUMN");
  const char* codelet = std::getenv("FLAGFFT_MACA_RADER_CODELET");
  const std::string mode = setting ? setting : (codelet && std::string(codelet) == "0" ? "0" : "pfa");
  return (mode == "1" || mode == "pfa") &&
      maca_663000_c2c_enabled(request) && request.input_layout == "contiguous" &&
      !request.requires_contiguous_copy;
}

inline bool maca_663000_real_enabled(const FFTRequest& request) {
  const int64_t length = request.packed_real_child ? 331500 : 663000;
  if (request.device_type != "maca" || request.device_arch != "102" ||
      request.raw_dim != 1 || request.origin_rank > 1 || !request.real_transform ||
      request.convolution_child || request.requested_n != length || request.fft_length != length ||
      (request.input_dtype != "complex64" && request.input_dtype != "complex128") ||
      request.output_dtype != request.input_dtype || request.input_layout != "contiguous" ||
      request.requires_contiguous_copy || request.input_strides.empty() ||
      request.input_strides.back() != 1) return false;
  const char* setting = std::getenv("FLAGFFT_MACA_663000_REAL");
  if (setting && std::string(setting) != "1") return false;
  const char* plan = std::getenv("FLAGFFT_MACA_TAIL_PLAN");
  if (plan && *plan) return false;
  const auto& kind = request.real_transform_kind;
  if (request.batch == 1 && request.direction == "forward" && (kind == "r2c" || kind == "d2z")) {
    return maca_tail_policy_enabled(request);
  }
  const char* batch_policy = std::getenv("FLAGFFT_MACA_1D_BATCH");
  return request.batch == 64 && request.direction == "inverse" && (kind == "c2r" || kind == "z2d") &&
      (batch_policy == nullptr || std::string(batch_policy) != "0");
}

inline bool maca_663000_packed_column_pfa_enabled(const FFTRequest& request) {
  constexpr int64_t length = 331500;
  if (request.device_type != "maca" || request.device_arch != "102" ||
      request.raw_dim != 1 || request.origin_rank > 1 || !request.packed_real_child ||
      !request.real_transform || request.convolution_child || request.requested_n != length ||
      request.fft_length != length || request.input_dtype != request.output_dtype ||
      request.input_layout != "contiguous" || request.requires_contiguous_copy ||
      (request.direction != "forward" && request.direction != "inverse") ||
      request.input_strides != std::vector<int64_t>{length, 1}) return false;

  const auto& kind = request.real_transform_kind;
  const bool forward = request.direction == "forward";
  if (request.input_dtype == "complex64") {
    if ((request.batch != 1 && request.batch != 64) || kind != (forward ? "r2c" : "c2r")) return false;
  } else if (request.input_dtype != "complex128" || request.batch != 64 ||
             kind != (forward ? "d2z" : "z2d")) {
    return false;
  }
  const char* setting = std::getenv("FLAGFFT_MACA_663000_PACKED_COLUMN_PFA");
  const bool legacy_inverse_override = request.batch == 64 && kind == "c2r" &&
      std::getenv("FLAGFFT_MACA_663000_PACKED_C2R_SPLIT");
  const char* real = std::getenv("FLAGFFT_MACA_663000_REAL");
  const char* batch = std::getenv("FLAGFFT_MACA_1D_BATCH");
  const char* plan = std::getenv("FLAGFFT_MACA_TAIL_PLAN");
  return (setting ? std::string(setting) == "1" : !legacy_inverse_override) &&
      (!real || std::string(real) == "1") &&
      (!plan || !*plan) && (request.batch == 1 ? maca_tail_policy_enabled(request) :
                           (!batch || std::string(batch) != "0"));
}

inline bool maca_663000_packed_c2r_split_enabled(const FFTRequest& request) {
  const char* setting = std::getenv("FLAGFFT_MACA_663000_PACKED_C2R_SPLIT");
  return (!setting || std::string(setting) == "1") && request.packed_real_child &&
      request.batch == 64 && request.input_dtype == "complex64" &&
      request.real_transform_kind == "c2r" &&
      request.input_strides == std::vector<int64_t>{331500, 1} &&
      !maca_663000_packed_column_pfa_enabled(request) &&
      maca_663000_real_enabled(request);
}

inline bool maca_663000_fp64_packed_column_order_enabled(const FFTRequest& request) {
  constexpr int64_t length = 331500;
  if (request.device_arch != "102" || request.requested_n != length || request.fft_length != length ||
      !request.real_transform || !request.packed_real_child || request.convolution_child ||
      request.input_dtype != "complex128" || request.output_dtype != "complex128" ||
      request.input_layout != "contiguous" || request.requires_contiguous_copy ||
      request.input_strides != std::vector<int64_t>{length, 1}) return false;
  if (!((request.direction == "forward" && request.real_transform_kind == "d2z") ||
        (request.direction == "inverse" && request.real_transform_kind == "z2d"))) return false;

  const char* setting = std::getenv("FLAGFFT_MACA_663000_FP64_PACKED_COLUMN_ORDER");
  const char* real = std::getenv("FLAGFFT_MACA_663000_REAL");
  const char* plan = std::getenv("FLAGFFT_MACA_TAIL_PLAN");
  return (!setting || std::string(setting) == "1") && (!real || std::string(real) == "1") &&
      (!plan || !*plan) && maca_tail_policy_enabled(request);
}

inline bool maca_batch_c2c_option_enabled(const FFTRequest& request, int64_t length,
                                         const char* option, bool default_value) {
  if (request.device_type != "maca" || request.device_arch != "102" ||
      request.raw_dim != 1 || request.origin_rank > 1 || request.batch != 64 ||
      request.requested_n != length || request.fft_length != length || request.real_transform ||
      request.convolution_child || request.packed_real_child ||
      (request.input_dtype != "complex64" && request.input_dtype != "complex128") ||
      request.input_dtype != request.output_dtype || request.input_layout != "contiguous" ||
      request.requires_contiguous_copy || request.input_strides != std::vector<int64_t>{length, 1}) {
    return false;
  }
  const char* setting = std::getenv(option);
  const char* batch_policy = std::getenv("FLAGFFT_MACA_1D_BATCH");
  const char* plan = std::getenv("FLAGFFT_MACA_TAIL_PLAN");
  return (setting ? std::string(setting) == "1" : default_value) &&
      (!batch_policy || std::string(batch_policy) != "0") && (!plan || !*plan);
}

inline bool maca_340200_batch_split_enabled(const FFTRequest& request) {
  return maca_batch_c2c_option_enabled(request, 340200, "FLAGFFT_MACA_340200_BATCH_SPLIT", true);
}

inline bool maca_batch_prime_c2c_four_step_enabled(const FFTRequest& request, int64_t length) {
  if ((length != 8191 && length != 16381) || request.input_dtype != "complex64") return false;
  const bool use_default = !std::getenv("FLAGFFT_MACA_BATCH_RADER_SPLIT") &&
                           !std::getenv("FLAGFFT_MACA_RADER_CODELET");
  return maca_batch_c2c_option_enabled(request, length,
                                       "FLAGFFT_MACA_BATCH_PRIME_RADER_FOUR_STEP", use_default);
}

inline bool maca_batch_prime_real_four_step_enabled(const FFTRequest& request, int64_t length,
                                                  bool require_direct_boundary = false) {
  if (request.device_type != "maca" || request.device_arch != "102" ||
      request.raw_dim != 1 || request.origin_rank > 1 || request.batch != 64 ||
      (length != 8191 && length != 16381) || request.requested_n != length || request.fft_length != length ||
      !request.real_transform || request.packed_real_child || request.convolution_child ||
      request.input_dtype != "complex64" || request.output_dtype != "complex64" ||
      request.input_layout != "contiguous" || request.requires_contiguous_copy ||
      request.input_strides.size() != 2 || request.input_strides.back() != 1) return false;

  const int64_t distance = request.input_strides.front();
  const bool forward = request.real_transform_kind == "r2c" && request.direction == "forward" &&
      (distance == length || distance == length + 1);
  const bool inverse = request.real_transform_kind == "c2r" && request.direction == "inverse" &&
      distance == length / 2 + 1;
  const char* setting = std::getenv("FLAGFFT_MACA_BATCH_PRIME_RADER_REAL_FOUR_STEP");
  const char* direct = std::getenv("FLAGFFT_MACA_BATCH_PRIME_RADER_REAL_DIRECT");
  const bool use_default = !std::getenv("FLAGFFT_MACA_BATCH_RADER_SPLIT") &&
                           !std::getenv("FLAGFFT_MACA_RADER_CODELET");
  const bool direct_enabled = direct ? std::string_view(direct) == "1" : use_default;
  const char* batch_policy = std::getenv("FLAGFFT_MACA_1D_BATCH");
  const char* plan = std::getenv("FLAGFFT_MACA_TAIL_PLAN");
  return (forward || inverse) && (setting ? std::string_view(setting) == "1" : (use_default || direct_enabled)) &&
      (!require_direct_boundary || direct_enabled) &&
      (!batch_policy || std::string_view(batch_policy) != "0") && (!plan || !*plan);
}

inline bool maca_dense_rank1_fp64_complex(const FFTRequest& request, int64_t length) {
  return request.device_type == "maca" && request.device_arch == "102" && request.raw_dim == 1 &&
      request.origin_rank <= 1 && request.requested_n == length && request.fft_length == length &&
      !request.real_transform && !request.convolution_child && !request.packed_real_child &&
      request.input_dtype == "complex128" && request.output_dtype == "complex128" &&
      request.input_layout == "contiguous" && !request.requires_contiguous_copy &&
      request.input_strides == std::vector<int64_t>{length, 1};
}

inline bool maca_fp64_fft16_native_enabled(const FFTRequest& request) {
  if ((request.batch != 1 && request.batch != 64) || !maca_dense_rank1_fp64_complex(request, 16)) {
    return false;
  }
  const char* setting = std::getenv("FLAGFFT_MACA_16_FP64_NATIVE");
  const char* plan = std::getenv("FLAGFFT_MACA_TAIL_PLAN");
  if ((setting && std::string_view(setting) != "1") || (plan && *plan)) return false;
  if (request.batch == 1) return maca_tail_policy_enabled(request);
  const char* pack16 = std::getenv("FLAGFFT_MACA_16_FP64_BATCH_PACK16");
  if (!setting && pack16 && std::string_view(pack16) == "1") return false;
  const char* batch_policy = std::getenv("FLAGFFT_MACA_1D_BATCH");
  return !batch_policy || std::string_view(batch_policy) != "0";
}

inline bool maca_fp64_fft23_native_enabled(const FFTRequest& request) {
  if (request.batch != 1 || !maca_dense_rank1_fp64_complex(request, 23)) return false;
  const char* setting = std::getenv("FLAGFFT_MACA_23_FP64_NATIVE");
  const char* plan = std::getenv("FLAGFFT_MACA_TAIL_PLAN");
  return (!setting || std::string_view(setting) == "1") && (!plan || !*plan) &&
      maca_tail_policy_enabled(request);
}

inline bool maca_fp64_d2z16_native_enabled(const FFTRequest& request) {
  constexpr int64_t length = 16;
  if (request.device_type != "maca" || request.device_arch != "102" ||
      request.raw_dim != 1 || request.origin_rank > 1 ||
      request.requested_n != length || request.fft_length != length ||
      !request.real_transform || request.real_transform_kind != "d2z" || request.direction != "forward" ||
      request.convolution_child || request.packed_real_child ||
      request.input_dtype != "complex128" || request.output_dtype != "complex128" ||
      request.input_layout != "contiguous" || request.requires_contiguous_copy ||
      request.input_strides.size() != 2 || request.input_strides.back() != 1 ||
      (request.input_strides.front() != length && request.input_strides.front() != length + 2)) {
    return false;
  }
  const char* setting = std::getenv("FLAGFFT_MACA_16_D2Z_NATIVE");
  const char* plan = std::getenv("FLAGFFT_MACA_TAIL_PLAN");
  if ((setting && std::string_view(setting) != "1") || (plan && *plan)) return false;
  if (request.batch == 1) return maca_tail_policy_enabled(request);
  const char* batch_policy = std::getenv("FLAGFFT_MACA_1D_BATCH");
  return request.batch == 64 && (!batch_policy || std::string_view(batch_policy) != "0");
}

inline bool maca_340200_fp32_batch_900x378_enabled() {
  const char* setting = std::getenv("FLAGFFT_MACA_340200_FP32_BATCH_900X378");
  if (setting) return std::string_view(setting) == "1";
  return std::getenv("FLAGFFT_MACA_340200_FP32_RADIX_ORDER") == nullptr;
}

inline bool maca_340200_fp64_forward_column_phase_enabled(const FFTRequest& request) {
  const char* setting = std::getenv("FLAGFFT_MACA_340200_FP64_FORWARD_COLUMN_PHASE");
  return request.input_dtype == "complex128" && request.direction == "forward" &&
      (!setting || std::string_view(setting) == "1");
}

inline bool maca_340200_single_split_enabled(const FFTRequest& request) {
  constexpr int64_t length = 340200;
  const bool fp64 = request.input_dtype == "complex128";
  if (request.device_arch != "102" || request.requested_n != length || request.fft_length != length ||
      request.real_transform || request.convolution_child || request.packed_real_child ||
      (!fp64 && request.input_dtype != "complex64") || request.output_dtype != request.input_dtype ||
      request.input_layout != "contiguous" || request.requires_contiguous_copy ||
      request.input_strides != std::vector<int64_t>{length, 1}) return false;
  const char* setting = std::getenv(fp64 ? "FLAGFFT_MACA_340200_FP64_SINGLE_SPLIT"
                                         : "FLAGFFT_MACA_340200_FP32_SINGLE_SPLIT");
  const char* plan = std::getenv("FLAGFFT_MACA_TAIL_PLAN");
  const bool default_enabled = !fp64 || request.direction == "inverse" ||
      maca_340200_fp64_forward_column_phase_enabled(request);
  return (setting ? std::string_view(setting) == "1" : default_enabled) && (!plan || !*plan) &&
      maca_tail_policy_enabled(request);
}

inline bool maca_batch_packed_real_option_enabled(const FFTRequest& request, int64_t length,
                                                  const char* option, bool default_value = false) {
  if (request.device_type != "maca" || request.device_arch != "102" ||
      request.raw_dim != 1 || request.origin_rank > 1 || request.batch != 64 ||
      request.requested_n != length || request.fft_length != length ||
      !request.real_transform || !request.packed_real_child || request.convolution_child ||
      (request.input_dtype != "complex64" && request.input_dtype != "complex128") ||
      request.input_dtype != request.output_dtype ||
      request.input_layout != "contiguous" || request.requires_contiguous_copy ||
      request.input_strides != std::vector<int64_t>{length, 1}) return false;
  const bool fp64 = request.input_dtype == "complex128";
  if (!((request.direction == "forward" && request.real_transform_kind == (fp64 ? "d2z" : "r2c")) ||
        (request.direction == "inverse" && request.real_transform_kind == (fp64 ? "z2d" : "c2r")))) return false;

  const char* setting = std::getenv(option);
  const char* batch_policy = std::getenv("FLAGFFT_MACA_1D_BATCH");
  const char* plan = std::getenv("FLAGFFT_MACA_TAIL_PLAN");
  return (setting ? std::string(setting) == "1" : default_value) &&
      (!batch_policy || std::string(batch_policy) != "0") && (!plan || !*plan);
}

inline bool maca_340200_packed_real_split_enabled(const FFTRequest& request) {
  return request.input_dtype == "complex64" &&
      maca_batch_packed_real_option_enabled(request, 170100, "FLAGFFT_MACA_340200_PACKED_REAL_SPLIT", true);
}

inline bool maca_16384_packed_real_split_enabled(const FFTRequest& request) {
  return maca_batch_packed_real_option_enabled(request, 8192, "FLAGFFT_MACA_16384_PACKED_REAL_SPLIT", true);
}

inline bool maca_16384_fp32_batch_row_twiddle_enabled(const FFTRequest& request) {
  return request.input_dtype == "complex64" &&
      maca_batch_c2c_option_enabled(request, 16384, "FLAGFFT_MACA_16384_FP32_BATCH_ROW_TWIDDLE", true);
}

inline bool maca_16384_fp32_batch_row_phase_enabled(const FFTRequest& request) {
  const char* setting = std::getenv("FLAGFFT_MACA_16384_FP32_BATCH_ROW_PHASE");
  return (!setting || std::string_view(setting) == "1") && maca_16384_fp32_batch_row_twiddle_enabled(request);
}

inline bool maca_524288_packed_real_split_enabled(const FFTRequest& request) {
  const char* row_pack = std::getenv("FLAGFFT_MACA_524288_PACKED_REAL_ROW_PACK4");
  return request.input_dtype == "complex64" &&
      maca_batch_packed_real_option_enabled(request, 262144, "FLAGFFT_MACA_524288_PACKED_REAL_SPLIT",
                                           !row_pack || std::string(row_pack) != "1");
}

inline bool maca_185640_packed_real_split_enabled(const FFTRequest& request) {
  return maca_batch_packed_real_option_enabled(request, 92820, "FLAGFFT_MACA_185640_PACKED_REAL_SPLIT", true);
}

inline bool maca_16384_fp64_batch_split_enabled(const FFTRequest& request) {
  return request.input_dtype == "complex128" &&
      maca_batch_c2c_option_enabled(request, 16384, "FLAGFFT_MACA_16384_FP64_BATCH_SPLIT", true);
}

inline bool maca_210_fp32_batch_radix6_enabled(const FFTRequest& request) {
  return request.input_dtype == "complex64" &&
      maca_batch_c2c_option_enabled(request, 210, "FLAGFFT_MACA_210_BATCH_RADIX6", false);
}

inline bool maca_524288_fp64_register_pack_enabled(const FFTRequest& request) {
  constexpr int64_t length = 524288;
  if (request.requested_n != length || request.fft_length != length || request.device_arch != "102" ||
      request.real_transform || request.convolution_child || request.packed_real_child ||
      request.input_dtype != "complex128" || request.output_dtype != "complex128" ||
      request.input_layout != "contiguous" || request.requires_contiguous_copy ||
      request.input_strides != std::vector<int64_t>{length, 1}) return false;
  const char* setting = std::getenv("FLAGFFT_MACA_524288_FP64_REGISTER_PACK");
  const char* plan = std::getenv("FLAGFFT_MACA_TAIL_PLAN");
  return (!setting || std::string(setting) == "1") && (!plan || !*plan) && maca_tail_policy_enabled(request);
}

inline std::string_view maca_524288_fp64_single_z2d_mode(const FFTRequest& request) {
  const int64_t length = request.packed_real_child ? 262144 : 524288;
  const int64_t distance = request.packed_real_child ? length : length / 2 + 1;
  if (request.device_arch != "102" || request.requested_n != length || request.fft_length != length ||
      !request.real_transform || request.convolution_child || request.real_transform_kind != "z2d" ||
      request.direction != "inverse" || request.input_dtype != "complex128" ||
      request.output_dtype != "complex128" || request.input_layout != "contiguous" ||
      request.requires_contiguous_copy || request.input_strides != std::vector<int64_t>{distance, 1} ||
      !maca_tail_policy_enabled(request)) return {};
  const char* setting = std::getenv("FLAGFFT_MACA_524288_FP64_SINGLE_Z2D");
  const std::string_view mode = setting ? setting : "paired";
  return mode == "vector" || mode == "paired" ? mode : std::string_view{};
}

inline bool maca_prime_dif_requested(const FFTRequest& request) {
  if (request.device_type != "maca" || request.device_arch != "102" ||
      request.raw_dim != 1 || request.origin_rank > 1 ||
      (request.batch != 1 && request.batch != 64) || request.convolution_child ||
      request.packed_real_child || request.fft_length != request.requested_n ||
      (request.requested_n != 997 && request.requested_n != 1009) ||
      (request.input_dtype != "complex64" && request.input_dtype != "complex128") ||
      request.output_dtype != request.input_dtype || request.input_layout != "contiguous" ||
      request.requires_contiguous_copy || request.input_strides.empty() ||
      request.input_strides.back() != 1 ||
      (request.real_transform && request.real_transform_kind.empty())) return false;
  const char* setting = std::getenv("FLAGFFT_MACA_PRIME_DIF");
  if (setting) return std::string(setting) == "1";
  const char* plan = std::getenv("FLAGFFT_MACA_TAIL_PLAN");
  if (plan && *plan) return false;
  if (request.batch == 1) return maca_tail_policy_enabled(request);
  const char* batch_policy = std::getenv("FLAGFFT_MACA_1D_BATCH");
  return batch_policy == nullptr || std::string(batch_policy) != "0";
}

inline bool maca_batch_prime_bluestein_enabled(const FFTRequest& request) {
  const auto n = request.requested_n;
  if (request.device_type != "maca" || request.device_arch != "102" ||
      request.raw_dim != 1 || request.origin_rank > 1 || request.batch != 64 ||
      (n != 8191 && n != 16381) || request.fft_length != n || request.real_transform ||
      request.convolution_child || request.packed_real_child ||
      request.input_dtype != "complex64" || request.output_dtype != "complex64" ||
      request.input_layout != "contiguous" || request.requires_contiguous_copy ||
      request.input_strides != std::vector<int64_t>{n, 1}) return false;
  const char* batch_policy = std::getenv("FLAGFFT_MACA_1D_BATCH");
  const char* setting = std::getenv("FLAGFFT_MACA_BATCH_PRIME_BLUESTEIN");
  return setting && std::string(setting) == "1" &&
      (!batch_policy || std::string(batch_policy) != "0");
}

// An explicit old root experiment, including 'default', wins over the policy.
inline std::string maca_tail_automatic_plan(const FFTRequest& request) {
  const char* old = std::getenv("FLAGFFT_MACA_TAIL_PLAN");
  if (old && *old) return {};
  if (maca_prime_dif_requested(request)) return "bs2048";
  if (maca_batch_prime_bluestein_enabled(request)) {
    return request.requested_n == 8191 ? "bs16384" : "bs32768";
  }
  if (maca_663000_packed_c2r_split_enabled(request)) return "ct340x975";
  if (request.batch == 64 && maca_663000_c2c_enabled(request)) {
    return maca_663000_batch_split_column_enabled(request) ? "ct375x1768" : "ct780x850";
  }
  if (!maca_tail_policy_enabled(request)) return {};
  const auto n = request.requested_n;
  if (request.real_transform) {
    // Only these real-prime/API pairs have paired device evidence.
    if (request.input_dtype == "complex128" && n == 997 &&
        (request.real_transform_kind == "d2z" || request.real_transform_kind == "z2d")) {
      return "bs2048";
    }
    if (request.input_dtype == "complex64" && n == 1009 &&
        (request.real_transform_kind == "r2c" || request.real_transform_kind == "c2r")) {
      return "bs2048";
    }
    return {};
  }
  if (request.input_dtype != request.output_dtype) return {};
  if (request.input_dtype == "complex128") {
    if (n == 997) return "bs2048";
    if (n == 1048576) return "ct1024x1024";
    if (n == 524287 && request.fft_length == n && request.device_arch == "102" &&
        !request.convolution_child && !request.packed_real_child &&
        request.input_layout == "contiguous" && !request.requires_contiguous_copy &&
        request.input_strides == std::vector<int64_t>{n, 1}) return "bs1048576";
  } else if (request.input_dtype == "complex64") {
    if (n == 1009) return "bs2048";
  }
  return {};
}

inline bool maca_tail_real_direct_dft(const FFTRequest& request) {
  if (!maca_tail_policy_enabled(request) || !request.real_transform ||
      (request.input_dtype != "complex64" && request.input_dtype != "complex128")) {
    return false;
  }
  // For small DirectDFT roots, boundary fusion is useful while launch/setup
  // overhead dominates. Device evidence covers the 23..37 interval; keep the
  // lower bound explicit because n=1 is not a meaningful FFT workload.
  if (request.requested_n >= 2 && request.requested_n <= 23) return true;
  // At the upper end, only R2C/C2R crossed the useful gate. D2Z/Z2D stays on
  // the production path until a separate measurement proves otherwise.
  return request.requested_n >= 24 && request.requested_n <= 37 &&
         (request.real_transform_kind == "r2c" || request.real_transform_kind == "c2r");
}

inline std::string maca_tail_codegen_root(const FFTRequest& request) {
  if (maca_tail_real_direct_dft(request)) return "real-direct";
  if (maca_340200_single_split_enabled(request)) {
    if (request.input_dtype == "complex128") {
      return maca_340200_fp64_forward_column_phase_enabled(request)
          ? "z2z340200-forward-column-phase" : "z2z340200-single";
    }
    const char* inverse_pack = std::getenv("FLAGFFT_MACA_340200_FP32_SINGLE_INVERSE_PACK");
    const bool use_inverse_pack = request.direction == "inverse" &&
        (!inverse_pack || std::string_view(inverse_pack) == "1");
    const char* row_phase = std::getenv("FLAGFFT_MACA_340200_FP32_SINGLE_ROW_PHASE");
    if (!row_phase || std::string_view(row_phase) == "1") {
      return use_inverse_pack ? "c2c340200-inverse-pack-row-phase" : "c2c340200-single-row-phase";
    }
    return use_inverse_pack ? "c2c340200-inverse-pack" : "c2c340200-single";
  }
  if (maca_340200_batch_split_enabled(request)) {
    if (request.input_dtype == "complex128") return "z2z340200-split";
    return maca_340200_fp32_batch_900x378_enabled() ? "c2c340200-batch-900x378" : "c2c340200-split";
  }
  if (maca_340200_packed_real_split_enabled(request)) return "batch-real340200-split";
  if (maca_16384_packed_real_split_enabled(request)) return "batch-real16384-split";
  if (maca_524288_packed_real_split_enabled(request)) return "batch-real524288-split";
  if (maca_185640_packed_real_split_enabled(request)) {
    return request.input_dtype == "complex128" ? "batch-real185640-p4" : "batch-real185640-p2";
  }
  if (maca_16384_fp32_batch_row_twiddle_enabled(request)) {
    if (maca_16384_fp32_batch_row_phase_enabled(request)) return "c2c16384-tiled-row-phase";
    return "c2c16384-row-twiddle";
  }
  if (maca_16384_fp64_batch_split_enabled(request)) return "z2z16384-split";
  if (maca_524288_fp64_register_pack_enabled(request)) return "z2z524288-pack4";
  if (request.packed_real_child && !maca_524288_fp64_single_z2d_mode(request).empty()) return "batch-vecio";
  if (request.input_dtype == "complex128" &&
      maca_batch_c2c_option_enabled(request, 16, "FLAGFFT_MACA_16_FP64_BATCH_PACK16", false)) {
    return "z2z16-pack16";
  }
  if (maca_663000_c2c_enabled(request)) {
    return request.batch == 64 && !maca_663000_batch_split_column_enabled(request)
        ? "c2c663000-batch" : "c2c663000";
  }
  if (request.packed_real_child && maca_663000_real_enabled(request)) {
    if (maca_663000_packed_c2r_split_enabled(request)) return "batch-c2r663000-split";
    if (request.batch == 64) return "batch-c2r663000";
    if (request.input_dtype == "complex64") return "r2c663000";
  }
  if (request.device_type == "maca" && request.device_arch == "102" &&
      request.raw_dim == 1 && request.origin_rank <= 1 && request.batch == 64 &&
      request.fft_length == request.requested_n && !request.input_strides.empty() &&
      request.input_strides.back() == 1) {
    const char* batch_policy = std::getenv("FLAGFFT_MACA_1D_BATCH");
    if (!batch_policy || std::string(batch_policy) != "0") {
      const char* vec_policy = std::getenv("FLAGFFT_MACA_BATCH_VEC_IO");
      if ((!vec_policy || std::string(vec_policy) != "0") &&
          (request.input_dtype == "complex64" || request.input_dtype == "complex128") &&
          ((!request.real_transform && !request.packed_real_child &&
            (request.requested_n == 16384 || request.requested_n == 524288 ||
             (request.requested_n == 185640 && request.input_dtype == "complex64"))) ||
           (request.real_transform && request.packed_real_child &&
            (request.requested_n == 8192 || request.requested_n == 262144)))) {
        if (request.input_dtype == "complex64" && maca_batch_packed_real_option_enabled(
                request, 262144, "FLAGFFT_MACA_524288_PACKED_REAL_ROW_PACK4")) {
          return "batch-real524288-row4";
        }
        return "batch-vecio";
      }
      const char* pack_policy = std::getenv("FLAGFFT_MACA_BATCH_REAL_PACK2");
      if ((!pack_policy || std::string(pack_policy) != "0") &&
          request.real_transform && request.packed_real_child &&
          request.input_dtype == "complex128" && request.requested_n == 92820) {
        return "batch-real-pack2";
      }
      const char* c2c_pack_policy = std::getenv("FLAGFFT_MACA_BATCH_C2C_PACK2");
      if ((!c2c_pack_policy || std::string(c2c_pack_policy) != "0") &&
          !request.real_transform && !request.packed_real_child &&
          request.input_dtype == "complex128" && request.output_dtype == "complex128" &&
          request.requested_n == 185640) return "batch-c2c-pack2";
      const char* prime_vec_policy = std::getenv("FLAGFFT_MACA_BATCH_PRIME_VEC_IO");
      if ((!prime_vec_policy || std::string(prime_vec_policy) != "0") &&
          !request.packed_real_child &&
          (request.input_dtype == "complex64" || request.input_dtype == "complex128") &&
          (request.requested_n == 8191 || request.requested_n == 16381)) {
        return "batch-prime-vecio";
      }
      const char* c2c_tree_policy = std::getenv("FLAGFFT_MACA_BATCH_C2C_TREE");
      if ((!c2c_tree_policy || std::string(c2c_tree_policy) != "0") &&
          !request.real_transform && !request.packed_real_child &&
          request.requested_n == 23 && request.input_dtype == "complex128" &&
          request.output_dtype == "complex128") return "batch-c2c-tree";
    }
  }
  // Batch-64 N=23 real transforms already use the fused direct DFT.  Give
  // that root the same FP64 tree reduction and distinct JIT identity as the
  // measured single-transform path, while honoring both batch opt-outs.
  if (request.device_type == "maca" && request.device_arch == "102" &&
      request.raw_dim == 1 && request.origin_rank <= 1 && request.batch == 64 &&
      request.requested_n == 23 && request.fft_length == 23 && request.real_transform &&
      (request.input_dtype == "complex64" || request.input_dtype == "complex128")) {
    const char* batch_policy = std::getenv("FLAGFFT_MACA_1D_BATCH");
    const char* direct_policy = std::getenv("FLAGFFT_MACA_BATCH_REAL_DIRECT_DFT");
    if ((!batch_policy || std::string(batch_policy) != "0") &&
        (!direct_policy || std::string(direct_policy) != "0")) return "real-direct";
  }
  if (maca_tail_automatic_plan(request) == "ct1024x1024") return "p4w4";
  return "off";
}

inline std::string maca_tail_kernel_mode(const std::string& root, KernelKind kind,
                                         const std::string& dtype, int64_t length,
                                         int64_t n1, int64_t n2) {
  if (dtype == "complex128" && n2 == 1024 &&
      ((root == "p4w4" && n1 == 1024) || (root == "z2z524288-pack4" && n1 == 512)) &&
      ((kind == KernelKind::FourStepRow && length == n1) ||
       (kind == KernelKind::FourStepCol && length == n2))) return root;
  if (root == "real-direct" && length >= 2 && length <= 37 &&
      (kind == KernelKind::DirectDftR2C || kind == KernelKind::DirectDftC2R)) return "real-direct";
  if (root == "batch-c2c-tree" && kind == KernelKind::DirectDft &&
      dtype == "complex128" && length == 23) return root;
  if (root == "z2z16-pack16" && kind == KernelKind::Leaf &&
      dtype == "complex128" && length == 16) return root;
  if (root == "batch-prime-vecio" &&
      (kind == KernelKind::RaderPrepare || kind == KernelKind::RaderFinalize) &&
      (dtype == "complex64" || dtype == "complex128")) return "batch-prime-rader-vecio";
  if ((root == "c2c16384-row-twiddle" || root == "c2c16384-tiled-row-phase") &&
      dtype == "complex64" && n1 == 256 && n2 == 64 &&
      ((kind == KernelKind::FourStepRow && length == n1) ||
       (kind == KernelKind::FourStepCol && length == n2))) return root;
  if (((dtype == "complex64" && (root == "c2c340200-single" || root == "c2c340200-inverse-pack" ||
                               root == "c2c340200-single-row-phase" || root == "c2c340200-inverse-pack-row-phase" ||
                               root == "c2c340200-batch-900x378")) ||
       (dtype == "complex128" && (root == "z2z340200-single" ||
                                 root == "z2z340200-forward-column-phase"))) && n1 == 900 && n2 == 378 &&
      ((kind == KernelKind::FourStepRow && length == n1) ||
       (kind == KernelKind::FourStepCol && length == n2))) return root;
  if ((kind == KernelKind::FourStepRow || kind == KernelKind::FourStepCol) &&
      (length == n1 || length == n2)) {
    if (dtype == "complex64" &&
        ((root == "c2c663000" && n1 == 375 && n2 == 1768) ||
         (root == "c2c663000-batch" && n1 == 780 && n2 == 850))) return root;
    if (n1 == 300 && n2 == 1105 &&
        ((root == "r2c663000" && dtype == "complex64") ||
         (root == "batch-c2r663000" && (dtype == "complex64" || dtype == "complex128")))) return root;
    if (root == "batch-c2r663000-split" && dtype == "complex64" && n1 == 340 && n2 == 975) return root;
    if ((root == "c2c340200-split" && dtype == "complex64" && n1 == 567 && n2 == 600) ||
        (root == "z2z340200-split" && dtype == "complex128" && n1 == 504 && n2 == 675)) return root;
    if (root == "batch-real340200-split" && dtype == "complex64" && n1 == 405 && n2 == 420) return root;
    if (root == "batch-real16384-split" && (dtype == "complex64" || dtype == "complex128") &&
        n1 == 128 && n2 == 64) return root;
    if (root == "batch-real524288-row4" && dtype == "complex64" && n1 == 512 && n2 == 512) {
      return kind == KernelKind::FourStepRow ? root : "batch-vecio";
    }
    if (root == "batch-real524288-split" && dtype == "complex64" && n1 == 256 && n2 == 1024) return root;
    if (n1 == 420 && n2 == 221 &&
        ((root == "batch-real185640-p2" && dtype == "complex64") ||
         (root == "batch-real185640-p4" && dtype == "complex128"))) return root;
    if (root == "z2z16384-split" && dtype == "complex128" && n1 == 256 && n2 == 64) return root;
    if (root == "batch-vecio" && (dtype == "complex64" || dtype == "complex128") &&
        ((n1 == 256 && n2 == 64) || (n1 == 128 && n2 == 128) ||
         (n1 == 64 && n2 == 128) ||
         (n1 == 390 && n2 == 476) || (n1 == 512 && n2 == 1024) ||
         (n1 == 512 && n2 == 512))) return root;
    if (root == "batch-real-pack2" && dtype == "complex128" &&
        n1 * n2 == 92820) return root;
    if (root == "batch-c2c-pack2" && dtype == "complex128" &&
        n1 == 390 && n2 == 476) return root;
    if (root == "batch-prime-vecio" &&
        (dtype == "complex64" || dtype == "complex128") &&
        ((n1 == 90 && n2 == 91) || (n1 == 91 && n2 == 180))) return root;
  }
  return "off";
}

// Also isolate the legacy tree/pack experiments when no automatic policy is
// enabled. Length-prefixed values distinguish unset, empty, and arbitrary text.
inline std::string maca_tail_codegen_identity(const std::string& mode) {
  std::string identity = ";maca-tail-v1=" + mode;
  for (const char* name : {"FLAGFFT_MACA_REAL_DFT_REDUCTION", "FLAGFFT_MACA_EXCHANGE",
                           "FLAGFFT_MACA_BATCH_PACK", "FLAGFFT_MACA_3D_N64_PACK",
                           "FLAGFFT_MACA_3D_N32_FP64_PACK",
                           "FLAGFFT_MACA_3D_PERMSTORE_PACK",
                           "FLAGFFT_MACA_FP64_REGISTER_PACK", "FLAGFFT_MACA_INNER_PACK",
                           "FLAGFFT_MACA_FOURSTEP_ROW_PACK", "FLAGFFT_MACA_FOURSTEP_COL_PACK",
                           "FLAGFFT_MACA_MAX_WARPS", "FLAGFFT_MACA_SPLIT_ORDER",
                           "FLAGFFT_MACA_VEC_IO", "FLAGFFT_MACA_LANE_MIN",
                           "FLAGFFT_MACA_MIXED_EXCHANGE", "FLAGFFT_MACA_RADER_FULL_LEAF",
                           "FLAGFFT_MACA_RADER_BOUNDARY_LEAF", "FLAGFFT_MACA_RADER_VEC_IO",
                           "FLAGFFT_MACA_TRANSPOSE3D", "FLAGFFT_MACA_TRANSPOSE3D_WARPS",
                           "FLAGFFT_MACA_TRANSPOSE3D_FP64", "FLAGFFT_MACA_TRANSPOSE3D_TRAVERSAL",
                           "FLAGFFT_MACA_TRANSPOSE3D_SLICE_GROUP",
                           "FLAGFFT_MACA_3D_FP64_COMPACT_TRANSPOSE",
                           "FLAGFFT_MACA_3D_FIRST_STORE",
                           "FLAGFFT_MACA_3D_FINAL_STORE",
                           "FLAGFFT_MACA_3D_R2C_FIRST_STORE",
                           "FLAGFFT_MACA_3D_REAL_FUSED16",
                           "FLAGFFT_MACA_3D_C2C_FUSED32",
                           "FLAGFFT_MACA_3D_C2C32_MORE_LANES",
                           "FLAGFFT_MACA_RADER_CODELET",
                           "FLAGFFT_MACA_COMPACT_STAGES"}) {
    const char* value = std::getenv(name);
    identity += std::string(";") + name + "=";
    identity += value ? std::to_string(std::string(value).size()) + ":" + value : "unset";
  }
  return identity;
}

}  // namespace flagfft
