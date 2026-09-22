// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "flagfft/base.hpp"

namespace flagfft {

inline bool maca_tail_policy_enabled(const FFTRequest& request) {
  const char* value = std::getenv("FLAGFFT_MACA_TAIL_POLICY");
  if (request.device_type != "maca" || request.raw_dim != 1 || request.batch != 1 ||
      (request.origin_rank != 0 && request.origin_rank != 1)) return false;
  if (!value || !*value || std::string(value) == "0") return false;
  if (std::string(value) != "1")
    throw std::runtime_error("FLAGFFT_MACA_TAIL_POLICY must be 0 or 1");
  return true;
}

// An explicit old root experiment, including 'default', wins over the policy.
inline std::string maca_tail_automatic_plan(const FFTRequest& request) {
  const char* old = std::getenv("FLAGFFT_MACA_TAIL_PLAN");
  if (!maca_tail_policy_enabled(request) || (old && *old) || request.real_transform ||
      request.input_dtype != request.output_dtype) return {};
  const auto n = request.requested_n;
  if (request.input_dtype == "complex128") {
    if (n == 997) return "bs2048";
    if (n == 1048576) return "ct1024x1024";
  } else if (request.input_dtype == "complex64") {
    if (n == 1009) return "bs2048";
    if (n == 328050) return "ct405x810";
  }
  return {};
}

inline bool maca_tail_real23(const FFTRequest& request) {
  return maca_tail_policy_enabled(request) && request.requested_n == 23 &&
         request.real_transform &&
         (request.input_dtype == "complex64" || request.input_dtype == "complex128");
}

inline std::string maca_tail_codegen_root(const FFTRequest& request) {
  if (maca_tail_real23(request)) return "real23";
  if (maca_tail_automatic_plan(request) == "ct1024x1024") return "p4w4";
  return "off";
}

inline std::string maca_tail_kernel_mode(const std::string& root, KernelKind kind,
                                         const std::string& dtype, int64_t length,
                                         int64_t n1, int64_t n2) {
  if (root == "p4w4" && dtype == "complex128" && length == 1024 &&
      n1 == 1024 && n2 == 1024 &&
      (kind == KernelKind::FourStepRow || kind == KernelKind::FourStepCol)) return "p4w4";
  if (root == "real23" && length == 23 &&
      (kind == KernelKind::DirectDftR2C || kind == KernelKind::DirectDftC2R)) return "real23";
  return "off";
}

// Also isolate the legacy tree/pack experiments when no automatic policy is
// enabled. Length-prefixed values distinguish unset, empty, and arbitrary text.
inline std::string maca_tail_codegen_identity(const std::string& mode) {
  std::string identity = ";maca-tail-v1=" + mode;
  for (const char* name : {"FLAGFFT_MACA_REAL_DFT_REDUCTION", "FLAGFFT_MACA_EXCHANGE",
                           "FLAGFFT_MACA_FP64_REGISTER_PACK", "FLAGFFT_MACA_INNER_PACK",
                           "FLAGFFT_MACA_MAX_WARPS", "FLAGFFT_MACA_SPLIT_ORDER",
                           "FLAGFFT_MACA_VEC_IO", "FLAGFFT_MACA_LANE_MIN",
                           "FLAGFFT_MACA_MIXED_EXCHANGE"}) {
    const char* value = std::getenv(name);
    identity += std::string(";") + name + "=";
    identity += value ? std::to_string(std::string(value).size()) + ":" + value : "unset";
  }
  return identity;
}

}  // namespace flagfft
