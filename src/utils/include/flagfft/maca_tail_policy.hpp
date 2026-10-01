// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "flagfft/base.hpp"

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

// An explicit old root experiment, including 'default', wins over the policy.
inline std::string maca_tail_automatic_plan(const FFTRequest& request) {
  const char* old = std::getenv("FLAGFFT_MACA_TAIL_PLAN");
  if (!maca_tail_policy_enabled(request) || (old && *old)) return {};
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
  if (root == "p4w4" && dtype == "complex128" && length == 1024 &&
      n1 == 1024 && n2 == 1024 &&
      (kind == KernelKind::FourStepRow || kind == KernelKind::FourStepCol)) return "p4w4";
  if (root == "real-direct" && length >= 2 && length <= 37 &&
      (kind == KernelKind::DirectDftR2C || kind == KernelKind::DirectDftC2R)) return "real-direct";
  if (root == "batch-c2c-tree" && kind == KernelKind::DirectDft &&
      dtype == "complex128" && length == 23) return root;
  if (root == "batch-prime-vecio" &&
      (kind == KernelKind::RaderPrepare || kind == KernelKind::RaderFinalize) &&
      (dtype == "complex64" || dtype == "complex128")) return "batch-prime-rader-vecio";
  if ((kind == KernelKind::FourStepRow || kind == KernelKind::FourStepCol) &&
      (length == n1 || length == n2)) {
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
                           "FLAGFFT_MACA_3D_PERMSTORE_PACK",
                           "FLAGFFT_MACA_FP64_REGISTER_PACK", "FLAGFFT_MACA_INNER_PACK",
                           "FLAGFFT_MACA_MAX_WARPS", "FLAGFFT_MACA_SPLIT_ORDER",
                           "FLAGFFT_MACA_VEC_IO", "FLAGFFT_MACA_LANE_MIN",
                           "FLAGFFT_MACA_MIXED_EXCHANGE", "FLAGFFT_MACA_RADER_FULL_LEAF",
                           "FLAGFFT_MACA_RADER_BOUNDARY_LEAF", "FLAGFFT_MACA_RADER_VEC_IO",
                           "FLAGFFT_MACA_TRANSPOSE3D", "FLAGFFT_MACA_TRANSPOSE3D_WARPS",
                           "FLAGFFT_MACA_TRANSPOSE3D_FP64", "FLAGFFT_MACA_TRANSPOSE3D_TRAVERSAL",
                           "FLAGFFT_MACA_3D_FIRST_STORE",
                           "FLAGFFT_MACA_3D_MIDDLE_STORE",
                           "FLAGFFT_MACA_3D_MIDDLE_STORE_PACK",
                           "FLAGFFT_MACA_3D_R2C_FIRST_STORE",
                           "FLAGFFT_MACA_3D_REAL_FUSED16",
                           "FLAGFFT_MACA_3D_C2C_FUSED32"}) {
    const char* value = std::getenv(name);
    identity += std::string(";") + name + "=";
    identity += value ? std::to_string(std::string(value).size()) + ":" + value : "unset";
  }
  return identity;
}

}  // namespace flagfft
