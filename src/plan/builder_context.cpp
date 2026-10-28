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

#include "adaptor/adaptor.h"
#include "flagfft/core.hpp"
#include "maca_tail_plans.hpp"

#include <cstdlib>
#include <string>

namespace flagfft {

bool PlanBuilder::RequestContext::operator==(const RequestContext &other) const {
  return input_dtype == other.input_dtype && output_dtype == other.output_dtype &&
         real_transform_kind == other.real_transform_kind &&
         origin_real_transform_kind == other.origin_real_transform_kind &&
         device_type == other.device_type && device_index == other.device_index &&
         device_arch == other.device_arch &&
         musa_1d_fourstep_split == other.musa_1d_fourstep_split && origin_rank == other.origin_rank &&
         requested_n == other.requested_n &&
         batch == other.batch &&
         maca_3d_c2c32_single_cube == other.maca_3d_c2c32_single_cube &&
         ix_short_single == other.ix_short_single &&
         ix_ct_batch == other.ix_ct_batch &&
         max_dynamic_smem_bytes == other.max_dynamic_smem_bytes;
}

PlanBuilder::RequestContext PlanBuilder::make_request_context(const FFTRequest &request) const {
  RequestContext context;
  context.input_dtype = request.input_dtype;
  context.output_dtype = request.output_dtype;
  context.real_transform_kind = request.real_transform_kind;
  context.origin_real_transform_kind = request.origin_real_transform_kind;
  context.device_type = request.device_type;
  context.device_index = request.device_index;
  context.device_arch = request.device_arch;
  if (request.device_type == "musa" && request.device_arch == "31") {
    const char *split = std::getenv("FLAGFFT_MUSA_1D_FOURSTEP_SPLIT");
    if (split != nullptr) context.musa_1d_fourstep_split = split;
  }
  context.origin_rank = request.origin_rank;
  context.requested_n = request.requested_n;
  context.batch = request.batch;
  context.maca_3d_c2c32_single_cube = request.maca_3d_c2c32_single_cube;
  context.ix_short_single = request.requested_n == 1024 && ix_ct_single_policy_enabled(request);
  context.ix_ct_batch = ix_ct_batch_policy_enabled(request);
  if (request.device_type == adaptor::backend_name()) {
    context.max_dynamic_smem_bytes = adaptor::max_dynamic_smem_bytes(request.device_index);
  }
  return context;
}

void PlanBuilder::set_request_context(const FFTRequest &request) {
  RequestContext next = make_request_context(request);
  if (request_context_.has_value() && *request_context_ == next) {
    return;
  }
  request_context_ = std::move(next);
  node_cache_.clear();
  cost_cache_.clear();
  tune_candidate_cache_.clear();
}

const PlanBuilder::RequestContext &PlanBuilder::request_context() const {
  if (!request_context_.has_value()) {
    throw std::runtime_error("PlanBuilder request context is not initialized");
  }
  return *request_context_;
}

PlanNodePtr PlanBuilder::build(int64_t n, const FFTRequest &request) {
  set_request_context(request);
  if (n <= 0) {
    throw std::runtime_error("FFT length must be positive");
  }
  // Allow a root 1D batched transform on MUSA to screen alternative four-step
  // decompositions without changing child plans.
  const char *musa_fourstep_split = std::getenv("FLAGFFT_MUSA_1D_FOURSTEP_SPLIT");
  const bool musa_1d_c2c_root = !request.real_transform &&
      (request.input_dtype == "complex64" || request.input_dtype == "complex128") &&
      request.input_dtype == request.output_dtype;
  const bool musa_1d_real_root = request.real_transform &&
      (request.real_transform_kind == "r2c" || request.real_transform_kind == "d2z" ||
       request.real_transform_kind == "c2r" || request.real_transform_kind == "z2d");
  const bool musa_default_131072_split = request.device_type == "musa" &&
      request.device_arch == "31" && request.raw_dim == 1 && request.origin_rank == 1 &&
      request.batch > 1 && request.requested_n == n && n == 131072 &&
      (musa_1d_c2c_root || musa_1d_real_root);
  const bool musa_default_1024_split = request.device_type == "musa" &&
      request.device_arch == "31" && request.raw_dim == 1 && request.origin_rank == 1 &&
      request.batch > 1 && request.requested_n == n && n == 1024 &&
      (musa_1d_c2c_root || musa_1d_real_root);
  const bool musa_default_8192_split = request.device_type == "musa" &&
      request.device_arch == "31" && request.raw_dim == 1 && request.origin_rank == 1 &&
      request.batch > 1 && request.requested_n == n && n == 8192 &&
      (musa_1d_c2c_root || musa_1d_real_root);
  const bool musa_default_300000_fp64_c2c_split =
      request.device_type == "musa" && request.device_arch == "31" && request.raw_dim == 1 &&
      request.origin_rank == 1 && request.batch >= 64 && request.requested_n == n && n == 300000 &&
      musa_1d_c2c_root && request.input_dtype == "complex128";
  const bool musa_packed_fp32_c2r_child_root =
      request.device_type == "musa" && request.device_arch == "31" && request.raw_dim == 1 &&
      request.origin_rank == 1 && request.batch > 1 && request.packed_real_child &&
      request.requested_n == n && request.input_dtype == "complex64" &&
      (request.origin_real_transform_kind == "c2r" || request.real_transform_kind == "c2r");
  const bool musa_default_packed_300000_fp64_c2r_split =
      request.device_type == "musa" && request.device_arch == "31" && request.raw_dim == 1 &&
      request.origin_rank == 1 && request.batch >= 64 && request.packed_real_child &&
      request.requested_n == n && n == 300000 && request.input_dtype == "complex128" &&
      (request.origin_real_transform_kind == "c2r" || request.real_transform_kind == "c2r");
  const bool musa_default_packed_c2r_32768_split = musa_packed_fp32_c2r_child_root && n == 32768;
  const bool musa_default_packed_c2r_65536_split = musa_packed_fp32_c2r_child_root && n == 65536;
  const bool musa_default_350000_fp64_c2c_split =
      request.device_type == "musa" && request.device_arch == "31" && request.raw_dim == 1 &&
      request.origin_rank == 1 && request.batch > 1 && request.requested_n == n && n == 350000 &&
      musa_1d_c2c_root && request.input_dtype == "complex128";
  const bool musa_default_600000_fp64_c2c_split =
      request.device_type == "musa" && request.device_arch == "31" && request.raw_dim == 1 &&
      request.origin_rank == 1 && request.batch > 1 && request.requested_n == n && n == 600000 &&
      musa_1d_c2c_root && request.input_dtype == "complex128";
  const bool musa_root_fourstep_split =
      request.device_type == "musa" && request.device_arch == "31" && request.raw_dim == 1 &&
      request.origin_rank == 1 && request.batch > 1 && request.requested_n == n &&
      (musa_1d_c2c_root || musa_1d_real_root) &&
      ((musa_fourstep_split != nullptr && *musa_fourstep_split != '\0') || musa_default_131072_split ||
       musa_default_1024_split || musa_default_8192_split || musa_default_300000_fp64_c2c_split ||
       musa_default_350000_fp64_c2c_split || musa_default_600000_fp64_c2c_split);
  if (musa_root_fourstep_split || musa_packed_fp32_c2r_child_root ||
      musa_default_packed_300000_fp64_c2r_split) {
    // MTT S5000 screens favor 128x8 over the CT leaf at N=1024, 128x64 over
    // 64x128 at N=8192, 512x256 over 128x1024 at N=131072, and 256x128 / 256x256
    // for the packed FP32 C2R children at N=32768 / 65536. FP64 C2C N=300000
    // and the packed FP64 C2R child at N=300000 favor 750x400; the FP64 C2C
    // roots favor 700x500 at N=350000 and 750x800 at N=600000.
    std::string spec;
    if (musa_fourstep_split != nullptr && *musa_fourstep_split != '\0') {
      spec = musa_fourstep_split;
    } else if (musa_default_131072_split) {
      spec = "131072:512";
    } else if (musa_default_1024_split) {
      spec = "1024:128";
    } else if (musa_default_8192_split) {
      spec = "8192:128";
    } else if (musa_default_packed_c2r_32768_split) {
      spec = "32768:256";
    } else if (musa_default_packed_c2r_65536_split) {
      spec = "65536:256";
    } else if (musa_default_300000_fp64_c2c_split || musa_default_packed_300000_fp64_c2r_split) {
      spec = "300000:750";
    } else if (musa_default_350000_fp64_c2c_split) {
      spec = "350000:700";
    } else {
      spec = "600000:750";
    }
    const auto separator = spec.find(':');
    if (separator == std::string::npos) {
      throw std::runtime_error("FLAGFFT_MUSA_1D_FOURSTEP_SPLIT must be <length>:<n1>");
    }
    std::size_t parsed = 0;
    const int64_t target_length = std::stoll(spec.substr(0, separator), &parsed);
    if (parsed != separator) {
      throw std::runtime_error("FLAGFFT_MUSA_1D_FOURSTEP_SPLIT must be <length>:<n1>");
    }
    const std::string n1_text = spec.substr(separator + 1);
    const int64_t n1 = std::stoll(n1_text, &parsed);
    if (parsed != n1_text.size()) {
      throw std::runtime_error("FLAGFFT_MUSA_1D_FOURSTEP_SPLIT must be <length>:<n1>");
    }
    if (target_length == n) {
      if (n1 <= 1 || n1 >= n || n % n1 != 0) {
        throw std::runtime_error("FLAGFFT_MUSA_1D_FOURSTEP_SPLIT n1 must divide the requested length");
      }
      const int64_t n2 = n / n1;
      return std::make_shared<FourStepPlanNode>(n, n1, n2,
                                                build_auto_node(n1, false), build_auto_node(n2, false));
    }
  }
  // Opt in to an explicit HCU single C2R split while screening alternative
  // 1D four-step decompositions. Keep the override at the public root so it
  // cannot change child FFTs used by another algorithm.
  const bool hcu_single_c2r_root =
      request.device_type == "hcu" && request.raw_dim == 1 && request.origin_rank == 1 &&
      request.requested_n == n && request.real_transform &&
      (request.real_transform_kind == "c2r" || request.real_transform_kind == "z2d" ||
       request.origin_real_transform_kind == "c2r" ||
       request.origin_real_transform_kind == "z2d");
  const char *hcu_c2r_split = std::getenv("FLAGFFT_HCU_C2R_FOURSTEP_SPLIT");
  if (hcu_single_c2r_root && hcu_c2r_split != nullptr && *hcu_c2r_split != '\0') {
    const std::string spec(hcu_c2r_split);
    const auto separator = spec.find(':');
    if (separator == std::string::npos) {
      throw std::runtime_error("FLAGFFT_HCU_C2R_FOURSTEP_SPLIT must be <length>:<n1>");
    }
    std::size_t parsed = 0;
    const int64_t target_length = std::stoll(spec.substr(0, separator), &parsed);
    if (parsed != separator) {
      throw std::runtime_error("FLAGFFT_HCU_C2R_FOURSTEP_SPLIT must be <length>:<n1>");
    }
    const std::string n1_text = spec.substr(separator + 1);
    const int64_t n1 = std::stoll(n1_text, &parsed);
    if (parsed != n1_text.size()) {
      throw std::runtime_error("FLAGFFT_HCU_C2R_FOURSTEP_SPLIT must be <length>:<n1>");
    }
    if (target_length == n) {
      if (n1 <= 1 || n1 >= n || n % n1 != 0) {
        throw std::runtime_error("FLAGFFT_HCU_C2R_FOURSTEP_SPLIT n1 must divide the requested length");
      }
      const int64_t n2 = n / n1;
      return std::make_shared<FourStepPlanNode>(n, n1, n2,
                                                build_auto_node(n1, false), build_auto_node(n2, false));
    }
  }
  if (hcu_single_c2r_root && n == 340200) {
    // The measured 567x600 pair uses lane-richer leaves than the automatic
    // 504x675 pair and improves both HCU C2R precisions.
    constexpr int64_t n1 = 567;
    constexpr int64_t n2 = 600;
    return std::make_shared<FourStepPlanNode>(n, n1, n2,
                                              build_auto_node(n1, false), build_auto_node(n2, false));
  }
  // An opt-in split lets Ascend qualify the existing generic FourStep path
  // with Stockham/Bluestein children before changing its automatic policy.
  // Scope the override to the requested 1D root, never a convolution child.
  const char *npu_split = std::getenv("FLAGFFT_NPU_FOURSTEP_SPLIT");
  if (request.device_type == "npu" && request.origin_rank == 1 && npu_split != nullptr && *npu_split != '\0') {
    const std::string spec(npu_split);
    const auto separator = spec.find(':');
    if (separator == std::string::npos) {
      throw std::runtime_error("FLAGFFT_NPU_FOURSTEP_SPLIT must be <length>:<n1>");
    }
    std::size_t parsed = 0;
    const int64_t target_length = std::stoll(spec.substr(0, separator), &parsed);
    if (parsed != separator) {
      throw std::runtime_error("FLAGFFT_NPU_FOURSTEP_SPLIT must be <length>:<n1>");
    }
    const std::string n1_text = spec.substr(separator + 1);
    const int64_t n1 = std::stoll(n1_text, &parsed);
    if (parsed != n1_text.size()) {
      throw std::runtime_error("FLAGFFT_NPU_FOURSTEP_SPLIT must be <length>:<n1>");
    }
    if (target_length == n) {
      if (n1 <= 1 || n1 >= n || n % n1 != 0) {
        throw std::runtime_error("FLAGFFT_NPU_FOURSTEP_SPLIT must divide the requested length");
      }
      const int64_t n2 = n / n1;
      const char *leaf_mode = std::getenv("FLAGFFT_NPU_FOURSTEP_LEAF");
      if (leaf_mode != nullptr && std::string(leaf_mode) == "1") {
        const auto row_factors = select_leaf_factors(n1);
        const auto col_factors = select_leaf_factors(n2);
        if (!should_use_leaf(n1, row_factors) || !should_use_leaf(n2, col_factors)) {
          throw std::runtime_error("FLAGFFT_NPU_FOURSTEP_LEAF requires two supported leaf lengths");
        }
        return std::make_shared<FourStepPlanNode>(n, n1, n2,
                                                  make_leaf_plan(n1, row_factors),
                                                  make_leaf_plan(n2, col_factors));
      }
      return std::make_shared<FourStepPlanNode>(n, n1, n2,
                                                build_auto_node(n1, false), build_auto_node(n2, false));
    }
  }
  // Let 2D axis plans reuse the existing FourStep plan and fused-leaf
  // execution for long axes. Keep this opt-in so the current 2D policy stays
  // unchanged while splits are screened on Ascend.
  const char *npu_2d_split = std::getenv("FLAGFFT_NPU_2D_FOURSTEP_SPLIT");
  if (request.device_type == "npu" && request.origin_rank == 2 && request.raw_dim == 1 &&
      request.requested_n == n && request.input_dtype == "complex64" && request.output_dtype == "complex64" &&
      npu_2d_split != nullptr && *npu_2d_split != '\0') {
    const std::string spec(npu_2d_split);
    const auto separator = spec.find(':');
    if (separator == std::string::npos) {
      throw std::runtime_error("FLAGFFT_NPU_2D_FOURSTEP_SPLIT must be <length>:<n1>");
    }
    std::size_t parsed = 0;
    const int64_t target_length = std::stoll(spec.substr(0, separator), &parsed);
    if (parsed != separator) {
      throw std::runtime_error("FLAGFFT_NPU_2D_FOURSTEP_SPLIT must be <length>:<n1>");
    }
    const std::string n1_text = spec.substr(separator + 1);
    const int64_t n1 = std::stoll(n1_text, &parsed);
    if (parsed != n1_text.size()) {
      throw std::runtime_error("FLAGFFT_NPU_2D_FOURSTEP_SPLIT must be <length>:<n1>");
    }
    if (target_length == n) {
      if (n1 <= 1 || n1 >= n || n % n1 != 0) {
        throw std::runtime_error("FLAGFFT_NPU_2D_FOURSTEP_SPLIT must divide the requested length");
      }
      const int64_t n2 = n / n1;
      const char *leaf_mode = std::getenv("FLAGFFT_NPU_FOURSTEP_LEAF");
      if (leaf_mode != nullptr && std::string(leaf_mode) == "1") {
        const auto row_factors = select_leaf_factors(n1);
        const auto col_factors = select_leaf_factors(n2);
        if (!should_use_leaf(n1, row_factors) || !should_use_leaf(n2, col_factors)) {
          throw std::runtime_error("FLAGFFT_NPU_FOURSTEP_LEAF requires two supported leaf lengths");
        }
        return std::make_shared<FourStepPlanNode>(n,
                                                  n1,
                                                  n2,
                                                  make_leaf_plan(n1, row_factors),
                                                  make_leaf_plan(n2, col_factors));
      }
      return std::make_shared<FourStepPlanNode>(n,
                                                n1,
                                                n2,
                                                build_auto_node(n1, false),
                                                build_auto_node(n2, false));
    }
  }
  const char *npu_3d_aiv64 = std::getenv("FLAGFFT_NPU_3D_AIV64");
  const bool use_npu_3d_aiv64 = request.device_type == "npu" && request.origin_rank == 3 &&
                                request.raw_dim == 1 && request.input_dtype == "complex64" &&
                                request.output_dtype == "complex64" && npu_3d_aiv64 != nullptr &&
                                std::string(npu_3d_aiv64) == "1";
  if (use_npu_3d_aiv64 && n == 64) {
    const auto factors = select_leaf_factors(n);
    if (should_use_leaf(n, factors)) return make_leaf_plan(n, factors);
  }
  const char *npu_3d_split = std::getenv("FLAGFFT_NPU_3D_FOURSTEP_SPLIT");
  if (request.device_type == "npu" && request.origin_rank == 3 && request.raw_dim == 1 &&
      request.requested_n == n && request.input_dtype == "complex64" && request.output_dtype == "complex64" &&
      npu_3d_split != nullptr && *npu_3d_split != '\0') {
    const std::string spec(npu_3d_split);
    const auto separator = spec.find(':');
    if (separator == std::string::npos) {
      throw std::runtime_error("FLAGFFT_NPU_3D_FOURSTEP_SPLIT must be <length>:<n1>");
    }
    std::size_t parsed = 0;
    const int64_t target_length = std::stoll(spec.substr(0, separator), &parsed);
    if (parsed != separator) {
      throw std::runtime_error("FLAGFFT_NPU_3D_FOURSTEP_SPLIT must be <length>:<n1>");
    }
    const std::string n1_text = spec.substr(separator + 1);
    const int64_t n1 = std::stoll(n1_text, &parsed);
    if (parsed != n1_text.size()) {
      throw std::runtime_error("FLAGFFT_NPU_3D_FOURSTEP_SPLIT must be <length>:<n1>");
    }
    if (target_length == n) {
      if (n1 <= 1 || n1 >= n || n % n1 != 0) {
        throw std::runtime_error("FLAGFFT_NPU_3D_FOURSTEP_SPLIT must divide the requested length");
      }
      const int64_t n2 = n / n1;
      const char *leaf_mode = std::getenv("FLAGFFT_NPU_FOURSTEP_LEAF");
      auto build_3d_child = [&](int64_t length) -> PlanNodePtr {
        if (use_npu_3d_aiv64 && length == 64) {
          const auto factors = select_leaf_factors(length);
          if (should_use_leaf(length, factors)) return make_leaf_plan(length, factors);
        }
        if (leaf_mode != nullptr && std::string(leaf_mode) == "1") {
          const auto factors = select_leaf_factors(length);
          if (!should_use_leaf(length, factors)) {
            throw std::runtime_error("FLAGFFT_NPU_FOURSTEP_LEAF requires supported leaf children");
          }
          return make_leaf_plan(length, factors);
        }
        return build_auto_node(length, false);
      };
      return std::make_shared<FourStepPlanNode>(n, n1,
                                                n2, build_3d_child(n1), build_3d_child(n2));
    }
  }
  const bool npu_fourstep_operator = request.device_type == "npu" &&
                                     request.raw_dim == 1 && request.origin_rank == 1 &&
                                     request.requested_n == n &&
                                     (request.input_dtype == "complex64" ||
                                      request.output_dtype == "complex64");
  if (npu_fourstep_operator) {
    // Keep the existing 1D FourStep operator family on a FourStep plan root
    // for every transform type and batch mode. These splits also avoid making
    // the full root a Bluestein convolution for the largest awkward lengths.
    int64_t n1 = 0;
    switch (n) {
      case 16384: n1 = 128; break;
      case 46189: n1 = 209; break;
      case 185640: n1 = 420; break;
      case 340200: n1 = 567; break;
      case 524288: n1 = 512; break;
      case 663000: n1 = 663; break;
      default: break;
    }
    if (n1 != 0) {
      const int64_t n2 = n / n1;
      return std::make_shared<FourStepPlanNode>(n, n1, n2,
                                                build_auto_node(n1, false), build_auto_node(n2, false));
    }
  }
  const auto experiments = detail::maca_tail_plans(n, request, false);
  if (!experiments.empty()) {
    return detail::build_maca_tail_plan(n, experiments.front(),
                                        [&](int64_t length, bool heuristic) {
                                          return build_auto_node(length, heuristic);
                                        });
  }
  const char *maca_batch_setting = std::getenv("FLAGFFT_MACA_1D_BATCH");
  const bool maca_batch_c2c = request.device_type == "maca" && request.device_arch == "102" &&
                              request.raw_dim == 1 && request.origin_rank <= 1 && request.batch == 64 &&
                              n == request.requested_n && !request.real_transform &&
                              request.input_dtype == "complex64" && request.output_dtype == "complex64" &&
                              (maca_batch_setting == nullptr || std::string(maca_batch_setting) != "0");
  if (maca_batch_c2c && (n == 16384 || n == 663000)) {
    // The measured batch splits distribute the C550 column stage more evenly.
    const int64_t n1 = n == 16384 ? 256 : 884;
    const int64_t n2 = n / n1;
    return std::make_shared<FourStepPlanNode>(n, n1, n2,
                                              build_auto_node(n1, false), build_auto_node(n2, false));
  }
  return build_auto_node(n, true);
}

double PlanBuilder::cost_for(int64_t n, const FFTRequest &request) {
  set_request_context(request);
  return cost_for(n);
}

double PlanBuilder::cost_for(int64_t n) {
  auto it = cost_cache_.find(n);
  if (it != cost_cache_.end()) {
    return it->second;
  }
  const bool previous_heuristic = parallel_leaf_heuristic_enabled_;
  parallel_leaf_heuristic_enabled_ = false;
  std::vector<PlanCandidate> candidates;
  try {
    candidates = build_auto_candidates(n);
  } catch (...) {
    parallel_leaf_heuristic_enabled_ = previous_heuristic;
    throw;
  }
  parallel_leaf_heuristic_enabled_ = previous_heuristic;
  if (candidates.empty()) {
    throw std::runtime_error("length " + std::to_string(n) + " has no supported FFT implementation route");
  }
  auto best = select_candidate(candidates);
  cost_cache_[n] = best.cost;
  return best.cost;
}

}  // namespace flagfft
