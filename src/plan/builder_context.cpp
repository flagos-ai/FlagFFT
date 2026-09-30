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
         device_type == other.device_type && device_index == other.device_index &&
         device_arch == other.device_arch && origin_rank == other.origin_rank &&
         requested_n == other.requested_n &&
         batch == other.batch && ix_short_single == other.ix_short_single &&
         ix_ct_batch == other.ix_ct_batch &&
         max_dynamic_smem_bytes == other.max_dynamic_smem_bytes;
}

PlanBuilder::RequestContext PlanBuilder::make_request_context(const FFTRequest &request) const {
  RequestContext context;
  context.input_dtype = request.input_dtype;
  context.output_dtype = request.output_dtype;
  context.real_transform_kind = request.real_transform_kind;
  context.device_type = request.device_type;
  context.device_index = request.device_index;
  context.device_arch = request.device_arch;
  context.origin_rank = request.origin_rank;
  context.requested_n = request.requested_n;
  context.batch = request.batch;
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
  // Screen a single CT leaf for each large 2D axis before using it as the
  // default. This keeps the TwoDimPlanNode and only changes its 1D children.
  const char *npu_2d_leaf = std::getenv("FLAGFFT_NPU_2D_AXIS_LEAF");
  if (request.device_type == "npu" && request.origin_rank == 2 && request.raw_dim == 1 &&
      request.requested_n == n && request.input_dtype == "complex64" && request.output_dtype == "complex64" &&
      npu_2d_leaf != nullptr && *npu_2d_leaf != '\0') {
    std::size_t parsed = 0;
    int64_t target_length = 0;
    try {
      target_length = std::stoll(npu_2d_leaf, &parsed);
    } catch (const std::exception &) {
      throw std::runtime_error("FLAGFFT_NPU_2D_AXIS_LEAF must be a positive transform length");
    }
    if (parsed != std::string(npu_2d_leaf).size() || target_length <= 0) {
      throw std::runtime_error("FLAGFFT_NPU_2D_AXIS_LEAF must be a positive transform length");
    }
    if (target_length == n) {
      const auto factors = select_leaf_factors(n);
      if (!should_use_leaf(n, factors)) {
        throw std::runtime_error("FLAGFFT_NPU_2D_AXIS_LEAF length does not fit a supported CT leaf");
      }
      return make_leaf_plan(n, factors);
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
