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
  const bool measured_npu_single_c2c =
      request.device_type == "npu" && request.device_arch == "102" &&
      request.raw_dim == 1 && request.origin_rank == 1 && request.requested_n == n &&
      request.batch == 1 && !request.real_transform &&
      request.input_dtype == "complex64" && request.output_dtype == "complex64";
  if (measured_npu_single_c2c && n == 185640) {
    // Split the 524288-point root Bluestein convolution into a smooth 420
    // Stockham child and a 442-point Bluestein child (1024 convolution).
    return std::make_shared<FourStepPlanNode>(n, 420, 442,
                                              build_auto_node(420, false), build_auto_node(442, false));
  }
  if (measured_npu_single_c2c && n == 663000) {
    // Replace the 2097152-point root Bluestein convolution with a 663-point
    // Bluestein child (2048 convolution) and a smooth 1000-point child.
    return std::make_shared<FourStepPlanNode>(n, 663, 1000,
                                              build_auto_node(663, false), build_auto_node(1000, false));
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
