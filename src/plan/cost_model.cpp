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

#include <string_view>

namespace flagfft {
namespace {

  constexpr int64_t kLargeBatchFourStepMinBatch = 65;
  constexpr int64_t kSmallBatchFourStepMaxBatch = 8;
  constexpr double kMeasuredFourStepPenaltyScale = 100.0;
  constexpr std::string_view kMeasuredFourStepDeviceArch = "sm_80";

  int64_t measured_small_batch_four_step_n1(int64_t n, int64_t batch, const std::string &device_arch) {
    if (device_arch != kMeasuredFourStepDeviceArch) {
      return 0;
    }
    if (n == 8192 && batch > 1) {
      return 128;
    }
    if (n == 16384) {
      return 256;
    }
    if (n == (int64_t {1} << 20)) {
      return 1024;
    }
    return 0;
  }

  int64_t measured_large_batch_four_step_n1(int64_t n,
                                            const std::string &dtype,
                                            const std::string &device_arch) {
    if (device_arch != kMeasuredFourStepDeviceArch) {
      return 0;
    }
    const bool is_double = dtype == "complex128" || dtype == "float64";
    if (n == 8192) {
      return is_double ? 256 : 128;
    }
    if (n == 16384) {
      return is_double ? 512 : 256;
    }
    if (n == (int64_t {1} << 20)) {
      return 1024;
    }
    return 0;
  }

}  // namespace

double PlanBuilder::estimate_leaf_warm_cost(int64_t n) {
  auto factors = select_leaf_factors(n);
  return estimate_leaf_warm_cost(n, factors);
}

double PlanBuilder::estimate_leaf_warm_cost(int64_t n, const std::vector<int64_t> &factors) {
  int64_t lanes = choose_lanes(n, factors);
  int64_t warps = choose_num_warps(lanes);
  int64_t generic_stage_count = 0;
  for (int64_t radix : factors) {
    if (!contains(kSpecializedButterflyRadices, radix) &&
        !contains(kSpecializedDirectCodeletRadices, radix)) {
      ++generic_stage_count;
    }
  }
  double base =
      static_cast<double>(n * static_cast<int64_t>(factors.size()) * warps) / static_cast<double>(lanes);
  const RequestContext &context = request_context();
  const bool is_double = context.input_dtype == "complex128" || context.input_dtype == "float64";
  const int64_t real_bytes = is_double ? 8 : 4;
  const int64_t shared_bytes = 4 * ceil_power_of_two(n) * real_bytes;
  constexpr int64_t kFullResidencySharedBytes = 64 * 1024;
  const double shared_pressure =
      std::max(1.0, static_cast<double>(shared_bytes) / static_cast<double>(kFullResidencySharedBytes));
  return (base + static_cast<double>(generic_stage_count * n)) * shared_pressure;
}

double PlanBuilder::estimate_direct_dft_cost(int64_t n) {
  return static_cast<double>(n * n);
}

double PlanBuilder::four_step_cost(int64_t n1, int64_t n2) {
  const int64_t n = n1 * n2;
  double cost = static_cast<double>(n2) * cost_for(n1) + static_cast<double>(n1) * cost_for(n2) +
                static_cast<double>(n);
  const RequestContext &context = request_context();
  int64_t preferred_n1 = 0;
  if (context.batch <= kSmallBatchFourStepMaxBatch) {
    preferred_n1 = measured_small_batch_four_step_n1(n, context.batch, context.device_arch);
  } else if (context.batch >= kLargeBatchFourStepMinBatch) {
    preferred_n1 = measured_large_batch_four_step_n1(n, context.input_dtype, context.device_arch);
  }
  if (preferred_n1 > 0 && n1 != preferred_n1) {
    cost += static_cast<double>(n) * kMeasuredFourStepPenaltyScale;
  }
  return cost;
}

double PlanBuilder::bluestein_cost(int64_t n, int64_t conv_length) {
  return 3.0 * cost_for(conv_length) + static_cast<double>(n + conv_length);
}

double PlanBuilder::rader_cost(int64_t n) {
  int64_t conv_length = n - 1;
  return 3.0 * cost_for(conv_length) + static_cast<double>(n + 3 * conv_length);
}

int PlanBuilder::priority(const PlanNodePtr &node) {
  if (node->kind == PlanNodeKind::CtLeaf) {
    return 0;
  }
  if (node->kind == PlanNodeKind::FourStep) {
    return 1;
  }
  if (node->kind == PlanNodeKind::StockhamAutosort) {
    return 2;
  }
  if (node->kind == PlanNodeKind::Bluestein) {
    return 3;
  }
  if (node->kind == PlanNodeKind::Rader) {
    return 3;
  }
  if (node->kind == PlanNodeKind::DirectDft) {
    return 4;
  }
  if (node->kind == PlanNodeKind::TwoDim) {
    return 5;
  }
  return 99;
}

}  // namespace flagfft
