#include "flagfft/core.hpp"

namespace flagfft {

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
  return base + static_cast<double>(generic_stage_count * n);
}

double PlanBuilder::estimate_direct_dft_cost(int64_t n) {
  return static_cast<double>(n * n);
}

double PlanBuilder::four_step_cost(int64_t n1, int64_t n2) {
  return static_cast<double>(n2) * cost_for(n1) + static_cast<double>(n1) * cost_for(n2) +
         static_cast<double>(n1 * n2);
}

double PlanBuilder::bluestein_cost(int64_t n, int64_t conv_length) {
  return 3.0 * cost_for(conv_length) + static_cast<double>(n + conv_length);
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
  if (node->kind == PlanNodeKind::DirectDft) {
    return 4;
  }
  if (node->kind == PlanNodeKind::TwoDim) {
    return 5;
  }
  return 99;
}

}  // namespace flagfft
