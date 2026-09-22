// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "flagfft/core.hpp"
#include "rader_utils.hpp"

namespace flagfft::detail {

// Deliberately a closed experiment matrix, not another automatic planner.
struct MacaTailPlan {
  std::string name;
  int64_t first = 0;
  int64_t second = 0;
};

inline std::vector<MacaTailPlan> maca_tail_plans(int64_t n, const FFTRequest& request,
                                               bool tuning) {
  const char* value = std::getenv("FLAGFFT_MACA_TAIL_PLAN");
  if (value == nullptr || *value == '\0' || request.device_type != "maca" ||
      request.raw_dim != 1 || request.batch != 1 || n != request.requested_n) {
    return {};
  }
  const std::string setting(value);
  // These are experiment prerequisites: DB lookup precedes PlanBuilder in
  // the C API, and packed-real compilation can replace the full-length root.
  const char* disabled = std::getenv("FLAGFFT_TUNE_DISABLE");
  const char* packed = std::getenv("FLAGFFT_PACKED_REAL");
  if (disabled == nullptr || std::string(disabled) != "1" ||
      packed == nullptr || std::string(packed) != "0") {
    throw std::runtime_error("FLAGFFT_MACA_TAIL_PLAN requires FLAGFFT_TUNE_DISABLE=1 and "
                             "FLAGFFT_PACKED_REAL=0");
  }
  const bool fp32 = (request.input_dtype == "complex64" && request.output_dtype == "complex64") ||
                    (request.input_dtype == "float32" && request.output_dtype == "complex64") ||
                    (request.input_dtype == "complex64" && request.output_dtype == "float32");
  const bool fp64 = (request.input_dtype == "complex128" && request.output_dtype == "complex128") ||
                    (request.input_dtype == "float64" && request.output_dtype == "complex128") ||
                    (request.input_dtype == "complex128" && request.output_dtype == "float64");
  std::vector<MacaTailPlan> plans;
  if (n == 997 && fp64) {
    plans = {{"bs2000", 2000}, {"bs2048", 2048}};
  } else if (n == 1009 && fp32) {
    plans = {{"rader1008", 1008}, {"bs2048", 2048}};
  } else if (n == 1048576 && fp64) {
    plans = {{"ct512x2048", 512, 2048}, {"ct1024x1024", 1024, 1024}};
  } else if (n == 663000 && (fp32 || fp64)) {
    plans = {{"ct375x1768", 375, 1768}, {"ct650x1020", 650, 1020}, {"ct780x850", 780, 850}};
  } else if (n == 328050 && (fp32 || fp64)) {
    plans = {{"ct450x729", 450, 729}, {"ct486x675", 486, 675}, {"ct405x810", 405, 810}};
  }
  if (plans.empty()) {
    throw std::runtime_error("FLAGFFT_MACA_TAIL_PLAN: unsupported experiment length/dtype");
  }
  if (setting == "compare") {
    if (!tuning) {
      throw std::runtime_error("FLAGFFT_MACA_TAIL_PLAN=compare is only valid for tune");
    }
    return plans;
  }
  if (setting == "default") {
    return {{"default"}};
  }
  for (const auto& plan : plans) {
    if (plan.name == setting) {
      return {plan};
    }
  }
  throw std::runtime_error("FLAGFFT_MACA_TAIL_PLAN: invalid plan for this experiment: " + setting);
}

// Reuse the normal child planner (including device resource limits and radix
// selection). The callback never enters the opt-in root selector recursively.
template <typename BuildAuto>
PlanNodePtr build_maca_tail_plan(int64_t n, const MacaTailPlan& plan, BuildAuto build_auto) {
  if (plan.name == "default") {
    return build_auto(n, true);
  }
  if (plan.second != 0) {
    return std::make_shared<FourStepPlanNode>(n, plan.first, plan.second,
                                             build_auto(plan.first, false),
                                             build_auto(plan.second, false));
  }
  if (plan.name == "rader1008") {
    const int64_t root = find_primitive_root(n);
    return std::make_shared<RaderPlanNode>(n, root, build_rader_index_table(n, root),
                                          build_auto(n - 1, false));
  }
  return std::make_shared<BluesteinPlanNode>(n, plan.first, build_auto(plan.first, false));
}

}  // namespace flagfft::detail
