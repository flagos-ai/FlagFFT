// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "flagfft/core.hpp"
#include "flagfft/maca_tail_policy.hpp"
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
  const auto automatic = maca_tail_automatic_plan(request);
  if (n == request.requested_n && !automatic.empty()) {
    if (automatic == "bs2048") return {{automatic, 2048}};
    if (automatic == "bs16384") return {{automatic, 16384}};
    if (automatic == "bs32768") return {{automatic, 32768}};
    if (automatic == "bs1048576") return {{automatic, 1048576}};
    if (automatic == "ct1024x1024") return {{automatic, 1024, 1024}};
    if (automatic == "ct405x810") return {{automatic, 405, 810}};
    if (automatic == "ct340x975") return {{automatic, 340, 975}};
    if (automatic == "ct780x850") return {{automatic, 780, 850}};
    if (automatic == "ct375x1768") return {{automatic, 375, 1768}};
  }
  const bool c550_large_batch = request.device_arch == "102" && request.origin_rank <= 1 &&
      request.batch == 64 && n == 663000 && !request.real_transform &&
      (request.input_dtype == "complex64" || request.input_dtype == "complex128") &&
      request.input_dtype == request.output_dtype &&
      !request.input_strides.empty() && request.input_strides.back() == 1;
  if (value == nullptr || *value == '\0' || request.device_type != "maca" ||
      request.raw_dim != 1 || (request.batch != 1 && !c550_large_batch) ||
      n != request.requested_n) {
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
  } else if (n == 524287 && request.device_arch == "102" && request.origin_rank <= 1 &&
             request.batch == 1 && !request.real_transform &&
             request.input_dtype == "complex128" && request.output_dtype == "complex128") {
    plans = {{"bs1048576", 1048576}};
  } else if (n == 1048576 && fp64) {
    plans = {{"ct512x2048", 512, 2048}, {"ct1024x1024", 1024, 1024}};
  } else if (n == 663000 && (fp32 || fp64)) {
    plans = {{"ct375x1768", 375, 1768}, {"ct650x1020", 650, 1020}, {"ct780x850", 780, 850},
             {"ct680x975", 680, 975}, {"ct975x680", 975, 680}, {"ct884x750", 884, 750},
             {"ct750x884", 750, 884}, {"ct663x1000", 663, 1000}, {"ct1000x663", 1000, 663}};
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
  if (plan.name == "bs16384" || plan.name == "bs32768" || plan.name == "bs1048576") {
    const int64_t n1 = plan.first == 1048576 ? 1024 : 128;
    const int64_t n2 = plan.first / n1;
    auto conv = std::make_shared<FourStepPlanNode>(
        plan.first, n1, n2, build_auto(n1, false), build_auto(n2, false));
    return std::make_shared<BluesteinPlanNode>(n, plan.first, std::move(conv));
  }
  return std::make_shared<BluesteinPlanNode>(n, plan.first, build_auto(plan.first, false));
}

}  // namespace flagfft::detail
