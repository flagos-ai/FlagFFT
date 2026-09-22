// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "exec/c_api_internal.hpp"
#include "flagfft/core.hpp"

#include <cstdlib>
#include <set>

namespace {
using namespace flagfft;

class Env {
 public:
  explicit Env(const char* name) : name_(name) {
    if (const char* old = std::getenv(name)) old_ = old;
  }
  ~Env() {
    if (old_) setenv(name_, old_->c_str(), 1);
    else unsetenv(name_);
  }
  void set(const char* value) {
    if (value) setenv(name_, value, 1);
    else unsetenv(name_);
  }
 private:
  const char* name_;
  std::optional<std::string> old_;
};

class MacaTailPlans : public ::testing::Test {
 protected:
  Env setting{"FLAGFFT_MACA_TAIL_PLAN"};
  Env policy{"FLAGFFT_MACA_TAIL_POLICY"};
  Env db{"FLAGFFT_TUNE_DISABLE"};
  Env packed{"FLAGFFT_PACKED_REAL"};
  void SetUp() override {
    setting.set(nullptr);
    // These tests describe the tuner candidate matrix and the unmodified
    // planner default, so the automatic tail policy is switched off here.
    policy.set("0");
    db.set("1");
    packed.set("0");
  }
};

FFTRequest request_for(int64_t n, bool fp64) {
  FlagFFTPlanDesc desc;
  desc.rank = 1;
  desc.n = {n};
  desc.batch = 1;
  desc.idist = desc.odist = n;
  desc.precision = fp64 ? FlagFFTPrecision::Float64 : FlagFFTPrecision::Float32;
  desc.type = fp64 ? FLAGFFT_Z2Z : FLAGFFT_C2C;
  desc.device_arch = "C550";
  return request_from_desc(desc, "forward");
}

std::string key(const PlanNodePtr& node) { return PlanKey::from_node(node).repr(); }

std::string label(const PlanNodePtr& node) {
  if (auto bs = std::dynamic_pointer_cast<BluesteinPlanNode>(node))
    return "bs" + std::to_string(bs->conv_length);
  if (auto ct = std::dynamic_pointer_cast<FourStepPlanNode>(node))
    return "ct" + std::to_string(ct->n1) + "x" + std::to_string(ct->n2);
  if (auto rader = std::dynamic_pointer_cast<RaderPlanNode>(node))
    return "rader" + std::to_string(rader->conv_plan->length);
  return node->describe();
}

void expect_legal(const PlanNodePtr& node) {
  ASSERT_NE(node, nullptr);
  EXPECT_TRUE(raw_supported_node(node));
  if (auto leaf = std::dynamic_pointer_cast<LeafPlanNode>(node)) {
    EXPECT_EQ(product(leaf->factors), leaf->length);
    EXPECT_EQ(leaf->remainder, 1);
    EXPECT_GT(leaf->lanes, 0);
    EXPECT_GT(leaf->num_warps, 0);
    for (auto radix : leaf->factors) EXPECT_TRUE(contains(kSupportedRadices, radix));
  } else if (auto ct = std::dynamic_pointer_cast<FourStepPlanNode>(node)) {
    EXPECT_EQ(ct->n1 * ct->n2, ct->length);
    EXPECT_EQ(ct->row_plan->length, ct->n1);
    EXPECT_EQ(ct->col_plan->length, ct->n2);
    expect_legal(ct->row_plan);
    expect_legal(ct->col_plan);
  } else if (auto bs = std::dynamic_pointer_cast<BluesteinPlanNode>(node)) {
    EXPECT_GE(bs->conv_length, 2 * bs->length - 1);
    EXPECT_EQ(bs->fft_plan->length, bs->conv_length);
    expect_legal(bs->fft_plan);
  } else if (auto rader = std::dynamic_pointer_cast<RaderPlanNode>(node)) {
    EXPECT_EQ(rader->conv_plan->length, rader->length - 1);
    const std::set<int64_t> indices(rader->idx.begin(), rader->idx.end());
    EXPECT_EQ(indices.size(), static_cast<size_t>(rader->length - 1));
    EXPECT_EQ(*indices.begin(), 1);
    EXPECT_EQ(*indices.rbegin(), rader->length - 1);
    expect_legal(rader->conv_plan);
  } else {
    FAIL() << "Unexpected experiment node: " << node->describe();
  }
}

struct Experiment {
  int64_t n;
  bool fp64;
  std::vector<std::string> names;
};
const std::vector<Experiment> experiments = {
  {997, true, {"bs2000", "bs2048"}},
  {1009, false, {"rader1008", "bs2048"}},
  {1048576, true, {"ct512x2048", "ct1024x1024"}},
  {663000, false, {"ct375x1768", "ct650x1020", "ct780x850"}},
  {663000, true, {"ct375x1768", "ct650x1020", "ct780x850"}},
  {328050, false, {"ct450x729", "ct486x675", "ct405x810"}},
  {328050, true, {"ct450x729", "ct486x675", "ct405x810"}},
};

TEST_F(MacaTailPlans, AutomaticPolicyIsOnByDefaultForWhitelistedRoots) {
  for (const auto& experiment : experiments) {
    SCOPED_TRACE(std::to_string(experiment.n) + (experiment.fp64 ? " FP64" : " FP32"));
    auto request = request_for(experiment.n, experiment.fp64);
    PlanBuilder builder;
    policy.set("0");
    const auto planner_default = label(builder.build(experiment.n, request));
    EXPECT_EQ(planner_default, experiment.names.front());
    policy.set(nullptr);
    const auto automatic = label(builder.build(experiment.n, request));
    const bool whitelisted = (experiment.n == 997 && experiment.fp64) ||
                             (experiment.n == 1009 && !experiment.fp64) ||
                             (experiment.n == 1048576 && experiment.fp64);
    if (whitelisted) {
      EXPECT_EQ(automatic, experiment.names.back());
    } else {
      EXPECT_EQ(automatic, planner_default);
    }
  }
}

TEST_F(MacaTailPlans, ExactBoundedMatrixAndForcedRootsMatchTuner) {
  for (const auto& experiment : experiments) {
    SCOPED_TRACE(std::to_string(experiment.n) + (experiment.fp64 ? " FP64" : " FP32"));
    auto request = request_for(experiment.n, experiment.fp64);
    PlanBuilder builder;
    setting.set(nullptr);
    const auto automatic = builder.build(experiment.n, request);
    EXPECT_EQ(label(automatic), experiment.names.front());
    setting.set("compare");
    const auto candidates = builder.build_decomposition_tune_candidates(experiment.n, request, 100);
    ASSERT_EQ(candidates.size(), experiment.names.size());
    EXPECT_EQ(key(candidates.front().node), key(automatic));
    EXPECT_THROW(builder.build_decomposition_tune_candidates(experiment.n, request, candidates.size() - 1),
                 std::runtime_error);
    EXPECT_THROW(builder.build(experiment.n, request), std::runtime_error);
    std::set<std::string> seen;
    for (size_t i = 0; i < candidates.size(); ++i) {
      EXPECT_EQ(label(candidates[i].node), experiment.names[i]);
      EXPECT_TRUE(seen.insert(key(candidates[i].node)).second);
      expect_legal(candidates[i].node);
      setting.set(experiment.names[i].c_str());
      EXPECT_EQ(key(builder.build(experiment.n, request)), key(candidates[i].node));
      const auto single = builder.build_decomposition_tune_candidates(experiment.n, request, 1);
      ASSERT_EQ(single.size(), 1u);
      EXPECT_EQ(key(single[0].node), key(candidates[i].node));
    }
    setting.set("default");
    EXPECT_EQ(key(builder.build(experiment.n, request)), key(automatic));
    setting.set(nullptr);
    EXPECT_EQ(key(builder.build(experiment.n, request)), key(automatic));
    PlanBuilder fresh;
    EXPECT_EQ(key(fresh.build(experiment.n, request)), key(automatic));
  }
}

TEST_F(MacaTailPlans, InverseAndRealRequestsKeepFullLengthCandidates) {
  for (const auto& experiment : experiments) {
    auto request = request_for(experiment.n, experiment.fp64);
    const std::string real = experiment.fp64 ? "float64" : "float32";
    const std::string complex = request.input_dtype;
    PlanBuilder builder;
    for (const auto& name : experiment.names) {
      setting.set(name.c_str());
      const auto expected = key(builder.build(experiment.n, request));
      for (const auto* direction : {"forward", "inverse"}) {
        for (int mode : {0, 1, 2}) {
          auto variant = request;
          variant.direction = direction;
          variant.input_dtype = mode == 1 ? real : complex;
          variant.output_dtype = mode == 2 ? real : complex;
          EXPECT_EQ(key(builder.build(experiment.n, variant)), expected);
          auto tuned = builder.build_decomposition_tune_candidates(experiment.n, variant, 1);
          ASSERT_EQ(tuned.size(), 1u);
          EXPECT_EQ(key(tuned.front().node), expected);
          EXPECT_EQ(tuned.front().node->length, experiment.n);
        }
      }
    }
  }
}

TEST_F(MacaTailPlans, BalancedFp64ChildrenQualifyForSeparateRegisterPackTrial) {
  auto request = request_for(1048576, true);
  PlanBuilder builder;
  setting.set("ct1024x1024");
  auto balanced = std::dynamic_pointer_cast<FourStepPlanNode>(builder.build(1048576, request));
  ASSERT_NE(balanced, nullptr);
  // The codegen opt-in is maintained separately. Verify the actual forced
  // children satisfy its 512/1024 power-of-two, unpadded-leaf prerequisites.
  for (const auto& node : {balanced->row_plan, balanced->col_plan}) {
    auto leaf = std::dynamic_pointer_cast<LeafPlanNode>(node);
    ASSERT_NE(leaf, nullptr);
    EXPECT_EQ(leaf->length, 1024);
    EXPECT_EQ(leaf->smem_size, leaf->length);
    EXPECT_GT(leaf->factors.size(), 1u);
    for (int64_t radix : leaf->factors) EXPECT_EQ(radix & (radix - 1), 0);
  }
  setting.set("ct512x2048");
  auto baseline = std::dynamic_pointer_cast<FourStepPlanNode>(builder.build(1048576, request));
  ASSERT_NE(baseline, nullptr);
  EXPECT_EQ(baseline->row_plan->length, 512);
  EXPECT_EQ(baseline->col_plan->length, 2048);
}

TEST_F(MacaTailPlans, OtherBackendsBatchesRanksAndChildBuildsRemainUnchanged) {
  for (const auto* backend : {"cuda", "musa", "hcu", "ppu", "ix", "npu", "maca"}) {
    for (int mode : {0, 1, 2, 3}) {
      if (std::string(backend) == "maca" && mode == 0) continue;
      auto request = request_for(997, true);
      request.device_type = backend;
      if (mode == 1) request.batch = 2;
      if (mode == 2) request.raw_dim = 2;
      if (mode == 3) request.raw_dim = 3;
      PlanBuilder builder;
      setting.set(nullptr);
      const auto automatic = key(builder.build(997, request));
      const auto candidates = builder.build_decomposition_tune_candidates(997, request, 3);
      // Invalid settings and missing prerequisites are ignored outside scope.
      setting.set("not-a-plan");
      db.set(nullptr);
      packed.set(nullptr);
      EXPECT_EQ(key(builder.build(997, request)), automatic);
      const auto unchanged = builder.build_decomposition_tune_candidates(997, request, 3);
      ASSERT_EQ(unchanged.size(), candidates.size());
      for (size_t i = 0; i < candidates.size(); ++i)
        EXPECT_EQ(key(unchanged[i].node), key(candidates[i].node));
    }
  }
  auto request = request_for(997, true);
  PlanBuilder builder;
  setting.set(nullptr);
  const auto child = key(builder.build(2048, request));
  setting.set("bs2000");
  EXPECT_EQ(key(builder.build(2048, request)), child);
}

TEST_F(MacaTailPlans, RejectsUnsupportedOrAmbiguousExperiments) {
  auto request = request_for(997, true);
  PlanBuilder builder;
  for (const char* invalid : {"1", "BS2048", "bs2000 ", "bs1992", "rader1008", "ct1x997"}) {
    setting.set(invalid);
    EXPECT_THROW(builder.build(997, request), std::runtime_error);
    EXPECT_THROW(builder.build_decomposition_tune_candidates(997, request, 3), std::runtime_error);
  }
  setting.set("bs2048");
  db.set(nullptr);
  EXPECT_THROW(builder.build(997, request), std::runtime_error);
  db.set("1");
  packed.set("1");
  EXPECT_THROW(builder.build(997, request), std::runtime_error);
  packed.set("0");
  for (const auto& unsupported : {request_for(997, false), request_for(1009, true),
                                   request_for(1048576, false), request_for(1024, true)}) {
    EXPECT_THROW(builder.build(unsupported.requested_n, unsupported), std::runtime_error);
  }
  request.output_dtype = "complex64";
  EXPECT_THROW(builder.build(997, request), std::runtime_error);
  EXPECT_THROW(builder.build(0, request), std::runtime_error);
  EXPECT_THROW(builder.build_decomposition_tune_candidates(997, request, 0), std::runtime_error);
}
}  // namespace
