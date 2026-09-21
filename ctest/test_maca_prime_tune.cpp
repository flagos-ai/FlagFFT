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

#include <gtest/gtest.h>

#include "flagfft/core.hpp"

#include <set>

// CPU-only planner target: supply the C550 capabilities recorded in the
// acceptance manifest, without linking or initializing a device runtime.
namespace flagfft::adaptor {
std::string backend_name() { return "maca"; }
int64_t max_dynamic_smem_bytes(int) { return 64 * 1024; }
}  // namespace flagfft::adaptor

TEST(Plan1D, MacaSinglePrimeTuneAddsPowerOfTwoConvolution) {
  flagfft::FFTRequest request;
  request.fft_length = request.requested_n = 997;
  request.input_dtype = request.output_dtype = "complex128";
  request.device_type = "maca";
  request.device_arch = "C550";
  request.direction = "forward";
  request.batch = 1;
  flagfft::PlanBuilder builder;
  const auto automatic = flagfft::PlanKey::from_node(builder.build(997, request)).repr();
  const auto plans = builder.build_decomposition_tune_candidates(997, request, 8);
  ASSERT_EQ(plans.size(), 3u);
  EXPECT_EQ(flagfft::PlanKey::from_node(plans.front().node).repr(), automatic);
  auto original = std::dynamic_pointer_cast<flagfft::BluesteinPlanNode>(plans[0].node);
  ASSERT_NE(original, nullptr);
  EXPECT_EQ(original->conv_length, 2000);
  EXPECT_NE(std::dynamic_pointer_cast<flagfft::RaderPlanNode>(plans[1].node), nullptr);
  auto power = std::dynamic_pointer_cast<flagfft::BluesteinPlanNode>(plans[2].node);
  ASSERT_NE(power, nullptr);
  EXPECT_EQ(power->conv_length, 2048);
  EXPECT_NE(std::dynamic_pointer_cast<flagfft::LeafPlanNode>(power->fft_plan), nullptr);
  for (int64_t limit : {1, 2, 3}) {
    const auto limited = builder.build_decomposition_tune_candidates(997, request, limit);
    ASSERT_EQ(limited.size(), static_cast<std::size_t>(limit));
    for (std::size_t i = 0; i < limited.size(); ++i) {
      EXPECT_EQ(flagfft::PlanKey::from_node(limited[i].node).repr(),
                flagfft::PlanKey::from_node(plans[i].node).repr());
    }
  }
  EXPECT_EQ(flagfft::PlanKey::from_node(builder.build(997, request)).repr(), automatic);
}

TEST(Plan1D, MacaSinglePrimeTuneAddsBalancedConvolutionSplit) {
  flagfft::FFTRequest request;
  request.fft_length = request.requested_n = 524287;
  request.input_dtype = request.output_dtype = "complex128";
  request.device_type = "maca";
  request.device_arch = "C550";
  request.direction = "forward";
  request.batch = 1;
  flagfft::PlanBuilder builder;
  const auto automatic = flagfft::PlanKey::from_node(builder.build(524287, request)).repr();
  const auto plans = builder.build_decomposition_tune_candidates(524287, request, 8);
  ASSERT_EQ(plans.size(), 2u);
  EXPECT_EQ(flagfft::PlanKey::from_node(plans.front().node).repr(), automatic);
  for (std::size_t i = 0; i < plans.size(); ++i) {
    auto bs = std::dynamic_pointer_cast<flagfft::BluesteinPlanNode>(plans[i].node);
    ASSERT_NE(bs, nullptr);
    EXPECT_EQ(bs->conv_length, 1048576);
    auto child = std::dynamic_pointer_cast<flagfft::FourStepPlanNode>(bs->fft_plan);
    ASSERT_NE(child, nullptr);
    EXPECT_EQ(child->n1, i == 0 ? 512 : 1024);
    EXPECT_EQ(child->n2, i == 0 ? 2048 : 1024);
    EXPECT_NE(std::dynamic_pointer_cast<flagfft::LeafPlanNode>(child->row_plan), nullptr);
    EXPECT_NE(std::dynamic_pointer_cast<flagfft::LeafPlanNode>(child->col_plan), nullptr);
  }
  EXPECT_EQ(builder.build_decomposition_tune_candidates(524287, request, 1).size(), 1u);
  EXPECT_EQ(flagfft::PlanKey::from_node(builder.build(524287, request)).repr(), automatic);
}

TEST(Plan1D, MacaPrimeTuneKeepsOtherTargetsAndBatchCandidates) {
  flagfft::FFTRequest request;
  request.input_dtype = request.output_dtype = "complex128";
  request.device_arch = "C550";
  request.direction = "forward";
  for (const auto& target : {std::pair {"maca", 2}, std::pair {"unit-test", 1}}) {
    request.device_type = target.first;
    request.batch = target.second;
    for (int64_t n : {997, 524287}) {
      request.fft_length = request.requested_n = n;
      flagfft::PlanBuilder builder;
      const auto automatic = flagfft::PlanKey::from_node(builder.build(n, request)).repr();
      const auto plans = builder.build_decomposition_tune_candidates(n, request, 8);
      ASSERT_EQ(plans.size(), n == 997 ? 2u : 1u);
      EXPECT_EQ(flagfft::PlanKey::from_node(plans.front().node).repr(), automatic);
    }
  }
}

TEST(Plan1D, MacaSinglePrimeTuneDeduplicatesAndBoundsLengths) {
  flagfft::FFTRequest request;
  request.input_dtype = request.output_dtype = "complex64";
  request.device_type = "maca";
  request.device_arch = "C550";
  request.direction = "forward";
  request.batch = 1;
  flagfft::PlanBuilder builder;
  for (int64_t n : {23, 997, 524287}) {
    request.fft_length = request.requested_n = n;
    const auto plans = builder.build_decomposition_tune_candidates(n, request, 8);
    EXPECT_EQ(plans.size(), n == 997 ? 2u : 1u);
    std::set<std::string> keys;
    for (const auto& candidate : plans) {
      EXPECT_TRUE(keys.insert(flagfft::PlanKey::from_node(candidate.node).repr()).second);
    }
  }
  EXPECT_THROW(builder.build_decomposition_tune_candidates(0, request, 8), std::runtime_error);
  EXPECT_THROW(builder.build_decomposition_tune_candidates(997, request, 0), std::runtime_error);
  EXPECT_THROW(builder.build_decomposition_tune_candidates((int64_t {1} << 61) + 1, request, 8),
               std::runtime_error);
}

