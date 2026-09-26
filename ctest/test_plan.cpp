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
#include "flagfft_test.h"

#include <cstdlib>
#include <cstring>
#include <set>
#include <string>

namespace {

void ExpectPlanContains(flagfftHandle plan, const std::string& expected) {
  const char* raw_desc = flagfftGetPlanDescription(plan);
  ASSERT_NE(raw_desc, nullptr);
  std::string desc(raw_desc);
  EXPECT_NE(desc.find(expected), std::string::npos);
}

}  // namespace

// =========================================================================
// 1D plan tests
// =========================================================================

TEST(Plan1D, IxCtSinglePolicyScope) {
  const char* original = std::getenv("FLAGFFT_IX_CT_SINGLE");
  const std::optional<std::string> saved = original ? std::optional<std::string>(original) : std::nullopt;
  struct Restore {
    std::optional<std::string> value;
    ~Restore() {
      if (value) setenv("FLAGFFT_IX_CT_SINGLE", value->c_str(), 1);
      else unsetenv("FLAGFFT_IX_CT_SINGLE");
    }
  } restore{saved};
  unsetenv("FLAGFFT_IX_CT_SINGLE");
  flagfft::FFTRequest request;
  request.device_type = "ix";
  request.device_arch = "71";
  request.raw_dim = 1;
  request.batch = 1;
  request.fft_length = request.requested_n = 2048;
  request.input_dtype = request.output_dtype = "complex64";
  request.input_strides = {2048, 1};
  EXPECT_TRUE(flagfft::ix_ct_single_policy_enabled(request));
  for (int64_t n : {1024, 2048, 16384}) {
    auto small = request;
    small.fft_length = small.requested_n = n;
    EXPECT_TRUE(flagfft::ix_ct_single_policy_enabled(small));
    EXPECT_EQ(flagfft::ix_ct_single_tle_policy(small), 0);
  }
  auto changed = request;
  changed.batch = 2;
  EXPECT_FALSE(flagfft::ix_ct_single_policy_enabled(changed));
  changed = request;
  changed.raw_dim = 2;
  EXPECT_FALSE(flagfft::ix_ct_single_policy_enabled(changed));
  changed = request;
  changed.fft_length = changed.requested_n = 1048576;
  EXPECT_FALSE(flagfft::ix_ct_single_policy_enabled(changed));
  changed = request;
  changed.input_dtype = changed.output_dtype = "complex128";
  EXPECT_FALSE(flagfft::ix_ct_single_policy_enabled(changed));
  changed = request;
  changed.device_arch = "other";
  EXPECT_FALSE(flagfft::ix_ct_single_policy_enabled(changed));
  changed = request;
  changed.input_strides.back() = 2;
  EXPECT_FALSE(flagfft::ix_ct_single_policy_enabled(changed));
  auto packed = request;
  EXPECT_FALSE(flagfft::ix_packed_real_policy_enabled(packed));
  for (int64_t n : {328050, 340200, 663000, 1048576}) {
    packed.fft_length = packed.requested_n = n;
    EXPECT_TRUE(flagfft::ix_packed_real_policy_enabled(packed));
    EXPECT_EQ(flagfft::ix_ct_single_tle_policy(packed), n == 1048576 ? 2 : (n == 663000 ? 0 : 1));
  }
  auto packed_changed = packed;
  packed_changed.batch = 2;
  EXPECT_FALSE(flagfft::ix_packed_real_policy_enabled(packed_changed));
  EXPECT_EQ(flagfft::ix_ct_single_tle_policy(packed_changed), 0);
  packed_changed = packed;
  packed_changed.raw_dim = 2;
  EXPECT_FALSE(flagfft::ix_packed_real_policy_enabled(packed_changed));
  packed_changed = packed;
  packed_changed.input_dtype = packed_changed.output_dtype = "complex128";
  EXPECT_FALSE(flagfft::ix_packed_real_policy_enabled(packed_changed));
  packed_changed = packed;
  packed_changed.device_arch = "other";
  EXPECT_FALSE(flagfft::ix_packed_real_policy_enabled(packed_changed));
  packed_changed = packed;
  packed_changed.input_strides.back() = 2;
  EXPECT_FALSE(flagfft::ix_packed_real_policy_enabled(packed_changed));
  setenv("FLAGFFT_IX_CT_SINGLE", "0", 1);
  EXPECT_FALSE(flagfft::ix_ct_single_policy_enabled(request));
  EXPECT_FALSE(flagfft::ix_packed_real_policy_enabled(packed));
  EXPECT_EQ(flagfft::ix_ct_single_tle_policy(packed), 0);
  setenv("FLAGFFT_IX_CT_SINGLE", "1", 1);
  EXPECT_TRUE(flagfft::ix_ct_single_policy_enabled(request));
  flagfft::PlanBuilder builder;
  auto small_request = request;
  small_request.n = small_request.fft_length = small_request.requested_n = 1024;
  auto optimized = std::dynamic_pointer_cast<flagfft::LeafPlanNode>(builder.build(1024, small_request));
  ASSERT_NE(optimized, nullptr);
  EXPECT_EQ(optimized->factors, (std::vector<int64_t>{16, 8, 8}));
  setenv("FLAGFFT_IX_CT_SINGLE", "0", 1);
  auto baseline = std::dynamic_pointer_cast<flagfft::LeafPlanNode>(builder.build(1024, small_request));
  ASSERT_NE(baseline, nullptr);
  EXPECT_EQ(baseline->factors, (std::vector<int64_t>{8, 8, 4, 4}));
  setenv("FLAGFFT_IX_CT_SINGLE", "1", 1);
  optimized = std::dynamic_pointer_cast<flagfft::LeafPlanNode>(builder.build(1024, small_request));
  ASSERT_NE(optimized, nullptr);
  EXPECT_EQ(optimized->factors, (std::vector<int64_t>{16, 8, 8}));
  setenv("FLAGFFT_IX_CT_SINGLE", "invalid", 1);
  EXPECT_THROW(flagfft::ix_ct_single_policy_enabled(request), std::runtime_error);
  EXPECT_THROW(flagfft::ix_packed_real_policy_enabled(packed), std::runtime_error);
  EXPECT_THROW(flagfft::ix_ct_single_tle_policy(packed), std::runtime_error);
}

TEST(Plan1D, IxCtBatchPolicyScope) {
  const char* original = std::getenv("FLAGFFT_IX_CT_BATCH");
  const std::optional<std::string> saved = original ? std::optional<std::string>(original) : std::nullopt;
  struct Restore {
    std::optional<std::string> value;
    ~Restore() {
      if (value) setenv("FLAGFFT_IX_CT_BATCH", value->c_str(), 1);
      else unsetenv("FLAGFFT_IX_CT_BATCH");
    }
  } restore{saved};
  unsetenv("FLAGFFT_IX_CT_BATCH");

  flagfft::FFTRequest request;
  request.device_type = "ix";
  request.device_arch = "71";
  request.raw_dim = 1;
  request.batch = 64;
  request.n = request.fft_length = request.requested_n = 1024;
  request.input_dtype = request.output_dtype = "complex64";
  request.input_strides = {1024, 1};
  EXPECT_TRUE(flagfft::ix_ct_batch_policy_enabled(request));
  auto longer = request;
  longer.n = longer.fft_length = longer.requested_n = 2048;
  EXPECT_TRUE(flagfft::ix_ct_batch_policy_enabled(longer));
  for (auto changed : {16, 210, 4096}) {
    auto other = request;
    other.n = other.fft_length = other.requested_n = changed;
    EXPECT_FALSE(flagfft::ix_ct_batch_policy_enabled(other));
  }
  auto changed = request;
  changed.batch = 1;
  EXPECT_FALSE(flagfft::ix_ct_batch_policy_enabled(changed));
  changed = request;
  changed.raw_dim = 2;
  EXPECT_FALSE(flagfft::ix_ct_batch_policy_enabled(changed));
  changed = request;
  changed.output_dtype = "complex128";
  EXPECT_FALSE(flagfft::ix_ct_batch_policy_enabled(changed));
  changed = request;
  changed.device_arch = "other";
  EXPECT_FALSE(flagfft::ix_ct_batch_policy_enabled(changed));
  changed = request;
  changed.input_strides.back() = 2;
  EXPECT_FALSE(flagfft::ix_ct_batch_policy_enabled(changed));

  flagfft::PlanBuilder builder;
  auto optimized = std::dynamic_pointer_cast<flagfft::LeafPlanNode>(builder.build(1024, request));
  ASSERT_NE(optimized, nullptr);
  EXPECT_EQ(optimized->factors, (std::vector<int64_t>{8, 8, 4, 4}));
  setenv("FLAGFFT_IX_CT_BATCH", "0", 1);
  EXPECT_FALSE(flagfft::ix_ct_batch_policy_enabled(request));
  auto baseline = std::dynamic_pointer_cast<flagfft::LeafPlanNode>(builder.build(1024, request));
  ASSERT_NE(baseline, nullptr);
  EXPECT_EQ(baseline->factors, (std::vector<int64_t>{32, 32}));
  setenv("FLAGFFT_IX_CT_BATCH", "invalid", 1);
  EXPECT_THROW(flagfft::ix_ct_batch_policy_enabled(request), std::runtime_error);
}

TEST(Plan1D, CreateDestroyAllTypes) {
  flagfftType types[] = {FLAGFFT_C2C, FLAGFFT_Z2Z, FLAGFFT_R2C, FLAGFFT_D2Z, FLAGFFT_C2R, FLAGFFT_Z2D};
  for (auto type : types) {
    flagfftHandle plan = nullptr;
    EXPECT_EQ(flagfftPlan1d(&plan, 256, type, 1), FLAGFFT_SUCCESS);
    EXPECT_NE(plan, nullptr);
    EXPECT_EQ(flagfftDestroy(plan), FLAGFFT_SUCCESS);
  }
}

TEST(Plan1D, WithBatch) {
  flagfftHandle plan = nullptr;
  EXPECT_EQ(flagfftPlan1d(&plan, 128, FLAGFFT_C2C, 4), FLAGFFT_SUCCESS);
  EXPECT_NE(plan, nullptr);
  EXPECT_EQ(flagfftDestroy(plan), FLAGFFT_SUCCESS);
}

TEST(Plan1D, InvalidParameters) {
  flagfftHandle plan = nullptr;
  // Zero size
  flagfftResult r = flagfftPlan1d(&plan, 0, FLAGFFT_C2C, 1);
  EXPECT_EQ(r, FLAGFFT_INVALID_SIZE);
  EXPECT_EQ(plan, nullptr);

  // Null plan pointer
  r = flagfftPlan1d(nullptr, 256, FLAGFFT_C2C, 1);
  EXPECT_EQ(r, FLAGFFT_INVALID_VALUE);
}

TEST(Plan1D, GetDescription) {
  flagfftHandle plan = nullptr;
  ASSERT_EQ(flagfftPlan1d(&plan, 256, FLAGFFT_C2C, 1), FLAGFFT_SUCCESS);
  const char* desc = flagfftGetPlanDescription(plan);
  EXPECT_NE(desc, nullptr);
  EXPECT_GT(std::strlen(desc), 0u);
  EXPECT_EQ(flagfftDestroy(plan), FLAGFFT_SUCCESS);
}

TEST(Plan1D, PrimeLengthUsesRader) {
  flagfftHandle plan = nullptr;
  ASSERT_EQ(flagfftPlan1d(&plan, 67, FLAGFFT_C2C, 1), FLAGFFT_SUCCESS);
  const char* desc = flagfftGetPlanDescription(plan);
  ASSERT_NE(desc, nullptr);
  EXPECT_NE(std::strstr(desc, "DirectDFT"), nullptr) << desc;
  EXPECT_EQ(flagfftDestroy(plan), FLAGFFT_SUCCESS);
}

TEST(Plan1D, PrimeLengthRaderSupportsRealWrappers) {
  flagfftHandle plan = nullptr;
  EXPECT_EQ(flagfftPlan1d(&plan, 67, FLAGFFT_R2C, 1), FLAGFFT_SUCCESS);
  EXPECT_NE(plan, nullptr);
  EXPECT_EQ(flagfftDestroy(plan), FLAGFFT_SUCCESS);
}

TEST(Plan1D, LargeBatchFourStepUsesMeasuredSplit) {
  if (flagfft::adaptor::backend_name() != "cuda") GTEST_SKIP() << "Measured CUDA split";
  setenv("FLAGFFT_TUNE_DISABLE", "1", 1);

  flagfftHandle plan8192 = nullptr;
  ASSERT_EQ(flagfftPlan1d(&plan8192, 8192, FLAGFFT_C2C, 256), FLAGFFT_SUCCESS);
  ExpectPlanContains(plan8192, "FourStep(n=8192, n1=128, n2=64)");
  EXPECT_EQ(flagfftDestroy(plan8192), FLAGFFT_SUCCESS);

  flagfftHandle plan8192z = nullptr;
  ASSERT_EQ(flagfftPlan1d(&plan8192z, 8192, FLAGFFT_Z2Z, 256), FLAGFFT_SUCCESS);
  ExpectPlanContains(plan8192z, "FourStep(n=8192, n1=256, n2=32)");
  EXPECT_EQ(flagfftDestroy(plan8192z), FLAGFFT_SUCCESS);

  flagfftHandle plan16384 = nullptr;
  ASSERT_EQ(flagfftPlan1d(&plan16384, 16384, FLAGFFT_C2C, 256), FLAGFFT_SUCCESS);
  ExpectPlanContains(plan16384, "FourStep(n=16384, n1=256, n2=64)");
  EXPECT_EQ(flagfftDestroy(plan16384), FLAGFFT_SUCCESS);

  flagfftHandle plan16384z = nullptr;
  ASSERT_EQ(flagfftPlan1d(&plan16384z, 16384, FLAGFFT_Z2Z, 256), FLAGFFT_SUCCESS);
  ExpectPlanContains(plan16384z, "FourStep(n=16384, n1=512, n2=32)");
  EXPECT_EQ(flagfftDestroy(plan16384z), FLAGFFT_SUCCESS);
}

TEST(Plan1D, SmallBatch16384UsesMeasuredSplit) {
  if (flagfft::adaptor::backend_name() != "cuda") GTEST_SKIP() << "Measured CUDA split";
  setenv("FLAGFFT_TUNE_DISABLE", "1", 1);

  flagfftHandle plan16384 = nullptr;
  ASSERT_EQ(flagfftPlan1d(&plan16384, 16384, FLAGFFT_C2C, 4), FLAGFFT_SUCCESS);
  ExpectPlanContains(plan16384, "FourStep(n=16384, n1=256, n2=64)");
  EXPECT_EQ(flagfftDestroy(plan16384), FLAGFFT_SUCCESS);

  flagfftHandle plan16384z = nullptr;
  ASSERT_EQ(flagfftPlan1d(&plan16384z, 16384, FLAGFFT_Z2Z, 4), FLAGFFT_SUCCESS);
  ExpectPlanContains(plan16384z, "FourStep(n=16384, n1=256, n2=64)");
  EXPECT_EQ(flagfftDestroy(plan16384z), FLAGFFT_SUCCESS);
}

TEST(Plan1D, Size2P20UsesTleOptimizedSplit) {
  if (flagfft::adaptor::backend_name() != "cuda") GTEST_SKIP() << "CUDA thread-local leaf layout";
  setenv("FLAGFFT_TUNE_DISABLE", "1", 1);

  flagfftHandle plan = nullptr;
  ASSERT_EQ(flagfftPlan1d(&plan, 1 << 20, FLAGFFT_C2C, 1), FLAGFFT_SUCCESS);
  ExpectPlanContains(plan, "FourStep(n=1048576, n1=1024, n2=1024)");
  ExpectPlanContains(plan, "LeafPlan(n=1024, factors=[32,32], lanes=32, num_warps=2");
  ExpectPlanContains(plan, "CompiledRawFourStepFused(n=1048576");
  EXPECT_EQ(flagfftDestroy(plan), FLAGFFT_SUCCESS);
}

TEST(Plan1D, LargeMixedRadicesUseThreadLocalRectangularLeaves) {
  if (flagfft::adaptor::backend_name() != "cuda") GTEST_SKIP() << "CUDA thread-local leaf layout";
  setenv("FLAGFFT_TUNE_DISABLE", "1", 1);

  struct MixedCase {
    int64_t length;
    int64_t n1;
    int64_t register_radix;
  };
  const MixedCase cases[] = {
      { 9 * (int64_t {1} << 16), 576, 18},
      { 3 * (int64_t {1} << 18), 768, 24},
      { 5 * (int64_t {1} << 17), 640, 20},
      {25 * (int64_t {1} << 15), 800, 25},
      {27 * (int64_t {1} << 15), 864, 27},
      { 7 * (int64_t {1} << 17), 896, 28},
      {15 * (int64_t {1} << 16), 960, 30},
  };

  for (const MixedCase& test_case : cases) {
    flagfftHandle plan = nullptr;
    ASSERT_EQ(flagfftPlan1d(&plan, test_case.length, FLAGFFT_C2C, 1), FLAGFFT_SUCCESS);
    ExpectPlanContains(plan,
                       "FourStep(n=" + std::to_string(test_case.length) +
                           ", n1=" + std::to_string(test_case.n1) + ", n2=1024)");
    ExpectPlanContains(plan,
                       "LeafPlan(n=" + std::to_string(test_case.n1) + ", factors=[" +
                           std::to_string(test_case.register_radix) +
                           ",32], lanes=" + std::to_string(test_case.register_radix));
    ExpectPlanContains(plan, "LeafPlan(n=1024, factors=[32,32], lanes=32");
    EXPECT_EQ(flagfftDestroy(plan), FLAGFFT_SUCCESS);
  }
}

TEST(Plan1D, Size2P20DecompositionTuneCandidatesFreezeOnePlanPerSplit) {
  flagfft::FFTRequest request;
  request.fft_length = int64_t {1} << 20;
  request.requested_n = request.fft_length;
  request.input_dtype = "complex64";
  request.output_dtype = "complex64";
  request.device_type = "unit-test";
  request.device_arch = "sm80";
  request.direction = "forward";
  request.batch = 1;

  flagfft::PlanBuilder builder;
  auto candidates = builder.build_decomposition_tune_candidates(request.fft_length, request, 5);
  ASSERT_EQ(candidates.size(), 5U);

  std::set<std::pair<int64_t, int64_t>> splits;
  for (const auto& candidate : candidates) {
    auto four_step = std::dynamic_pointer_cast<flagfft::FourStepPlanNode>(candidate.node);
    ASSERT_NE(four_step, nullptr);
    EXPECT_TRUE(splits.insert({four_step->n1, four_step->n2}).second);
  }
  EXPECT_EQ(splits.count({1024, 1024}), 1U);
}

TEST(Plan1D, DoublePlanAvoidsHighSharedMemoryLeaf) {
  flagfft::FFTRequest request;
  request.fft_length = 655360;
  request.requested_n = request.fft_length;
  request.input_dtype = "complex128";
  request.output_dtype = "complex128";
  request.device_type = "unit-test";
  request.device_arch = "sm_80";
  request.direction = "forward";
  request.batch = 1;

  flagfft::PlanBuilder builder;
  auto node = builder.build(request.fft_length, request);
  auto four_step = std::dynamic_pointer_cast<flagfft::FourStepPlanNode>(node);
  ASSERT_NE(four_step, nullptr);
  EXPECT_EQ(four_step->n1, 640);
  EXPECT_EQ(four_step->n2, 1024);

  auto row = std::dynamic_pointer_cast<flagfft::LeafPlanNode>(four_step->row_plan);
  auto col = std::dynamic_pointer_cast<flagfft::LeafPlanNode>(four_step->col_plan);
  ASSERT_NE(row, nullptr);
  ASSERT_NE(col, nullptr);
  EXPECT_EQ(row->factors, (std::vector<int64_t> {20, 32}));
  EXPECT_EQ(col->factors, (std::vector<int64_t> {32, 32}));
  EXPECT_EQ(row->smem_size, 1024);
  EXPECT_EQ(col->smem_size, 1024);
}

TEST(Plan1D, A100DoubleUsesOnlyLeafFusedBluesteinOverRader) {
  if (flagfft::adaptor::backend_name() != "cuda") GTEST_SKIP() << "A100 fused Bluestein policy";
  flagfft::FFTRequest request;
  request.input_dtype = "complex128";
  request.output_dtype = "complex128";
  request.device_type = flagfft::adaptor::backend_name();
  request.device_index = 0;
  request.device_arch = "sm_80";
  request.direction = "forward";
  request.batch = 1;

  request.fft_length = 1009;
  request.requested_n = request.fft_length;
  flagfft::PlanBuilder leaf_builder;
  auto leaf_fused = leaf_builder.build(request.fft_length, request);
  EXPECT_NE(std::dynamic_pointer_cast<flagfft::BluesteinPlanNode>(leaf_fused), nullptr);

  request.fft_length = 8191;
  request.requested_n = request.fft_length;
  flagfft::PlanBuilder four_step_builder;
  auto four_step = four_step_builder.build(request.fft_length, request);
  EXPECT_NE(std::dynamic_pointer_cast<flagfft::RaderPlanNode>(four_step), nullptr);
}

TEST(Plan1D, BluesteinBoundaryKernelsKeepPackingMetadata) {
  if (flagfft::adaptor::backend_name() == "maca") GTEST_SKIP() << "MACA uses split Bluestein kernels";
  flagfft::FFTRequest request;
  request.input_dtype = "complex64";
  request.output_dtype = "complex64";
  request.device_type = flagfft::adaptor::backend_name();
  request.device_index = 0;
  request.device_arch = flagfft::adaptor::device_architecture(0);
  request.direction = "forward";
  request.batch = 1;
  request.fft_length = request.requested_n = 8191;
  flagfft::PlanBuilder builder;
  auto leaf = builder.build(128, request);
  auto convolution = std::make_shared<flagfft::FourStepPlanNode>(16384, 128, 128, leaf, leaf);
  auto plan = std::make_shared<flagfft::BluesteinPlanNode>(8191, 16384, convolution);
  flagfft::TritonCompiler compiler;
  auto compiled = std::dynamic_pointer_cast<flagfft::CompiledRawBluesteinFourStepNode>(
      compiler.compile_raw_node(plan, request, 1));
  ASSERT_NE(compiled, nullptr);
  // The boundary kernels and plain column kernel use identical 128-point
  // leaf layouts. Losing metadata on any boundary used to inflate its grid.
  ASSERT_GT(compiled->first_col_kernel->inner_pack, 1);
  EXPECT_EQ(compiled->prepare_row_kernel->inner_pack, compiled->first_col_kernel->inner_pack);
  EXPECT_EQ(compiled->pointwise_row_kernel->inner_pack, compiled->first_col_kernel->inner_pack);
  EXPECT_EQ(compiled->finish_col_kernel->inner_pack, compiled->first_col_kernel->inner_pack);
}

TEST(Plan1D, MusaBatched8191PrefersBluesteinWithoutChangingLeafRader) {
  flagfft::FFTRequest request;
  request.input_dtype = request.output_dtype = "complex128";
  request.device_type = "musa";
  request.device_arch = "31";
  request.device_index = 0;
  request.direction = "forward";
  flagfft::PlanBuilder builder;
  request.fft_length = request.requested_n = 8191;
  request.batch = 16;
  auto batched = builder.build(8191, request);
  auto bluestein = std::dynamic_pointer_cast<flagfft::BluesteinPlanNode>(batched);
  ASSERT_NE(bluestein, nullptr);
  EXPECT_EQ(bluestein->conv_length, 16384);

  request.batch = 1;
  EXPECT_NE(std::dynamic_pointer_cast<flagfft::RaderPlanNode>(builder.build(8191, request)), nullptr);
  request.batch = 16;
  request.fft_length = request.requested_n = 1009;
  EXPECT_NE(std::dynamic_pointer_cast<flagfft::RaderPlanNode>(builder.build(1009, request)), nullptr);
  request.device_type = "cuda";
  request.device_arch = "sm_80";
  request.fft_length = request.requested_n = 8191;
  EXPECT_NE(std::dynamic_pointer_cast<flagfft::RaderPlanNode>(builder.build(8191, request)), nullptr);
}

TEST(Plan1D, BatchedPrimeTunerIncludesBothAlgorithms) {
  flagfft::FFTRequest request;
  request.device_index = 0;
  request.input_dtype = request.output_dtype = "complex128";
  request.device_type = "musa";
  request.device_arch = "31";
  request.direction = "forward";
  flagfft::PlanBuilder builder;
  request.batch = 64;
  for (int64_t n : {4093, 8191, 12289, 16381}) {
    request.fft_length = request.requested_n = n;
    auto plans = builder.build_decomposition_tune_candidates(n, request, 3);
    ASSERT_EQ(plans.size(), 2u);
    bool bs = false, rader = false;
    for (const auto& p : plans) {
      bs |= std::dynamic_pointer_cast<flagfft::BluesteinPlanNode>(p.node) != nullptr;
      rader |= std::dynamic_pointer_cast<flagfft::RaderPlanNode>(p.node) != nullptr;
    }
    EXPECT_TRUE(bs);
    EXPECT_TRUE(rader);
    EXPECT_EQ(builder.build_decomposition_tune_candidates(n, request, 1).size(), 1u);
  }
}

TEST(Plan1D, BatchedCompositeTunerIncludesBothSplitOrientations) {
  flagfft::FFTRequest request;
  request.device_index = 0;
  request.device_type = "musa";
  request.device_arch = "31";
  request.direction = "forward";
  flagfft::PlanBuilder builder;
  for (const char* dtype : {"complex64", "complex128"}) {
    request.input_dtype = request.output_dtype = dtype;
    request.batch = 64;
    for (int64_t n : {98304, 131072, 196608}) {
      request.fft_length = request.requested_n = n;
      auto plans = builder.build_decomposition_tune_candidates(n, request, 5);
      bool row_shorter = false, col_shorter = false;
      for (const auto& p : plans) {
        auto plan = std::dynamic_pointer_cast<flagfft::FourStepPlanNode>(p.node);
        if (!plan) continue;
        row_shorter |= plan->n1 < plan->n2;
        col_shorter |= plan->n1 > plan->n2;
      }
      EXPECT_TRUE(row_shorter);
      EXPECT_TRUE(col_shorter);
    }
  }
  EXPECT_NE(flagfft::batch_bucket(16), flagfft::batch_bucket(64));
  EXPECT_NE(flagfft::batch_bucket(512), flagfft::batch_bucket(513));
}

TEST(Plan1D, DoubleMixedPlansModelCooperativeStagesAndRowPacking) {
  if (flagfft::adaptor::backend_name() != "cuda") GTEST_SKIP() << "Measured CUDA splits and packing";
  struct MixedCase {
    int64_t length;
    int64_t n1;
    int64_t n2;
    std::vector<int64_t> row_factors;
    std::vector<int64_t> col_factors;
  };
  const MixedCase cases[] = {
      { 477360, 510,  936,    {30, 17},   {13, 9, 8}},
      { 855712, 442, 1936, {17, 13, 2}, {16, 11, 11}},
      { 961875, 475, 2025,    {25, 19},  {15, 15, 9}},
      {1001385, 495, 2023, {15, 11, 3},  {17, 17, 7}},
  };

  for (const MixedCase& test_case : cases) {
    flagfft::FFTRequest request;
    request.fft_length = test_case.length;
    request.requested_n = request.fft_length;
    request.input_dtype = "complex128";
    request.output_dtype = "complex128";
    request.device_type = flagfft::adaptor::backend_name();
    request.device_index = 0;
    request.device_arch = "sm_80";
    request.direction = "forward";
    request.batch = 1;

    flagfft::PlanBuilder builder;
    auto node = builder.build(request.fft_length, request);
    auto four_step = std::dynamic_pointer_cast<flagfft::FourStepPlanNode>(node);
    ASSERT_NE(four_step, nullptr);
    EXPECT_EQ(four_step->n1, test_case.n1);
    EXPECT_EQ(four_step->n2, test_case.n2);

    auto row = std::dynamic_pointer_cast<flagfft::LeafPlanNode>(four_step->row_plan);
    auto col = std::dynamic_pointer_cast<flagfft::LeafPlanNode>(four_step->col_plan);
    ASSERT_NE(row, nullptr);
    ASSERT_NE(col, nullptr);
    EXPECT_EQ(row->factors, test_case.row_factors);
    EXPECT_EQ(col->factors, test_case.col_factors);
  }
}

TEST(Plan1D, LargeMixedDecompositionTuneCandidatesIncludeBalancedSplit) {
  flagfft::FFTRequest request;
  request.fft_length = 663000;
  request.requested_n = request.fft_length;
  request.input_dtype = "complex64";
  request.output_dtype = "complex64";
  request.device_type = "unit-test";
  request.device_arch = "sm80";
  request.direction = "forward";
  request.batch = 1;

  flagfft::PlanBuilder builder;
  auto candidates = builder.build_decomposition_tune_candidates(request.fft_length, request, 2);
  ASSERT_EQ(candidates.size(), 2U);

  std::set<std::pair<int64_t, int64_t>> splits;
  for (const auto& candidate : candidates) {
    auto four_step = std::dynamic_pointer_cast<flagfft::FourStepPlanNode>(candidate.node);
    ASSERT_NE(four_step, nullptr);
    splits.insert({four_step->n1, four_step->n2});
  }
  EXPECT_EQ(splits.count({425, 1560}), 1U);
  EXPECT_EQ(splits.count({780, 850}), 1U);
}

TEST(Plan1D, BatchFour8192UsesMeasuredSplit) {
  if (flagfft::adaptor::backend_name() != "cuda") GTEST_SKIP() << "Measured CUDA split";
  setenv("FLAGFFT_TUNE_DISABLE", "1", 1);

  flagfftHandle plan8192 = nullptr;
  ASSERT_EQ(flagfftPlan1d(&plan8192, 8192, FLAGFFT_C2C, 4), FLAGFFT_SUCCESS);
  ExpectPlanContains(plan8192, "FourStep(n=8192, n1=128, n2=64)");
  EXPECT_EQ(flagfftDestroy(plan8192), FLAGFFT_SUCCESS);

  flagfftHandle plan8192z = nullptr;
  ASSERT_EQ(flagfftPlan1d(&plan8192z, 8192, FLAGFFT_Z2Z, 4), FLAGFFT_SUCCESS);
  ExpectPlanContains(plan8192z, "FourStep(n=8192, n1=128, n2=64)");
  EXPECT_EQ(flagfftDestroy(plan8192z), FLAGFFT_SUCCESS);

  flagfftHandle batchOne = nullptr;
  ASSERT_EQ(flagfftPlan1d(&batchOne, 8192, FLAGFFT_C2C, 1), FLAGFFT_SUCCESS);
  ExpectPlanContains(batchOne, "FourStep(n=8192, n1=64, n2=128)");
  EXPECT_EQ(flagfftDestroy(batchOne), FLAGFFT_SUCCESS);
}

TEST(Plan1D, R2CFourStepReadsRealInputAndWritesHalfOutputDirectly) {
  setenv("FLAGFFT_TUNE_DISABLE", "1", 1);

  flagfftHandle plan = nullptr;
  ASSERT_EQ(flagfftPlan1d(&plan, 8192, FLAGFFT_R2C, 256), FLAGFFT_SUCCESS);

  const char* raw_desc = flagfftGetPlanDescription(plan);
  ASSERT_NE(raw_desc, nullptr);
  std::string desc(raw_desc);
  EXPECT_NE(desc.find("CompiledRawR2CFourStepRealInHalfOut"), std::string::npos);
  EXPECT_EQ(desc.find("expand_kernel=_real_to_complex"), std::string::npos);
  EXPECT_EQ(desc.find("pack_kernel=_r2c_half_pack"), std::string::npos);

  EXPECT_EQ(flagfftDestroy(plan), FLAGFFT_SUCCESS);
}

TEST(Plan1D, C2RFourStepReadsCompactInputAndWritesRealOutputDirectly) {
  setenv("FLAGFFT_TUNE_DISABLE", "1", 1);

  flagfftHandle plan = nullptr;
  ASSERT_EQ(flagfftPlan1d(&plan, 8192, FLAGFFT_C2R, 256), FLAGFFT_SUCCESS);

  const char* raw_desc = flagfftGetPlanDescription(plan);
  ASSERT_NE(raw_desc, nullptr);
  std::string desc(raw_desc);
  EXPECT_NE(desc.find("CompiledRawC2RFourStepCompactInRealOut"), std::string::npos);
  EXPECT_EQ(desc.find("expand_kernel=_compact_to_hermitian_full"), std::string::npos);
  EXPECT_EQ(desc.find("pack_kernel=_complex_to_real"), std::string::npos);

  EXPECT_EQ(flagfftDestroy(plan), FLAGFFT_SUCCESS);
}

TEST(Plan1D, R2CLeafReadsRealInputAndWritesHalfOutputDirectly) {
  setenv("FLAGFFT_TUNE_DISABLE", "1", 1);

  flagfftHandle plan = nullptr;
  ASSERT_EQ(flagfftPlan1d(&plan, 4096, FLAGFFT_R2C, 256), FLAGFFT_SUCCESS);

  const char* raw_desc = flagfftGetPlanDescription(plan);
  ASSERT_NE(raw_desc, nullptr);
  std::string desc(raw_desc);
  EXPECT_NE(desc.find("LeafPlan(n=4096"), std::string::npos);
  EXPECT_NE(desc.find("CompiledRawR2CLeaf"), std::string::npos);
  EXPECT_EQ(desc.find("expand_kernel=_real_to_complex"), std::string::npos);
  EXPECT_EQ(desc.find("pack_kernel=_r2c_half_pack"), std::string::npos);

  EXPECT_EQ(flagfftDestroy(plan), FLAGFFT_SUCCESS);
}

TEST(Plan1D, C2RLeafReadsCompactInputAndWritesRealOutputDirectly) {
  setenv("FLAGFFT_TUNE_DISABLE", "1", 1);

  flagfftHandle plan = nullptr;
  ASSERT_EQ(flagfftPlan1d(&plan, 4096, FLAGFFT_C2R, 256), FLAGFFT_SUCCESS);

  const char* raw_desc = flagfftGetPlanDescription(plan);
  ASSERT_NE(raw_desc, nullptr);
  std::string desc(raw_desc);
  EXPECT_NE(desc.find("LeafPlan(n=4096"), std::string::npos);
  EXPECT_NE(desc.find("CompiledRawC2RLeaf"), std::string::npos);
  EXPECT_EQ(desc.find("expand_kernel=_compact_to_hermitian_full"), std::string::npos);
  EXPECT_EQ(desc.find("pack_kernel=_complex_to_real"), std::string::npos);

  EXPECT_EQ(flagfftDestroy(plan), FLAGFFT_SUCCESS);
}

// =========================================================================
// 2D plan tests
// =========================================================================

TEST(Plan2D, CreateDestroyAllTypes) {
  flagfftType types[] = {FLAGFFT_C2C, FLAGFFT_Z2Z};
  for (auto type : types) {
    flagfftHandle plan = nullptr;
    EXPECT_EQ(flagfftPlan2d(&plan, 64, 32, type), FLAGFFT_SUCCESS);
    EXPECT_NE(plan, nullptr);
    EXPECT_EQ(flagfftDestroy(plan), FLAGFFT_SUCCESS);
  }
}

TEST(Plan2D, BatchedPlanManyComplex) {
  int n[2] = {64, 32};
  const int dist = n[0] * n[1];
  flagfftHandle plan = nullptr;
  EXPECT_EQ(flagfftPlanMany(&plan, 2, n, nullptr, 1, dist, nullptr, 1, dist, FLAGFFT_C2C, 4),
            FLAGFFT_SUCCESS);
  EXPECT_NE(plan, nullptr);
  EXPECT_EQ(flagfftDestroy(plan), FLAGFFT_SUCCESS);
}

TEST(Plan2D, RealTypesSupported) {
  // R2C/D2Z forward and C2R/Z2D inverse are now supported
  flagfftType forward_types[] = {FLAGFFT_R2C, FLAGFFT_D2Z};
  for (auto type : forward_types) {
    flagfftHandle plan = nullptr;
    EXPECT_EQ(flagfftPlan2d(&plan, 64, 32, type), FLAGFFT_SUCCESS);
    EXPECT_NE(plan, nullptr);
    EXPECT_EQ(flagfftDestroy(plan), FLAGFFT_SUCCESS);
  }
  flagfftType inverse_types[] = {FLAGFFT_C2R, FLAGFFT_Z2D};
  for (auto type : inverse_types) {
    flagfftHandle plan = nullptr;
    EXPECT_EQ(flagfftPlan2d(&plan, 64, 32, type), FLAGFFT_SUCCESS);
    EXPECT_NE(plan, nullptr);
    EXPECT_EQ(flagfftDestroy(plan), FLAGFFT_SUCCESS);
  }
}

TEST(Plan2D, CustomStrideNotSupportedYet) {
  int n[2] = {64, 32};
  flagfftHandle plan = nullptr;
  EXPECT_EQ(flagfftPlanMany(&plan, 2, n, nullptr, 2, n[0] * n[1], nullptr, 1, n[0] * n[1], FLAGFFT_C2C, 1),
            FLAGFFT_NOT_SUPPORTED);
  EXPECT_EQ(plan, nullptr);
}

TEST(Plan2D, PaddedDistRejected) {
  // idist/odist > logical size must be rejected until 2D execution handles strides
  int n[2] = {64, 32};
  const int logical = n[0] * n[1];
  flagfftHandle plan = nullptr;
  EXPECT_EQ(flagfftPlanMany(&plan, 2, n, nullptr, 1, logical + 16, nullptr, 1, logical, FLAGFFT_C2C, 1),
            FLAGFFT_NOT_SUPPORTED);
  EXPECT_EQ(plan, nullptr);
  EXPECT_EQ(flagfftPlanMany(&plan, 2, n, nullptr, 1, logical, nullptr, 1, logical + 16, FLAGFFT_C2C, 1),
            FLAGFFT_NOT_SUPPORTED);
  EXPECT_EQ(plan, nullptr);
}

TEST(Plan2D, InvalidParameters) {
  flagfftHandle plan = nullptr;
  EXPECT_EQ(flagfftPlan2d(&plan, 0, 32, FLAGFFT_C2C), FLAGFFT_INVALID_SIZE);
  EXPECT_EQ(plan, nullptr);
  EXPECT_EQ(flagfftPlan2d(nullptr, 64, 32, FLAGFFT_C2C), FLAGFFT_INVALID_VALUE);
}

TEST(Plan2D, GetDescription) {
  flagfftHandle plan = nullptr;
  ASSERT_EQ(flagfftPlan2d(&plan, 64, 32, FLAGFFT_C2C), FLAGFFT_SUCCESS);
  const char* desc = flagfftGetPlanDescription(plan);
  EXPECT_NE(desc, nullptr);
  EXPECT_GT(std::strlen(desc), 0u);
  EXPECT_EQ(flagfftDestroy(plan), FLAGFFT_SUCCESS);
}

TEST(Plan2D, PrimeAxisUsesRader) {
  flagfftHandle plan = nullptr;
  ASSERT_EQ(flagfftPlan2d(&plan, 67, 32, FLAGFFT_C2C), FLAGFFT_SUCCESS);
  const char* desc = flagfftGetPlanDescription(plan);
  ASSERT_NE(desc, nullptr);
  EXPECT_NE(std::strstr(desc, "DirectDFT"), nullptr) << desc;
  EXPECT_EQ(flagfftDestroy(plan), FLAGFFT_SUCCESS);
}

// =========================================================================
// 3D plan tests
// =========================================================================

TEST(Plan3D, CreateDestroyAllTypes) {
  flagfftType types[] = {FLAGFFT_C2C, FLAGFFT_Z2Z, FLAGFFT_R2C, FLAGFFT_D2Z, FLAGFFT_C2R, FLAGFFT_Z2D};
  for (auto type : types) {
    flagfftHandle plan = nullptr;
    ASSERT_EQ(flagfftPlan3d(&plan, 32, 16, 8, type), FLAGFFT_SUCCESS);
    ASSERT_NE(plan, nullptr);
    const char* desc = flagfftGetPlanDescription(plan);
    ASSERT_NE(desc, nullptr);
    EXPECT_GT(std::strlen(desc), 0u);
    EXPECT_NE(std::strstr(desc, "ThreeDim"), nullptr) << desc;
    EXPECT_EQ(flagfftDestroy(plan), FLAGFFT_SUCCESS);
  }
}

TEST(Plan3D, PrimeAxisUsesRader) {
  flagfftHandle plan = nullptr;
  ASSERT_EQ(flagfftPlan3d(&plan, 32, 67, 8, FLAGFFT_C2C), FLAGFFT_SUCCESS);
  const char* desc = flagfftGetPlanDescription(plan);
  ASSERT_NE(desc, nullptr);
  EXPECT_NE(std::strstr(desc, "DirectDFT"), nullptr) << desc;
  EXPECT_EQ(flagfftDestroy(plan), FLAGFFT_SUCCESS);
}

TEST(Plan3D, InvalidParameters) {
  flagfftHandle plan = nullptr;
  EXPECT_EQ(flagfftPlan3d(&plan, 0, 16, 8, FLAGFFT_C2C), FLAGFFT_INVALID_SIZE);
  EXPECT_EQ(plan, nullptr);
  EXPECT_EQ(flagfftPlan3d(nullptr, 32, 16, 8, FLAGFFT_C2C), FLAGFFT_INVALID_VALUE);
}
