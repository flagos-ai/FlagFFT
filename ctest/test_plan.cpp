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
  EXPECT_NE(std::strstr(desc, "Rader"), nullptr) << desc;
  EXPECT_EQ(flagfftDestroy(plan), FLAGFFT_SUCCESS);
}

TEST(Plan1D, PrimeLengthRaderSupportsRealWrappers) {
  flagfftHandle plan = nullptr;
  EXPECT_EQ(flagfftPlan1d(&plan, 67, FLAGFFT_R2C, 1), FLAGFFT_SUCCESS);
  EXPECT_NE(plan, nullptr);
  EXPECT_EQ(flagfftDestroy(plan), FLAGFFT_SUCCESS);
}

TEST(Plan1D, LargeBatchFourStepUsesMeasuredSplit) {
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
  setenv("FLAGFFT_TUNE_DISABLE", "1", 1);

  flagfftHandle plan = nullptr;
  ASSERT_EQ(flagfftPlan1d(&plan, 1 << 20, FLAGFFT_C2C, 1), FLAGFFT_SUCCESS);
  ExpectPlanContains(plan, "FourStep(n=1048576, n1=1024, n2=1024)");
  ExpectPlanContains(plan, "LeafPlan(n=1024, factors=[32,32], lanes=32, num_warps=2");
  ExpectPlanContains(plan, "CompiledRawFourStepFused(n=1048576");
  EXPECT_EQ(flagfftDestroy(plan), FLAGFFT_SUCCESS);
}

TEST(Plan1D, LargeMixedRadicesUseThreadLocalRectangularLeaves) {
  setenv("FLAGFFT_TUNE_DISABLE", "1", 1);

  struct MixedCase {
    int64_t length;
    int64_t n1;
    int64_t register_radix;
  };
  const MixedCase cases[] = {
      {3 * (int64_t {1} << 18), 768, 24},
      {5 * (int64_t {1} << 17), 640, 20},
      {7 * (int64_t {1} << 17), 896, 28},
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

TEST(Plan1D, BatchFour8192UsesMeasuredSplit) {
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
  EXPECT_NE(std::strstr(desc, "Rader"), nullptr) << desc;
  EXPECT_EQ(flagfftDestroy(plan), FLAGFFT_SUCCESS);
}

// =========================================================================
// 3D plan tests
// =========================================================================

TEST(Plan3D, CreateDestroyAllTypes) {
  flagfftType types[] = {FLAGFFT_C2C, FLAGFFT_Z2Z, FLAGFFT_R2C, FLAGFFT_D2Z, FLAGFFT_C2R, FLAGFFT_Z2D};
  for (auto type : types) {
    flagfftHandle plan = nullptr;
    EXPECT_EQ(flagfftPlan3d(&plan, 32, 16, 8, type), FLAGFFT_NOT_SUPPORTED);
    EXPECT_EQ(plan, nullptr);
  }
}

TEST(Plan3D, InvalidParameters) {
  flagfftHandle plan = nullptr;
  EXPECT_EQ(flagfftPlan3d(&plan, 0, 16, 8, FLAGFFT_C2C), FLAGFFT_INVALID_SIZE);
  EXPECT_EQ(plan, nullptr);
  EXPECT_EQ(flagfftPlan3d(nullptr, 32, 16, 8, FLAGFFT_C2C), FLAGFFT_INVALID_VALUE);
}
