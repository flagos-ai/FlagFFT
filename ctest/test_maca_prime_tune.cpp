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

#include <cstdlib>
#include <set>
#include <string>

// CPU-only planner target: supply the C550 capabilities recorded in the
// acceptance manifest, without linking or initializing a device runtime.
namespace flagfft::adaptor {
std::string backend_name() { return "maca"; }
int64_t max_dynamic_smem_bytes(int) { return 64 * 1024; }
}  // namespace flagfft::adaptor

namespace {

class ScopedEnv {
 public:
  ScopedEnv(const char *name, const char *value) : name_(name), existed_(std::getenv(name) != nullptr) {
    if (existed_) previous_ = std::getenv(name);
    if (value != nullptr) {
      setenv(name, value, 1);
    } else {
      unsetenv(name);
    }
  }

  ~ScopedEnv() {
    if (existed_) {
      setenv(name_.c_str(), previous_.c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

 private:
  std::string name_;
  bool existed_ = false;
  std::string previous_;
};

}  // namespace

TEST(HcuFourStep, BalancedRegisterStagesAndPolicyCacheAreBounded) {
  flagfft::FFTRequest request;
  request.device_type = "hcu";
  request.device_arch = "gfx936";
  request.origin_rank = request.raw_dim = 1;
  request.batch = 7;
  request.input_dtype = request.output_dtype = "complex128";
  request.input_layout = "contiguous";
  flagfft::PlanBuilder builder;
  for (int64_t n : {65536, 131072, 262144, 524288, 1048576}) {
    request.n = request.requested_n = request.fft_length = n;
    request.input_shape = {request.batch, n};
    request.input_strides = {n, 1};
    std::vector<std::string> descriptions;
    for (const char *cap : {"auto", "8", "16", "auto"}) {
      ScopedEnv policy("FLAGFFT_HCU_FOURSTEP_RADIX_CAP", cap);
      auto root = builder.build(n, request);
      descriptions.push_back(root->describe());
      auto pair = std::dynamic_pointer_cast<flagfft::FourStepPlanNode>(root);
      ASSERT_NE(pair, nullptr);
      for (const auto &child : {pair->row_plan, pair->col_plan}) {
        auto leaf = std::dynamic_pointer_cast<flagfft::LeafPlanNode>(child);
        ASSERT_NE(leaf, nullptr);
        int64_t product = 1;
        for (auto radix : leaf->factors) {
          product *= radix;
          if (std::string(cap) != "auto") {
            EXPECT_LE(radix, std::stoi(cap));
            EXPECT_GE(radix, 4);
          }
        }
        EXPECT_EQ(product, leaf->length);
      }
    }
    EXPECT_EQ(descriptions.front(), descriptions.back());
    EXPECT_NE(descriptions.front(), descriptions[1]);
  }
}

TEST(HcuFourStep, RegisterStageExperimentPreservesPrimeAndOtherBackendPlans) {
  for (const char *backend : {"hcu", "cuda"}) {
    flagfft::FFTRequest request;
    request.device_type = backend;
    request.device_arch = std::string(backend) == "hcu" ? "gfx936" : "sm_80";
    request.origin_rank = request.raw_dim = 1;
    request.batch = 3;
    request.n = request.requested_n = request.fft_length = 8191;
    request.input_dtype = request.output_dtype = "complex64";
    request.input_layout = "contiguous";
    request.input_shape = {3, 8191};
    request.input_strides = {8191, 1};
    flagfft::PlanBuilder builder;
    ScopedEnv original("FLAGFFT_HCU_FOURSTEP_RADIX_CAP", "auto");
    auto baseline = builder.build(8191, request)->describe();
    ScopedEnv candidate("FLAGFFT_HCU_FOURSTEP_RADIX_CAP", "8");
    EXPECT_EQ(builder.build(8191, request)->describe(), baseline);
    request.n = request.requested_n = request.fft_length = 524287;
    request.input_shape = {3, 524287};
    request.input_strides = {524287, 1};
    auto large_candidate = builder.build(524287, request)->describe();
    ScopedEnv reset("FLAGFFT_HCU_FOURSTEP_RADIX_CAP", "auto");
    EXPECT_EQ(builder.build(524287, request)->describe(), large_candidate);
  }
}

TEST(Plan2D, MacaRealRowsPolicyIsScopedAndNarrow) {
  flagfft::FFTRequest request;
  request.device_type = "maca";
  request.raw_dim = 2;
  request.batch = 1;
  request.input_dtype = request.output_dtype = "complex64";
  request.input_layout = "contiguous";
  request.requires_contiguous_copy = false;

  ScopedEnv gate("FLAGFFT_MACA_2D_REAL_ROWS", nullptr);
  EXPECT_FALSE(flagfft::maca_2d_real_rows_enabled(request, 1, 2048, 2048));
  EXPECT_TRUE(flagfft::maca_2d_real_rows_enabled(request, 1, 2048, 2048, true));

  setenv("FLAGFFT_MACA_2D_REAL_ROWS", "0", 1);
  EXPECT_FALSE(flagfft::maca_2d_real_rows_enabled(request, 1, 2048, 2048, true));

  setenv("FLAGFFT_MACA_2D_REAL_ROWS", "1", 1);
  EXPECT_TRUE(flagfft::maca_2d_real_rows_enabled(request, 1, 2048, 2048));
  EXPECT_FALSE(flagfft::maca_2d_real_rows_enabled(request, 1, 256, 2048));
  EXPECT_FALSE(flagfft::maca_2d_real_rows_enabled(request, 1, 2048, 1025));

  request.device_type = "cuda";
  EXPECT_FALSE(flagfft::maca_2d_real_rows_enabled(request, 1, 2048, 2048));
  request.device_type = "maca";
  request.input_dtype = request.output_dtype = "complex128";
  EXPECT_FALSE(flagfft::maca_2d_real_rows_enabled(request, 1, 2048, 2048));
  request.input_dtype = request.output_dtype = "complex64";
  request.batch = 2;
  EXPECT_FALSE(flagfft::maca_2d_real_rows_enabled(request, 2, 2048, 2048));
  request.batch = 1;
  request.raw_dim = 1;
  EXPECT_FALSE(flagfft::maca_2d_real_rows_enabled(request, 1, 2048, 2048));
  request.raw_dim = 2;
  request.requires_contiguous_copy = true;
  EXPECT_FALSE(flagfft::maca_2d_real_rows_enabled(request, 1, 2048, 2048));
}

TEST(Plan2D, RcPreservesOnlySmallMixedBatchedRows) {
  const auto leaf = [](int64_t n, std::vector<int64_t> factors, int64_t lanes = 1) {
    return std::make_shared<flagfft::LeafPlanNode>(n, std::move(factors), 1, lanes, 2,
                                                   std::vector<int64_t> {}, 256);
  };
  const auto row = leaf(209, {19, 11});
  const auto col = leaf(221, {17, 13});
  const auto pair = [](const flagfft::PlanNodePtr &a, const flagfft::PlanNodePtr &b) {
    return std::make_shared<flagfft::FourStepPlanNode>(a->length * b->length, a->length, b->length, a, b);
  };
  EXPECT_TRUE(flagfft::maca_2d_rc_preserve_batched_row(pair(row, col)));
  EXPECT_FALSE(flagfft::maca_2d_rc_preserve_batched_row(row));
  EXPECT_FALSE(flagfft::maca_2d_rc_preserve_batched_row(pair(row, leaf(128, {8, 4, 4}))));
  EXPECT_FALSE(flagfft::maca_2d_rc_preserve_batched_row(pair(row, leaf(273, {21, 13}))));
  EXPECT_FALSE(flagfft::maca_2d_rc_preserve_batched_row(pair(row, leaf(221, {221}))));
  EXPECT_FALSE(flagfft::maca_2d_rc_preserve_batched_row(pair(leaf(209, {19, 11}, 2), col)));
  EXPECT_FALSE(flagfft::maca_2d_rc_preserve_batched_row(
      pair(row, std::make_shared<flagfft::DirectDFTPlanNode>(23))));
}

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

TEST(IXLeafPlan, BalancedRadicesPreserveLengthsAcrossPowerOfTwoLeaves) {
  ScopedEnv enabled("FLAGFFT_IX_BALANCED_RADICES", "1");
  flagfft::FFTRequest request;
  request.device_type = "ix";
  request.device_arch = "71";
  request.input_dtype = request.output_dtype = "complex64";
  request.batch = 3;
  request.raw_dim = request.origin_rank = 1;
  request.input_layout = "contiguous";
  for (int64_t n : {64, 128, 256, 512, 1024, 2048, 4096}) {
    request.fft_length = request.requested_n = n;
    request.input_shape = {3, n};
    request.input_strides = {n, 1};
    flagfft::PlanBuilder builder;
    auto root = builder.build(n, request);
    RecordProperty("ix_plan_" + std::to_string(n), root->describe());
    int checked = 0;
    const auto verify = [&](const auto &self, const flagfft::PlanNodePtr &node) -> void {
      if (auto leaf = std::dynamic_pointer_cast<flagfft::LeafPlanNode>(node)) {
        int64_t product = 1;
        for (int64_t radix : leaf->factors) {
          product *= radix;
          if (leaf->length >= 64) EXPECT_TRUE(radix == 8 || radix == 16);
        }
        EXPECT_EQ(product, leaf->length);
        if (leaf->length >= 64) ++checked;
      } else if (auto pair = std::dynamic_pointer_cast<flagfft::FourStepPlanNode>(node)) {
        self(self, pair->row_plan);
        self(self, pair->col_plan);
      } else {
        FAIL() << node->describe();
      }
    };
    verify(verify, root);
    EXPECT_GT(checked, 0) << n;
  }
}

TEST(IXLeafPlan, BalancedRadicesLeaveOtherTargetsAndMixedLengthsUnchanged) {
  ScopedEnv enabled("FLAGFFT_IX_BALANCED_RADICES", "0");
  flagfft::FFTRequest request;
  request.device_type = "ix";
  request.device_arch = "71";
  request.input_dtype = request.output_dtype = "complex64";
  request.batch = 3;
  request.raw_dim = request.origin_rank = 1;
  for (int target = 0; target < 3; ++target) {
    const int64_t n = target == 0 ? 210 : 1024;
    request.device_type = target == 1 ? "musa" : "ix";
    request.input_dtype = request.output_dtype = target == 2 ? "complex128" : "complex64";
    request.fft_length = request.requested_n = n;
    request.input_shape = {3, n};
    request.input_strides = {n, 1};
    setenv("FLAGFFT_IX_BALANCED_RADICES", "0", 1);
    flagfft::PlanBuilder baseline;
    auto original = baseline.build(n, request)->describe();
    setenv("FLAGFFT_IX_BALANCED_RADICES", "1", 1);
    flagfft::PlanBuilder candidate;
    EXPECT_EQ(candidate.build(n, request)->describe(), original);
  }
}

TEST(IXLeafPlan, ExplicitFourStepSplitScopesRootAndPackedChild) {
  ScopedEnv gate("FLAGFFT_IX_FOURSTEP_SPLIT", "16384:256");
  flagfft::FFTRequest request;
  request.device_type = "ix";
  request.device_arch = "71";
  request.input_dtype = request.output_dtype = "complex64";
  request.batch = 64;
  request.raw_dim = request.origin_rank = 1;
  request.fft_length = request.requested_n = 16384;
  request.input_shape = {64, 16384};
  request.input_strides = {16384, 1};
  for (bool packed : {false, true}) {
    request.packed_real_child = packed;
    flagfft::PlanBuilder builder;
    auto pair = std::dynamic_pointer_cast<flagfft::FourStepPlanNode>(builder.build(16384, request));
    ASSERT_NE(pair, nullptr);
    EXPECT_EQ(pair->n1, 256);
    EXPECT_EQ(pair->n2, 64);
    EXPECT_NE(std::dynamic_pointer_cast<flagfft::LeafPlanNode>(pair->row_plan), nullptr);
    EXPECT_NE(std::dynamic_pointer_cast<flagfft::LeafPlanNode>(pair->col_plan), nullptr);
  }
  setenv("FLAGFFT_IX_FOURSTEP_SPLIT", "16384:127", 1);
  flagfft::PlanBuilder invalid;
  EXPECT_THROW(invalid.build(16384, request), std::runtime_error);
  // A rank-3 axis must ignore the 1D override, even an invalid split.
  request.origin_rank = 3;
  flagfft::PlanBuilder axis;
  EXPECT_NO_THROW(axis.build(16384, request));
}

TEST(IXLeafPlan, Radix8DistributesTheRegisterWorkAcrossMoreLanes) {
  ScopedEnv gate("FLAGFFT_IX_RADIX8", "1");
  flagfft::FFTRequest request;
  request.device_type = "ix";
  request.device_arch = "71";
  request.input_dtype = request.output_dtype = "complex64";
  request.batch = 3;
  request.raw_dim = 1;
  request.origin_rank = 3;
  for (int64_t n : {128,256,512,1024,2048}) {
    request.fft_length = request.requested_n = n;
    request.input_shape = {3,n}; request.input_strides = {n,1};
    flagfft::PlanBuilder builder;
    auto leaf = std::dynamic_pointer_cast<flagfft::LeafPlanNode>(builder.build(n,request));
    ASSERT_NE(leaf,nullptr) << n;
    int64_t product=1;
    for (int64_t radix : leaf->factors) {
      EXPECT_TRUE(radix==2 || radix==4 || radix==8);
      product *= radix;
    }
    EXPECT_EQ(product,n);
    EXPECT_EQ(leaf->lanes,n/8);
  }
}

TEST(IXLeafPlan, ExplicitFactorsAreLengthScopedAndValidated) {
  flagfft::FFTRequest request;
  request.device_type="ix"; request.device_arch="71";
  request.input_dtype=request.output_dtype="complex64";
  request.batch=1; request.raw_dim=request.origin_rank=1;
  request.fft_length=request.requested_n=210;
  request.input_shape={1,210}; request.input_strides={210,1};
  std::vector<int64_t> baseline;
  {
    ScopedEnv absent("FLAGFFT_IX_LEAF_FACTORS",nullptr);
    flagfft::PlanBuilder baseline_builder;
    auto leaf=std::dynamic_pointer_cast<flagfft::LeafPlanNode>(baseline_builder.build(210,request));
    ASSERT_NE(leaf,nullptr);
    baseline=leaf->factors;
  }
  ScopedEnv override("FLAGFFT_IX_LEAF_FACTORS","210:30x7");
  flagfft::PlanBuilder builder;
  auto leaf=std::dynamic_pointer_cast<flagfft::LeafPlanNode>(builder.build(210,request));
  ASSERT_NE(leaf,nullptr);
  EXPECT_EQ(leaf->factors,(std::vector<int64_t>{30,7}));
  for (const char *invalid : {"210:30x0","210:30x6","210:30x7x","210:30x7junk","210:30x7:2","210:15x14"}) {
    setenv("FLAGFFT_IX_LEAF_FACTORS",invalid,1);
    flagfft::PlanBuilder invalid_builder;
    EXPECT_THROW(invalid_builder.build(210,request),std::runtime_error) << invalid;
  }
  setenv("FLAGFFT_IX_LEAF_FACTORS","1024:16x8x8",1);
  flagfft::PlanBuilder other_length;
  auto original=std::dynamic_pointer_cast<flagfft::LeafPlanNode>(other_length.build(210,request));
  ASSERT_NE(original,nullptr);
  EXPECT_EQ(original->factors,baseline);
}

TEST(IXNDPolicy, QualifiedShapesAndExplicitRollbackAreScoped) {
  ScopedEnv policy("FLAGFFT_EXECUTION_POLICY","balanced");
  ScopedEnv gate("FLAGFFT_IX_ND_WARP",nullptr);
  ScopedEnv warp("FLAGFFT_IX_WARP_FFT",nullptr);
  ScopedEnv warp256("FLAGFFT_IX_WARP256",nullptr);
  ScopedEnv store2d("FLAGFFT_IX_2D_TRANSPOSE_STORE",nullptr);
  ScopedEnv store3d("FLAGFFT_3D_FUSED_STORE",nullptr);
  ScopedEnv hybrid("FLAGFFT_IX_3D_HYBRID",nullptr);
  ScopedEnv real_hybrid("FLAGFFT_IX_3D_REAL_HYBRID",nullptr);
  ScopedEnv real_rtrt("FLAGFFT_IX_3D_REAL_RTRT",nullptr);
  ScopedEnv middle("FLAGFFT_IX_3D_R2C_FUSED_MIDDLE",nullptr);
  flagfft::FFTRequest request;
  request.device_type="ix"; request.device_arch="71";
  request.input_dtype=request.output_dtype="complex64";
  request.input_layout="contiguous";
  request.raw_dim=request.origin_rank=2;
  EXPECT_EQ(flagfft::ix_nd_warp_default_pack(request,1,2048,2048),16);
  EXPECT_EQ(flagfft::ix_nd_warp_default_pack(request,4,2048,2048),16);
  EXPECT_EQ(flagfft::ix_nd_warp_default_pack(request,3,2048,2048),0);
  EXPECT_EQ(flagfft::ix_nd_warp_default_pack(request,1,1024,2048),0);
  request.input_dtype="complex128";
  EXPECT_EQ(flagfft::ix_nd_warp_default_pack(request,1,2048,2048),0);
  request.input_dtype="complex64";
  request.requires_contiguous_copy=true;
  EXPECT_EQ(flagfft::ix_nd_warp_default_pack(request,1,2048,2048),0);
  request.requires_contiguous_copy=false;
  setenv("FLAGFFT_IX_ND_WARP","0",1);
  EXPECT_EQ(flagfft::ix_nd_warp_default_pack(request,1,2048,2048),0);
  unsetenv("FLAGFFT_IX_ND_WARP");
  setenv("FLAGFFT_IX_WARP_FFT","0",1);
  EXPECT_EQ(flagfft::ix_nd_warp_default_pack(request,1,2048,2048),0);
  unsetenv("FLAGFFT_IX_WARP_FFT");
  setenv("FLAGFFT_IX_2D_TRANSPOSE_STORE","0",1);
  EXPECT_EQ(flagfft::ix_nd_warp_default_pack(request,1,2048,2048),0);
  unsetenv("FLAGFFT_IX_2D_TRANSPOSE_STORE");
  request.raw_dim=request.origin_rank=3;
  EXPECT_EQ(flagfft::ix_nd_warp_default_pack(request,4,256,256,256),16);
  EXPECT_EQ(flagfft::ix_nd_warp_default_pack(request,4,128,2048,64),16);
  request.real_transform_kind="c2r";
  EXPECT_EQ(flagfft::ix_nd_warp_default_pack(request,4,128,2048,64),0);
  request.real_transform_kind="r2c";
  EXPECT_EQ(flagfft::ix_nd_warp_default_pack(request,4,128,2048,64),16);
  request.real_transform_kind.clear();
  setenv("FLAGFFT_IX_3D_HYBRID","1",1);
  EXPECT_EQ(flagfft::ix_nd_warp_default_pack(request,4,256,256,256),0);
  unsetenv("FLAGFFT_IX_3D_HYBRID");
  setenv("FLAGFFT_3D_FUSED_STORE","0",1);
  EXPECT_EQ(flagfft::ix_nd_warp_default_pack(request,4,256,256,256),0);
  unsetenv("FLAGFFT_3D_FUSED_STORE");
  request.real_transform_kind="r2c";
  EXPECT_EQ(flagfft::ix_nd_warp_default_pack(request,4,256,256,256),16);
  setenv("FLAGFFT_IX_3D_R2C_FUSED_MIDDLE","0",1);
  EXPECT_EQ(flagfft::ix_nd_warp_default_pack(request,4,256,256,256),0);
  unsetenv("FLAGFFT_IX_3D_R2C_FUSED_MIDDLE");
  setenv("FLAGFFT_EXECUTION_POLICY","legacy",1);
  EXPECT_EQ(flagfft::ix_nd_warp_default_pack(request,4,256,256,256),0);
}

TEST(IXNDPolicy, PrimeAxisPolicyIsScopedToQualifiedParent) {
  ScopedEnv policy("FLAGFFT_EXECUTION_POLICY","balanced");
  ScopedEnv enabled("FLAGFFT_IX_PRIME_3D",nullptr);
  flagfft::FFTRequest request;
  request.device_type="ix";request.device_arch="71";
  request.raw_dim=request.origin_rank=3;
  request.input_dtype=request.output_dtype="complex64";
  request.input_layout="contiguous";
  EXPECT_TRUE(flagfft::ix_prime_3d_default_enabled(request,1,16,997,64));
  EXPECT_TRUE(flagfft::ix_prime_3d_default_enabled(request,4,16,997,64));
  EXPECT_FALSE(flagfft::ix_prime_3d_default_enabled(request,3,16,997,64));
  EXPECT_FALSE(flagfft::ix_prime_3d_default_enabled(request,4,128,2048,64));
  request.origin_rank=1;
  EXPECT_FALSE(flagfft::ix_prime_3d_default_enabled(request,4,16,997,64));
  request.origin_rank=3;request.requires_contiguous_copy=true;
  EXPECT_FALSE(flagfft::ix_prime_3d_default_enabled(request,4,16,997,64));
  request.requires_contiguous_copy=false;request.input_dtype="complex128";
  EXPECT_FALSE(flagfft::ix_prime_3d_default_enabled(request,4,16,997,64));
  request.input_dtype="complex64";request.device_type="maca";
  EXPECT_FALSE(flagfft::ix_prime_3d_default_enabled(request,4,16,997,64));
  request.device_type="ix";
  setenv("FLAGFFT_IX_PRIME_3D","0",1);
  EXPECT_FALSE(flagfft::ix_prime_3d_default_enabled(request,4,16,997,64));
  setenv("FLAGFFT_IX_PRIME_3D","1",1);
  setenv("FLAGFFT_EXECUTION_POLICY","legacy",1);
  EXPECT_FALSE(flagfft::ix_prime_3d_default_enabled(request,4,16,997,64));
}

TEST(IXNDPolicy, SmallRealWarpPolicyOnlySelectsPublicSingleFP32) {
  ScopedEnv policy("FLAGFFT_EXECUTION_POLICY","balanced");
  ScopedEnv enabled("FLAGFFT_IX_SMALL_REAL",nullptr);
  flagfft::FFTRequest request;
  request.device_type="ix";request.device_arch="71";
  request.raw_dim=request.origin_rank=1;request.batch=1;
  request.requested_n=request.fft_length=16;
  request.input_dtype=request.output_dtype="complex64";
  request.input_layout="contiguous";request.real_transform_kind="r2c";
  EXPECT_TRUE(flagfft::ix_small_real_warp_default_enabled(request));
  request.real_transform_kind="c2r";
  EXPECT_TRUE(flagfft::ix_small_real_warp_default_enabled(request));
  request.real_transform_kind.clear();
  EXPECT_FALSE(flagfft::ix_small_real_warp_default_enabled(request));
  request.real_transform_kind="r2c";request.batch=64;
  EXPECT_FALSE(flagfft::ix_small_real_warp_default_enabled(request));
  request.batch=1;request.origin_rank=3;
  EXPECT_FALSE(flagfft::ix_small_real_warp_default_enabled(request));
  request.origin_rank=1;request.input_dtype="complex128";
  EXPECT_FALSE(flagfft::ix_small_real_warp_default_enabled(request));
  request.input_dtype="complex64";
  setenv("FLAGFFT_IX_SMALL_REAL","0",1);
  EXPECT_FALSE(flagfft::ix_small_real_warp_default_enabled(request));
  setenv("FLAGFFT_IX_SMALL_REAL","1",1);
  setenv("FLAGFFT_EXECUTION_POLICY","legacy",1);
  EXPECT_FALSE(flagfft::ix_small_real_warp_default_enabled(request));
}

TEST(IXLeafPlan, SinglePackedIoDefaultsStayAtThePublicContiguousFP32Boundary) {
  ScopedEnv policy("FLAGFFT_EXECUTION_POLICY","balanced");
  ScopedEnv enabled("FLAGFFT_IX_CT_SINGLE",nullptr);
  flagfft::FFTRequest request;
  request.device_type="ix"; request.device_arch="71";
  request.input_dtype=request.output_dtype="complex64";
  request.batch=1; request.raw_dim=request.origin_rank=1;
  request.input_strides={1}; request.real_transform_kind="c2r";
  request.input_layout="contiguous";
  for (int64_t n : {210,1024,2048}) {
    request.fft_length=request.requested_n=n;
    for (const auto &kind : {"c2r","r2c",""}) {
      request.real_transform_kind=kind;
      EXPECT_TRUE(flagfft::ix_real_single_pack_policy_enabled(request));
      auto other=request; other.origin_rank=3;
      EXPECT_FALSE(flagfft::ix_real_single_pack_policy_enabled(other));
      other=request; other.batch=64;
      EXPECT_FALSE(flagfft::ix_real_single_pack_policy_enabled(other));
      other=request; other.device_type="maca";
      EXPECT_FALSE(flagfft::ix_real_single_pack_policy_enabled(other));
      other=request; other.input_dtype=other.output_dtype="complex128";
      EXPECT_FALSE(flagfft::ix_real_single_pack_policy_enabled(other));
      other=request; other.input_strides={2};
      EXPECT_FALSE(flagfft::ix_real_single_pack_policy_enabled(other));
      other=request; other.real_transform_kind="unknown";
      EXPECT_FALSE(flagfft::ix_real_single_pack_policy_enabled(other));
      other=request; other.requires_contiguous_copy=true;
      EXPECT_FALSE(flagfft::ix_real_single_pack_policy_enabled(other));
    }
  }
  request.fft_length=request.requested_n=16;
  request.real_transform_kind="";
  EXPECT_TRUE(flagfft::ix_real_single_pack_policy_enabled(request));
  request.real_transform_kind="r2c";
  EXPECT_FALSE(flagfft::ix_real_single_pack_policy_enabled(request));
  request.fft_length=request.requested_n=2048;
  setenv("FLAGFFT_IX_CT_SINGLE","0",1);
  EXPECT_FALSE(flagfft::ix_real_single_pack_policy_enabled(request));
  setenv("FLAGFFT_IX_CT_SINGLE","1",1);
  setenv("FLAGFFT_EXECUTION_POLICY","legacy",1);
  EXPECT_FALSE(flagfft::ix_real_single_pack_policy_enabled(request));
}

TEST(IXLeafPlan, ShortSingleRealWarpDefaultsRemainParentScopedAndReversible) {
  ScopedEnv policy("FLAGFFT_EXECUTION_POLICY","balanced");
  ScopedEnv real("FLAGFFT_IX_SHORT_REAL",nullptr);
  ScopedEnv dense("FLAGFFT_IX_WARP_FOURSTEP",nullptr);
  ScopedEnv warp("FLAGFFT_IX_WARP_FFT",nullptr);
  flagfft::FFTRequest request;
  request.device_type="ix"; request.device_arch="71";
  request.input_dtype=request.output_dtype="complex64";
  request.batch=1; request.raw_dim=request.origin_rank=1;
  request.input_layout="contiguous"; request.input_strides={1};
  request.fft_length=request.requested_n=16384;
  request.real_transform_kind="c2r";
  EXPECT_TRUE(flagfft::ix_short_real_warp_default_enabled(request));
  request.real_transform_kind="r2c";
  EXPECT_TRUE(flagfft::ix_short_real_warp_default_enabled(request));
  auto other=request; other.batch=64;
  EXPECT_FALSE(flagfft::ix_short_real_warp_default_enabled(other));
  other=request; other.origin_rank=3;
  EXPECT_FALSE(flagfft::ix_short_real_warp_default_enabled(other));
  other=request; other.real_transform_kind="";
  EXPECT_FALSE(flagfft::ix_short_real_warp_default_enabled(other));
  other=request; other.requires_contiguous_copy=true;
  EXPECT_FALSE(flagfft::ix_short_real_warp_default_enabled(other));
  for (const char *name : {"FLAGFFT_IX_SHORT_REAL","FLAGFFT_IX_WARP_FOURSTEP","FLAGFFT_IX_WARP_FFT"}) {
    ScopedEnv disabled(name,"0");
    EXPECT_FALSE(flagfft::ix_short_real_warp_default_enabled(request));
  }
}

TEST(IXLeafPlan, FourStepPresetsRequireTheQualifiedPublicDenseRequest) {
  ScopedEnv policy("FLAGFFT_EXECUTION_POLICY","balanced");
  ScopedEnv gate("FLAGFFT_IX_TLE_FOURSTEP_DEFAULTS","1");
  ScopedEnv single("FLAGFFT_IX_CT_SINGLE",nullptr);
  ScopedEnv batched("FLAGFFT_IX_CT_BATCH",nullptr);
  flagfft::FFTRequest request;
  request.device_type="ix";request.device_arch="71";
  request.input_dtype=request.output_dtype="complex64";
  request.raw_dim=request.origin_rank=1;request.batch=64;
  request.input_layout="contiguous";
  for (const auto &[n,preset] : std::vector<std::pair<int64_t,int>>{{8191,1},{16381,2},{16384,4},{663000,6}}) {
    request.fft_length=request.requested_n=n;
    request.input_strides={n,1};
    EXPECT_EQ(flagfft::ix_fourstep_default_preset(request),preset);
    auto other=request;other.origin_rank=3;
    EXPECT_EQ(flagfft::ix_fourstep_default_preset(other),0);
    other=request;other.batch=4;
    EXPECT_EQ(flagfft::ix_fourstep_default_preset(other),0);
    other=request;other.input_strides={n+8,1};
    EXPECT_EQ(flagfft::ix_fourstep_default_preset(other),0);
    other=request;other.input_dtype=other.output_dtype="complex128";
    EXPECT_EQ(flagfft::ix_fourstep_default_preset(other),0);
    other=request;other.device_type="maca";
    EXPECT_EQ(flagfft::ix_fourstep_default_preset(other),0);
    other=request;other.real_transform_kind="c2r";other.input_strides={n/2+1,1};
    EXPECT_EQ(flagfft::ix_fourstep_default_preset(other),n==16384?7:preset);
  }
  request.batch=1;
  EXPECT_EQ(flagfft::ix_fourstep_default_preset(request),5);
  request.requested_n=request.fft_length=16384;request.input_strides={16384,1};
  EXPECT_EQ(flagfft::ix_fourstep_default_preset(request),3);
  setenv("FLAGFFT_IX_CT_SINGLE","0",1);
  EXPECT_EQ(flagfft::ix_fourstep_default_preset(request),0);
  unsetenv("FLAGFFT_IX_CT_SINGLE");
  setenv("FLAGFFT_IX_TLE_FOURSTEP_DEFAULTS","0",1);
  EXPECT_EQ(flagfft::ix_fourstep_default_preset(request),0);
  unsetenv("FLAGFFT_IX_TLE_FOURSTEP_DEFAULTS");
  EXPECT_EQ(flagfft::ix_fourstep_default_preset(request),3);
  request.batch=64;
  for (const auto &[n,preset] : std::vector<std::pair<int64_t,int>>{{8191,1},{16381,2}}) {
    request.fft_length=request.requested_n=n;request.input_strides={n,1};
    EXPECT_EQ(flagfft::ix_fourstep_default_preset(request),preset);
    ScopedEnv disabled("FLAGFFT_IX_CT_BATCH","0");
    EXPECT_EQ(flagfft::ix_fourstep_default_preset(request),0);
  }
}

TEST(IXLeafPlan, BatchedRealCtaDefaultHasTypedScopeAndExistingRollbacks) {
  ScopedEnv policy("FLAGFFT_EXECUTION_POLICY","balanced");
  ScopedEnv gate("FLAGFFT_IX_TLE_FOURSTEP_DEFAULTS",nullptr);
  flagfft::FFTRequest request;
  request.device_type="ix";request.device_arch="71";
  request.input_dtype=request.output_dtype="complex64";
  request.raw_dim=request.origin_rank=1;request.batch=64;
  request.input_layout="contiguous";
  request.fft_length=request.requested_n=16384;
  request.real_transform_kind="c2r";request.input_strides={8193,1};
  EXPECT_EQ(flagfft::ix_fourstep_default_preset(request),7);
  for (const char *name : {"FLAGFFT_IX_FUSED_C2R","FLAGFFT_IX_FOURSTEP_CTA",
                            "FLAGFFT_IX_PORTABLE_LEAF","FLAGFFT_PACKED_REAL",
                            "FLAGFFT_IX_CT_BATCH","FLAGFFT_IX_TLE_FOURSTEP_DEFAULTS"}) {
    ScopedEnv disabled(name,"0");
    EXPECT_EQ(flagfft::ix_fourstep_default_preset(request),0);
  }
  request.real_transform_kind="r2c";request.input_strides={16384,1};
  EXPECT_EQ(flagfft::ix_fourstep_default_preset(request),8);
  for (const char *name : {"FLAGFFT_IX_FUSED_R2C","FLAGFFT_IX_FOURSTEP_CTA",
                            "FLAGFFT_IX_PORTABLE_LEAF","FLAGFFT_PACKED_REAL",
                            "FLAGFFT_IX_CT_BATCH","FLAGFFT_IX_TLE_FOURSTEP_DEFAULTS"}) {
    ScopedEnv disabled(name,"0");
    EXPECT_EQ(flagfft::ix_fourstep_default_preset(request),0);
  }
  request.real_transform_kind="";
  EXPECT_EQ(flagfft::ix_fourstep_default_preset(request),4);
  {
    ScopedEnv disabled("FLAGFFT_IX_FOURSTEP_CTA","0");
    EXPECT_EQ(flagfft::ix_fourstep_default_preset(request),0);
  }
  {
    ScopedEnv portable("FLAGFFT_IX_PORTABLE_LEAF","1");
    EXPECT_EQ(flagfft::ix_fourstep_default_preset(request),0);
  }
}

TEST(IXLeafPlan, CtBatchDefaultsKeepPublicDenseScopeAndQualifiedFactors) {
  ScopedEnv policy("FLAGFFT_EXECUTION_POLICY","balanced");
  ScopedEnv packed("FLAGFFT_IX_CT_BATCH_PACKED_IO",nullptr);
  ScopedEnv rollback("FLAGFFT_IX_CT_BATCH",nullptr);
  ScopedEnv gate("FLAGFFT_IX_TLE_FOURSTEP_DEFAULTS",nullptr);
  ScopedEnv balanced("FLAGFFT_IX_BALANCED_RADICES",nullptr);
  flagfft::FFTRequest request;
  request.device_type="ix";request.device_arch="71";
  request.input_dtype=request.output_dtype="complex64";
  request.raw_dim=request.origin_rank=1;request.batch=64;
  request.input_layout="contiguous";
  for (int64_t n : {16,210,1024,2048}) {
    request.fft_length=request.requested_n=n;
    request.input_shape={64,n};
    for (const auto &kind : {"","r2c","c2r"}) {
      request.real_transform_kind=kind;
      request.input_strides={std::string(kind)=="c2r" ? n/2+1 : n,1};
      const int preset=n==1024 ? 11 : n==210 ? (std::string(kind)=="r2c" ? 13 : 12) : 9;
      EXPECT_EQ(flagfft::ix_fourstep_default_preset(request),preset);
      request.ix_fourstep_preset=0;
      flagfft::PlanBuilder before;
      const auto geometry=before.build(n,request)->describe();
      request.ix_fourstep_preset=preset;
      flagfft::PlanBuilder after;
      auto node=after.build(n,request);
      if (n==1024) {
        auto leaf=std::dynamic_pointer_cast<flagfft::LeafPlanNode>(node);
        ASSERT_NE(leaf,nullptr);
        EXPECT_EQ(leaf->factors,(std::vector<int64_t>{16,8,8}));
        EXPECT_EQ(leaf->lanes,64);
      } else {
        EXPECT_EQ(node->describe(),geometry);
      }
      auto other=request;other.origin_rank=3;
      EXPECT_EQ(flagfft::ix_fourstep_default_preset(other),0);
      other=request;other.input_strides.front()++;
      EXPECT_EQ(flagfft::ix_fourstep_default_preset(other),0);
      other=request;other.batch=4;
      EXPECT_EQ(flagfft::ix_fourstep_default_preset(other),0);
    }
  }
  setenv("FLAGFFT_IX_CT_BATCH_PACKED_IO","0",1);
  EXPECT_EQ(flagfft::ix_fourstep_default_preset(request),0);
  setenv("FLAGFFT_IX_CT_BATCH_PACKED_IO","1",1);
  setenv("FLAGFFT_IX_CT_BATCH","0",1);
  EXPECT_EQ(flagfft::ix_fourstep_default_preset(request),0);
}


TEST(IXLeafPlan, MixedBatch340200DefaultPreservesGeometryAndTypedPublicScope) {
  ScopedEnv policy("FLAGFFT_EXECUTION_POLICY","balanced");
  ScopedEnv gate("FLAGFFT_IX_TLE_FOURSTEP_DEFAULTS",nullptr);
  ScopedEnv rollback("FLAGFFT_IX_CT_BATCH",nullptr);
  flagfft::FFTRequest request;
  request.device_type="ix"; request.device_arch="71";
  request.input_dtype=request.output_dtype="complex64";
  request.raw_dim=request.origin_rank=1; request.batch=64;
  request.input_layout="contiguous";
  request.fft_length=request.requested_n=340200;
  request.input_shape={64,340200}; request.input_strides={340200,1};
  EXPECT_EQ(flagfft::ix_fourstep_default_preset(request),10);
  flagfft::PlanBuilder old_builder;
  const auto geometry=old_builder.build(340200,request)->describe();
  request.ix_fourstep_preset=10;
  flagfft::PlanBuilder new_builder;
  EXPECT_EQ(new_builder.build(340200,request)->describe(),geometry);
  for (const auto &kind : {"r2c","c2r"}) {
    auto other=request; other.real_transform_kind=kind;
    other.input_strides.front()=std::string(kind)=="c2r" ? 170101 : 340200;
    EXPECT_EQ(flagfft::ix_fourstep_default_preset(other),0);
  }
  auto other=request; other.origin_rank=3;
  EXPECT_EQ(flagfft::ix_fourstep_default_preset(other),0);
  other=request; other.batch=1;
  EXPECT_EQ(flagfft::ix_fourstep_default_preset(other),0);
  other=request; other.input_strides.front()++;
  EXPECT_EQ(flagfft::ix_fourstep_default_preset(other),0);
  other=request; other.input_dtype=other.output_dtype="complex128";
  EXPECT_EQ(flagfft::ix_fourstep_default_preset(other),0);
  for (const char *name : {"FLAGFFT_IX_CT_BATCH","FLAGFFT_IX_TLE_FOURSTEP_DEFAULTS"}) {
    ScopedEnv disabled(name,"0");
    EXPECT_EQ(flagfft::ix_fourstep_default_preset(request),0);
  }
}
