// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include "flagfft/core.hpp"
#include "flagfft/maca_tail_policy.hpp"
#include "exec/c_api_internal.hpp"

namespace {
class TailPolicy : public ::testing::Test {
 protected:
  std::vector<std::pair<std::string, std::optional<std::string>>> saved;
  void SetUp() override {
    for (const char* name : {"FLAGFFT_MACA_TAIL_POLICY", "FLAGFFT_MACA_TAIL_PLAN",
                             "FLAGFFT_TUNE_DISABLE", "FLAGFFT_PACKED_REAL",
                             "FLAGFFT_MACA_REAL_DFT_REDUCTION", "FLAGFFT_MACA_1D_BATCH",
                             "FLAGFFT_MACA_BATCH_VEC_IO", "FLAGFFT_MACA_BATCH_REAL_PACK2",
                             "FLAGFFT_MACA_BATCH_C2C_PACK2",
                             "FLAGFFT_MACA_BATCH_PRIME_VEC_IO",
                             "FLAGFFT_MACA_BATCH_C2C_TREE", "FLAGFFT_MACA_PRIME_DIF",
                             "FLAGFFT_MACA_663000_REAL", "FLAGFFT_MACA_663000_C2C",
                             "FLAGFFT_MACA_663000_BATCH_SPLIT_COLUMN",
                             "FLAGFFT_MACA_663000_PACKED_C2R_SPLIT",
                             "FLAGFFT_MACA_663000_PACKED_COLUMN_PFA",
                             "FLAGFFT_MACA_663000_FP64_PACKED_COLUMN_ORDER",
                             "FLAGFFT_MACA_340200_BATCH_SPLIT",
                             "FLAGFFT_MACA_340200_PACKED_REAL_SPLIT",
                             "FLAGFFT_MACA_16384_PACKED_REAL_SPLIT",
                             "FLAGFFT_MACA_16384_FP64_BATCH_SPLIT",
                             "FLAGFFT_MACA_524288_FP64_REGISTER_PACK",
                             "FLAGFFT_MACA_524288_PACKED_REAL_ROW_PACK4",
                             "FLAGFFT_MACA_524288_PACKED_REAL_SPLIT",
                             "FLAGFFT_MACA_185640_PACKED_REAL_SPLIT",
                             "FLAGFFT_MACA_16_FP64_BATCH_PACK16",
                             "FLAGFFT_MACA_210_BATCH_RADIX6",
                             "FLAGFFT_MACA_RADER_CODELET",
                             "FLAGFFT_MACA_BATCH_PRIME_BLUESTEIN"}) {
      const char* value = std::getenv(name);
      saved.emplace_back(name, value ? std::optional<std::string>(value) : std::nullopt);
      unsetenv(name);
    }
  }
  void TearDown() override {
    for (const auto& [name, value] : saved) {
      if (value) setenv(name.c_str(), value->c_str(), 1);
      else unsetenv(name.c_str());
    }
  }
  flagfft::FFTRequest request(int64_t n, const std::string& dtype) {
    flagfft::FFTRequest r;
    r.raw_dim = r.origin_rank = r.batch = 1;
    r.device_type = "maca";
    r.device_arch = "C550";
    r.input_dtype = r.output_dtype = dtype;
    r.fft_length = r.requested_n = n;
    return r;
  }
};

TEST_F(TailPolicy, ExactWhitelistAndSameBuilderToggle) {
  flagfft::PlanBuilder builder;
  for (auto [n, dtype, expected] : {
      std::tuple<int64_t, std::string, std::string>{997, "complex128", "bs2048"},
      {1009, "complex64", "bs2048"}, {1048576, "complex128", "ct1024x1024"},
      }) {
    auto r = request(n, dtype);
    setenv("FLAGFFT_MACA_TAIL_POLICY", "0", 1);
    const auto original = flagfft::PlanKey::from_node(builder.build(n, r)).repr();
    EXPECT_TRUE(flagfft::maca_tail_automatic_plan(r).empty());
    unsetenv("FLAGFFT_MACA_TAIL_POLICY");
    EXPECT_EQ(flagfft::maca_tail_automatic_plan(r), expected);
    const auto node = builder.build(n, r);
    if (auto bs = std::dynamic_pointer_cast<flagfft::BluesteinPlanNode>(node)) {
      EXPECT_EQ(bs->conv_length, 2048);
    } else {
      auto ct = std::dynamic_pointer_cast<flagfft::FourStepPlanNode>(node);
      ASSERT_NE(ct, nullptr);
      EXPECT_EQ(ct->n1, 1024);
      EXPECT_EQ(ct->n2, 1024);
    }
    setenv("FLAGFFT_MACA_TAIL_POLICY", "1", 1);
    EXPECT_EQ(flagfft::maca_tail_automatic_plan(r), expected);
    setenv("FLAGFFT_MACA_TAIL_POLICY", "0", 1);
    EXPECT_EQ(flagfft::PlanKey::from_node(builder.build(n, r)).repr(), original);
  }
  for (auto [n, dtype, kind] : {
      std::tuple<int64_t, std::string, std::string>{997, "complex128", "d2z"},
      {997, "complex128", "z2d"}, {1009, "complex64", "r2c"},
      {1009, "complex64", "c2r"}}) {
    auto r = request(n, dtype);
    r.real_transform = true;
    r.real_transform_kind = kind;
    unsetenv("FLAGFFT_MACA_TAIL_POLICY");
    EXPECT_EQ(flagfft::maca_tail_automatic_plan(r), "bs2048");
    setenv("FLAGFFT_MACA_TAIL_POLICY", "0", 1);
    EXPECT_EQ(flagfft::maca_tail_automatic_plan(r), "");
    setenv("FLAGFFT_MACA_TAIL_POLICY", "1", 1);
    EXPECT_EQ(flagfft::maca_tail_automatic_plan(r), "bs2048");
  }
}

TEST_F(TailPolicy, ScopeExclusionsAndOverride) {
  setenv("FLAGFFT_MACA_TAIL_POLICY", "1", 1);
  const auto base = request(1048576, "complex128");
  for (int variant = 0; variant < 8; ++variant) {
    auto r = base;
    if (variant == 0) r.raw_dim = 2;
    if (variant == 1) r.origin_rank = 3;
    if (variant == 2) r.batch = 2;
    if (variant == 3) r.real_transform = true;
    if (variant == 4) r.output_dtype = "float64";
    if (variant == 5) r.input_dtype = r.output_dtype = "complex64";
    if (variant == 6) r.requested_n = 524287;
    if (variant == 7) r.device_type = "cuda";
    EXPECT_TRUE(flagfft::maca_tail_automatic_plan(r).empty()) << variant;
  }
  setenv("FLAGFFT_MACA_TAIL_PLAN", "default", 1);
  EXPECT_TRUE(flagfft::maca_tail_automatic_plan(base).empty());
  unsetenv("FLAGFFT_MACA_TAIL_PLAN");
  auto real = request(23, "complex128");
  EXPECT_FALSE(flagfft::maca_tail_real_direct_dft(real));
  real.real_transform = true;
  real.real_transform_kind = "d2z";
  EXPECT_TRUE(flagfft::maca_tail_real_direct_dft(real));
  real.requested_n = 29;
  EXPECT_FALSE(flagfft::maca_tail_real_direct_dft(real));
  real.real_transform_kind = "r2c";
  EXPECT_TRUE(flagfft::maca_tail_real_direct_dft(real));
  real.real_transform_kind = "d2z";
  EXPECT_EQ(flagfft::maca_tail_automatic_plan(real), "");
  setenv("FLAGFFT_MACA_TAIL_POLICY", "yes", 1);
  EXPECT_THROW(flagfft::maca_tail_automatic_plan(base), std::runtime_error);
}

TEST_F(TailPolicy, FusedC550PrimeDefaultScopeAndFallback) {
  auto r = request(1009, "complex128");
  r.device_arch = "102";
  r.input_layout = "contiguous";
  r.input_strides = {1009, 1};
  for (int64_t n : {1009, 997}) {
    r.requested_n = r.fft_length = r.input_strides.front() = n;
    for (int64_t batch : {1, 64}) {
      r.batch = batch;
      EXPECT_TRUE(flagfft::maca_prime_dif_requested(r));
      EXPECT_EQ(flagfft::maca_tail_automatic_plan(r), "bs2048");
      const char* root_policy = batch == 1 ? "FLAGFFT_MACA_TAIL_POLICY" : "FLAGFFT_MACA_1D_BATCH";
      setenv(root_policy, "0", 1);
      EXPECT_FALSE(flagfft::maca_prime_dif_requested(r));
      unsetenv(root_policy);
      setenv("FLAGFFT_MACA_PRIME_DIF", "0", 1);
      EXPECT_FALSE(flagfft::maca_prime_dif_requested(r));
      unsetenv("FLAGFFT_MACA_PRIME_DIF");
    }
  }
  setenv("FLAGFFT_MACA_TAIL_PLAN", "default", 1);
  EXPECT_FALSE(flagfft::maca_prime_dif_requested(r));
  unsetenv("FLAGFFT_MACA_TAIL_PLAN");
  r.real_transform = true;
  r.real_transform_kind = "d2z";
  EXPECT_TRUE(flagfft::maca_prime_dif_requested(r));
  r.real_transform_kind.clear();
  EXPECT_FALSE(flagfft::maca_prime_dif_requested(r));
  r.real_transform = false;
  setenv("FLAGFFT_MACA_PRIME_DIF", "1", 1);
  EXPECT_TRUE(flagfft::maca_prime_dif_requested(r));
  r.origin_rank = 3;
  EXPECT_FALSE(flagfft::maca_prime_dif_requested(r));
  r.origin_rank = 1;
  r.requested_n = r.fft_length = r.input_strides.front() = 8191;
  EXPECT_FALSE(flagfft::maca_prime_dif_requested(r));
}

TEST_F(TailPolicy, C550Packed663000ResourcesMatchMeasuredRequests) {
  using flagfft::KernelKind;
  auto r = request(663000, "complex64");
  r.device_arch = "102";
  r.input_layout = "contiguous";
  r.input_strides = {663000, 1};
  r.real_transform = true;
  r.real_transform_kind = "r2c";
  r.direction = "forward";
  EXPECT_TRUE(flagfft::maca_663000_real_enabled(r));
  EXPECT_EQ(flagfft::maca_tail_codegen_root(r), "off");
  r.packed_real_child = true;
  r.fft_length = r.requested_n = 331500;
  r.input_strides.front() = 331500;
  EXPECT_EQ(flagfft::maca_tail_codegen_root(r), "r2c663000");
  EXPECT_EQ(flagfft::maca_tail_kernel_mode("r2c663000", KernelKind::FourStepCol,
                                         "complex64", 1105, 300, 1105), "r2c663000");
  EXPECT_EQ(flagfft::maca_tail_kernel_mode("r2c663000", KernelKind::FourStepCol,
                                         "complex64", 884, 375, 884), "off");
  r.input_dtype = r.output_dtype = "complex128";
  r.real_transform_kind = "d2z";
  EXPECT_TRUE(flagfft::maca_663000_real_enabled(r));
  EXPECT_EQ(flagfft::maca_tail_codegen_root(r), "off");
  r.batch = 64;
  r.real_transform_kind = "z2d";
  r.direction = "inverse";
  EXPECT_EQ(flagfft::maca_tail_codegen_root(r), "batch-c2r663000");
  setenv("FLAGFFT_MACA_663000_REAL", "0", 1);
  EXPECT_FALSE(flagfft::maca_663000_real_enabled(r));
  EXPECT_EQ(flagfft::maca_tail_codegen_root(r), "off");
}

TEST_F(TailPolicy, C550Packed663000SplitAppliesToFp32BatchInverseChild) {
  using flagfft::KernelKind;
  auto r = request(331500, "complex64");
  r.device_arch = "102";
  r.input_layout = "contiguous";
  r.input_strides = {331500, 1};
  r.batch = 64;
  r.real_transform = r.packed_real_child = true;
  r.real_transform_kind = "c2r";
  r.direction = "inverse";
  flagfft::PlanBuilder builder;
  setenv("FLAGFFT_MACA_663000_PACKED_COLUMN_PFA", "0", 1);
  setenv("FLAGFFT_MACA_663000_PACKED_C2R_SPLIT", "0", 1);
  EXPECT_TRUE(flagfft::maca_tail_automatic_plan(r).empty());
  const auto baseline = std::dynamic_pointer_cast<flagfft::FourStepPlanNode>(builder.build(331500, r));
  ASSERT_NE(baseline, nullptr);
  EXPECT_EQ(baseline->n1, 300);
  EXPECT_EQ(baseline->n2, 1105);
  unsetenv("FLAGFFT_MACA_663000_PACKED_C2R_SPLIT");
  EXPECT_EQ(flagfft::maca_tail_automatic_plan(r), "ct340x975");
  const auto mode = flagfft::maca_tail_codegen_root(r);
  EXPECT_EQ(mode, "batch-c2r663000-split");
  EXPECT_EQ(flagfft::maca_tail_kernel_mode(mode, KernelKind::FourStepRow,
                                         "complex64", 340, 340, 975), mode);
  EXPECT_EQ(flagfft::maca_tail_kernel_mode(mode, KernelKind::FourStepCol,
                                         "complex64", 1105, 300, 1105), "off");

  auto plan = std::dynamic_pointer_cast<flagfft::FourStepPlanNode>(builder.build(331500, r));
  ASSERT_NE(plan, nullptr);
  EXPECT_EQ(plan->n1, 340);
  EXPECT_EQ(plan->n2, 975);
  const auto row = std::dynamic_pointer_cast<flagfft::LeafPlanNode>(plan->row_plan);
  const auto col = std::dynamic_pointer_cast<flagfft::LeafPlanNode>(plan->col_plan);
  ASSERT_NE(row, nullptr);
  ASSERT_NE(col, nullptr);
  EXPECT_EQ(row->factors, (std::vector<int64_t>{17, 5, 4}));
  EXPECT_EQ(col->factors, (std::vector<int64_t>{13, 5, 5, 3}));

  r.packed_real_child = false;
  EXPECT_TRUE(flagfft::maca_tail_automatic_plan(r).empty());
  r.packed_real_child = true;
  r.batch = 1;
  EXPECT_TRUE(flagfft::maca_tail_automatic_plan(r).empty());
  r.batch = 64;
  r.input_dtype = r.output_dtype = "complex128";
  r.real_transform_kind = "z2d";
  EXPECT_TRUE(flagfft::maca_tail_automatic_plan(r).empty());
  EXPECT_FALSE(flagfft::maca_663000_packed_column_pfa_enabled(r));
  unsetenv("FLAGFFT_MACA_663000_PACKED_COLUMN_PFA");
  EXPECT_TRUE(flagfft::maca_663000_packed_column_pfa_enabled(r));
  r.batch = 1;
  EXPECT_FALSE(flagfft::maca_663000_packed_column_pfa_enabled(r));
  r.batch = 64;
  r.input_dtype = r.output_dtype = "complex64";
  r.real_transform_kind = "c2r";
  EXPECT_TRUE(flagfft::maca_663000_packed_column_pfa_enabled(r));
  const auto packed = std::dynamic_pointer_cast<flagfft::FourStepPlanNode>(builder.build(331500, r));
  ASSERT_NE(packed, nullptr);
  EXPECT_EQ(packed->n1, 300);
  EXPECT_EQ(packed->n2, 1105);
  setenv("FLAGFFT_MACA_663000_PACKED_C2R_SPLIT", "1", 1);
  EXPECT_FALSE(flagfft::maca_663000_packed_column_pfa_enabled(r));
  EXPECT_EQ(flagfft::maca_tail_automatic_plan(r), "ct340x975");
  setenv("FLAGFFT_MACA_663000_PACKED_C2R_SPLIT", "0", 1);
  EXPECT_FALSE(flagfft::maca_663000_packed_column_pfa_enabled(r));
  EXPECT_TRUE(flagfft::maca_tail_automatic_plan(r).empty());
  setenv("FLAGFFT_MACA_663000_PACKED_COLUMN_PFA", "1", 1);
  EXPECT_TRUE(flagfft::maca_663000_packed_column_pfa_enabled(r));
  r.input_strides.front() += 1;
  EXPECT_FALSE(flagfft::maca_663000_packed_column_pfa_enabled(r));
}

TEST_F(TailPolicy, C550Batched340200SplitUsesTheMeasuredLeafFactors) {
  auto r = request(340200, "complex64");
  r.device_arch = "102";
  r.input_layout = "contiguous";
  r.input_strides = {340200, 1};
  r.batch = 64;
  flagfft::PlanBuilder builder;
  setenv("FLAGFFT_MACA_340200_BATCH_SPLIT", "0", 1);
  const auto baseline = flagfft::PlanKey::from_node(builder.build(340200, r)).repr();
  EXPECT_FALSE(flagfft::maca_340200_batch_split_enabled(r));

  unsetenv("FLAGFFT_MACA_340200_BATCH_SPLIT");
  const auto plan = std::dynamic_pointer_cast<flagfft::FourStepPlanNode>(builder.build(340200, r));
  ASSERT_NE(plan, nullptr);
  EXPECT_EQ(plan->n1, 567);
  EXPECT_EQ(plan->n2, 600);
  const auto row = std::dynamic_pointer_cast<flagfft::LeafPlanNode>(plan->row_plan);
  const auto col = std::dynamic_pointer_cast<flagfft::LeafPlanNode>(plan->col_plan);
  ASSERT_NE(row, nullptr);
  ASSERT_NE(col, nullptr);
  EXPECT_EQ(row->factors, (std::vector<int64_t>{7, 9, 9}));
  EXPECT_EQ(col->factors, (std::vector<int64_t>{6, 10, 10}));
  const auto mode = flagfft::maca_tail_codegen_root(r);
  EXPECT_EQ(mode, "c2c340200-split");
  EXPECT_EQ(flagfft::maca_tail_kernel_mode(mode, flagfft::KernelKind::FourStepRow,
                                         "complex64", 567, 567, 600), mode);
  setenv("FLAGFFT_MACA_1D_BATCH", "0", 1);
  EXPECT_EQ(flagfft::PlanKey::from_node(builder.build(340200, r)).repr(), baseline);
  unsetenv("FLAGFFT_MACA_1D_BATCH");
  r.input_strides.front() += 5;
  EXPECT_FALSE(flagfft::maca_340200_batch_split_enabled(r));

  r.input_strides.front() = 340200;
  r.input_dtype = r.output_dtype = "complex128";
  const auto wide = std::dynamic_pointer_cast<flagfft::FourStepPlanNode>(builder.build(340200, r));
  ASSERT_NE(wide, nullptr);
  EXPECT_EQ(wide->n1, 504);
  EXPECT_EQ(wide->n2, 675);
  const auto wide_row = std::dynamic_pointer_cast<flagfft::LeafPlanNode>(wide->row_plan);
  const auto wide_col = std::dynamic_pointer_cast<flagfft::LeafPlanNode>(wide->col_plan);
  ASSERT_NE(wide_row, nullptr);
  ASSERT_NE(wide_col, nullptr);
  EXPECT_EQ(wide_row->factors, (std::vector<int64_t>{7, 8, 9}));
  EXPECT_EQ(wide_col->factors, (std::vector<int64_t>{5, 15, 9}));
  EXPECT_EQ(flagfft::maca_tail_codegen_root(r), "z2z340200-split");
}

TEST_F(TailPolicy, C550SinglePackedReal663000Fp64UsesTheMeasuredColumnOrder) {
  auto r = request(331500, "complex128");
  r.device_arch = "102";
  r.input_layout = "contiguous";
  r.input_strides = {331500, 1};
  r.real_transform = r.packed_real_child = true;
  r.real_transform_kind = "d2z";
  r.direction = "forward";
  flagfft::PlanBuilder builder;
  setenv("FLAGFFT_MACA_663000_FP64_PACKED_COLUMN_ORDER", "0", 1);
  const auto baseline = flagfft::PlanKey::from_node(builder.build(331500, r)).repr();
  EXPECT_FALSE(flagfft::maca_663000_fp64_packed_column_order_enabled(r));

  unsetenv("FLAGFFT_MACA_663000_FP64_PACKED_COLUMN_ORDER");
  const auto plan = std::dynamic_pointer_cast<flagfft::FourStepPlanNode>(builder.build(331500, r));
  ASSERT_NE(plan, nullptr);
  EXPECT_EQ(plan->n1, 300);
  EXPECT_EQ(plan->n2, 1105);
  const auto col = std::dynamic_pointer_cast<flagfft::LeafPlanNode>(plan->col_plan);
  ASSERT_NE(col, nullptr);
  EXPECT_EQ(col->factors, (std::vector<int64_t>{13, 5, 17}));
  setenv("FLAGFFT_MACA_TAIL_POLICY", "0", 1);
  EXPECT_EQ(flagfft::PlanKey::from_node(builder.build(331500, r)).repr(), baseline);
  unsetenv("FLAGFFT_MACA_TAIL_POLICY");
  r.direction = "inverse";
  r.real_transform_kind = "z2d";
  EXPECT_TRUE(flagfft::maca_663000_fp64_packed_column_order_enabled(r));
  r.batch = 64;
  EXPECT_FALSE(flagfft::maca_663000_fp64_packed_column_order_enabled(r));
}

TEST_F(TailPolicy, C550BatchedPackedReal340200UsesTheMeasuredSplit) {
  using flagfft::KernelKind;
  auto r = request(170100, "complex64");
  r.device_arch = "102";
  r.input_layout = "contiguous";
  r.input_strides = {170100, 1};
  r.batch = 64;
  r.real_transform = r.packed_real_child = true;
  r.real_transform_kind = "r2c";
  r.direction = "forward";
  flagfft::PlanBuilder builder;
  setenv("FLAGFFT_MACA_340200_PACKED_REAL_SPLIT", "0", 1);
  const auto baseline = flagfft::PlanKey::from_node(builder.build(170100, r)).repr();
  EXPECT_FALSE(flagfft::maca_340200_packed_real_split_enabled(r));

  unsetenv("FLAGFFT_MACA_340200_PACKED_REAL_SPLIT");
  const auto plan = std::dynamic_pointer_cast<flagfft::FourStepPlanNode>(builder.build(170100, r));
  ASSERT_NE(plan, nullptr);
  EXPECT_EQ(plan->n1, 405);
  EXPECT_EQ(plan->n2, 420);
  const auto row = std::dynamic_pointer_cast<flagfft::LeafPlanNode>(plan->row_plan);
  const auto col = std::dynamic_pointer_cast<flagfft::LeafPlanNode>(plan->col_plan);
  ASSERT_NE(row, nullptr);
  ASSERT_NE(col, nullptr);
  EXPECT_EQ(row->factors, (std::vector<int64_t>{5, 9, 9}));
  EXPECT_EQ(col->factors, (std::vector<int64_t>{7, 6, 10}));
  const auto mode = flagfft::maca_tail_codegen_root(r);
  EXPECT_EQ(mode, "batch-real340200-split");
  EXPECT_EQ(flagfft::maca_tail_kernel_mode(mode, KernelKind::FourStepCol,
                                         "complex64", 420, 405, 420), mode);
  setenv("FLAGFFT_MACA_1D_BATCH", "0", 1);
  EXPECT_EQ(flagfft::PlanKey::from_node(builder.build(170100, r)).repr(), baseline);
  unsetenv("FLAGFFT_MACA_1D_BATCH");

  r.direction = "inverse";
  r.real_transform_kind = "c2r";
  EXPECT_TRUE(flagfft::maca_340200_packed_real_split_enabled(r));
  r.packed_real_child = false;
  EXPECT_FALSE(flagfft::maca_340200_packed_real_split_enabled(r));
  r.packed_real_child = true;
  r.input_dtype = r.output_dtype = "complex128";
  EXPECT_FALSE(flagfft::maca_340200_packed_real_split_enabled(r));
}

TEST_F(TailPolicy, C550BatchedPackedReal16384UsesTheMeasuredSplit) {
  using flagfft::KernelKind;
  auto r = request(8192, "complex128");
  r.device_arch = "102";
  r.input_layout = "contiguous";
  r.input_strides = {8192, 1};
  r.batch = 64;
  r.real_transform = r.packed_real_child = true;
  r.direction = "inverse";
  r.real_transform_kind = "z2d";
  flagfft::PlanBuilder builder;
  setenv("FLAGFFT_MACA_16384_PACKED_REAL_SPLIT", "0", 1);
  const auto baseline = flagfft::PlanKey::from_node(builder.build(8192, r)).repr();
  unsetenv("FLAGFFT_MACA_16384_PACKED_REAL_SPLIT");
  const auto plan = std::dynamic_pointer_cast<flagfft::FourStepPlanNode>(builder.build(8192, r));
  ASSERT_NE(plan, nullptr);
  EXPECT_EQ(plan->n1, 128);
  EXPECT_EQ(plan->n2, 64);
  EXPECT_NE(flagfft::PlanKey::from_node(plan).repr(), baseline);
  const auto mode = flagfft::maca_tail_codegen_root(r);
  EXPECT_EQ(mode, "batch-real16384-split");
  EXPECT_EQ(flagfft::maca_tail_kernel_mode(mode, KernelKind::FourStepRow,
                                         "complex128", 128, 128, 64), mode);
  r.input_dtype = r.output_dtype = "complex64";
  r.direction = "forward";
  r.real_transform_kind = "r2c";
  EXPECT_TRUE(flagfft::maca_16384_packed_real_split_enabled(r));
  r.input_strides.front() += 5;
  EXPECT_FALSE(flagfft::maca_16384_packed_real_split_enabled(r));
}

TEST_F(TailPolicy, C550BatchedFp64_16384UsesTheMeasuredSplit) {
  auto r = request(16384, "complex128");
  r.device_arch = "102";
  r.input_layout = "contiguous";
  r.input_strides = {16384, 1};
  r.batch = 64;
  flagfft::PlanBuilder builder;
  setenv("FLAGFFT_MACA_16384_FP64_BATCH_SPLIT", "0", 1);
  const auto baseline = flagfft::PlanKey::from_node(builder.build(16384, r)).repr();
  EXPECT_EQ(flagfft::maca_tail_codegen_root(r), "batch-vecio");
  unsetenv("FLAGFFT_MACA_16384_FP64_BATCH_SPLIT");
  const auto plan = std::dynamic_pointer_cast<flagfft::FourStepPlanNode>(builder.build(16384, r));
  ASSERT_NE(plan, nullptr);
  EXPECT_EQ(plan->n1, 256);
  EXPECT_EQ(plan->n2, 64);
  EXPECT_NE(flagfft::PlanKey::from_node(plan).repr(), baseline);
  EXPECT_EQ(flagfft::maca_tail_codegen_root(r), "z2z16384-split");
  r.batch = 1;
  EXPECT_FALSE(flagfft::maca_16384_fp64_batch_split_enabled(r));
  r.batch = 64;
  r.input_dtype = r.output_dtype = "complex64";
  EXPECT_FALSE(flagfft::maca_16384_fp64_batch_split_enabled(r));
}

TEST_F(TailPolicy, C550BatchedPackedReal524288UsesMeasuredSplit) {
  using flagfft::KernelKind;
  auto r = request(262144, "complex64");
  r.device_arch = "102";
  r.input_layout = "contiguous";
  r.input_strides = {262144, 1};
  r.batch = 64;
  r.real_transform = r.packed_real_child = true;
  r.direction = "forward";
  r.real_transform_kind = "r2c";
  EXPECT_EQ(flagfft::maca_tail_codegen_root(r), "batch-real524288-split");
  setenv("FLAGFFT_MACA_524288_PACKED_REAL_SPLIT", "0", 1);
  EXPECT_EQ(flagfft::maca_tail_codegen_root(r), "batch-vecio");

  unsetenv("FLAGFFT_MACA_524288_PACKED_REAL_SPLIT");
  setenv("FLAGFFT_MACA_524288_PACKED_REAL_ROW_PACK4", "1", 1);
  const auto mode = flagfft::maca_tail_codegen_root(r);
  EXPECT_EQ(mode, "batch-real524288-row4");
  EXPECT_EQ(flagfft::maca_tail_kernel_mode(mode, KernelKind::FourStepRow,
                                         "complex64", 512, 512, 512), mode);
  EXPECT_EQ(flagfft::maca_tail_kernel_mode(mode, KernelKind::FourStepCol,
                                         "complex64", 512, 512, 512), "batch-vecio");

  setenv("FLAGFFT_MACA_524288_PACKED_REAL_SPLIT", "1", 1);
  flagfft::PlanBuilder builder;
  const auto plan = std::dynamic_pointer_cast<flagfft::FourStepPlanNode>(builder.build(262144, r));
  ASSERT_NE(plan, nullptr);
  EXPECT_EQ(plan->n1, 256);
  EXPECT_EQ(plan->n2, 1024);
  const auto split_mode = flagfft::maca_tail_codegen_root(r);
  EXPECT_EQ(split_mode, "batch-real524288-split");
  EXPECT_EQ(flagfft::maca_tail_kernel_mode(split_mode, KernelKind::FourStepCol,
                                         "complex64", 1024, 256, 1024), split_mode);
  unsetenv("FLAGFFT_MACA_524288_PACKED_REAL_SPLIT");
  unsetenv("FLAGFFT_MACA_524288_PACKED_REAL_ROW_PACK4");
  r.direction = "inverse";
  r.real_transform_kind = "c2r";
  EXPECT_EQ(flagfft::maca_tail_codegen_root(r), "batch-real524288-split");
  r.input_dtype = r.output_dtype = "complex128";
  r.real_transform_kind = "z2d";
  EXPECT_EQ(flagfft::maca_tail_codegen_root(r), "batch-vecio");
}

TEST_F(TailPolicy, C550BatchedPackedReal185640UsesTwoMeasuredLeafPacks) {
  using flagfft::KernelKind;
  auto r = request(92820, "complex64");
  r.device_arch = "102";
  r.input_layout = "contiguous";
  r.input_strides = {92820, 1};
  r.batch = 64;
  r.real_transform = r.packed_real_child = true;
  r.direction = "inverse";
  r.real_transform_kind = "c2r";
  EXPECT_FALSE(flagfft::maca_185640_packed_real_split_enabled(r));

  setenv("FLAGFFT_MACA_185640_PACKED_REAL_SPLIT", "1", 1);
  flagfft::PlanBuilder builder;
  const auto plan = std::dynamic_pointer_cast<flagfft::FourStepPlanNode>(builder.build(92820, r));
  ASSERT_NE(plan, nullptr);
  EXPECT_EQ(plan->n1, 420);
  EXPECT_EQ(plan->n2, 221);
  EXPECT_EQ(flagfft::maca_tail_codegen_root(r), "batch-real185640-p2");
  r.input_dtype = r.output_dtype = "complex128";
  r.real_transform_kind = "z2d";
  const auto mode = flagfft::maca_tail_codegen_root(r);
  EXPECT_EQ(mode, "batch-real185640-p4");
  EXPECT_EQ(flagfft::maca_tail_kernel_mode(mode, KernelKind::FourStepCol,
                                         "complex128", 221, 420, 221), mode);
  r.batch = 1;
  EXPECT_FALSE(flagfft::maca_185640_packed_real_split_enabled(r));
}

TEST_F(TailPolicy, C550ShortBatchedComplexOptionsStayLocal) {
  using flagfft::KernelKind;
  auto r = request(16, "complex128");
  r.device_arch = "102";
  r.input_layout = "contiguous";
  r.input_strides = {16, 1};
  r.batch = 64;
  EXPECT_EQ(flagfft::maca_tail_codegen_root(r), "off");
  setenv("FLAGFFT_MACA_16_FP64_BATCH_PACK16", "1", 1);
  const auto mode = flagfft::maca_tail_codegen_root(r);
  EXPECT_EQ(mode, "z2z16-pack16");
  EXPECT_EQ(flagfft::maca_tail_kernel_mode(mode, KernelKind::Leaf, "complex128", 16, 0, 0), mode);
  r.batch = 1;
  EXPECT_NE(flagfft::maca_tail_codegen_root(r), mode);

  r.batch = 64;
  r.input_dtype = r.output_dtype = "complex64";
  r.input_strides = {210, 1};
  r.fft_length = r.requested_n = 210;
  EXPECT_FALSE(flagfft::maca_210_fp32_batch_radix6_enabled(r));
  setenv("FLAGFFT_MACA_210_BATCH_RADIX6", "1", 1);
  flagfft::PlanBuilder builder;
  const auto leaf = std::dynamic_pointer_cast<flagfft::LeafPlanNode>(builder.build(210, r));
  ASSERT_NE(leaf, nullptr);
  EXPECT_EQ(leaf->factors, (std::vector<int64_t>{7, 6, 5}));
  r.real_transform = true;
  EXPECT_FALSE(flagfft::maca_210_fp32_batch_radix6_enabled(r));
}

TEST_F(TailPolicy, C550Fp64_524288Pack4IsScopedToDenseSingleComplex) {
  using flagfft::KernelKind;
  auto r = request(524288, "complex128");
  r.device_arch = "102";
  r.input_layout = "contiguous";
  r.input_strides = {524288, 1};
  setenv("FLAGFFT_MACA_524288_FP64_REGISTER_PACK", "0", 1);
  EXPECT_EQ(flagfft::maca_tail_codegen_root(r), "off");
  unsetenv("FLAGFFT_MACA_524288_FP64_REGISTER_PACK");
  const auto mode = flagfft::maca_tail_codegen_root(r);
  EXPECT_EQ(mode, "z2z524288-pack4");
  EXPECT_EQ(flagfft::maca_tail_kernel_mode(mode, KernelKind::FourStepRow,
                                         "complex128", 512, 512, 1024), mode);
  EXPECT_EQ(flagfft::maca_tail_kernel_mode(mode, KernelKind::FourStepCol,
                                         "complex128", 1024, 1024, 1024), "off");
  r.batch = 64;
  EXPECT_FALSE(flagfft::maca_524288_fp64_register_pack_enabled(r));
  r.batch = 1;
  r.real_transform = true;
  EXPECT_FALSE(flagfft::maca_524288_fp64_register_pack_enabled(r));
}

TEST_F(TailPolicy, C550BatchedSplitColumnUsesTheMatchingFourStepPlan) {
  auto r = request(663000, "complex64");
  r.device_arch = "102";
  r.batch = 64;
  r.input_layout = "contiguous";
  r.input_strides = {663000, 1};
  EXPECT_EQ(flagfft::maca_tail_automatic_plan(r), "ct375x1768");
  setenv("FLAGFFT_MACA_663000_BATCH_SPLIT_COLUMN", "0", 1);
  EXPECT_EQ(flagfft::maca_tail_automatic_plan(r), "ct780x850");
  unsetenv("FLAGFFT_MACA_663000_BATCH_SPLIT_COLUMN");
  setenv("FLAGFFT_MACA_RADER_CODELET", "0", 1);
  EXPECT_EQ(flagfft::maca_tail_automatic_plan(r), "ct780x850");
  unsetenv("FLAGFFT_MACA_RADER_CODELET");
  setenv("FLAGFFT_MACA_663000_BATCH_SPLIT_COLUMN", "1", 1);
  EXPECT_EQ(flagfft::maca_tail_automatic_plan(r), "ct375x1768");
  EXPECT_EQ(flagfft::maca_tail_codegen_root(r), "c2c663000");
  flagfft::PlanBuilder builder;
  auto fourstep = std::dynamic_pointer_cast<flagfft::FourStepPlanNode>(builder.build(663000, r));
  ASSERT_NE(fourstep, nullptr);
  EXPECT_EQ(fourstep->n1, 375);
  EXPECT_EQ(fourstep->n2, 1768);
  setenv("FLAGFFT_MACA_663000_BATCH_SPLIT_COLUMN", "pfa", 1);
  EXPECT_EQ(flagfft::maca_tail_automatic_plan(r), "ct375x1768");
  r.requires_contiguous_copy = true;
  EXPECT_EQ(flagfft::maca_tail_automatic_plan(r), "ct780x850");
  r.requires_contiguous_copy = false;
  setenv("FLAGFFT_MACA_1D_BATCH", "0", 1);
  EXPECT_TRUE(flagfft::maca_tail_automatic_plan(r).empty());
}

TEST_F(TailPolicy, C550BatchedPrimeBluesteinUsesTwoLeafAxes) {
  flagfft::PlanBuilder builder;
  for (const int64_t n : {8191, 16381}) {
    auto r = request(n, "complex64");
    r.device_arch = "102";
    r.batch = 64;
    r.input_layout = "contiguous";
    r.input_strides = {n, 1};
    EXPECT_TRUE(flagfft::maca_tail_automatic_plan(r).empty());
    setenv("FLAGFFT_MACA_BATCH_PRIME_BLUESTEIN", "1", 1);
    auto root = std::dynamic_pointer_cast<flagfft::BluesteinPlanNode>(builder.build(n, r));
    ASSERT_NE(root, nullptr);
    EXPECT_EQ(root->conv_length, n == 8191 ? 16384 : 32768);
    auto conv = std::dynamic_pointer_cast<flagfft::FourStepPlanNode>(root->fft_plan);
    ASSERT_NE(conv, nullptr);
    EXPECT_EQ(conv->n1, 128);
    EXPECT_EQ(conv->n1 * conv->n2, root->conv_length);
    EXPECT_NE(std::dynamic_pointer_cast<flagfft::LeafPlanNode>(conv->row_plan), nullptr);
    EXPECT_NE(std::dynamic_pointer_cast<flagfft::LeafPlanNode>(conv->col_plan), nullptr);
    r.input_dtype = r.output_dtype = "complex128";
    EXPECT_TRUE(flagfft::maca_tail_automatic_plan(r).empty());
    unsetenv("FLAGFFT_MACA_BATCH_PRIME_BLUESTEIN");
  }
}

TEST_F(TailPolicy, NativeDescriptorKeepsOriginalApiAndRank) {
  setenv("FLAGFFT_MACA_TAIL_POLICY", "1", 1);
  flagfft::FlagFFTPlanDesc desc;
  desc.rank = desc.batch = 1;
  desc.n = {23};
  desc.type = FLAGFFT_D2Z;
  desc.precision = flagfft::FlagFFTPrecision::Float64;
  auto r = flagfft::request_from_desc(desc, "forward");
  EXPECT_EQ(r.input_dtype, "complex128");
  EXPECT_EQ(r.output_dtype, "complex128");
  EXPECT_TRUE(r.real_transform);
  EXPECT_EQ(r.real_transform_kind, "d2z");
  EXPECT_TRUE(flagfft::maca_tail_real_direct_dft(r));
  for (int rank : {2, 3}) {
    desc.type = FLAGFFT_D2Z;
    desc.n = {23};
    r = flagfft::request_from_desc(desc, "forward", rank);
    EXPECT_EQ(r.raw_dim, 1);
    EXPECT_EQ(r.batch, 1);
    EXPECT_FALSE(flagfft::maca_tail_real_direct_dft(r));
    desc.type = FLAGFFT_Z2Z;
    desc.n = {1048576};
    r = flagfft::request_from_desc(desc, "inverse", rank);
    EXPECT_TRUE(flagfft::maca_tail_automatic_plan(r).empty());
  }
}

TEST_F(TailPolicy, KernelModeIsNarrowAndCacheSeparatesLegacyTree) {
  using flagfft::KernelKind;
  using flagfft::maca_tail_kernel_mode;
  EXPECT_EQ(maca_tail_kernel_mode("p4w4", KernelKind::FourStepCol,
                                  "complex128", 1024, 1024, 1024), "p4w4");
  for (auto kind : {KernelKind::FourStepRealRow, KernelKind::FourStepColStrided,
                    KernelKind::FourStepR2CCol, KernelKind::FourStepC2RCol,
                    KernelKind::BluesteinFourStepPointwiseRow, KernelKind::Leaf}) {
    EXPECT_EQ(maca_tail_kernel_mode("p4w4", kind, "complex128", 1024, 1024, 1024), "off");
  }
  EXPECT_EQ(maca_tail_kernel_mode("real-direct", KernelKind::DirectDft, "complex128", 23, 0, 0), "off");
  EXPECT_EQ(maca_tail_kernel_mode("real-direct", KernelKind::DirectDftR2C, "complex128", 23, 0, 0), "real-direct");
  const auto baseline = flagfft::maca_tail_codegen_identity("off");
  EXPECT_NE(baseline, flagfft::maca_tail_codegen_identity("p4w4"));
  setenv("FLAGFFT_MACA_REAL_DFT_REDUCTION", "tree", 1);
  const auto tree = flagfft::maca_tail_codegen_identity("off");
  EXPECT_NE(baseline, tree);
  setenv("FLAGFFT_MACA_REAL_DFT_REDUCTION", "kahan", 1);
  EXPECT_NE(tree, flagfft::maca_tail_codegen_identity("off"));
  unsetenv("FLAGFFT_MACA_REAL_DFT_REDUCTION");
  setenv("FLAGFFT_MACA_TRANSPOSE3D_SLICE_GROUP", "2", 1);
  EXPECT_NE(baseline, flagfft::maca_tail_codegen_identity("off"));
  unsetenv("FLAGFFT_MACA_TRANSPOSE3D_SLICE_GROUP");
  setenv("FLAGFFT_MACA_3D_REAL_FUSED16", "1", 1);
  EXPECT_NE(baseline, flagfft::maca_tail_codegen_identity("off"));
  unsetenv("FLAGFFT_MACA_3D_REAL_FUSED16");
  setenv("FLAGFFT_MACA_3D_C2C_FUSED32", "1", 1);
  EXPECT_NE(baseline, flagfft::maca_tail_codegen_identity("off"));
  unsetenv("FLAGFFT_MACA_3D_C2C_FUSED32");
  setenv("FLAGFFT_MACA_3D_FINAL_STORE", "1", 1);
  EXPECT_NE(baseline, flagfft::maca_tail_codegen_identity("off"));
  unsetenv("FLAGFFT_MACA_3D_FINAL_STORE");
  EXPECT_EQ(baseline, flagfft::maca_tail_codegen_identity("off"));
}

TEST_F(TailPolicy, BatchResourceModesStayWithinMeasuredRequestsAndKernels) {
  using flagfft::KernelKind;
  auto complex = request(16384, "complex64");
  complex.device_arch = "102";
  complex.batch = 64;
  complex.input_strides = {16384, 1};
  EXPECT_EQ(flagfft::maca_tail_codegen_root(complex), "batch-vecio");
  EXPECT_EQ(flagfft::maca_tail_kernel_mode("batch-vecio", KernelKind::FourStepRow,
                                         "complex64", 256, 256, 64), "batch-vecio");
  EXPECT_EQ(flagfft::maca_tail_kernel_mode("batch-vecio", KernelKind::FourStepCol,
                                         "complex64", 64, 256, 64), "batch-vecio");
  EXPECT_EQ(flagfft::maca_tail_kernel_mode("batch-vecio", KernelKind::FourStepRow,
                                         "complex64", 64, 64, 128), "batch-vecio");
  EXPECT_EQ(flagfft::maca_tail_kernel_mode("batch-vecio", KernelKind::FourStepCol,
                                         "complex128", 128, 128, 128), "batch-vecio");
  EXPECT_EQ(flagfft::maca_tail_kernel_mode("batch-vecio", KernelKind::FourStepRow,
                                         "complex128", 512, 512, 512), "batch-vecio");
  EXPECT_EQ(flagfft::maca_tail_kernel_mode("batch-vecio", KernelKind::Leaf,
                                         "complex64", 256, 256, 64), "off");
  complex.batch = 1;
  EXPECT_EQ(flagfft::maca_tail_codegen_root(complex), "off");
  complex.batch = 64;
  complex.origin_rank = 2;
  EXPECT_EQ(flagfft::maca_tail_codegen_root(complex), "off");
  complex.origin_rank = 1;
  complex.requested_n = complex.fft_length = 262144;
  complex.input_strides = {262144, 1};
  complex.real_transform = complex.packed_real_child = true;
  EXPECT_EQ(flagfft::maca_tail_codegen_root(complex), "batch-vecio");
  complex.requested_n = complex.fft_length = 16384;
  complex.input_strides = {16384, 1};
  complex.real_transform = complex.packed_real_child = false;
  setenv("FLAGFFT_MACA_BATCH_VEC_IO", "0", 1);
  EXPECT_EQ(flagfft::maca_tail_codegen_root(complex), "off");
  unsetenv("FLAGFFT_MACA_BATCH_VEC_IO");

  auto real_child = request(92820, "complex128");
  real_child.device_arch = "102";
  real_child.batch = 64;
  real_child.input_strides = {92820, 1};
  real_child.real_transform = real_child.packed_real_child = true;
  EXPECT_EQ(flagfft::maca_tail_codegen_root(real_child), "batch-real-pack2");
  EXPECT_EQ(flagfft::maca_tail_kernel_mode("batch-real-pack2", KernelKind::FourStepCol,
                                         "complex128", 357, 260, 357), "batch-real-pack2");
  real_child.packed_real_child = false;
  EXPECT_EQ(flagfft::maca_tail_codegen_root(real_child), "off");
  real_child.packed_real_child = true;
  setenv("FLAGFFT_MACA_1D_BATCH", "0", 1);
  EXPECT_EQ(flagfft::maca_tail_codegen_root(real_child), "off");
  unsetenv("FLAGFFT_MACA_1D_BATCH");

  auto c2c = request(185640, "complex128");
  c2c.device_arch = "102";
  c2c.batch = 64;
  c2c.input_strides = {185640, 1};
  EXPECT_EQ(flagfft::maca_tail_codegen_root(c2c), "batch-c2c-pack2");
  EXPECT_EQ(flagfft::maca_tail_kernel_mode("batch-c2c-pack2", KernelKind::FourStepRow,
                                         "complex128", 390, 390, 476), "batch-c2c-pack2");
  setenv("FLAGFFT_MACA_BATCH_C2C_PACK2", "0", 1);
  EXPECT_EQ(flagfft::maca_tail_codegen_root(c2c), "off");

  auto prime = request(8191, "complex64");
  prime.device_arch = "102";
  prime.batch = 64;
  prime.input_strides = {8191, 1};
  EXPECT_EQ(flagfft::maca_tail_codegen_root(prime), "batch-prime-vecio");
  EXPECT_EQ(flagfft::maca_tail_kernel_mode("batch-prime-vecio", KernelKind::FourStepRow,
                                         "complex64", 90, 90, 91), "batch-prime-vecio");
  EXPECT_EQ(flagfft::maca_tail_kernel_mode("batch-prime-vecio", KernelKind::FourStepCol,
                                         "complex128", 180, 91, 180), "batch-prime-vecio");
  EXPECT_EQ(flagfft::maca_tail_kernel_mode("batch-prime-vecio", KernelKind::RaderPrepare,
                                         "complex64", 0, 0, 0), "batch-prime-rader-vecio");
  EXPECT_EQ(flagfft::maca_tail_kernel_mode("batch-prime-vecio", KernelKind::RaderFinalize,
                                         "complex128", 0, 0, 0), "batch-prime-rader-vecio");
  EXPECT_EQ(flagfft::maca_tail_kernel_mode("batch-prime-vecio", KernelKind::RaderPointwise,
                                         "complex64", 0, 0, 0), "off");
  EXPECT_EQ(flagfft::maca_tail_kernel_mode("batch-prime-vecio", KernelKind::Leaf,
                                         "complex64", 90, 90, 91), "off");
  setenv("FLAGFFT_MACA_BATCH_PRIME_VEC_IO", "0", 1);
  EXPECT_EQ(flagfft::maca_tail_codegen_root(prime), "off");

  auto small_prime = request(23, "complex128");
  small_prime.device_arch = "102";
  small_prime.batch = 64;
  small_prime.input_strides = {23, 1};
  EXPECT_EQ(flagfft::maca_tail_codegen_root(small_prime), "batch-c2c-tree");
  EXPECT_EQ(flagfft::maca_tail_kernel_mode("batch-c2c-tree", KernelKind::DirectDft,
                                         "complex128", 23, 0, 0), "batch-c2c-tree");
  setenv("FLAGFFT_MACA_BATCH_C2C_TREE", "0", 1);
  EXPECT_EQ(flagfft::maca_tail_codegen_root(small_prime), "off");
}
}  // namespace
