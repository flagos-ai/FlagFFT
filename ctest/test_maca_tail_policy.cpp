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
                             "FLAGFFT_MACA_REAL_DFT_REDUCTION"}) {
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
      {328050, "complex64", "ct405x810"}}) {
    auto r = request(n, dtype);
    unsetenv("FLAGFFT_MACA_TAIL_POLICY");
    const auto original = flagfft::PlanKey::from_node(builder.build(n, r)).repr();
    EXPECT_TRUE(flagfft::maca_tail_automatic_plan(r).empty());
    setenv("FLAGFFT_MACA_TAIL_POLICY", "1", 1);
    EXPECT_EQ(flagfft::maca_tail_automatic_plan(r), expected);
    const auto node = builder.build(n, r);
    if (auto bs = std::dynamic_pointer_cast<flagfft::BluesteinPlanNode>(node)) {
      EXPECT_EQ(bs->conv_length, 2048);
    } else {
      auto ct = std::dynamic_pointer_cast<flagfft::FourStepPlanNode>(node);
      ASSERT_NE(ct, nullptr);
      EXPECT_EQ(ct->n1, n == 1048576 ? 1024 : 405);
      EXPECT_EQ(ct->n2, n == 1048576 ? 1024 : 810);
    }
    setenv("FLAGFFT_MACA_TAIL_POLICY", "0", 1);
    EXPECT_EQ(flagfft::PlanKey::from_node(builder.build(n, r)).repr(), original);
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
  setenv("FLAGFFT_MACA_TAIL_POLICY", "yes", 1);
  EXPECT_THROW(flagfft::maca_tail_automatic_plan(base), std::runtime_error);
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
  EXPECT_EQ(baseline, flagfft::maca_tail_codegen_identity("off"));
}
}  // namespace
