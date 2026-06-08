#include "flagfft_test.h"

#include <cstdlib>
#include <cstring>
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

TEST(Plan1D, Bluestein997UsesFastPowerOfTwoConvolution) {
  setenv("FLAGFFT_TUNE_DISABLE", "1", 1);

  flagfftHandle plan = nullptr;
  ASSERT_EQ(flagfftPlan1d(&plan, 997, FLAGFFT_C2C, 1), FLAGFFT_SUCCESS);

  const char* raw_desc = flagfftGetPlanDescription(plan);
  ASSERT_NE(raw_desc, nullptr);
  std::string desc(raw_desc);
  EXPECT_NE(desc.find("Bluestein(n=997, conv_length=2048)"), std::string::npos);
  EXPECT_NE(desc.find("LeafPlan(n=2048"), std::string::npos);

  EXPECT_EQ(flagfftDestroy(plan), FLAGFFT_SUCCESS);
}

TEST(Plan1D, Size23UsesRawDirectDft) {
  setenv("FLAGFFT_TUNE_DISABLE", "1", 1);

  flagfftHandle plan = nullptr;
  ASSERT_EQ(flagfftPlan1d(&plan, 23, FLAGFFT_C2C, 256), FLAGFFT_SUCCESS);

  const char* raw_desc = flagfftGetPlanDescription(plan);
  ASSERT_NE(raw_desc, nullptr);
  std::string desc(raw_desc);
  EXPECT_NE(desc.find("DirectDFT(n=23)"), std::string::npos);
  EXPECT_NE(desc.find("CompiledRawDirectDft(n=23"), std::string::npos);
  EXPECT_EQ(desc.find("Bluestein(n=23"), std::string::npos);

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
  flagfftType types[] = {FLAGFFT_C2C, FLAGFFT_Z2Z, FLAGFFT_R2C, FLAGFFT_D2Z, FLAGFFT_C2R, FLAGFFT_Z2D};
  for (auto type : types) {
    flagfftHandle plan = nullptr;
    EXPECT_EQ(flagfftPlan2d(&plan, 64, 32, type), FLAGFFT_NOT_SUPPORTED);
    EXPECT_EQ(plan, nullptr);
  }
}

TEST(Plan2D, InvalidParameters) {
  flagfftHandle plan = nullptr;
  EXPECT_EQ(flagfftPlan2d(&plan, 0, 32, FLAGFFT_C2C), FLAGFFT_INVALID_SIZE);
  EXPECT_EQ(plan, nullptr);
  EXPECT_EQ(flagfftPlan2d(nullptr, 64, 32, FLAGFFT_C2C), FLAGFFT_INVALID_VALUE);
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
