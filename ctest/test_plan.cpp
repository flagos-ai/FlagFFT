#include "flagfft_test.h"

using namespace flagfft_test::adaptor;

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
  EXPECT_NE(r, FLAGFFT_SUCCESS);
  EXPECT_EQ(plan, nullptr);

  // Null plan pointer
  r = flagfftPlan1d(nullptr, 256, FLAGFFT_C2C, 1);
  EXPECT_NE(r, FLAGFFT_SUCCESS);
}

TEST(Plan1D, GetDescription) {
  flagfftHandle plan = nullptr;
  ASSERT_EQ(flagfftPlan1d(&plan, 256, FLAGFFT_C2C, 1), FLAGFFT_SUCCESS);
  const char* desc = flagfftGetPlanDescription(plan);
  EXPECT_NE(desc, nullptr);
  EXPECT_GT(std::strlen(desc), 0u);
  EXPECT_EQ(flagfftDestroy(plan), FLAGFFT_SUCCESS);
}

// =========================================================================
// 2D plan tests
// =========================================================================

TEST(Plan2D, CreateDestroyAllTypes) {
  flagfftType types[] = {FLAGFFT_C2C, FLAGFFT_Z2Z, FLAGFFT_R2C, FLAGFFT_D2Z, FLAGFFT_C2R, FLAGFFT_Z2D};
  for (auto type : types) {
    flagfftHandle plan = nullptr;
    EXPECT_EQ(flagfftPlan2d(&plan, 64, 32, type), FLAGFFT_SUCCESS);
    EXPECT_NE(plan, nullptr);
    EXPECT_EQ(flagfftDestroy(plan), FLAGFFT_SUCCESS);
  }
}

TEST(Plan2D, InvalidParameters) {
  flagfftHandle plan = nullptr;
  EXPECT_NE(flagfftPlan2d(&plan, 0, 32, FLAGFFT_C2C), FLAGFFT_SUCCESS);
  EXPECT_NE(flagfftPlan2d(nullptr, 64, 32, FLAGFFT_C2C), FLAGFFT_SUCCESS);
}

// =========================================================================
// 3D plan tests
// =========================================================================

TEST(Plan3D, CreateDestroyAllTypes) {
  flagfftType types[] = {FLAGFFT_C2C, FLAGFFT_Z2Z, FLAGFFT_R2C, FLAGFFT_D2Z, FLAGFFT_C2R, FLAGFFT_Z2D};
  for (auto type : types) {
    flagfftHandle plan = nullptr;
    EXPECT_EQ(flagfftPlan3d(&plan, 32, 16, 8, type), FLAGFFT_SUCCESS);
    EXPECT_NE(plan, nullptr);
    EXPECT_EQ(flagfftDestroy(plan), FLAGFFT_SUCCESS);
  }
}

TEST(Plan3D, InvalidParameters) {
  flagfftHandle plan = nullptr;
  EXPECT_NE(flagfftPlan3d(&plan, 0, 16, 8, FLAGFFT_C2C), FLAGFFT_SUCCESS);
  EXPECT_NE(flagfftPlan3d(nullptr, 32, 16, 8, FLAGFFT_C2C), FLAGFFT_SUCCESS);
}
