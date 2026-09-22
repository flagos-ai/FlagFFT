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

#include "flagfft_test.h"

#include <algorithm>
#include <exception>
#include <string>
#include <vector>

namespace {

struct Dense2DShape {
  int n0;
  int n1;
};

struct PlanGuard {
  ~PlanGuard() {
    if (plan != nullptr) {
      flagfftDestroy(plan);
    }
  }

  flagfftHandle plan = nullptr;
};

bool IsUsableMaca() {
  if (flagfft::adaptor::backend_name() != "maca") {
    return false;
  }
  try {
    return flagfft::adaptor::device_count() > 0;
  } catch (const std::exception&) {
    return false;
  }
}

std::vector<flagfftReal> BoundaryInput(const Dense2DShape& shape) {
  const int64_t total = static_cast<int64_t>(shape.n0) * shape.n1;
  constexpr double dc = 1.25;
  constexpr double nyquist = -0.5;
  std::vector<flagfftReal> input(static_cast<std::size_t>(total));
  for (int i = 0; i < shape.n0; ++i) {
    for (int j = 0; j < shape.n1; ++j) {
      const double checker = ((i + j) & 1) == 0 ? 1.0 : -1.0;
      input[static_cast<std::size_t>(i) * shape.n1 + j] =
          static_cast<flagfftReal>(dc + nyquist * checker);
    }
  }
  return input;
}

void AssertRowBoundaryDescription(flagfftHandle forward, flagfftHandle inverse, const Dense2DShape& shape) {
  const char* forward_description = flagfftGetPlanDescription(forward);
  const char* inverse_description = flagfftGetPlanDescription(inverse);
  ASSERT_NE(forward_description, nullptr) << "missing R2C plan description for " << shape.n0 << "x" << shape.n1;
  ASSERT_NE(inverse_description, nullptr) << "missing C2R plan description for " << shape.n0 << "x" << shape.n1;
  EXPECT_NE(std::string(forward_description).find("CompiledRaw2DR2CRow("), std::string::npos)
      << forward_description;
  EXPECT_NE(std::string(inverse_description).find("CompiledRaw2DC2RRow("), std::string::npos)
      << inverse_description;
}

void AssertShortRealBoundaryPack4Description(flagfftHandle forward,
                                             flagfftHandle inverse,
                                             const Dense2DShape& shape) {
  const char* forward_description = flagfftGetPlanDescription(forward);
  const char* inverse_description = flagfftGetPlanDescription(inverse);
  ASSERT_NE(forward_description, nullptr);
  ASSERT_NE(inverse_description, nullptr);

  const std::string forward_text(forward_description);
  const std::string inverse_text(inverse_description);
  EXPECT_NE(forward_text.find("CompiledRawR2CLeaf(n=" + std::to_string(shape.n1)), std::string::npos)
      << forward_text;
  EXPECT_NE(inverse_text.find("CompiledRawC2RLeaf(n=" + std::to_string(shape.n1)), std::string::npos)
      << inverse_text;
  EXPECT_NE(forward_text.find("batch_per_block=4"), std::string::npos) << forward_text;
  EXPECT_NE(inverse_text.find("batch_per_block=4"), std::string::npos) << inverse_text;
}

void AssertR2CBoundaries(const std::vector<flagfftComplex>& spectrum,
                         const Dense2DShape& shape,
                         double dc,
                         double nyquist) {
  const int64_t total = static_cast<int64_t>(shape.n0) * shape.n1;
  const int half_n1 = shape.n1 / 2 + 1;
  const double tolerance = std::max(1.0e-3, 1.0e-4 * static_cast<double>(total));
  const auto at = [&](int k0, int k1) -> const flagfftComplex& {
    return spectrum[static_cast<std::size_t>(k0) * half_n1 + k1];
  };

  ASSERT_EQ(spectrum.size(), static_cast<std::size_t>(shape.n0) * half_n1);
  EXPECT_NEAR(at(0, 0).x, dc * total, tolerance);
  EXPECT_NEAR(at(0, 0).y, 0.0, tolerance);
  EXPECT_NEAR(at(shape.n0 / 2, shape.n1 / 2).x, nyquist * total, tolerance);
  EXPECT_NEAR(at(shape.n0 / 2, shape.n1 / 2).y, 0.0, tolerance);

  for (int k0 = 0; k0 < shape.n0; ++k0) {
    for (int k1 = 0; k1 < half_n1; ++k1) {
      if ((k0 == 0 && k1 == 0) || (k0 == shape.n0 / 2 && k1 == shape.n1 / 2)) {
        continue;
      }
      EXPECT_NEAR(at(k0, k1).x, 0.0, tolerance);
      EXPECT_NEAR(at(k0, k1).y, 0.0, tolerance);
    }
  }
}

void AssertRoundtrip(const std::vector<flagfftReal>& output,
                     const std::vector<flagfftReal>& input,
                     const Dense2DShape& shape) {
  const int64_t total = static_cast<int64_t>(shape.n0) * shape.n1;
  const double tolerance = std::max(1.0e-3, 2.0e-4 * static_cast<double>(total));
  ASSERT_EQ(output.size(), input.size());
  ASSERT_EQ(output.size(), static_cast<std::size_t>(total));
  for (std::size_t i = 0; i < output.size(); ++i) {
    EXPECT_NEAR(output[i], static_cast<double>(input[i]) * total, tolerance);
  }
}

void MakeRealPlans(const Dense2DShape& shape,
                   int batch,
                   PlanGuard& forward,
                   PlanGuard& inverse) {
  int n[2] = {shape.n0, shape.n1};
  const int total = shape.n0 * shape.n1;
  const int half = shape.n0 * (shape.n1 / 2 + 1);
  ASSERT_EQ(flagfftPlanMany(&forward.plan,
                            2,
                            n,
                            nullptr,
                            1,
                            total,
                            nullptr,
                            1,
                            half,
                            FLAGFFT_R2C,
                            batch),
            FLAGFFT_SUCCESS);
  ASSERT_EQ(flagfftPlanMany(&inverse.plan,
                            2,
                            n,
                            nullptr,
                            1,
                            half,
                            nullptr,
                            1,
                            total,
                            FLAGFFT_C2R,
                            batch),
            FLAGFFT_SUCCESS);
}

void RunOutOfPlace(const Dense2DShape& shape) {
  const int total = shape.n0 * shape.n1;
  const int half = shape.n0 * (shape.n1 / 2 + 1);
  const auto input = BoundaryInput(shape);
  PlanGuard forward;
  PlanGuard inverse;
  MakeRealPlans(shape, 1, forward, inverse);
  AssertRowBoundaryDescription(forward.plan, inverse.plan, shape);
  if (shape.n1 >= 16 && shape.n1 <= 128) {
    AssertShortRealBoundaryPack4Description(forward.plan, inverse.plan, shape);
  }

  flagfft::adaptor::Memory real_mem(static_cast<std::size_t>(total) * sizeof(flagfftReal));
  flagfft::adaptor::Memory spectrum_mem(static_cast<std::size_t>(half) * sizeof(flagfftComplex));
  flagfft::adaptor::Memory output_mem(static_cast<std::size_t>(total) * sizeof(flagfftReal));
  real_mem.copy_from_host(input.data(), input.size() * sizeof(flagfftReal));

  ASSERT_EQ(flagfftExecR2C(forward.plan,
                           static_cast<flagfftReal*>(real_mem.data()),
                           static_cast<flagfftComplex*>(spectrum_mem.data())),
            FLAGFFT_SUCCESS);
  flagfft::adaptor::synchronize();
  std::vector<flagfftComplex> spectrum(static_cast<std::size_t>(half));
  spectrum_mem.copy_to_host(spectrum.data(), spectrum.size() * sizeof(flagfftComplex));
  AssertR2CBoundaries(spectrum, shape, 1.25, -0.5);

  ASSERT_EQ(flagfftExecC2R(inverse.plan,
                           static_cast<flagfftComplex*>(spectrum_mem.data()),
                           static_cast<flagfftReal*>(output_mem.data())),
            FLAGFFT_SUCCESS);
  flagfft::adaptor::synchronize();
  std::vector<flagfftReal> output(static_cast<std::size_t>(total));
  output_mem.copy_to_host(output.data(), output.size() * sizeof(flagfftReal));
  AssertRoundtrip(output, input, shape);
}

void RunInPlace(const Dense2DShape& shape) {
  const int total = shape.n0 * shape.n1;
  const int half = shape.n0 * (shape.n1 / 2 + 1);
  const auto input = BoundaryInput(shape);
  PlanGuard forward;
  PlanGuard inverse;
  MakeRealPlans(shape, 1, forward, inverse);
  AssertRowBoundaryDescription(forward.plan, inverse.plan, shape);
  if (shape.n1 >= 16 && shape.n1 <= 128) {
    AssertShortRealBoundaryPack4Description(forward.plan, inverse.plan, shape);
  }

  const std::size_t real_bytes = static_cast<std::size_t>(total) * sizeof(flagfftReal);
  const std::size_t compact_bytes = static_cast<std::size_t>(half) * sizeof(flagfftComplex);
  flagfft::adaptor::Memory in_place(std::max(real_bytes, compact_bytes));
  in_place.copy_from_host(input.data(), real_bytes);

  ASSERT_EQ(flagfftExecR2C(forward.plan,
                           static_cast<flagfftReal*>(in_place.data()),
                           static_cast<flagfftComplex*>(in_place.data())),
            FLAGFFT_SUCCESS);
  flagfft::adaptor::synchronize();
  std::vector<flagfftComplex> spectrum(static_cast<std::size_t>(half));
  in_place.copy_to_host(spectrum.data(), compact_bytes);
  AssertR2CBoundaries(spectrum, shape, 1.25, -0.5);

  ASSERT_EQ(flagfftExecC2R(inverse.plan,
                           static_cast<flagfftComplex*>(in_place.data()),
                           static_cast<flagfftReal*>(in_place.data())),
            FLAGFFT_SUCCESS);
  flagfft::adaptor::synchronize();
  std::vector<flagfftReal> output(static_cast<std::size_t>(total));
  in_place.copy_to_host(output.data(), real_bytes);
  AssertRoundtrip(output, input, shape);
}

void ExpectFallbackDescription(flagfftType type,
                               const Dense2DShape& shape,
                               int batch,
                               const char* node_name,
                               int idist,
                               int odist) {
  int n[2] = {shape.n0, shape.n1};
  PlanGuard plan;
  ASSERT_EQ(flagfftPlanMany(&plan.plan,
                            2,
                            n,
                            nullptr,
                            1,
                            idist,
                            nullptr,
                            1,
                            odist,
                            type,
                            batch),
            FLAGFFT_SUCCESS);
  const char* raw_description = flagfftGetPlanDescription(plan.plan);
  ASSERT_NE(raw_description, nullptr);
  const std::string description(raw_description);
  EXPECT_NE(description.find(node_name), std::string::npos) << description;
  EXPECT_EQ(description.find("CompiledRaw2DR2CRow("), std::string::npos) << description;
  EXPECT_EQ(description.find("CompiledRaw2DC2RRow("), std::string::npos) << description;
}

class Maca2DRealRows : public ::testing::TestWithParam<Dense2DShape> {
 protected:
  void SetUp() override {
    if (!IsUsableMaca()) {
      GTEST_SKIP() << "MACA-only real-row C API test requires a usable MACA device";
    }
  }
};

TEST_P(Maca2DRealRows, DenseOutOfPlaceBoundaryAndRoundtrip) {
  RunOutOfPlace(GetParam());
}

TEST_P(Maca2DRealRows, DenseInPlaceBoundaryAndRoundtrip) {
  RunInPlace(GetParam());
}

INSTANTIATE_TEST_SUITE_P(Dense,
                         Maca2DRealRows,
                         ::testing::Values(Dense2DShape {512, 1024},
                                           Dense2DShape {512, 128},
                                           Dense2DShape {512, 64}),
                         [](const auto& info) {
                           return std::to_string(info.param.n0) + "x" + std::to_string(info.param.n1);
                         });

void ExpectUnitAxisDescription(flagfftType type,
                               const Dense2DShape& shape,
                               int idist,
                               int odist,
                               const char* forbidden_row_node) {
  int n[2] = {shape.n0, shape.n1};
  PlanGuard plan;
  ASSERT_EQ(flagfftPlanMany(&plan.plan,
                            2,
                            n,
                            nullptr,
                            1,
                            idist,
                            nullptr,
                            1,
                            odist,
                            type,
                            1),
            FLAGFFT_SUCCESS);
  const char* raw_description = flagfftGetPlanDescription(plan.plan);
  ASSERT_NE(raw_description, nullptr);
  const std::string description(raw_description);
  EXPECT_NE(description.find("CompiledRaw1DAs2D("), std::string::npos) << description;
  EXPECT_EQ(description.find(forbidden_row_node), std::string::npos) << description;
  EXPECT_EQ(description.find("batch_per_block=4"), std::string::npos) << description;
}

TEST(Maca2DRealRowsFallback, FP64AndBatch2KeepTheExistingNodes) {
  if (!IsUsableMaca()) {
    GTEST_SKIP() << "MACA-only real-row C API test requires a usable MACA device";
  }
  const Dense2DShape shape {512, 64};
  const int total = shape.n0 * shape.n1;
  const int half = shape.n0 * (shape.n1 / 2 + 1);

  ExpectFallbackDescription(FLAGFFT_D2Z, shape, 1, "CompiledRaw2DR2C(", total, half);
  ExpectFallbackDescription(FLAGFFT_Z2D, shape, 1, "CompiledRaw2DC2R(", half, total);
  ExpectFallbackDescription(FLAGFFT_R2C, shape, 2, "CompiledRaw2DR2C(", total, half);
  ExpectFallbackDescription(FLAGFFT_C2R, shape, 2, "CompiledRaw2DC2R(", half, total);
}

TEST(Maca2DRealRowsFallback, UnitAxisAndOddWidthDoNotUseRealRows) {
  if (!IsUsableMaca()) {
    GTEST_SKIP() << "MACA-only real-row C API test requires a usable MACA device";
  }

  const Dense2DShape unit_axis {1, 64};
  ExpectUnitAxisDescription(FLAGFFT_R2C,
                            unit_axis,
                            unit_axis.n0 * unit_axis.n1,
                            unit_axis.n0 * (unit_axis.n1 / 2 + 1),
                            "CompiledRaw2DR2CRow(");
  ExpectUnitAxisDescription(FLAGFFT_C2R,
                            unit_axis,
                            unit_axis.n0 * (unit_axis.n1 / 2 + 1),
                            unit_axis.n0 * unit_axis.n1,
                            "CompiledRaw2DC2RRow(");

  const Dense2DShape odd_width {512, 65};
  const int total = odd_width.n0 * odd_width.n1;
  const int half = odd_width.n0 * (odd_width.n1 / 2 + 1);
  ExpectFallbackDescription(FLAGFFT_R2C, odd_width, 1, "CompiledRaw2DR2C(", total, half);
  ExpectFallbackDescription(FLAGFFT_C2R, odd_width, 1, "CompiledRaw2DC2R(", half, total);
}

}  // namespace
