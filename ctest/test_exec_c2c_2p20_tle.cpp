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

using namespace flagfft_test;

namespace {

void ExpectLarge1DMatchesReference(int n, int direction) {
  constexpr int batch = 1;
  const int total = n * batch;
  const std::size_t bytes = static_cast<std::size_t>(total) * sizeof(flagfftComplex);

  flagfftHandle plan = nullptr;
  Plan1d(&plan, n, FLAGFFT_C2C, batch);

  RefPlanHandle reference_plan;
  ref_plan_1d(reference_plan, n, FLAGFFT_C2C, batch);

  auto input = random_complex(total, accuracy_seed(FLAGFFT_C2C, n, batch));
  std::vector<flagfftComplex> output(total);
  std::vector<flagfftComplex> reference(total);

  flagfft::adaptor::Memory input_memory(bytes);
  flagfft::adaptor::Memory output_memory(bytes);
  flagfft::adaptor::Memory reference_memory(bytes);
  auto* device_input = static_cast<flagfftComplex*>(input_memory.data());
  auto* device_output = static_cast<flagfftComplex*>(output_memory.data());
  auto* device_reference = static_cast<flagfftComplex*>(reference_memory.data());

  for (double scale : kAccuracyInputScales) {
    auto scaled_input = input;
    scale_input(scaled_input, scale);
    input_memory.copy_from_host(scaled_input.data(), bytes);
    ExecC2C(plan, device_input, device_output, direction);
    ref_exec_c2c(reference_plan, device_input, device_reference, direction);
    output_memory.copy_to_host(output.data(), bytes);
    reference_memory.copy_to_host(reference.data(), bytes);
    expect_reference_accuracy(error_stats(output.data(), reference.data(), n, batch),
                              FLAGFFT_C2C,
                              n,
                              batch,
                              input_scale_name(scale));
  }

  EXPECT_EQ(flagfftDestroy(plan), FLAGFFT_SUCCESS);
}

}  // namespace

TEST(C2C2P20Tle, ForwardVsReference) {
  ExpectLarge1DMatchesReference(1 << 20, FLAGFFT_FORWARD);
}

TEST(C2C2P20Tle, InverseVsReference) {
  ExpectLarge1DMatchesReference(1 << 20, FLAGFFT_INVERSE);
}

class C2CLargeMixedTle : public ::testing::TestWithParam<int> {};

TEST_P(C2CLargeMixedTle, ForwardVsReference) {
  ExpectLarge1DMatchesReference(GetParam(), FLAGFFT_FORWARD);
}

TEST_P(C2CLargeMixedTle, InverseVsReference) {
  ExpectLarge1DMatchesReference(GetParam(), FLAGFFT_INVERSE);
}

std::string MixedSizeName(const ::testing::TestParamInfo<int>& info) {
  if (info.param == 9 * (1 << 16)) {
    return "Radix3Squared";
  }
  if (info.param == 3 * (1 << 18)) {
    return "Radix3";
  }
  if (info.param == 5 * (1 << 17)) {
    return "Radix5";
  }
  if (info.param == 25 * (1 << 15)) {
    return "Radix5Squared";
  }
  if (info.param == 27 * (1 << 15)) {
    return "Radix3Cubed";
  }
  if (info.param == 7 * (1 << 17)) {
    return "Radix7";
  }
  return "Radix3Times5";
}

INSTANTIATE_TEST_SUITE_P(LargeMixedBases,
                         C2CLargeMixedTle,
                         ::testing::Values(9 * (1 << 16),
                                           3 * (1 << 18),
                                           5 * (1 << 17),
                                           25 * (1 << 15),
                                           27 * (1 << 15),
                                           7 * (1 << 17),
                                           15 * (1 << 16)),
                         MixedSizeName);
