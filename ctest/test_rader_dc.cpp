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

TEST(RaderDC, InPlaceAndSeparateBuffersMatchReference) {
  const int n = g_test_params.nx > 0 ? g_test_params.nx : 16381;
  const int batch = g_test_params.batch > 0 ? g_test_params.batch : 4;
  const std::size_t count = static_cast<std::size_t>(n) * batch;
  std::vector<flagfftDoubleComplex> input(count), expected(count), actual(count);
  for (std::size_t i = 0; i < count; ++i) {
    // Nonzero and different DC in each batch makes a missing x[0] or a
    // batch-stride error visible in the DC bin as well as the whole FFT.
    input[i] = {std::sin(i * 0.071) + 0.1 * (i/n + 1), std::cos(i * 0.039)};
  }
  flagfft::adaptor::Memory in(count * sizeof(input[0])), out(count * sizeof(input[0]));
  flagfft::adaptor::Memory ref_in(count * sizeof(input[0])), ref_out(count * sizeof(input[0]));
  RefPlanHandle ref;
  ref_plan_1d(ref, n, FLAGFFT_Z2Z, batch);
  flagfftHandle plan = nullptr;
  ASSERT_EQ(flagfftPlan1d(&plan, n, FLAGFFT_Z2Z, batch), FLAGFFT_SUCCESS);
  ASSERT_NE(std::string(flagfftGetPlanDescription(plan)).find("CompiledRawRader"), std::string::npos);
  for (int direction : {FLAGFFT_FORWARD, FLAGFFT_INVERSE}) {
    ref_in.copy_from_host(input.data(), count * sizeof(input[0]));
    ref_exec_z2z(ref, static_cast<flagfftDoubleComplex*>(ref_in.data()),
                 static_cast<flagfftDoubleComplex*>(ref_out.data()), direction);
    ref_out.copy_to_host(expected.data(), count * sizeof(input[0]));
    for (bool in_place : {false, true}) {
      SCOPED_TRACE(direction);
      SCOPED_TRACE(in_place);
      in.copy_from_host(input.data(), count * sizeof(input[0]));
      auto *result = static_cast<flagfftDoubleComplex*>(in_place ? in.data() : out.data());
      ASSERT_EQ(flagfftExecZ2Z(plan, static_cast<flagfftDoubleComplex*>(in.data()), result, direction), FLAGFFT_SUCCESS);
      (in_place ? in : out).copy_to_host(actual.data(), count * sizeof(input[0]));
      expect_reference_accuracy(error_stats(actual.data(), expected.data(), n, batch),
                                FLAGFFT_Z2Z, n, batch, "Rader DC reuse");
      for (int b = 0; b < batch; ++b) {
        EXPECT_NEAR(actual[b*n].x, expected[b*n].x, 1e-10 * n);
        EXPECT_NEAR(actual[b*n].y, expected[b*n].y, 1e-10 * n);
      }
    }
  }
  ASSERT_EQ(flagfftDestroy(plan), FLAGFFT_SUCCESS);
}
