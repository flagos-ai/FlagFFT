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
struct PackedSetting {
  std::string previous;
  bool existed;
  PackedSetting() : existed(std::getenv("FLAGFFT_PACKED_REAL") != nullptr) {
    if (existed) previous = std::getenv("FLAGFFT_PACKED_REAL");
    setenv("FLAGFFT_PACKED_REAL", "1", 1);
  }
  ~PackedSetting() {
    if (existed) setenv("FLAGFFT_PACKED_REAL", previous.c_str(), 1);
    else unsetenv("FLAGFFT_PACKED_REAL");
  }
};
}

// Different data in every batch detects accidental batch-0 reuse. Compare
// each layout with a dense platform transform, including Nyquist padding.
TEST(PackedRealBatch, DensePaddedAndInPlaceMatchReference) {
  PackedSetting setting;
  const int n = g_test_params.nx > 0 ? g_test_params.nx : 65536;
  const int batch = g_test_params.batch > 0 ? g_test_params.batch : 4;
  const int half = n / 2 + 1;
  auto input = random_double_real(n * batch, 3127);
  flagfft::adaptor::Memory dense_in(input.size() * sizeof(double));
  flagfft::adaptor::Memory dense_spectrum(half * batch * sizeof(flagfftDoubleComplex));
  flagfft::adaptor::Memory dense_inverse(input.size() * sizeof(double));
  dense_in.copy_from_host(input.data(), input.size() * sizeof(double));
  RefPlanHandle fwd, inv;
  ref_plan_1d(fwd, n, FLAGFFT_D2Z, batch);
  ref_plan_1d(inv, n, FLAGFFT_Z2D, batch);
  ref_exec_d2z(fwd, static_cast<double*>(dense_in.data()),
               static_cast<flagfftDoubleComplex*>(dense_spectrum.data()));
  std::vector<flagfftDoubleComplex> reference(half * batch);
  dense_spectrum.copy_to_host(reference.data(), reference.size() * sizeof(reference[0]));
  ref_exec_z2d(inv, static_cast<flagfftDoubleComplex*>(dense_spectrum.data()),
               static_cast<double*>(dense_inverse.data()));
  std::vector<double> inverse_reference(n * batch);
  dense_inverse.copy_to_host(inverse_reference.data(), inverse_reference.size() * sizeof(double));

  flagfftHandle unsupported = nullptr;
  int shape_for_rejection = n;
  EXPECT_EQ(flagfftPlanMany(&unsupported, 1, &shape_for_rejection, nullptr, 1, n+6,
                           nullptr, 1, half+3, FLAGFFT_D2Z, batch), FLAGFFT_NOT_SUPPORTED);
  EXPECT_EQ(unsupported, nullptr);

  for (int layout = 0; layout < 3; ++layout) {
    SCOPED_TRACE(layout);
    const bool in_place = layout == 2;
    const int real_distance = layout == 0 ? n : 2 * half;
    const int complex_distance = half;
    std::vector<double> real_host(real_distance * batch, -123.0);
    std::vector<flagfftDoubleComplex> complex_host(complex_distance * batch);
    for (int b = 0; b < batch; ++b) {
      std::copy_n(input.data() + b*n, n, real_host.data() + b*real_distance);
      std::copy_n(reference.data() + b*half, half, complex_host.data() + b*complex_distance);
    }
    flagfft::adaptor::Memory real(real_host.size() * sizeof(double));
    flagfft::adaptor::Memory spectrum(complex_host.size() * sizeof(complex_host[0]));
    auto real_ptr = static_cast<double*>(real.data());
    auto complex_ptr = in_place ? reinterpret_cast<flagfftDoubleComplex*>(real_ptr)
                                : static_cast<flagfftDoubleComplex*>(spectrum.data());
    flagfftHandle plan = nullptr;
    int shape = n;
    ASSERT_EQ(flagfftPlanMany(&plan, 1, &shape, nullptr, 1, real_distance, nullptr, 1,
                              complex_distance, FLAGFFT_D2Z, batch), FLAGFFT_SUCCESS);
    EXPECT_NE(std::string(flagfftGetPlanDescription(plan)).find("CompiledRawPackedR2C"), std::string::npos);
    real.copy_from_host(real_host.data(), real_host.size() * sizeof(double));
    ASSERT_EQ(flagfftExecD2Z(plan, real_ptr, complex_ptr), FLAGFFT_SUCCESS);
    (in_place ? real : spectrum).copy_to_host(complex_host.data(), complex_host.size() * sizeof(complex_host[0]));
    for (int b = 0; b < batch; ++b) {
      expect_reference_accuracy(error_stats(complex_host.data()+b*complex_distance, reference.data()+b*half, half, 1),
                                FLAGFFT_D2Z, n, 1, "batch-layout");
    }
    ASSERT_EQ(flagfftDestroy(plan), FLAGFFT_SUCCESS);

    for (int b = 0; b < batch; ++b)
      std::copy_n(reference.data()+b*half, half, complex_host.data()+b*complex_distance);
    (in_place ? real : spectrum).copy_from_host(complex_host.data(), complex_host.size()*sizeof(complex_host[0]));
    ASSERT_EQ(flagfftPlanMany(&plan, 1, &shape, nullptr, 1, complex_distance, nullptr, 1,
                              real_distance, FLAGFFT_Z2D, batch), FLAGFFT_SUCCESS);
    EXPECT_NE(std::string(flagfftGetPlanDescription(plan)).find("CompiledRawPackedC2R"), std::string::npos);
    ASSERT_EQ(flagfftExecZ2D(plan, complex_ptr, real_ptr), FLAGFFT_SUCCESS);
    real.copy_to_host(real_host.data(), real_host.size()*sizeof(double));
    for (int b = 0; b < batch; ++b) {
      expect_reference_accuracy(error_stats(real_host.data()+b*real_distance, inverse_reference.data()+b*n, n, 1),
                                FLAGFFT_Z2D, n, 1, "batch-layout");
    }
    ASSERT_EQ(flagfftDestroy(plan), FLAGFFT_SUCCESS);
  }
}
