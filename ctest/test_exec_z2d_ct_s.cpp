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

class Z2DCTSingle_Test : public ::testing::TestWithParam<Test1DParam> {
 protected:
  void SetUp() override {
    auto p = GetParam();
    N = p.N;
    batch = p.batch;
    total_in = (N / 2 + 1) * batch;
    total_out = N * batch;

    plan = nullptr;
    Plan1d(&plan, N, FLAGFFT_Z2D, batch);

    h_in = random_double_complex(total_in, accuracy_seed(FLAGFFT_Z2D, N, batch));

    for (int b = 0; b < batch; ++b) {
      h_in[b * (N / 2 + 1) + 0].y = 0.0;
      if (N % 2 == 0) h_in[b * (N / 2 + 1) + N / 2].y = 0.0;
    }

    in_memory.allocate(total_in * sizeof(flagfftDoubleComplex));
    out_memory.allocate(total_out * sizeof(flagfftDoubleReal));
    ref_memory.allocate(total_out * sizeof(flagfftDoubleReal));
    d_in = static_cast<flagfftDoubleComplex*>(in_memory.data());
    d_out = static_cast<flagfftDoubleReal*>(out_memory.data());
    d_ref = static_cast<flagfftDoubleReal*>(ref_memory.data());
    ASSERT_NE(d_in, nullptr);
    ASSERT_NE(d_out, nullptr);
    ASSERT_NE(d_ref, nullptr);

    in_memory.copy_from_host(h_in.data(), total_in * sizeof(flagfftDoubleComplex));
  }

  void TearDown() override {
    if (plan) flagfftDestroy(plan);
  }

  int N = 0;
  int batch = 0;
  int total_in = 0;
  int total_out = 0;
  flagfftHandle plan = nullptr;
  std::vector<flagfftDoubleComplex> h_in;
  flagfft::adaptor::Memory in_memory;
  flagfft::adaptor::Memory out_memory;
  flagfft::adaptor::Memory ref_memory;
  flagfftDoubleComplex* d_in = nullptr;
  flagfftDoubleReal* d_out = nullptr;
  flagfftDoubleReal* d_ref = nullptr;
};

TEST_P(Z2DCTSingle_Test, InverseVsReference) {
  RefPlanHandle ref;
  ref_plan_1d(ref, N, FLAGFFT_Z2D, batch);
  std::vector<flagfftDoubleReal> h_out(total_out);
  std::vector<flagfftDoubleReal> h_ref_out(total_out);
  for (double scale : filter_scales()) {
    auto input = h_in;
    scale_input(input, scale);
    in_memory.copy_from_host(input.data(), total_in * sizeof(flagfftDoubleComplex));
    ExecZ2D(plan, d_in, d_out);
    ref_exec_z2d(ref, d_in, d_ref);
    out_memory.copy_to_host(h_out.data(), total_out * sizeof(flagfftDoubleReal));
    ref_memory.copy_to_host(h_ref_out.data(), total_out * sizeof(flagfftDoubleReal));
    expect_reference_accuracy(error_stats(h_out.data(), h_ref_out.data(), N, batch),
                              FLAGFFT_Z2D,
                              N,
                              batch,
                              input_scale_name(scale));
  }
}

TEST(SmokeZ2DAccuracy, ZeroInputIsExact) {
  constexpr int kN = 256;
  constexpr int kInputCount = kN / 2 + 1;
  flagfftHandle plan = nullptr;
  Plan1d(&plan, kN, FLAGFFT_Z2D);
  std::vector<flagfftDoubleComplex> h_in(kInputCount, {0.0, 0.0});
  std::vector<flagfftDoubleReal> h_out(kN);
  flagfft::adaptor::Memory in_memory(kInputCount * sizeof(flagfftDoubleComplex));
  flagfft::adaptor::Memory out_memory(kN * sizeof(flagfftDoubleReal));
  auto* d_in = static_cast<flagfftDoubleComplex*>(in_memory.data());
  auto* d_out = static_cast<flagfftDoubleReal*>(out_memory.data());
  ASSERT_NE(d_in, nullptr);
  ASSERT_NE(d_out, nullptr);
  in_memory.copy_from_host(h_in.data(), kInputCount * sizeof(flagfftDoubleComplex));
  ExecZ2D(plan, d_in, d_out);
  out_memory.copy_to_host(h_out.data(), kN * sizeof(flagfftDoubleReal));
  for (flagfftDoubleReal value : h_out) {
    EXPECT_EQ(value, 0.0);
    EXPECT_TRUE(std::isfinite(value));
  }
  EXPECT_EQ(flagfftDestroy(plan), FLAGFFT_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(All,
                         Z2DCTSingle_Test,
                         ::testing::ValuesIn(override_params(Generate1DParamsCTAllSingle())),
                         [](const auto& info) {
                           return std::to_string(info.param.N) + "x" + std::to_string(info.param.batch);
                         });
