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

class R2CCTSingle_Test : public ::testing::TestWithParam<Test1DParam> {
 protected:
  void SetUp() override {
    auto p = GetParam();
    N = p.N;
    batch = p.batch;
    total_in = N * batch;
    total_out = (N / 2 + 1) * batch;

    plan = nullptr;
    Plan1d(&plan, N, FLAGFFT_R2C, batch);

    h_in = random_real(total_in, accuracy_seed(FLAGFFT_R2C, N, batch));

    in_memory.allocate(total_in * sizeof(flagfftReal));
    out_memory.allocate(total_out * sizeof(flagfftComplex));
    ref_memory.allocate(total_out * sizeof(flagfftComplex));
    d_in = static_cast<flagfftReal*>(in_memory.data());
    d_out = static_cast<flagfftComplex*>(out_memory.data());
    d_ref = static_cast<flagfftComplex*>(ref_memory.data());
    ASSERT_NE(d_in, nullptr);
    ASSERT_NE(d_out, nullptr);
    ASSERT_NE(d_ref, nullptr);

    in_memory.copy_from_host(h_in.data(), total_in * sizeof(flagfftReal));
  }

  void TearDown() override {
    if (plan) flagfftDestroy(plan);
  }

  int N = 0;
  int batch = 0;
  int total_in = 0;
  int total_out = 0;
  flagfftHandle plan = nullptr;
  std::vector<flagfftReal> h_in;
  flagfft::adaptor::Memory in_memory;
  flagfft::adaptor::Memory out_memory;
  flagfft::adaptor::Memory ref_memory;
  flagfftReal* d_in = nullptr;
  flagfftComplex* d_out = nullptr;
  flagfftComplex* d_ref = nullptr;
};

TEST_P(R2CCTSingle_Test, ForwardVsReference) {
  RefPlanHandle ref;
  ref_plan_1d(ref, N, FLAGFFT_R2C, batch);
  std::vector<flagfftComplex> h_out(total_out);
  std::vector<flagfftComplex> h_ref_out(total_out);
  for (double scale : filter_scales()) {
    auto input = h_in;
    scale_input(input, scale);
    in_memory.copy_from_host(input.data(), total_in * sizeof(flagfftReal));
    ExecR2C(plan, d_in, d_out);
    ref_exec_r2c(ref, d_in, d_ref);
    out_memory.copy_to_host(h_out.data(), total_out * sizeof(flagfftComplex));
    ref_memory.copy_to_host(h_ref_out.data(), total_out * sizeof(flagfftComplex));
    expect_reference_accuracy(error_stats(h_out.data(), h_ref_out.data(), N / 2 + 1, batch),
                              FLAGFFT_R2C,
                              N,
                              batch,
                              input_scale_name(scale));
  }
}

INSTANTIATE_TEST_SUITE_P(All,
                         R2CCTSingle_Test,
                         ::testing::ValuesIn(override_params(Generate1DParamsCTAllSingle())),
                         [](const auto& info) {
                           return std::to_string(info.param.N) + "x" + std::to_string(info.param.batch);
                         });
