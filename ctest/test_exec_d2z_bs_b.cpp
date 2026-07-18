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

class D2ZBSBatch_Test : public ::testing::TestWithParam<Test1DParam> {
 protected:
  void SetUp() override {
    auto p = GetParam();
    N = p.N;
    batch = p.batch;
    total_in = N * batch;
    total_out = (N / 2 + 1) * batch;

    plan = nullptr;
    Plan1d(&plan, N, FLAGFFT_D2Z, batch);

    h_in = random_double_real(total_in, accuracy_seed(FLAGFFT_D2Z, N, batch));

    in_memory.allocate(total_in * sizeof(flagfftDoubleReal));
    out_memory.allocate(total_out * sizeof(flagfftDoubleComplex));
    ref_memory.allocate(total_out * sizeof(flagfftDoubleComplex));
    d_in = static_cast<flagfftDoubleReal*>(in_memory.data());
    d_out = static_cast<flagfftDoubleComplex*>(out_memory.data());
    d_ref = static_cast<flagfftDoubleComplex*>(ref_memory.data());
    ASSERT_NE(d_in, nullptr);
    ASSERT_NE(d_out, nullptr);
    ASSERT_NE(d_ref, nullptr);

    in_memory.copy_from_host(h_in.data(), total_in * sizeof(flagfftDoubleReal));
  }

  void TearDown() override {
    if (plan) flagfftDestroy(plan);
  }

  int N = 0;
  int batch = 0;
  int total_in = 0;
  int total_out = 0;
  flagfftHandle plan = nullptr;
  std::vector<flagfftDoubleReal> h_in;
  flagfft::adaptor::Memory in_memory;
  flagfft::adaptor::Memory out_memory;
  flagfft::adaptor::Memory ref_memory;
  flagfftDoubleReal* d_in = nullptr;
  flagfftDoubleComplex* d_out = nullptr;
  flagfftDoubleComplex* d_ref = nullptr;
};

TEST_P(D2ZBSBatch_Test, ForwardVsReference) {
  RefPlanHandle ref;
  ref_plan_1d(ref, N, FLAGFFT_D2Z, batch);
  std::vector<flagfftDoubleComplex> h_out(total_out);
  std::vector<flagfftDoubleComplex> h_ref_out(total_out);
  for (double scale : filter_scales()) {
    auto input = h_in;
    scale_input(input, scale);
    in_memory.copy_from_host(input.data(), total_in * sizeof(flagfftDoubleReal));
    ExecD2Z(plan, d_in, d_out);
    ref_exec_d2z(ref, d_in, d_ref);
    out_memory.copy_to_host(h_out.data(), total_out * sizeof(flagfftDoubleComplex));
    ref_memory.copy_to_host(h_ref_out.data(), total_out * sizeof(flagfftDoubleComplex));
    expect_reference_accuracy(error_stats(h_out.data(), h_ref_out.data(), N / 2 + 1, batch),
                              FLAGFFT_D2Z,
                              N,
                              batch,
                              input_scale_name(scale));
  }
}

INSTANTIATE_TEST_SUITE_P(All,
                         D2ZBSBatch_Test,
                         ::testing::ValuesIn(override_params(Generate1DParamsBSExtendedSmallBatch())),
                         [](const auto& info) {
                           return std::to_string(info.param.N) + "x" + std::to_string(info.param.batch);
                         });
