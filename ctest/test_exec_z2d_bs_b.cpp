#include "flagfft_test.h"

using namespace flagfft_test;

class Z2DBSBatch_Test : public ::testing::TestWithParam<Test1DParam> {
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

TEST_P(Z2DBSBatch_Test, InverseVsReference) {
  RefPlanHandle ref;
  ref_plan_1d(ref, N, FLAGFFT_Z2D, batch);
  std::vector<flagfftDoubleReal> h_out(total_out);
  std::vector<flagfftDoubleReal> h_ref_out(total_out);
  for (double scale : kAccuracyInputScales) {
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

INSTANTIATE_TEST_SUITE_P(All,
                         Z2DBSBatch_Test,
                         ::testing::ValuesIn(override_params(Generate1DParamsBSExtendedSmallBatch())),
                         [](const auto& info) {
                           return std::to_string(info.param.N) + "x" + std::to_string(info.param.batch);
                         });
