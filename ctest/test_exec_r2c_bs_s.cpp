#include "flagfft_test.h"

using namespace flagfft_test;

class R2CBSSingle_Test : public ::testing::TestWithParam<Test1DParam> {
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

TEST_P(R2CBSSingle_Test, ForwardVsReference) {
  RefPlanHandle ref;
  ref_plan_1d(ref, N, FLAGFFT_R2C, batch);
  std::vector<flagfftComplex> h_out(total_out);
  std::vector<flagfftComplex> h_ref_out(total_out);
  for (double scale : kAccuracyInputScales) {
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

INSTANTIATE_TEST_SUITE_P(Smoke,
                         R2CBSSingle_Test,
                         ::testing::ValuesIn(Generate1DParamsBSSmokeSingle()),
                         [](const auto& info) {
                           return std::to_string(info.param.N) + "x" + std::to_string(info.param.batch);
                         });
INSTANTIATE_TEST_SUITE_P(ExtendedSmall,
                         R2CBSSingle_Test,
                         ::testing::ValuesIn(Generate1DParamsBSExtendedSmallSingle()),
                         [](const auto& info) {
                           return std::to_string(info.param.N) + "x" + std::to_string(info.param.batch);
                         });
INSTANTIATE_TEST_SUITE_P(ExtendedMedium,
                         R2CBSSingle_Test,
                         ::testing::ValuesIn(Generate1DParamsBSExtendedMediumSingle()),
                         [](const auto& info) {
                           return std::to_string(info.param.N) + "x" + std::to_string(info.param.batch);
                         });
INSTANTIATE_TEST_SUITE_P(ExtendedLarge,
                         R2CBSSingle_Test,
                         ::testing::ValuesIn(Generate1DParamsBSExtendedLargeSingle()),
                         [](const auto& info) {
                           return std::to_string(info.param.N) + "x" + std::to_string(info.param.batch);
                         });
