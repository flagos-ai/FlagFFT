#include "flagfft_test.h"

using namespace flagfft_test::adaptor;

class R2C_1D_Test : public ::testing::TestWithParam<Test1DParam> {
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

    d_in = static_cast<flagfftReal*>(allocate_device(total_in * sizeof(flagfftReal)));
    d_out = static_cast<flagfftComplex*>(allocate_device(total_out * sizeof(flagfftComplex)));
    d_ref = static_cast<flagfftComplex*>(allocate_device(total_out * sizeof(flagfftComplex)));
    ASSERT_NE(d_in, nullptr);
    ASSERT_NE(d_out, nullptr);
    ASSERT_NE(d_ref, nullptr);

    copy_host_to_device(h_in.data(), d_in, total_in * sizeof(flagfftReal));
  }

  void TearDown() override {
    if (d_in) free_device(d_in);
    if (d_out) free_device(d_out);
    if (d_ref) free_device(d_ref);
    if (plan) flagfftDestroy(plan);
  }

  int N = 0;
  int batch = 0;
  int total_in = 0;
  int total_out = 0;
  flagfftHandle plan = nullptr;
  std::vector<flagfftReal> h_in;
  flagfftReal* d_in = nullptr;
  flagfftComplex* d_out = nullptr;
  flagfftComplex* d_ref = nullptr;
};

TEST_P(R2C_1D_Test, ForwardVsReference) {
  RefHandle ref;
  ref_plan_1d(ref, N, FLAGFFT_R2C, batch);
  std::vector<flagfftComplex> h_out(total_out);
  std::vector<flagfftComplex> h_ref_out(total_out);
  for (double scale : kAccuracyInputScales) {
    auto input = h_in;
    scale_input(input, scale);
    copy_host_to_device(input.data(), d_in, total_in * sizeof(flagfftReal));
    ExecR2C(plan, d_in, d_out);
    ref_exec_r2c(ref, d_in, d_ref);
    copy_device_to_host(d_out, h_out.data(), total_out * sizeof(flagfftComplex));
    copy_device_to_host(d_ref, h_ref_out.data(), total_out * sizeof(flagfftComplex));
    expect_reference_accuracy(error_stats(h_out.data(), h_ref_out.data(), N / 2 + 1, batch),
                              FLAGFFT_R2C,
                              N,
                              batch,
                              input_scale_name(scale));
  }
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         R2C_1D_Test,
                         ::testing::ValuesIn(Generate1DParamsSmoke()),
                         [](const auto& info) {
                           return std::to_string(info.param.N) + "x" + std::to_string(info.param.batch);
                         });
INSTANTIATE_TEST_SUITE_P(ExtendedSmall,
                         R2C_1D_Test,
                         ::testing::ValuesIn(Generate1DParamsExtendedSmall()),
                         [](const auto& info) {
                           return std::to_string(info.param.N) + "x" + std::to_string(info.param.batch);
                         });
INSTANTIATE_TEST_SUITE_P(ExtendedMedium,
                         R2C_1D_Test,
                         ::testing::ValuesIn(Generate1DParamsExtendedMedium()),
                         [](const auto& info) {
                           return std::to_string(info.param.N) + "x" + std::to_string(info.param.batch);
                         });
INSTANTIATE_TEST_SUITE_P(ExtendedLarge,
                         R2C_1D_Test,
                         ::testing::ValuesIn(Generate1DParamsExtendedLarge()),
                         [](const auto& info) {
                           return std::to_string(info.param.N) + "x" + std::to_string(info.param.batch);
                         });
