#include "flagfft_test.h"

using namespace flagfft_test::adaptor;

constexpr double kRelTol = 1e-4;

class R2C_C2R_Roundtrip_Test : public ::testing::TestWithParam<Test1DParam> {
 protected:
  void SetUp() override {
    auto p = GetParam();
    N = p.N;
    batch = p.batch;
    total_real = N * batch;
    total_complex = (N / 2 + 1) * batch;

    plan_fwd = nullptr;
    plan_inv = nullptr;
    Plan1d(&plan_fwd, N, FLAGFFT_R2C, batch);
    Plan1d(&plan_inv, N, FLAGFFT_C2R, batch);

    h_in = random_real(total_real);

    d_in = static_cast<flagfftReal*>(allocate_device(total_real * sizeof(flagfftReal)));
    d_mid = static_cast<flagfftComplex*>(allocate_device(total_complex * sizeof(flagfftComplex)));
    d_out = static_cast<flagfftReal*>(allocate_device(total_real * sizeof(flagfftReal)));
    ASSERT_NE(d_in, nullptr);
    ASSERT_NE(d_mid, nullptr);
    ASSERT_NE(d_out, nullptr);

    copy_host_to_device(h_in.data(), d_in, total_real * sizeof(flagfftReal));
  }

  void TearDown() override {
    if (d_in) free_device(d_in);
    if (d_mid) free_device(d_mid);
    if (d_out) free_device(d_out);
    if (plan_fwd) flagfftDestroy(plan_fwd);
    if (plan_inv) flagfftDestroy(plan_inv);
  }

  int N = 0;
  int batch = 0;
  int total_real = 0;
  int total_complex = 0;
  flagfftHandle plan_fwd = nullptr;
  flagfftHandle plan_inv = nullptr;
  std::vector<flagfftReal> h_in;
  flagfftReal* d_in = nullptr;
  flagfftComplex* d_mid = nullptr;
  flagfftReal* d_out = nullptr;
};

TEST_P(R2C_C2R_Roundtrip_Test, Roundtrip1D) {
  ExecR2C(plan_fwd, d_in, d_mid);
  ExecC2R(plan_inv, d_mid, d_out);

  std::vector<flagfftReal> h_out(total_real);
  copy_device_to_host(d_out, h_out.data(), total_real * sizeof(flagfftReal));

  for (int i = 0; i < total_real; ++i) {
    double expected = static_cast<double>(h_in[i]) * N;
    EXPECT_NEAR(static_cast<double>(h_out[i]), expected, N * kRelTol)
        << "N=" << N << " batch=" << batch << " i=" << i;
  }
}

INSTANTIATE_TEST_SUITE_P(Small,
                         R2C_C2R_Roundtrip_Test,
                         ::testing::ValuesIn(Generate1DParamsSmall()),
                         [](const auto& info) {
                           return std::to_string(info.param.N) + "x" + std::to_string(info.param.batch);
                         });
INSTANTIATE_TEST_SUITE_P(Medium,
                         R2C_C2R_Roundtrip_Test,
                         ::testing::ValuesIn(Generate1DParamsMedium()),
                         [](const auto& info) {
                           return std::to_string(info.param.N) + "x" + std::to_string(info.param.batch);
                         });
INSTANTIATE_TEST_SUITE_P(Large,
                         R2C_C2R_Roundtrip_Test,
                         ::testing::ValuesIn(Generate1DParamsLarge()),
                         [](const auto& info) {
                           return std::to_string(info.param.N) + "x" + std::to_string(info.param.batch);
                         });
