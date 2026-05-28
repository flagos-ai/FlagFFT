#include "flagfft_test.h"

using namespace flagfft_test;

class R2C_C2RCTBatch_Test : public ::testing::TestWithParam<Test1DParam> {
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

    h_in = random_real(total_real, accuracy_seed(FLAGFFT_R2C, N, batch));

    in_memory.allocate(total_real * sizeof(flagfftReal));
    mid_memory.allocate(total_complex * sizeof(flagfftComplex));
    out_memory.allocate(total_real * sizeof(flagfftReal));
    d_in = static_cast<flagfftReal*>(in_memory.data());
    d_mid = static_cast<flagfftComplex*>(mid_memory.data());
    d_out = static_cast<flagfftReal*>(out_memory.data());
    ASSERT_NE(d_in, nullptr);
    ASSERT_NE(d_mid, nullptr);
    ASSERT_NE(d_out, nullptr);

    in_memory.copy_from_host(h_in.data(), total_real * sizeof(flagfftReal));
  }

  void TearDown() override {
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
  flagfft::adaptor::Memory in_memory;
  flagfft::adaptor::Memory mid_memory;
  flagfft::adaptor::Memory out_memory;
  flagfftReal* d_in = nullptr;
  flagfftComplex* d_mid = nullptr;
  flagfftReal* d_out = nullptr;
};

TEST_P(R2C_C2RCTBatch_Test, Roundtrip1D) {
  ExecR2C(plan_fwd, d_in, d_mid);
  ExecC2R(plan_inv, d_mid, d_out);

  std::vector<flagfftReal> h_out(total_real);
  std::vector<flagfftReal> h_expected(total_real);
  out_memory.copy_to_host(h_out.data(), total_real * sizeof(flagfftReal));

  for (int i = 0; i < total_real; ++i) {
    h_expected[i] = h_in[i] * N;
  }
  expect_roundtrip_accuracy(error_stats(h_out.data(), h_expected.data(), N, batch),
                            FLAGFFT_R2C,
                            FLAGFFT_C2R,
                            N,
                            batch);
}

INSTANTIATE_TEST_SUITE_P(ExtendedSmall,
                         R2C_C2RCTBatch_Test,
                         ::testing::ValuesIn(Generate1DParamsCTExtendedSmallBatch()),
                         [](const auto& info) {
                           return std::to_string(info.param.N) + "x" + std::to_string(info.param.batch);
                         });
INSTANTIATE_TEST_SUITE_P(ExtendedMedium,
                         R2C_C2RCTBatch_Test,
                         ::testing::ValuesIn(Generate1DParamsCTExtendedMediumBatch()),
                         [](const auto& info) {
                           return std::to_string(info.param.N) + "x" + std::to_string(info.param.batch);
                         });
INSTANTIATE_TEST_SUITE_P(ExtendedLarge,
                         R2C_C2RCTBatch_Test,
                         ::testing::ValuesIn(Generate1DParamsCTExtendedLargeBatch()),
                         [](const auto& info) {
                           return std::to_string(info.param.N) + "x" + std::to_string(info.param.batch);
                         });
