#include "flagfft_test.h"

using namespace flagfft_test;

class D2Z_Z2DCTBatch_Test : public ::testing::TestWithParam<Test1DParam> {
 protected:
  void SetUp() override {
    auto p = GetParam();
    N = p.N;
    batch = p.batch;
    total_real = N * batch;
    total_complex = (N / 2 + 1) * batch;

    plan_fwd = nullptr;
    plan_inv = nullptr;
    Plan1d(&plan_fwd, N, FLAGFFT_D2Z, batch);
    Plan1d(&plan_inv, N, FLAGFFT_Z2D, batch);

    h_in = random_double_real(total_real, accuracy_seed(FLAGFFT_D2Z, N, batch));

    in_memory.allocate(total_real * sizeof(flagfftDoubleReal));
    mid_memory.allocate(total_complex * sizeof(flagfftDoubleComplex));
    out_memory.allocate(total_real * sizeof(flagfftDoubleReal));
    d_in = static_cast<flagfftDoubleReal*>(in_memory.data());
    d_mid = static_cast<flagfftDoubleComplex*>(mid_memory.data());
    d_out = static_cast<flagfftDoubleReal*>(out_memory.data());
    ASSERT_NE(d_in, nullptr);
    ASSERT_NE(d_mid, nullptr);
    ASSERT_NE(d_out, nullptr);

    in_memory.copy_from_host(h_in.data(), total_real * sizeof(flagfftDoubleReal));
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
  std::vector<flagfftDoubleReal> h_in;
  flagfft::adaptor::Memory in_memory;
  flagfft::adaptor::Memory mid_memory;
  flagfft::adaptor::Memory out_memory;
  flagfftDoubleReal* d_in = nullptr;
  flagfftDoubleComplex* d_mid = nullptr;
  flagfftDoubleReal* d_out = nullptr;
};

TEST_P(D2Z_Z2DCTBatch_Test, Roundtrip1D) {
  ExecD2Z(plan_fwd, d_in, d_mid);
  ExecZ2D(plan_inv, d_mid, d_out);

  std::vector<flagfftDoubleReal> h_out(total_real);
  std::vector<flagfftDoubleReal> h_expected(total_real);
  out_memory.copy_to_host(h_out.data(), total_real * sizeof(flagfftDoubleReal));

  for (int i = 0; i < total_real; ++i) {
    h_expected[i] = h_in[i] * N;
  }
  expect_roundtrip_accuracy(error_stats(h_out.data(), h_expected.data(), N, batch),
                            FLAGFFT_D2Z,
                            FLAGFFT_Z2D,
                            N,
                            batch);
}

INSTANTIATE_TEST_SUITE_P(ExtendedSmall,
                         D2Z_Z2DCTBatch_Test,
                         ::testing::ValuesIn(Generate1DParamsCTExtendedSmallBatch()),
                         [](const auto& info) {
                           return std::to_string(info.param.N) + "x" + std::to_string(info.param.batch);
                         });
INSTANTIATE_TEST_SUITE_P(ExtendedMedium,
                         D2Z_Z2DCTBatch_Test,
                         ::testing::ValuesIn(Generate1DParamsCTExtendedMediumBatch()),
                         [](const auto& info) {
                           return std::to_string(info.param.N) + "x" + std::to_string(info.param.batch);
                         });
INSTANTIATE_TEST_SUITE_P(ExtendedLarge,
                         D2Z_Z2DCTBatch_Test,
                         ::testing::ValuesIn(Generate1DParamsCTExtendedLargeBatch()),
                         [](const auto& info) {
                           return std::to_string(info.param.N) + "x" + std::to_string(info.param.batch);
                         });
