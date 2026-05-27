#include "flagfft_test.h"

using namespace flagfft_test::adaptor;

constexpr double kRelTol = 2e-4;

class C2R_1D_Test : public ::testing::TestWithParam<Test1DParam> {
 protected:
  void SetUp() override {
    auto p = GetParam();
    N = p.N;
    batch = p.batch;
    total_in = (N / 2 + 1) * batch;
    total_out = N * batch;

    plan = nullptr;
    Plan1d(&plan, N, FLAGFFT_C2R, batch);

    h_in = random_complex(total_in);

    // Ensure Hermitian symmetry: DC and Nyquist bins must have zero imaginary part
    for (int b = 0; b < batch; ++b) {
      h_in[b * (N / 2 + 1) + 0].y = 0.0f;
      if (N % 2 == 0) h_in[b * (N / 2 + 1) + N / 2].y = 0.0f;
    }

    d_in = static_cast<flagfftComplex*>(allocate_device(total_in * sizeof(flagfftComplex)));
    d_out = static_cast<flagfftReal*>(allocate_device(total_out * sizeof(flagfftReal)));
    d_ref = static_cast<flagfftReal*>(allocate_device(total_out * sizeof(flagfftReal)));
    ASSERT_NE(d_in, nullptr);
    ASSERT_NE(d_out, nullptr);
    ASSERT_NE(d_ref, nullptr);

    copy_host_to_device(h_in.data(), d_in, total_in * sizeof(flagfftComplex));
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
  std::vector<flagfftComplex> h_in;
  flagfftComplex* d_in = nullptr;
  flagfftReal* d_out = nullptr;
  flagfftReal* d_ref = nullptr;
};

TEST_P(C2R_1D_Test, InverseVsReference) {
  ExecC2R(plan, d_in, d_out);

  RefHandle ref;
  ref_plan_1d(ref, N, FLAGFFT_C2R, batch);
  ref_exec_c2r(ref, d_in, d_ref);

  std::vector<flagfftReal> h_out(total_out);
  std::vector<flagfftReal> h_ref_out(total_out);
  copy_device_to_host(d_out, h_out.data(), total_out * sizeof(flagfftReal));
  copy_device_to_host(d_ref, h_ref_out.data(), total_out * sizeof(flagfftReal));

  double max_err = max_relative_error_real(h_out.data(), h_ref_out.data(), total_out);
  EXPECT_LT(max_err, kRelTol) << "N=" << N << " batch=" << batch << " max relative error: " << max_err;
}

INSTANTIATE_TEST_SUITE_P(Small,
                         C2R_1D_Test,
                         ::testing::ValuesIn(Generate1DParamsSmall()),
                         [](const auto& info) {
                           return std::to_string(info.param.N) + "x" + std::to_string(info.param.batch);
                         });
INSTANTIATE_TEST_SUITE_P(Medium,
                         C2R_1D_Test,
                         ::testing::ValuesIn(Generate1DParamsMedium()),
                         [](const auto& info) {
                           return std::to_string(info.param.N) + "x" + std::to_string(info.param.batch);
                         });
INSTANTIATE_TEST_SUITE_P(Large,
                         C2R_1D_Test,
                         ::testing::ValuesIn(Generate1DParamsLarge()),
                         [](const auto& info) {
                           return std::to_string(info.param.N) + "x" + std::to_string(info.param.batch);
                         });
