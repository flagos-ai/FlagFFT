#include "flagfft_test.h"

using namespace flagfft_test::adaptor;

constexpr double kRelTol = 1e-10;

class D2Z_1D_Test : public ::testing::TestWithParam<Test1DParam> {
 protected:
  void SetUp() override {
    auto p = GetParam();
    N = p.N;
    batch = p.batch;
    total_in = N * batch;
    total_out = (N / 2 + 1) * batch;

    plan = nullptr;
    Plan1d(&plan, N, FLAGFFT_D2Z, batch);

    h_in = random_double_real(total_in);

    d_in = static_cast<flagfftDoubleReal*>(allocate_device(total_in * sizeof(flagfftDoubleReal)));
    d_out = static_cast<flagfftDoubleComplex*>(allocate_device(total_out * sizeof(flagfftDoubleComplex)));
    d_ref = static_cast<flagfftDoubleComplex*>(allocate_device(total_out * sizeof(flagfftDoubleComplex)));
    ASSERT_NE(d_in, nullptr);
    ASSERT_NE(d_out, nullptr);
    ASSERT_NE(d_ref, nullptr);

    copy_host_to_device(h_in.data(), d_in, total_in * sizeof(flagfftDoubleReal));
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
  std::vector<flagfftDoubleReal> h_in;
  flagfftDoubleReal* d_in = nullptr;
  flagfftDoubleComplex* d_out = nullptr;
  flagfftDoubleComplex* d_ref = nullptr;
};

TEST_P(D2Z_1D_Test, ForwardVsReference) {
  ExecD2Z(plan, d_in, d_out);

  RefHandle ref;
  ref_plan_1d(ref, N, FLAGFFT_D2Z, batch);
  ref_exec_d2z(ref, d_in, d_ref);

  std::vector<flagfftDoubleComplex> h_out(total_out);
  std::vector<flagfftDoubleComplex> h_ref_out(total_out);
  copy_device_to_host(d_out, h_out.data(), total_out * sizeof(flagfftDoubleComplex));
  copy_device_to_host(d_ref, h_ref_out.data(), total_out * sizeof(flagfftDoubleComplex));

  double max_err = max_relative_error(h_out.data(), h_ref_out.data(), total_out);
  EXPECT_LT(max_err, kRelTol) << "N=" << N << " batch=" << batch << " max relative error: " << max_err;
}

INSTANTIATE_TEST_SUITE_P(Small,
                         D2Z_1D_Test,
                         ::testing::ValuesIn(Generate1DParamsSmall()),
                         [](const auto& info) {
                           return std::to_string(info.param.N) + "x" + std::to_string(info.param.batch);
                         });
INSTANTIATE_TEST_SUITE_P(Medium,
                         D2Z_1D_Test,
                         ::testing::ValuesIn(Generate1DParamsMedium()),
                         [](const auto& info) {
                           return std::to_string(info.param.N) + "x" + std::to_string(info.param.batch);
                         });
INSTANTIATE_TEST_SUITE_P(Large,
                         D2Z_1D_Test,
                         ::testing::ValuesIn(Generate1DParamsLarge()),
                         [](const auto& info) {
                           return std::to_string(info.param.N) + "x" + std::to_string(info.param.batch);
                         });
