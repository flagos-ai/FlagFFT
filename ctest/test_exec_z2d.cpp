#include "flagfft_test.h"

using namespace flagfft_test::adaptor;

constexpr double kRelTol = 1e-10;

class Z2D_1D_Test : public ::testing::TestWithParam<Test1DParam> {
 protected:
  void SetUp() override {
    auto p = GetParam();
    N = p.N;
    batch = p.batch;
    total_in = (N / 2 + 1) * batch;
    total_out = N * batch;

    plan = nullptr;
    Plan1d(&plan, N, FLAGFFT_Z2D, batch);

    h_in = random_double_complex(total_in);

    d_in = static_cast<flagfftDoubleComplex*>(allocate_device(total_in * sizeof(flagfftDoubleComplex)));
    d_out = static_cast<flagfftDoubleReal*>(allocate_device(total_out * sizeof(flagfftDoubleReal)));
    d_ref = static_cast<flagfftDoubleReal*>(allocate_device(total_out * sizeof(flagfftDoubleReal)));
    ASSERT_NE(d_in, nullptr);
    ASSERT_NE(d_out, nullptr);
    ASSERT_NE(d_ref, nullptr);

    copy_host_to_device(h_in.data(), d_in, total_in * sizeof(flagfftDoubleComplex));
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
  std::vector<flagfftDoubleComplex> h_in;
  flagfftDoubleComplex* d_in = nullptr;
  flagfftDoubleReal* d_out = nullptr;
  flagfftDoubleReal* d_ref = nullptr;
};

TEST_P(Z2D_1D_Test, InverseVsReference) {
  ExecZ2D(plan, d_in, d_out);

  RefHandle ref;
  ref_plan_1d(ref, N, FLAGFFT_Z2D, batch);
  ref_exec_z2d(ref, d_in, d_ref);

  std::vector<flagfftDoubleReal> h_out(total_out);
  std::vector<flagfftDoubleReal> h_ref_out(total_out);
  copy_device_to_host(d_out, h_out.data(), total_out * sizeof(flagfftDoubleReal));
  copy_device_to_host(d_ref, h_ref_out.data(), total_out * sizeof(flagfftDoubleReal));

  double max_err = max_relative_error_real(h_out.data(), h_ref_out.data(), total_out);
  EXPECT_LT(max_err, kRelTol) << "N=" << N << " batch=" << batch << " max relative error: " << max_err;
}

INSTANTIATE_TEST_SUITE_P(Small,
                         Z2D_1D_Test,
                         ::testing::ValuesIn(Generate1DParamsSmall()),
                         [](const auto& info) {
                           return std::to_string(info.param.N) + "x" + std::to_string(info.param.batch);
                         });
INSTANTIATE_TEST_SUITE_P(Medium,
                         Z2D_1D_Test,
                         ::testing::ValuesIn(Generate1DParamsMedium()),
                         [](const auto& info) {
                           return std::to_string(info.param.N) + "x" + std::to_string(info.param.batch);
                         });
INSTANTIATE_TEST_SUITE_P(Large,
                         Z2D_1D_Test,
                         ::testing::ValuesIn(Generate1DParamsLarge()),
                         [](const auto& info) {
                           return std::to_string(info.param.N) + "x" + std::to_string(info.param.batch);
                         });
