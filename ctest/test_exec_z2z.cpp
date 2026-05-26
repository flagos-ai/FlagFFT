#include "flagfft_test.h"

using namespace flagfft_test::adaptor;

constexpr double kRelTol = 1e-10;

class Z2Z_1D_Test : public ::testing::TestWithParam<Test1DParam> {
 protected:
  void SetUp() override {
    auto p = GetParam();
    N = p.N;
    batch = p.batch;
    total = N * batch;

    plan = nullptr;
    Plan1d(&plan, N, FLAGFFT_Z2Z, batch);

    h_in = random_double_complex(total);

    auto bytes = total * sizeof(flagfftDoubleComplex);
    d_in = static_cast<flagfftDoubleComplex*>(allocate_device(bytes));
    d_out = static_cast<flagfftDoubleComplex*>(allocate_device(bytes));
    d_aux = static_cast<flagfftDoubleComplex*>(allocate_device(bytes));
    ASSERT_NE(d_in, nullptr);
    ASSERT_NE(d_out, nullptr);
    ASSERT_NE(d_aux, nullptr);

    copy_host_to_device(h_in.data(), d_in, bytes);
  }

  void TearDown() override {
    if (d_in) free_device(d_in);
    if (d_out) free_device(d_out);
    if (d_aux) free_device(d_aux);
    if (plan) flagfftDestroy(plan);
  }

  int N = 0;
  int batch = 0;
  int total = 0;
  flagfftHandle plan = nullptr;
  std::vector<flagfftDoubleComplex> h_in;
  flagfftDoubleComplex* d_in = nullptr;
  flagfftDoubleComplex* d_out = nullptr;
  flagfftDoubleComplex* d_aux = nullptr;
};

TEST_P(Z2Z_1D_Test, ForwardVsReference) {
  ExecZ2Z(plan, d_in, d_out, FLAGFFT_FORWARD);

  RefHandle ref;
  ref_plan_1d(ref, N, FLAGFFT_Z2Z, batch);
  ref_exec_z2z(ref, d_in, d_aux, FLAGFFT_FORWARD);

  std::vector<flagfftDoubleComplex> h_out(total);
  std::vector<flagfftDoubleComplex> h_ref(total);
  copy_device_to_host(d_out, h_out.data(), total * sizeof(flagfftDoubleComplex));
  copy_device_to_host(d_aux, h_ref.data(), total * sizeof(flagfftDoubleComplex));

  double max_err = max_relative_error(h_out.data(), h_ref.data(), total);
  EXPECT_LT(max_err, kRelTol) << "N=" << N << " batch=" << batch << " max relative error: " << max_err;
}

TEST_P(Z2Z_1D_Test, InverseVsReference) {
  ExecZ2Z(plan, d_in, d_out, FLAGFFT_INVERSE);

  RefHandle ref;
  ref_plan_1d(ref, N, FLAGFFT_Z2Z, batch);
  ref_exec_z2z(ref, d_in, d_aux, FLAGFFT_INVERSE);

  std::vector<flagfftDoubleComplex> h_out(total);
  std::vector<flagfftDoubleComplex> h_ref(total);
  copy_device_to_host(d_out, h_out.data(), total * sizeof(flagfftDoubleComplex));
  copy_device_to_host(d_aux, h_ref.data(), total * sizeof(flagfftDoubleComplex));

  double max_err = max_relative_error(h_out.data(), h_ref.data(), total);
  EXPECT_LT(max_err, kRelTol) << "N=" << N << " batch=" << batch << " max relative error: " << max_err;
}

TEST_P(Z2Z_1D_Test, Roundtrip) {
  ExecZ2Z(plan, d_in, d_aux, FLAGFFT_FORWARD);
  ExecZ2Z(plan, d_aux, d_out, FLAGFFT_INVERSE);

  std::vector<flagfftDoubleComplex> h_out(total);
  copy_device_to_host(d_out, h_out.data(), total * sizeof(flagfftDoubleComplex));

  for (int i = 0; i < total; ++i) {
    double expected_x = h_in[i].x * N;
    double expected_y = h_in[i].y * N;
    EXPECT_NEAR(h_out[i].x, expected_x, N * kRelTol)
        << "N=" << N << " batch=" << batch << " i=" << i << " (real)";
    EXPECT_NEAR(h_out[i].y, expected_y, N * kRelTol)
        << "N=" << N << " batch=" << batch << " i=" << i << " (imag)";
  }
}

INSTANTIATE_TEST_SUITE_P(Small,
                         Z2Z_1D_Test,
                         ::testing::ValuesIn(Generate1DParamsSmall()),
                         [](const auto& info) {
                           return std::to_string(info.param.N) + "x" + std::to_string(info.param.batch);
                         });
INSTANTIATE_TEST_SUITE_P(Medium,
                         Z2Z_1D_Test,
                         ::testing::ValuesIn(Generate1DParamsMedium()),
                         [](const auto& info) {
                           return std::to_string(info.param.N) + "x" + std::to_string(info.param.batch);
                         });
INSTANTIATE_TEST_SUITE_P(Large,
                         Z2Z_1D_Test,
                         ::testing::ValuesIn(Generate1DParamsLarge()),
                         [](const auto& info) {
                           return std::to_string(info.param.N) + "x" + std::to_string(info.param.batch);
                         });
