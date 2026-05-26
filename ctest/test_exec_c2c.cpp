#include "flagfft_test.h"

using namespace flagfft_test::adaptor;

constexpr double kRelTol = 1e-4;

class C2C_1D_Test : public ::testing::TestWithParam<Test1DParam> {
 protected:
  void SetUp() override {
    auto p = GetParam();
    N = p.N;
    batch = p.batch;
    total = N * batch;

    plan = nullptr;
    Plan1d(&plan, N, FLAGFFT_C2C, batch);

    h_in = random_complex(total);

    auto bytes = total * sizeof(flagfftComplex);
    d_in = static_cast<flagfftComplex*>(allocate_device(bytes));
    d_out = static_cast<flagfftComplex*>(allocate_device(bytes));
    d_aux = static_cast<flagfftComplex*>(allocate_device(bytes));
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
  std::vector<flagfftComplex> h_in;
  flagfftComplex* d_in = nullptr;
  flagfftComplex* d_out = nullptr;
  flagfftComplex* d_aux = nullptr;
};

TEST_P(C2C_1D_Test, ForwardVsReference) {
  ExecC2C(plan, d_in, d_out, FLAGFFT_FORWARD);

  RefHandle ref;
  ref_plan_1d(ref, N, FLAGFFT_C2C, batch);
  ref_exec_c2c(ref, d_in, d_aux, FLAGFFT_FORWARD);

  std::vector<flagfftComplex> h_out(total);
  std::vector<flagfftComplex> h_ref(total);
  copy_device_to_host(d_out, h_out.data(), total * sizeof(flagfftComplex));
  copy_device_to_host(d_aux, h_ref.data(), total * sizeof(flagfftComplex));

  double max_err = max_relative_error(h_out.data(), h_ref.data(), total);
  EXPECT_LT(max_err, kRelTol) << "N=" << N << " batch=" << batch << " max relative error: " << max_err;
}

TEST_P(C2C_1D_Test, InverseVsReference) {
  ExecC2C(plan, d_in, d_out, FLAGFFT_INVERSE);

  RefHandle ref;
  ref_plan_1d(ref, N, FLAGFFT_C2C, batch);
  ref_exec_c2c(ref, d_in, d_aux, FLAGFFT_INVERSE);

  std::vector<flagfftComplex> h_out(total);
  std::vector<flagfftComplex> h_ref(total);
  copy_device_to_host(d_out, h_out.data(), total * sizeof(flagfftComplex));
  copy_device_to_host(d_aux, h_ref.data(), total * sizeof(flagfftComplex));

  double max_err = max_relative_error(h_out.data(), h_ref.data(), total);
  EXPECT_LT(max_err, kRelTol) << "N=" << N << " batch=" << batch << " max relative error: " << max_err;
}

TEST_P(C2C_1D_Test, Roundtrip) {
  ExecC2C(plan, d_in, d_aux, FLAGFFT_FORWARD);
  ExecC2C(plan, d_aux, d_out, FLAGFFT_INVERSE);

  std::vector<flagfftComplex> h_out(total);
  copy_device_to_host(d_out, h_out.data(), total * sizeof(flagfftComplex));

  for (int i = 0; i < total; ++i) {
    double expected_x = h_in[i].x * N;
    double expected_y = h_in[i].y * N;
    EXPECT_NEAR(h_out[i].x, expected_x, N * kRelTol)
        << "N=" << N << " batch=" << batch << " i=" << i << " (real)";
    EXPECT_NEAR(h_out[i].y, expected_y, N * kRelTol)
        << "N=" << N << " batch=" << batch << " i=" << i << " (imag)";
  }
}

INSTANTIATE_TEST_SUITE_P(Small, C2C_1D_Test, ::testing::ValuesIn(Generate1DParamsSmall()));
INSTANTIATE_TEST_SUITE_P(Medium, C2C_1D_Test, ::testing::ValuesIn(Generate1DParamsMedium()));
INSTANTIATE_TEST_SUITE_P(Large, C2C_1D_Test, ::testing::ValuesIn(Generate1DParamsLarge()));
