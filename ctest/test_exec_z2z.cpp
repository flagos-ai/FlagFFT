#include "flagfft_test.h"

using namespace flagfft_test::adaptor;

class Z2Z_1D_Test : public ::testing::TestWithParam<Test1DParam> {
 protected:
  void SetUp() override {
    auto p = GetParam();
    N = p.N;
    batch = p.batch;
    total = N * batch;

    plan = nullptr;
    Plan1d(&plan, N, FLAGFFT_Z2Z, batch);

    h_in = random_double_complex(total, accuracy_seed(FLAGFFT_Z2Z, N, batch));

    auto bytes = total * sizeof(flagfftDoubleComplex);
    d_in = static_cast<flagfftDoubleComplex*>(allocate_device(bytes));
    d_out = static_cast<flagfftDoubleComplex*>(allocate_device(bytes));
    d_ref = static_cast<flagfftDoubleComplex*>(allocate_device(bytes));
    ASSERT_NE(d_in, nullptr);
    ASSERT_NE(d_out, nullptr);
    ASSERT_NE(d_ref, nullptr);

    copy_host_to_device(h_in.data(), d_in, bytes);
  }

  void TearDown() override {
    if (d_in) free_device(d_in);
    if (d_out) free_device(d_out);
    if (d_ref) free_device(d_ref);
    if (plan) flagfftDestroy(plan);
  }

  int N = 0;
  int batch = 0;
  int total = 0;
  flagfftHandle plan = nullptr;
  std::vector<flagfftDoubleComplex> h_in;
  flagfftDoubleComplex* d_in = nullptr;
  flagfftDoubleComplex* d_out = nullptr;
  flagfftDoubleComplex* d_ref = nullptr;
};

TEST_P(Z2Z_1D_Test, ForwardVsReference) {
  RefHandle ref;
  ref_plan_1d(ref, N, FLAGFFT_Z2Z, batch);
  std::vector<flagfftDoubleComplex> h_out(total);
  std::vector<flagfftDoubleComplex> h_ref(total);
  for (double scale : kAccuracyInputScales) {
    auto input = h_in;
    scale_input(input, scale);
    copy_host_to_device(input.data(), d_in, total * sizeof(flagfftDoubleComplex));
    ExecZ2Z(plan, d_in, d_out, FLAGFFT_FORWARD);
    ref_exec_z2z(ref, d_in, d_ref, FLAGFFT_FORWARD);
    copy_device_to_host(d_out, h_out.data(), total * sizeof(flagfftDoubleComplex));
    copy_device_to_host(d_ref, h_ref.data(), total * sizeof(flagfftDoubleComplex));
    expect_reference_accuracy(error_stats(h_out.data(), h_ref.data(), N, batch),
                              FLAGFFT_Z2Z,
                              N,
                              batch,
                              input_scale_name(scale));
  }
}

TEST_P(Z2Z_1D_Test, InverseVsReference) {
  RefHandle ref;
  ref_plan_1d(ref, N, FLAGFFT_Z2Z, batch);
  std::vector<flagfftDoubleComplex> h_out(total);
  std::vector<flagfftDoubleComplex> h_ref(total);
  for (double scale : kAccuracyInputScales) {
    auto input = h_in;
    scale_input(input, scale);
    copy_host_to_device(input.data(), d_in, total * sizeof(flagfftDoubleComplex));
    ExecZ2Z(plan, d_in, d_out, FLAGFFT_INVERSE);
    ref_exec_z2z(ref, d_in, d_ref, FLAGFFT_INVERSE);
    copy_device_to_host(d_out, h_out.data(), total * sizeof(flagfftDoubleComplex));
    copy_device_to_host(d_ref, h_ref.data(), total * sizeof(flagfftDoubleComplex));
    expect_reference_accuracy(error_stats(h_out.data(), h_ref.data(), N, batch),
                              FLAGFFT_Z2Z,
                              N,
                              batch,
                              input_scale_name(scale));
  }
}

TEST_P(Z2Z_1D_Test, Roundtrip) {
  auto* d_mid = d_ref;  // reuse reference buffer as intermediate
  ExecZ2Z(plan, d_in, d_mid, FLAGFFT_FORWARD);
  ExecZ2Z(plan, d_mid, d_out, FLAGFFT_INVERSE);

  std::vector<flagfftDoubleComplex> h_out(total);
  std::vector<flagfftDoubleComplex> h_expected(total);
  copy_device_to_host(d_out, h_out.data(), total * sizeof(flagfftDoubleComplex));

  for (int i = 0; i < total; ++i) {
    h_expected[i].x = h_in[i].x * N;
    h_expected[i].y = h_in[i].y * N;
  }
  expect_roundtrip_accuracy(error_stats(h_out.data(), h_expected.data(), N, batch),
                            FLAGFFT_Z2Z,
                            FLAGFFT_Z2Z,
                            N,
                            batch);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         Z2Z_1D_Test,
                         ::testing::ValuesIn(Generate1DParamsSmoke()),
                         [](const auto& info) {
                           return std::to_string(info.param.N) + "x" + std::to_string(info.param.batch);
                         });
INSTANTIATE_TEST_SUITE_P(ExtendedSmall,
                         Z2Z_1D_Test,
                         ::testing::ValuesIn(Generate1DParamsExtendedSmall()),
                         [](const auto& info) {
                           return std::to_string(info.param.N) + "x" + std::to_string(info.param.batch);
                         });
INSTANTIATE_TEST_SUITE_P(ExtendedMedium,
                         Z2Z_1D_Test,
                         ::testing::ValuesIn(Generate1DParamsExtendedMedium()),
                         [](const auto& info) {
                           return std::to_string(info.param.N) + "x" + std::to_string(info.param.batch);
                         });
INSTANTIATE_TEST_SUITE_P(ExtendedLarge,
                         Z2Z_1D_Test,
                         ::testing::ValuesIn(Generate1DParamsExtendedLarge()),
                         [](const auto& info) {
                           return std::to_string(info.param.N) + "x" + std::to_string(info.param.batch);
                         });
