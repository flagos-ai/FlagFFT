#include "flagfft_test.h"

using namespace flagfft_test::adaptor;

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

    h_in = random_double_complex(total_in, accuracy_seed(FLAGFFT_Z2D, N, batch));

    // Ensure Hermitian symmetry: DC and Nyquist bins must have zero imaginary part
    for (int b = 0; b < batch; ++b) {
      h_in[b * (N / 2 + 1) + 0].y = 0.0;
      if (N % 2 == 0) h_in[b * (N / 2 + 1) + N / 2].y = 0.0;
    }

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
  RefHandle ref;
  ref_plan_1d(ref, N, FLAGFFT_Z2D, batch);
  std::vector<flagfftDoubleReal> h_out(total_out);
  std::vector<flagfftDoubleReal> h_ref_out(total_out);
  for (double scale : kAccuracyInputScales) {
    auto input = h_in;
    scale_input(input, scale);
    copy_host_to_device(input.data(), d_in, total_in * sizeof(flagfftDoubleComplex));
    ExecZ2D(plan, d_in, d_out);
    ref_exec_z2d(ref, d_in, d_ref);
    copy_device_to_host(d_out, h_out.data(), total_out * sizeof(flagfftDoubleReal));
    copy_device_to_host(d_ref, h_ref_out.data(), total_out * sizeof(flagfftDoubleReal));
    expect_reference_accuracy(error_stats(h_out.data(), h_ref_out.data(), N, batch),
                              FLAGFFT_Z2D,
                              N,
                              batch,
                              input_scale_name(scale));
  }
}

TEST(SmokeZ2DAccuracy, ZeroInputIsExact) {
  constexpr int kN = 256;
  constexpr int kInputCount = kN / 2 + 1;
  flagfftHandle plan = nullptr;
  Plan1d(&plan, kN, FLAGFFT_Z2D);
  std::vector<flagfftDoubleComplex> h_in(kInputCount, {0.0, 0.0});
  std::vector<flagfftDoubleReal> h_out(kN);
  auto* d_in =
      static_cast<flagfftDoubleComplex*>(allocate_device(kInputCount * sizeof(flagfftDoubleComplex)));
  auto* d_out = static_cast<flagfftDoubleReal*>(allocate_device(kN * sizeof(flagfftDoubleReal)));
  ASSERT_NE(d_in, nullptr);
  ASSERT_NE(d_out, nullptr);
  copy_host_to_device(h_in.data(), d_in, kInputCount * sizeof(flagfftDoubleComplex));
  ExecZ2D(plan, d_in, d_out);
  copy_device_to_host(d_out, h_out.data(), kN * sizeof(flagfftDoubleReal));
  for (flagfftDoubleReal value : h_out) {
    EXPECT_EQ(value, 0.0);
    EXPECT_TRUE(std::isfinite(value));
  }
  free_device(d_in);
  free_device(d_out);
  EXPECT_EQ(flagfftDestroy(plan), FLAGFFT_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         Z2D_1D_Test,
                         ::testing::ValuesIn(Generate1DParamsSmoke()),
                         [](const auto& info) {
                           return std::to_string(info.param.N) + "x" + std::to_string(info.param.batch);
                         });
INSTANTIATE_TEST_SUITE_P(ExtendedSmall,
                         Z2D_1D_Test,
                         ::testing::ValuesIn(Generate1DParamsExtendedSmall()),
                         [](const auto& info) {
                           return std::to_string(info.param.N) + "x" + std::to_string(info.param.batch);
                         });
INSTANTIATE_TEST_SUITE_P(ExtendedMedium,
                         Z2D_1D_Test,
                         ::testing::ValuesIn(Generate1DParamsExtendedMedium()),
                         [](const auto& info) {
                           return std::to_string(info.param.N) + "x" + std::to_string(info.param.batch);
                         });
INSTANTIATE_TEST_SUITE_P(ExtendedLarge,
                         Z2D_1D_Test,
                         ::testing::ValuesIn(Generate1DParamsExtendedLarge()),
                         [](const auto& info) {
                           return std::to_string(info.param.N) + "x" + std::to_string(info.param.batch);
                         });
