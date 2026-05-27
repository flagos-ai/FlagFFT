#include "flagfft_test.h"

using namespace flagfft_test::adaptor;

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

    h_in = random_complex(total_in, accuracy_seed(FLAGFFT_C2R, N, batch));

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
  RefHandle ref;
  ref_plan_1d(ref, N, FLAGFFT_C2R, batch);
  std::vector<flagfftReal> h_out(total_out);
  std::vector<flagfftReal> h_ref_out(total_out);
  for (double scale : kAccuracyInputScales) {
    auto input = h_in;
    scale_input(input, scale);
    copy_host_to_device(input.data(), d_in, total_in * sizeof(flagfftComplex));
    ExecC2R(plan, d_in, d_out);
    ref_exec_c2r(ref, d_in, d_ref);
    copy_device_to_host(d_out, h_out.data(), total_out * sizeof(flagfftReal));
    copy_device_to_host(d_ref, h_ref_out.data(), total_out * sizeof(flagfftReal));
    expect_reference_accuracy(error_stats(h_out.data(), h_ref_out.data(), N, batch),
                              FLAGFFT_C2R,
                              N,
                              batch,
                              input_scale_name(scale));
  }
}

TEST(ExtendedC2RRegression, LargeBatchMultipleSeeds) {
  constexpr int kN = 16384;
  constexpr int kBatch = 256;
  constexpr int kTotalIn = (kN / 2 + 1) * kBatch;
  constexpr int kTotalOut = kN * kBatch;
  flagfftHandle plan = nullptr;
  Plan1d(&plan, kN, FLAGFFT_C2R, kBatch);

  RefHandle ref;
  ref_plan_1d(ref, kN, FLAGFFT_C2R, kBatch);
  auto* d_in = static_cast<flagfftComplex*>(allocate_device(kTotalIn * sizeof(flagfftComplex)));
  auto* d_out = static_cast<flagfftReal*>(allocate_device(kTotalOut * sizeof(flagfftReal)));
  auto* d_ref = static_cast<flagfftReal*>(allocate_device(kTotalOut * sizeof(flagfftReal)));
  ASSERT_NE(d_in, nullptr);
  ASSERT_NE(d_out, nullptr);
  ASSERT_NE(d_ref, nullptr);

  std::vector<flagfftReal> h_out(kTotalOut);
  std::vector<flagfftReal> h_ref_out(kTotalOut);
  for (std::uint64_t seed = 1; seed <= 10; ++seed) {
    SCOPED_TRACE("seed=" + std::to_string(seed));
    auto h_in = random_complex(kTotalIn, accuracy_seed(FLAGFFT_C2R, kN, kBatch, seed));
    for (int b = 0; b < kBatch; ++b) {
      h_in[b * (kN / 2 + 1)].y = 0.0f;
      h_in[b * (kN / 2 + 1) + kN / 2].y = 0.0f;
    }
    copy_host_to_device(h_in.data(), d_in, kTotalIn * sizeof(flagfftComplex));
    ExecC2R(plan, d_in, d_out);
    ref_exec_c2r(ref, d_in, d_ref);
    copy_device_to_host(d_out, h_out.data(), kTotalOut * sizeof(flagfftReal));
    copy_device_to_host(d_ref, h_ref_out.data(), kTotalOut * sizeof(flagfftReal));
    expect_reference_accuracy(error_stats(h_out.data(), h_ref_out.data(), kN, kBatch),
                              FLAGFFT_C2R,
                              kN,
                              kBatch);
  }

  free_device(d_in);
  free_device(d_out);
  free_device(d_ref);
  EXPECT_EQ(flagfftDestroy(plan), FLAGFFT_SUCCESS);
}

TEST(SmokeC2RAccuracy, ZeroInputIsExact) {
  constexpr int kN = 256;
  constexpr int kInputCount = kN / 2 + 1;
  flagfftHandle plan = nullptr;
  Plan1d(&plan, kN, FLAGFFT_C2R);
  std::vector<flagfftComplex> h_in(kInputCount, {0.0f, 0.0f});
  std::vector<flagfftReal> h_out(kN);
  auto* d_in = static_cast<flagfftComplex*>(allocate_device(kInputCount * sizeof(flagfftComplex)));
  auto* d_out = static_cast<flagfftReal*>(allocate_device(kN * sizeof(flagfftReal)));
  ASSERT_NE(d_in, nullptr);
  ASSERT_NE(d_out, nullptr);
  copy_host_to_device(h_in.data(), d_in, kInputCount * sizeof(flagfftComplex));
  ExecC2R(plan, d_in, d_out);
  copy_device_to_host(d_out, h_out.data(), kN * sizeof(flagfftReal));
  for (flagfftReal value : h_out) {
    EXPECT_EQ(value, 0.0f);
    EXPECT_TRUE(std::isfinite(value));
  }
  free_device(d_in);
  free_device(d_out);
  EXPECT_EQ(flagfftDestroy(plan), FLAGFFT_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         C2R_1D_Test,
                         ::testing::ValuesIn(Generate1DParamsSmoke()),
                         [](const auto& info) {
                           return std::to_string(info.param.N) + "x" + std::to_string(info.param.batch);
                         });
INSTANTIATE_TEST_SUITE_P(ExtendedSmall,
                         C2R_1D_Test,
                         ::testing::ValuesIn(Generate1DParamsExtendedSmall()),
                         [](const auto& info) {
                           return std::to_string(info.param.N) + "x" + std::to_string(info.param.batch);
                         });
INSTANTIATE_TEST_SUITE_P(ExtendedMedium,
                         C2R_1D_Test,
                         ::testing::ValuesIn(Generate1DParamsExtendedMedium()),
                         [](const auto& info) {
                           return std::to_string(info.param.N) + "x" + std::to_string(info.param.batch);
                         });
INSTANTIATE_TEST_SUITE_P(ExtendedLarge,
                         C2R_1D_Test,
                         ::testing::ValuesIn(Generate1DParamsExtendedLarge()),
                         [](const auto& info) {
                           return std::to_string(info.param.N) + "x" + std::to_string(info.param.batch);
                         });
