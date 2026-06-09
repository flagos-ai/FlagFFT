#include "flagfft_test.h"

using namespace flagfft_test;

class C2RCTSingle_Test : public ::testing::TestWithParam<Test1DParam> {
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

    for (int b = 0; b < batch; ++b) {
      h_in[b * (N / 2 + 1) + 0].y = 0.0f;
      if (N % 2 == 0) h_in[b * (N / 2 + 1) + N / 2].y = 0.0f;
    }

    in_memory.allocate(total_in * sizeof(flagfftComplex));
    out_memory.allocate(total_out * sizeof(flagfftReal));
    ref_memory.allocate(total_out * sizeof(flagfftReal));
    d_in = static_cast<flagfftComplex*>(in_memory.data());
    d_out = static_cast<flagfftReal*>(out_memory.data());
    d_ref = static_cast<flagfftReal*>(ref_memory.data());
    ASSERT_NE(d_in, nullptr);
    ASSERT_NE(d_out, nullptr);
    ASSERT_NE(d_ref, nullptr);

    in_memory.copy_from_host(h_in.data(), total_in * sizeof(flagfftComplex));
  }

  void TearDown() override {
    if (plan) flagfftDestroy(plan);
  }

  int N = 0;
  int batch = 0;
  int total_in = 0;
  int total_out = 0;
  flagfftHandle plan = nullptr;
  std::vector<flagfftComplex> h_in;
  flagfft::adaptor::Memory in_memory;
  flagfft::adaptor::Memory out_memory;
  flagfft::adaptor::Memory ref_memory;
  flagfftComplex* d_in = nullptr;
  flagfftReal* d_out = nullptr;
  flagfftReal* d_ref = nullptr;
};

TEST_P(C2RCTSingle_Test, InverseVsReference) {
  RefPlanHandle ref;
  ref_plan_1d(ref, N, FLAGFFT_C2R, batch);
  std::vector<flagfftReal> h_out(total_out);
  std::vector<flagfftReal> h_ref_out(total_out);
  for (double scale : filter_scales()) {
    auto input = h_in;
    scale_input(input, scale);
    in_memory.copy_from_host(input.data(), total_in * sizeof(flagfftComplex));
    ExecC2R(plan, d_in, d_out);
    ref_exec_c2r(ref, d_in, d_ref);
    out_memory.copy_to_host(h_out.data(), total_out * sizeof(flagfftReal));
    ref_memory.copy_to_host(h_ref_out.data(), total_out * sizeof(flagfftReal));
    expect_reference_accuracy(error_stats(h_out.data(), h_ref_out.data(), N, batch),
                              FLAGFFT_C2R,
                              N,
                              batch,
                              input_scale_name(scale));
  }
}

TEST(SmokeC2RAccuracy, ZeroInputIsExact) {
  constexpr int kN = 256;
  constexpr int kInputCount = kN / 2 + 1;
  flagfftHandle plan = nullptr;
  Plan1d(&plan, kN, FLAGFFT_C2R);
  std::vector<flagfftComplex> h_in(kInputCount, {0.0f, 0.0f});
  std::vector<flagfftReal> h_out(kN);
  flagfft::adaptor::Memory in_memory(kInputCount * sizeof(flagfftComplex));
  flagfft::adaptor::Memory out_memory(kN * sizeof(flagfftReal));
  auto* d_in = static_cast<flagfftComplex*>(in_memory.data());
  auto* d_out = static_cast<flagfftReal*>(out_memory.data());
  ASSERT_NE(d_in, nullptr);
  ASSERT_NE(d_out, nullptr);
  in_memory.copy_from_host(h_in.data(), kInputCount * sizeof(flagfftComplex));
  ExecC2R(plan, d_in, d_out);
  out_memory.copy_to_host(h_out.data(), kN * sizeof(flagfftReal));
  for (flagfftReal value : h_out) {
    EXPECT_EQ(value, 0.0f);
    EXPECT_TRUE(std::isfinite(value));
  }
  EXPECT_EQ(flagfftDestroy(plan), FLAGFFT_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(All,
                         C2RCTSingle_Test,
                         ::testing::ValuesIn(override_params(Generate1DParamsCTAllSingle())),
                         [](const auto& info) {
                           return std::to_string(info.param.N) + "x" + std::to_string(info.param.batch);
                         });
