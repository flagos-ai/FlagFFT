#include "flagfft_test.h"

using namespace flagfft_test;

class C2CInvCTBatch_Test : public ::testing::TestWithParam<Test1DParam> {
 protected:
  void SetUp() override {
    auto p = GetParam();
    N = p.N;
    batch = p.batch;
    total = N * batch;

    plan = nullptr;
    Plan1d(&plan, N, FLAGFFT_C2C, batch);

    h_in = random_complex(total, accuracy_seed(FLAGFFT_C2C, N, batch));

    auto bytes = total * sizeof(flagfftComplex);
    in_memory.allocate(bytes);
    out_memory.allocate(bytes);
    ref_memory.allocate(bytes);
    d_in = static_cast<flagfftComplex*>(in_memory.data());
    d_out = static_cast<flagfftComplex*>(out_memory.data());
    d_ref = static_cast<flagfftComplex*>(ref_memory.data());
    ASSERT_NE(d_in, nullptr);
    ASSERT_NE(d_out, nullptr);
    ASSERT_NE(d_ref, nullptr);

    in_memory.copy_from_host(h_in.data(), bytes);
  }

  void TearDown() override {
    if (plan) flagfftDestroy(plan);
  }

  int N = 0;
  int batch = 0;
  int total = 0;
  flagfftHandle plan = nullptr;
  std::vector<flagfftComplex> h_in;
  flagfft::adaptor::Memory in_memory;
  flagfft::adaptor::Memory out_memory;
  flagfft::adaptor::Memory ref_memory;
  flagfftComplex* d_in = nullptr;
  flagfftComplex* d_out = nullptr;
  flagfftComplex* d_ref = nullptr;
};

TEST_P(C2CInvCTBatch_Test, InverseVsReference) {
  if (should_skip_direction(FLAGFFT_INVERSE)) {
    GTEST_SKIP() << "direction=inverse skipped by --direction flag";
  }
  RefPlanHandle ref;
  ref_plan_1d(ref, N, FLAGFFT_C2C, batch);
  std::vector<flagfftComplex> h_out(total);
  std::vector<flagfftComplex> h_ref(total);
  for (double scale : kAccuracyInputScales) {
    auto input = h_in;
    scale_input(input, scale);
    in_memory.copy_from_host(input.data(), total * sizeof(flagfftComplex));
    ExecC2C(plan, d_in, d_out, FLAGFFT_INVERSE);
    ref_exec_c2c(ref, d_in, d_ref, FLAGFFT_INVERSE);
    out_memory.copy_to_host(h_out.data(), total * sizeof(flagfftComplex));
    ref_memory.copy_to_host(h_ref.data(), total * sizeof(flagfftComplex));
    expect_reference_accuracy(error_stats(h_out.data(), h_ref.data(), N, batch),
                              FLAGFFT_C2C,
                              N,
                              batch,
                              input_scale_name(scale));
  }
}

TEST_P(C2CInvCTBatch_Test, Roundtrip) {
  auto* d_mid = d_ref;
  ExecC2C(plan, d_in, d_mid, FLAGFFT_FORWARD);
  ExecC2C(plan, d_mid, d_out, FLAGFFT_INVERSE);

  std::vector<flagfftComplex> h_out(total);
  std::vector<flagfftComplex> h_expected(total);
  out_memory.copy_to_host(h_out.data(), total * sizeof(flagfftComplex));

  for (int i = 0; i < total; ++i) {
    h_expected[i].x = h_in[i].x * N;
    h_expected[i].y = h_in[i].y * N;
  }
  expect_roundtrip_accuracy(error_stats(h_out.data(), h_expected.data(), N, batch),
                            FLAGFFT_C2C,
                            FLAGFFT_C2C,
                            N,
                            batch);
}

INSTANTIATE_TEST_SUITE_P(All,
                         C2CInvCTBatch_Test,
                         ::testing::ValuesIn(override_params(Generate1DParamsCTExtendedSmallBatch())),
                         [](const auto& info) {
                           return std::to_string(info.param.N) + "x" + std::to_string(info.param.batch);
                         });
