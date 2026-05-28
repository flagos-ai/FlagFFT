#include "flagfft_test.h"

using namespace flagfft_test;

class Z2ZFwdBSBatch_Test : public ::testing::TestWithParam<Test1DParam> {
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
    in_memory.allocate(bytes);
    out_memory.allocate(bytes);
    ref_memory.allocate(bytes);
    d_in = static_cast<flagfftDoubleComplex*>(in_memory.data());
    d_out = static_cast<flagfftDoubleComplex*>(out_memory.data());
    d_ref = static_cast<flagfftDoubleComplex*>(ref_memory.data());
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
  std::vector<flagfftDoubleComplex> h_in;
  flagfft::adaptor::Memory in_memory;
  flagfft::adaptor::Memory out_memory;
  flagfft::adaptor::Memory ref_memory;
  flagfftDoubleComplex* d_in = nullptr;
  flagfftDoubleComplex* d_out = nullptr;
  flagfftDoubleComplex* d_ref = nullptr;
};

TEST_P(Z2ZFwdBSBatch_Test, ForwardVsReference) {
  RefPlanHandle ref;
  ref_plan_1d(ref, N, FLAGFFT_Z2Z, batch);
  std::vector<flagfftDoubleComplex> h_out(total);
  std::vector<flagfftDoubleComplex> h_ref(total);
  for (double scale : kAccuracyInputScales) {
    auto input = h_in;
    scale_input(input, scale);
    in_memory.copy_from_host(input.data(), total * sizeof(flagfftDoubleComplex));
    ExecZ2Z(plan, d_in, d_out, FLAGFFT_FORWARD);
    ref_exec_z2z(ref, d_in, d_ref, FLAGFFT_FORWARD);
    out_memory.copy_to_host(h_out.data(), total * sizeof(flagfftDoubleComplex));
    ref_memory.copy_to_host(h_ref.data(), total * sizeof(flagfftDoubleComplex));
    expect_reference_accuracy(error_stats(h_out.data(), h_ref.data(), N, batch),
                              FLAGFFT_Z2Z,
                              N,
                              batch,
                              input_scale_name(scale));
  }
}

INSTANTIATE_TEST_SUITE_P(ExtendedSmall,
                         Z2ZFwdBSBatch_Test,
                         ::testing::ValuesIn(Generate1DParamsBSExtendedSmallBatch()),
                         [](const auto& info) {
                           return std::to_string(info.param.N) + "x" + std::to_string(info.param.batch);
                         });
INSTANTIATE_TEST_SUITE_P(ExtendedMedium,
                         Z2ZFwdBSBatch_Test,
                         ::testing::ValuesIn(Generate1DParamsBSExtendedMediumBatch()),
                         [](const auto& info) {
                           return std::to_string(info.param.N) + "x" + std::to_string(info.param.batch);
                         });
INSTANTIATE_TEST_SUITE_P(ExtendedLarge,
                         Z2ZFwdBSBatch_Test,
                         ::testing::ValuesIn(Generate1DParamsBSExtendedLargeBatch()),
                         [](const auto& info) {
                           return std::to_string(info.param.N) + "x" + std::to_string(info.param.batch);
                         });
