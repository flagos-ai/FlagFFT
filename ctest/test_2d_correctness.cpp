#include "flagfft_test.h"

#include <cstring>
#include <exception>
#include <vector>

namespace {

struct Test2DSize {
  int n0;
  int n1;
};

constexpr Test2DSize k2DSmoke[] = {
    {16, 16},
};

constexpr Test2DSize k2DSizes[] = {
    { 16,  16},
    { 64,  64},
    {128, 256},
    {256, 128},
    {256, 256},
};

// Non-smooth sizes to exercise Bluestein path (one or both dimensions).
constexpr Test2DSize k2DBluesteinSizes[] = {
    { 23,  23},
    {997,  16},
    { 16, 997},
};

constexpr int k2DNumSizes = sizeof(k2DSizes) / sizeof(k2DSizes[0]);
constexpr int k2DBatchSingle[] = {1};
constexpr int k2DBatchMulti[] = {4};

bool HasUsableDevice() {
  try {
    return flagfft::adaptor::device_count() > 0;
  } catch (const std::exception&) {
    return false;
  }
}

struct Test2DParam {
  int n0;
  int n1;
  int batch;
};

std::vector<Test2DParam> Generate2DParams(const Test2DSize* sizes,
                                          int numSizes,
                                          const int* batches,
                                          int numBatches) {
  std::vector<Test2DParam> params;
  for (int i = 0; i < numSizes; ++i)
    for (int b = 0; b < numBatches; ++b) params.push_back({sizes[i].n0, sizes[i].n1, batches[b]});
  return params;
}

// =========================================================================
// C2C 2D tests
// =========================================================================

class C2C2D : public ::testing::TestWithParam<Test2DParam> {
 protected:
  void SetUp() override {
    if (!HasUsableDevice()) return;
    n0 = GetParam().n0;
    n1 = GetParam().n1;
    batch = GetParam().batch;
    total = n0 * n1;
    bytes = total * batch * sizeof(flagfftComplex);

    flagfft_test::PlanMany2d(&plan, n0, n1, FLAGFFT_C2C, batch);

    std::uint64_t seed = flagfft_test::accuracy_seed(FLAGFFT_C2C, total, batch);
    h_in = flagfft_test::random_complex(total * batch, seed);
    h_out.resize(total * batch);
    h_roundtrip.resize(total * batch);

    in_mem.allocate(bytes);
    out_mem.allocate(bytes);
    d_in = static_cast<flagfftComplex*>(in_mem.data());
    d_out = static_cast<flagfftComplex*>(out_mem.data());
    in_mem.copy_from_host(h_in.data(), bytes);
  }

  void TearDown() override {
    if (plan) flagfftDestroy(plan);
  }

  int n0 = 0, n1 = 0, batch = 0, total = 0;
  std::size_t bytes = 0;
  flagfftHandle plan = nullptr;
  std::vector<flagfftComplex> h_in, h_out, h_roundtrip;
  flagfft::adaptor::Memory in_mem, out_mem;
  flagfftComplex* d_in = nullptr;
  flagfftComplex* d_out = nullptr;
};

TEST_P(C2C2D, ForwardInverse) {
  if (!HasUsableDevice()) GTEST_SKIP() << "no device";

  // Forward
  flagfft_test::ExecC2C(plan, d_in, d_out, FLAGFFT_FORWARD);
  out_mem.copy_to_host(h_out.data(), bytes);

  // Copy output to input for inverse
  in_mem.copy_from_host(h_out.data(), bytes);

  // Inverse
  flagfft_test::ExecC2C(plan, d_in, d_out, FLAGFFT_INVERSE);
  out_mem.copy_to_host(h_roundtrip.data(), bytes);

  // Inverse FFT doesn't normalize, so expected = input * N
  const int N = total;
  std::vector<flagfftComplex> h_expected(total * batch);
  for (int i = 0; i < total * batch; ++i) {
    h_expected[i].x = h_in[i].x * N;
    h_expected[i].y = h_in[i].y * N;
  }

  flagfft_test::ErrorStats stats =
      flagfft_test::error_stats(h_roundtrip.data(), h_expected.data(), total, batch);
  flagfft_test::expect_roundtrip_accuracy(stats, FLAGFFT_C2C, FLAGFFT_C2C, total, batch);
}

TEST_P(C2C2D, ForwardReference) {
  if (!HasUsableDevice()) GTEST_SKIP() << "no device";

  flagfft_test::RefPlanHandle ref_plan;
  flagfft_test::ref_plan_2d(ref_plan, n0, n1, FLAGFFT_C2C);

  flagfft::adaptor::Memory ref_in_mem(total * sizeof(flagfftComplex));
  flagfft::adaptor::Memory ref_out_mem(total * sizeof(flagfftComplex));
  auto* d_ref_in = static_cast<flagfftComplex*>(ref_in_mem.data());
  auto* d_ref_out = static_cast<flagfftComplex*>(ref_out_mem.data());

  for (double scale : flagfft_test::kAccuracyInputScales) {
    auto input = h_in;
    flagfft_test::scale_input(input, scale);
    in_mem.copy_from_host(input.data(), bytes);

    // FlagFFT forward
    flagfft_test::ExecC2C(plan, d_in, d_out, FLAGFFT_FORWARD);
    out_mem.copy_to_host(h_out.data(), bytes);

    // Reference forward - process each batch separately
    std::vector<flagfftComplex> h_ref(total * batch);
    for (int b = 0; b < batch; ++b) {
      ref_in_mem.copy_from_host(input.data() + b * total, total * sizeof(flagfftComplex));
      flagfft_test::ref_exec_c2c(ref_plan, d_ref_in, d_ref_out, FLAGFFT_FORWARD);
      ref_out_mem.copy_to_host(h_ref.data() + b * total, total * sizeof(flagfftComplex));
    }

    flagfft_test::ErrorStats stats = flagfft_test::error_stats(h_out.data(), h_ref.data(), total, batch);
    flagfft_test::expect_reference_accuracy(stats,
                                            FLAGFFT_C2C,
                                            total,
                                            batch,
                                            flagfft_test::input_scale_name(scale));
  }
}

TEST_P(C2C2D, InverseReference) {
  if (!HasUsableDevice()) GTEST_SKIP() << "no device";

  flagfft_test::RefPlanHandle ref_plan;
  flagfft_test::ref_plan_2d(ref_plan, n0, n1, FLAGFFT_C2C);

  flagfft::adaptor::Memory ref_in_mem(total * sizeof(flagfftComplex));
  flagfft::adaptor::Memory ref_out_mem(total * sizeof(flagfftComplex));
  auto* d_ref_in = static_cast<flagfftComplex*>(ref_in_mem.data());
  auto* d_ref_out = static_cast<flagfftComplex*>(ref_out_mem.data());

  for (double scale : flagfft_test::kAccuracyInputScales) {
    auto input = h_in;
    flagfft_test::scale_input(input, scale);
    in_mem.copy_from_host(input.data(), bytes);

    // FlagFFT inverse
    flagfft_test::ExecC2C(plan, d_in, d_out, FLAGFFT_INVERSE);
    out_mem.copy_to_host(h_out.data(), bytes);

    // Reference inverse - process each batch separately
    std::vector<flagfftComplex> h_ref(total * batch);
    for (int b = 0; b < batch; ++b) {
      ref_in_mem.copy_from_host(input.data() + b * total, total * sizeof(flagfftComplex));
      flagfft_test::ref_exec_c2c(ref_plan, d_ref_in, d_ref_out, FLAGFFT_INVERSE);
      ref_out_mem.copy_to_host(h_ref.data() + b * total, total * sizeof(flagfftComplex));
    }

    flagfft_test::ErrorStats stats = flagfft_test::error_stats(h_out.data(), h_ref.data(), total, batch);
    flagfft_test::expect_reference_accuracy(stats,
                                            FLAGFFT_C2C,
                                            total,
                                            batch,
                                            flagfft_test::input_scale_name(scale));
  }
}

INSTANTIATE_TEST_SUITE_P(Smoke, C2C2D, ::testing::ValuesIn(Generate2DParams(k2DSmoke, 1, k2DBatchSingle, 1)));

INSTANTIATE_TEST_SUITE_P(Extended,
                         C2C2D,
                         ::testing::ValuesIn(Generate2DParams(k2DSizes, k2DNumSizes, k2DBatchSingle, 1)));

INSTANTIATE_TEST_SUITE_P(Batch,
                         C2C2D,
                         ::testing::ValuesIn(Generate2DParams(k2DSizes, k2DNumSizes, k2DBatchMulti, 1)));

INSTANTIATE_TEST_SUITE_P(
    Bluestein,
    C2C2D,
    ::testing::ValuesIn(Generate2DParams(
        k2DBluesteinSizes, sizeof(k2DBluesteinSizes) / sizeof(k2DBluesteinSizes[0]), k2DBatchSingle, 1)));

// =========================================================================
// Z2Z 2D tests
// =========================================================================

class Z2Z2D : public ::testing::TestWithParam<Test2DParam> {
 protected:
  void SetUp() override {
    if (!HasUsableDevice()) return;
    n0 = GetParam().n0;
    n1 = GetParam().n1;
    batch = GetParam().batch;
    total = n0 * n1;
    bytes = total * batch * sizeof(flagfftDoubleComplex);

    flagfft_test::PlanMany2d(&plan, n0, n1, FLAGFFT_Z2Z, batch);

    std::uint64_t seed = flagfft_test::accuracy_seed(FLAGFFT_Z2Z, total, batch);
    h_in = flagfft_test::random_double_complex(total * batch, seed);
    h_out.resize(total * batch);
    h_roundtrip.resize(total * batch);

    in_mem.allocate(bytes);
    out_mem.allocate(bytes);
    d_in = static_cast<flagfftDoubleComplex*>(in_mem.data());
    d_out = static_cast<flagfftDoubleComplex*>(out_mem.data());
    in_mem.copy_from_host(h_in.data(), bytes);
  }

  void TearDown() override {
    if (plan) flagfftDestroy(plan);
  }

  int n0 = 0, n1 = 0, batch = 0, total = 0;
  std::size_t bytes = 0;
  flagfftHandle plan = nullptr;
  std::vector<flagfftDoubleComplex> h_in, h_out, h_roundtrip;
  flagfft::adaptor::Memory in_mem, out_mem;
  flagfftDoubleComplex* d_in = nullptr;
  flagfftDoubleComplex* d_out = nullptr;
};

TEST_P(Z2Z2D, ForwardInverse) {
  if (!HasUsableDevice()) GTEST_SKIP() << "no device";

  flagfft_test::ExecZ2Z(plan, d_in, d_out, FLAGFFT_FORWARD);
  out_mem.copy_to_host(h_out.data(), bytes);
  in_mem.copy_from_host(h_out.data(), bytes);

  flagfft_test::ExecZ2Z(plan, d_in, d_out, FLAGFFT_INVERSE);
  out_mem.copy_to_host(h_roundtrip.data(), bytes);

  const int N = total;
  std::vector<flagfftDoubleComplex> h_expected(total * batch);
  for (int i = 0; i < total * batch; ++i) {
    h_expected[i].x = h_in[i].x * N;
    h_expected[i].y = h_in[i].y * N;
  }

  flagfft_test::ErrorStats stats =
      flagfft_test::error_stats(h_roundtrip.data(), h_expected.data(), total, batch);
  flagfft_test::expect_roundtrip_accuracy(stats, FLAGFFT_Z2Z, FLAGFFT_Z2Z, total, batch);
}

TEST_P(Z2Z2D, ForwardReference) {
  if (!HasUsableDevice()) GTEST_SKIP() << "no device";

  flagfft_test::RefPlanHandle ref_plan;
  flagfft_test::ref_plan_2d(ref_plan, n0, n1, FLAGFFT_Z2Z);

  flagfft::adaptor::Memory ref_in_mem(total * sizeof(flagfftDoubleComplex));
  flagfft::adaptor::Memory ref_out_mem(total * sizeof(flagfftDoubleComplex));
  auto* d_ref_in = static_cast<flagfftDoubleComplex*>(ref_in_mem.data());
  auto* d_ref_out = static_cast<flagfftDoubleComplex*>(ref_out_mem.data());

  for (double scale : flagfft_test::kAccuracyInputScales) {
    auto input = h_in;
    flagfft_test::scale_input(input, scale);
    in_mem.copy_from_host(input.data(), bytes);

    flagfft_test::ExecZ2Z(plan, d_in, d_out, FLAGFFT_FORWARD);
    out_mem.copy_to_host(h_out.data(), bytes);

    std::vector<flagfftDoubleComplex> h_ref(total * batch);
    for (int b = 0; b < batch; ++b) {
      ref_in_mem.copy_from_host(input.data() + b * total, total * sizeof(flagfftDoubleComplex));
      flagfft_test::ref_exec_z2z(ref_plan, d_ref_in, d_ref_out, FLAGFFT_FORWARD);
      ref_out_mem.copy_to_host(h_ref.data() + b * total, total * sizeof(flagfftDoubleComplex));
    }

    flagfft_test::ErrorStats stats = flagfft_test::error_stats(h_out.data(), h_ref.data(), total, batch);
    flagfft_test::expect_reference_accuracy(stats,
                                            FLAGFFT_Z2Z,
                                            total,
                                            batch,
                                            flagfft_test::input_scale_name(scale));
  }
}

TEST_P(Z2Z2D, InverseReference) {
  if (!HasUsableDevice()) GTEST_SKIP() << "no device";

  flagfft_test::RefPlanHandle ref_plan;
  flagfft_test::ref_plan_2d(ref_plan, n0, n1, FLAGFFT_Z2Z);

  flagfft::adaptor::Memory ref_in_mem(total * sizeof(flagfftDoubleComplex));
  flagfft::adaptor::Memory ref_out_mem(total * sizeof(flagfftDoubleComplex));
  auto* d_ref_in = static_cast<flagfftDoubleComplex*>(ref_in_mem.data());
  auto* d_ref_out = static_cast<flagfftDoubleComplex*>(ref_out_mem.data());

  for (double scale : flagfft_test::kAccuracyInputScales) {
    auto input = h_in;
    flagfft_test::scale_input(input, scale);
    in_mem.copy_from_host(input.data(), bytes);

    flagfft_test::ExecZ2Z(plan, d_in, d_out, FLAGFFT_INVERSE);
    out_mem.copy_to_host(h_out.data(), bytes);

    std::vector<flagfftDoubleComplex> h_ref(total * batch);
    for (int b = 0; b < batch; ++b) {
      ref_in_mem.copy_from_host(input.data() + b * total, total * sizeof(flagfftDoubleComplex));
      flagfft_test::ref_exec_z2z(ref_plan, d_ref_in, d_ref_out, FLAGFFT_INVERSE);
      ref_out_mem.copy_to_host(h_ref.data() + b * total, total * sizeof(flagfftDoubleComplex));
    }

    flagfft_test::ErrorStats stats = flagfft_test::error_stats(h_out.data(), h_ref.data(), total, batch);
    flagfft_test::expect_reference_accuracy(stats,
                                            FLAGFFT_Z2Z,
                                            total,
                                            batch,
                                            flagfft_test::input_scale_name(scale));
  }
}

INSTANTIATE_TEST_SUITE_P(Smoke, Z2Z2D, ::testing::ValuesIn(Generate2DParams(k2DSmoke, 1, k2DBatchSingle, 1)));

INSTANTIATE_TEST_SUITE_P(Extended,
                         Z2Z2D,
                         ::testing::ValuesIn(Generate2DParams(k2DSizes, k2DNumSizes, k2DBatchSingle, 1)));

INSTANTIATE_TEST_SUITE_P(Batch,
                         Z2Z2D,
                         ::testing::ValuesIn(Generate2DParams(k2DSizes, k2DNumSizes, k2DBatchMulti, 1)));

INSTANTIATE_TEST_SUITE_P(
    Bluestein,
    Z2Z2D,
    ::testing::ValuesIn(Generate2DParams(
        k2DBluesteinSizes, sizeof(k2DBluesteinSizes) / sizeof(k2DBluesteinSizes[0]), k2DBatchSingle, 1)));

// =========================================================================
// R2C + C2R 2D roundtrip tests
// =========================================================================

class R2CC2R2D : public ::testing::TestWithParam<Test2DParam> {
 protected:
  void SetUp() override {
    // R2C/C2R 2D not yet supported
  }

  void TearDown() override {
  }
};

TEST_P(R2CC2R2D, Roundtrip) {
  GTEST_SKIP() << "R2C/C2R 2D not yet supported";
}

TEST_P(R2CC2R2D, ForwardReference) {
  GTEST_SKIP() << "R2C/C2R 2D not yet supported";
}

INSTANTIATE_TEST_SUITE_P(Smoke, R2CC2R2D, ::testing::Values(Test2DParam {16, 16, 1}));

INSTANTIATE_TEST_SUITE_P(Extended,
                         R2CC2R2D,
                         ::testing::ValuesIn(Generate2DParams(k2DSizes, k2DNumSizes, k2DBatchSingle, 1)));

// =========================================================================
// D2Z + Z2D 2D roundtrip tests
// =========================================================================

class D2ZZ2D2D : public ::testing::TestWithParam<Test2DParam> {
 protected:
  void SetUp() override {
    // D2Z/Z2D 2D not yet supported
  }

  void TearDown() override {
  }
};

TEST_P(D2ZZ2D2D, Roundtrip) {
  GTEST_SKIP() << "D2Z/Z2D 2D not yet supported";
}

TEST_P(D2ZZ2D2D, ForwardReference) {
  GTEST_SKIP() << "D2Z/Z2D 2D not yet supported";
}

INSTANTIATE_TEST_SUITE_P(Smoke, D2ZZ2D2D, ::testing::Values(Test2DParam {16, 16, 1}));

INSTANTIATE_TEST_SUITE_P(Extended,
                         D2ZZ2D2D,
                         ::testing::ValuesIn(Generate2DParams(k2DSizes, k2DNumSizes, k2DBatchSingle, 1)));

}  // namespace
