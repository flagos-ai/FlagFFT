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
constexpr int k2DNumBluesteinSizes = sizeof(k2DBluesteinSizes) / sizeof(k2DBluesteinSizes[0]);
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
// R2C 2D tests
// =========================================================================

class R2C2D : public ::testing::TestWithParam<Test2DParam> {
 protected:
  void SetUp() override {
    if (!HasUsableDevice()) return;
    n0 = GetParam().n0;
    n1 = GetParam().n1;
    batch = GetParam().batch;
    total_in = n0 * n1;
    total_out = n0 * (n1 / 2 + 1);

    // Use PlanMany to support batch > 1
    int n[2] = {n0, n1};
    const int idist = n0 * n1;
    const int odist = n0 * (n1 / 2 + 1);
    flagfftResult r = flagfftPlanMany(&plan, 2, n, nullptr, 1, idist, nullptr, 1, odist, FLAGFFT_R2C, batch);
    if (r != FLAGFFT_SUCCESS) {
      FAIL() << "flagfftPlanMany(R2C, n0=" << n0 << ", n1=" << n1 << ", batch=" << batch
             << ") failed with code " << r;
    }

    std::uint64_t seed = flagfft_test::accuracy_seed(FLAGFFT_R2C, total_in, batch);
    h_in = flagfft_test::random_real(total_in * batch, seed);
    h_out.resize(total_out * batch);

    in_mem.allocate(total_in * batch * sizeof(flagfftReal));
    out_mem.allocate(total_out * batch * sizeof(flagfftComplex));
    d_in = static_cast<flagfftReal*>(in_mem.data());
    d_out = static_cast<flagfftComplex*>(out_mem.data());
    in_mem.copy_from_host(h_in.data(), total_in * batch * sizeof(flagfftReal));
  }

  void TearDown() override {
    if (plan) flagfftDestroy(plan);
  }

  int n0 = 0, n1 = 0, batch = 0;
  int total_in = 0, total_out = 0;
  flagfftHandle plan = nullptr;
  std::vector<flagfftReal> h_in;
  std::vector<flagfftComplex> h_out;
  flagfft::adaptor::Memory in_mem, out_mem;
  flagfftReal* d_in = nullptr;
  flagfftComplex* d_out = nullptr;
};

TEST_P(R2C2D, ForwardReference) {
  if (!HasUsableDevice()) GTEST_SKIP() << "no device";

  flagfft_test::RefPlanHandle ref_plan;
  flagfft_test::ref_plan_2d(ref_plan, n0, n1, FLAGFFT_R2C);

  flagfft::adaptor::Memory ref_in_mem(total_in * sizeof(flagfftReal));
  flagfft::adaptor::Memory ref_out_mem(total_out * sizeof(flagfftComplex));
  auto* d_ref_in = static_cast<flagfftReal*>(ref_in_mem.data());
  auto* d_ref_out = static_cast<flagfftComplex*>(ref_out_mem.data());

  for (double scale : flagfft_test::kAccuracyInputScales) {
    auto input = h_in;
    flagfft_test::scale_input(input, scale);
    in_mem.copy_from_host(input.data(), total_in * batch * sizeof(flagfftReal));

    // FlagFFT forward
    flagfft_test::ExecR2C(plan, d_in, d_out);
    out_mem.copy_to_host(h_out.data(), total_out * batch * sizeof(flagfftComplex));

    // Reference forward - process each batch separately
    std::vector<flagfftComplex> h_ref(total_out * batch);
    for (int b = 0; b < batch; ++b) {
      ref_in_mem.copy_from_host(input.data() + b * total_in, total_in * sizeof(flagfftReal));
      flagfft_test::ref_exec_r2c(ref_plan, d_ref_in, d_ref_out);
      ref_out_mem.copy_to_host(h_ref.data() + b * total_out, total_out * sizeof(flagfftComplex));
    }

    flagfft_test::ErrorStats stats = flagfft_test::error_stats(h_out.data(), h_ref.data(), total_out, batch);
    flagfft_test::expect_reference_accuracy(stats,
                                            FLAGFFT_R2C,
                                            total_in,
                                            batch,
                                            flagfft_test::input_scale_name(scale));
  }
}

INSTANTIATE_TEST_SUITE_P(Smoke, R2C2D, ::testing::ValuesIn(Generate2DParams(k2DSmoke, 1, k2DBatchSingle, 1)));

INSTANTIATE_TEST_SUITE_P(Extended,
                         R2C2D,
                         ::testing::ValuesIn(Generate2DParams(k2DSizes, k2DNumSizes, k2DBatchSingle, 1)));

INSTANTIATE_TEST_SUITE_P(Batch,
                         R2C2D,
                         ::testing::ValuesIn(Generate2DParams(k2DSizes, k2DNumSizes, k2DBatchMulti, 1)));

INSTANTIATE_TEST_SUITE_P(
    Bluestein,
    R2C2D,
    ::testing::ValuesIn(Generate2DParams(k2DBluesteinSizes, k2DNumBluesteinSizes, k2DBatchSingle, 1)));

// =========================================================================
// C2R 2D tests
// =========================================================================

class C2R2D : public ::testing::TestWithParam<Test2DParam> {
 protected:
  void SetUp() override {
    if (!HasUsableDevice()) return;
    n0 = GetParam().n0;
    n1 = GetParam().n1;
    batch = GetParam().batch;
    total_in = n0 * (n1 / 2 + 1);
    total_out = n0 * n1;

    // Use PlanMany to support batch > 1
    int n[2] = {n0, n1};
    const int idist = n0 * (n1 / 2 + 1);
    const int odist = n0 * n1;
    flagfftResult r = flagfftPlanMany(&plan, 2, n, nullptr, 1, idist, nullptr, 1, odist, FLAGFFT_C2R, batch);
    if (r != FLAGFFT_SUCCESS) {
      FAIL() << "flagfftPlanMany(C2R, n0=" << n0 << ", n1=" << n1 << ", batch=" << batch
             << ") failed with code " << r;
    }

    std::uint64_t seed = flagfft_test::accuracy_seed(FLAGFFT_C2R, total_out, batch);
    h_in = flagfft_test::random_complex(total_in * batch, seed);
    h_out.resize(total_out * batch);

    in_mem.allocate(total_in * batch * sizeof(flagfftComplex));
    out_mem.allocate(total_out * batch * sizeof(flagfftReal));
    d_in = static_cast<flagfftComplex*>(in_mem.data());
    d_out = static_cast<flagfftReal*>(out_mem.data());
    in_mem.copy_from_host(h_in.data(), total_in * batch * sizeof(flagfftComplex));
  }

  void TearDown() override {
    if (plan) flagfftDestroy(plan);
  }

  int n0 = 0, n1 = 0, batch = 0;
  int total_in = 0, total_out = 0;
  flagfftHandle plan = nullptr;
  std::vector<flagfftComplex> h_in;
  std::vector<flagfftReal> h_out;
  flagfft::adaptor::Memory in_mem, out_mem;
  flagfftComplex* d_in = nullptr;
  flagfftReal* d_out = nullptr;
};

TEST_P(C2R2D, InverseReference) {
  if (!HasUsableDevice()) GTEST_SKIP() << "no device";

  flagfft_test::RefPlanHandle ref_plan;
  flagfft_test::ref_plan_2d(ref_plan, n0, n1, FLAGFFT_C2R);

  flagfft::adaptor::Memory ref_in_mem(total_in * sizeof(flagfftComplex));
  flagfft::adaptor::Memory ref_out_mem(total_out * sizeof(flagfftReal));
  auto* d_ref_in = static_cast<flagfftComplex*>(ref_in_mem.data());
  auto* d_ref_out = static_cast<flagfftReal*>(ref_out_mem.data());

  for (double scale : flagfft_test::kAccuracyInputScales) {
    auto input = h_in;
    flagfft_test::scale_input(input, scale);
    in_mem.copy_from_host(input.data(), total_in * batch * sizeof(flagfftComplex));

    // FlagFFT inverse
    flagfft_test::ExecC2R(plan, d_in, d_out);
    out_mem.copy_to_host(h_out.data(), total_out * batch * sizeof(flagfftReal));

    // Reference inverse - process each batch separately
    std::vector<flagfftReal> h_ref(total_out * batch);
    for (int b = 0; b < batch; ++b) {
      ref_in_mem.copy_from_host(input.data() + b * total_in, total_in * sizeof(flagfftComplex));
      flagfft_test::ref_exec_c2r(ref_plan, d_ref_in, d_ref_out);
      ref_out_mem.copy_to_host(h_ref.data() + b * total_out, total_out * sizeof(flagfftReal));
    }

    flagfft_test::ErrorStats stats = flagfft_test::error_stats(h_out.data(), h_ref.data(), total_out, batch);
    flagfft_test::expect_reference_accuracy(stats,
                                            FLAGFFT_C2R,
                                            total_out,
                                            batch,
                                            flagfft_test::input_scale_name(scale));
  }
}

INSTANTIATE_TEST_SUITE_P(Smoke, C2R2D, ::testing::ValuesIn(Generate2DParams(k2DSmoke, 1, k2DBatchSingle, 1)));

INSTANTIATE_TEST_SUITE_P(Extended,
                         C2R2D,
                         ::testing::ValuesIn(Generate2DParams(k2DSizes, k2DNumSizes, k2DBatchSingle, 1)));

INSTANTIATE_TEST_SUITE_P(Batch,
                         C2R2D,
                         ::testing::ValuesIn(Generate2DParams(k2DSizes, k2DNumSizes, k2DBatchMulti, 1)));

INSTANTIATE_TEST_SUITE_P(
    Bluestein,
    C2R2D,
    ::testing::ValuesIn(Generate2DParams(k2DBluesteinSizes, k2DNumBluesteinSizes, k2DBatchSingle, 1)));

// =========================================================================
// D2Z 2D tests
// =========================================================================

class D2Z2D : public ::testing::TestWithParam<Test2DParam> {
 protected:
  void SetUp() override {
    if (!HasUsableDevice()) return;
    n0 = GetParam().n0;
    n1 = GetParam().n1;
    batch = GetParam().batch;
    total_in = n0 * n1;
    total_out = n0 * (n1 / 2 + 1);

    // Use PlanMany to support batch > 1
    int n[2] = {n0, n1};
    const int idist = n0 * n1;
    const int odist = n0 * (n1 / 2 + 1);
    flagfftResult r = flagfftPlanMany(&plan, 2, n, nullptr, 1, idist, nullptr, 1, odist, FLAGFFT_D2Z, batch);
    if (r != FLAGFFT_SUCCESS) {
      FAIL() << "flagfftPlanMany(D2Z, n0=" << n0 << ", n1=" << n1 << ", batch=" << batch
             << ") failed with code " << r;
    }

    std::uint64_t seed = flagfft_test::accuracy_seed(FLAGFFT_D2Z, total_in, batch);
    h_in = flagfft_test::random_double_real(total_in * batch, seed);
    h_out.resize(total_out * batch);

    in_mem.allocate(total_in * batch * sizeof(flagfftDoubleReal));
    out_mem.allocate(total_out * batch * sizeof(flagfftDoubleComplex));
    d_in = static_cast<flagfftDoubleReal*>(in_mem.data());
    d_out = static_cast<flagfftDoubleComplex*>(out_mem.data());
    in_mem.copy_from_host(h_in.data(), total_in * batch * sizeof(flagfftDoubleReal));
  }

  void TearDown() override {
    if (plan) flagfftDestroy(plan);
  }

  int n0 = 0, n1 = 0, batch = 0;
  int total_in = 0, total_out = 0;
  flagfftHandle plan = nullptr;
  std::vector<flagfftDoubleReal> h_in;
  std::vector<flagfftDoubleComplex> h_out;
  flagfft::adaptor::Memory in_mem, out_mem;
  flagfftDoubleReal* d_in = nullptr;
  flagfftDoubleComplex* d_out = nullptr;
};

TEST_P(D2Z2D, ForwardReference) {
  if (!HasUsableDevice()) GTEST_SKIP() << "no device";

  flagfft_test::RefPlanHandle ref_plan;
  flagfft_test::ref_plan_2d(ref_plan, n0, n1, FLAGFFT_D2Z);

  flagfft::adaptor::Memory ref_in_mem(total_in * sizeof(flagfftDoubleReal));
  flagfft::adaptor::Memory ref_out_mem(total_out * sizeof(flagfftDoubleComplex));
  auto* d_ref_in = static_cast<flagfftDoubleReal*>(ref_in_mem.data());
  auto* d_ref_out = static_cast<flagfftDoubleComplex*>(ref_out_mem.data());

  for (double scale : flagfft_test::kAccuracyInputScales) {
    auto input = h_in;
    flagfft_test::scale_input(input, scale);
    in_mem.copy_from_host(input.data(), total_in * batch * sizeof(flagfftDoubleReal));

    flagfft_test::ExecD2Z(plan, d_in, d_out);
    out_mem.copy_to_host(h_out.data(), total_out * batch * sizeof(flagfftDoubleComplex));

    std::vector<flagfftDoubleComplex> h_ref(total_out * batch);
    for (int b = 0; b < batch; ++b) {
      ref_in_mem.copy_from_host(input.data() + b * total_in, total_in * sizeof(flagfftDoubleReal));
      flagfft_test::ref_exec_d2z(ref_plan, d_ref_in, d_ref_out);
      ref_out_mem.copy_to_host(h_ref.data() + b * total_out, total_out * sizeof(flagfftDoubleComplex));
    }

    flagfft_test::ErrorStats stats = flagfft_test::error_stats(h_out.data(), h_ref.data(), total_out, batch);
    flagfft_test::expect_reference_accuracy(stats,
                                            FLAGFFT_D2Z,
                                            total_in,
                                            batch,
                                            flagfft_test::input_scale_name(scale));
  }
}

INSTANTIATE_TEST_SUITE_P(Smoke, D2Z2D, ::testing::ValuesIn(Generate2DParams(k2DSmoke, 1, k2DBatchSingle, 1)));

INSTANTIATE_TEST_SUITE_P(Extended,
                         D2Z2D,
                         ::testing::ValuesIn(Generate2DParams(k2DSizes, k2DNumSizes, k2DBatchSingle, 1)));

INSTANTIATE_TEST_SUITE_P(Batch,
                         D2Z2D,
                         ::testing::ValuesIn(Generate2DParams(k2DSizes, k2DNumSizes, k2DBatchMulti, 1)));

INSTANTIATE_TEST_SUITE_P(
    Bluestein,
    D2Z2D,
    ::testing::ValuesIn(Generate2DParams(k2DBluesteinSizes, k2DNumBluesteinSizes, k2DBatchSingle, 1)));

// =========================================================================
// Z2D 2D tests
// =========================================================================

class Z2D2D : public ::testing::TestWithParam<Test2DParam> {
 protected:
  void SetUp() override {
    if (!HasUsableDevice()) return;
    n0 = GetParam().n0;
    n1 = GetParam().n1;
    batch = GetParam().batch;
    total_in = n0 * (n1 / 2 + 1);
    total_out = n0 * n1;

    // Use PlanMany to support batch > 1
    int n[2] = {n0, n1};
    const int idist = n0 * (n1 / 2 + 1);
    const int odist = n0 * n1;
    flagfftResult r = flagfftPlanMany(&plan, 2, n, nullptr, 1, idist, nullptr, 1, odist, FLAGFFT_Z2D, batch);
    if (r != FLAGFFT_SUCCESS) {
      FAIL() << "flagfftPlanMany(Z2D, n0=" << n0 << ", n1=" << n1 << ", batch=" << batch
             << ") failed with code " << r;
    }

    std::uint64_t seed = flagfft_test::accuracy_seed(FLAGFFT_Z2D, total_out, batch);
    h_in = flagfft_test::random_double_complex(total_in * batch, seed);
    h_out.resize(total_out * batch);

    in_mem.allocate(total_in * batch * sizeof(flagfftDoubleComplex));
    out_mem.allocate(total_out * batch * sizeof(flagfftDoubleReal));
    d_in = static_cast<flagfftDoubleComplex*>(in_mem.data());
    d_out = static_cast<flagfftDoubleReal*>(out_mem.data());
    in_mem.copy_from_host(h_in.data(), total_in * batch * sizeof(flagfftDoubleComplex));
  }

  void TearDown() override {
    if (plan) flagfftDestroy(plan);
  }

  int n0 = 0, n1 = 0, batch = 0;
  int total_in = 0, total_out = 0;
  flagfftHandle plan = nullptr;
  std::vector<flagfftDoubleComplex> h_in;
  std::vector<flagfftDoubleReal> h_out;
  flagfft::adaptor::Memory in_mem, out_mem;
  flagfftDoubleComplex* d_in = nullptr;
  flagfftDoubleReal* d_out = nullptr;
};

TEST_P(Z2D2D, InverseReference) {
  if (!HasUsableDevice()) GTEST_SKIP() << "no device";

  flagfft_test::RefPlanHandle ref_plan;
  flagfft_test::ref_plan_2d(ref_plan, n0, n1, FLAGFFT_Z2D);

  flagfft::adaptor::Memory ref_in_mem(total_in * sizeof(flagfftDoubleComplex));
  flagfft::adaptor::Memory ref_out_mem(total_out * sizeof(flagfftDoubleReal));
  auto* d_ref_in = static_cast<flagfftDoubleComplex*>(ref_in_mem.data());
  auto* d_ref_out = static_cast<flagfftDoubleReal*>(ref_out_mem.data());

  for (double scale : flagfft_test::kAccuracyInputScales) {
    auto input = h_in;
    flagfft_test::scale_input(input, scale);
    in_mem.copy_from_host(input.data(), total_in * batch * sizeof(flagfftDoubleComplex));

    flagfft_test::ExecZ2D(plan, d_in, d_out);
    out_mem.copy_to_host(h_out.data(), total_out * batch * sizeof(flagfftDoubleReal));

    std::vector<flagfftDoubleReal> h_ref(total_out * batch);
    for (int b = 0; b < batch; ++b) {
      ref_in_mem.copy_from_host(input.data() + b * total_in, total_in * sizeof(flagfftDoubleComplex));
      flagfft_test::ref_exec_z2d(ref_plan, d_ref_in, d_ref_out);
      ref_out_mem.copy_to_host(h_ref.data() + b * total_out, total_out * sizeof(flagfftDoubleReal));
    }

    flagfft_test::ErrorStats stats = flagfft_test::error_stats(h_out.data(), h_ref.data(), total_out, batch);
    flagfft_test::expect_reference_accuracy(stats,
                                            FLAGFFT_Z2D,
                                            total_out,
                                            batch,
                                            flagfft_test::input_scale_name(scale));
  }
}

INSTANTIATE_TEST_SUITE_P(Smoke, Z2D2D, ::testing::ValuesIn(Generate2DParams(k2DSmoke, 1, k2DBatchSingle, 1)));

INSTANTIATE_TEST_SUITE_P(Extended,
                         Z2D2D,
                         ::testing::ValuesIn(Generate2DParams(k2DSizes, k2DNumSizes, k2DBatchSingle, 1)));

INSTANTIATE_TEST_SUITE_P(Batch,
                         Z2D2D,
                         ::testing::ValuesIn(Generate2DParams(k2DSizes, k2DNumSizes, k2DBatchMulti, 1)));

INSTANTIATE_TEST_SUITE_P(
    Bluestein,
    Z2D2D,
    ::testing::ValuesIn(Generate2DParams(k2DBluesteinSizes, k2DNumBluesteinSizes, k2DBatchSingle, 1)));

// =========================================================================
// R2C + C2R roundtrip 2D tests
// =========================================================================

class R2CC2RRoundtrip2D : public ::testing::TestWithParam<Test2DParam> {
 protected:
  void SetUp() override {
    if (!HasUsableDevice()) return;
    n0 = GetParam().n0;
    n1 = GetParam().n1;
    batch = GetParam().batch;
    total_real = n0 * n1;
    total_half = n0 * (n1 / 2 + 1);

    // Use PlanMany to support batch > 1
    int n[2] = {n0, n1};
    const int r2c_idist = n0 * n1;
    const int r2c_odist = n0 * (n1 / 2 + 1);
    const int c2r_idist = n0 * (n1 / 2 + 1);
    const int c2r_odist = n0 * n1;
    flagfftResult r1 =
        flagfftPlanMany(&fwd_plan, 2, n, nullptr, 1, r2c_idist, nullptr, 1, r2c_odist, FLAGFFT_R2C, batch);
    if (r1 != FLAGFFT_SUCCESS) {
      FAIL() << "flagfftPlanMany(R2C) failed with code " << r1;
    }
    flagfftResult r2 =
        flagfftPlanMany(&inv_plan, 2, n, nullptr, 1, c2r_idist, nullptr, 1, c2r_odist, FLAGFFT_C2R, batch);
    if (r2 != FLAGFFT_SUCCESS) {
      FAIL() << "flagfftPlanMany(C2R) failed with code " << r2;
    }

    std::uint64_t seed = flagfft_test::accuracy_seed(FLAGFFT_R2C, total_real, batch);
    h_in = flagfft_test::random_real(total_real * batch, seed);
    h_freq.resize(total_half * batch);
    h_roundtrip.resize(total_real * batch);

    in_mem.allocate(total_real * batch * sizeof(flagfftReal));
    freq_mem.allocate(total_half * batch * sizeof(flagfftComplex));
    out_mem.allocate(total_real * batch * sizeof(flagfftReal));
    d_in = static_cast<flagfftReal*>(in_mem.data());
    d_freq = static_cast<flagfftComplex*>(freq_mem.data());
    d_out = static_cast<flagfftReal*>(out_mem.data());
    in_mem.copy_from_host(h_in.data(), total_real * batch * sizeof(flagfftReal));
  }

  void TearDown() override {
    if (fwd_plan) flagfftDestroy(fwd_plan);
    if (inv_plan) flagfftDestroy(inv_plan);
  }

  int n0 = 0, n1 = 0, batch = 0;
  int total_real = 0, total_half = 0;
  flagfftHandle fwd_plan = nullptr;
  flagfftHandle inv_plan = nullptr;
  std::vector<flagfftReal> h_in, h_roundtrip;
  std::vector<flagfftComplex> h_freq;
  flagfft::adaptor::Memory in_mem, freq_mem, out_mem;
  flagfftReal* d_in = nullptr;
  flagfftComplex* d_freq = nullptr;
  flagfftReal* d_out = nullptr;
};

TEST_P(R2CC2RRoundtrip2D, ForwardInverse) {
  if (!HasUsableDevice()) GTEST_SKIP() << "no device";

  // Forward R2C
  flagfft_test::ExecR2C(fwd_plan, d_in, d_freq);
  freq_mem.copy_to_host(h_freq.data(), total_half * batch * sizeof(flagfftComplex));

  // Inverse C2R
  freq_mem.copy_from_host(h_freq.data(), total_half * batch * sizeof(flagfftComplex));
  flagfft_test::ExecC2R(inv_plan, d_freq, d_out);
  out_mem.copy_to_host(h_roundtrip.data(), total_real * batch * sizeof(flagfftReal));

  // C2R doesn't normalize, so expected = input * N
  const int N = total_real;
  std::vector<flagfftReal> h_expected(total_real * batch);
  for (int i = 0; i < total_real * batch; ++i) {
    h_expected[i] = h_in[i] * N;
  }

  flagfft_test::ErrorStats stats =
      flagfft_test::error_stats(h_roundtrip.data(), h_expected.data(), total_real, batch);
  flagfft_test::expect_roundtrip_accuracy(stats, FLAGFFT_R2C, FLAGFFT_C2R, total_real, batch);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         R2CC2RRoundtrip2D,
                         ::testing::ValuesIn(Generate2DParams(k2DSmoke, 1, k2DBatchSingle, 1)));

INSTANTIATE_TEST_SUITE_P(Extended,
                         R2CC2RRoundtrip2D,
                         ::testing::ValuesIn(Generate2DParams(k2DSizes, k2DNumSizes, k2DBatchSingle, 1)));

INSTANTIATE_TEST_SUITE_P(Batch,
                         R2CC2RRoundtrip2D,
                         ::testing::ValuesIn(Generate2DParams(k2DSizes, k2DNumSizes, k2DBatchMulti, 1)));

// =========================================================================
// D2Z + Z2D roundtrip 2D tests
// =========================================================================

class D2ZZ2DRoundtrip2D : public ::testing::TestWithParam<Test2DParam> {
 protected:
  void SetUp() override {
    if (!HasUsableDevice()) return;
    n0 = GetParam().n0;
    n1 = GetParam().n1;
    batch = GetParam().batch;
    total_real = n0 * n1;
    total_half = n0 * (n1 / 2 + 1);

    // Use PlanMany to support batch > 1
    int n[2] = {n0, n1};
    const int d2z_idist = n0 * n1;
    const int d2z_odist = n0 * (n1 / 2 + 1);
    const int z2d_idist = n0 * (n1 / 2 + 1);
    const int z2d_odist = n0 * n1;
    flagfftResult r1 =
        flagfftPlanMany(&fwd_plan, 2, n, nullptr, 1, d2z_idist, nullptr, 1, d2z_odist, FLAGFFT_D2Z, batch);
    if (r1 != FLAGFFT_SUCCESS) {
      FAIL() << "flagfftPlanMany(D2Z) failed with code " << r1;
    }
    flagfftResult r2 =
        flagfftPlanMany(&inv_plan, 2, n, nullptr, 1, z2d_idist, nullptr, 1, z2d_odist, FLAGFFT_Z2D, batch);
    if (r2 != FLAGFFT_SUCCESS) {
      FAIL() << "flagfftPlanMany(Z2D) failed with code " << r2;
    }

    std::uint64_t seed = flagfft_test::accuracy_seed(FLAGFFT_D2Z, total_real, batch);
    h_in = flagfft_test::random_double_real(total_real * batch, seed);
    h_freq.resize(total_half * batch);
    h_roundtrip.resize(total_real * batch);

    in_mem.allocate(total_real * batch * sizeof(flagfftDoubleReal));
    freq_mem.allocate(total_half * batch * sizeof(flagfftDoubleComplex));
    out_mem.allocate(total_real * batch * sizeof(flagfftDoubleReal));
    d_in = static_cast<flagfftDoubleReal*>(in_mem.data());
    d_freq = static_cast<flagfftDoubleComplex*>(freq_mem.data());
    d_out = static_cast<flagfftDoubleReal*>(out_mem.data());
    in_mem.copy_from_host(h_in.data(), total_real * batch * sizeof(flagfftDoubleReal));
  }

  void TearDown() override {
    if (fwd_plan) flagfftDestroy(fwd_plan);
    if (inv_plan) flagfftDestroy(inv_plan);
  }

  int n0 = 0, n1 = 0, batch = 0;
  int total_real = 0, total_half = 0;
  flagfftHandle fwd_plan = nullptr;
  flagfftHandle inv_plan = nullptr;
  std::vector<flagfftDoubleReal> h_in, h_roundtrip;
  std::vector<flagfftDoubleComplex> h_freq;
  flagfft::adaptor::Memory in_mem, freq_mem, out_mem;
  flagfftDoubleReal* d_in = nullptr;
  flagfftDoubleComplex* d_freq = nullptr;
  flagfftDoubleReal* d_out = nullptr;
};

TEST_P(D2ZZ2DRoundtrip2D, ForwardInverse) {
  if (!HasUsableDevice()) GTEST_SKIP() << "no device";

  // Forward D2Z
  flagfft_test::ExecD2Z(fwd_plan, d_in, d_freq);
  freq_mem.copy_to_host(h_freq.data(), total_half * batch * sizeof(flagfftDoubleComplex));

  // Inverse Z2D
  freq_mem.copy_from_host(h_freq.data(), total_half * batch * sizeof(flagfftDoubleComplex));
  flagfft_test::ExecZ2D(inv_plan, d_freq, d_out);
  out_mem.copy_to_host(h_roundtrip.data(), total_real * batch * sizeof(flagfftDoubleReal));

  // Z2D doesn't normalize, so expected = input * N
  const int N = total_real;
  std::vector<flagfftDoubleReal> h_expected(total_real * batch);
  for (int i = 0; i < total_real * batch; ++i) {
    h_expected[i] = h_in[i] * N;
  }

  flagfft_test::ErrorStats stats =
      flagfft_test::error_stats(h_roundtrip.data(), h_expected.data(), total_real, batch);
  flagfft_test::expect_roundtrip_accuracy(stats, FLAGFFT_D2Z, FLAGFFT_Z2D, total_real, batch);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         D2ZZ2DRoundtrip2D,
                         ::testing::ValuesIn(Generate2DParams(k2DSmoke, 1, k2DBatchSingle, 1)));

INSTANTIATE_TEST_SUITE_P(Extended,
                         D2ZZ2DRoundtrip2D,
                         ::testing::ValuesIn(Generate2DParams(k2DSizes, k2DNumSizes, k2DBatchSingle, 1)));

INSTANTIATE_TEST_SUITE_P(Batch,
                         D2ZZ2DRoundtrip2D,
                         ::testing::ValuesIn(Generate2DParams(k2DSizes, k2DNumSizes, k2DBatchMulti, 1)));

}  // namespace
