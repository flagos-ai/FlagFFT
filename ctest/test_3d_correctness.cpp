// Copyright 2026 FlagOS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "flagfft_test.h"
#include "exec/c_api_internal.hpp"

#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <vector>
#include <thread>

namespace {

struct Test3DSize {
  int n0;
  int n1;
  int n2;
};

constexpr Test3DSize k3DSizes[] = {
    {    1,    16,    32},
    {   16,     1,    32},
    {   16,    32,     1},
    {   23,    30,    67},
    {   30,    67,    23},
    {   67,    23,    30},
    {  997,    16,    64},
    {   16,   997,    64},
    {   16,    64,   997},
    {    4,    32, 16384},
    {   32, 16384,     4},
    {16384,     4,    32},
    {   31,    33,    35},
    {   64,    64,    64},
};
constexpr int k3DNumSizes = sizeof(k3DSizes) / sizeof(k3DSizes[0]);
constexpr int k3DBatchSingle[] = {1};

bool HasUsableDevice() {
  try {
    return flagfft::adaptor::device_count() > 0;
  } catch (const std::exception&) {
    return false;
  }
}

struct Test3DParam {
  int n0;
  int n1;
  int n2;
  int batch;
};

std::vector<Test3DParam> Generate3DParams(const Test3DSize* sizes,
                                          int numSizes,
                                          const int* batches,
                                          int numBatches) {
  std::vector<Test3DParam> params;
  for (int i = 0; i < numSizes; ++i)
    for (int b = 0; b < numBatches; ++b)
      params.push_back({sizes[i].n0, sizes[i].n1, sizes[i].n2, batches[b]});
  return params;
}

std::vector<Test3DParam> All3DParams() {
  return Generate3DParams(k3DSizes, k3DNumSizes, k3DBatchSingle, 1);
}

std::vector<Test3DParam> Filter3DParams(std::vector<Test3DParam> defaults) {
  if (defaults.empty()) return defaults;
  if (flagfft_test::g_test_params.nx > 0 && flagfft_test::g_test_params.ny > 0 &&
      flagfft_test::g_test_params.nz > 0) {
    int batch = flagfft_test::g_test_params.batch > 0 ? flagfft_test::g_test_params.batch : 1;
    return {
        {flagfft_test::g_test_params.nx,
         flagfft_test::g_test_params.ny,
         flagfft_test::g_test_params.nz,
         batch}
    };
  }
  return defaults;
}
std::vector<flagfftComplex> ValidC2RInput3D(int n0, int n1, int n2, int batch, std::uint64_t seed) {
  const int total_real = n0 * n1 * n2;
  const int total_half = n0 * n1 * (n2 / 2 + 1);
  auto real_input = flagfft_test::random_real(total_real * batch, seed);
  std::vector<flagfftComplex> spectrum(total_half * batch);
  flagfft_test::RefPlanHandle plan;
  flagfft_test::ref_plan_3d(plan, n0, n1, n2, FLAGFFT_R2C);
  flagfft::adaptor::Memory real_mem(total_real * sizeof(flagfftReal));
  flagfft::adaptor::Memory spectrum_mem(total_half * sizeof(flagfftComplex));
  for (int b = 0; b < batch; ++b) {
    real_mem.copy_from_host(real_input.data() + b * total_real, total_real * sizeof(flagfftReal));
    flagfft_test::ref_exec_r2c(plan,
                               static_cast<flagfftReal*>(real_mem.data()),
                               static_cast<flagfftComplex*>(spectrum_mem.data()));
    spectrum_mem.copy_to_host(spectrum.data() + b * total_half, total_half * sizeof(flagfftComplex));
  }
  return spectrum;
}

std::vector<flagfftDoubleComplex> ValidZ2DInput3D(int n0, int n1, int n2, int batch, std::uint64_t seed) {
  const int total_real = n0 * n1 * n2;
  const int total_half = n0 * n1 * (n2 / 2 + 1);
  auto real_input = flagfft_test::random_double_real(total_real * batch, seed);
  std::vector<flagfftDoubleComplex> spectrum(total_half * batch);
  flagfft_test::RefPlanHandle plan;
  flagfft_test::ref_plan_3d(plan, n0, n1, n2, FLAGFFT_D2Z);
  flagfft::adaptor::Memory real_mem(total_real * sizeof(flagfftDoubleReal));
  flagfft::adaptor::Memory spectrum_mem(total_half * sizeof(flagfftDoubleComplex));
  for (int b = 0; b < batch; ++b) {
    real_mem.copy_from_host(real_input.data() + b * total_real, total_real * sizeof(flagfftDoubleReal));
    flagfft_test::ref_exec_d2z(plan,
                               static_cast<flagfftDoubleReal*>(real_mem.data()),
                               static_cast<flagfftDoubleComplex*>(spectrum_mem.data()));
    spectrum_mem.copy_to_host(spectrum.data() + b * total_half, total_half * sizeof(flagfftDoubleComplex));
  }
  return spectrum;
}

// =========================================================================
// C2C 3D tests
// =========================================================================

class C2C3D : public ::testing::TestWithParam<Test3DParam> {
 protected:
  void SetUp() override {
    if (flagfft_test::should_skip_api("c2c")) GTEST_SKIP() << "filtered by --api";
    if (!HasUsableDevice()) return;
    n0 = GetParam().n0;
    n1 = GetParam().n1;
    n2 = GetParam().n2;
    batch = GetParam().batch;
    total = n0 * n1 * n2;
    bytes = total * batch * sizeof(flagfftComplex);

    flagfft_test::PlanMany3d(&plan, n0, n1, n2, FLAGFFT_C2C, batch);

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

  int n0 = 0, n1 = 0, n2 = 0, batch = 0, total = 0;
  std::size_t bytes = 0;
  flagfftHandle plan = nullptr;
  std::vector<flagfftComplex> h_in, h_out, h_roundtrip;
  flagfft::adaptor::Memory in_mem, out_mem;
  flagfftComplex* d_in = nullptr;
  flagfftComplex* d_out = nullptr;
};

TEST_P(C2C3D, ForwardInverse) {
  if (!HasUsableDevice()) GTEST_SKIP() << "no device";
  if (flagfft_test::should_skip_duplicate_roundtrip()) GTEST_SKIP() << "roundtrip runs with forward case";

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

TEST_P(C2C3D, ForwardReference) {
  if (!HasUsableDevice()) GTEST_SKIP() << "no device";
  if (flagfft_test::should_skip_direction(FLAGFFT_FORWARD)) GTEST_SKIP();

  flagfft_test::RefPlanHandle ref_plan;
  flagfft_test::ref_plan_3d(ref_plan, n0, n1, n2, FLAGFFT_C2C);

  flagfft::adaptor::Memory ref_in_mem(total * sizeof(flagfftComplex));
  flagfft::adaptor::Memory ref_out_mem(total * sizeof(flagfftComplex));
  auto* d_ref_in = static_cast<flagfftComplex*>(ref_in_mem.data());
  auto* d_ref_out = static_cast<flagfftComplex*>(ref_out_mem.data());

  for (double scale : flagfft_test::filter_scales()) {
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

TEST_P(C2C3D, InverseReference) {
  if (!HasUsableDevice()) GTEST_SKIP() << "no device";
  if (flagfft_test::should_skip_direction(FLAGFFT_INVERSE)) GTEST_SKIP();

  flagfft_test::RefPlanHandle ref_plan;
  flagfft_test::ref_plan_3d(ref_plan, n0, n1, n2, FLAGFFT_C2C);

  flagfft::adaptor::Memory ref_in_mem(total * sizeof(flagfftComplex));
  flagfft::adaptor::Memory ref_out_mem(total * sizeof(flagfftComplex));
  auto* d_ref_in = static_cast<flagfftComplex*>(ref_in_mem.data());
  auto* d_ref_out = static_cast<flagfftComplex*>(ref_out_mem.data());

  for (double scale : flagfft_test::filter_scales()) {
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

INSTANTIATE_TEST_SUITE_P(All, C2C3D, ::testing::ValuesIn(Filter3DParams(All3DParams())));

// =========================================================================
// Z2Z 3D tests
// =========================================================================

class Z2Z3D : public ::testing::TestWithParam<Test3DParam> {
 protected:
  void SetUp() override {
    if (flagfft_test::should_skip_api("z2z")) GTEST_SKIP() << "filtered by --api";
    if (!HasUsableDevice()) return;
    n0 = GetParam().n0;
    n1 = GetParam().n1;
    n2 = GetParam().n2;
    batch = GetParam().batch;
    total = n0 * n1 * n2;
    bytes = total * batch * sizeof(flagfftDoubleComplex);

    flagfft_test::PlanMany3d(&plan, n0, n1, n2, FLAGFFT_Z2Z, batch);

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

  int n0 = 0, n1 = 0, n2 = 0, batch = 0, total = 0;
  std::size_t bytes = 0;
  flagfftHandle plan = nullptr;
  std::vector<flagfftDoubleComplex> h_in, h_out, h_roundtrip;
  flagfft::adaptor::Memory in_mem, out_mem;
  flagfftDoubleComplex* d_in = nullptr;
  flagfftDoubleComplex* d_out = nullptr;
};

TEST_P(Z2Z3D, ForwardInverse) {
  if (!HasUsableDevice()) GTEST_SKIP() << "no device";
  if (flagfft_test::should_skip_duplicate_roundtrip()) GTEST_SKIP() << "roundtrip runs with forward case";

  // Forward
  flagfft_test::ExecZ2Z(plan, d_in, d_out, FLAGFFT_FORWARD);
  out_mem.copy_to_host(h_out.data(), bytes);

  // Copy output to input for inverse
  in_mem.copy_from_host(h_out.data(), bytes);

  // Inverse
  flagfft_test::ExecZ2Z(plan, d_in, d_out, FLAGFFT_INVERSE);
  out_mem.copy_to_host(h_roundtrip.data(), bytes);

  // Inverse FFT doesn't normalize, so expected = input * N
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

TEST_P(Z2Z3D, ForwardReference) {
  if (!HasUsableDevice()) GTEST_SKIP() << "no device";
  if (flagfft_test::should_skip_direction(FLAGFFT_FORWARD)) GTEST_SKIP();

  flagfft_test::RefPlanHandle ref_plan;
  flagfft_test::ref_plan_3d(ref_plan, n0, n1, n2, FLAGFFT_Z2Z);

  flagfft::adaptor::Memory ref_in_mem(total * sizeof(flagfftDoubleComplex));
  flagfft::adaptor::Memory ref_out_mem(total * sizeof(flagfftDoubleComplex));
  auto* d_ref_in = static_cast<flagfftDoubleComplex*>(ref_in_mem.data());
  auto* d_ref_out = static_cast<flagfftDoubleComplex*>(ref_out_mem.data());

  for (double scale : flagfft_test::filter_scales()) {
    auto input = h_in;
    flagfft_test::scale_input(input, scale);
    in_mem.copy_from_host(input.data(), bytes);

    // FlagFFT forward
    flagfft_test::ExecZ2Z(plan, d_in, d_out, FLAGFFT_FORWARD);
    out_mem.copy_to_host(h_out.data(), bytes);

    // Reference forward - process each batch separately
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

TEST_P(Z2Z3D, InverseReference) {
  if (!HasUsableDevice()) GTEST_SKIP() << "no device";
  if (flagfft_test::should_skip_direction(FLAGFFT_INVERSE)) GTEST_SKIP();

  flagfft_test::RefPlanHandle ref_plan;
  flagfft_test::ref_plan_3d(ref_plan, n0, n1, n2, FLAGFFT_Z2Z);

  flagfft::adaptor::Memory ref_in_mem(total * sizeof(flagfftDoubleComplex));
  flagfft::adaptor::Memory ref_out_mem(total * sizeof(flagfftDoubleComplex));
  auto* d_ref_in = static_cast<flagfftDoubleComplex*>(ref_in_mem.data());
  auto* d_ref_out = static_cast<flagfftDoubleComplex*>(ref_out_mem.data());

  for (double scale : flagfft_test::filter_scales()) {
    auto input = h_in;
    flagfft_test::scale_input(input, scale);
    in_mem.copy_from_host(input.data(), bytes);

    // FlagFFT inverse
    flagfft_test::ExecZ2Z(plan, d_in, d_out, FLAGFFT_INVERSE);
    out_mem.copy_to_host(h_out.data(), bytes);

    // Reference inverse - process each batch separately
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

INSTANTIATE_TEST_SUITE_P(All, Z2Z3D, ::testing::ValuesIn(Filter3DParams(All3DParams())));

// =========================================================================
// R2C 3D tests
// =========================================================================

class R2C3D : public ::testing::TestWithParam<Test3DParam> {
 protected:
  void SetUp() override {
    if (flagfft_test::should_skip_api("r2c")) GTEST_SKIP() << "filtered by --api";
    if (!HasUsableDevice()) return;
    n0 = GetParam().n0;
    n1 = GetParam().n1;
    n2 = GetParam().n2;
    batch = GetParam().batch;
    total_in = n0 * n1 * n2;
    total_out = n0 * n1 * (n2 / 2 + 1);

    flagfft_test::PlanMany3d(&plan, n0, n1, n2, FLAGFFT_R2C, batch);

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

  int n0 = 0, n1 = 0, n2 = 0, batch = 0;
  int total_in = 0, total_out = 0;
  flagfftHandle plan = nullptr;
  std::vector<flagfftReal> h_in;
  std::vector<flagfftComplex> h_out;
  flagfft::adaptor::Memory in_mem, out_mem;
  flagfftReal* d_in = nullptr;
  flagfftComplex* d_out = nullptr;
};

TEST_P(R2C3D, ForwardReference) {
  if (!HasUsableDevice()) GTEST_SKIP() << "no device";

  flagfft_test::RefPlanHandle ref_plan;
  flagfft_test::ref_plan_3d(ref_plan, n0, n1, n2, FLAGFFT_R2C);

  flagfft::adaptor::Memory ref_in_mem(total_in * sizeof(flagfftReal));
  flagfft::adaptor::Memory ref_out_mem(total_out * sizeof(flagfftComplex));
  auto* d_ref_in = static_cast<flagfftReal*>(ref_in_mem.data());
  auto* d_ref_out = static_cast<flagfftComplex*>(ref_out_mem.data());

  for (double scale : flagfft_test::filter_scales()) {
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

INSTANTIATE_TEST_SUITE_P(All, R2C3D, ::testing::ValuesIn(Filter3DParams(All3DParams())));

// =========================================================================
// D2Z 3D tests
// =========================================================================

class D2Z3D : public ::testing::TestWithParam<Test3DParam> {
 protected:
  void SetUp() override {
    if (flagfft_test::should_skip_api("d2z")) GTEST_SKIP() << "filtered by --api";
    if (!HasUsableDevice()) return;
    n0 = GetParam().n0;
    n1 = GetParam().n1;
    n2 = GetParam().n2;
    batch = GetParam().batch;
    total_in = n0 * n1 * n2;
    total_out = n0 * n1 * (n2 / 2 + 1);

    flagfft_test::PlanMany3d(&plan, n0, n1, n2, FLAGFFT_D2Z, batch);

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

  int n0 = 0, n1 = 0, n2 = 0, batch = 0;
  int total_in = 0, total_out = 0;
  flagfftHandle plan = nullptr;
  std::vector<flagfftDoubleReal> h_in;
  std::vector<flagfftDoubleComplex> h_out;
  flagfft::adaptor::Memory in_mem, out_mem;
  flagfftDoubleReal* d_in = nullptr;
  flagfftDoubleComplex* d_out = nullptr;
};

TEST_P(D2Z3D, ForwardReference) {
  if (!HasUsableDevice()) GTEST_SKIP() << "no device";

  flagfft_test::RefPlanHandle ref_plan;
  flagfft_test::ref_plan_3d(ref_plan, n0, n1, n2, FLAGFFT_D2Z);

  flagfft::adaptor::Memory ref_in_mem(total_in * sizeof(flagfftDoubleReal));
  flagfft::adaptor::Memory ref_out_mem(total_out * sizeof(flagfftDoubleComplex));
  auto* d_ref_in = static_cast<flagfftDoubleReal*>(ref_in_mem.data());
  auto* d_ref_out = static_cast<flagfftDoubleComplex*>(ref_out_mem.data());

  for (double scale : flagfft_test::filter_scales()) {
    auto input = h_in;
    flagfft_test::scale_input(input, scale);
    in_mem.copy_from_host(input.data(), total_in * batch * sizeof(flagfftDoubleReal));

    // FlagFFT forward
    flagfft_test::ExecD2Z(plan, d_in, d_out);
    out_mem.copy_to_host(h_out.data(), total_out * batch * sizeof(flagfftDoubleComplex));

    // Reference forward - process each batch separately
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

INSTANTIATE_TEST_SUITE_P(All, D2Z3D, ::testing::ValuesIn(Filter3DParams(All3DParams())));

// =========================================================================
// C2R 3D tests
// =========================================================================

class C2R3D : public ::testing::TestWithParam<Test3DParam> {
 protected:
  void SetUp() override {
    if (flagfft_test::should_skip_api("c2r")) GTEST_SKIP() << "filtered by --api";
    if (!HasUsableDevice()) return;
    n0 = GetParam().n0;
    n1 = GetParam().n1;
    n2 = GetParam().n2;
    batch = GetParam().batch;
    total_in = n0 * n1 * (n2 / 2 + 1);
    total_out = n0 * n1 * n2;

    flagfft_test::PlanMany3d(&plan, n0, n1, n2, FLAGFFT_C2R, batch);

    std::uint64_t seed = flagfft_test::accuracy_seed(FLAGFFT_C2R, total_out, batch);
    h_in = ValidC2RInput3D(n0, n1, n2, batch, seed);
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

  int n0 = 0, n1 = 0, n2 = 0, batch = 0;
  int total_in = 0, total_out = 0;
  flagfftHandle plan = nullptr;
  std::vector<flagfftComplex> h_in;
  std::vector<flagfftReal> h_out;
  flagfft::adaptor::Memory in_mem, out_mem;
  flagfftComplex* d_in = nullptr;
  flagfftReal* d_out = nullptr;
};

TEST_P(C2R3D, InverseReference) {
  if (!HasUsableDevice()) GTEST_SKIP() << "no device";

  flagfft_test::RefPlanHandle ref_plan;
  flagfft_test::ref_plan_3d(ref_plan, n0, n1, n2, FLAGFFT_C2R);

  flagfft::adaptor::Memory ref_in_mem(total_in * sizeof(flagfftComplex));
  flagfft::adaptor::Memory ref_out_mem(total_out * sizeof(flagfftReal));
  auto* d_ref_in = static_cast<flagfftComplex*>(ref_in_mem.data());
  auto* d_ref_out = static_cast<flagfftReal*>(ref_out_mem.data());

  for (double scale : flagfft_test::filter_scales()) {
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

INSTANTIATE_TEST_SUITE_P(All, C2R3D, ::testing::ValuesIn(Filter3DParams(All3DParams())));

// =========================================================================
// Z2D 3D tests
// =========================================================================

class Z2D3D : public ::testing::TestWithParam<Test3DParam> {
 protected:
  void SetUp() override {
    if (flagfft_test::should_skip_api("z2d")) GTEST_SKIP() << "filtered by --api";
    if (!HasUsableDevice()) return;
    n0 = GetParam().n0;
    n1 = GetParam().n1;
    n2 = GetParam().n2;
    batch = GetParam().batch;
    total_in = n0 * n1 * (n2 / 2 + 1);
    total_out = n0 * n1 * n2;

    flagfft_test::PlanMany3d(&plan, n0, n1, n2, FLAGFFT_Z2D, batch);

    std::uint64_t seed = flagfft_test::accuracy_seed(FLAGFFT_Z2D, total_out, batch);
    h_in = ValidZ2DInput3D(n0, n1, n2, batch, seed);
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

  int n0 = 0, n1 = 0, n2 = 0, batch = 0;
  int total_in = 0, total_out = 0;
  flagfftHandle plan = nullptr;
  std::vector<flagfftDoubleComplex> h_in;
  std::vector<flagfftDoubleReal> h_out;
  flagfft::adaptor::Memory in_mem, out_mem;
  flagfftDoubleComplex* d_in = nullptr;
  flagfftDoubleReal* d_out = nullptr;
};

TEST_P(Z2D3D, InverseReference) {
  if (!HasUsableDevice()) GTEST_SKIP() << "no device";

  flagfft_test::RefPlanHandle ref_plan;
  flagfft_test::ref_plan_3d(ref_plan, n0, n1, n2, FLAGFFT_Z2D);

  flagfft::adaptor::Memory ref_in_mem(total_in * sizeof(flagfftDoubleComplex));
  flagfft::adaptor::Memory ref_out_mem(total_out * sizeof(flagfftDoubleReal));
  auto* d_ref_in = static_cast<flagfftDoubleComplex*>(ref_in_mem.data());
  auto* d_ref_out = static_cast<flagfftDoubleReal*>(ref_out_mem.data());

  for (double scale : flagfft_test::filter_scales()) {
    auto input = h_in;
    flagfft_test::scale_input(input, scale);
    in_mem.copy_from_host(input.data(), total_in * batch * sizeof(flagfftDoubleComplex));

    // FlagFFT inverse
    flagfft_test::ExecZ2D(plan, d_in, d_out);
    out_mem.copy_to_host(h_out.data(), total_out * batch * sizeof(flagfftDoubleReal));

    // Reference inverse - process each batch separately
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

INSTANTIATE_TEST_SUITE_P(All, Z2D3D, ::testing::ValuesIn(Filter3DParams(All3DParams())));

// =========================================================================
// R2C + C2R roundtrip 3D tests
// =========================================================================

class R2CC2RRoundtrip3D : public ::testing::TestWithParam<Test3DParam> {
 protected:
  void SetUp() override {
    if (flagfft_test::should_skip_api("r2c")) GTEST_SKIP() << "filtered by --api";
    if (!HasUsableDevice()) return;
    n0 = GetParam().n0;
    n1 = GetParam().n1;
    n2 = GetParam().n2;
    batch = GetParam().batch;
    total_real = n0 * n1 * n2;
    total_half = n0 * n1 * (n2 / 2 + 1);

    flagfft_test::PlanMany3d(&fwd_plan, n0, n1, n2, FLAGFFT_R2C, batch);
    flagfft_test::PlanMany3d(&inv_plan, n0, n1, n2, FLAGFFT_C2R, batch);

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

  int n0 = 0, n1 = 0, n2 = 0, batch = 0;
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

TEST_P(R2CC2RRoundtrip3D, ForwardInverse) {
  if (!HasUsableDevice()) GTEST_SKIP() << "no device";

  flagfft_test::ExecR2C(fwd_plan, d_in, d_freq);
  freq_mem.copy_to_host(h_freq.data(), total_half * batch * sizeof(flagfftComplex));

  freq_mem.copy_from_host(h_freq.data(), total_half * batch * sizeof(flagfftComplex));
  flagfft_test::ExecC2R(inv_plan, d_freq, d_out);
  out_mem.copy_to_host(h_roundtrip.data(), total_real * batch * sizeof(flagfftReal));

  const int transform_size = total_real;
  std::vector<flagfftReal> h_expected(total_real * batch);
  for (int i = 0; i < total_real * batch; ++i) {
    h_expected[i] = h_in[i] * transform_size;
  }

  flagfft_test::ErrorStats stats =
      flagfft_test::error_stats(h_roundtrip.data(), h_expected.data(), total_real, batch);
  flagfft_test::expect_roundtrip_accuracy(stats, FLAGFFT_R2C, FLAGFFT_C2R, total_real, batch);
}

INSTANTIATE_TEST_SUITE_P(All, R2CC2RRoundtrip3D, ::testing::ValuesIn(Filter3DParams(All3DParams())));

// =========================================================================
// D2Z + Z2D roundtrip 3D tests
// =========================================================================

class D2ZZ2DRoundtrip3D : public ::testing::TestWithParam<Test3DParam> {
 protected:
  void SetUp() override {
    if (flagfft_test::should_skip_api("d2z")) GTEST_SKIP() << "filtered by --api";
    if (!HasUsableDevice()) return;
    n0 = GetParam().n0;
    n1 = GetParam().n1;
    n2 = GetParam().n2;
    batch = GetParam().batch;
    total_real = n0 * n1 * n2;
    total_half = n0 * n1 * (n2 / 2 + 1);

    flagfft_test::PlanMany3d(&fwd_plan, n0, n1, n2, FLAGFFT_D2Z, batch);
    flagfft_test::PlanMany3d(&inv_plan, n0, n1, n2, FLAGFFT_Z2D, batch);

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

  int n0 = 0, n1 = 0, n2 = 0, batch = 0;
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

TEST_P(D2ZZ2DRoundtrip3D, ForwardInverse) {
  if (!HasUsableDevice()) GTEST_SKIP() << "no device";

  flagfft_test::ExecD2Z(fwd_plan, d_in, d_freq);
  freq_mem.copy_to_host(h_freq.data(), total_half * batch * sizeof(flagfftDoubleComplex));

  freq_mem.copy_from_host(h_freq.data(), total_half * batch * sizeof(flagfftDoubleComplex));
  flagfft_test::ExecZ2D(inv_plan, d_freq, d_out);
  out_mem.copy_to_host(h_roundtrip.data(), total_real * batch * sizeof(flagfftDoubleReal));

  const int transform_size = total_real;
  std::vector<flagfftDoubleReal> h_expected(total_real * batch);
  for (int i = 0; i < total_real * batch; ++i) {
    h_expected[i] = h_in[i] * transform_size;
  }

  flagfft_test::ErrorStats stats =
      flagfft_test::error_stats(h_roundtrip.data(), h_expected.data(), total_real, batch);
  flagfft_test::expect_roundtrip_accuracy(stats, FLAGFFT_D2Z, FLAGFFT_Z2D, total_real, batch);
}

INSTANTIATE_TEST_SUITE_P(All, D2ZZ2DRoundtrip3D, ::testing::ValuesIn(Filter3DParams(All3DParams())));

void CheckCudaExecutionGraphReplay(bool rank_one, bool rank_two = false) {
  if (flagfft::adaptor::backend_name() != "cuda" || !HasUsableDevice()) {
    GTEST_SKIP() << "requires CUDA";
  }
  struct GraphSetting {
    const char *name;
    bool existed;
    std::string previous;
    explicit GraphSetting(const char *variable)
        : name(variable), existed(std::getenv(name) != nullptr),
          previous(existed ? std::getenv(name) : "") { setenv(name, "1", 1); }
    ~GraphSetting() {
      if (existed) setenv(name, previous.c_str(), 1);
      else unsetenv(name);
    }
  } setting(rank_one ? "FLAGFFT_CUDA_1D_GRAPH" : "FLAGFFT_CUDA_MULTID_GRAPH");
  constexpr int n = 16;
  const int total = rank_two ? n * n : n * n * n;
  constexpr int batch = 2;
  const std::size_t bytes = total * batch * sizeof(flagfftComplex);
  for (bool inplace : {false, true}) {
    struct PlanGuard {
      flagfftHandle handle = nullptr;
      ~PlanGuard() { if (handle) flagfftDestroy(handle); }
    } plan;
    if (rank_one) {
      ASSERT_EQ(flagfftPlan1d(&plan.handle, total, FLAGFFT_C2C, batch), FLAGFFT_SUCCESS);
    } else if (rank_two) {
      ASSERT_NO_FATAL_FAILURE(flagfft_test::PlanMany2d(&plan.handle,n,n,FLAGFFT_C2C,batch));
    } else {
      ASSERT_NO_FATAL_FAILURE(
          flagfft_test::PlanMany3d(&plan.handle, n, n, n, FLAGFFT_C2C, batch));
    }
    ASSERT_NE(plan.handle, nullptr);
    flagfft::adaptor::Stream stream;
    ASSERT_EQ(flagfftSetStream(plan.handle, stream.get()), FLAGFFT_SUCCESS);
    // The route is qualified for sm_80. Other CUDA architectures keep their
    // existing dispatch and do not participate in this replay regression.
    auto *implementation = static_cast<flagfft::FlagFFTPlan *>(plan.handle->impl);
    auto graph = std::dynamic_pointer_cast<flagfft::CompiledRawGraphNode>(
        implementation->executable.forward);
    if (!graph) GTEST_SKIP() << "graph route not selected on this device";
    flagfft::adaptor::Memory inputs[2] = {
        flagfft::adaptor::Memory(bytes), flagfft::adaptor::Memory(bytes)};
    flagfft::adaptor::Memory outputs[2] = {
        flagfft::adaptor::Memory(bytes), flagfft::adaptor::Memory(bytes)};
    std::vector<flagfftComplex> input(total * batch), output(total * batch);
    for (int iteration = 0; iteration < 4; ++iteration) {
      const int slot = iteration % 2;
      std::fill(input.begin(), input.end(), flagfftComplex{0.0f, 0.0f});
      for (int b = 0; b < batch; ++b) {
        input[b * total].x = static_cast<float>(1 + iteration + b);
      }
      inputs[slot].copy_from_host(input.data(), bytes);
      auto &destination = inplace ? inputs[slot] : outputs[slot];
      flagfftResult status=FLAGFFT_INTERNAL_ERROR;
      const auto execute=[&] {
        status=flagfftExecC2C(plan.handle,
            static_cast<flagfftComplex *>(inputs[slot].data()),
            static_cast<flagfftComplex *>(destination.data()),FLAGFFT_FORWARD);
      };
      // Replay the captured slot on a fresh host thread, whose CUDA current
      // context must be initialized without relying on the creator thread.
      if (iteration==2) { std::thread worker(execute);worker.join(); }
      else execute();
      ASSERT_EQ(status,FLAGFFT_SUCCESS);
      stream.sync();
      EXPECT_NE(graph->graph, nullptr);
      EXPECT_FALSE(graph->graph_failed);
      destination.copy_to_host(output.data(), bytes);
      for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < total; ++i) {
          ASSERT_NEAR(output[b * total + i].x, 1 + iteration + b, 1e-5f)
              << "inplace=" << inplace << " iteration=" << iteration << " index=" << i;
          ASSERT_NEAR(output[b * total + i].y, 0.0f, 1e-5f);
        }
      }
    }
  }
}

TEST(CudaMultidGraph, FirstInPlaceExecutionReplayAndPointerChanges) {
  CheckCudaExecutionGraphReplay(false);
}

TEST(Cuda1DGraph, FirstInPlaceExecutionReplayAndPointerChanges) {
  CheckCudaExecutionGraphReplay(true);
}

TEST(Cuda2DGraph, FirstInPlaceExecutionReplayAndPointerChanges) {
  CheckCudaExecutionGraphReplay(false,true);
}

}  // namespace
