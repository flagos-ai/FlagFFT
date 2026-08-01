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

#include <cstring>
#include <exception>
#include <vector>

namespace {

struct Test3DSize {
  int n0;
  int n1;
  int n2;
};

constexpr Test3DSize k3DSmoke[] = {
    {16, 16, 16},
};

constexpr Test3DSize k3DSizes[] = {
    { 16,  16,  16},
    { 32,  16,   8},
    {  8,  32,  16},
    { 16,  32,  64},
    { 64,  32,  16},
    {128,  64,  32},
};

// Non-smooth sizes to exercise Rader/Bluestein on one or more axes.
constexpr Test3DSize k3DBluesteinSizes[] = {
    { 23,  16,  16},
    { 16, 997,   8},
    { 16,  16, 997},
};

constexpr int k3DNumSizes = sizeof(k3DSizes) / sizeof(k3DSizes[0]);
constexpr int k3DNumBluesteinSizes = sizeof(k3DBluesteinSizes) / sizeof(k3DBluesteinSizes[0]);
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
    for (int b = 0; b < numBatches; ++b) params.push_back({sizes[i].n0, sizes[i].n1, sizes[i].n2, batches[b]});
  return params;
}

std::vector<Test3DParam> All3DParams() {
  auto params = Generate3DParams(k3DSmoke, 1, k3DBatchSingle, 1);
  auto ext = Generate3DParams(k3DSizes, k3DNumSizes, k3DBatchSingle, 1);
  params.insert(params.end(), ext.begin(), ext.end());
  ext = Generate3DParams(k3DBluesteinSizes, k3DNumBluesteinSizes, k3DBatchSingle, 1);
  params.insert(params.end(), ext.begin(), ext.end());
  return params;
}

std::vector<Test3DParam> Filter3DParams(std::vector<Test3DParam> defaults) {
  if (defaults.empty()) return defaults;
  if (flagfft_test::g_test_params.nx > 0 && flagfft_test::g_test_params.ny > 0 &&
      flagfft_test::g_test_params.nz > 0) {
    int batch = flagfft_test::g_test_params.batch > 0 ? flagfft_test::g_test_params.batch : 1;
    return {
        {flagfft_test::g_test_params.nx, flagfft_test::g_test_params.ny, flagfft_test::g_test_params.nz,
         batch}
    };
  }
  return defaults;
}

// =========================================================================
// C2C 3D tests
// =========================================================================

class C2C3D : public ::testing::TestWithParam<Test3DParam> {
 protected:
  void SetUp() override {
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
    if (!HasUsableDevice()) return;
    n0 = GetParam().n0;
    n1 = GetParam().n1;
    n2 = GetParam().n2;
    batch = GetParam().batch;
    total_in = n0 * n1 * (n2 / 2 + 1);
    total_out = n0 * n1 * n2;

    flagfft_test::PlanMany3d(&plan, n0, n1, n2, FLAGFFT_C2R, batch);

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
    if (!HasUsableDevice()) return;
    n0 = GetParam().n0;
    n1 = GetParam().n1;
    n2 = GetParam().n2;
    batch = GetParam().batch;
    total_in = n0 * n1 * (n2 / 2 + 1);
    total_out = n0 * n1 * n2;

    flagfft_test::PlanMany3d(&plan, n0, n1, n2, FLAGFFT_Z2D, batch);

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

}  // namespace
