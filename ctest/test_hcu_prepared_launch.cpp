// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "flagfft_test.h"
#include "exec/c_api_internal.hpp"
#include "triton_jit/triton_kernel.h"

#include <array>
#include <complex>
#include <type_traits>

namespace {

struct RuntimeSetting {
  std::string name;
  std::string previous;
  bool existed;
  explicit RuntimeSetting(const char *value, const char *variable = "FLAGFFT_HCU_PREPARED_LAUNCH")
      : name(variable), existed(std::getenv(variable) != nullptr) {
    if (existed) previous = std::getenv(name.c_str());
    if (value) setenv(name.c_str(), value, 1);
    else unsetenv(name.c_str());
  }
  ~RuntimeSetting() {
    if (existed) setenv(name.c_str(), previous.c_str(), 1);
    else unsetenv(name.c_str());
  }
};

template <typename Scalar, typename Complex>
void CheckRealDispatch(int n, bool alias) {
  SCOPED_TRACE(::testing::Message() << "n=" << n << " alias=" << alias);
  constexpr bool fp64 = std::is_same_v<Scalar, double>;
  const auto forward_type = fp64 ? FLAGFFT_D2Z : FLAGFFT_R2C;
  const auto inverse_type = fp64 ? FLAGFFT_Z2D : FLAGFFT_C2R;
  const int half = n / 2 + 1;
  const int real_distance = alias ? 2 * half : n;
  std::array<flagfft::adaptor::Memory, 2> real, spectrum, restored;
  for (int slot = 0; slot < 2; ++slot) {
    real[slot].allocate(real_distance * sizeof(Scalar));
    spectrum[slot].allocate(half * sizeof(Complex));
    restored[slot].allocate(n * sizeof(Scalar));
  }
  flagfftHandle forward = nullptr, inverse = nullptr;
  ASSERT_EQ(flagfftPlanMany(&forward, 1, &n, nullptr, 1, real_distance,
                            nullptr, 1, half, forward_type, 1), FLAGFFT_SUCCESS);
  ASSERT_EQ(flagfftPlanMany(&inverse, 1, &n, nullptr, 1, half,
                            nullptr, 1, real_distance, inverse_type, 1), FLAGFFT_SUCCESS);
  auto fwd = std::dynamic_pointer_cast<flagfft::CompiledRawR2CLeafNode>(
      flagfft::checked_plan(forward)->executable.forward);
  auto inv = std::dynamic_pointer_cast<flagfft::CompiledRawC2RLeafNode>(
      flagfft::checked_plan(inverse)->executable.inverse);
  ASSERT_NE(fwd, nullptr);
  ASSERT_NE(inv, nullptr);
  flagfft::adaptor::Stream stream_a, stream_b;
  const std::array<int, 6> slots {0, 0, 1, 0, 1, 0};
  const std::array<flagfftStream_t, 6> streams {
      nullptr, stream_a.get(), stream_b.get(), stream_b.get(), stream_a.get(), stream_a.get()};
  for (int step = 0; step < 6; ++step) {
    SCOPED_TRACE(step);
    const int slot = slots[step];
    std::vector<Scalar> input(n), output(n), expected_inverse(n);
    for (int j = 0; j < n; ++j) {
      input[j] = static_cast<Scalar>(std::sin((j + 1) * (step + 1) * 0.173) + step * 0.1);
      expected_inverse[j] = static_cast<Scalar>(n * input[j]);
    }
    std::vector<Complex> expected(half), actual(half);
    const long double pi = std::acos(-1.0L);
    for (int k = 0; k < half; ++k) {
      std::complex<long double> sum {};
      for (int j = 0; j < n; ++j) {
        const long double angle = -2 * pi * j * k / n;
        sum += static_cast<long double>(input[j]) *
               std::complex<long double>(std::cos(angle), std::sin(angle));
      }
      expected[k] = {static_cast<Scalar>(sum.real()), static_cast<Scalar>(sum.imag())};
    }
    real[slot].copy_from_host(input.data(), n * sizeof(Scalar));
    auto *input_ptr = static_cast<Scalar *>(real[slot].data());
    auto *spectrum_ptr = alias ? reinterpret_cast<Complex *>(input_ptr)
                               : static_cast<Complex *>(spectrum[slot].data());
    auto *restored_ptr = alias ? input_ptr : static_cast<Scalar *>(restored[slot].data());
    ASSERT_EQ(flagfftSetStream(forward, streams[step]), FLAGFFT_SUCCESS);
    ASSERT_EQ(flagfftSetStream(inverse, streams[step]), FLAGFFT_SUCCESS);
    if constexpr (fp64) {
      ASSERT_EQ(flagfftExecD2Z(forward, input_ptr, spectrum_ptr), FLAGFFT_SUCCESS);
    } else {
      ASSERT_EQ(flagfftExecR2C(forward, input_ptr, spectrum_ptr), FLAGFFT_SUCCESS);
    }
    flagfft::adaptor::synchronize();
    (alias ? real[slot] : spectrum[slot]).copy_to_host(actual.data(), half * sizeof(Complex));
    auto forward_error = flagfft_test::error_stats(actual.data(), expected.data(), half, 1);
    EXPECT_TRUE(forward_error.finite);
    EXPECT_LT(forward_error.rel_l2, fp64 ? 5e-13 : 5e-6);
    if constexpr (fp64) {
      ASSERT_EQ(flagfftExecZ2D(inverse, spectrum_ptr, restored_ptr), FLAGFFT_SUCCESS);
    } else {
      ASSERT_EQ(flagfftExecC2R(inverse, spectrum_ptr, restored_ptr), FLAGFFT_SUCCESS);
    }
    flagfft::adaptor::synchronize();
    (alias ? real[slot] : restored[slot]).copy_to_host(output.data(), n * sizeof(Scalar));
    auto inverse_error = flagfft_test::error_stats(output.data(), expected_inverse.data(), n, 1);
    EXPECT_TRUE(inverse_error.finite);
    EXPECT_LT(inverse_error.rel_l2, fp64 ? 5e-13 : 5e-6);
    EXPECT_NE(fwd->kernel->direct_launch, nullptr);
    EXPECT_NE(inv->kernel->direct_launch, nullptr);
  }
  EXPECT_EQ(flagfftDestroy(forward), FLAGFFT_SUCCESS);
  EXPECT_EQ(flagfftDestroy(inverse), FLAGFFT_SUCCESS);
}

TEST(HcuPreparedLaunch, DefaultHandlesChangedDataAddressesStreamsAndInPlace) {
  RuntimeSetting setting(nullptr);
  for (int n : {48, 105}) {
    for (bool alias : {false, true}) {
      CheckRealDispatch<float, flagfftComplex>(n, alias);
      CheckRealDispatch<double, flagfftDoubleComplex>(n, alias);
    }
  }
}

TEST(HcuPreparedLaunch, OverrideRetainsGenericDispatch) {
  RuntimeSetting setting("0");
  CheckRealDispatch<float, flagfftComplex>(48, false);
}

TEST(HcuPreparedLaunch, PreservesLaunchHooks) {
  RuntimeSetting prepared("1");
  struct ClearHooks {
    ~ClearHooks() { triton_jit::clear_launch_hooks(); }
  } cleanup;
  int entered = 0, exited = 0;
  triton_jit::set_launch_enter_hook([&](const triton_jit::LaunchMetadata &metadata) {
    ++entered;
    EXPECT_FALSE(metadata.kernel_name.empty());
    EXPECT_FALSE(metadata.signature.empty());
    EXPECT_GT(metadata.num_warps, 0);
    EXPECT_EQ(metadata.grid_x, 1);
  });
  triton_jit::set_launch_exit_hook([&](const triton_jit::LaunchMetadata &) { ++exited; });
  CheckRealDispatch<float, flagfftComplex>(48, false);
  EXPECT_EQ(entered, 12);
  EXPECT_EQ(exited, entered);
}

}  // namespace
