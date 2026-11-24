// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "flagfft_test.h"

#include <numeric>
#include <utility>

namespace {

template <typename Scalar, typename Complex>
void CheckNativeBatch(const std::vector<int>& shape,
                      flagfftType complex_type,
                      flagfftType forward_type,
                      flagfftType inverse_type) {
  constexpr int batch = 4;
  const int total = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  const int half = total / shape.back() * (shape.back() / 2 + 1);
  for (flagfftType type : {complex_type, forward_type, inverse_type}) {
    const bool real_input = type == forward_type;
    const bool real_output = type == inverse_type;
    const int input_scalars = real_input ? total : 2 * (real_output ? half : total);
    const int output_scalars = real_output ? total : 2 * (real_input ? half : total);
    std::vector<Scalar> input(batch * input_scalars, Scalar{0});
    std::vector<Scalar> output(batch * output_scalars);
    // Distinct DC/impulse amplitudes detect an incorrect batch distance or
    // executing only batch zero. A single execution must fill all batches.
    for (int b = 0; b < batch; ++b) input[b * input_scalars] = Scalar(b + 1);
    flagfft::adaptor::Memory device_input(input.size() * sizeof(Scalar));
    flagfft::adaptor::Memory device_output(output.size() * sizeof(Scalar));
    device_input.copy_from_host(input.data(), input.size() * sizeof(Scalar));
    flagfft::test_adaptor::RefPlanHandle original;
    original.configure_case(real_output ? FLAGFFT_INVERSE : FLAGFFT_FORWARD, batch);
    if (shape.size() == 2) {
      flagfft::test_adaptor::ref_plan_2d(original, shape[0], shape[1], type);
    } else {
      flagfft::test_adaptor::ref_plan_3d(original, shape[0], shape[1], shape[2], type);
    }
    auto plan = std::move(original);
    EXPECT_EQ(plan.batch(), batch);
    flagfft::adaptor::Stream stream;
    flagfft::test_adaptor::ref_set_stream(plan, stream.get());
    auto* input_real = static_cast<Scalar*>(device_input.data());
    auto* output_real = static_cast<Scalar*>(device_output.data());
    auto* input_complex = reinterpret_cast<Complex*>(input_real);
    auto* output_complex = reinterpret_cast<Complex*>(output_real);
    if constexpr (sizeof(Scalar) == sizeof(float)) {
      if (real_input) flagfft::test_adaptor::ref_exec_r2c(plan, input_real, output_complex);
      else if (real_output) flagfft::test_adaptor::ref_exec_c2r(plan, input_complex, output_real);
      else flagfft::test_adaptor::ref_exec_c2c(plan, input_complex, output_complex, FLAGFFT_FORWARD);
    } else {
      if (real_input) flagfft::test_adaptor::ref_exec_d2z(plan, input_real, output_complex);
      else if (real_output) flagfft::test_adaptor::ref_exec_z2d(plan, input_complex, output_real);
      else flagfft::test_adaptor::ref_exec_z2z(plan, input_complex, output_complex, FLAGFFT_FORWARD);
    }
    stream.sync();
    device_output.copy_to_host(output.data(), output.size() * sizeof(Scalar));
    for (int b = 0; b < batch; ++b) {
      for (int i = 0; i < output_scalars; ++i) {
        const Scalar expected = real_output || i % 2 == 0 ? Scalar(b + 1) : Scalar{0};
        ASSERT_NEAR(output[b * output_scalars + i], expected, 1e-6)
            << "rank=" << shape.size() << " type=" << type << " batch=" << b << " index=" << i;
      }
    }
  }
}

TEST(CudaMultidReference, DenseBatchDistancesAndMovedMetadata) {
  if (flagfft::adaptor::device_count() == 0) GTEST_SKIP() << "requires CUDA";
  for (const std::vector<int>& shape : {std::vector<int>{4, 8}, std::vector<int>{4, 8, 16}}) {
    CheckNativeBatch<float, flagfftComplex>(shape, FLAGFFT_C2C, FLAGFFT_R2C, FLAGFFT_C2R);
    CheckNativeBatch<double, flagfftDoubleComplex>(shape, FLAGFFT_Z2Z, FLAGFFT_D2Z, FLAGFFT_Z2D);
  }
}

}  // namespace
