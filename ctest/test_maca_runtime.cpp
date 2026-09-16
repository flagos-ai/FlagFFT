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
#include "flagfft/core.hpp"

#include <complex>
#include <numbers>
#include <type_traits>

namespace {

template <typename Complex>
void check_prime_algorithms() {
  using Scalar = decltype(Complex{}.x);
  constexpr int n = 257;
  constexpr int batch = 7;
  constexpr bool is_double = std::is_same_v<Scalar, double>;
  std::vector<Complex> input(n * batch), output(n * batch);
  for (int i = 0; i < n * batch; ++i) {
    input[i] = {static_cast<Scalar>(std::sin(i * 0.071) + 0.1 * (i / n + 1)),
                static_cast<Scalar>(std::cos(i * 0.039))};
  }
  flagfft::adaptor::Memory in(input.size() * sizeof(Complex));
  flagfft::adaptor::Memory out(output.size() * sizeof(Complex));
  flagfft::adaptor::Stream stream;
  flagfft::TritonCompiler compiler;
  for (int direction : {FLAGFFT_FORWARD, FLAGFFT_INVERSE}) {
    flagfft::FFTRequest request;
    request.fft_length = request.requested_n = n;
    request.input_dtype = request.output_dtype = is_double ? "complex128" : "complex64";
    request.device_type = flagfft::adaptor::backend_name();
    int device = 0;
    ASSERT_EQ(flagfft::adaptor::ensure_device(device, request.device_arch), FLAGFFT_SUCCESS);
    request.device_index = device;
    request.direction = direction == FLAGFFT_FORWARD ? "forward" : "inverse";
    request.batch = batch;
    flagfft::PlanBuilder builder;
    auto candidates = builder.build_decomposition_tune_candidates(n, request, 3);
    ASSERT_EQ(candidates.size(), 2U);

    // A direct CPU DFT checks forced algorithms independently of mcFFT.
    std::vector<std::complex<double>> expected(n * batch);
    for (int b = 0; b < batch; ++b) {
      for (int k = 0; k < n; ++k) {
        for (int j = 0; j < n; ++j) {
          const double angle = direction * 2.0 * std::numbers::pi * k * j / n;
          const auto value = input[b * n + j];
          expected[b * n + k] += std::complex<double>(value.x, value.y) * std::polar(1.0, angle);
        }
      }
    }
    for (const auto &candidate : candidates) {
      SCOPED_TRACE(candidate.node->describe());
      auto compiled = compiler.compile_raw_node(candidate.node, request, batch);
      if (candidate.node->kind == flagfft::PlanNodeKind::Bluestein) {
        ASSERT_NE(std::dynamic_pointer_cast<flagfft::CompiledRawBluesteinNode>(compiled), nullptr);
      } else {
        ASSERT_EQ(candidate.node->kind, flagfft::PlanNodeKind::Rader);
      }
      const flagfft::RawExecutionContext context{request, stream.get(), batch};
      auto verify = [&](flagfft::adaptor::Memory &result) {
        stream.sync();
        result.copy_to_host(output.data(), output.size() * sizeof(Complex));
        double error = 0.0, norm = 0.0;
        for (int i = 0; i < n * batch; ++i) {
          error += std::norm(std::complex<double>(output[i].x, output[i].y) - expected[i]);
          norm += std::norm(expected[i]);
        }
        EXPECT_LT(std::sqrt(error / norm), is_double ? 2e-12 : 2e-6);
      };
      in.copy_from_host(input.data(), input.size() * sizeof(Complex));
      // Warm execution also completes the convolution precomputation before capture.
      ASSERT_EQ(compiled->execute(in.get(), out.get(), context), FLAGFFT_SUCCESS);
      verify(out);
      flagfft::adaptor::CudaGraph graph;
      graph.begin_capture(stream.get());
      ASSERT_EQ(compiled->execute(in.get(), out.get(), context), FLAGFFT_SUCCESS);
      graph.end_capture(stream.get());
      ASSERT_TRUE(graph.valid());
      for (int replay = 0; replay < 3; ++replay) {
        graph.launch(stream.get());
      }
      verify(out);
      in.copy_from_host(input.data(), input.size() * sizeof(Complex));
      ASSERT_EQ(compiled->execute(in.get(), in.get(), context), FLAGFFT_SUCCESS);
      verify(in);
    }
  }
}

}  // namespace

TEST(MacaRuntime, ForcedRaderAndBluesteinSupportStreamsGraphsAndInPlace) {
  ASSERT_EQ(flagfft::adaptor::device_count(), 1) << "Select one physical card before starting this test";
  check_prime_algorithms<flagfftComplex>();
  check_prime_algorithms<flagfftDoubleComplex>();
}
