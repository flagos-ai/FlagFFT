// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

// Analytic real-transform checks that do not require a device-pointer vendor
// reference API. Run with FLAGFFT_PACKED_REAL=1 to also cover batched fallback.
#include "adaptor/adaptor.h"
#include "flagfft.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void check(flagfftResult result) {
  if (result != FLAGFFT_SUCCESS)
    throw std::runtime_error("FlagFFT returned " + std::to_string(result));
}

struct Plan {
  flagfftHandle handle = nullptr;
  ~Plan() { if (handle) flagfftDestroy(handle); }
};

void near(double value, double expected, double tolerance) {
  if (!std::isfinite(value) || std::abs(value - expected) > tolerance)
    throw std::runtime_error("expected " + std::to_string(expected) + ", got " + std::to_string(value));
}

void run(int n, int batch, bool in_place) {
  const int half = n / 2 + 1;
  const int distance = in_place ? 2 * half : n;
  Plan forward, inverse;
  check(flagfftPlanMany(&forward.handle, 1, &n, nullptr, 1, distance,
                       nullptr, 1, half, FLAGFFT_R2C, batch));
  check(flagfftPlanMany(&inverse.handle, 1, &n, nullptr, 1, half,
                       nullptr, 1, distance, FLAGFFT_C2R, batch));
  flagfft::adaptor::Memory real(static_cast<size_t>(distance) * batch * sizeof(float));
  flagfft::adaptor::Memory spectrum(static_cast<size_t>(half) * batch * sizeof(flagfftComplex));
  auto* real_ptr = static_cast<float*>(real.data());
  auto* complex_ptr = in_place ? reinterpret_cast<flagfftComplex*>(real_ptr)
                               : static_cast<flagfftComplex*>(spectrum.data());
  auto& complex_memory = in_place ? real : spectrum;
  std::vector<float> input(static_cast<size_t>(distance) * batch), output(input.size());
  std::vector<flagfftComplex> frequencies(static_cast<size_t>(half) * batch);
  for (float scale : {1.0e-6f, 1.0f, 1.0e6f}) {
    for (int pattern = 0; pattern < 3; ++pattern) {
      std::fill(input.begin(), input.end(), -99.0f);
      for (int b = 0; b < batch; ++b) {
        const float amplitude = scale * (b + 1);
        for (int k = 0; k < n; ++k)
          input[b * distance + k] = pattern == 0 ? amplitude :
              pattern == 1 ? ((k % 2) ? -amplitude : amplitude) :
              (k == 0 ? amplitude : 0.0f);
      }
      real.copy_from_host(input.data(), input.size() * sizeof(float));
      check(flagfftExecR2C(forward.handle, real_ptr, complex_ptr));
      flagfft::adaptor::synchronize();
      complex_memory.copy_to_host(frequencies.data(), frequencies.size() * sizeof(flagfftComplex));
      for (int b = 0; b < batch; ++b) {
        const double amplitude = scale * (b + 1);
        for (int k = 0; k < half; ++k) {
          const double expected = pattern == 2 ? amplitude :
              ((pattern == 0 && k == 0) || (pattern == 1 && k == n / 2) ? n * amplitude : 0.0);
          near(frequencies[b * half + k].x, expected, 2.0e-5 * n * amplitude);
          near(frequencies[b * half + k].y, 0.0, 2.0e-5 * n * amplitude);
        }
      }
      // Feed an independent analytic spectrum into C2R, so matching errors in
      // the forward and inverse kernels cannot cancel in a round trip.
      for (int b = 0; b < batch; ++b) {
        const float amplitude = scale * (b + 1);
        for (int k = 0; k < half; ++k) {
          const float value = pattern == 2 ? amplitude :
              ((pattern == 0 && k == 0) || (pattern == 1 && k == n / 2) ? n * amplitude : 0.0f);
          frequencies[b * half + k] = {value, 0.0f};
        }
      }
      complex_memory.copy_from_host(frequencies.data(), frequencies.size() * sizeof(flagfftComplex));
      check(flagfftExecC2R(inverse.handle, complex_ptr, real_ptr));
      flagfft::adaptor::synchronize();
      real.copy_to_host(output.data(), output.size() * sizeof(float));
      for (int b = 0; b < batch; ++b)
        for (int k = 0; k < n; ++k)
          near(output[b * distance + k], n * static_cast<double>(input[b * distance + k]),
               2.0e-5 * n * scale * (b + 1));
    }
  }
  std::cout << "Passed n=" << n << " batch=" << batch << " in_place=" << in_place << '\n';
}
}  // namespace

int main() {
  try {
    for (int n : {16, 1024, 2048})
      for (bool in_place : {false, true}) run(n, 1, in_place);
    // In-place batches carry padding between real rows and must take the
    // distance-aware fallback when packed transforms are forced.
    run(16, 3, false);
    run(16, 3, true);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
