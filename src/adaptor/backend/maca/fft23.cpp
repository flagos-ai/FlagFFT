// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include <mc_runtime.h>

namespace {

constexpr unsigned n = 23;
constexpr unsigned warp = 64;
constexpr unsigned long long full_wave = ~0ULL;

__global__ void cooperative(const double* input, double* output, const double2* roots) {
  constexpr unsigned width = 32;
  const unsigned k = blockIdx.x * (warp / width) + threadIdx.x / width;
  const unsigned lane = threadIdx.x & (width - 1);
  double real = 0.0;
  double imag = 0.0;
  if (k < n && lane < n) {
    const auto q = roots[(k * lane) % n];
    const double re = input[2 * lane];
    const double im = input[2 * lane + 1];
    real = re * q.x - im * q.y;
    imag = im * q.x + re * q.y;
  }
  #pragma unroll
  for (unsigned step = width / 2; step != 0; step /= 2) {
    real += __shfl_xor_sync(full_wave, real, step, width);
    imag += __shfl_xor_sync(full_wave, imag, step, width);
  }
  if (k < n && lane == 0) {
    double* dst = output + 2 * k;
    dst[0] = real;
    dst[1] = imag;
  }
}

__global__ void resident(const double* input, double* output, const double2* roots) {
  const unsigned k = threadIdx.x;
  if (k >= 32) return;
  double x = 0.0;
  double y = 0.0;
  if (k < n) {
    x = input[2 * k];
    y = input[2 * k + 1];
  }
  double real = 0.0;
  double imag = 0.0;
  #pragma unroll
  for (unsigned j = 0; j < n; ++j) {
    const double re = __shfl_sync(0xffffffffULL, x, j, 32);
    const double im = __shfl_sync(0xffffffffULL, y, j, 32);
    const auto q = roots[(k * j) % n];
    real += re * q.x - im * q.y;
    imag += im * q.x + re * q.y;
  }
  if (k < n) {
    double* dst = output + 2 * k;
    dst[0] = real;
    dst[1] = imag;
  }
}

}  // namespace

extern "C" __attribute__((visibility("hidden"))) int flagfft_maca_launch_fp64_fft23(
    const double* input, double* output, const void* roots, void* stream) {
  // The resident wave reads the whole transform into registers before the
  // first write. The disjoint path can distribute frequencies across blocks.
  const bool inplace = input == output;
  const auto kernel = inplace ? resident : cooperative;
  void* arguments[] = {&input, &output, &roots};
  return mcLaunchKernel(reinterpret_cast<const void*>(kernel), dim3(inplace ? 1 : (n + 1) / 2),
                        dim3(warp), arguments, 0, reinterpret_cast<mcStream_t>(stream));
}
