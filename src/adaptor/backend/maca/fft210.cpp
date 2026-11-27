// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include <mc_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {

constexpr unsigned outer_length = 7;
constexpr unsigned middle_length = 5;
constexpr unsigned inner_length = 6;
constexpr unsigned plane = middle_length * inner_length;
constexpr unsigned length = outer_length * plane;
constexpr unsigned pitch = 224;
constexpr unsigned threads = 128;

template<unsigned Radix, unsigned Stride>
__device__ float2 radix_component(const float2* data, unsigned index, const float2* roots) {
  const unsigned digit = index / Stride % Radix;
  const unsigned first = index - digit * Stride;
  auto value = data[first];
  #pragma unroll
  for (unsigned term = 1; term < Radix; ++term) {
    const auto input = data[first + term * Stride];
    const auto root = roots[(digit * term % Radix) * (length / Radix)];
    value.x = fmaf(input.x, root.x, value.x);
    value.x = fmaf(-input.y, root.y, value.x);
    value.y = fmaf(input.x, root.y, value.y);
    value.y = fmaf(input.y, root.x, value.y);
  }
  return value;
}

template<bool Aligned>
__global__ void fft210(const float* input, float* output, const float2* roots) {
  __shared__ float2 first[pitch];
  __shared__ float2 second[pitch];
  const unsigned thread = threadIdx.x;
  const auto base = static_cast<std::size_t>(blockIdx.x) * length;
  for (unsigned index = thread; index < length; index += threads) {
    const unsigned outer = index / plane;
    const unsigned middle = index / inner_length % middle_length;
    const unsigned inner = index % inner_length;
    // Good–Thomas input strides are N divided by the corresponding radix.
    const auto pos = base + (outer * plane + middle * (length / middle_length) +
                             inner * (length / inner_length)) % length;
    if constexpr (Aligned) {
      first[index] = reinterpret_cast<const float2*>(input)[pos];
    } else {
      first[index] = make_float2(input[2 * pos], input[2 * pos + 1]);
    }
  }
  __syncthreads();

  for (unsigned index = thread; index < length; index += threads) {
    second[index] = radix_component<inner_length, 1>(first, index, roots);
  }
  __syncthreads();

  for (unsigned index = thread; index < length; index += threads) {
    first[index] = radix_component<middle_length, inner_length>(second, index, roots);
  }
  __syncthreads();

  for (unsigned index = thread; index < length; index += threads) {
    const unsigned outer = index / plane;
    const unsigned middle = index / inner_length % middle_length;
    const unsigned inner = index % inner_length;
    // These CRT weights restore the natural frequency order for 7 × 5 × 6.
    const auto pos = base + (outer * 120 + middle * 126 + inner * 175) % length;
    const auto value = radix_component<outer_length, plane>(first, index, roots);
    if constexpr (Aligned) {
      reinterpret_cast<float2*>(output)[pos] = value;
    } else {
      output[2 * pos] = value.x;
      output[2 * pos + 1] = value.y;
    }
  }
}

}  // namespace

extern "C" __attribute__((visibility("hidden"))) int flagfft_maca_launch_fp32_fft210(
    const void* input, void* output, const void* roots, unsigned batch, void* stream) {
  const auto address = reinterpret_cast<std::uintptr_t>(input) | reinterpret_cast<std::uintptr_t>(output);
  const auto kernel = address % alignof(float2) == 0 ? fft210<true> : fft210<false>;
  void* args[] = {&input, &output, &roots};
  return mcLaunchKernel(reinterpret_cast<const void*>(kernel), dim3(batch), dim3(threads), args, 0,
                        reinterpret_cast<mcStream_t>(stream));
}
