// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include <mc_runtime.h>

#include <cstdint>

namespace {

constexpr unsigned length = 16;
constexpr unsigned block_threads = 64;
constexpr unsigned long long active_lanes = (1ULL << length) - 1;

template<int Sign>
__device__ double2 rotate(double2 value, unsigned frequency) {
  constexpr double cos_pi_8 = 0.92387953251128675613;
  constexpr double sin_pi_8 = 0.38268343236508977173;
  constexpr double sqrt_half = 0.70710678118654752440;
  const unsigned reduced = frequency <= 4 ? frequency : 8 - frequency;
  const double cosine = reduced == 0 ? 1.0 : reduced == 1 ? cos_pi_8 :
                        reduced == 2 ? sqrt_half : reduced == 3 ? sin_pi_8 : 0.0;
  const double sine = reduced == 0 ? 0.0 : reduced == 1 ? sin_pi_8 :
                      reduced == 2 ? sqrt_half : reduced == 3 ? cos_pi_8 : 1.0;
  const double re = frequency <= 4 ? cosine : -cosine;
  const double im = Sign * sine;
  return make_double2(value.x * re - value.y * im, value.x * im + value.y * re);
}

template<int Sign, unsigned Half>
__device__ double2 butterfly(double2 value, unsigned lane) {
  const double peer_re = __shfl_xor_sync(active_lanes, value.x, Half, length);
  const double peer_im = __shfl_xor_sync(active_lanes, value.y, Half, length);
  const double2 peer = make_double2(peer_re, peer_im);
  const bool upper = (lane & Half) != 0;
  const double2 even = upper ? peer : value;
  double2 odd = upper ? value : peer;
  if constexpr (Half > 1) {
    odd = rotate<Sign>(odd, (lane & (Half - 1)) * (8 / Half));
  }
  return upper ? make_double2(even.x - odd.x, even.y - odd.y)
               : make_double2(even.x + odd.x, even.y + odd.y);
}

__device__ unsigned reverse_four_bits(unsigned lane) {
  return ((lane & 1) << 3) | ((lane & 2) << 1) | ((lane & 4) >> 1) | ((lane & 8) >> 3);
}

template<int Sign>
__device__ double2 fft16_butterflies(double2 value, unsigned lane) {
  value = butterfly<Sign, 1>(value, lane);
  value = butterfly<Sign, 2>(value, lane);
  value = butterfly<Sign, 4>(value, lane);
  return butterfly<Sign, 8>(value, lane);
}

template<int Sign>
__global__ void fft16(const double* input, double* output) {
  const unsigned lane = threadIdx.x;
  if (lane >= length) return;
  const unsigned row_offset = 2 * blockIdx.x * length;
  // Complex FP64 buffers may have only eight-byte alignment.
  const auto* src = input + row_offset + 2 * reverse_four_bits(lane);
  const auto value = fft16_butterflies<Sign>(make_double2(src[0], src[1]), lane);
  auto* dst = output + row_offset + 2 * lane;
  dst[0] = value.x;
  dst[1] = value.y;
}

__global__ void d2z16(const double* input, double* output, std::int64_t input_distance,
                      std::int64_t output_distance) {
  const unsigned lane = threadIdx.x;
  if (lane >= length) return;
  const auto* src = input + blockIdx.x * input_distance;
  const auto value = fft16_butterflies<-1>(make_double2(src[reverse_four_bits(lane)], 0.0), lane);
  if (lane <= length / 2) {
    auto* dst = output + 2 * (blockIdx.x * output_distance + lane);
    dst[0] = value.x;
    dst[1] = value.y;
  }
}

}  // namespace

extern "C" __attribute__((visibility("hidden"))) int flagfft_maca_launch_fp64_fft16(
    const double* input, double* output, unsigned batch, int direction, void* stream) {
  const auto kernel = direction < 0 ? fft16<-1> : fft16<1>;
  void* args[] = {&input, &output};
  return mcLaunchKernel(reinterpret_cast<const void*>(kernel), dim3(batch), dim3(block_threads),
                        args, 0, reinterpret_cast<mcStream_t>(stream));
}

extern "C" __attribute__((visibility("hidden"))) int flagfft_maca_launch_fp64_d2z16(
    const double* input, double* output, unsigned batch, std::int64_t input_distance,
    std::int64_t output_distance, void* stream) {
  void* args[] = {&input, &output, &input_distance, &output_distance};
  return mcLaunchKernel(reinterpret_cast<const void*>(d2z16), dim3(batch), dim3(block_threads),
                        args, 0, reinterpret_cast<mcStream_t>(stream));
}
