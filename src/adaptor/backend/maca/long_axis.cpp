// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include <mc_runtime.h>

#include <cstddef>

namespace {

constexpr unsigned wave = 64;
constexpr unsigned long long lanes = ~0ULL;

template<class T>
struct Complex {
  T re;
  T im;
};

template<class T>
__device__ Complex<T> rotate(Complex<T> value, Complex<T> twiddle) {
  return {value.re * twiddle.re - value.im * twiddle.im,
          value.im * twiddle.re + value.re * twiddle.im};
}

template<unsigned Span, unsigned Length, unsigned Registers, class T>
__device__ void dif(Complex<T> (&values)[Registers][Length / wave], const Complex<T>* roots, unsigned lane) {
  const bool high = (lane & Span) != 0;
  const unsigned phase = (lane & (Span - 1)) * (Length / (2 * Span));
  Complex<T> twiddle;
  if constexpr (Span > 1) twiddle = roots[phase];
  #pragma unroll
  for (unsigned row = 0; row < Registers; ++row) {
    #pragma unroll
    for (unsigned branch = 0; branch < Length / wave; ++branch) {
      const auto own = values[row][branch];
      const T peer_re = __shfl_xor_sync(lanes, own.re, Span, wave);
      const T peer_im = __shfl_xor_sync(lanes, own.im, Span, wave);
      Complex<T> next = high ? Complex<T>{peer_re - own.re, peer_im - own.im}
                             : Complex<T>{own.re + peer_re, own.im + peer_im};
      if constexpr (Span > 1) {
        if (high) next = rotate(next, twiddle);
      }
      values[row][branch] = next;
    }
  }
}

template<unsigned Length, unsigned Rows, unsigned Warps, unsigned Pad, class T>
__global__ void fft_transpose(const T* input, T* output, const Complex<T>* roots) {
  constexpr unsigned registers = Rows / Warps;
  constexpr unsigned per_plane = Length == 64 ? 2048 : 64;
  constexpr unsigned tiles_per_plane = per_plane / Rows;
  constexpr unsigned cube = 128 * 2048 * 64;
  // Each wave transforms complete rows. The shared tile turns them into
  // contiguous stores along the short dimension of the destination.
  __shared__ Complex<T> transposed[Length][Rows + Pad];
  const unsigned thread = threadIdx.x;
  const unsigned warp = thread / wave;
  const unsigned lane = thread & (wave - 1);
  const unsigned plane = blockIdx.x / tiles_per_plane;
  const unsigned tile = (blockIdx.x % tiles_per_plane) * Rows;
  const std::size_t base = static_cast<std::size_t>(plane) * per_plane * Length;
  Complex<T> values[registers][Length / wave];
  #pragma unroll
  for (unsigned row = 0; row < registers; ++row) {
    const auto row_start = base + (tile + row * Warps + warp) * Length;
    const auto* first = input + 2 * (row_start + lane);
    values[row][0] = {first[0], first[1]};
    if constexpr (Length == 128) {
      const auto* second = first + 2 * wave;
      values[row][1] = rotate(Complex<T>{first[0] - second[0], first[1] - second[1]}, roots[lane]);
      values[row][0] = {first[0] + second[0], first[1] + second[1]};
    }
  }
  dif<32, Length, registers>(values, roots, lane);
  dif<16, Length, registers>(values, roots, lane);
  dif<8, Length, registers>(values, roots, lane);
  dif<4, Length, registers>(values, roots, lane);
  dif<2, Length, registers>(values, roots, lane);
  dif<1, Length, registers>(values, roots, lane);

  const unsigned reversed = __brev(lane) >> 26;
  #pragma unroll
  for (unsigned row = 0; row < registers; ++row) {
    const unsigned slot = row * Warps + warp;
    #pragma unroll
    for (unsigned branch = 0; branch < Length / wave; ++branch) {
      transposed[(Length / wave) * reversed + branch][slot] = values[row][branch];
    }
  }
  __syncthreads();

  for (unsigned element = thread; element < Length * Rows; element += Warps * wave) {
    const unsigned frequency = element / Rows;
    const unsigned slot = element % Rows;
    std::size_t address;
    if constexpr (Length == 64) {
      address = base + frequency * per_plane + tile + slot;
    } else {
      const auto cube_base = static_cast<std::size_t>(plane / 2048) * cube;
      address = cube_base + (frequency * 2048 + (plane % 2048)) * per_plane + tile + slot;
    }
    const auto value = transposed[frequency][slot];
    output[2 * address] = value.re;
    output[2 * address + 1] = value.im;
  }
}

template<class T, unsigned Length, unsigned Rows, unsigned Warps, unsigned Pad>
int launch(const void* input, void* output, const void* roots, unsigned batch, void* stream) {
  constexpr unsigned rows_per_cube = Length == 64 ? 128 * 2048 : 2048 * 64;
  const void* kernel = reinterpret_cast<const void*>(fft_transpose<Length, Rows, Warps, Pad, T>);
  void* arguments[] = {&input, &output, &roots};
  return mcLaunchKernel(kernel, dim3(batch * rows_per_cube / Rows), dim3(Warps * wave), arguments,
                        0, reinterpret_cast<mcStream_t>(stream));
}

}  // namespace

extern "C" __attribute__((visibility("hidden"))) int flagfft_maca_launch_long_axis(
    const void* input, void* output, const void* roots, unsigned batch,
    bool innermost_fp64, void* stream) {
  if (innermost_fp64) return launch<double, 64, 32, 8, 0>(input, output, roots, batch, stream);
  return launch<float, 128, 16, 4, 1>(input, output, roots, batch, stream);
}
