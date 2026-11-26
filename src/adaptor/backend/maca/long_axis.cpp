// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include <mc_runtime.h>

#include <cstddef>
#include <cstdint>

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

constexpr unsigned first_axis = 64;
constexpr unsigned first_rows_per_plane = 2048;
constexpr unsigned first_tile_rows = 32;
constexpr unsigned first_warps = 4;
constexpr unsigned subgroup = 16;

__device__ float2 add(float2 a, float2 b) {
  return make_float2(a.x + b.x, a.y + b.y);
}

__device__ float2 subtract(float2 a, float2 b) {
  return make_float2(a.x - b.x, a.y - b.y);
}

__device__ float2 multiply(float2 value, float2 twiddle) {
  const auto rotated = rotate(Complex<float>{value.x, value.y}, Complex<float>{twiddle.x, twiddle.y});
  return make_float2(rotated.re, rotated.im);
}

template<bool Inverse>
__device__ float2 quarter_turn(float2 value) {
  return Inverse ? make_float2(-value.y, value.x) : make_float2(value.y, -value.x);
}

template<bool Inverse>
__device__ void radix4(float2 (&x)[4]) {
  const auto even = add(x[0], x[2]);
  const auto odd = add(x[1], x[3]);
  const auto delta = subtract(x[0], x[2]);
  const auto turned = quarter_turn<Inverse>(subtract(x[1], x[3]));
  x[0] = add(even, odd);
  x[1] = add(delta, turned);
  x[2] = subtract(even, odd);
  x[3] = subtract(delta, turned);
}

template<unsigned Mask>
__device__ float2 exchange(float2 value) {
  return make_float2(__shfl_xor_sync(lanes, value.x, Mask, wave),
                     __shfl_xor_sync(lanes, value.y, Mask, wave));
}

template<unsigned Stride, bool Inverse>
__device__ float2 radix4_lane(float2 own, unsigned lane) {
  const auto peer1 = exchange<Stride>(own);
  const auto peer2 = exchange<2 * Stride>(own);
  const auto peer3 = exchange<3 * Stride>(own);
  const unsigned digit = (lane / Stride) & 3;
  const bool high = (digit & 2) != 0;
  if ((digit & 1) == 0) {
    const auto even = add(own, peer2);
    const auto odd = add(peer1, peer3);
    return high ? subtract(even, odd) : add(even, odd);
  }
  const auto even_delta = subtract(peer1, peer3);
  const auto turned = quarter_turn<Inverse>(subtract(own, peer2));
  return high ? subtract(turned, even_delta) : add(even_delta, turned);
}

template<bool Aligned>
__device__ float2 load_complex(const float* input, std::size_t offset) {
  if constexpr (Aligned) {
    return reinterpret_cast<const float2*>(input)[offset];
  } else {
    return make_float2(input[2 * offset], input[2 * offset + 1]);
  }
}

template<bool Inverse, bool Aligned>
__global__ void fft64_fp32_first(const float* input, float2* output, const float2* roots) {
  constexpr unsigned threads = first_warps * wave;
  constexpr unsigned groups = threads / subgroup;
  constexpr unsigned tiles_per_plane = first_rows_per_plane / first_tile_rows;
  __shared__ float2 tile[first_tile_rows][first_axis + 1];
  const unsigned thread = threadIdx.x;
  const unsigned group = thread / subgroup;
  const unsigned lane = thread % subgroup;
  const unsigned plane = blockIdx.x / tiles_per_plane;
  const unsigned row_start = (blockIdx.x % tiles_per_plane) * first_tile_rows;
  const auto base = static_cast<std::size_t>(plane) * first_rows_per_plane * first_axis;
  const unsigned out_base = ((lane >> 2) << 2) + ((lane & 3) << 4);
  const unsigned phase = (lane & 3) * (lane >> 2) * 4;
  // Four 16-lane subgroups per wave each transform a separate 64-value row.
  #pragma unroll
  for (unsigned repeat = 0; repeat < first_tile_rows / groups; ++repeat) {
    const unsigned row = group + repeat * groups;
    const auto source = base + (row_start + row) * first_axis + lane;
    float2 x[4] = {load_complex<Aligned>(input, source),
                   load_complex<Aligned>(input, source + subgroup),
                   load_complex<Aligned>(input, source + 2 * subgroup),
                   load_complex<Aligned>(input, source + 3 * subgroup)};
    radix4<Inverse>(x);
    #pragma unroll
    for (unsigned branch = 0; branch < 4; ++branch) {
      auto value = x[branch];
      if (branch != 0) value = multiply(value, roots[branch * lane]);
      value = radix4_lane<4, Inverse>(value, lane);
      value = radix4_lane<1, Inverse>(multiply(value, roots[phase]), lane);
      tile[row][out_base + branch] = value;
    }
  }
  __syncthreads();

  for (unsigned element = thread; element < first_axis * first_tile_rows; element += threads) {
    const unsigned frequency = element / first_tile_rows;
    const unsigned row = element % first_tile_rows;
    output[base + frequency * first_rows_per_plane + row_start + row] = tile[row][frequency];
  }
}

constexpr unsigned middle_tile_rows = 32;
constexpr unsigned middle_tile_cols = 32;
constexpr unsigned middle_warps = 4;
constexpr unsigned middle_tiles_per_plane = (128 / middle_tile_rows) * (2048 / middle_tile_cols);

__global__ void transpose_fp32_middle(const float2* input, float2* output) {
  constexpr unsigned col_tiles = 2048 / middle_tile_cols;
  constexpr unsigned tile_elements = middle_tile_rows * middle_tile_cols;
  constexpr unsigned threads = middle_warps * wave;
  __shared__ float2 tile[middle_tile_rows][middle_tile_cols + 1];
  const unsigned batch_plane = blockIdx.x / middle_tiles_per_plane;
  const unsigned plane = batch_plane % 64;
  const unsigned block = blockIdx.x % middle_tiles_per_plane;
  const unsigned row_start = (block / col_tiles) * middle_tile_rows;
  const unsigned col_start = (block % col_tiles) * middle_tile_cols;
  const auto base = static_cast<std::size_t>(batch_plane / 64) * 128 * 64 * 2048;

  for (unsigned i = threadIdx.x; i < tile_elements; i += threads) {
    const unsigned row = i / middle_tile_cols;
    const unsigned col = i % middle_tile_cols;
    const auto index = base + ((row_start + row) * 64 + plane) * 2048 + col_start + col;
    tile[row][col] = input[index];
  }
  __syncthreads();
  for (unsigned i = threadIdx.x; i < tile_elements; i += threads) {
    const unsigned col = i / middle_tile_rows;
    const unsigned row = i % middle_tile_rows;
    const auto index = base + ((col_start + col) * 64 + plane) * 128 + row_start + row;
    output[index] = tile[row][col];
  }
}

}  // namespace

extern "C" __attribute__((visibility("hidden"))) int flagfft_maca_launch_long_axis(
    const void* input, void* output, const void* roots, unsigned batch,
    bool innermost_fp64, void* stream) {
  if (innermost_fp64) return launch<double, 64, 32, 8, 0>(input, output, roots, batch, stream);
  return launch<float, 128, 16, 4, 1>(input, output, roots, batch, stream);
}

extern "C" __attribute__((visibility("hidden"))) int flagfft_maca_launch_long_axis_fp32_first(
    const void* input, void* output, const void* roots, unsigned batch, bool inverse, void* stream) {
  const bool aligned = reinterpret_cast<std::uintptr_t>(input) % alignof(float2) == 0;
  const auto kernel = inverse
      ? (aligned ? fft64_fp32_first<true, true> : fft64_fp32_first<true, false>)
      : (aligned ? fft64_fp32_first<false, true> : fft64_fp32_first<false, false>);
  void* args[] = {&input, &output, &roots};
  return mcLaunchKernel(reinterpret_cast<const void*>(kernel),
                        dim3(batch * 128 * first_rows_per_plane / first_tile_rows),
                        dim3(first_warps * wave), args, 0, reinterpret_cast<mcStream_t>(stream));
}

extern "C" __attribute__((visibility("hidden"))) int flagfft_maca_launch_long_axis_fp32_middle(
    const void* input, void* output, unsigned batch, void* stream) {
  void* args[] = {&input, &output};
  return mcLaunchKernel(reinterpret_cast<const void*>(transpose_fp32_middle),
                        dim3(batch * 64 * middle_tiles_per_plane), dim3(middle_warps * wave), args,
                        0, reinterpret_cast<mcStream_t>(stream));
}
