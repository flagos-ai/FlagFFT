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

template<bool Wide, class T>
__device__ Complex<T> load_pair(const T* pointer) {
  if constexpr (Wide) {
    const auto pair = *reinterpret_cast<const float2*>(pointer);
    return {pair.x, pair.y};
  }
  return {pointer[0], pointer[1]};
}

template<unsigned Length, unsigned Rows, unsigned Warps, unsigned Pad, class T, bool Wide>
__global__ void fft_transpose(const T* input, T* output, const Complex<T>* roots) {
  static_assert(!Wide || sizeof(T) == sizeof(float));
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
    const auto* address = input + 2 * (row_start + lane);
    const auto first = load_pair<Wide>(address);
    values[row][0] = first;
    if constexpr (Length == 128) {
      const auto second = load_pair<Wide>(address + 2 * wave);
      values[row][1] = rotate(Complex<T>{first.re - second.re, first.im - second.im}, roots[lane]);
      values[row][0] = {first.re + second.re, first.im + second.im};
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
      const unsigned frequency = (Length / wave) * reversed + branch;
      // Spread a wave's strided frequency stores over the shared-memory banks.
      const unsigned shared_slot = Wide ? slot ^ (frequency & (Rows - 1)) : slot;
      transposed[frequency][shared_slot] = values[row][branch];
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
    const unsigned shared_slot = Wide ? slot ^ (frequency & (Rows - 1)) : slot;
    const auto value = transposed[frequency][shared_slot];
    if constexpr (Wide) {
      reinterpret_cast<float2*>(output)[address] = make_float2(value.re, value.im);
    } else {
      output[2 * address] = value.re;
      output[2 * address + 1] = value.im;
    }
  }
}

template<class T, unsigned Length, unsigned Rows, unsigned Warps, unsigned Pad, bool Wide = false>
int launch(const void* input, void* output, const void* roots, unsigned batch, void* stream) {
  constexpr unsigned rows_per_cube = Length == 64 ? 128 * 2048 : 2048 * 64;
  const void* kernel = reinterpret_cast<const void*>(fft_transpose<Length, Rows, Warps, Pad, T, Wide>);
  void* arguments[] = {&input, &output, &roots};
  return mcLaunchKernel(kernel, dim3(batch * rows_per_cube / Rows), dim3(Warps * wave), arguments,
                        0, reinterpret_cast<mcStream_t>(stream));
}

constexpr unsigned first_axis = 64;
constexpr unsigned first_rows_per_plane = 2048;
constexpr unsigned first_tile_rows = 32;
constexpr unsigned subgroup = 16;

template<bool TwoShuffle>
constexpr unsigned first_warps = TwoShuffle ? 2 : 4;

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

__device__ unsigned reverse_radix4_digit(unsigned digit) {
  return ((digit & 1) << 1) | (digit >> 1);
}

template<unsigned Stride, bool Inverse, bool TwoShuffle>
__device__ float2 first_axis_radix4(float2 own, unsigned lane) {
  if constexpr (TwoShuffle) {
    const unsigned digit = (lane / Stride) & 3;
    const auto opposite = exchange<2 * Stride>(own);
    auto value = (digit & 2) ? subtract(opposite, own) : add(own, opposite);
    if (digit == 3) value = quarter_turn<Inverse>(value);
    const auto partner = exchange<Stride>(value);
    // The two radix-2 butterflies leave the frequency digit bit-reversed.
    return (digit & 1) ? subtract(partner, value) : add(partner, value);
  }
  return radix4_lane<Stride, Inverse>(own, lane);
}

template<bool Aligned>
__device__ float2 load_complex(const float* input, std::size_t offset) {
  if constexpr (Aligned) {
    return reinterpret_cast<const float2*>(input)[offset];
  } else {
    return make_float2(input[2 * offset], input[2 * offset + 1]);
  }
}

template<unsigned Mask>
__device__ Complex<double> exchange_fp64(Complex<double> value) {
  return {__shfl_xor_sync(lanes, value.re, Mask, wave),
          __shfl_xor_sync(lanes, value.im, Mask, wave)};
}

template<unsigned Stride, bool Inverse>
__device__ Complex<double> radix4_fp64(Complex<double> value, const Complex<double>* roots, unsigned lane) {
  const unsigned digit = (lane / Stride) & 3;
  const auto opposite = exchange_fp64<2 * Stride>(value);
  auto first = (digit & 2) ? Complex<double>{opposite.re - value.re, opposite.im - value.im}
                           : Complex<double>{opposite.re + value.re, opposite.im + value.im};
  if (digit == 3) {
    first = Inverse ? Complex<double>{-first.im, first.re} : Complex<double>{first.im, -first.re};
  }
  const auto partner = exchange_fp64<Stride>(first);
  const auto second = (digit & 1) ? Complex<double>{partner.re - first.re, partner.im - first.im}
                                  : Complex<double>{partner.re + first.re, partner.im + first.im};
  if constexpr (Stride == 1) {
    return second;
  } else {
    const unsigned phase = reverse_radix4_digit(digit) * (lane % Stride) * (wave / (4 * Stride));
    return rotate(second, roots[phase]);
  }
}

constexpr unsigned fp64_first_warps = 8;

template<bool Inverse, bool Aligned>
__global__ void fft64_fp64_radix4(const double* input, double* output, const Complex<double>* roots) {
  constexpr unsigned tiles = first_rows_per_plane / first_tile_rows;
  __shared__ Complex<double> tile[wave][first_tile_rows];

  const unsigned thread = threadIdx.x;
  const unsigned warp = thread / wave;
  const unsigned lane = thread % wave;
  const unsigned plane = blockIdx.x / tiles;
  const unsigned first_row = (blockIdx.x % tiles) * first_tile_rows;
  const auto base = static_cast<std::size_t>(plane) * first_rows_per_plane * wave;
  const unsigned frequency = __brev(lane) >> 26;
  #pragma unroll
  for (unsigned group = 0; group < first_tile_rows / fp64_first_warps; ++group) {
    const unsigned slot = group * fp64_first_warps + warp;
    const auto index = base + (first_row + slot) * wave + lane;
    Complex<double> value;
    if constexpr (Aligned) {
      const auto pair = reinterpret_cast<const double2*>(input)[index];
      value = {pair.x, pair.y};
    } else {
      value = {input[2 * index], input[2 * index + 1]};
    }
    value = radix4_fp64<16, Inverse>(value, roots, lane);
    value = radix4_fp64<4, Inverse>(value, roots, lane);
    value = radix4_fp64<1, Inverse>(value, roots, lane);
    tile[frequency][slot] = value;
  }
  __syncthreads();

  for (unsigned index = thread; index < wave * first_tile_rows; index += fp64_first_warps * wave) {
    const unsigned k = index / first_tile_rows;
    const unsigned slot = index % first_tile_rows;
    const auto dst = base + k * first_rows_per_plane + first_row + slot;
    const auto value = tile[k][slot];
    if constexpr (Aligned) {
      reinterpret_cast<double2*>(output)[dst] = make_double2(value.re, value.im);
    } else {
      output[2 * dst] = value.re;
      output[2 * dst + 1] = value.im;
    }
  }
}

constexpr unsigned prime_real_tile_rows = 32;
constexpr unsigned prime_real_warps = 4;

template<bool Aligned>
__global__ void prime_r2c64_fp32(const float* input, float* output, const float2* roots) {
  constexpr unsigned packed_length = 32;
  constexpr unsigned half = 33;
  constexpr unsigned middle_length = 997;
  constexpr unsigned tiles_per_plane = (middle_length + prime_real_tile_rows - 1) / prime_real_tile_rows;
  constexpr unsigned rows_per_round = prime_real_warps * wave / packed_length;
  __shared__ float2 tile[prime_real_tile_rows][half + 1];

  const unsigned thread = threadIdx.x;
  const unsigned lane = thread % packed_length;
  const unsigned first_row = (blockIdx.x % tiles_per_plane) * prime_real_tile_rows;
  const unsigned plane = blockIdx.x / tiles_per_plane;
  const unsigned frequency = __brev(lane) >> 27;
  const unsigned mirror_lane = (__brev((-frequency) & (packed_length - 1)) >> 27) +
                               ((thread % wave) & packed_length);
  const auto phase = roots[packed_length + frequency];
  const unsigned first_phase = reverse_radix4_digit((lane >> 3) & 3) * (lane & 7);
  const unsigned second_phase = reverse_radix4_digit((lane >> 1) & 3) * (lane & 1) * 4;

  #pragma unroll
  for (unsigned round = 0; round < prime_real_tile_rows / rows_per_round; ++round) {
    const unsigned row = thread / packed_length + round * rows_per_round;
    const unsigned middle = first_row + row;
    const auto index = (static_cast<std::size_t>(plane) * middle_length + middle) * packed_length + lane;
    auto value = middle < middle_length ? load_complex<Aligned>(input, index) : make_float2(0.0f, 0.0f);
    value = multiply(first_axis_radix4<8, false, true>(value, lane), roots[first_phase]);
    value = multiply(first_axis_radix4<2, false, true>(value, lane), roots[second_phase]);
    const auto partner = exchange<1>(value);
    value = (lane & 1) ? subtract(partner, value) : add(value, partner);
    const auto mirror = make_float2(__shfl_sync(lanes, value.x, mirror_lane, wave),
                                    __shfl_sync(lanes, value.y, mirror_lane, wave));
    const float dr = value.x - mirror.x;
    const float di = value.y + mirror.y;
    tile[row][frequency] = make_float2(0.5f * (value.x + mirror.x + dr * phase.y + di * phase.x),
                                        0.5f * (value.y - mirror.y - dr * phase.x + di * phase.y));
    if (lane == 0) tile[row][packed_length] = make_float2(value.x - value.y, 0.0f);
  }
  __syncthreads();

  for (unsigned flat = thread; flat < half * prime_real_tile_rows; flat += prime_real_warps * wave) {
    const unsigned row = flat % prime_real_tile_rows;
    const unsigned k = flat / prime_real_tile_rows;
    if (first_row + row < middle_length) {
      const auto offset = (static_cast<std::size_t>(plane) * half + k) * middle_length + first_row + row;
      const auto value = tile[row][k];
      if constexpr (Aligned) {
        reinterpret_cast<float2*>(output)[offset] = value;
      } else {
        output[2 * offset] = value.x;
        output[2 * offset + 1] = value.y;
      }
    }
  }
}

template<bool Inverse, bool Aligned, bool TwoShuffle>
__global__ void fft64_fp32_first(const float* input, float2* output, const float2* roots) {
  constexpr unsigned threads = first_warps<TwoShuffle> * wave;
  constexpr unsigned groups = threads / subgroup;
  constexpr unsigned tiles_per_plane = first_rows_per_plane / first_tile_rows;
  __shared__ float2 tile[first_tile_rows][first_axis + 1];
  const unsigned thread = threadIdx.x;
  const unsigned group = thread / subgroup;
  const unsigned lane = thread % subgroup;
  const unsigned plane = blockIdx.x / tiles_per_plane;
  const unsigned row_start = (blockIdx.x % tiles_per_plane) * first_tile_rows;
  const auto base = static_cast<std::size_t>(plane) * first_rows_per_plane * first_axis;
  const unsigned middle = TwoShuffle ? reverse_radix4_digit(lane >> 2) : lane >> 2;
  const unsigned last = TwoShuffle ? reverse_radix4_digit(lane & 3) : lane & 3;
  const unsigned out_base = middle * 4 + last * 16;
  const unsigned phase = (lane & 3) * middle * 4;
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
      value = first_axis_radix4<4, Inverse, TwoShuffle>(value, lane);
      value = first_axis_radix4<1, Inverse, TwoShuffle>(multiply(value, roots[phase]), lane);
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

template<bool TwoShuffle>
int launch_fp32_first(const void* input, void* output, const void* roots, unsigned batch,
                     bool inverse, void* stream) {
  const bool aligned = reinterpret_cast<std::uintptr_t>(input) % alignof(float2) == 0;
  const auto kernel = inverse
      ? (aligned ? fft64_fp32_first<true, true, TwoShuffle> : fft64_fp32_first<true, false, TwoShuffle>)
      : (aligned ? fft64_fp32_first<false, true, TwoShuffle> : fft64_fp32_first<false, false, TwoShuffle>);
  void* args[] = {&input, &output, &roots};
  return mcLaunchKernel(reinterpret_cast<const void*>(kernel),
                        dim3(batch * 128 * first_rows_per_plane / first_tile_rows),
                        dim3(first_warps<TwoShuffle> * wave), args, 0, reinterpret_cast<mcStream_t>(stream));
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

extern "C" __attribute__((visibility("hidden"))) int flagfft_maca_launch_long_axis_fp64_radix4(
    const void* input, void* output, const void* roots, unsigned batch, bool inverse, void* stream) {
  const auto address = reinterpret_cast<std::uintptr_t>(input) | reinterpret_cast<std::uintptr_t>(output);
  const bool aligned = address % alignof(double2) == 0;
  const auto kernel = inverse
      ? (aligned ? fft64_fp64_radix4<true, true> : fft64_fp64_radix4<true, false>)
      : (aligned ? fft64_fp64_radix4<false, true> : fft64_fp64_radix4<false, false>);
  void* args[] = {&input, &output, &roots};
  return mcLaunchKernel(reinterpret_cast<const void*>(kernel),
                        dim3(batch * 128 * first_rows_per_plane / first_tile_rows),
                        dim3(fp64_first_warps * wave), args, 0, reinterpret_cast<mcStream_t>(stream));
}

extern "C" __attribute__((visibility("hidden"))) int flagfft_maca_launch_long_axis_fp32_last_wide(
    const void* input, void* output, const void* roots, unsigned batch, void* stream) {
  const auto addresses = reinterpret_cast<std::uintptr_t>(input) | reinterpret_cast<std::uintptr_t>(output);
  if (addresses % alignof(float2) != 0) {
    return launch<float, 128, 16, 4, 1>(input, output, roots, batch, stream);
  }
  return launch<float, 128, 32, 4, 0, true>(input, output, roots, batch, stream);
}

extern "C" __attribute__((visibility("hidden"))) int flagfft_maca_launch_long_axis_fp32_first(
    const void* input, void* output, const void* roots, unsigned batch, bool inverse, bool two_shuffle,
    void* stream) {
  if (two_shuffle) return launch_fp32_first<true>(input, output, roots, batch, inverse, stream);
  return launch_fp32_first<false>(input, output, roots, batch, inverse, stream);
}

extern "C" __attribute__((visibility("hidden"))) int flagfft_maca_launch_long_axis_fp32_middle(
    const void* input, void* output, unsigned batch, void* stream) {
  void* args[] = {&input, &output};
  return mcLaunchKernel(reinterpret_cast<const void*>(transpose_fp32_middle),
                        dim3(batch * 64 * middle_tiles_per_plane), dim3(middle_warps * wave), args,
                        0, reinterpret_cast<mcStream_t>(stream));
}

extern "C" __attribute__((visibility("hidden"))) int flagfft_maca_launch_prime_r2c64_fp32(
    const void* input, void* output, const void* roots, unsigned padded_rows, void* stream) {
  const auto addresses = reinterpret_cast<std::uintptr_t>(input) | reinterpret_cast<std::uintptr_t>(output);
  const auto kernel = addresses % alignof(float2) == 0 ? prime_r2c64_fp32<true> : prime_r2c64_fp32<false>;
  void* args[] = {&input, &output, &roots};
  return mcLaunchKernel(reinterpret_cast<const void*>(kernel), dim3(padded_rows / prime_real_tile_rows),
                        dim3(prime_real_warps * wave), args, 0, reinterpret_cast<mcStream_t>(stream));
}
