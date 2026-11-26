// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include <mc_runtime.h>

namespace {

constexpr unsigned length = 2048;
constexpr unsigned half = length / 2;
constexpr unsigned wave = 64;
constexpr unsigned residues = 8;
constexpr unsigned long long full_wave = ~0ULL;

__device__ float2 sum(float2 a, float2 b) {
  return make_float2(a.x + b.x, a.y + b.y);
}

__device__ float2 diff(float2 a, float2 b) {
  return make_float2(a.x - b.x, a.y - b.y);
}

__device__ float2 rotate(float2 value, float2 twiddle) {
  return make_float2(value.x * twiddle.x - value.y * twiddle.y,
                     value.y * twiddle.x + value.x * twiddle.y);
}

__device__ float2 read_pair(const float* input, unsigned j) {
  return make_float2(input[2 * j], input[2 * j + 1]);
}

__device__ void radix4(float2 (&values)[4]) {
  const float2 even = sum(values[0], values[2]);
  const float2 odd = sum(values[1], values[3]);
  const float2 even_difference = diff(values[0], values[2]);
  const float2 odd_difference = diff(values[1], values[3]);
  const float2 rotated = make_float2(odd_difference.y, -odd_difference.x);
  values[0] = sum(even, odd);
  values[1] = sum(even_difference, rotated);
  values[2] = diff(even, odd);
  values[3] = diff(even_difference, rotated);
}

__device__ float2 radix4_component(float2 a, float2 b, float2 c, float2 d, unsigned k) {
  if ((k & 1) == 0) {
    const float2 even = sum(a, c);
    const float2 odd = sum(b, d);
    return k == 0 ? sum(even, odd) : diff(even, odd);
  }
  const float2 even = diff(a, c);
  const float2 delta = diff(b, d);
  const float2 odd = make_float2(delta.y, -delta.x);
  return k == 1 ? sum(even, odd) : diff(even, odd);
}

__device__ float2 load_residue(const float* src, const float2* roots, unsigned j, unsigned k) {
  constexpr unsigned axis = half / residues;
  constexpr unsigned step = half / 4;
  const unsigned first_frequency = k & 3;
  const float2 even = radix4_component(read_pair(src, j), read_pair(src, j + step),
      read_pair(src, j + 2 * step), read_pair(src, j + 3 * step), first_frequency);
  const float2 odd = radix4_component(read_pair(src, j + axis), read_pair(src, j + axis + step),
      read_pair(src, j + axis + 2 * step), read_pair(src, j + axis + 3 * step), first_frequency);
  const auto value = sum(even, rotate(odd, roots[k * (length / residues)]));
  return rotate(value, roots[2 * j * k]);
}

template<unsigned Span, unsigned Registers>
__device__ void wave_butterfly(float2 (&values)[Registers], const float2* roots, unsigned lane) {
  const bool high = (lane & Span) != 0;
  float2 twiddle;
  if constexpr (Span > 1) {
    twiddle = roots[(lane & (Span - 1)) * (length / (2 * Span))];
  }
  #pragma unroll
  for (unsigned i = 0; i < Registers; ++i) {
    const float imaginary = __shfl_xor_sync(full_wave, values[i].y, Span, wave);
    const float real = __shfl_xor_sync(full_wave, values[i].x, Span, wave);
    const float2 peer = make_float2(real, imaginary);
    const float2 even = high ? peer : values[i];
    float2 odd = high ? values[i] : peer;
    if constexpr (Span > 1) odd = rotate(odd, twiddle);
    values[i] = high ? diff(even, odd) : sum(even, odd);
  }
}

template<unsigned Registers>
__device__ void wave_fft(float2 (&values)[Registers], const float2* roots,
                         unsigned lane, unsigned reversed_lane) {
  if constexpr (Registers == 4) {
    radix4(values);
  } else {
    const auto even = sum(values[0], values[1]);
    values[1] = diff(values[0], values[1]);
    values[0] = even;
  }
  #pragma unroll
  for (unsigned k = 1; k < Registers; ++k) {
    values[k] = rotate(values[k], roots[(length / (wave * Registers)) * reversed_lane * k]);
  }
  wave_butterfly<1>(values, roots, lane);
  wave_butterfly<2>(values, roots, lane);
  wave_butterfly<4>(values, roots, lane);
  wave_butterfly<8>(values, roots, lane);
  wave_butterfly<16>(values, roots, lane);
  wave_butterfly<32>(values, roots, lane);
}

__device__ void reconstruct(float* out, unsigned k, float2 evenodd, float2 reflected,
                            const float2* roots) {
  const float2 terms = diff(evenodd, make_float2(reflected.x, -reflected.y));
  const float2 total = sum(evenodd, make_float2(reflected.x, -reflected.y));
  const float2 phase = roots[k];
  const float2 shifted = rotate(terms, make_float2(phase.y, -phase.x));
  out[2 * k] = 0.5f * (total.x + shifted.x);
  out[2 * k + 1] = 0.5f * (total.y + shifted.y);
  if (k == 0) {
    out[length] = evenodd.x - evenodd.y;
    out[length + 1] = 0.0f;
  }
}

__global__ void shared_fft(const float* input, float* output, const float2* roots) {
  // Partition the packed 1024-point FFT into four 256-point wave FFTs.
  __shared__ float2 first[half];
  constexpr unsigned axis = half / 4;
  const unsigned thread = threadIdx.x;
  const unsigned lane = thread & (wave - 1);
  const unsigned warp = thread / wave;
  const unsigned rev = __brev(lane) >> 26;
  float2 values[4];
  #pragma unroll
  for (unsigned i = 0; i < 4; ++i) values[i] = read_pair(input, thread + axis * i);
  radix4(values);
  #pragma unroll
  for (unsigned i = 1; i < 4; ++i) values[i] = rotate(values[i], roots[2 * thread * i]);
  #pragma unroll
  for (unsigned i = 0; i < 4; ++i) first[axis * i + thread] = values[i];
  __syncthreads();
  #pragma unroll
  for (unsigned i = 0; i < 4; ++i) values[i] = first[axis * warp + rev + wave * i];
  wave_fft(values, roots, lane, rev);
  __syncthreads();
  #pragma unroll
  for (unsigned i = 0; i < 4; ++i) first[4 * (lane * 4 + i) + warp] = values[i];
  __syncthreads();
  #pragma unroll
  for (unsigned i = 0; i < 4; ++i) {
    const unsigned k = thread + axis * i;
    reconstruct(output, k, first[k], first[-k & (half - 1)], roots);
  }
}

__global__ void residue_fft(const float* input, float* output, const float2* roots) {
  // One block computes a residue class and recomputes its reflected pair.
  constexpr unsigned axis = half / residues;
  constexpr unsigned registers = axis / wave;
  __shared__ float2 spectra[2][axis];
  const unsigned residue = blockIdx.x;
  const unsigned reflected = -residue & (residues - 1);
  const unsigned warp = threadIdx.x / wave;
  const unsigned lane = threadIdx.x & (wave - 1);
  const unsigned rev = __brev(lane) >> 26;
  const unsigned requested = warp == 0 ? residue : reflected;
  float2 values[registers];
  #pragma unroll
  for (unsigned i = 0; i < registers; ++i) {
    values[i] = load_residue(input, roots, rev + wave * i, requested);
  }
  wave_fft(values, roots, lane, rev);
  #pragma unroll
  for (unsigned i = 0; i < registers; ++i) spectra[warp][lane * registers + i] = values[i];
  __syncthreads();
  const unsigned k = threadIdx.x;
  const unsigned mirror = (-k - (residue != 0)) & (axis - 1);
  reconstruct(output, k * residues + residue, spectra[0][k], spectra[1][mirror], roots);
}

}  // namespace

extern "C" __attribute__((visibility("hidden"))) int flagfft_maca_launch_fp32_r2c2048(
    const float* input, float* output, const void* phase, bool use_residues, void* stream) {
  // The independent residue blocks share the input. Exact in-place requests
  // use the shared-memory schedule, which consumes all samples before storing.
  const bool split = use_residues && input != output;
  const auto kernel = split ? residue_fft : shared_fft;
  void* arguments[] = {&input, &output, &phase};
  return mcLaunchKernel(reinterpret_cast<const void*>(kernel), dim3(split ? residues : 1),
                        dim3(split ? 2 * wave : 4 * wave), arguments, 0,
                        reinterpret_cast<mcStream_t>(stream));
}
