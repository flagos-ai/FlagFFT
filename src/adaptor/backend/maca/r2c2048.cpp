// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include <mc_runtime.h>
#include <cstdint>

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

template<bool Batched>
__global__ void shared_fft(const float* input, float* output, const float2* roots,
                           std::int64_t input_distance, std::int64_t output_distance) {
  // Partition the packed 1024-point FFT into four 256-point wave FFTs.
  __shared__ float2 first[half];
  if constexpr (Batched) {
    input += static_cast<std::int64_t>(blockIdx.x) * input_distance;
    output += static_cast<std::int64_t>(blockIdx.x) * output_distance * 2;
  }
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

__device__ float2 pack_inverse(const float* input, const float2* forward_roots, unsigned k) {
  auto a = read_pair(input, k);
  auto b = read_pair(input, half - k);
  if (k == 0) {
    a.y = 0.0f;
    b.y = 0.0f;
  }
  const float dr = a.x - b.x;
  const float di = a.y + b.y;
  const float co = forward_roots[k].x;
  const float si = -forward_roots[k].y;
  // Conjugate the packed spectrum to reuse the forward wave FFT.
  return make_float2(a.x + b.x - dr * si - di * co,
                     b.y - a.y - dr * co + di * si);
}

__device__ void store_inverse(float* output, unsigned k, float2 value) {
  output[2 * k] = value.x;
  output[2 * k + 1] = -value.y;
}

__global__ void inverse_shared_fft(const float* input, float* output, const float2* roots) {
  constexpr unsigned axis = half / 4;
  __shared__ float2 first[half];
  const unsigned thread = threadIdx.x;
  const unsigned lane = thread & (wave - 1);
  const unsigned warp = thread / wave;
  const unsigned rev = __brev(lane) >> 26;
  float2 values[4];
  #pragma unroll
  for (unsigned i = 0; i < 4; ++i) values[i] = pack_inverse(input, roots, thread + axis * i);
  radix4(values);
  #pragma unroll
  for (unsigned i = 1; i < 4; ++i) values[i] = rotate(values[i], roots[2 * thread * i]);
  #pragma unroll
  for (unsigned i = 0; i < 4; ++i) first[axis * i + thread] = values[i];
  __syncthreads();
  #pragma unroll
  for (unsigned i = 0; i < 4; ++i) values[i] = first[axis * warp + rev + wave * i];
  wave_fft(values, roots, lane, rev);
  #pragma unroll
  for (unsigned i = 0; i < 4; ++i) store_inverse(output, 4 * (lane * 4 + i) + warp, values[i]);
}

__device__ float2 load_inverse_residue(const float* input, const float2* roots, unsigned j, unsigned k) {
  constexpr unsigned step = half / 4;
  const unsigned first_frequency = k & 3;
  const auto even = radix4_component(pack_inverse(input, roots, j),
      pack_inverse(input, roots, j + step), pack_inverse(input, roots, j + 2 * step),
      pack_inverse(input, roots, j + 3 * step), first_frequency);
  const unsigned odd_j = j + half / residues;
  const auto odd = radix4_component(pack_inverse(input, roots, odd_j),
      pack_inverse(input, roots, odd_j + step), pack_inverse(input, roots, odd_j + 2 * step),
      pack_inverse(input, roots, odd_j + 3 * step), first_frequency);
  const auto value = sum(even, rotate(odd, roots[k * (length / residues)]));
  return rotate(value, roots[2 * j * k]);
}

__global__ void inverse_residue_fft(const float* input, float* output, const float2* roots) {
  constexpr unsigned registers = half / (residues * wave);
  const unsigned residue = blockIdx.x;
  const unsigned lane = threadIdx.x;
  const unsigned rev = __brev(lane) >> 26;
  float2 values[registers];
  #pragma unroll
  for (unsigned i = 0; i < registers; ++i) {
    values[i] = load_inverse_residue(input, roots, rev + wave * i, residue);
  }
  wave_fft(values, roots, lane, rev);
  #pragma unroll
  for (unsigned i = 0; i < registers; ++i) {
    store_inverse(output, (lane * registers + i) * residues + residue, values[i]);
  }
}

}  // namespace

extern "C" __attribute__((visibility("hidden"))) int flagfft_maca_launch_fp32_r2c2048(
    const float* input, float* output, const void* phase, unsigned batch,
    std::int64_t input_distance, std::int64_t output_distance, bool use_residues, void* stream) {
  // The independent residue blocks share the input. Exact in-place requests
  // use the shared-memory schedule, which consumes all samples before storing.
  auto maca_stream = reinterpret_cast<mcStream_t>(stream);
  if (use_residues && input != output) {
    void* arguments[] = {&input, &output, &phase};
    return mcLaunchKernel(reinterpret_cast<const void*>(residue_fft), dim3(residues), dim3(2 * wave),
                          arguments, 0, maca_stream);
  }
  const auto kernel = batch == 1 ? shared_fft<false> : shared_fft<true>;
  void* arguments[] = {&input, &output, &phase, &input_distance, &output_distance};
  return mcLaunchKernel(reinterpret_cast<const void*>(kernel), dim3(batch), dim3(4 * wave),
                        arguments, 0, maca_stream);
}

extern "C" __attribute__((visibility("hidden"))) int flagfft_maca_launch_fp32_c2r2048(
    const float* input, float* output, const void* roots, void* stream) {
  const auto source = reinterpret_cast<std::uintptr_t>(input);
  const auto destination = reinterpret_cast<std::uintptr_t>(output);
  const bool overlap = destination >= source ? destination - source < (length + 2) * sizeof(float)
                                             : source - destination < length * sizeof(float);
  // The shared kernel reads the full spectrum before writing aliased output.
  const auto kernel = overlap ? inverse_shared_fft : inverse_residue_fft;
  void* arguments[] = {&input, &output, &roots};
  return mcLaunchKernel(reinterpret_cast<const void*>(kernel), dim3(overlap ? 1 : residues),
                        dim3(overlap ? 4 * wave : wave), arguments, 0,
                        reinterpret_cast<mcStream_t>(stream));
}
