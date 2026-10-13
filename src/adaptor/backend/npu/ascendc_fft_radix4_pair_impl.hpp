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

#pragma once

#include "kernel_operator.h"

using namespace AscendC;

namespace {

constexpr uint32_t fft_radix4_pair_log2(uint32_t value) {
  return value <= 1 ? 0 : 1 + fft_radix4_pair_log2(value / 2);
}

template <uint32_t N, uint32_t GroupSize>
class FftRadix4PairAiv {
 public:
  static constexpr uint32_t kStages = fft_radix4_pair_log2(N);
  static constexpr uint32_t kPairCount = kStages / 2;
  static constexpr uint32_t kGroupN = N * GroupSize;
  static constexpr uint32_t kQuarter = kGroupN / 4;

  __aicore__ inline void Init(
      GM_ADDR input, GM_ADDR output, GM_ADDR indices, GM_ADDR twiddles, uint32_t block_count, uint32_t mode) {
    input_ptr_ = reinterpret_cast<__gm__ float *>(input);
    output_ptr_ = reinterpret_cast<__gm__ float *>(output);
    indices_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(indices));
    twiddles_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(twiddles));
    block_count_ = block_count;
    mode_ = mode;

    pipe_.InitBuffer(input_buf_, 2 * kGroupN * sizeof(float));
    pipe_.InitBuffer(work_buf_, 7 * kGroupN * sizeof(float));
    pipe_.InitBuffer(index_buf_, 3 * kGroupN * sizeof(uint32_t));
    pipe_.InitBuffer(twiddle_buf_, 2 * kGroupN * sizeof(float));
  }

  __aicore__ inline void Process() {
    const uint32_t block = GetBlockIdx();
    if (block >= block_count_) return;

    const uint32_t transform = block * GroupSize;
    const uint32_t half_spectrum = N / 2 + 1;
    const uint32_t input_count = mode_ == kModeRealForward   ? kGroupN
                                 : mode_ == kModeRealInverse ? 2 * half_spectrum * GroupSize
                                                             : 2 * kGroupN;
    GlobalTensor<float> src;
    LocalTensor<float> input_local = input_buf_.Get<float>();
    const uint32_t input_base = mode_ == kModeRealForward   ? transform * N
                                : mode_ == kModeRealInverse ? transform * half_spectrum * 2
                                                            : transform * N * 2;
    src.SetGlobalBuffer(input_ptr_ + input_base);
    DataCopy(input_local, src, input_count);

    LocalTensor<float> work = work_buf_.Get<float>();
    LocalTensor<float> current_real = work[0 * kGroupN];
    LocalTensor<float> current_imag = work[1 * kGroupN];
    LocalTensor<float> next_real = work[2 * kGroupN];
    LocalTensor<float> next_imag = work[3 * kGroupN];
    LocalTensor<float> x0_real = work[4 * kGroupN + 0 * kQuarter];
    LocalTensor<float> x0_imag = work[4 * kGroupN + 1 * kQuarter];
    LocalTensor<float> x1_real = work[4 * kGroupN + 2 * kQuarter];
    LocalTensor<float> x1_imag = work[4 * kGroupN + 3 * kQuarter];
    LocalTensor<float> x2_real = work[4 * kGroupN + 4 * kQuarter];
    LocalTensor<float> x2_imag = work[4 * kGroupN + 5 * kQuarter];
    LocalTensor<float> x3_real = work[4 * kGroupN + 6 * kQuarter];
    LocalTensor<float> x3_imag = work[4 * kGroupN + 7 * kQuarter];
    LocalTensor<float> product_real = work[6 * kGroupN];
    LocalTensor<float> product_imag = work[6 * kGroupN + kQuarter];
    LocalTensor<float> product_temp = work[6 * kGroupN + 2 * kQuarter];

    constexpr uint32_t pair_index_count = kPairCount * 4 * kQuarter;
    constexpr uint32_t output_index_base = pair_index_count;
    constexpr uint32_t leftover_index_base = output_index_base + 2 * kGroupN;
    constexpr uint32_t expand_index_base = leftover_index_base + 2 * kGroupN;
    constexpr uint32_t pair_twiddle_count = kPairCount * 6 * kQuarter;
    constexpr uint32_t leftover_twiddle_base = pair_twiddle_count;
    constexpr uint32_t expand_sign_base = leftover_twiddle_base + 2 * kGroupN;

    LocalTensor<uint32_t> index_local = index_buf_.Get<uint32_t>();
    LocalTensor<float> twiddle_local = twiddle_buf_.Get<float>();

    if (mode_ == kModeRealInverse) {
      // Expand the Hermitian half spectrum into planar local vectors. The
      // first radix-4 pair then consumes the same bit-reversed layout as C2C.
      DataCopy(index_local, indices_[expand_index_base], 3 * kGroupN);
      DataCopy(twiddle_local, twiddles_[expand_sign_base], kGroupN);
      PipeBarrier<PIPE_ALL>();
      Gather(current_real, input_local, index_local, 0, kGroupN);
      Gather(current_imag, input_local, index_local[kGroupN], 0, kGroupN);
      Mul(current_imag, current_imag, twiddle_local, kGroupN);
      PipeBarrier<PIPE_ALL>();
    }

    for (uint32_t pair = 0; pair < kPairCount; ++pair) {
      const uint32_t pair_index_base = pair * 4 * kQuarter;
      const uint32_t pair_twiddle_base = pair * 6 * kQuarter;
      DataCopy(index_local, indices_[pair_index_base], 4 * kQuarter);
      DataCopy(twiddle_local, twiddles_[pair_twiddle_base], 6 * kQuarter);
      PipeBarrier<PIPE_ALL>();

      const LocalTensor<uint32_t> index0 = index_local[0 * kQuarter];
      const LocalTensor<uint32_t> index1 = index_local[1 * kQuarter];
      const LocalTensor<uint32_t> index2 = index_local[2 * kQuarter];
      const LocalTensor<uint32_t> index3 = index_local[3 * kQuarter];

      if (pair == 0 && mode_ != kModeRealInverse) {
        Gather(x0_real, input_local, index0, 0, kQuarter);
        Gather(x1_real, input_local, index1, 0, kQuarter);
        Gather(x2_real, input_local, index2, 0, kQuarter);
        Gather(x3_real, input_local, index3, 0, kQuarter);
        if (mode_ == kModeRealForward) {
          Duplicate(x0_imag, 0.0f, kQuarter);
          Duplicate(x1_imag, 0.0f, kQuarter);
          Duplicate(x2_imag, 0.0f, kQuarter);
          Duplicate(x3_imag, 0.0f, kQuarter);
        } else {
          Gather(x0_imag, input_local, index0, sizeof(float), kQuarter);
          Gather(x1_imag, input_local, index1, sizeof(float), kQuarter);
          Gather(x2_imag, input_local, index2, sizeof(float), kQuarter);
          Gather(x3_imag, input_local, index3, sizeof(float), kQuarter);
        }
      } else {
        Gather(x0_real, current_real, index0, 0, kQuarter);
        Gather(x0_imag, current_imag, index0, 0, kQuarter);
        Gather(x1_real, current_real, index1, 0, kQuarter);
        Gather(x1_imag, current_imag, index1, 0, kQuarter);
        Gather(x2_real, current_real, index2, 0, kQuarter);
        Gather(x2_imag, current_imag, index2, 0, kQuarter);
        Gather(x3_real, current_real, index3, 0, kQuarter);
        Gather(x3_imag, current_imag, index3, 0, kQuarter);
      }

      const LocalTensor<float> twiddle1_real = twiddle_local[0 * kQuarter];
      const LocalTensor<float> twiddle1_imag = twiddle_local[1 * kQuarter];
      const LocalTensor<float> twiddle2lo_real = twiddle_local[2 * kQuarter];
      const LocalTensor<float> twiddle2lo_imag = twiddle_local[3 * kQuarter];
      const LocalTensor<float> twiddle2hi_real = twiddle_local[4 * kQuarter];
      const LocalTensor<float> twiddle2hi_imag = twiddle_local[5 * kQuarter];

      ComplexMultiply(product_real,
                      product_imag,
                      product_temp,
                      x1_real,
                      x1_imag,
                      twiddle1_real,
                      twiddle1_imag);
      Sub(x1_real, x0_real, product_real, kQuarter);
      Sub(x1_imag, x0_imag, product_imag, kQuarter);
      Add(x0_real, x0_real, product_real, kQuarter);
      Add(x0_imag, x0_imag, product_imag, kQuarter);

      ComplexMultiply(product_real,
                      product_imag,
                      product_temp,
                      x3_real,
                      x3_imag,
                      twiddle1_real,
                      twiddle1_imag);
      Sub(x3_real, x2_real, product_real, kQuarter);
      Sub(x3_imag, x2_imag, product_imag, kQuarter);
      Add(x2_real, x2_real, product_real, kQuarter);
      Add(x2_imag, x2_imag, product_imag, kQuarter);

      ComplexMultiply(product_real,
                      product_imag,
                      product_temp,
                      x2_real,
                      x2_imag,
                      twiddle2lo_real,
                      twiddle2lo_imag);
      Add(next_real[0 * kQuarter], x0_real, product_real, kQuarter);
      Add(next_imag[0 * kQuarter], x0_imag, product_imag, kQuarter);
      Sub(next_real[2 * kQuarter], x0_real, product_real, kQuarter);
      Sub(next_imag[2 * kQuarter], x0_imag, product_imag, kQuarter);

      ComplexMultiply(product_real,
                      product_imag,
                      product_temp,
                      x3_real,
                      x3_imag,
                      twiddle2hi_real,
                      twiddle2hi_imag);
      Add(next_real[1 * kQuarter], x1_real, product_real, kQuarter);
      Add(next_imag[1 * kQuarter], x1_imag, product_imag, kQuarter);
      Sub(next_real[3 * kQuarter], x1_real, product_real, kQuarter);
      Sub(next_imag[3 * kQuarter], x1_imag, product_imag, kQuarter);
      PipeBarrier<PIPE_ALL>();

      LocalTensor<float> swap = current_real;
      current_real = next_real;
      next_real = swap;
      swap = current_imag;
      current_imag = next_imag;
      next_imag = swap;
    }

    if constexpr ((kStages & 1) != 0) {
      DataCopy(index_local, indices_[leftover_index_base], 2 * kGroupN);
      DataCopy(twiddle_local, twiddles_[leftover_twiddle_base], 2 * kGroupN);
      PipeBarrier<PIPE_ALL>();
      const LocalTensor<uint32_t> stage_a = index_local;
      const LocalTensor<uint32_t> stage_b = index_local[kGroupN];
      for (uint32_t tile = 0; tile < kGroupN; tile += kQuarter) {
        Gather(x0_real, current_real, stage_a[tile], 0, kQuarter);
        Gather(x0_imag, current_imag, stage_a[tile], 0, kQuarter);
        Gather(x1_real, current_real, stage_b[tile], 0, kQuarter);
        Gather(x1_imag, current_imag, stage_b[tile], 0, kQuarter);
        ComplexMultiply(product_real,
                        product_imag,
                        product_temp,
                        x1_real,
                        x1_imag,
                        twiddle_local[tile],
                        twiddle_local[kGroupN + tile]);
        Add(next_real[tile], x0_real, product_real, kQuarter);
        Add(next_imag[tile], x0_imag, product_imag, kQuarter);
      }
      PipeBarrier<PIPE_ALL>();
      LocalTensor<float> swap = current_real;
      current_real = next_real;
      next_real = swap;
      swap = current_imag;
      current_imag = next_imag;
      next_imag = swap;
    }

    // Reuse the butterfly scratch space for the gathered output to keep UB
    // usage low enough for multiple resident AIV blocks on 910B.
    LocalTensor<float> output_local = work[4 * kGroupN];
    uint32_t output_count = 2 * kGroupN;
    if (mode_ == kModeRealForward) output_count = 2 * half_spectrum * GroupSize;
    DataCopy(index_local, indices_[output_index_base], 2 * kGroupN);
    PipeBarrier<PIPE_ALL>();
    if (mode_ == kModeRealInverse) {
      output_count = kGroupN;
      Gather(output_local, current_real, index_local, 0, output_count);
    } else {
      DataCopy(input_local, current_real, kGroupN);
      DataCopy(input_local[kGroupN], current_imag, kGroupN);
      PipeBarrier<PIPE_ALL>();
      Gather(output_local, input_local, index_local, 0, output_count);
    }
    PipeBarrier<PIPE_ALL>();

    GlobalTensor<float> dst;
    const uint32_t output_base = mode_ == kModeRealForward   ? transform * half_spectrum * 2
                                 : mode_ == kModeRealInverse ? transform * N
                                                             : transform * N * 2;
    dst.SetGlobalBuffer(output_ptr_ + output_base);
    DataCopy(dst, output_local, output_count);
  }

 private:
  __aicore__ inline void ComplexMultiply(LocalTensor<float> &out_real,
                                         LocalTensor<float> &out_imag,
                                         LocalTensor<float> &temporary,
                                         const LocalTensor<float> &in_real,
                                         const LocalTensor<float> &in_imag,
                                         const LocalTensor<float> &twiddle_real,
                                         const LocalTensor<float> &twiddle_imag,
                                         uint32_t count = kQuarter) {
    Mul(out_real, in_real, twiddle_real, count);
    Mul(temporary, in_imag, twiddle_imag, count);
    Sub(out_real, out_real, temporary, count);
    Mul(out_imag, in_real, twiddle_imag, count);
    Mul(temporary, in_imag, twiddle_real, count);
    Add(out_imag, out_imag, temporary, count);
  }

  static constexpr uint32_t kModeComplex = 0;
  static constexpr uint32_t kModeRealForward = 1;
  static constexpr uint32_t kModeRealInverse = 2;

  TPipe pipe_;
  TBuf<TPosition::VECIN> input_buf_;
  TBuf<TPosition::VECCALC> work_buf_;
  TBuf<TPosition::VECCALC> index_buf_;
  TBuf<TPosition::VECCALC> twiddle_buf_;
  GlobalTensor<uint32_t> indices_;
  GlobalTensor<float> twiddles_;
  __gm__ float *input_ptr_ = nullptr;
  __gm__ float *output_ptr_ = nullptr;
  uint32_t block_count_ = 0;
  uint32_t mode_ = 0;
};

}  // namespace
