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
constexpr uint32_t kRadix4N = 256;
constexpr uint32_t kRadix4GroupSize = 8;
constexpr uint32_t kRadix4GroupN = kRadix4N * kRadix4GroupSize;
constexpr uint32_t kRadix4Quarter = kRadix4GroupN / 4;
constexpr uint32_t kRadix4PairCount = 4;
constexpr uint32_t kRadix4IndexCount = 4 * kRadix4PairCount * kRadix4Quarter + 2 * kRadix4GroupN;
constexpr uint32_t kRadix4TwiddleCount = 6 * kRadix4PairCount * kRadix4Quarter;

class Fft256AivPairRadix4Group8 {
 public:
  __aicore__ inline void Init(GM_ADDR input,
                              GM_ADDR output,
                              GM_ADDR indices,
                              GM_ADDR twiddles,
                              uint32_t transform_count,
                              uint32_t output_row_stride,
                              uint32_t output_transform_offset) {
    input_ptr_ = reinterpret_cast<__gm__ float *>(input);
    output_ptr_ = reinterpret_cast<__gm__ float *>(output);
    indices_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(indices));
    twiddles_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(twiddles));
    transform_count_ = transform_count;
    output_row_stride_ = output_row_stride;
    output_transform_offset_ = output_transform_offset;

    pipe_.InitBuffer(input_buf_, 2 * kRadix4GroupN * sizeof(float));
    pipe_.InitBuffer(work_buf_, 7 * kRadix4GroupN * sizeof(float));
    pipe_.InitBuffer(index_buf_, 3 * kRadix4GroupN * sizeof(uint32_t));
    pipe_.InitBuffer(twiddle_buf_, 6 * kRadix4Quarter * sizeof(float));
    pipe_.InitBuffer(output_buf_, 2 * kRadix4GroupN * sizeof(float));
  }

  __aicore__ inline void Process() {
    const uint32_t transform = GetBlockIdx() * kRadix4GroupSize;
    if (transform >= transform_count_) return;

    GlobalTensor<float> src;
    src.SetGlobalBuffer(input_ptr_ + transform * 2 * kRadix4N);
    LocalTensor<float> packed = input_buf_.Get<float>();
    DataCopy(packed, src, 2 * kRadix4GroupN);

    LocalTensor<float> work = work_buf_.Get<float>();
    LocalTensor<float> current_real = work[0 * kRadix4GroupN];
    LocalTensor<float> current_imag = work[1 * kRadix4GroupN];
    LocalTensor<float> next_real = work[2 * kRadix4GroupN];
    LocalTensor<float> next_imag = work[3 * kRadix4GroupN];

    LocalTensor<float> x0_real = work[4 * kRadix4GroupN + 0 * kRadix4Quarter];
    LocalTensor<float> x0_imag = work[4 * kRadix4GroupN + 1 * kRadix4Quarter];
    LocalTensor<float> x1_real = work[4 * kRadix4GroupN + 2 * kRadix4Quarter];
    LocalTensor<float> x1_imag = work[4 * kRadix4GroupN + 3 * kRadix4Quarter];
    LocalTensor<float> x2_real = work[4 * kRadix4GroupN + 4 * kRadix4Quarter];
    LocalTensor<float> x2_imag = work[4 * kRadix4GroupN + 5 * kRadix4Quarter];
    LocalTensor<float> x3_real = work[4 * kRadix4GroupN + 6 * kRadix4Quarter];
    LocalTensor<float> x3_imag = work[4 * kRadix4GroupN + 7 * kRadix4Quarter];

    LocalTensor<float> product_real = work[6 * kRadix4GroupN];
    LocalTensor<float> product_imag = work[6 * kRadix4GroupN + kRadix4Quarter];
    LocalTensor<float> product_temp = work[6 * kRadix4GroupN + 2 * kRadix4Quarter];

    LocalTensor<uint32_t> index_local = index_buf_.Get<uint32_t>();
    DataCopy(index_local[kRadix4GroupN], indices_[4 * kRadix4GroupN], 2 * kRadix4GroupN);
    PipeBarrier<PIPE_ALL>();

    LocalTensor<float> twiddle_local = twiddle_buf_.Get<float>();
    for (uint32_t pair = 0; pair < kRadix4PairCount; ++pair) {
      const uint32_t pair_index_base = pair * 4 * kRadix4Quarter;
      const uint32_t pair_twiddle_base = pair * 6 * kRadix4Quarter;
      DataCopy(index_local, indices_[pair_index_base], 4 * kRadix4Quarter);
      DataCopy(twiddle_local, twiddles_[pair_twiddle_base], 6 * kRadix4Quarter);
      PipeBarrier<PIPE_ALL>();

      const LocalTensor<uint32_t> index0 = index_local[0 * kRadix4Quarter];
      const LocalTensor<uint32_t> index1 = index_local[1 * kRadix4Quarter];
      const LocalTensor<uint32_t> index2 = index_local[2 * kRadix4Quarter];
      const LocalTensor<uint32_t> index3 = index_local[3 * kRadix4Quarter];

      if (pair == 0) {
        Gather(x0_real, packed, index0, 0, kRadix4Quarter);
        Gather(x0_imag, packed, index0, sizeof(float), kRadix4Quarter);
        Gather(x1_real, packed, index1, 0, kRadix4Quarter);
        Gather(x1_imag, packed, index1, sizeof(float), kRadix4Quarter);
        Gather(x2_real, packed, index2, 0, kRadix4Quarter);
        Gather(x2_imag, packed, index2, sizeof(float), kRadix4Quarter);
        Gather(x3_real, packed, index3, 0, kRadix4Quarter);
        Gather(x3_imag, packed, index3, sizeof(float), kRadix4Quarter);
      } else {
        Gather(x0_real, current_real, index0, 0, kRadix4Quarter);
        Gather(x0_imag, current_imag, index0, 0, kRadix4Quarter);
        Gather(x1_real, current_real, index1, 0, kRadix4Quarter);
        Gather(x1_imag, current_imag, index1, 0, kRadix4Quarter);
        Gather(x2_real, current_real, index2, 0, kRadix4Quarter);
        Gather(x2_imag, current_imag, index2, 0, kRadix4Quarter);
        Gather(x3_real, current_real, index3, 0, kRadix4Quarter);
        Gather(x3_imag, current_imag, index3, 0, kRadix4Quarter);
      }

      const LocalTensor<float> twiddle1_real = twiddle_local[0 * kRadix4Quarter];
      const LocalTensor<float> twiddle1_imag = twiddle_local[1 * kRadix4Quarter];
      const LocalTensor<float> twiddle2lo_real = twiddle_local[2 * kRadix4Quarter];
      const LocalTensor<float> twiddle2lo_imag = twiddle_local[3 * kRadix4Quarter];
      const LocalTensor<float> twiddle2hi_real = twiddle_local[4 * kRadix4Quarter];
      const LocalTensor<float> twiddle2hi_imag = twiddle_local[5 * kRadix4Quarter];

      ComplexMultiply(product_real,
                      product_imag,
                      product_temp,
                      x1_real,
                      x1_imag,
                      twiddle1_real,
                      twiddle1_imag);
      Sub(x1_real, x0_real, product_real, kRadix4Quarter);
      Sub(x1_imag, x0_imag, product_imag, kRadix4Quarter);
      Add(x0_real, x0_real, product_real, kRadix4Quarter);
      Add(x0_imag, x0_imag, product_imag, kRadix4Quarter);

      ComplexMultiply(product_real,
                      product_imag,
                      product_temp,
                      x3_real,
                      x3_imag,
                      twiddle1_real,
                      twiddle1_imag);
      Sub(x3_real, x2_real, product_real, kRadix4Quarter);
      Sub(x3_imag, x2_imag, product_imag, kRadix4Quarter);
      Add(x2_real, x2_real, product_real, kRadix4Quarter);
      Add(x2_imag, x2_imag, product_imag, kRadix4Quarter);

      ComplexMultiply(product_real,
                      product_imag,
                      product_temp,
                      x2_real,
                      x2_imag,
                      twiddle2lo_real,
                      twiddle2lo_imag);
      Add(next_real[0 * kRadix4Quarter], x0_real, product_real, kRadix4Quarter);
      Add(next_imag[0 * kRadix4Quarter], x0_imag, product_imag, kRadix4Quarter);
      Sub(next_real[2 * kRadix4Quarter], x0_real, product_real, kRadix4Quarter);
      Sub(next_imag[2 * kRadix4Quarter], x0_imag, product_imag, kRadix4Quarter);

      ComplexMultiply(product_real,
                      product_imag,
                      product_temp,
                      x3_real,
                      x3_imag,
                      twiddle2hi_real,
                      twiddle2hi_imag);
      Add(next_real[1 * kRadix4Quarter], x1_real, product_real, kRadix4Quarter);
      Add(next_imag[1 * kRadix4Quarter], x1_imag, product_imag, kRadix4Quarter);
      Sub(next_real[3 * kRadix4Quarter], x1_real, product_real, kRadix4Quarter);
      Sub(next_imag[3 * kRadix4Quarter], x1_imag, product_imag, kRadix4Quarter);
      PipeBarrier<PIPE_ALL>();

      LocalTensor<float> swap = current_real;
      current_real = next_real;
      next_real = swap;
      swap = current_imag;
      current_imag = next_imag;
      next_imag = swap;
    }

    DataCopy(packed, current_real, kRadix4GroupN);
    DataCopy(packed[kRadix4GroupN], current_imag, kRadix4GroupN);
    PipeBarrier<PIPE_ALL>();
    LocalTensor<float> output_local = output_buf_.Get<float>();
    Gather(output_local, packed, index_local[kRadix4GroupN], 0, 2 * kRadix4GroupN);
    PipeBarrier<PIPE_ALL>();

    LocalTensor<uint64_t> output_complex = output_local.ReinterpretCast<uint64_t>();
    constexpr uint32_t kOutputRowsPerPair = kRadix4N / 4;
    const DataCopyExtParams output_params(kOutputRowsPerPair,
                                          kRadix4GroupSize * sizeof(uint64_t),
                                          0,
                                          (output_row_stride_ - kRadix4GroupSize) * sizeof(uint64_t),
                                          0);
    for (uint32_t slot = 0; slot < 4; ++slot) {
      GlobalTensor<uint64_t> dst;
      dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(output_ptr_) + output_transform_offset_ +
                          transform + slot * kOutputRowsPerPair * output_row_stride_);
      DataCopyPad(dst, output_complex[slot * kRadix4Quarter], output_params);
    }
  }

 private:
  __aicore__ inline void ComplexMultiply(LocalTensor<float> &out_real,
                                         LocalTensor<float> &out_imag,
                                         LocalTensor<float> &temporary,
                                         const LocalTensor<float> &in_real,
                                         const LocalTensor<float> &in_imag,
                                         const LocalTensor<float> &twiddle_real,
                                         const LocalTensor<float> &twiddle_imag) {
    Mul(out_real, in_real, twiddle_real, kRadix4Quarter);
    Mul(temporary, in_imag, twiddle_imag, kRadix4Quarter);
    Sub(out_real, out_real, temporary, kRadix4Quarter);
    Mul(out_imag, in_real, twiddle_imag, kRadix4Quarter);
    Mul(temporary, in_imag, twiddle_real, kRadix4Quarter);
    Add(out_imag, out_imag, temporary, kRadix4Quarter);
  }

  TPipe pipe_;
  TBuf<TPosition::VECIN> input_buf_;
  TBuf<TPosition::VECCALC> work_buf_;
  TBuf<TPosition::VECCALC> index_buf_;
  TBuf<TPosition::VECCALC> twiddle_buf_;
  TBuf<TPosition::VECOUT> output_buf_;
  __gm__ float *input_ptr_ = nullptr;
  __gm__ float *output_ptr_ = nullptr;
  GlobalTensor<uint32_t> indices_;
  GlobalTensor<float> twiddles_;
  uint32_t transform_count_ = 0;
  uint32_t output_row_stride_ = 0;
  uint32_t output_transform_offset_ = 0;
};
}  // namespace
