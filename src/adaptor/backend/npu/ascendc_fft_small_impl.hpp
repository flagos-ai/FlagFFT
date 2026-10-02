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

template <uint32_t N, uint32_t Stages, uint32_t GroupSize>
class FftSmallAiv {
 public:
  __aicore__ inline void Init(GM_ADDR input,
                              GM_ADDR output,
                              GM_ADDR indices,
                              GM_ADDR twiddles,
                              uint32_t block_count,
                              uint32_t stride) {
    input_ptr_ = reinterpret_cast<__gm__ float *>(input);
    output_ptr_ = reinterpret_cast<__gm__ float *>(output);
    indices_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(indices));
    twiddles_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(twiddles));
    block_count_ = block_count;
    stride_ = stride;

    constexpr uint32_t kGroupN = N * GroupSize;
    constexpr uint32_t kIndexCount = (3 + 2 * Stages) * kGroupN;
    constexpr uint32_t kTwiddleCount = 2 * Stages * kGroupN;
    pipe_.InitBuffer(input_buf_, 2 * kGroupN * sizeof(float));
    pipe_.InitBuffer(work_buf_, 11 * kGroupN * sizeof(float));
    pipe_.InitBuffer(index_buf_, kIndexCount * sizeof(uint32_t));
    pipe_.InitBuffer(twiddle_buf_, kTwiddleCount * sizeof(float));
    pipe_.InitBuffer(output_buf_, 2 * kGroupN * sizeof(float));
    pipe_.InitBuffer(merge_buf_, 2 * kGroupN * sizeof(float));
  }

  __aicore__ inline void Process() {
    constexpr uint32_t kGroupN = N * GroupSize;
    constexpr uint32_t kIndexCount = (3 + 2 * Stages) * kGroupN;
    constexpr uint32_t kTwiddleCount = 2 * Stages * kGroupN;
    const uint32_t block = GetBlockIdx();
    if (block >= block_count_ ||
        (stride_ != 1 && (stride_ < GroupSize || stride_ % GroupSize != 0))) return;

    const uint32_t transform_base = block * GroupSize;
    const uint32_t source_base = stride_ == 1
        ? transform_base * N
        : (transform_base / stride_) * N * stride_ + transform_base % stride_;
    LocalTensor<float> input_local = input_buf_.Get<float>();
    if (stride_ == 1) {
      GlobalTensor<float> src;
      src.SetGlobalBuffer(input_ptr_ + transform_base * N * 2);
      DataCopy(input_local, src, 2 * kGroupN);
    } else {
      GlobalTensor<uint64_t> src;
      src.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(input_ptr_ + source_base * 2));
      LocalTensor<uint64_t> input_complex = input_local.ReinterpretCast<uint64_t>();
      const DataCopyExtParams params(
          N, GroupSize * sizeof(uint64_t), (stride_ - GroupSize) * sizeof(uint64_t), 0, 0);
      const DataCopyPadExtParams<uint64_t> pad;
      DataCopyPad(input_complex, src, params, pad);
    }

    LocalTensor<uint32_t> index_local = index_buf_.Get<uint32_t>();
    LocalTensor<float> twiddle_local = twiddle_buf_.Get<float>();
    DataCopy(index_local, indices_, kIndexCount);
    DataCopy(twiddle_local, twiddles_, kTwiddleCount);
    PipeBarrier<PIPE_ALL>();

    LocalTensor<float> work = work_buf_.Get<float>();
    LocalTensor<float> current_real = work[0 * kGroupN];
    LocalTensor<float> current_imag = work[1 * kGroupN];
    LocalTensor<float> next_real = work[2 * kGroupN];
    LocalTensor<float> next_imag = work[3 * kGroupN];
    LocalTensor<float> a_real = work[4 * kGroupN];
    LocalTensor<float> a_imag = work[5 * kGroupN];
    LocalTensor<float> b_real = work[6 * kGroupN];
    LocalTensor<float> b_imag = work[7 * kGroupN];
    LocalTensor<float> product0 = work[8 * kGroupN];
    LocalTensor<float> product1 = work[9 * kGroupN];
    LocalTensor<float> product2 = work[10 * kGroupN];

    Gather(current_real, input_local, index_local, 0, kGroupN);
    Gather(current_imag, input_local, index_local, sizeof(float), kGroupN);

    constexpr uint32_t kStageA = 3 * kGroupN;
    constexpr uint32_t kStageB = (3 + Stages) * kGroupN;
    for (uint32_t stage = 0; stage < Stages; ++stage) {
      const LocalTensor<uint32_t> stage_a = index_local[kStageA + stage * kGroupN];
      const LocalTensor<uint32_t> stage_b = index_local[kStageB + stage * kGroupN];
      const LocalTensor<float> twiddle_real = twiddle_local[stage * kGroupN];
      const LocalTensor<float> twiddle_imag = twiddle_local[(Stages + stage) * kGroupN];

      Gather(a_real, current_real, stage_a, 0, kGroupN);
      Gather(a_imag, current_imag, stage_a, 0, kGroupN);
      Gather(b_real, current_real, stage_b, 0, kGroupN);
      Gather(b_imag, current_imag, stage_b, 0, kGroupN);
      Mul(product0, b_real, twiddle_real, kGroupN);
      Mul(product1, b_imag, twiddle_imag, kGroupN);
      Sub(product0, product0, product1, kGroupN);
      Mul(product1, b_real, twiddle_imag, kGroupN);
      Mul(product2, b_imag, twiddle_real, kGroupN);
      Add(product1, product1, product2, kGroupN);
      Add(next_real, a_real, product0, kGroupN);
      Add(next_imag, a_imag, product1, kGroupN);

      LocalTensor<float> swap = current_real;
      current_real = next_real;
      next_real = swap;
      swap = current_imag;
      current_imag = next_imag;
      next_imag = swap;
    }

    LocalTensor<float> merged = merge_buf_.Get<float>();
    DataCopy(merged, current_real, kGroupN);
    DataCopy(merged[kGroupN], current_imag, kGroupN);
    PipeBarrier<PIPE_ALL>();
    LocalTensor<float> output_local = output_buf_.Get<float>();
    Gather(output_local, merged, index_local[kGroupN], 0, 2 * kGroupN);
    PipeBarrier<PIPE_ALL>();

    if (stride_ == 1) {
      GlobalTensor<float> dst;
      dst.SetGlobalBuffer(output_ptr_ + transform_base * N * 2);
      DataCopy(dst, output_local, 2 * kGroupN);
    } else {
      GlobalTensor<uint64_t> dst;
      dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(output_ptr_ + source_base * 2));
      LocalTensor<uint64_t> output_complex = output_local.ReinterpretCast<uint64_t>();
      const DataCopyExtParams params(
          N, GroupSize * sizeof(uint64_t), 0, (stride_ - GroupSize) * sizeof(uint64_t), 0);
      DataCopyPad(dst, output_complex, params);
    }
  }

 private:
  TPipe pipe_;
  TBuf<QuePosition::VECCALC> input_buf_;
  TBuf<QuePosition::VECCALC> work_buf_;
  TBuf<QuePosition::VECCALC> index_buf_;
  TBuf<QuePosition::VECCALC> twiddle_buf_;
  TBuf<QuePosition::VECCALC> output_buf_;
  TBuf<QuePosition::VECCALC> merge_buf_;
  __gm__ float *input_ptr_ = nullptr;
  __gm__ float *output_ptr_ = nullptr;
  GlobalTensor<uint32_t> indices_;
  GlobalTensor<float> twiddles_;
  uint32_t block_count_ = 0;
  uint32_t stride_ = 1;
};
