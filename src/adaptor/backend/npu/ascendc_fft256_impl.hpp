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
constexpr uint32_t kN = 256;
constexpr uint32_t kStages = 8;
constexpr uint32_t kStageIndexBase = 2 * kN;
constexpr uint32_t kOutputIndexBase = kStageIndexBase + 2 * kStages * kN;
constexpr uint32_t kOutputIndexCount = 2 * kN;
constexpr uint32_t kLocalIndexCount = kOutputIndexBase + kOutputIndexCount;
constexpr uint32_t kTwiddleCount = 2 * kStages * kN;

template <uint32_t GroupSize>
class Fft256Aiv {
 public:
  __aicore__ inline void Init(GM_ADDR input,
                              GM_ADDR output,
                              GM_ADDR indices,
                              GM_ADDR twiddles,
                              uint32_t transform_count) {
    input_ptr_ = reinterpret_cast<__gm__ float *>(input);
    output_ptr_ = reinterpret_cast<__gm__ float *>(output);
    indices_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(indices));
    twiddles_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(twiddles));
    transform_count_ = transform_count;

    constexpr uint32_t group_n = kN * GroupSize;
    constexpr uint32_t index_count = GroupSize == 1 ? kLocalIndexCount : 5 * group_n;
    constexpr uint32_t twiddle_count = GroupSize == 1 ? kTwiddleCount : 2 * group_n;
    pipe_.InitBuffer(input_buf_, 2 * group_n * sizeof(float));
    pipe_.InitBuffer(work_buf_, 11 * group_n * sizeof(float));
    pipe_.InitBuffer(index_buf_, index_count * sizeof(uint32_t));
    pipe_.InitBuffer(twiddle_buf_, twiddle_count * sizeof(float));
    pipe_.InitBuffer(output_buf_, 2 * group_n * sizeof(float));
  }

  __aicore__ inline void Process() {
    const uint32_t transform = GetBlockIdx() * GroupSize;
    if (transform >= transform_count_) return;

    if constexpr (GroupSize == 4) {
      ProcessGroup4(transform);
      return;
    }

    GlobalTensor<float> src;
    src.SetGlobalBuffer(input_ptr_ + transform * 2 * kN);
    LocalTensor<float> input_local = input_buf_.Get<float>();
    DataCopy(input_local, src, 2 * kN);

    LocalTensor<uint32_t> index_local = index_buf_.Get<uint32_t>();
    DataCopy(index_local, indices_, 2 * kN);
    DataCopy(index_local[kStageIndexBase],
             indices_[kStageIndexBase],
             2 * kStages * kN);
    DataCopy(index_local[kOutputIndexBase],
             indices_[kOutputIndexBase],
             kOutputIndexCount);
    LocalTensor<float> twiddle_local = twiddle_buf_.Get<float>();
    DataCopy(twiddle_local, twiddles_, kTwiddleCount);
    PipeBarrier<PIPE_ALL>();

    LocalTensor<float> work = work_buf_.Get<float>();
    LocalTensor<float> current_real = work[0 * kN];
    LocalTensor<float> current_imag = work[1 * kN];
    LocalTensor<float> next_real = work[2 * kN];
    LocalTensor<float> next_imag = work[3 * kN];
    LocalTensor<float> a_real = work[4 * kN];
    LocalTensor<float> a_imag = work[5 * kN];
    LocalTensor<float> b_real = work[6 * kN];
    LocalTensor<float> b_imag = work[7 * kN];
    LocalTensor<float> product0 = work[8 * kN];
    LocalTensor<float> product1 = work[9 * kN];
    LocalTensor<float> product2 = work[10 * kN];

    Gather(current_real, input_local, index_local, 0, kN);
    Gather(current_imag, input_local, index_local, sizeof(float), kN);

    for (uint32_t stage = 0; stage < kStages; ++stage) {
      const LocalTensor<uint32_t> stage_a =
          index_local[kStageIndexBase + stage * kN];
      const LocalTensor<uint32_t> stage_b =
          index_local[kStageIndexBase + kStages * kN + stage * kN];
      const LocalTensor<float> twiddle_real = twiddle_local[stage * kN];
      const LocalTensor<float> twiddle_imag =
          twiddle_local[kStages * kN + stage * kN];

      Gather(a_real, current_real, stage_a, 0, kN);
      Gather(a_imag, current_imag, stage_a, 0, kN);
      Gather(b_real, current_real, stage_b, 0, kN);
      Gather(b_imag, current_imag, stage_b, 0, kN);

      Mul(product0, b_real, twiddle_real, kN);
      Mul(product1, b_imag, twiddle_imag, kN);
      Sub(product0, product0, product1, kN);
      Mul(product1, b_real, twiddle_imag, kN);
      Mul(product2, b_imag, twiddle_real, kN);
      Add(product1, product1, product2, kN);
      Add(next_real, a_real, product0, kN);
      Add(next_imag, a_imag, product1, kN);

      LocalTensor<float> swap = current_real;
      current_real = next_real;
      next_real = swap;
      swap = current_imag;
      current_imag = next_imag;
      next_imag = swap;
    }

    LocalTensor<float> packed = work[4 * kN];
    DataCopy(packed, current_real, kN);
    DataCopy(packed[kN], current_imag, kN);
    PipeBarrier<PIPE_ALL>();
    LocalTensor<float> output_local = output_buf_.Get<float>();
    Gather(output_local, packed, index_local[kOutputIndexBase], 0, kOutputIndexCount);
    PipeBarrier<PIPE_ALL>();
    GlobalTensor<float> dst;
    dst.SetGlobalBuffer(output_ptr_ + transform * 2 * kN);
    DataCopy(dst, output_local, kOutputIndexCount);
  }

  __aicore__ inline void ProcessGroup4(uint32_t transform) {
    constexpr uint32_t group_n = kN * GroupSize;
    constexpr uint32_t output_index_base = group_n;
    constexpr uint32_t stage_a_local = 3 * group_n;
    constexpr uint32_t stage_b_local = 4 * group_n;
    constexpr uint32_t stage_a_global = 3 * group_n;
    constexpr uint32_t stage_b_global = 11 * group_n;

    GlobalTensor<float> src;
    src.SetGlobalBuffer(input_ptr_ + transform * 2 * kN);
    LocalTensor<float> input_local = input_buf_.Get<float>();
    DataCopy(input_local, src, 2 * group_n);

    LocalTensor<uint32_t> index_local = index_buf_.Get<uint32_t>();
    DataCopy(index_local, indices_, group_n);
    DataCopy(index_local[output_index_base], indices_[output_index_base], 2 * group_n);
    PipeBarrier<PIPE_ALL>();

    LocalTensor<float> work = work_buf_.Get<float>();
    LocalTensor<float> current_real = work[0 * group_n];
    LocalTensor<float> current_imag = work[1 * group_n];
    LocalTensor<float> next_real = work[2 * group_n];
    LocalTensor<float> next_imag = work[3 * group_n];
    LocalTensor<float> a_real = work[4 * group_n];
    LocalTensor<float> a_imag = work[5 * group_n];
    LocalTensor<float> b_real = work[6 * group_n];
    LocalTensor<float> b_imag = work[7 * group_n];
    LocalTensor<float> product0 = work[8 * group_n];
    LocalTensor<float> product1 = work[9 * group_n];
    LocalTensor<float> product2 = work[10 * group_n];

    Gather(current_real, input_local, index_local, 0, group_n);
    Gather(current_imag, input_local, index_local, sizeof(float), group_n);

    LocalTensor<float> twiddle_local = twiddle_buf_.Get<float>();
    for (uint32_t stage = 0; stage < kStages; ++stage) {
      const uint32_t stage_offset = stage * group_n;
      DataCopy(index_local[stage_a_local], indices_[stage_a_global + stage_offset], group_n);
      DataCopy(index_local[stage_b_local], indices_[stage_b_global + stage_offset], group_n);
      DataCopy(twiddle_local, twiddles_[stage_offset], group_n);
      DataCopy(twiddle_local[group_n], twiddles_[kStages * group_n + stage_offset], group_n);
      PipeBarrier<PIPE_ALL>();

      Gather(a_real, current_real, index_local[stage_a_local], 0, group_n);
      Gather(a_imag, current_imag, index_local[stage_a_local], 0, group_n);
      Gather(b_real, current_real, index_local[stage_b_local], 0, group_n);
      Gather(b_imag, current_imag, index_local[stage_b_local], 0, group_n);

      Mul(product0, b_real, twiddle_local, group_n);
      Mul(product1, b_imag, twiddle_local[group_n], group_n);
      Sub(product0, product0, product1, group_n);
      Mul(product1, b_real, twiddle_local[group_n], group_n);
      Mul(product2, b_imag, twiddle_local, group_n);
      Add(product1, product1, product2, group_n);
      Add(next_real, a_real, product0, group_n);
      Add(next_imag, a_imag, product1, group_n);

      LocalTensor<float> swap = current_real;
      current_real = next_real;
      next_real = swap;
      swap = current_imag;
      current_imag = next_imag;
      next_imag = swap;
    }

    LocalTensor<float> merged = work[8 * group_n];
    DataCopy(merged, current_real, group_n);
    DataCopy(merged[group_n], current_imag, group_n);
    PipeBarrier<PIPE_ALL>();
    LocalTensor<float> output_local = output_buf_.Get<float>();
    Gather(output_local, merged, index_local[output_index_base], 0, 2 * group_n);
    PipeBarrier<PIPE_ALL>();
    GlobalTensor<float> dst;
    dst.SetGlobalBuffer(output_ptr_ + transform * 2 * kN);
    DataCopy(dst, output_local, 2 * group_n);
  }

 private:
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
};
}  // namespace

