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

template <uint32_t GroupSize,
          bool PairButterflies = false,
          bool TransposedOutput = false,
          bool RealForward = false,
          bool RealInverse = false>
class Fft256Aiv {
 public:
  __aicore__ inline void Init(GM_ADDR input,
                              GM_ADDR output,
                              GM_ADDR indices,
                              GM_ADDR twiddles,
                              uint32_t transform_count,
                              uint32_t output_row_stride = 0,
                              uint32_t output_transform_offset = 0) {
    input_ptr_ = reinterpret_cast<__gm__ float *>(input);
    output_ptr_ = reinterpret_cast<__gm__ float *>(output);
    indices_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(indices));
    twiddles_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(twiddles));
    transform_count_ = transform_count;
    output_row_stride_ = output_row_stride;
    output_transform_offset_ = output_transform_offset;

    constexpr uint32_t group_n = kN * GroupSize;
    constexpr uint32_t index_count = PairButterflies ? 4 * group_n :
        (GroupSize == 1 ? kLocalIndexCount : 5 * group_n);
    constexpr uint32_t twiddle_count = PairButterflies ? group_n :
        (GroupSize == 1 ? kTwiddleCount : 2 * group_n);
    pipe_.InitBuffer(input_buf_, 2 * group_n * sizeof(float));
    constexpr uint32_t work_vectors = PairButterflies ? 8 : 11;
    static_assert(!(RealForward && RealInverse));
    static_assert((!RealForward && !RealInverse) ||
                  (GroupSize == 8 && PairButterflies && !TransposedOutput));
    pipe_.InitBuffer(work_buf_, work_vectors * group_n * sizeof(float));
    pipe_.InitBuffer(index_buf_, index_count * sizeof(uint32_t));
    pipe_.InitBuffer(twiddle_buf_, twiddle_count * sizeof(float));
    pipe_.InitBuffer(output_buf_, 2 * group_n * sizeof(float));
  }

  __aicore__ inline void Process() {
    const uint32_t transform = GetBlockIdx() * GroupSize;
    if (transform >= transform_count_) return;

    if constexpr (PairButterflies) {
      if constexpr (RealForward || RealInverse) {
        const uint32_t block_stride = static_cast<uint32_t>(GetBlockNum()) * GroupSize;
        for (uint32_t row = transform; row < transform_count_; row += block_stride) {
          ProcessPairGroup8(row);
        }
      } else {
        ProcessPairGroup8(transform);
      }
      return;
    }

    if constexpr (GroupSize == 4 || GroupSize == 8) {
      ProcessGrouped(transform);
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

  __aicore__ inline void ProcessGrouped(uint32_t transform) {
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
      PipeBarrier<PIPE_ALL>();
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

  __aicore__ inline void ProcessPairGroup8(uint32_t transform) {
    constexpr uint32_t group_n = kN * GroupSize;
    constexpr uint32_t half_group = group_n / 2;
    constexpr uint32_t half = kN / 2 + 1;
    constexpr uint32_t output_index_base = group_n;
    constexpr uint32_t stage_a_local = 3 * group_n;
    constexpr uint32_t stage_b_local = stage_a_local + half_group;
    constexpr uint32_t stage_a_global = 3 * group_n;
    constexpr uint32_t stage_b_global = 7 * group_n;
    static_assert(GroupSize == 8);

    GlobalTensor<float> src;
    if constexpr (RealForward) {
      src.SetGlobalBuffer(input_ptr_ + transform * kN);
    } else if constexpr (RealInverse) {
      src.SetGlobalBuffer(input_ptr_ + transform * 2 * half);
    } else {
      src.SetGlobalBuffer(input_ptr_ + transform * 2 * kN);
    }
    LocalTensor<float> input_local = input_buf_.Get<float>();
    if constexpr (RealForward) {
      DataCopy(input_local, src, group_n);
    } else if constexpr (RealInverse) {
      DataCopy(input_local, src, 2 * half * GroupSize);
    } else {
      DataCopy(input_local, src, 2 * group_n);
    }

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
    LocalTensor<float> a_imag = work[4 * group_n + half_group];
    LocalTensor<float> b_real = work[5 * group_n];
    LocalTensor<float> b_imag = work[5 * group_n + half_group];
    LocalTensor<float> product0 = work[6 * group_n];
    LocalTensor<float> product1 = work[6 * group_n + half_group];
    LocalTensor<float> product2 = work[7 * group_n];

    LocalTensor<float> twiddle_local = twiddle_buf_.Get<float>();
    Gather(current_real, input_local, index_local, 0, group_n);
    if constexpr (RealForward) {
      Duplicate(current_imag, 0.0f, group_n);
    } else {
      Gather(current_imag, input_local, index_local, sizeof(float), group_n);
      if constexpr (RealInverse) {
        DataCopy(twiddle_local, twiddles_[kStages * group_n], group_n);
        Mul(current_imag, current_imag, twiddle_local, group_n);
        PipeBarrier<PIPE_ALL>();
      }
    }

    for (uint32_t stage = 0; stage < kStages; ++stage) {
      const uint32_t stage_offset = stage * group_n;
      DataCopy(index_local[stage_a_local], indices_[stage_a_global + stage * half_group], half_group);
      DataCopy(index_local[stage_b_local], indices_[stage_b_global + stage * half_group], half_group);
      DataCopy(twiddle_local, twiddles_[stage_offset], group_n);
      PipeBarrier<PIPE_ALL>();

      Gather(a_real, current_real, index_local[stage_a_local], 0, half_group);
      Gather(a_imag, current_imag, index_local[stage_a_local], 0, half_group);
      Gather(b_real, current_real, index_local[stage_b_local], 0, half_group);
      Gather(b_imag, current_imag, index_local[stage_b_local], 0, half_group);

      Mul(product0, b_real, twiddle_local, half_group);
      Mul(product1, b_imag, twiddle_local[half_group], half_group);
      Sub(product0, product0, product1, half_group);
      Mul(product1, b_real, twiddle_local[half_group], half_group);
      Mul(product2, b_imag, twiddle_local, half_group);
      Add(product1, product1, product2, half_group);

      Add(next_real, a_real, product0, half_group);
      Add(next_imag, a_imag, product1, half_group);
      Sub(next_real[half_group], a_real, product0, half_group);
      Sub(next_imag[half_group], a_imag, product1, half_group);
      PipeBarrier<PIPE_ALL>();

      LocalTensor<float> swap = current_real;
      current_real = next_real;
      next_real = swap;
      swap = current_imag;
      current_imag = next_imag;
      next_imag = swap;
    }

    LocalTensor<float> merged = work[6 * group_n];
    DataCopy(merged, current_real, group_n);
    DataCopy(merged[group_n], current_imag, group_n);
    PipeBarrier<PIPE_ALL>();
    LocalTensor<float> output_local = output_buf_.Get<float>();
    if constexpr (RealForward) {
      Gather(output_local, merged, index_local[output_index_base], 0, 2 * half * GroupSize);
      PipeBarrier<PIPE_ALL>();
      GlobalTensor<float> dst;
      dst.SetGlobalBuffer(output_ptr_ + transform * 2 * half);
      DataCopy(dst, output_local, 2 * half * GroupSize);
      PipeBarrier<PIPE_ALL>();
    } else if constexpr (RealInverse) {
      Gather(output_local, current_real, index_local[output_index_base], 0, group_n);
      PipeBarrier<PIPE_ALL>();
      GlobalTensor<float> dst;
      dst.SetGlobalBuffer(output_ptr_ + transform * kN);
      DataCopy(dst, output_local, group_n);
      PipeBarrier<PIPE_ALL>();
    } else {
      Gather(output_local, merged, index_local[output_index_base], 0, 2 * group_n);
      PipeBarrier<PIPE_ALL>();
      if constexpr (TransposedOutput) {
        LocalTensor<uint64_t> output_complex = output_local.ReinterpretCast<uint64_t>();
        GlobalTensor<uint64_t> dst;
        dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(output_ptr_) +
                            output_transform_offset_ + transform);
        const DataCopyExtParams output_params(
            kN,
            GroupSize * sizeof(uint64_t),
            0,
            (output_row_stride_ - GroupSize) * sizeof(uint64_t),
            0);
        DataCopyPad(dst, output_complex, output_params);
      } else {
        GlobalTensor<float> dst;
        dst.SetGlobalBuffer(output_ptr_ + transform * 2 * kN);
        DataCopy(dst, output_local, 2 * group_n);
      }
    }
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
  uint32_t output_row_stride_ = 0;
  uint32_t output_transform_offset_ = 0;
};
}  // namespace
