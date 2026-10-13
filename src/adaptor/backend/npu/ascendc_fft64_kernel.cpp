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

#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t kN = 64;
constexpr uint32_t kStages = 6;
constexpr uint32_t kStageIndexBase = 2 * kN;
constexpr uint32_t kOutputIndexBase = kStageIndexBase + 2 * kStages * kN;
constexpr uint32_t kOutputIndexCount = 2 * kN;
constexpr uint32_t kLocalIndexCount = kOutputIndexBase + kOutputIndexCount;
constexpr uint32_t kTwiddleCount = 2 * kStages * kN;
constexpr uint32_t kWorkArrays = 11;
constexpr uint32_t kComplexBytes = 2 * sizeof(float);
constexpr uint32_t kComputeGroupSize = 4;
constexpr uint32_t kComputeGroupN = kN * kComputeGroupSize;
constexpr uint32_t kComputeOutputIndexBase = kComputeGroupN;
constexpr uint32_t kComputeStageIndexBase = 3 * kComputeGroupN;
constexpr uint32_t kComputeStageBIndexBase = kComputeStageIndexBase + kStages * kComputeGroupN;
constexpr uint32_t kComputeIndexCount = 3 * kComputeGroupN + 2 * kStages * kComputeGroupN;
constexpr uint32_t kComputeTwiddleCount = 2 * kStages * kComputeGroupN;
constexpr uint32_t kModeComplex = 0;
constexpr uint32_t kModeRealForward = 1;
constexpr uint32_t kModeRealInverse = 2;

class Fft64Aiv {
 public:
  __aicore__ inline void Init(GM_ADDR input,
                              GM_ADDR output,
                              GM_ADDR indices,
                              GM_ADDR twiddles,
                              uint32_t transform_count,
                              uint32_t stride,
                              uint32_t group_size,
                              uint32_t mode) {
    input_ptr_ = reinterpret_cast<__gm__ float *>(input);
    output_ptr_ = reinterpret_cast<__gm__ float *>(output);
    indices_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(indices));
    twiddles_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(twiddles));
    transform_count_ = transform_count;
    stride_ = stride;
    group_size_ = group_size;
    mode_ = mode;

    // Strided DataCopyPad moves each 8-byte complex value through a 32-byte
    // VECIN block, so reserve one block per input point.
    const bool grouped = group_size_ > 1;
    const uint32_t input_group_n = kN * group_size_;
    const uint32_t compute_group_n = group_size_ == 8 ? kComputeGroupN : input_group_n;
    const uint32_t group_index_count =
        kComputeIndexCount + (group_size_ == 8 ? kComputeGroupN : 0) +
        (mode_ == kModeRealInverse ? compute_group_n * (group_size_ == 8 ? 2 : 1) : 0);
    const uint32_t group_twiddle_count =
        2 * kStages * compute_group_n + (mode_ == kModeRealInverse ? compute_group_n : 0);
    pipe_.InitBuffer(input_buf_, (grouped ? input_group_n * 2 : kN * 8) * sizeof(float));
    pipe_.InitBuffer(work_buf_, (grouped ? kWorkArrays * compute_group_n : kWorkArrays * kN) * sizeof(float));
    pipe_.InitBuffer(index_buf_, (grouped ? group_index_count : kLocalIndexCount) * sizeof(uint32_t));
    pipe_.InitBuffer(twiddle_buf_, (grouped ? group_twiddle_count : kTwiddleCount) * sizeof(float));
    // MTE3 likewise reads one 8-byte complex value from each 32-byte VECOUT
    // block when emitting a strided column.
    pipe_.InitBuffer(output_buf_, (grouped ? 2 * compute_group_n : kN * 8) * sizeof(float));
    if (grouped) pipe_.InitBuffer(merge_buf_, 2 * compute_group_n * sizeof(float));
    if (mode_ == kModeRealInverse) {
      pipe_.InitBuffer(compact_buf_, group_size_ * 2 * (kN / 2 + 1) * sizeof(float));
    }
  }

  __aicore__ inline void Process() {
    const uint32_t transform = GetBlockIdx();
    if (transform >= transform_count_ || stride_ == 0 ||
        (group_size_ != 1 && stride_ != 1 && stride_ != kN)) return;

    const bool strided = stride_ != 1;
    if (group_size_ == 4) {
      ProcessGroup4(transform);
      return;
    }
    if (group_size_ == 8) {
      ProcessGroup8(transform);
      return;
    }
    const uint32_t batch_index = strided ? transform / stride_ : transform / kN;
    const uint32_t column = strided ? transform % stride_ : 0;
    const uint32_t source_base = mode_ == kModeRealInverse
        ? transform * (kN / 2 + 1)
        : mode_ == kModeRealForward
              ? transform * kN
              : strided ? batch_index * kN * stride_ + column : transform * kN;
    const uint32_t output_base = mode_ == kModeRealForward
        ? transform * (kN / 2 + 1)
        : mode_ == kModeRealInverse ? transform * kN : source_base;

    GlobalTensor<float> src;
    LocalTensor<float> input_local = input_buf_.Get<float>();
    if (mode_ == kModeRealInverse) {
      src.SetGlobalBuffer(input_ptr_ + source_base * 2);
      LocalTensor<float> compact_local = compact_buf_.Get<float>();
      DataCopy(compact_local, src, 2 * (kN / 2 + 1));
    } else if (strided) {
      src.SetGlobalBuffer(input_ptr_ + source_base * 2);
      // Read one complex value from each matrix row into padded VECIN blocks.
      // DataCopyExtParams strides are byte offsets for the GM source operand.
      GlobalTensor<uint64_t> src_complex;
      src_complex.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(input_ptr_ + source_base * 2));
      LocalTensor<uint64_t> input_complex = input_local.ReinterpretCast<uint64_t>();
      const DataCopyExtParams input_params(kN, sizeof(uint64_t), (stride_ - 1) * kComplexBytes, 0, 0);
      const DataCopyPadExtParams<uint64_t> input_pad;
      DataCopyPad(input_complex, src_complex, input_params, input_pad);
    } else if (mode_ == kModeRealForward) {
      src.SetGlobalBuffer(input_ptr_ + source_base);
      DataCopy(input_local, src, kN);
    } else {
      // Copy one interleaved row in a single contiguous transfer.
      src.SetGlobalBuffer(input_ptr_ + source_base * 2);
      DataCopy(input_local, src, 2 * kN);
    }

    LocalTensor<uint32_t> index_local = index_buf_.Get<uint32_t>();
    LocalTensor<float> twiddle_local = twiddle_buf_.Get<float>();
    DataCopy(index_local, indices_, 2 * kN);
    DataCopy(index_local[kStageIndexBase], indices_[kStageIndexBase],
             2 * kStages * kN);
    DataCopy(index_local[kOutputIndexBase], indices_[kOutputIndexBase], kOutputIndexCount);
    DataCopy(twiddle_local,
             twiddles_,
             kTwiddleCount + (mode_ == kModeRealInverse ? kN : 0));
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

    const LocalTensor<uint32_t> input_indices = strided ? index_local[kN] : index_local;
    if (mode_ == kModeRealInverse) {
      const LocalTensor<float> compact_local = compact_buf_.Get<float>();
      Gather(current_real, compact_local, input_indices, 0, kN);
      Gather(current_imag, compact_local, index_local[kN], 0, kN);
      Mul(current_imag, current_imag, twiddle_local[kTwiddleCount], kN);
    } else {
      Gather(current_real, input_local, input_indices, 0, kN);
      if (mode_ == kModeRealForward) {
        Duplicate(current_imag, 0.0f, kN);
      } else {
        Gather(current_imag, input_local, input_indices, sizeof(float), kN);
      }
    }

    for (uint32_t stage = 0; stage < kStages; ++stage) {
      const LocalTensor<uint32_t> stage_a = index_local[kStageIndexBase + stage * kN];
      const LocalTensor<uint32_t> stage_b =
          index_local[kStageIndexBase + kStages * kN + stage * kN];
      const LocalTensor<float> twiddle_real = twiddle_local[stage * kN];
      const LocalTensor<float> twiddle_imag = twiddle_local[kStages * kN + stage * kN];

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

    if (mode_ == kModeRealInverse) {
      LocalTensor<float> output_local = output_buf_.Get<float>();
      Gather(output_local, current_real, index_local[kOutputIndexBase], 0, kN);
      PipeBarrier<PIPE_ALL>();
      GlobalTensor<float> dst;
      dst.SetGlobalBuffer(output_ptr_ + output_base);
      DataCopy(dst, output_local, kN);
    } else if (mode_ == kModeRealForward) {
      PipeBarrier<PIPE_ALL>();
      LocalTensor<float> packed = work[4 * kN];
      DataCopy(packed, current_real, kN);
      DataCopy(packed[kN], current_imag, kN);
      PipeBarrier<PIPE_ALL>();
      LocalTensor<float> output_local = output_buf_.Get<float>();
      constexpr uint32_t half_count = 2 * (kN / 2 + 1);
      Gather(output_local, packed, index_local[kOutputIndexBase], 0, half_count);
      PipeBarrier<PIPE_ALL>();
      GlobalTensor<float> dst;
      dst.SetGlobalBuffer(output_ptr_ + output_base * 2);
      DataCopy(dst, output_local, half_count);
    } else if (strided) {
      // Pack complex values into padded VECOUT blocks before the strided MTE3
      // write. AIV scalar stores to this GM column layout are not reliable.
      LocalTensor<float> output_local = output_buf_.Get<float>();
      for (uint32_t i = 0; i < kN; ++i) {
        output_local.SetValue(8 * i, current_real.GetValue(i));
        output_local.SetValue(8 * i + 1, current_imag.GetValue(i));
      }
      PipeBarrier<PIPE_ALL>();
      GlobalTensor<uint64_t> dst_complex;
      dst_complex.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(output_ptr_ + output_base * 2));
      LocalTensor<uint64_t> output_complex = output_local.ReinterpretCast<uint64_t>();
      const DataCopyExtParams output_params(
          kN, sizeof(uint64_t), 0, (stride_ - 1) * kComplexBytes, 0);
      DataCopyPad(dst_complex, output_complex, output_params);
    } else {
      // Interleave the real and imaginary planes in local memory, then let
      // MTE3 write the contiguous row instead of issuing scalar GM stores.
      PipeBarrier<PIPE_ALL>();
      LocalTensor<float> packed = work[4 * kN];
      DataCopy(packed, current_real, kN);
      DataCopy(packed[kN], current_imag, kN);
      PipeBarrier<PIPE_ALL>();
      LocalTensor<float> output_local = output_buf_.Get<float>();
      Gather(output_local, packed, index_local[kOutputIndexBase], 0, kOutputIndexCount);
      PipeBarrier<PIPE_ALL>();
      GlobalTensor<float> dst;
      dst.SetGlobalBuffer(output_ptr_ + output_base * 2);
      DataCopy(dst, output_local, kOutputIndexCount);
    }
  }

  __aicore__ inline void ProcessGroup4(uint32_t transform) {
    constexpr uint32_t group_n = kComputeGroupN;
    const uint32_t group_index_count =
        kComputeIndexCount + (mode_ == kModeRealInverse ? kComputeGroupN : 0);
    const uint32_t group_twiddle_count =
        kComputeTwiddleCount + (mode_ == kModeRealInverse ? kComputeGroupN : 0);
    constexpr uint32_t groups_per_matrix = kN / kComputeGroupSize;
    const uint32_t batch_index = transform / groups_per_matrix;
    const uint32_t group = transform % groups_per_matrix;
    const bool strided = stride_ == kN;
    const uint32_t source_base = mode_ == kModeRealInverse
        ? batch_index * kN * (kN / 2 + 1) + group * kComputeGroupSize * (kN / 2 + 1)
        : batch_index * kN * kN +
              (strided ? group * kComputeGroupSize : group * kComputeGroupSize * kN);
    const uint32_t output_base = mode_ == kModeRealForward
        ? batch_index * kN * (kN / 2 + 1) + group * kComputeGroupSize * (kN / 2 + 1)
        : mode_ == kModeRealInverse
              ? batch_index * kN * kN + group * kComputeGroupSize * kN
              : source_base;

    LocalTensor<float> input_local = input_buf_.Get<float>();
    if (mode_ == kModeRealInverse) {
      GlobalTensor<float> src;
      src.SetGlobalBuffer(input_ptr_ + source_base * 2);
      LocalTensor<float> compact_local = compact_buf_.Get<float>();
      DataCopy(compact_local, src, 2 * (kN / 2 + 1) * kComputeGroupSize);
    } else if (strided) {
      GlobalTensor<uint64_t> src_complex;
      src_complex.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(input_ptr_ + source_base * 2));
      LocalTensor<uint64_t> input_complex = input_local.ReinterpretCast<uint64_t>();
      const DataCopyExtParams input_params(
          kN, kComputeGroupSize * sizeof(uint64_t),
          (kN - kComputeGroupSize) * kComplexBytes, 0, 0);
      const DataCopyPadExtParams<uint64_t> input_pad;
      DataCopyPad(input_complex, src_complex, input_params, input_pad);
    } else if (mode_ == kModeRealForward) {
      GlobalTensor<float> src;
      src.SetGlobalBuffer(input_ptr_ + source_base);
      DataCopy(input_local, src, kN * kComputeGroupSize);
    } else {
      GlobalTensor<float> src;
      src.SetGlobalBuffer(input_ptr_ + source_base * 2);
      DataCopy(input_local, src, 2 * group_n);
    }

    LocalTensor<uint32_t> index_local = index_buf_.Get<uint32_t>();
    LocalTensor<float> twiddle_local = twiddle_buf_.Get<float>();
    DataCopy(index_local, indices_, group_index_count);
    DataCopy(twiddle_local, twiddles_, group_twiddle_count);
    PipeBarrier<PIPE_ALL>();

    TransformGroup4(0, output_base, false);
  }

  __aicore__ inline void ProcessGroup8(uint32_t transform) {
    constexpr uint32_t groups_per_matrix = kN / 8;
    const uint32_t batch_index = transform / groups_per_matrix;
    const uint32_t group = transform % groups_per_matrix;
    const bool strided = stride_ == kN;
    const uint32_t source_base = mode_ == kModeRealInverse
        ? batch_index * kN * (kN / 2 + 1) + group * 8 * (kN / 2 + 1)
        : batch_index * kN * kN + (strided ? group * 8 : group * 8 * kN);
    const uint32_t output_base = mode_ == kModeRealForward
        ? batch_index * kN * (kN / 2 + 1) + group * 8 * (kN / 2 + 1)
        : mode_ == kModeRealInverse
              ? batch_index * kN * kN + group * 8 * kN
              : source_base;

    LocalTensor<float> input_local = input_buf_.Get<float>();
    if (mode_ == kModeRealInverse) {
      GlobalTensor<float> src;
      src.SetGlobalBuffer(input_ptr_ + source_base * 2);
      LocalTensor<float> compact_local = compact_buf_.Get<float>();
      DataCopy(compact_local, src, 2 * (kN / 2 + 1) * 8);
    } else if (strided) {
      GlobalTensor<uint64_t> src_complex;
      src_complex.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(input_ptr_ + source_base * 2));
      LocalTensor<uint64_t> input_complex = input_local.ReinterpretCast<uint64_t>();
      const DataCopyExtParams input_params(
          kN, 8 * sizeof(uint64_t), (kN - 8) * kComplexBytes, 0, 0);
      const DataCopyPadExtParams<uint64_t> input_pad;
      DataCopyPad(input_complex, src_complex, input_params, input_pad);
    } else if (mode_ == kModeRealForward) {
      GlobalTensor<float> src;
      src.SetGlobalBuffer(input_ptr_ + source_base);
      DataCopy(input_local, src, kN * 8);
    } else {
      GlobalTensor<float> src;
      src.SetGlobalBuffer(input_ptr_ + source_base * 2);
      DataCopy(input_local, src, 2 * kN * 8);
    }

    LocalTensor<uint32_t> index_local = index_buf_.Get<uint32_t>();
    LocalTensor<float> twiddle_local = twiddle_buf_.Get<float>();
    const uint32_t group_index_count =
        kComputeIndexCount + kComputeGroupN +
        (mode_ == kModeRealInverse ? 2 * kComputeGroupN : 0);
    DataCopy(index_local, indices_, group_index_count);
    DataCopy(twiddle_local,
             twiddles_,
             kComputeTwiddleCount + (mode_ == kModeRealInverse ? kComputeGroupN : 0));
    PipeBarrier<PIPE_ALL>();

    TransformGroup4(0, output_base, strided);
    const uint32_t second_output_base = mode_ == kModeRealForward
        ? output_base + 4 * (kN / 2 + 1)
        : mode_ == kModeRealInverse ? output_base + 4 * kN
                                     : source_base + (strided ? 4 : 4 * kN);
    TransformGroup4(kComputeIndexCount, second_output_base, false);
  }

  __aicore__ inline void TransformGroup4(uint32_t input_index_offset,
                                         uint32_t output_base,
                                         bool wait_before_reuse) {
    constexpr uint32_t group_n = kComputeGroupN;
    constexpr uint32_t output_index_base = kComputeOutputIndexBase;
    constexpr uint32_t stage_index_base = kComputeStageIndexBase;
    constexpr uint32_t stage_b_index_base = kComputeStageBIndexBase;
    const bool strided = stride_ == kN;

    LocalTensor<float> input_local = input_buf_.Get<float>();
    LocalTensor<float> twiddle_local = twiddle_buf_.Get<float>();
    LocalTensor<float> work = work_buf_.Get<float>();
    LocalTensor<uint32_t> index_local = index_buf_.Get<uint32_t>();
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

    const LocalTensor<uint32_t> input_indices = index_local[input_index_offset];
    if (mode_ == kModeRealInverse) {
      const uint32_t imag_input_base =
          kComputeIndexCount + (group_size_ == 8 ? kComputeGroupN : 0);
      const uint32_t imag_offset = input_index_offset == 0 ? 0 : kComputeGroupN;
      const LocalTensor<float> compact_local = compact_buf_.Get<float>();
      Gather(current_real, compact_local, input_indices, 0, group_n);
      Gather(current_imag,
             compact_local,
             index_local[imag_input_base + imag_offset],
             0,
             group_n);
      Mul(current_imag, current_imag, twiddle_local[2 * kStages * group_n], group_n);
    } else {
      Gather(current_real, input_local, input_indices, 0, group_n);
      if (mode_ == kModeRealForward) {
        Duplicate(current_imag, 0.0f, group_n);
      } else {
        Gather(current_imag, input_local, input_indices, sizeof(float), group_n);
      }
    }

    for (uint32_t stage = 0; stage < kStages; ++stage) {
      const LocalTensor<uint32_t> stage_a = index_local[stage_index_base + stage * group_n];
      const LocalTensor<uint32_t> stage_b = index_local[stage_b_index_base + stage * group_n];
      const LocalTensor<float> twiddle_real = twiddle_local[stage * group_n];
      const LocalTensor<float> twiddle_imag = twiddle_local[kStages * group_n + stage * group_n];

      Gather(a_real, current_real, stage_a, 0, group_n);
      Gather(a_imag, current_imag, stage_a, 0, group_n);
      Gather(b_real, current_real, stage_b, 0, group_n);
      Gather(b_imag, current_imag, stage_b, 0, group_n);

      Mul(product0, b_real, twiddle_real, group_n);
      Mul(product1, b_imag, twiddle_imag, group_n);
      Sub(product0, product0, product1, group_n);
      Mul(product1, b_real, twiddle_imag, group_n);
      Mul(product2, b_imag, twiddle_real, group_n);
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

    LocalTensor<float> output_local = output_buf_.Get<float>();
    if (mode_ == kModeRealInverse) {
      Gather(output_local, current_real, index_local[output_index_base], 0, group_n);
      PipeBarrier<PIPE_ALL>();
      GlobalTensor<float> dst_real;
      dst_real.SetGlobalBuffer(output_ptr_ + output_base);
      DataCopy(dst_real, output_local, group_n);
    } else {
      // Merge the real and imaginary planes and gather them into the requested
      // grouped row or adjacent-column output layout.
      LocalTensor<float> merged = merge_buf_.Get<float>();
      DataCopy(merged, current_real, group_n);
      DataCopy(merged[group_n], current_imag, group_n);
      PipeBarrier<PIPE_ALL>();
      const uint32_t output_count = mode_ == kModeRealForward
          ? 2 * (kN / 2 + 1) * kComputeGroupSize
          : 2 * group_n;
      Gather(output_local, merged, index_local[output_index_base], 0, output_count);
      PipeBarrier<PIPE_ALL>();

      if (strided) {
        GlobalTensor<uint64_t> dst_complex;
        dst_complex.SetGlobalBuffer(
            reinterpret_cast<__gm__ uint64_t *>(output_ptr_ + output_base * 2));
        LocalTensor<uint64_t> output_complex = output_local.ReinterpretCast<uint64_t>();
        const DataCopyExtParams output_params(
            kN, kComputeGroupSize * sizeof(uint64_t), 0,
            (kN - kComputeGroupSize) * kComplexBytes, 0);
        DataCopyPad(dst_complex, output_complex, output_params);
      } else {
        GlobalTensor<float> dst;
        dst.SetGlobalBuffer(output_ptr_ + output_base * 2);
        DataCopy(dst, output_local, output_count);
      }
    }
    if (wait_before_reuse) PipeBarrier<PIPE_ALL>();
  }

 private:
  TPipe pipe_;
  TBuf<QuePosition::VECCALC> input_buf_;
  TBuf<QuePosition::VECCALC> work_buf_;
  TBuf<QuePosition::VECCALC> index_buf_;
  TBuf<QuePosition::VECCALC> twiddle_buf_;
  TBuf<QuePosition::VECCALC> output_buf_;
  TBuf<QuePosition::VECCALC> merge_buf_;
  TBuf<QuePosition::VECCALC> compact_buf_;
  __gm__ float *input_ptr_ = nullptr;
  __gm__ float *output_ptr_ = nullptr;
  GlobalTensor<uint32_t> indices_;
  GlobalTensor<float> twiddles_;
  uint32_t transform_count_ = 0;
  uint32_t stride_ = 1;
  uint32_t group_size_ = 1;
  uint32_t mode_ = kModeComplex;
};
}  // namespace

extern "C" __global__ __aicore__ void flagfft_npu_fft64(GM_ADDR input,
                                                        GM_ADDR output,
                                                        GM_ADDR indices,
                                                        GM_ADDR twiddles,
                                                        uint32_t transform_count,
                                                        uint32_t stride,
                                                        uint32_t group_size,
                                                        uint32_t mode) {
  Fft64Aiv op;
  op.Init(input, output, indices, twiddles, transform_count, stride, group_size, mode);
  op.Process();
}
