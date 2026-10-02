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
constexpr uint32_t kGroupSize = 4;
constexpr uint32_t kGroupN = kN * kGroupSize;
constexpr uint32_t kStageIndexBase = 2 * kN;
constexpr uint32_t kOutputIndexBase = kStageIndexBase + 2 * kStages * kN;
constexpr uint32_t kOutputIndexCount = 2 * kN;
constexpr uint32_t kLocalIndexCount = kOutputIndexBase + kOutputIndexCount;
constexpr uint32_t kTwiddleCount = 2 * kStages * kN;
constexpr uint32_t kGroupOutputIndexBase = kGroupN;
constexpr uint32_t kGroupStageIndexBase = 3 * kGroupN;
constexpr uint32_t kGroupStageBIndexBase = kGroupStageIndexBase + kStages * kGroupN;
constexpr uint32_t kGroupIndexCount = 3 * kGroupN + 2 * kStages * kGroupN;
constexpr uint32_t kGroupTwiddleCount = 2 * kStages * kGroupN;
constexpr uint32_t kWorkArrays = 11;
constexpr uint32_t kComplexBytes = 2 * sizeof(float);

class Fft64Aiv {
 public:
  __aicore__ inline void Init(GM_ADDR input,
                              GM_ADDR output,
                              GM_ADDR indices,
                              GM_ADDR twiddles,
                              uint32_t transform_count,
                              uint32_t stride,
                              uint32_t group_size) {
    input_ptr_ = reinterpret_cast<__gm__ float *>(input);
    output_ptr_ = reinterpret_cast<__gm__ float *>(output);
    indices_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(indices));
    twiddles_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(twiddles));
    transform_count_ = transform_count;
    stride_ = stride;
    group_size_ = group_size;

    // Strided DataCopyPad moves each 8-byte complex value through a 32-byte
    // VECIN block, so reserve one block per input point.
    const bool group4 = group_size_ == kGroupSize;
    pipe_.InitBuffer(input_buf_, group4 ? kGroupN * 2 * sizeof(float) : kN * 8 * sizeof(float));
    pipe_.InitBuffer(work_buf_, (group4 ? kWorkArrays * kGroupN : kWorkArrays * kN) * sizeof(float));
    pipe_.InitBuffer(index_buf_, (group4 ? kGroupIndexCount : kLocalIndexCount) * sizeof(uint32_t));
    pipe_.InitBuffer(twiddle_buf_, (group4 ? kGroupTwiddleCount : kTwiddleCount) * sizeof(float));
    // MTE3 likewise reads one 8-byte complex value from each 32-byte VECOUT
    // block when emitting a strided column.
    pipe_.InitBuffer(output_buf_, (group4 ? 2 * kGroupN : kN * 8) * sizeof(float));
    if (group4) pipe_.InitBuffer(merge_buf_, 2 * kGroupN * sizeof(float));
  }

  __aicore__ inline void Process() {
    const uint32_t transform = GetBlockIdx();
    if (transform >= transform_count_ || (stride_ != 1 && stride_ != kN)) return;

    const bool strided = stride_ == kN;
    if (group_size_ == kGroupSize) {
      ProcessGroup4(transform);
      return;
    }
    const uint32_t batch_index = transform / kN;
    const uint32_t column = strided ? transform % kN : 0;
    const uint32_t source_base = strided ? batch_index * kN * kN + column : transform * kN;
    const uint32_t output_base = source_base;

    GlobalTensor<float> src;
    src.SetGlobalBuffer(input_ptr_ + source_base * 2);
    LocalTensor<float> input_local = input_buf_.Get<float>();
    if (strided) {
      // Read one complex value from each matrix row into padded VECIN blocks.
      // DataCopyExtParams strides are byte offsets for the GM source operand.
      GlobalTensor<uint64_t> src_complex;
      src_complex.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(input_ptr_ + source_base * 2));
      LocalTensor<uint64_t> input_complex = input_local.ReinterpretCast<uint64_t>();
      const DataCopyExtParams input_params(
          kN, sizeof(uint64_t), (kN - 1) * kComplexBytes, 0, 0);
      const DataCopyPadExtParams<uint64_t> input_pad;
      DataCopyPad(input_complex, src_complex, input_params, input_pad);
    } else {
      // Copy one interleaved row in a single contiguous transfer.
      DataCopy(input_local, src, 2 * kN);
    }

    LocalTensor<uint32_t> index_local = index_buf_.Get<uint32_t>();
    LocalTensor<float> twiddle_local = twiddle_buf_.Get<float>();
    DataCopy(index_local, indices_, 2 * kN);
    DataCopy(index_local[kStageIndexBase], indices_[kStageIndexBase],
             2 * kStages * kN);
    DataCopy(index_local[kOutputIndexBase], indices_[kOutputIndexBase], kOutputIndexCount);
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

    const LocalTensor<uint32_t> input_indices = strided ? index_local[kN] : index_local;
    Gather(current_real, input_local, input_indices, 0, kN);
    Gather(current_imag, input_local, input_indices, sizeof(float), kN);

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

    if (strided) {
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
    constexpr uint32_t groups_per_matrix = kN / kGroupSize;
    const uint32_t batch_index = transform / groups_per_matrix;
    const uint32_t group = transform % groups_per_matrix;
    const bool strided = stride_ == kN;
    const uint32_t source_base = batch_index * kN * kN +
        (strided ? group * kGroupSize : group * kGroupSize * kN);

    LocalTensor<float> input_local = input_buf_.Get<float>();
    if (strided) {
      GlobalTensor<uint64_t> src_complex;
      src_complex.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(input_ptr_ + source_base * 2));
      LocalTensor<uint64_t> input_complex = input_local.ReinterpretCast<uint64_t>();
      const DataCopyExtParams input_params(
          kN, kGroupSize * sizeof(uint64_t), (kN - kGroupSize) * kComplexBytes, 0, 0);
      const DataCopyPadExtParams<uint64_t> input_pad;
      DataCopyPad(input_complex, src_complex, input_params, input_pad);
    } else {
      GlobalTensor<float> src;
      src.SetGlobalBuffer(input_ptr_ + source_base * 2);
      DataCopy(input_local, src, 2 * kGroupN);
    }

    LocalTensor<uint32_t> index_local = index_buf_.Get<uint32_t>();
    LocalTensor<float> twiddle_local = twiddle_buf_.Get<float>();
    DataCopy(index_local, indices_, kGroupIndexCount);
    DataCopy(twiddle_local, twiddles_, kGroupTwiddleCount);
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

    const LocalTensor<uint32_t> input_indices = index_local;
    Gather(current_real, input_local, input_indices, 0, kGroupN);
    Gather(current_imag, input_local, input_indices, sizeof(float), kGroupN);

    for (uint32_t stage = 0; stage < kStages; ++stage) {
      const LocalTensor<uint32_t> stage_a = index_local[kGroupStageIndexBase + stage * kGroupN];
      const LocalTensor<uint32_t> stage_b = index_local[kGroupStageBIndexBase + stage * kGroupN];
      const LocalTensor<float> twiddle_real = twiddle_local[stage * kGroupN];
      const LocalTensor<float> twiddle_imag = twiddle_local[kStages * kGroupN + stage * kGroupN];

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

    // Merge real and imaginary planes and use a Vector gather to form either
    // four contiguous rows or four adjacent columns in the requested layout.
    LocalTensor<float> merged = merge_buf_.Get<float>();
    DataCopy(merged, current_real, kGroupN);
    DataCopy(merged[kGroupN], current_imag, kGroupN);
    PipeBarrier<PIPE_ALL>();
    LocalTensor<float> output_local = output_buf_.Get<float>();
    Gather(output_local, merged, index_local[kGroupOutputIndexBase], 0, 2 * kGroupN);
    PipeBarrier<PIPE_ALL>();

    if (strided) {
      GlobalTensor<uint64_t> dst_complex;
      dst_complex.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(output_ptr_ + source_base * 2));
      LocalTensor<uint64_t> output_complex = output_local.ReinterpretCast<uint64_t>();
      const DataCopyExtParams output_params(
          kN, kGroupSize * sizeof(uint64_t), 0, (kN - kGroupSize) * kComplexBytes, 0);
      DataCopyPad(dst_complex, output_complex, output_params);
    } else {
      GlobalTensor<float> dst;
      dst.SetGlobalBuffer(output_ptr_ + source_base * 2);
      DataCopy(dst, output_local, 2 * kGroupN);
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
  uint32_t transform_count_ = 0;
  uint32_t stride_ = 1;
  uint32_t group_size_ = 1;
};
}  // namespace

extern "C" __global__ __aicore__ void flagfft_npu_fft64(GM_ADDR input,
                                                        GM_ADDR output,
                                                        GM_ADDR indices,
                                                        GM_ADDR twiddles,
                                                        uint32_t transform_count,
                                                        uint32_t stride,
                                                        uint32_t group_size) {
  Fft64Aiv op;
  op.Init(input, output, indices, twiddles, transform_count, stride, group_size);
  op.Process();
}
