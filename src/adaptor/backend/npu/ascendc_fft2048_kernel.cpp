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
constexpr uint32_t kN = 2048;
constexpr uint32_t kStages = 11;
constexpr uint32_t kInputIndexCount = kN;
constexpr uint32_t kStageAIndexBase = kInputIndexCount;
constexpr uint32_t kStageBIndexBase = kStageAIndexBase + kStages * kN;
constexpr uint32_t kOutputIndexBase = kStageBIndexBase + kStages * kN;
constexpr uint32_t kWorkArrays = 11;

class Fft2048Aiv {
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

    // Keep the full transform and its current/next vectors in UB. Index and
    // twiddle storage is reused for every radix-2 stage to stay below the
    // 910B vector-local memory budget.
    pipe_.InitBuffer(input_buf_, 2 * kN * sizeof(float));
    pipe_.InitBuffer(work_buf_, kWorkArrays * kN * sizeof(float));
    pipe_.InitBuffer(index_buf_, 3 * kN * sizeof(uint32_t));
    pipe_.InitBuffer(twiddle_buf_, 2 * kN * sizeof(float));
    pipe_.InitBuffer(output_buf_, 2 * kN * sizeof(float));
  }

  __aicore__ inline void Process() {
    const uint32_t transform = GetBlockIdx();
    if (transform >= transform_count_) return;

    LocalTensor<float> input_local = input_buf_.Get<float>();
    GlobalTensor<float> src;
    src.SetGlobalBuffer(input_ptr_ + transform * 2 * kN);
    DataCopy(input_local, src, 2 * kN);

    LocalTensor<uint32_t> index_local = index_buf_.Get<uint32_t>();
    DataCopy(index_local, indices_, kInputIndexCount);
    PipeBarrier<PIPE_ALL>();

    LocalTensor<float> twiddle_local = twiddle_buf_.Get<float>();
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
    PipeBarrier<PIPE_ALL>();

    for (uint32_t stage = 0; stage < kStages; ++stage) {
      DataCopy(index_local[kN], indices_[kStageAIndexBase + stage * kN], kN);
      DataCopy(index_local[2 * kN], indices_[kStageBIndexBase + stage * kN], kN);
      DataCopy(twiddle_local, twiddles_[stage * 2 * kN], 2 * kN);
      PipeBarrier<PIPE_ALL>();

      const LocalTensor<uint32_t> stage_a = index_local[kN];
      const LocalTensor<uint32_t> stage_b = index_local[2 * kN];
      const LocalTensor<float> twiddle_real = twiddle_local;
      const LocalTensor<float> twiddle_imag = twiddle_local[kN];

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

      // The next iteration reloads this stage's index and twiddle tiles into
      // the same UB buffers. Finish the Vector reads before MTE2 reuses them.
      PipeBarrier<PIPE_ALL>();
    }

    DataCopy(input_local, current_real, kN);
    DataCopy(input_local[kN], current_imag, kN);
    DataCopy(index_local, indices_[kOutputIndexBase], 2 * kN);
    PipeBarrier<PIPE_ALL>();
    LocalTensor<float> output_local = output_buf_.Get<float>();
    Gather(output_local, input_local, index_local, 0, 2 * kN);
    PipeBarrier<PIPE_ALL>();
    GlobalTensor<float> dst;
    dst.SetGlobalBuffer(output_ptr_ + transform * 2 * kN);
    DataCopy(dst, output_local, 2 * kN);
  }

 private:
  TPipe pipe_;
  TBuf<QuePosition::VECCALC> input_buf_;
  TBuf<QuePosition::VECCALC> work_buf_;
  TBuf<QuePosition::VECCALC> index_buf_;
  TBuf<QuePosition::VECCALC> twiddle_buf_;
  TBuf<QuePosition::VECCALC> output_buf_;
  __gm__ float *input_ptr_ = nullptr;
  __gm__ float *output_ptr_ = nullptr;
  GlobalTensor<uint32_t> indices_;
  GlobalTensor<float> twiddles_;
  uint32_t transform_count_ = 0;
};
}  // namespace

extern "C" __global__ __aicore__ void flagfft_npu_fft2048(GM_ADDR input,
                                                           GM_ADDR output,
                                                           GM_ADDR indices,
                                                           GM_ADDR twiddles,
                                                           uint32_t transform_count) {
  Fft2048Aiv op;
  op.Init(input, output, indices, twiddles, transform_count);
  op.Process();
}
