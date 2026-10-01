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
constexpr uint32_t kIndexCount = kN + 2 * kStages * kN;
constexpr uint32_t kTwiddleCount = 2 * kStages * kN;
constexpr uint32_t kWorkArrays = 11;

class Fft64Aiv {
 public:
  __aicore__ inline void Init(GM_ADDR input,
                              GM_ADDR output,
                              GM_ADDR indices,
                              GM_ADDR twiddles,
                              uint32_t transform_count,
                              uint32_t stride) {
    input_ptr_ = reinterpret_cast<__gm__ float *>(input);
    output_ptr_ = reinterpret_cast<__gm__ float *>(output);
    indices_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(indices));
    twiddles_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(twiddles));
    transform_count_ = transform_count;
    stride_ = stride;

    pipe_.InitBuffer(input_queue_, 1, 2 * kN * sizeof(float));
    pipe_.InitBuffer(work_buf_, kWorkArrays * kN * sizeof(float));
    pipe_.InitBuffer(index_buf_, kIndexCount * sizeof(uint32_t));
    pipe_.InitBuffer(twiddle_buf_, kTwiddleCount * sizeof(float));
  }

  __aicore__ inline void Process() {
    const uint32_t transform = GetBlockIdx();
    if (transform >= transform_count_) return;

    const uint32_t base = stride_ == 1 ? transform * kN : (transform / kN) * kN * kN + transform % kN;
    GlobalTensor<float> src_real;
    GlobalTensor<float> src_imag;
    src_real.SetGlobalBuffer(input_ptr_ + base * 2);
    src_imag.SetGlobalBuffer(input_ptr_ + base * 2 + 1);

    const uint32_t source_gap = (2 * stride_ - 1) * sizeof(float);
    const DataCopyExtParams input_params(kN, sizeof(float), source_gap, 0, 0);
    const DataCopyPadExtParams<float> no_padding(false, 0, 0, 0.0f);

    LocalTensor<float> input_local = input_queue_.AllocTensor<float>();
    DataCopyPad(input_local[0], src_real, input_params, no_padding);
    DataCopyPad(input_local[kN], src_imag, input_params, no_padding);
    input_queue_.EnQue(input_local);
    input_local = input_queue_.DeQue<float>();

    LocalTensor<uint32_t> index_local = index_buf_.Get<uint32_t>();
    LocalTensor<float> twiddle_local = twiddle_buf_.Get<float>();
    DataCopy(index_local, indices_, kIndexCount);
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

    Gather(current_real, input_local[0], index_local, 0, kN);
    Gather(current_imag, input_local[kN], index_local, 0, kN);
    input_queue_.FreeTensor(input_local);

    for (uint32_t stage = 0; stage < kStages; ++stage) {
      const LocalTensor<uint32_t> stage_a = index_local[kN + stage * kN];
      const LocalTensor<uint32_t> stage_b = index_local[kN + kStages * kN + stage * kN];
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

    PipeBarrier<PIPE_V>();
    GlobalTensor<float> dst_real;
    GlobalTensor<float> dst_imag;
    dst_real.SetGlobalBuffer(output_ptr_ + base * 2);
    dst_imag.SetGlobalBuffer(output_ptr_ + base * 2 + 1);
    const uint32_t destination_gap = (2 * stride_ - 1) * sizeof(float);
    const DataCopyExtParams output_params(kN, sizeof(float), 0, destination_gap, 0);
    DataCopyPad(dst_real, current_real, output_params);
    DataCopyPad(dst_imag, current_imag, output_params);
  }

 private:
  TPipe pipe_;
  TQue<QuePosition::VECIN, 1> input_queue_;
  TBuf<QuePosition::VECCALC> work_buf_;
  TBuf<QuePosition::VECCALC> index_buf_;
  TBuf<QuePosition::VECCALC> twiddle_buf_;
  __gm__ float *input_ptr_ = nullptr;
  __gm__ float *output_ptr_ = nullptr;
  GlobalTensor<uint32_t> indices_;
  GlobalTensor<float> twiddles_;
  uint32_t transform_count_ = 0;
  uint32_t stride_ = 1;
};
}  // namespace

extern "C" __global__ __aicore__ void flagfft_npu_fft64(GM_ADDR input,
                                                        GM_ADDR output,
                                                        GM_ADDR indices,
                                                        GM_ADDR twiddles,
                                                        uint32_t transform_count,
                                                        uint32_t stride) {
  Fft64Aiv op;
  op.Init(input, output, indices, twiddles, transform_count, stride);
  op.Process();
}
