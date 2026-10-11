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
constexpr uint32_t kTile = 32;
constexpr uint32_t kMaxRadix = 19;
constexpr uint32_t kMaxRootFloats = (2 * kMaxRadix * kMaxRadix + 7) / 8 * 8;

class StockhamStageAiv {
 public:
  __aicore__ inline void Init(GM_ADDR input,
                              GM_ADDR output,
                              GM_ADDR stage_twiddles,
                              GM_ADDR radix_roots,
                              GM_ADDR index_table,
                              uint32_t length,
                              uint32_t radix,
                              uint32_t span,
                              uint32_t batch) {
    input_ptr_ = reinterpret_cast<__gm__ uint64_t *>(input);
    output_ptr_ = reinterpret_cast<__gm__ uint64_t *>(output);
    stage_twiddles_ptr_ = reinterpret_cast<__gm__ uint64_t *>(stage_twiddles);
    radix_roots_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(radix_roots));
    indices_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(index_table));
    length_ = length;
    radix_ = radix;
    span_ = span;
    batch_ = batch;
    ndiv_ = length_ / radix_;
    tiles_per_group_ = span_ == 1 ? 1 : (span_ + kTile - 1) / kTile;
    groups_per_transform_ = span_ == 1 ? 1 : ndiv_ / span_;
    tiles_per_transform_ = span_ == 1 ? (ndiv_ + kTile - 1) / kTile
                                       : groups_per_transform_ * tiles_per_group_;
    root_float_count_ = ((2 * radix_ * radix_ + 7) / 8) * 8;

    pipe_.InitBuffer(input_buf_, 2 * kTile * sizeof(float));
    pipe_.InitBuffer(twiddle_buf_, 2 * kTile * sizeof(float));
    pipe_.InitBuffer(digits_buf_, 2 * kMaxRadix * kTile * sizeof(float));
    pipe_.InitBuffer(roots_buf_, kMaxRootFloats * sizeof(float));
    pipe_.InitBuffer(indices_buf_, 2 * kTile * sizeof(uint32_t));
    pipe_.InitBuffer(root_indices_buf_, 2 * kTile * sizeof(uint32_t));
    pipe_.InitBuffer(work_buf_, 7 * kTile * sizeof(float));
    pipe_.InitBuffer(store_buf_, 8 * kTile * sizeof(float));

    LocalTensor<float> roots = roots_buf_.Get<float>();
    DataCopy(roots, radix_roots_, root_float_count_);
    LocalTensor<uint32_t> indices = indices_buf_.Get<uint32_t>();
    DataCopy(indices, indices_, 2 * kTile);
    PipeBarrier<PIPE_ALL>();
  }

  __aicore__ inline void Process() {
    const uint64_t total_tiles = static_cast<uint64_t>(batch_) * tiles_per_transform_;
    const uint64_t block_count = GetBlockNum();
    if (block_count == 0) return;
    for (uint64_t tile_id = GetBlockIdx(); tile_id < total_tiles; tile_id += block_count) {
      ProcessTile(tile_id);
    }
  }

 private:
  __aicore__ inline void ProcessTile(uint64_t tile_id) {
    const uint64_t batch_index = tile_id / tiles_per_transform_;
    const uint64_t local_tile = tile_id % tiles_per_transform_;
    uint64_t k_start = 0;
    uint32_t j_start = 0;
    uint32_t valid = 0;
    if (span_ == 1) {
      k_start = local_tile * kTile;
      valid = static_cast<uint32_t>(ndiv_ - k_start < kTile ? ndiv_ - k_start : kTile);
    } else {
      const uint64_t group = local_tile / tiles_per_group_;
      const uint64_t tile_in_group = local_tile % tiles_per_group_;
      j_start = static_cast<uint32_t>(tile_in_group * kTile);
      k_start = group * span_ + j_start;
      valid = span_ - j_start < kTile ? span_ - j_start : kTile;
    }

    LocalTensor<float> input_local = input_buf_.Get<float>();
    LocalTensor<uint64_t> input_complex = input_local.ReinterpretCast<uint64_t>();
    LocalTensor<uint32_t> indices = indices_buf_.Get<uint32_t>();
    LocalTensor<float> digits = digits_buf_.Get<float>();
    LocalTensor<uint32_t> root_indices = root_indices_buf_.Get<uint32_t>();
    LocalTensor<float> twiddle = twiddle_buf_.Get<float>();
    LocalTensor<float> roots = roots_buf_.Get<float>();
    LocalTensor<float> work = work_buf_.Get<float>();
    LocalTensor<float> tmp_real = work[2 * kTile];
    LocalTensor<float> tmp_imag = work[3 * kTile];
    LocalTensor<float> tmp_aux = work[4 * kTile];
    LocalTensor<float> twiddle_real = work[5 * kTile];
    LocalTensor<float> twiddle_imag = work[6 * kTile];

    for (uint32_t digit = 0; digit < radix_; ++digit) {
      Duplicate(input_local, 0.0f, 2 * kTile);
      PipeBarrier<PIPE_ALL>();
      const uint64_t input_offset = batch_index * length_ + k_start +
                                    static_cast<uint64_t>(digit) * ndiv_;
      GlobalTensor<uint64_t> src;
      src.SetGlobalBuffer(input_ptr_ + input_offset);
      if (valid == kTile) {
        DataCopy(input_complex, src, kTile);
      } else {
        const DataCopyExtParams params(1, valid * sizeof(uint64_t), 0, 0, 0);
        const DataCopyPadExtParams<uint64_t> pad;
        DataCopyPad(input_complex, src, params, pad);
      }
      PipeBarrier<PIPE_ALL>();

      LocalTensor<float> digit_real = digits[digit * 2 * kTile];
      LocalTensor<float> digit_imag = digits[digit * 2 * kTile + kTile];
      Gather(digit_real, input_local, indices, 0, kTile);
      Gather(digit_imag, input_local, indices[kTile], 0, kTile);
      PipeBarrier<PIPE_ALL>();

      if (digit > 0 && span_ > 1) {
        Duplicate(twiddle, 0.0f, 2 * kTile);
        PipeBarrier<PIPE_ALL>();
        GlobalTensor<uint64_t> ext_twiddle;
        ext_twiddle.SetGlobalBuffer(stage_twiddles_ptr_ +
                                    static_cast<uint64_t>(digit) * span_ + j_start);
        if (valid == kTile) {
          DataCopy(twiddle.ReinterpretCast<uint64_t>(), ext_twiddle, kTile);
        } else {
          const DataCopyExtParams params(1, valid * sizeof(uint64_t), 0, 0, 0);
          const DataCopyPadExtParams<uint64_t> pad;
          DataCopyPad(twiddle.ReinterpretCast<uint64_t>(), ext_twiddle, params, pad);
        }
        PipeBarrier<PIPE_ALL>();
        Gather(twiddle_real, twiddle, indices, 0, kTile);
        Gather(twiddle_imag, twiddle, indices[kTile], 0, kTile);
        Mul(tmp_real, digit_real, twiddle_real, kTile);
        Mul(tmp_imag, digit_imag, twiddle_imag, kTile);
        Sub(tmp_real, tmp_real, tmp_imag, kTile);
        Mul(tmp_imag, digit_real, twiddle_imag, kTile);
        Mul(tmp_aux, digit_imag, twiddle_real, kTile);
        Add(tmp_imag, tmp_imag, tmp_aux, kTile);
        PipeBarrier<PIPE_ALL>();
        DataCopy(digit_real, tmp_real, kTile);
        DataCopy(digit_imag, tmp_imag, kTile);
        PipeBarrier<PIPE_ALL>();
      }
    }

    LocalTensor<float> acc_real = work;
    LocalTensor<float> acc_imag = work[kTile];
    LocalTensor<float> store = store_buf_.Get<float>();
    const uint32_t output_stride = span_ == 1
        ? (2 * radix_ - 1) * sizeof(float)
        : sizeof(float);
    for (uint32_t q = 0; q < radix_; ++q) {
      Duplicate(acc_real, 0.0f, kTile);
      Duplicate(acc_imag, 0.0f, kTile);
      PipeBarrier<PIPE_ALL>();
      for (uint32_t digit = 0; digit < radix_; ++digit) {
        LocalTensor<float> digit_real = digits[digit * 2 * kTile];
        LocalTensor<float> digit_imag = digits[digit * 2 * kTile + kTile];
        if (q == 0) {
          Add(acc_real, acc_real, digit_real, kTile);
          Add(acc_imag, acc_imag, digit_imag, kTile);
        } else {
          const uint32_t coeff = 2 * (q * radix_ + digit);
          constexpr uint32_t kFloatBytes = static_cast<uint32_t>(sizeof(float));
          Duplicate(root_indices, coeff * kFloatBytes, kTile);
          Duplicate(root_indices[kTile], (coeff + 1) * kFloatBytes, kTile);
          PipeBarrier<PIPE_ALL>();
          Gather(twiddle_real, roots, root_indices, 0, kTile);
          Gather(twiddle_imag, roots, root_indices[kTile], 0, kTile);
          PipeBarrier<PIPE_ALL>();
          Mul(tmp_real, digit_real, twiddle_real, kTile);
          Mul(tmp_imag, digit_imag, twiddle_imag, kTile);
          Sub(tmp_real, tmp_real, tmp_imag, kTile);
          Mul(tmp_imag, digit_real, twiddle_imag, kTile);
          Mul(tmp_aux, digit_imag, twiddle_real, kTile);
          Add(tmp_imag, tmp_imag, tmp_aux, kTile);
          Add(acc_real, acc_real, tmp_real, kTile);
          Add(acc_imag, acc_imag, tmp_imag, kTile);
        }
      }
      const uint64_t output_base = batch_index * length_ +
          static_cast<uint64_t>(radix_) * k_start -
          static_cast<uint64_t>(radix_ - 1) * j_start +
          static_cast<uint64_t>(q) * span_;
      const DataCopyExtParams params(valid, sizeof(float), 0, output_stride, 0);
      // Brcb is supported for FP32 on the 910B AI Core. It places one scalar
      // at the start of each 32-byte VECOUT block for an MTE3 strided store.
      Brcb(store, acc_real, kTile / 8, {1, 8});
      PipeBarrier<PIPE_ALL>();
      GlobalTensor<float> dst_real;
      dst_real.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output_ptr_) +
                               2 * output_base);
      DataCopyPad(dst_real, store, params);
      PipeBarrier<PIPE_ALL>();

      Brcb(store, acc_imag, kTile / 8, {1, 8});
      PipeBarrier<PIPE_ALL>();
      GlobalTensor<float> dst_imag;
      dst_imag.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output_ptr_) +
                               2 * output_base + 1);
      DataCopyPad(dst_imag, store, params);
      PipeBarrier<PIPE_ALL>();
    }
  }

  TPipe pipe_;
  TBuf<TPosition::VECCALC> input_buf_;
  TBuf<TPosition::VECCALC> twiddle_buf_;
  TBuf<TPosition::VECCALC> digits_buf_;
  TBuf<TPosition::VECCALC> roots_buf_;
  TBuf<TPosition::VECCALC> indices_buf_;
  TBuf<TPosition::VECCALC> root_indices_buf_;
  TBuf<TPosition::VECCALC> work_buf_;
  TBuf<TPosition::VECCALC> store_buf_;
  __gm__ uint64_t *stage_twiddles_ptr_ = nullptr;
  GlobalTensor<float> radix_roots_;
  GlobalTensor<uint32_t> indices_;
  __gm__ uint64_t *input_ptr_ = nullptr;
  __gm__ uint64_t *output_ptr_ = nullptr;
  uint32_t length_ = 0;
  uint32_t radix_ = 0;
  uint32_t span_ = 0;
  uint32_t batch_ = 0;
  uint32_t ndiv_ = 0;
  uint32_t tiles_per_group_ = 0;
  uint32_t groups_per_transform_ = 0;
  uint32_t tiles_per_transform_ = 0;
  uint32_t root_float_count_ = 0;
};
}  // namespace

extern "C" __global__ __aicore__ void flagfft_npu_stockham_stage(GM_ADDR input,
                                                                  GM_ADDR output,
                                                                  GM_ADDR stage_twiddles,
                                                                  GM_ADDR radix_roots,
                                                                  GM_ADDR indices,
                                                                  uint32_t length,
                                                                  uint32_t radix,
                                                                  uint32_t span,
                                                                  uint32_t batch) {
  StockhamStageAiv kernel;
  kernel.Init(input, output, stage_twiddles, radix_roots, indices,
              length, radix, span, batch);
  kernel.Process();
}
