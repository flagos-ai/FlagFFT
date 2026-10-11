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
#include "adv_api/broadcast/broadcast.h"

using namespace AscendC;

namespace {
constexpr uint32_t kTile = 32;
constexpr uint32_t kMaxRadix = 19;
constexpr uint32_t kRootStride = 24;
constexpr uint32_t kMaxRootFloats = 2 * kMaxRadix * kRootStride;
constexpr uint32_t kBroadcastScratchBytes = 2048;

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
    root_float_count_ = 2 * radix_ * kRootStride;

    pipe_.InitBuffer(input_buf_, 2 * kTile * sizeof(float));
    pipe_.InitBuffer(twiddle_buf_, 2 * kTile * sizeof(float));
    pipe_.InitBuffer(digits_buf_, 2 * kMaxRadix * kTile * sizeof(float));
    pipe_.InitBuffer(roots_buf_, kMaxRootFloats * sizeof(float));
    pipe_.InitBuffer(indices_buf_, 5 * kTile * sizeof(uint32_t));
    pipe_.InitBuffer(root_indices_buf_, kTile * sizeof(uint32_t));
    pipe_.InitBuffer(work_buf_, 7 * kTile * sizeof(float));
    pipe_.InitBuffer(planar_buf_, 2 * kTile * sizeof(float));
    pipe_.InitBuffer(packed_buf_, 2 * kTile * sizeof(float));
    pipe_.InitBuffer(store_buf_, 8 * kTile * sizeof(float));
    pipe_.InitBuffer(q_input_buf_, 2 * kMaxRadix * kTile * sizeof(float));
    pipe_.InitBuffer(q_coeff_buf_, 2 * kMaxRadix * kTile * sizeof(float));
    pipe_.InitBuffer(q_acc_buf_, 2 * kMaxRadix * kTile * sizeof(float));
    pipe_.InitBuffer(q_tmp_buf_, 3 * kMaxRadix * kTile * sizeof(float));
    pipe_.InitBuffer(q_root_buf_, 2 * kTile * sizeof(float));
    pipe_.InitBuffer(broadcast_scratch_buf_, kBroadcastScratchBytes);

    LocalTensor<float> roots = roots_buf_.Get<float>();
    DataCopy(roots, radix_roots_, root_float_count_);
    LocalTensor<uint32_t> indices = indices_buf_.Get<uint32_t>();
    DataCopy(indices, indices_, 5 * kTile);
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

    LocalTensor<float> q_input_real = q_input_buf_.Get<float>();
    LocalTensor<float> q_input_imag = q_input_real[radix_ * kTile];
    LocalTensor<float> q_coeff_real = q_coeff_buf_.Get<float>();
    LocalTensor<float> q_coeff_imag = q_coeff_real[radix_ * kTile];
    LocalTensor<float> q_acc_real = q_acc_buf_.Get<float>();
    LocalTensor<float> q_acc_imag = q_acc_real[radix_ * kTile];
    LocalTensor<float> q_tmp_real = q_tmp_buf_.Get<float>();
    LocalTensor<float> q_tmp_imag = q_tmp_real[radix_ * kTile];
    LocalTensor<float> q_tmp_aux = q_tmp_imag[radix_ * kTile];
    LocalTensor<float> q_root_real = q_root_buf_.Get<float>();
    LocalTensor<float> q_root_imag = q_root_real[kTile];
    LocalTensor<uint32_t> q_root_offsets = indices[4 * kTile];
    LocalTensor<uint8_t> broadcast_scratch = broadcast_scratch_buf_.Get<uint8_t>();
    const uint32_t q_shape[2] = {radix_, kTile};
    const uint32_t input_shape[2] = {1, kTile};
    const uint32_t root_shape[2] = {radix_, 1};
    const BinaryRepeatParams q_repeat_params(
        1, 1, 1, kTile / 8, kTile / 8, kTile / 8);
    Duplicate(q_acc_real, 0.0f, radix_ * kTile);
    Duplicate(q_acc_imag, 0.0f, radix_ * kTile);
    PipeBarrier<PIPE_ALL>();

    for (uint32_t digit = 0; digit < radix_; ++digit) {
      LocalTensor<float> digit_real = digits[digit * 2 * kTile];
      LocalTensor<float> digit_imag = digits[digit * 2 * kTile + kTile];
      Broadcast<float, 2, 0>(q_input_real, digit_real, q_shape, input_shape,
                             broadcast_scratch);
      Broadcast<float, 2, 0>(q_input_imag, digit_imag, q_shape, input_shape,
                             broadcast_scratch);
      if (digit == 0) {
        Add(q_acc_real, q_acc_real, q_input_real, kTile, radix_,
            q_repeat_params);
        Add(q_acc_imag, q_acc_imag, q_input_imag, kTile, radix_,
            q_repeat_params);
        PipeBarrier<PIPE_ALL>();
        continue;
      }

      Duplicate(root_indices,
                digit * static_cast<uint32_t>(sizeof(float)), kTile);
      Add(root_indices, root_indices, q_root_offsets, kTile);
      Gather(q_root_real, roots, root_indices, 0, kTile);
      Duplicate(root_indices,
                (radix_ * kRootStride + digit) *
                    static_cast<uint32_t>(sizeof(float)),
                kTile);
      Add(root_indices, root_indices, q_root_offsets, kTile);
      Gather(q_root_imag, roots, root_indices, 0, kTile);
      PipeBarrier<PIPE_ALL>();
      Broadcast<float, 2, 1>(q_coeff_real, q_root_real, q_shape, root_shape,
                             broadcast_scratch);
      Broadcast<float, 2, 1>(q_coeff_imag, q_root_imag, q_shape, root_shape,
                             broadcast_scratch);
      Mul(q_tmp_real, q_input_real, q_coeff_real, kTile, radix_,
          q_repeat_params);
      Mul(q_tmp_imag, q_input_imag, q_coeff_imag, kTile, radix_,
          q_repeat_params);
      Sub(q_tmp_real, q_tmp_real, q_tmp_imag, kTile, radix_,
          q_repeat_params);
      Mul(q_tmp_imag, q_input_real, q_coeff_imag, kTile, radix_,
          q_repeat_params);
      Mul(q_tmp_aux, q_input_imag, q_coeff_real, kTile, radix_,
          q_repeat_params);
      Add(q_tmp_imag, q_tmp_imag, q_tmp_aux, kTile, radix_,
          q_repeat_params);
      Add(q_acc_real, q_acc_real, q_tmp_real, kTile, radix_,
          q_repeat_params);
      Add(q_acc_imag, q_acc_imag, q_tmp_imag, kTile, radix_,
          q_repeat_params);
      PipeBarrier<PIPE_ALL>();
    }

    LocalTensor<float> planar = planar_buf_.Get<float>();
    LocalTensor<float> packed = packed_buf_.Get<float>();
    LocalTensor<float> store = store_buf_.Get<float>();
    LocalTensor<uint64_t> store_complex = store.ReinterpretCast<uint64_t>();
    const uint32_t dst_stride = (radix_ - 1) * sizeof(uint64_t);
    for (uint32_t q = 0; q < radix_; ++q) {
      LocalTensor<float> acc_real = q_acc_real[q * kTile];
      LocalTensor<float> acc_imag = q_acc_imag[q * kTile];
      PipeBarrier<PIPE_ALL>();
      DataCopy(planar, acc_real, kTile);
      DataCopy(planar[kTile], acc_imag, kTile);
      PipeBarrier<PIPE_ALL>();
      Gather(packed, planar, indices[2 * kTile], 0, 2 * kTile);
      PipeBarrier<PIPE_ALL>();

      // MTE3 reads one 32-byte VECOUT block per transfer block even when the
      // GM block length is one complex value. Stage each packed value at the
      // start of a padded local block before the strided scatter.
      Duplicate(store, 0.0f, 8 * kTile);
      PipeBarrier<PIPE_ALL>();
      for (uint32_t lane = 0; lane < valid; ++lane) {
        store.SetValue(8 * lane, packed.GetValue(2 * lane));
        store.SetValue(8 * lane + 1, packed.GetValue(2 * lane + 1));
      }
      PipeBarrier<PIPE_ALL>();

      const uint64_t output_base = batch_index * length_ +
          static_cast<uint64_t>(radix_) * k_start -
          static_cast<uint64_t>(radix_ - 1) * j_start +
          static_cast<uint64_t>(q) * span_;
      GlobalTensor<uint64_t> dst;
      dst.SetGlobalBuffer(output_ptr_ + output_base);
      const uint32_t output_stride = span_ == 1 ? dst_stride : 0;
      const DataCopyExtParams params(valid,
                                     sizeof(uint64_t),
                                     0,
                                     output_stride,
                                     0);
      DataCopyPad(dst, store_complex, params);
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
  TBuf<TPosition::VECCALC> planar_buf_;
  TBuf<TPosition::VECCALC> packed_buf_;
  TBuf<TPosition::VECCALC> store_buf_;
  TBuf<TPosition::VECCALC> q_input_buf_;
  TBuf<TPosition::VECCALC> q_coeff_buf_;
  TBuf<TPosition::VECCALC> q_acc_buf_;
  TBuf<TPosition::VECCALC> q_tmp_buf_;
  TBuf<TPosition::VECCALC> q_root_buf_;
  TBuf<TPosition::VECCALC> broadcast_scratch_buf_;
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
