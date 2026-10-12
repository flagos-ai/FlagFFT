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
constexpr uint32_t kTile = 128;
constexpr uint32_t kVectorRepeatElements = 64;
constexpr uint32_t kMaxRadix = 19;
constexpr uint32_t kRootStride = 24;
constexpr uint32_t kMaxRootFloats = 2 * kMaxRadix * kRootStride;
constexpr uint32_t kBroadcastScratchBytes = 16384;

class StockhamStageAiv {
 public:
  __aicore__ inline void Init(GM_ADDR input,
                              GM_ADDR output,
                              GM_ADDR stage_twiddles,
                              GM_ADDR radix_roots,
                              GM_ADDR index_table,
                              GM_ADDR store_index_table,
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
    // Keep this mapping in sync with the host's short-extent tile count.
    batch_lane_extent_ = span_ == 1 ? ndiv_ : (span_ < kTile ? span_ : kTile);
    // Keep short-extent grouping aligned with the host helper: complex64 is
    // 8 bytes, so each DataCopyPad local destination must start every 4 lanes.
    batch_group_ = batch_lane_extent_ > 0 && batch_lane_extent_ < kTile &&
                           batch_lane_extent_ % 4U == 0
                       ? kTile / batch_lane_extent_
                       : 1;
    // Pack adjacent independent groups when a short non-aligned span cannot
    // use the batch-group path without violating DataCopyPad alignment.
    packed_groups_per_tile_ = span_ > 1 && span_ < kTile && batch_group_ == 1
                                  ? kTile / span_
                                  : 1;
    tiles_per_transform_ = span_ == 1
        ? (ndiv_ + kTile - 1) / kTile
        : ((groups_per_transform_ + packed_groups_per_tile_ - 1) /
           packed_groups_per_tile_) * tiles_per_group_;
    batch_groups_ = (static_cast<uint64_t>(batch_) + batch_group_ - 1) /
                    batch_group_;
    twiddle_stride_ = packed_groups_per_tile_ > 1 ? kTile : span_;
    root_float_count_ = 2 * radix_ * kRootStride;
    has_contiguous_store_ = store_index_table != nullptr && span_ <= kTile &&
                            packed_groups_per_tile_ == 1;
    if (has_contiguous_store_) {
      store_indices_.SetGlobalBuffer(
          reinterpret_cast<__gm__ uint32_t *>(store_index_table));
      const uint32_t store_lanes = span_ == 1
          ? (ndiv_ < kTile ? ndiv_ : kTile)
          : batch_lane_extent_;
      store_index_stride_ = (2 * radix_ * store_lanes + 7U) & ~7U;
    }

    pipe_.InitBuffer(input_buf_, 2 * kTile * sizeof(float));
    pipe_.InitBuffer(twiddle_buf_, 2 * kTile * sizeof(float));
    pipe_.InitBuffer(digits_buf_, 2 * kMaxRadix * kTile * sizeof(float));
    pipe_.InitBuffer(roots_buf_, kMaxRootFloats * sizeof(float));
    pipe_.InitBuffer(indices_buf_, 5 * kTile * sizeof(uint32_t));
    pipe_.InitBuffer(root_indices_buf_, kTile * sizeof(uint32_t));
    pipe_.InitBuffer(work_buf_, 7 * kTile * sizeof(float));
    pipe_.InitBuffer(planar_buf_, 2 * kTile * sizeof(float));
    pipe_.InitBuffer(packed_buf_, 2 * kTile * sizeof(float));
    const uint32_t store_buffer_floats = has_contiguous_store_
        ? (store_index_stride_ > 8 * kTile ? store_index_stride_ : 8 * kTile)
        : 8 * kTile;
    pipe_.InitBuffer(store_buf_, store_buffer_floats * sizeof(float));
    if (has_contiguous_store_) {
      pipe_.InitBuffer(store_indices_buf_,
                       static_cast<uint64_t>(store_index_stride_) * batch_group_ *
                           sizeof(uint32_t));
    }
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
    if (has_contiguous_store_) {
      LocalTensor<uint32_t> store_indices = store_indices_buf_.Get<uint32_t>();
      DataCopy(store_indices, store_indices_, store_index_stride_ * batch_group_);
    }
    PipeBarrier<PIPE_ALL>();
  }

  __aicore__ inline void Process() {
    const uint64_t total_tiles = batch_groups_ * tiles_per_transform_;
    const uint64_t block_count = GetBlockNum();
    if (block_count == 0) return;
    for (uint64_t tile_id = GetBlockIdx(); tile_id < total_tiles; tile_id += block_count) {
      ProcessTile(tile_id);
    }
  }

 private:
  __aicore__ inline void ProcessTile(uint64_t tile_id) {
    const uint64_t batch_group_index = tile_id / tiles_per_transform_;
    const uint64_t local_tile = tile_id % tiles_per_transform_;
    const uint64_t batch_index = batch_group_index * batch_group_;
    const uint32_t grouped_batch_count = static_cast<uint32_t>(
        batch_ - batch_index < batch_group_ ? batch_ - batch_index : batch_group_);
    uint64_t k_start = 0;
    uint32_t j_start = 0;
    uint32_t valid = 0;
    uint32_t valid_per_group = 0;
    uint32_t groups_in_tile = 1;
    if (span_ == 1) {
      k_start = local_tile * kTile;
      valid = static_cast<uint32_t>(ndiv_ - k_start < kTile ? ndiv_ - k_start : kTile);
      valid_per_group = valid;
    } else {
      const uint64_t group_tile = local_tile / tiles_per_group_;
      const uint64_t tile_in_group = local_tile % tiles_per_group_;
      j_start = static_cast<uint32_t>(tile_in_group * kTile);
      const uint64_t group = group_tile * packed_groups_per_tile_;
      k_start = group * span_ + j_start;
      valid_per_group = span_ - j_start < kTile ? span_ - j_start : kTile;
      groups_in_tile = static_cast<uint32_t>(
          groups_per_transform_ - group < packed_groups_per_tile_
              ? groups_per_transform_ - group
              : packed_groups_per_tile_);
      valid = packed_groups_per_tile_ > 1
                  ? groups_in_tile * valid_per_group
                  : valid_per_group;
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
      for (uint32_t transform = 0; transform < grouped_batch_count; ++transform) {
        const uint64_t input_offset = (batch_index + transform) * length_ + k_start +
                                      static_cast<uint64_t>(digit) * ndiv_;
        GlobalTensor<uint64_t> src;
        src.SetGlobalBuffer(input_ptr_ + input_offset);
        LocalTensor<uint64_t> input_dest =
            input_complex[transform * batch_lane_extent_];
        if (valid == kTile && batch_group_ == 1) {
          DataCopy(input_dest, src, kTile);
        } else {
          const DataCopyExtParams params(1, valid * sizeof(uint64_t), 0, 0, 0);
          const DataCopyPadExtParams<uint64_t> pad;
          DataCopyPad(input_dest, src, params, pad);
        }
      }
      PipeBarrier<PIPE_ALL>();

      LocalTensor<float> digit_real = digits[digit * 2 * kTile];
      LocalTensor<float> digit_imag = digits[digit * 2 * kTile + kTile];
      Gather(digit_real, input_local, indices, 0, kTile);
      Gather(digit_imag, input_local, indices[kTile], 0, kTile);
      PipeBarrier<PIPE_V>();

      if (digit > 0 && span_ > 1) {
        Duplicate(twiddle, 0.0f, 2 * kTile);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t transform = 0; transform < grouped_batch_count; ++transform) {
          GlobalTensor<uint64_t> ext_twiddle;
          ext_twiddle.SetGlobalBuffer(stage_twiddles_ptr_ +
                                      static_cast<uint64_t>(digit) * twiddle_stride_ + j_start);
          LocalTensor<uint64_t> twiddle_dest =
              twiddle.ReinterpretCast<uint64_t>()[
                  transform * batch_lane_extent_];
          if (valid == kTile && batch_group_ == 1) {
            DataCopy(twiddle_dest, ext_twiddle, kTile);
          } else {
            const DataCopyExtParams params(1, valid * sizeof(uint64_t), 0, 0, 0);
            const DataCopyPadExtParams<uint64_t> pad;
            DataCopyPad(twiddle_dest, ext_twiddle, params, pad);
          }
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
    LocalTensor<int32_t> root_indices_i = root_indices.ReinterpretCast<int32_t>();
    LocalTensor<int32_t> q_root_offsets_i =
        q_root_offsets.ReinterpretCast<int32_t>();
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
        for (uint32_t lane_offset = 0; lane_offset < kTile;
             lane_offset += kVectorRepeatElements) {
          Add(q_acc_real[lane_offset], q_acc_real[lane_offset],
              q_input_real[lane_offset], kVectorRepeatElements, radix_,
              q_repeat_params);
          Add(q_acc_imag[lane_offset], q_acc_imag[lane_offset],
              q_input_imag[lane_offset], kVectorRepeatElements, radix_,
              q_repeat_params);
        }
        PipeBarrier<PIPE_V>();
        continue;
      }

      Adds(root_indices_i, q_root_offsets_i,
           static_cast<int32_t>(digit * sizeof(float)), kTile);
      Gather(q_root_real, roots, root_indices, 0, kTile);
      Adds(root_indices_i, q_root_offsets_i,
           static_cast<int32_t>((radix_ * kRootStride + digit) *
                                sizeof(float)),
           kTile);
      Gather(q_root_imag, roots, root_indices, 0, kTile);
      PipeBarrier<PIPE_V>();
      Broadcast<float, 2, 1>(q_coeff_real, q_root_real, q_shape, root_shape,
                             broadcast_scratch);
      Broadcast<float, 2, 1>(q_coeff_imag, q_root_imag, q_shape, root_shape,
                             broadcast_scratch);
      for (uint32_t lane_offset = 0; lane_offset < kTile;
           lane_offset += kVectorRepeatElements) {
        Mul(q_tmp_real[lane_offset], q_input_real[lane_offset],
            q_coeff_real[lane_offset], kVectorRepeatElements, radix_,
            q_repeat_params);
        Mul(q_tmp_imag[lane_offset], q_input_imag[lane_offset],
            q_coeff_imag[lane_offset], kVectorRepeatElements, radix_,
            q_repeat_params);
        Sub(q_tmp_real[lane_offset], q_tmp_real[lane_offset],
            q_tmp_imag[lane_offset], kVectorRepeatElements, radix_,
            q_repeat_params);
        Mul(q_tmp_imag[lane_offset], q_input_real[lane_offset],
            q_coeff_imag[lane_offset], kVectorRepeatElements, radix_,
            q_repeat_params);
        Mul(q_tmp_aux[lane_offset], q_input_imag[lane_offset],
            q_coeff_real[lane_offset], kVectorRepeatElements, radix_,
            q_repeat_params);
        Add(q_tmp_imag[lane_offset], q_tmp_imag[lane_offset],
            q_tmp_aux[lane_offset], kVectorRepeatElements, radix_,
            q_repeat_params);
        Add(q_acc_real[lane_offset], q_acc_real[lane_offset],
            q_tmp_real[lane_offset], kVectorRepeatElements, radix_,
            q_repeat_params);
        Add(q_acc_imag[lane_offset], q_acc_imag[lane_offset],
            q_tmp_imag[lane_offset], kVectorRepeatElements, radix_,
            q_repeat_params);
      }
      PipeBarrier<PIPE_V>();
    }

    const uint32_t output_complex_count = radix_ * valid;
    bool use_contiguous_store = has_contiguous_store_ &&
                                (output_complex_count % 4U) == 0;
    // A contiguous GM transfer is only safe when both ends satisfy the
    // 32-byte DataCopy granularity. Irregular tails and batch boundaries use
    // the existing strided DataCopyPad path below.
    for (uint32_t transform = 0;
         use_contiguous_store && transform < grouped_batch_count;
         ++transform) {
      const uint64_t output_base =
          (batch_index + transform) * length_ +
          static_cast<uint64_t>(radix_) * k_start -
          static_cast<uint64_t>(radix_ - 1) * j_start;
      use_contiguous_store = (output_base % 4U) == 0;
    }

    if (use_contiguous_store) {
      LocalTensor<float> store = store_buf_.Get<float>();
      LocalTensor<uint32_t> store_indices = store_indices_buf_.Get<uint32_t>();
      LocalTensor<float> q_acc = q_acc_buf_.Get<float>();
      const uint32_t table_stride = store_index_stride_;
      // Repack the full radix tile once and issue one contiguous GM write.
      for (uint32_t transform = 0; transform < grouped_batch_count; ++transform) {
        const uint32_t index_offset = transform * table_stride;
        Gather(store, q_acc, store_indices[index_offset], 0, table_stride);
        PipeBarrier<PIPE_ALL>();

        const uint64_t output_base = (batch_index + transform) * length_ +
            static_cast<uint64_t>(radix_) * k_start -
            static_cast<uint64_t>(radix_ - 1) * j_start;
        GlobalTensor<uint64_t> dst;
        dst.SetGlobalBuffer(output_ptr_ + output_base);
        DataCopy(dst, store.ReinterpretCast<uint64_t>(), output_complex_count);
        PipeBarrier<PIPE_ALL>();
      }
      return;
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
      PipeBarrier<PIPE_V>();

      // MTE3 reads one 32-byte VECOUT block per transfer block even when the
      // GM block length is one complex value. Stage each packed value at the
      // start of a padded local block before the strided scatter.
      for (uint32_t transform = 0; transform < grouped_batch_count; ++transform) {
        const uint32_t groups_to_store = packed_groups_per_tile_ > 1
                                             ? groups_in_tile
                                             : 1;
        for (uint32_t group_slot = 0; group_slot < groups_to_store; ++group_slot) {
          const uint64_t output_base = (batch_index + transform) * length_ +
              static_cast<uint64_t>(radix_) * k_start -
              static_cast<uint64_t>(radix_ - 1) * j_start +
              (packed_groups_per_tile_ > 1
                   ? static_cast<uint64_t>(group_slot) * radix_ * span_
                   : 0) +
              static_cast<uint64_t>(q) * span_;
          const uint32_t group_valid = packed_groups_per_tile_ > 1
                                           ? valid_per_group
                                           : valid;
          const uint32_t lane_offset = transform * batch_lane_extent_ +
              (packed_groups_per_tile_ > 1 ? group_slot * span_ : 0);
          if (packed_groups_per_tile_ == 1 && span_ > 1 &&
              grouped_batch_count == 1 && group_valid == kTile &&
              (output_base % 4U) == 0) {
            // A full lane tile is already interleaved in `packed`; when its GM
            // destination is 32-byte aligned, write the contiguous segment
            // directly and avoid scalar SetValue packing into padded slots.
            GlobalTensor<uint64_t> dst;
            dst.SetGlobalBuffer(output_ptr_ + output_base);
            DataCopy(dst, packed.ReinterpretCast<uint64_t>()[lane_offset],
                     group_valid);
            PipeBarrier<PIPE_ALL>();
            continue;
          }

          Duplicate(store, 0.0f, 8 * kTile);
          PipeBarrier<PIPE_ALL>();
          for (uint32_t lane = 0; lane < group_valid; ++lane) {
            store.SetValue(8 * lane, packed.GetValue(2 * (lane_offset + lane)));
            store.SetValue(8 * lane + 1,
                           packed.GetValue(2 * (lane_offset + lane) + 1));
          }
          PipeBarrier<PIPE_ALL>();

          GlobalTensor<uint64_t> dst;
          dst.SetGlobalBuffer(output_ptr_ + output_base);
          const uint32_t output_stride = span_ == 1 ? dst_stride : 0;
          const DataCopyExtParams params(group_valid,
                                         sizeof(uint64_t),
                                         0,
                                         output_stride,
                                         0);
          DataCopyPad(dst, store_complex, params);
          PipeBarrier<PIPE_ALL>();
        }
      }
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
  TBuf<TPosition::VECCALC> store_indices_buf_;
  TBuf<TPosition::VECCALC> q_input_buf_;
  TBuf<TPosition::VECCALC> q_coeff_buf_;
  TBuf<TPosition::VECCALC> q_acc_buf_;
  TBuf<TPosition::VECCALC> q_tmp_buf_;
  TBuf<TPosition::VECCALC> q_root_buf_;
  TBuf<TPosition::VECCALC> broadcast_scratch_buf_;
  __gm__ uint64_t *stage_twiddles_ptr_ = nullptr;
  GlobalTensor<float> radix_roots_;
  GlobalTensor<uint32_t> indices_;
  GlobalTensor<uint32_t> store_indices_;
  __gm__ uint64_t *input_ptr_ = nullptr;
  __gm__ uint64_t *output_ptr_ = nullptr;
  uint32_t length_ = 0;
  uint32_t radix_ = 0;
  uint32_t span_ = 0;
  uint32_t batch_ = 0;
  uint32_t ndiv_ = 0;
  uint32_t tiles_per_group_ = 0;
  uint32_t groups_per_transform_ = 0;
  uint32_t packed_groups_per_tile_ = 1;
  uint32_t tiles_per_transform_ = 0;
  uint32_t batch_lane_extent_ = 0;
  uint32_t batch_group_ = 1;
  uint64_t batch_groups_ = 0;
  uint32_t root_float_count_ = 0;
  uint32_t store_index_stride_ = 0;
  uint32_t twiddle_stride_ = 0;
  bool has_contiguous_store_ = false;
};
}  // namespace

extern "C" __global__ __aicore__ void flagfft_npu_stockham_stage(GM_ADDR input,
                                                                  GM_ADDR output,
                                                                  GM_ADDR stage_twiddles,
                                                                  GM_ADDR radix_roots,
                                                                  GM_ADDR indices,
                                                                  GM_ADDR store_indices,
                                                                  uint32_t length,
                                                                  uint32_t radix,
                                                                  uint32_t span,
                                                                  uint32_t batch) {
  StockhamStageAiv kernel;
  kernel.Init(input, output, stage_twiddles, radix_roots, indices, store_indices,
              length, radix, span, batch);
  kernel.Process();
}
