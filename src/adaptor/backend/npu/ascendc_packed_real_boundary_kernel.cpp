// Copyright 2026 FlagOS Contributors
// Licensed under the Apache License, Version 2.0.

#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t kTile = 256;

class PackedRealBoundaryAiv {
 public:
  __aicore__ inline void Init(GM_ADDR input,
                              GM_ADDR output,
                              GM_ADDR indices,
                              GM_ADDR twiddles,
                              uint32_t length,
                              uint32_t input_distance,
                              uint32_t output_distance,
                              uint32_t batch,
                              uint32_t r2c_postprocess) {
    input_ptr_ = reinterpret_cast<__gm__ float *>(input);
    output_ptr_ = reinterpret_cast<__gm__ float *>(output);
    indices_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(indices));
    twiddles_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(twiddles));
    length_ = length;
    packed_ = length / 2;
    half_ = packed_ + 1;
    padded_half_ = ((half_ + kTile - 1) / kTile) * kTile;
    input_distance_ = input_distance;
    output_distance_ = output_distance;
    batch_ = batch;
    r2c_postprocess_ = r2c_postprocess != 0;
    output_count_ = r2c_postprocess_ ? half_ : packed_;
    group_rows_ = 1;
    while ((output_count_ * group_rows_) % 4U != 0) ++group_rows_;
    staging_row_pitch_ = ((output_count_ + kTile - 1) / kTile) * kTile;
    compact_count_ = ((group_rows_ * output_count_ + kTile - 1) / kTile) * kTile;
    compact_index_base_ = 3 * padded_half_ + 2 * kTile;
    twiddle_count_ = ((2 * half_ + 7U) / 8U) * 8U;
    index_count_ = compact_index_base_ + 2 * compact_count_;
    pipe_.InitBuffer(input_buf_, ((2 * half_ + 7U) / 8U) * 8U * sizeof(float));
    pipe_.InitBuffer(twiddle_buf_, twiddle_count_ * sizeof(float));
    pipe_.InitBuffer(indices_buf_, index_count_ * sizeof(uint32_t));
    pipe_.InitBuffer(work_buf_, 14 * kTile * sizeof(float));
    pipe_.InitBuffer(merge_buf_, 2 * kTile * sizeof(float));
    pipe_.InitBuffer(output_buf_, 2 * kTile * sizeof(float));
    pipe_.InitBuffer(staging_buf_, 2 * group_rows_ * staging_row_pitch_ * sizeof(float));
  }

  __aicore__ inline void Process() {
    const uint32_t grid = GetBlockNum();
    if (grid == 0 || length_ < 128 || length_ > 2048 || (length_ & 1U) != 0) return;
    const uint32_t input_count = 2 * (r2c_postprocess_ ? packed_ : half_);
    const uint32_t output_count = output_count_;
    LocalTensor<float> input = input_buf_.Get<float>();
    LocalTensor<float> twiddle = twiddle_buf_.Get<float>();
    LocalTensor<uint32_t> indices = indices_buf_.Get<uint32_t>();
    DataCopy(indices, indices_, index_count_);
    DataCopy(twiddle, twiddles_, twiddle_count_);
    PipeBarrier<PIPE_ALL>();

    LocalTensor<float> work = work_buf_.Get<float>();
    LocalTensor<float> a_real = work[0 * kTile];
    LocalTensor<float> a_imag = work[1 * kTile];
    LocalTensor<float> b_real = work[2 * kTile];
    LocalTensor<float> b_imag = work[3 * kTile];
    LocalTensor<float> w_real = work[4 * kTile];
    LocalTensor<float> w_imag = work[5 * kTile];
    LocalTensor<float> sum_real = work[6 * kTile];
    LocalTensor<float> sum_imag = work[7 * kTile];
    LocalTensor<float> diff_real = work[8 * kTile];
    LocalTensor<float> diff_imag = work[9 * kTile];
    LocalTensor<float> product_real = work[10 * kTile];
    LocalTensor<float> product_imag = work[11 * kTile];
    LocalTensor<float> output_real = work[12 * kTile];
    LocalTensor<float> output_imag = work[13 * kTile];
    LocalTensor<float> merge = merge_buf_.Get<float>();
    LocalTensor<float> output_local = output_buf_.Get<float>();
    LocalTensor<float> staging = staging_buf_.Get<float>();

    const uint32_t row_groups = (batch_ + group_rows_ - 1) / group_rows_;
    for (uint32_t group = GetBlockIdx(); group < row_groups; group += grid) {
      const uint32_t first_row = group * group_rows_;
      const uint32_t valid_rows = batch_ - first_row < group_rows_
                                      ? batch_ - first_row
                                      : group_rows_;
      for (uint32_t row_slot = 0; row_slot < valid_rows; ++row_slot) {
        const uint32_t row = first_row + row_slot;
        GlobalTensor<float> src;
        src.SetGlobalBuffer(input_ptr_ + static_cast<uint64_t>(row) * input_distance_ * 2);
        const DataCopyExtParams input_params(1, input_count * sizeof(float), 0, 0, 0);
        const DataCopyPadExtParams<float> input_pad;
        DataCopyPad(input, src, input_params, input_pad);
        PipeBarrier<PIPE_ALL>();

        for (uint32_t first = 0; first < output_count; first += kTile) {
          const uint32_t valid = output_count - first < kTile ? output_count - first : kTile;
          const LocalTensor<uint32_t> index_a = indices[first];
          const LocalTensor<uint32_t> index_b = indices[padded_half_ + first];
          const LocalTensor<uint32_t> index_w = indices[2 * padded_half_ + first];
          Gather(a_real, input, index_a, 0, kTile);
          Gather(a_imag, input, index_a, sizeof(float), kTile);
          Gather(b_real, input, index_b, 0, kTile);
          Gather(b_imag, input, index_b, sizeof(float), kTile);
          Gather(w_real, twiddle, index_w, 0, kTile);
          Gather(w_imag, twiddle, index_w, sizeof(float), kTile);
          PipeBarrier<PIPE_V>();

          // Diagnostic: dump the gathered real and imaginary planes before
          // interleaving or compaction, so those stores can be checked alone.
          if (r2c_postprocess_) {
            if (first < packed_) {
              const uint32_t packed_valid =
                  packed_ - first < kTile ? packed_ - first : kTile;
              GlobalTensor<float> real_dst;
              real_dst.SetGlobalBuffer(output_ptr_ +
                                       static_cast<uint64_t>(row) * output_distance_ * 2 + first);
              const DataCopyExtParams real_params(
                  1, packed_valid * sizeof(float), 0, 0, 0);
              DataCopyPad(real_dst, a_real, real_params);
              GlobalTensor<float> imag_dst;
              imag_dst.SetGlobalBuffer(output_ptr_ +
                                       static_cast<uint64_t>(row) * output_distance_ * 2 +
                                           output_count_ + first);
              DataCopyPad(imag_dst, a_imag, real_params);
            }
            PipeBarrier<PIPE_ALL>();
            continue;
          }

          Muls(b_imag, b_imag, -1.0f, kTile);
          Add(sum_real, a_real, b_real, kTile);
          Add(sum_imag, a_imag, b_imag, kTile);
          Sub(diff_real, a_real, b_real, kTile);
          Sub(diff_imag, a_imag, b_imag, kTile);
          if (r2c_postprocess_) {
            Mul(product_real, diff_real, w_real, kTile);
            Mul(product_imag, diff_imag, w_imag, kTile);
            Sub(product_real, product_real, product_imag, kTile);
            Mul(product_imag, diff_imag, w_real, kTile);
            Mul(output_real, diff_real, w_imag, kTile);
            Add(product_imag, product_imag, output_real, kTile);
            PipeBarrier<PIPE_V>();
            Muls(a_real, sum_real, 0.5f, kTile);
            Muls(a_imag, sum_imag, 0.5f, kTile);
            Muls(b_real, product_imag, 0.5f, kTile);
            Muls(b_imag, product_real, 0.5f, kTile);
            PipeBarrier<PIPE_V>();
            Add(output_real, a_real, b_real, kTile);
            Sub(output_imag, a_imag, b_imag, kTile);
            PipeBarrier<PIPE_V>();
            if (first == 0) output_imag.SetValue(0, 0.0f);
            if (first <= packed_ && first + valid > packed_) {
              output_imag.SetValue(packed_ - first, 0.0f);
            }
          } else {
            Mul(product_real, diff_real, w_real, kTile);
            Mul(product_imag, diff_imag, w_imag, kTile);
            Add(product_real, product_real, product_imag, kTile);
            Mul(product_imag, diff_imag, w_real, kTile);
            Mul(output_real, diff_real, w_imag, kTile);
            Sub(product_imag, product_imag, output_real, kTile);
            Sub(output_real, sum_real, product_imag, kTile);
            Add(output_imag, sum_imag, product_real, kTile);
          }
          PipeBarrier<PIPE_ALL>();

          DataCopy(merge, output_real, kTile);
          DataCopy(merge[kTile], output_imag, kTile);
          PipeBarrier<PIPE_ALL>();
          Gather(output_local, merge, indices[3 * padded_half_], 0, 2 * kTile);
          PipeBarrier<PIPE_V>();

          LocalTensor<float> row_staging =
              staging[row_slot * staging_row_pitch_ * 2 + first * 2];
          DataCopy(row_staging, output_local, 2 * kTile);
          PipeBarrier<PIPE_ALL>();
        }
      }

      if (r2c_postprocess_) continue;

      const uint32_t flat_count = valid_rows * output_count;
      for (uint32_t first = 0; first < flat_count; first += kTile) {
        const uint32_t valid = flat_count - first < kTile ? flat_count - first : kTile;
        Gather(output_local, staging,
               indices[compact_index_base_ + 2 * first], 0, 2 * kTile);
        PipeBarrier<PIPE_V>();

        GlobalTensor<float> dst;
        dst.SetGlobalBuffer(output_ptr_ +
                            (static_cast<uint64_t>(first_row) * output_count + first) * 2);
        const DataCopyExtParams output_params(1, 2 * valid * sizeof(float), 0, 0, 0);
        DataCopyPad(dst, output_local, output_params);
        PipeBarrier<PIPE_ALL>();
      }
    }
  }

 private:
  TPipe pipe_;
  TBuf<TPosition::VECCALC> input_buf_;
  TBuf<TPosition::VECCALC> twiddle_buf_;
  TBuf<TPosition::VECCALC> indices_buf_;
  TBuf<TPosition::VECCALC> work_buf_;
  TBuf<TPosition::VECCALC> merge_buf_;
  TBuf<TPosition::VECCALC> output_buf_;
  TBuf<TPosition::VECCALC> staging_buf_;
  GlobalTensor<uint32_t> indices_;
  GlobalTensor<float> twiddles_;
  __gm__ float *input_ptr_ = nullptr;
  __gm__ float *output_ptr_ = nullptr;
  uint32_t length_ = 0;
  uint32_t packed_ = 0;
  uint32_t half_ = 0;
  uint32_t padded_half_ = 0;
  uint32_t output_count_ = 0;
  uint32_t group_rows_ = 1;
  uint32_t staging_row_pitch_ = 0;
  uint32_t compact_count_ = 0;
  uint32_t compact_index_base_ = 0;
  uint32_t input_distance_ = 0;
  uint32_t output_distance_ = 0;
  uint32_t batch_ = 0;
  uint32_t twiddle_count_ = 0;
  uint32_t index_count_ = 0;
  bool r2c_postprocess_ = false;
};
}  // namespace

extern "C" __global__ __aicore__ void flagfft_npu_packed_real_boundary(
    GM_ADDR input,
    GM_ADDR output,
    GM_ADDR indices,
    GM_ADDR twiddles,
    uint32_t length,
    uint32_t input_distance,
    uint32_t output_distance,
    uint32_t batch,
    uint32_t r2c_postprocess) {
  PackedRealBoundaryAiv kernel;
  kernel.Init(input, output, indices, twiddles, length, input_distance,
              output_distance, batch, r2c_postprocess);
  kernel.Process();
}
