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
constexpr uint32_t kTile = 16;
constexpr uint32_t kTileComplexCount = kTile * kTile;
constexpr uint32_t kTileFloatCount = 2 * kTileComplexCount;
constexpr uint32_t kTileDataBlocks = kTile * sizeof(uint64_t) / 32;
constexpr uint32_t kFloatIndicesPerTable = kTileFloatCount;

class Transpose2DAiv {
 public:
  __aicore__ inline void Init(GM_ADDR input,
                              GM_ADDR output,
                              GM_ADDR indices,
                              uint32_t rows,
                              uint32_t cols,
                              uint32_t batch) {
    input_ptr_ = reinterpret_cast<__gm__ uint64_t *>(input);
    output_ptr_ = reinterpret_cast<__gm__ uint64_t *>(output);
    indices_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(indices));
    rows_ = rows;
    cols_ = cols;
    batch_ = batch;
    row_tiles_ = (rows + kTile - 1) / kTile;
    col_tiles_ = (cols + kTile - 1) / kTile;
    tiles_per_batch_ = static_cast<uint64_t>(row_tiles_) * col_tiles_;
    total_tiles_ = static_cast<uint64_t>(batch_) * tiles_per_batch_;

    pipe_.InitBuffer(input_buf_, kTileFloatCount * sizeof(float));
    pipe_.InitBuffer(output_buf_, kTileFloatCount * sizeof(float));
    pipe_.InitBuffer(index_buf_, kTileFloatCount * sizeof(uint32_t));
  }

  __aicore__ inline void Process() {
    const uint64_t block_count = GetBlockNum();
    for (uint64_t tile_id = GetBlockIdx(); tile_id < total_tiles_; tile_id += block_count) {
      ProcessTile(tile_id);
    }
  }

 private:
  __aicore__ inline void ProcessTile(uint64_t tile_id) {
    const uint32_t batch_index = static_cast<uint32_t>(tile_id / tiles_per_batch_);
    uint32_t tile_in_batch = static_cast<uint32_t>(tile_id % tiles_per_batch_);
    const uint32_t tile_col = tile_in_batch % col_tiles_;
    const uint32_t tile_row = tile_in_batch / col_tiles_;
    const uint32_t row_start = tile_row * kTile;
    const uint32_t col_start = tile_col * kTile;
    const uint32_t valid_rows = (rows_ - row_start < kTile) ? rows_ - row_start : kTile;
    uint32_t valid_cols = (cols_ - col_start < kTile) ? cols_ - col_start : kTile;
    uint32_t source_col_start = col_start;
    uint32_t source_cols = valid_cols;
    uint32_t tail_shift = 0;

    // Ascend GM transfers are most efficient in 32-byte units. For a short
    // final column tile, read an aligned window and rotate the gather indices
    // so the written edge still starts at the original column.
    if (valid_cols < 4 && cols_ >= 4) {
      const uint32_t aligned_cols = ((valid_cols + 3) / 4) * 4;
      const uint32_t aligned_start = cols_ - aligned_cols;
      if (col_start >= aligned_start) {
        tail_shift = col_start - aligned_start;
        source_col_start = aligned_start;
        source_cols = aligned_cols;
      }
    }

    LocalTensor<uint64_t> input_complex = input_buf_.Get<float>().ReinterpretCast<uint64_t>();
    GlobalTensor<uint64_t> input_global;
    const uint64_t input_offset =
        (static_cast<uint64_t>(batch_index) * rows_ + row_start) * cols_ + source_col_start;
    input_global.SetGlobalBuffer(input_ptr_ + input_offset);
    const uint32_t source_block_bytes = source_cols * sizeof(uint64_t);
    const DataCopyExtParams input_params(valid_rows,
                                         source_block_bytes,
                                         (cols_ - source_cols) * sizeof(uint64_t),
                                         kTileDataBlocks - (source_cols + 3) / 4,
                                         0);
    const DataCopyPadExtParams<uint64_t> input_pad;
    DataCopyPad(input_complex, input_global, input_params, input_pad);

    LocalTensor<uint32_t> index_local = index_buf_.Get<uint32_t>();
    DataCopy(index_local, indices_[tail_shift * kFloatIndicesPerTable], kTileFloatCount);
    PipeBarrier<PIPE_ALL>();

    LocalTensor<float> input_local = input_buf_.Get<float>();
    LocalTensor<float> output_local = output_buf_.Get<float>();
    Gather(output_local, input_local, index_local, 0, kTileFloatCount);
    PipeBarrier<PIPE_ALL>();

    LocalTensor<uint64_t> output_complex = output_local.ReinterpretCast<uint64_t>();
    for (uint32_t output_row = 0; output_row < valid_cols; ++output_row) {
      const uint64_t output_offset =
          (static_cast<uint64_t>(batch_index) * cols_ + col_start + output_row) * rows_ + row_start;
      GlobalTensor<uint64_t> output_global;
      output_global.SetGlobalBuffer(output_ptr_ + output_offset);
      const DataCopyExtParams output_params(valid_cols,
                                            valid_rows * sizeof(uint64_t),
                                            kTileDataBlocks - (valid_rows + 3) / 4,
                                            (rows_ - valid_rows) * sizeof(uint64_t),
                                            0);
      DataCopyPad(output_global,
                  output_complex[output_row * kTile],
                  output_params);
    }
    // The same UB buffers are reused by the next grid-stride tile.
    PipeBarrier<PIPE_ALL>();
  }

  TPipe pipe_;
  TBuf<TPosition::VECCALC> input_buf_;
  TBuf<TPosition::VECCALC> output_buf_;
  TBuf<TPosition::VECCALC> index_buf_;
  GlobalTensor<uint32_t> indices_;
  __gm__ uint64_t *input_ptr_ = nullptr;
  __gm__ uint64_t *output_ptr_ = nullptr;
  uint32_t rows_ = 0;
  uint32_t cols_ = 0;
  uint32_t batch_ = 0;
  uint32_t row_tiles_ = 0;
  uint32_t col_tiles_ = 0;
  uint64_t tiles_per_batch_ = 0;
  uint64_t total_tiles_ = 0;
};
}  // namespace

extern "C" __global__ __aicore__ void flagfft_npu_transpose2d(GM_ADDR input,
                                                               GM_ADDR output,
                                                               GM_ADDR indices,
                                                               uint32_t rows,
                                                               uint32_t cols,
                                                               uint32_t batch) {
  Transpose2DAiv kernel;
  kernel.Init(input, output, indices, rows, cols, batch);
  kernel.Process();
}
