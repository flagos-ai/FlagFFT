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
constexpr uint32_t kTileComplexCount = kTile * kTile * kTile;
constexpr uint32_t kTileFloatCount = 2 * kTileComplexCount;

class Transpose3DAiv {
 public:
  __aicore__ inline void Init(GM_ADDR input,
                              GM_ADDR output,
                              GM_ADDR indices,
                              uint32_t n0,
                              uint32_t n1,
                              uint32_t n2,
                              uint32_t batch,
                              uint32_t axis0,
                              uint32_t axis1,
                              uint32_t axis2) {
    input_ptr_ = reinterpret_cast<__gm__ float *>(input);
    output_ptr_ = reinterpret_cast<__gm__ float *>(output);
    indices_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(indices));
    dims_[0] = n0;
    dims_[1] = n1;
    dims_[2] = n2;
    axes_[0] = axis0;
    axes_[1] = axis1;
    axes_[2] = axis2;
    batch_ = batch;
    out_dims_[0] = dims_[axes_[0]];
    out_dims_[1] = dims_[axes_[1]];
    out_dims_[2] = dims_[axes_[2]];
    tile_counts_[0] = (out_dims_[0] + kTile - 1) / kTile;
    tile_counts_[1] = (out_dims_[1] + kTile - 1) / kTile;
    tile_counts_[2] = (out_dims_[2] + kTile - 1) / kTile;
    volume_ = static_cast<uint64_t>(n0) * n1 * n2;

    pipe_.InitBuffer(input_buf_, kTileFloatCount * sizeof(float));
    pipe_.InitBuffer(output_buf_, kTileFloatCount * sizeof(float));
    pipe_.InitBuffer(index_buf_, kTileFloatCount * sizeof(uint32_t));
  }

  __aicore__ inline void Process() {
    const uint32_t block = GetBlockIdx();
    const uint32_t tiles_per_batch = tile_counts_[0] * tile_counts_[1] * tile_counts_[2];
    const uint32_t batch_index = block / tiles_per_batch;
    if (batch_index >= batch_) return;
    uint32_t tile = block % tiles_per_batch;
    const uint32_t tile2 = tile % tile_counts_[2];
    tile /= tile_counts_[2];
    const uint32_t tile1 = tile % tile_counts_[1];
    const uint32_t tile0 = tile / tile_counts_[1];

    const uint32_t out_start[3] = {tile0 * kTile, tile1 * kTile, tile2 * kTile};
    const uint32_t remaining0 = out_dims_[0] - out_start[0];
    const uint32_t remaining1 = out_dims_[1] - out_start[1];
    const uint32_t remaining2 = out_dims_[2] - out_start[2];
    const uint32_t out_valid[3] = {
        remaining0 < kTile ? remaining0 : kTile,
        remaining1 < kTile ? remaining1 : kTile,
        remaining2 < kTile ? remaining2 : kTile,
    };
    uint32_t in_start[3] = {0, 0, 0};
    uint32_t in_valid[3] = {0, 0, 0};
    in_start[axes_[0]] = out_start[0];
    in_start[axes_[1]] = out_start[1];
    in_start[axes_[2]] = out_start[2];
    in_valid[axes_[0]] = out_valid[0];
    in_valid[axes_[1]] = out_valid[1];
    in_valid[axes_[2]] = out_valid[2];

    LocalTensor<uint64_t> input_complex = input_buf_.Get<float>().ReinterpretCast<uint64_t>();
    const uint32_t input_block_bytes = in_valid[2] * sizeof(uint64_t);
    const DataCopyPadExtParams<uint64_t> input_pad;
    for (uint32_t row0 = 0; row0 < in_valid[0]; ++row0) {
      const uint64_t input_offset = static_cast<uint64_t>(batch_index) * volume_ +
          static_cast<uint64_t>(in_start[0] + row0) * dims_[1] * dims_[2] +
          static_cast<uint64_t>(in_start[1]) * dims_[2] + in_start[2];
      GlobalTensor<uint64_t> input_global;
      input_global.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(input_ptr_ + 2 * input_offset));
      const DataCopyExtParams input_params(in_valid[1],
                                           input_block_bytes,
                                           (dims_[2] - in_valid[2]) * sizeof(uint64_t),
                                           (kTile - in_valid[2]) * sizeof(uint64_t),
                                           0);
      DataCopyPad(input_complex[row0 * kTile * kTile], input_global, input_params, input_pad);
    }
    LocalTensor<uint32_t> index_local = index_buf_.Get<uint32_t>();
    DataCopy(index_local, indices_, kTileFloatCount);
    PipeBarrier<PIPE_ALL>();

    LocalTensor<float> input_local = input_buf_.Get<float>();
    LocalTensor<float> output_local = output_buf_.Get<float>();
    Gather(output_local, input_local, index_local, 0, kTileFloatCount);
    PipeBarrier<PIPE_ALL>();

    LocalTensor<uint64_t> output_complex = output_local.ReinterpretCast<uint64_t>();
    const uint32_t output_block_bytes = out_valid[2] * sizeof(uint64_t);
    for (uint32_t row0 = 0; row0 < out_valid[0]; ++row0) {
      const uint64_t output_offset = static_cast<uint64_t>(batch_index) * volume_ +
          static_cast<uint64_t>(out_start[0] + row0) * out_dims_[1] * out_dims_[2] +
          static_cast<uint64_t>(out_start[1]) * out_dims_[2] + out_start[2];
      GlobalTensor<uint64_t> output_global;
      output_global.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(output_ptr_ + 2 * output_offset));
      const DataCopyExtParams output_params(out_valid[1],
                                            output_block_bytes,
                                            (kTile - out_valid[2]) * sizeof(uint64_t),
                                            (out_dims_[2] - out_valid[2]) * sizeof(uint64_t),
                                            0);
      DataCopyPad(output_global,
                  output_complex[row0 * kTile * kTile],
                  output_params);
    }
  }

 private:
  TPipe pipe_;
  TBuf<TPosition::VECCALC> input_buf_;
  TBuf<TPosition::VECCALC> output_buf_;
  TBuf<TPosition::VECCALC> index_buf_;
  GlobalTensor<uint32_t> indices_;
  __gm__ float *input_ptr_ = nullptr;
  __gm__ float *output_ptr_ = nullptr;
  uint32_t dims_[3] = {0, 0, 0};
  uint32_t out_dims_[3] = {0, 0, 0};
  uint32_t axes_[3] = {0, 1, 2};
  uint32_t tile_counts_[3] = {0, 0, 0};
  uint32_t batch_ = 0;
  uint64_t volume_ = 0;
};
}  // namespace

extern "C" __global__ __aicore__ void flagfft_npu_transpose3d(GM_ADDR input,
                                                               GM_ADDR output,
                                                               GM_ADDR indices,
                                                               uint32_t n0,
                                                               uint32_t n1,
                                                               uint32_t n2,
                                                               uint32_t batch,
                                                               uint32_t axis0,
                                                               uint32_t axis1,
                                                               uint32_t axis2) {
  Transpose3DAiv kernel;
  kernel.Init(input, output, indices, n0, n1, n2, batch, axis0, axis1, axis2);
  kernel.Process();
}
