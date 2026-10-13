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

#include "adaptor/backend/npu/ascendc_stockham_stage.hpp"

#include <acl/acl_rt.h>
#include <aclrtlaunch_flagfft_npu_stockham_stage.h>

#include <algorithm>
#include <limits>

namespace flagfft::adaptor::npu {

flagfftResult launch_ascendc_stockham_stage(DevicePtr input,
                                             DevicePtr output,
                                             DevicePtr stage_twiddles,
                                             DevicePtr radix_roots,
                                             DevicePtr indices,
                                             DevicePtr store_indices,
                                             uint32_t length,
                                             uint32_t radix,
                                             uint32_t span,
                                             uint32_t batch,
                                             StreamHandle stream) {
  if (input == 0 || output == 0 || radix_roots == 0 || indices == 0 || length == 0 ||
      radix < 2 || radix > 19 || span == 0 || length % radix != 0 ||
      length % (static_cast<uint64_t>(radix) * span) != 0 || batch == 0 ||
      (span > 1 && stage_twiddles == 0)) {
    return FLAGFFT_INVALID_SIZE;
  }

  const uint64_t ndiv = length / radix;
  const uint64_t tiles_per_group =
      span == 1 ? 1 : (span + kAscendCStockhamTile - 1) / kAscendCStockhamTile;
  const uint64_t groups_per_transform = span == 1 ? 1 : ndiv / span;
  // Fill otherwise idle lanes with independent transform segments.
  const uint64_t lane_extent = span == 1
                                   ? ndiv
                                   : std::min<uint64_t>(span, kAscendCStockhamTile);
  const uint64_t batch_group = ascendc_stockham_batch_group(lane_extent);
  // If batch grouping cannot fill a vector because the butterfly span is not
  // 32-byte aligned, pack adjacent independent Stockham groups into the tile.
  const uint64_t groups_per_tile =
      span > 1 && span < kAscendCStockhamTile && batch_group == 1
          ? kAscendCStockhamTile / span
          : 1;
  const uint64_t tiles_per_transform = span == 1
                                           ? (ndiv + kAscendCStockhamTile - 1) /
                                                 kAscendCStockhamTile
                                           : ((groups_per_transform + groups_per_tile - 1) /
                                              groups_per_tile) * tiles_per_group;
  const uint64_t batch_groups =
      (static_cast<uint64_t>(batch) + batch_group - 1) / batch_group;
  const uint64_t total_tiles = batch_groups * tiles_per_transform;
  if (tiles_per_transform == 0 || total_tiles == 0) return FLAGFFT_INVALID_SIZE;

  constexpr uint32_t kMaxGrid = 65535;
  const uint32_t blocks = static_cast<uint32_t>(std::min<uint64_t>(total_tiles, kMaxGrid));
  const uint32_t status = ACLRT_LAUNCH_KERNEL(flagfft_npu_stockham_stage)(
      blocks,
      reinterpret_cast<void *>(stream),
      reinterpret_cast<uint8_t *>(input),
      reinterpret_cast<uint8_t *>(output),
      reinterpret_cast<uint8_t *>(stage_twiddles),
      reinterpret_cast<uint8_t *>(radix_roots),
      reinterpret_cast<uint8_t *>(indices),
      reinterpret_cast<uint8_t *>(store_indices),
      length,
      radix,
      span,
      batch);
  return status == 0 ? FLAGFFT_SUCCESS : FLAGFFT_EXEC_FAILED;
}

}  // namespace flagfft::adaptor::npu
