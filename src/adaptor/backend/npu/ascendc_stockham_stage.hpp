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

#pragma once

#include "adaptor/adaptor.h"

#include <cstdint>

namespace flagfft::adaptor::npu {

// Keep one Vector Unit repeat per Stockham tile. Larger tiles reduce GM tile
// count but also increase per-block UB use and can reduce AIV occupancy.
inline constexpr uint32_t kAscendCStockhamTile = 32;
inline constexpr uint32_t kAscendCStockhamMaxRadix = 19;
inline constexpr uint32_t kAscendCStockhamRootStride = 24;
inline constexpr uint32_t kAscendCStockhamIndexVectors = 5;
inline constexpr uint32_t kAscendCStockhamComplexAlignment = 4;

// DataCopyPad local addresses must start on a 32-byte boundary. A complex64
// value occupies 8 bytes, so only combine transforms when each local segment
// begins at a multiple of four complex values.
inline constexpr uint64_t ascendc_stockham_batch_group(uint64_t lane_extent) {
  return lane_extent > 0 && lane_extent < kAscendCStockhamTile &&
                 lane_extent % kAscendCStockhamComplexAlignment == 0
             ? kAscendCStockhamTile / lane_extent
             : 1;
}

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
                                             StreamHandle stream);

}  // namespace flagfft::adaptor::npu
