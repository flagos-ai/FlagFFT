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

#include "adaptor/backend/npu/ascendc_transpose2d.hpp"

#include <acl/acl_rt.h>
#include <aclrtlaunch_flagfft_npu_transpose2d.h>

#include <algorithm>
#include <limits>

namespace flagfft::adaptor::npu {

flagfftResult launch_ascendc_transpose2d(DevicePtr input,
                                         DevicePtr output,
                                         DevicePtr indices,
                                         int32_t rows,
                                         int32_t cols,
                                         int32_t batch,
                                         StreamHandle stream) {
  if (input == 0 || output == 0 || indices == 0 || rows < 4 || cols < 4 || batch <= 0) {
    return FLAGFFT_INVALID_SIZE;
  }
  const uint64_t row_tiles = (static_cast<uint64_t>(rows) + 15) / 16;
  const uint64_t col_tiles = (static_cast<uint64_t>(cols) + 15) / 16;
  const uint64_t total_tiles = static_cast<uint64_t>(batch) * row_tiles * col_tiles;
  if (total_tiles == 0 || total_tiles > std::numeric_limits<uint32_t>::max()) {
    return FLAGFFT_INVALID_SIZE;
  }
  // ACL blockDim is limited to 65535. The kernel uses a grid-stride loop to
  // cover larger matrices without splitting the transpose across launches.
  const uint32_t blocks = static_cast<uint32_t>(std::min<uint64_t>(total_tiles, 65535));
  const uint32_t status = ACLRT_LAUNCH_KERNEL(flagfft_npu_transpose2d)(
      blocks,
      reinterpret_cast<void *>(stream),
      reinterpret_cast<uint8_t *>(input),
      reinterpret_cast<uint8_t *>(output),
      reinterpret_cast<uint8_t *>(indices),
      static_cast<uint32_t>(rows),
      static_cast<uint32_t>(cols),
      static_cast<uint32_t>(batch));
  return status == 0 ? FLAGFFT_SUCCESS : FLAGFFT_EXEC_FAILED;
}

}  // namespace flagfft::adaptor::npu
