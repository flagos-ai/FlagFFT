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

#include "adaptor/backend/npu/ascendc_transpose3d.hpp"

#include <acl/acl_rt.h>
#include <aclrtlaunch_flagfft_npu_transpose3d.h>

#include <algorithm>
#include <limits>

namespace flagfft::adaptor::npu {

flagfftResult launch_ascendc_transpose3d(DevicePtr input,
                                         DevicePtr output,
                                         DevicePtr indices,
                                         DevicePtr edge_indices,
                                         int32_t n0,
                                         int32_t n1,
                                         int32_t n2,
                                         int32_t batch,
                                         int32_t axis0,
                                         int32_t axis1,
                                         int32_t axis2,
                                         StreamHandle stream) {
  if (n0 <= 0 || n1 <= 0 || n2 <= 0 || batch <= 0 || indices == 0 ||
      axis0 < 0 || axis0 > 2 || axis1 < 0 || axis1 > 2 || axis2 < 0 || axis2 > 2 ||
      axis0 == axis1 || axis0 == axis2 || axis1 == axis2) {
    return FLAGFFT_INVALID_SIZE;
  }
  const int32_t dims[] = {n0, n1, n2};
  const uint64_t tiles0 = (static_cast<uint64_t>(dims[axis0]) + 15) / 16;
  const uint64_t tiles1 = (static_cast<uint64_t>(dims[axis1]) + 15) / 16;
  const uint64_t tiles2 = (static_cast<uint64_t>(dims[axis2]) + 15) / 16;
  const uint64_t blocks = static_cast<uint64_t>(batch) * tiles0 * tiles1 * tiles2;
  if (blocks == 0 || blocks > std::numeric_limits<uint32_t>::max()) {
    return FLAGFFT_INVALID_SIZE;
  }
  const uint32_t status = ACLRT_LAUNCH_KERNEL(flagfft_npu_transpose3d)(
      static_cast<uint32_t>(blocks),
      reinterpret_cast<void *>(stream),
      reinterpret_cast<uint8_t *>(input),
      reinterpret_cast<uint8_t *>(output),
      reinterpret_cast<uint8_t *>(indices),
      reinterpret_cast<uint8_t *>(edge_indices),
      static_cast<uint32_t>(n0),
      static_cast<uint32_t>(n1),
      static_cast<uint32_t>(n2),
      static_cast<uint32_t>(batch),
      static_cast<uint32_t>(axis0),
      static_cast<uint32_t>(axis1),
      static_cast<uint32_t>(axis2));
  return status == 0 ? FLAGFFT_SUCCESS : FLAGFFT_EXEC_FAILED;
}

}  // namespace flagfft::adaptor::npu
