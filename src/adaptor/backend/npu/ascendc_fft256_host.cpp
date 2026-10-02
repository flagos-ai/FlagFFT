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

#include "adaptor/backend/npu/ascendc_fft256.hpp"

#include <acl/acl_rt.h>
#include <aclrtlaunch_flagfft_npu_fft256.h>
#include <aclrtlaunch_flagfft_npu_fft256_group4.h>
#include <aclrtlaunch_flagfft_npu_fft256_group8.h>

namespace flagfft::adaptor::npu {

flagfftResult launch_ascendc_fft256(DevicePtr input,
                                    DevicePtr output,
                                    DevicePtr indices,
                                    DevicePtr twiddles,
                                    int32_t transform_count,
                                    int32_t group_size,
                                    StreamHandle stream) {
  if (transform_count <= 0 || (group_size != 1 && group_size != 4 && group_size != 8) ||
      transform_count % group_size != 0) return FLAGFFT_INVALID_SIZE;
  const uint32_t block_dim = static_cast<uint32_t>(transform_count / group_size);
  uint32_t status = 0;
  if (group_size == 8) {
    status = ACLRT_LAUNCH_KERNEL(flagfft_npu_fft256_group8)(
        block_dim,
        reinterpret_cast<void *>(stream),
        reinterpret_cast<uint8_t *>(input),
        reinterpret_cast<uint8_t *>(output),
        reinterpret_cast<uint8_t *>(indices),
        reinterpret_cast<uint8_t *>(twiddles),
        static_cast<uint32_t>(transform_count));
  } else if (group_size == 4) {
    status = ACLRT_LAUNCH_KERNEL(flagfft_npu_fft256_group4)(
        block_dim,
        reinterpret_cast<void *>(stream),
        reinterpret_cast<uint8_t *>(input),
        reinterpret_cast<uint8_t *>(output),
        reinterpret_cast<uint8_t *>(indices),
        reinterpret_cast<uint8_t *>(twiddles),
        static_cast<uint32_t>(transform_count));
  } else {
    status = ACLRT_LAUNCH_KERNEL(flagfft_npu_fft256)(
        block_dim,
        reinterpret_cast<void *>(stream),
        reinterpret_cast<uint8_t *>(input),
        reinterpret_cast<uint8_t *>(output),
        reinterpret_cast<uint8_t *>(indices),
        reinterpret_cast<uint8_t *>(twiddles),
        static_cast<uint32_t>(transform_count));
  }
  return status == 0 ? FLAGFFT_SUCCESS : FLAGFFT_EXEC_FAILED;
}

}  // namespace flagfft::adaptor::npu
