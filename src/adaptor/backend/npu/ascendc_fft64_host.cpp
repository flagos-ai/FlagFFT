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

#include "adaptor/backend/npu/ascendc_fft64.hpp"

#include <acl/acl_rt.h>
#include <aclrtlaunch_flagfft_npu_fft64.h>

namespace flagfft::adaptor::npu {

flagfftResult launch_ascendc_fft64(DevicePtr input,
                                   DevicePtr output,
                                   DevicePtr indices,
                                   DevicePtr twiddles,
                                   int32_t transform_count,
                                   int32_t stride,
                                   StreamHandle stream) {
  if (transform_count <= 0 || (stride != 1 && stride != 64)) return FLAGFFT_INVALID_SIZE;
  const uint32_t status = ACLRT_LAUNCH_KERNEL(flagfft_npu_fft64)(static_cast<uint32_t>(transform_count),
                                                                 reinterpret_cast<void *>(stream),
                                                                 reinterpret_cast<uint8_t *>(input),
                                                                 reinterpret_cast<uint8_t *>(output),
                                                                 reinterpret_cast<uint8_t *>(indices),
                                                                 reinterpret_cast<uint8_t *>(twiddles),
                                                                 static_cast<uint32_t>(transform_count),
                                                                 static_cast<uint32_t>(stride));
  return status == 0 ? FLAGFFT_SUCCESS : FLAGFFT_EXEC_FAILED;
}

}  // namespace flagfft::adaptor::npu
