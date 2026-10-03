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
#include <aclrtlaunch_flagfft_npu_fft256_pair_group8.h>
#include <aclrtlaunch_flagfft_npu_fft256_pair_radix4_store_group8.h>
#include <aclrtlaunch_flagfft_npu_fft256_pair_store_group8.h>
#include <aclrtlaunch_flagfft_npu_fft256_real_forward_group8.h>
#include <aclrtlaunch_flagfft_npu_fft256_real_inverse_group8.h>

#include <cstdio>

namespace flagfft::adaptor::npu {

flagfftResult launch_ascendc_fft256(DevicePtr input,
                                    DevicePtr output,
                                    DevicePtr indices,
                                    DevicePtr twiddles,
                                    int32_t transform_count,
                                    int32_t group_size,
                                    bool pair_mode,
                                    bool transposed_store,
                                    bool radix4_mode,
                                    int32_t mode,
                                    int32_t output_row_stride,
                                    int32_t output_transform_offset,
                                    StreamHandle stream) {
  if (transform_count <= 0 || (group_size != 1 && group_size != 4 && group_size != 8) ||
      (pair_mode && group_size != 8) || (transposed_store && !pair_mode) ||
      (radix4_mode && (!pair_mode || !transposed_store || group_size != 8)) ||
      mode < 0 || mode > 2 ||
      (mode != 0 && (!pair_mode || group_size != 8 || transposed_store || radix4_mode)) ||
      (transposed_store && output_row_stride < group_size) || output_transform_offset < 0 ||
      transform_count % group_size != 0) {
    return FLAGFFT_INVALID_SIZE;
  }
  constexpr uint32_t kMaxRealBlocksPerLaunch = 2;
  const uint32_t logical_block_dim = static_cast<uint32_t>(transform_count / group_size);
  const uint32_t block_dim = (mode == 1 || mode == 2) &&
                                     logical_block_dim > kMaxRealBlocksPerLaunch
      ? kMaxRealBlocksPerLaunch
      : logical_block_dim;
  if (mode == 1 || mode == 2) {
    std::fprintf(stderr,
                 "fft256 real launch mode=%d transforms=%d logical_blocks=%u block_dim=%u input=%p output=%p\\n",
                 mode, transform_count, logical_block_dim, block_dim,
                 reinterpret_cast<void *>(input), reinterpret_cast<void *>(output));
  }
  uint32_t status = 0;
  if (mode == 1) {
    status = ACLRT_LAUNCH_KERNEL(flagfft_npu_fft256_real_forward_group8)(
        block_dim,
        reinterpret_cast<void *>(stream),
        reinterpret_cast<uint8_t *>(input),
        reinterpret_cast<uint8_t *>(output),
        reinterpret_cast<uint8_t *>(indices),
        reinterpret_cast<uint8_t *>(twiddles),
        static_cast<uint32_t>(transform_count));
  } else if (mode == 2) {
    status = ACLRT_LAUNCH_KERNEL(flagfft_npu_fft256_real_inverse_group8)(
        block_dim,
        reinterpret_cast<void *>(stream),
        reinterpret_cast<uint8_t *>(input),
        reinterpret_cast<uint8_t *>(output),
        reinterpret_cast<uint8_t *>(indices),
        reinterpret_cast<uint8_t *>(twiddles),
        static_cast<uint32_t>(transform_count));
  } else if (group_size == 8) {
    if (pair_mode) {
      if (transposed_store) {
        if (radix4_mode) {
          status = ACLRT_LAUNCH_KERNEL(flagfft_npu_fft256_pair_radix4_store_group8)(
              block_dim,
              reinterpret_cast<void *>(stream),
              reinterpret_cast<uint8_t *>(input),
              reinterpret_cast<uint8_t *>(output),
              reinterpret_cast<uint8_t *>(indices),
              reinterpret_cast<uint8_t *>(twiddles),
              static_cast<uint32_t>(transform_count),
              static_cast<uint32_t>(output_row_stride),
              static_cast<uint32_t>(output_transform_offset));
        } else {
          status = ACLRT_LAUNCH_KERNEL(flagfft_npu_fft256_pair_store_group8)(
              block_dim,
              reinterpret_cast<void *>(stream),
              reinterpret_cast<uint8_t *>(input),
              reinterpret_cast<uint8_t *>(output),
              reinterpret_cast<uint8_t *>(indices),
              reinterpret_cast<uint8_t *>(twiddles),
              static_cast<uint32_t>(transform_count),
              static_cast<uint32_t>(output_row_stride),
              static_cast<uint32_t>(output_transform_offset));
        }
      } else {
        status = ACLRT_LAUNCH_KERNEL(flagfft_npu_fft256_pair_group8)(
            block_dim,
            reinterpret_cast<void *>(stream),
            reinterpret_cast<uint8_t *>(input),
            reinterpret_cast<uint8_t *>(output),
            reinterpret_cast<uint8_t *>(indices),
            reinterpret_cast<uint8_t *>(twiddles),
            static_cast<uint32_t>(transform_count));
      }
    } else {
      status = ACLRT_LAUNCH_KERNEL(flagfft_npu_fft256_group8)(
          block_dim,
          reinterpret_cast<void *>(stream),
          reinterpret_cast<uint8_t *>(input),
          reinterpret_cast<uint8_t *>(output),
          reinterpret_cast<uint8_t *>(indices),
          reinterpret_cast<uint8_t *>(twiddles),
          static_cast<uint32_t>(transform_count));
    }
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
