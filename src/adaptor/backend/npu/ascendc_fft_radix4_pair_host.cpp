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

#include "adaptor/backend/npu/ascendc_fft_radix4_pair.hpp"

#include <acl/acl_rt.h>
#include <aclrtlaunch_flagfft_npu_fft128_radix4_pair_g4.h>
#include <aclrtlaunch_flagfft_npu_fft128_radix4_pair_g8.h>
#include <aclrtlaunch_flagfft_npu_fft2048_radix4_pair_g1.h>
#include <aclrtlaunch_flagfft_npu_fft64_radix4_pair_g1.h>
#include <aclrtlaunch_flagfft_npu_fft64_radix4_pair_g4.h>
#include <aclrtlaunch_flagfft_npu_fft64_radix4_pair_g8.h>

namespace flagfft::adaptor::npu {

flagfftResult launch_ascendc_fft_radix4_pair(int32_t length,
                                             DevicePtr input,
                                             DevicePtr output,
                                             DevicePtr indices,
                                             DevicePtr twiddles,
                                             int32_t transform_count,
                                             int32_t group_size,
                                             int32_t mode,
                                             StreamHandle stream) {
  if (transform_count <= 0 || group_size <= 0 || transform_count % group_size != 0 || mode < 0 || mode > 2) {
    return FLAGFFT_INVALID_SIZE;
  }

  const uint32_t block_count = static_cast<uint32_t>(transform_count / group_size);
  const auto input_bytes = reinterpret_cast<uint8_t *>(input);
  const auto output_bytes = reinterpret_cast<uint8_t *>(output);
  const auto index_bytes = reinterpret_cast<uint8_t *>(indices);
  const auto twiddle_bytes = reinterpret_cast<uint8_t *>(twiddles);
  uint32_t status = 0;
  bool supported = true;

  if (length == 64 && group_size == 1) {
    status = ACLRT_LAUNCH_KERNEL(flagfft_npu_fft64_radix4_pair_g1)(block_count,
                                                                   reinterpret_cast<void *>(stream),
                                                                   input_bytes,
                                                                   output_bytes,
                                                                   index_bytes,
                                                                   twiddle_bytes,
                                                                   block_count,
                                                                   static_cast<uint32_t>(mode));
  } else if (length == 64 && group_size == 4) {
    status = ACLRT_LAUNCH_KERNEL(flagfft_npu_fft64_radix4_pair_g4)(block_count,
                                                                   reinterpret_cast<void *>(stream),
                                                                   input_bytes,
                                                                   output_bytes,
                                                                   index_bytes,
                                                                   twiddle_bytes,
                                                                   block_count,
                                                                   static_cast<uint32_t>(mode));
  } else if (length == 64 && group_size == 8) {
    status = ACLRT_LAUNCH_KERNEL(flagfft_npu_fft64_radix4_pair_g8)(block_count,
                                                                   reinterpret_cast<void *>(stream),
                                                                   input_bytes,
                                                                   output_bytes,
                                                                   index_bytes,
                                                                   twiddle_bytes,
                                                                   block_count,
                                                                   static_cast<uint32_t>(mode));
  } else if (length == 128 && group_size == 4) {
    status = ACLRT_LAUNCH_KERNEL(flagfft_npu_fft128_radix4_pair_g4)(block_count,
                                                                    reinterpret_cast<void *>(stream),
                                                                    input_bytes,
                                                                    output_bytes,
                                                                    index_bytes,
                                                                    twiddle_bytes,
                                                                    block_count,
                                                                    static_cast<uint32_t>(mode));
  } else if (length == 128 && group_size == 8) {
    status = ACLRT_LAUNCH_KERNEL(flagfft_npu_fft128_radix4_pair_g8)(block_count,
                                                                    reinterpret_cast<void *>(stream),
                                                                    input_bytes,
                                                                    output_bytes,
                                                                    index_bytes,
                                                                    twiddle_bytes,
                                                                    block_count,
                                                                    static_cast<uint32_t>(mode));
  } else if (length == 2048 && group_size == 1 && mode == 0) {
    status = ACLRT_LAUNCH_KERNEL(flagfft_npu_fft2048_radix4_pair_g1)(block_count,
                                                                     reinterpret_cast<void *>(stream),
                                                                     input_bytes,
                                                                     output_bytes,
                                                                     index_bytes,
                                                                     twiddle_bytes,
                                                                     block_count,
                                                                     static_cast<uint32_t>(mode));
  } else {
    supported = false;
  }

  if (!supported) return FLAGFFT_INVALID_SIZE;
  return status == 0 ? FLAGFFT_SUCCESS : FLAGFFT_EXEC_FAILED;
}

}  // namespace flagfft::adaptor::npu
