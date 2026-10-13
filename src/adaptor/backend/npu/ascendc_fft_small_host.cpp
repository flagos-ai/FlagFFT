// Copyright 2026 FlagOS Contributors
// Licensed under the Apache License, Version 2.0.

#include "adaptor/backend/npu/ascendc_fft_small.hpp"

#include <acl/acl_rt.h>
#include <aclrtlaunch_flagfft_npu_fft16.h>
#include <aclrtlaunch_flagfft_npu_fft16_g1.h>
#include <aclrtlaunch_flagfft_npu_fft32.h>
#include <aclrtlaunch_flagfft_npu_fft32_g1.h>

namespace flagfft::adaptor::npu {

flagfftResult launch_ascendc_fft_small(int32_t length,
                                       DevicePtr input,
                                       DevicePtr output,
                                       DevicePtr indices,
                                       DevicePtr twiddles,
                                       int32_t block_count,
                                       int32_t group_size,
                                       int32_t stride,
                                       int32_t output_stride,
                                       int32_t mode,
                                       StreamHandle stream) {
  const int32_t preferred_group = length == 16 ? 8 : length == 32 ? 4 : 0;
  if (block_count <= 0 || stride <= 0 || output_stride <= 0 || preferred_group == 0 ||
      (group_size != 1 && group_size != preferred_group) || mode < 0 || mode > 2 ||
      (mode != 0 && (stride != 1 || output_stride != 1)) ||
      (stride != 1 && (stride < group_size || stride % group_size != 0)) ||
      (output_stride != 1 &&
       (output_stride < group_size || output_stride % group_size != 0))) {
    return FLAGFFT_INVALID_SIZE;
  }
  uint32_t status = 0;
  if (length == 16 && group_size == 1) {
    status = ACLRT_LAUNCH_KERNEL(flagfft_npu_fft16_g1)(static_cast<uint32_t>(block_count),
                                                       reinterpret_cast<void *>(stream),
                                                       reinterpret_cast<uint8_t *>(input),
                                                       reinterpret_cast<uint8_t *>(output),
                                                       reinterpret_cast<uint8_t *>(indices),
                                                       reinterpret_cast<uint8_t *>(twiddles),
                                                       static_cast<uint32_t>(block_count),
                                                       static_cast<uint32_t>(stride),
                                                       static_cast<uint32_t>(output_stride),
                                                       static_cast<uint32_t>(mode));
  } else if (length == 16) {
    status = ACLRT_LAUNCH_KERNEL(flagfft_npu_fft16)(static_cast<uint32_t>(block_count),
                                                    reinterpret_cast<void *>(stream),
                                                    reinterpret_cast<uint8_t *>(input),
                                                    reinterpret_cast<uint8_t *>(output),
                                                    reinterpret_cast<uint8_t *>(indices),
                                                    reinterpret_cast<uint8_t *>(twiddles),
                                                    static_cast<uint32_t>(block_count),
                                                    static_cast<uint32_t>(stride),
                                                    static_cast<uint32_t>(output_stride),
                                                    static_cast<uint32_t>(mode));
  } else if (length == 32 && group_size == 1) {
    status = ACLRT_LAUNCH_KERNEL(flagfft_npu_fft32_g1)(static_cast<uint32_t>(block_count),
                                                       reinterpret_cast<void *>(stream),
                                                       reinterpret_cast<uint8_t *>(input),
                                                       reinterpret_cast<uint8_t *>(output),
                                                       reinterpret_cast<uint8_t *>(indices),
                                                       reinterpret_cast<uint8_t *>(twiddles),
                                                       static_cast<uint32_t>(block_count),
                                                       static_cast<uint32_t>(stride),
                                                       static_cast<uint32_t>(output_stride),
                                                       static_cast<uint32_t>(mode));
  } else if (length == 32) {
    status = ACLRT_LAUNCH_KERNEL(flagfft_npu_fft32)(static_cast<uint32_t>(block_count),
                                                    reinterpret_cast<void *>(stream),
                                                    reinterpret_cast<uint8_t *>(input),
                                                    reinterpret_cast<uint8_t *>(output),
                                                    reinterpret_cast<uint8_t *>(indices),
                                                    reinterpret_cast<uint8_t *>(twiddles),
                                                    static_cast<uint32_t>(block_count),
                                                    static_cast<uint32_t>(stride),
                                                    static_cast<uint32_t>(output_stride),
                                                    static_cast<uint32_t>(mode));
  } else {
    return FLAGFFT_INVALID_SIZE;
  }
  return status == 0 ? FLAGFFT_SUCCESS : FLAGFFT_EXEC_FAILED;
}

}  // namespace flagfft::adaptor::npu
