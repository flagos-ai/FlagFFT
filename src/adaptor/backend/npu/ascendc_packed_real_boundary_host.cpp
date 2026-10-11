// Copyright 2026 FlagOS Contributors
// Licensed under the Apache License, Version 2.0.

#include "adaptor/backend/npu/ascendc_packed_real_boundary.hpp"

#include <acl/acl_rt.h>
#include <aclrtlaunch_flagfft_npu_packed_real_boundary.h>

#include <algorithm>

namespace flagfft::adaptor::npu {

flagfftResult launch_ascendc_packed_real_boundary(DevicePtr input,
                                                  DevicePtr output,
                                                  DevicePtr indices,
                                                  DevicePtr twiddles,
                                                  int32_t length,
                                                  int32_t input_distance,
                                                  int32_t output_distance,
                                                  int32_t batch,
                                                  bool r2c_postprocess,
                                                  StreamHandle stream) {
  const int32_t packed = length / 2;
  const int32_t half = packed + 1;
  const int32_t minimum_input_distance = r2c_postprocess ? packed : half;
  const int32_t minimum_output_distance = r2c_postprocess ? half : packed;
  if (input == 0 || output == 0 || indices == 0 || twiddles == 0 || length < 128 ||
      length > 2048 || (length & 1) != 0 || input_distance < minimum_input_distance ||
      output_distance < minimum_output_distance || batch <= 0) {
    return FLAGFFT_INVALID_SIZE;
  }
  constexpr uint32_t kMaxBlocks = 65535;
  const uint32_t output_count = r2c_postprocess ? static_cast<uint32_t>(half)
                                               : static_cast<uint32_t>(packed);
  uint32_t group_rows = 1;
  while ((output_count * group_rows) % 4U != 0) ++group_rows;
  const uint32_t row_groups =
      (static_cast<uint32_t>(batch) + group_rows - 1) / group_rows;
  const uint32_t blocks = std::min<uint32_t>(row_groups, kMaxBlocks);
  const uint32_t status = ACLRT_LAUNCH_KERNEL(flagfft_npu_packed_real_boundary)(
      blocks,
      reinterpret_cast<void *>(stream),
      reinterpret_cast<uint8_t *>(input),
      reinterpret_cast<uint8_t *>(output),
      reinterpret_cast<uint8_t *>(indices),
      reinterpret_cast<uint8_t *>(twiddles),
      static_cast<uint32_t>(length),
      static_cast<uint32_t>(input_distance),
      static_cast<uint32_t>(output_distance),
      static_cast<uint32_t>(batch),
      static_cast<uint32_t>(r2c_postprocess ? 1 : 0));
  return status == 0 ? FLAGFFT_SUCCESS : FLAGFFT_EXEC_FAILED;
}

}  // namespace flagfft::adaptor::npu
