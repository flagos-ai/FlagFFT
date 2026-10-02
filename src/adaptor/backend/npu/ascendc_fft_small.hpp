// Copyright 2026 FlagOS Contributors
// Licensed under the Apache License, Version 2.0.

#pragma once

#include "adaptor/adaptor.h"

#include <cstdint>

namespace flagfft::adaptor::npu {

flagfftResult launch_ascendc_fft_small(int32_t length,
                                       DevicePtr input,
                                       DevicePtr output,
                                       DevicePtr indices,
                                       DevicePtr twiddles,
                                       int32_t block_count,
                                       int32_t group_size,
                                       int32_t stride,
                                       int32_t mode,
                                       StreamHandle stream);

}  // namespace flagfft::adaptor::npu
