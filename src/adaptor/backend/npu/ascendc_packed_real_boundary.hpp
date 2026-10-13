// Copyright 2026 FlagOS Contributors
// Licensed under the Apache License, Version 2.0.

#pragma once

#include "adaptor/adaptor.h"

#include <cstdint>

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
                                                  StreamHandle stream);

}  // namespace flagfft::adaptor::npu
