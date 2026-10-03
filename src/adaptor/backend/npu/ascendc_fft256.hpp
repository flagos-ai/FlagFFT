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

#pragma once

#include "adaptor/adaptor.h"

#include <cstdint>

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
                                    StreamHandle stream);

}  // namespace flagfft::adaptor::npu
