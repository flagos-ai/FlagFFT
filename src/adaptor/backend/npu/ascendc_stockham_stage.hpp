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

inline constexpr uint32_t kAscendCStockhamTile = 32;

flagfftResult launch_ascendc_stockham_stage(DevicePtr input,
                                             DevicePtr output,
                                             DevicePtr stage_twiddles,
                                             DevicePtr radix_roots,
                                             DevicePtr indices,
                                             uint32_t length,
                                             uint32_t radix,
                                             uint32_t span,
                                             uint32_t batch,
                                             StreamHandle stream);

}  // namespace flagfft::adaptor::npu
