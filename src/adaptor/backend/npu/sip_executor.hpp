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
#include "flagfft.h"

#include <cstdint>

namespace flagfft::adaptor::npu {

// The returned handle owns the SiP plan, workspace, and shape metadata. The
// input/output ACL tensors are cached after the first execute call.
void *sip_plan_create(int64_t length, int64_t batch, flagfftType type, int direction);
void sip_plan_destroy(void *plan);
flagfftResult sip_plan_set_stream(void *plan, StreamHandle stream);
flagfftResult sip_plan_execute(void *plan, void *input, void *output);

}  // namespace flagfft::adaptor::npu
