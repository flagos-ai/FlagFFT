// Copyright 2026 FlagOS Contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <algorithm>
#include <cstddef>

namespace flagfft::validation {

// The bounded capture mode can validate numerical output in smaller plans.
// A batch-sensitive path must explicitly preserve the benchmark's request.
inline int capture_batch_chunk(int requested_batch,
                               std::size_t input_bytes,
                               std::size_t output_bytes,
                               bool preserve_batch) {
  if (requested_batch <= 1 || preserve_batch) return requested_batch;
  constexpr std::size_t max_chunk_bytes = 128ULL * 1024ULL * 1024ULL;
  const auto batch = static_cast<std::size_t>(requested_batch);
  const std::size_t bytes_per_batch = input_bytes / batch + output_bytes / batch;
  const std::size_t max_batch = bytes_per_batch == 0
      ? batch : std::max<std::size_t>(1, max_chunk_bytes / bytes_per_batch);
  return static_cast<int>(std::min(batch, max_batch));
}

}  // namespace flagfft::validation
