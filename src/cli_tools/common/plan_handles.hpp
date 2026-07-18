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

#include "flagfft.h"

namespace flagfft::cli {

class FlagfftPlanHandle {
 public:
  FlagfftPlanHandle() = default;
  explicit FlagfftPlanHandle(flagfftHandle handle);
  ~FlagfftPlanHandle();

  FlagfftPlanHandle(const FlagfftPlanHandle&) = delete;
  FlagfftPlanHandle& operator=(const FlagfftPlanHandle&) = delete;
  FlagfftPlanHandle(FlagfftPlanHandle&& other) noexcept;
  FlagfftPlanHandle& operator=(FlagfftPlanHandle&& other) noexcept;

  flagfftHandle get() const noexcept;
  flagfftHandle release() noexcept;
  void reset(flagfftHandle handle = nullptr);

 private:
  flagfftHandle handle_ = nullptr;
};

}  // namespace flagfft::cli
