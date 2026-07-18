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

#include "plan_handles.hpp"

namespace flagfft::cli {

FlagfftPlanHandle::FlagfftPlanHandle(flagfftHandle handle) : handle_(handle) {
}

FlagfftPlanHandle::~FlagfftPlanHandle() {
  reset();
}

FlagfftPlanHandle::FlagfftPlanHandle(FlagfftPlanHandle&& other) noexcept : handle_(other.handle_) {
  other.handle_ = nullptr;
}

FlagfftPlanHandle& FlagfftPlanHandle::operator=(FlagfftPlanHandle&& other) noexcept {
  if (this != &other) {
    reset();
    handle_ = other.handle_;
    other.handle_ = nullptr;
  }
  return *this;
}

flagfftHandle FlagfftPlanHandle::get() const noexcept {
  return handle_;
}

flagfftHandle FlagfftPlanHandle::release() noexcept {
  flagfftHandle out = handle_;
  handle_ = nullptr;
  return out;
}

void FlagfftPlanHandle::reset(flagfftHandle handle) {
  if (handle_ != nullptr) {
    flagfftDestroy(handle_);
  }
  handle_ = handle;
}

}  // namespace flagfft::cli
