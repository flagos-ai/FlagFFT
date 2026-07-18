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

#include "flagfft/core.hpp"

namespace flagfft {

FFTRequest forward_child_request(const FFTRequest &request) {
  FFTRequest child = request;
  child.direction = "forward";
  child.norm = "backward";
  return child;
}

std::string triton_target_for_request(const FFTRequest &request) {
  return adaptor::triton_target(request.device_arch);
}

}  // namespace flagfft
