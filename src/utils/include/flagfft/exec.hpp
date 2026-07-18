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

#include "flagfft/codegen.hpp"

namespace flagfft {

int64_t four_step_col_inner_pack_for(int64_t n1, int64_t n2, const std::string &dtype);
std::string batch_bucket(int64_t batch);
bool env_flag_enabled(const char *value);
std::optional<std::filesystem::path> tuned_db_path();

}  // namespace flagfft
