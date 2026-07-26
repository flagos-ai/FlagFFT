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

#include <string>

#include <nlohmann/json.hpp>

namespace flagfft::cli::tune {

struct TuneOptions {
  int length = 1 << 20;
  int batch = 1;
  int max_candidates = 5;
  int finalists = 2;
  int screen_warmup = 10;
  int screen_iters = 50;
  int final_warmup = 50;
  int final_iters = 1000;
  std::string db_path;
  bool save = true;
};

nlohmann::json run_decomposition_tune(const TuneOptions& options);

}  // namespace flagfft::cli::tune
