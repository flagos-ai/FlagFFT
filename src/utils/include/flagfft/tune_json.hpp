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

#include "flagfft/exec.hpp"

#include <nlohmann/json.hpp>

namespace flagfft {

struct TuneFingerprints {
  std::string planner;
  std::string codegen;
  std::string runtime;
  std::string benchmark;
};

TuneFingerprints tune_fingerprints();

nlohmann::json plan_node_to_json(const PlanNodePtr &node);
nlohmann::json plan_key_to_json(const PlanKey &key);
nlohmann::json problem_key_to_json(const ProblemKey &key);
nlohmann::json request_to_json(const FFTRequest &request);
nlohmann::json wrap_plan_json(const PlanNodePtr &root, const FFTRequest &request, const std::string &source);
std::optional<nlohmann::json> lookup_tuned_plan_json(const FFTRequest &request);
PlanNodePtr plan_node_from_json(PlanBuilder &builder, const nlohmann::json &node);

}  // namespace flagfft
