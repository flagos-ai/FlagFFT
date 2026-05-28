#pragma once

#include <string>
#include <vector>

#include "cli_tools/bench/runner.hpp"
#include "cli_tools/common/case.hpp"

namespace flagfft::cli::bench {

std::string format_table(const std::vector<CaseSpec>& cases,
                         const std::vector<BenchResult>& results,
                         int warmup,
                         int iters);

nlohmann::json format_json(const std::vector<CaseSpec>& cases,
                           const std::vector<BenchResult>& results,
                           int warmup,
                           int iters);

}  // namespace flagfft::cli::bench
