#pragma once

#include <string>
#include <vector>

#include "cli_tools/common/case.hpp"

namespace flagfft::cli {

json run_tune_cases(const std::vector<CaseSpec> &cases,
                    const std::string &db_path,
                    int warmup,
                    int iters,
                    int static_limit,
                    int finalists,
                    bool retune);

}  // namespace flagfft::cli
