#pragma once

#include <string>

#include "case.hpp"

namespace flagfft::cli {

struct TimingConfig {
    int warmup = 10;
    int iters = 100;
    int launches_per_sample = 10;
};

json run_correctness(const CaseSpec &spec, bool include_path = false);
json run_benchmark(const CaseSpec &spec, const TimingConfig &timing, bool include_path);

}  // namespace flagfft::cli
