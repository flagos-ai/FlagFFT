#pragma once

#include <string>
#include <vector>

#include "cli_tools/common/case.hpp"

namespace flagfft::cli::bench {

struct TimingStats {
  double median_ms = 0.0;
  double p90_ms = 0.0;
  std::vector<float> samples;
};

struct BenchResult {
  TimingStats flagfft;
  TimingStats reference;
  double speedup = 0.0;
  std::string plan_description;
};

BenchResult run_benchmark(const CaseSpec& spec, int warmup, int iters, bool include_path);

}  // namespace flagfft::cli::bench
