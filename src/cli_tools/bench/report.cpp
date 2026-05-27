#include "cli_tools/bench/report.hpp"

#include <iomanip>
#include <sstream>

#include "cli_tools/common/cli_utils.hpp"

namespace flagfft::cli::bench {

std::string format_table(const std::vector<CaseSpec>& cases, const std::vector<BenchResult>& results) {
  std::ostringstream out;
  out << std::left << std::setw(12) << "shape" << std::setw(8) << "api" << std::setw(7) << "batch"
      << std::setw(11) << "direction" << std::setw(14) << "placement" << std::setw(17) << "flagfft_median_ms"
      << std::setw(14) << "ref_median_ms"
      << "speedup\n";

  for (std::size_t i = 0; i < cases.size(); ++i) {
    const auto& spec = cases[i];
    const auto& r = results[i];

    std::string shape_str;
    for (std::size_t j = 0; j < spec.shape.size(); ++j) {
      if (j > 0) shape_str += "x";
      shape_str += std::to_string(spec.shape[j]);
    }

    out << std::left << std::setw(12) << shape_str << std::setw(8) << fft_api_name(spec.api) << std::setw(7)
        << spec.batch << std::setw(11) << direction_name(spec.direction) << std::setw(14)
        << placement_name(spec.placement) << std::fixed << std::setprecision(4) << std::setw(17)
        << r.flagfft.median_ms << std::setw(14) << r.reference.median_ms << std::setprecision(2) << r.speedup
        << "x\n";
  }
  return out.str();
}

nlohmann::json format_json(const std::vector<CaseSpec>& cases,
                           const std::vector<BenchResult>& results,
                           int warmup,
                           int iters) {
  json j_cases = json::array();
  for (std::size_t i = 0; i < cases.size(); ++i) {
    const auto& spec = cases[i];
    const auto& r = results[i];
    json entry = case_json(spec);
    entry["timing"] = {
        {"flagfft_median_ms",   r.flagfft.median_ms},
        {   "flagfft_p90_ms",      r.flagfft.p90_ms},
        {    "ref_median_ms", r.reference.median_ms},
        {       "ref_p90_ms",    r.reference.p90_ms},
        {          "speedup",             r.speedup},
        {           "warmup",                warmup},
        {            "iters",                 iters},
    };
    if (!r.plan_description.empty()) {
      entry["plan_description"] = r.plan_description;
    }
    j_cases.push_back(entry);
  }
  return {
      { "status", "passed"},
      {"command",  "bench"},
      {  "cases",  j_cases},
  };
}

}  // namespace flagfft::cli::bench
