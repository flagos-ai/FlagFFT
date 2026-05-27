#include "case.hpp"

#include <sstream>

namespace flagfft::cli {

std::vector<int> parse_shape(const std::string& value) {
  std::vector<int> shape;
  std::size_t start = 0;
  while (start <= value.size()) {
    const std::size_t end = value.find_first_of("xX", start);
    const std::string part = value.substr(start, end == std::string::npos ? end : end - start);
    if (part.empty()) {
      throw AssertionFailure("invalid --shape: " + value);
    }
    try {
      std::size_t consumed = 0;
      const int n = std::stoi(part, &consumed);
      if (consumed != part.size()) {
        throw AssertionFailure("invalid --shape: " + value);
      }
      if (n <= 0) {
        throw AssertionFailure("shape dimensions must be positive");
      }
      shape.push_back(n);
    } catch (const std::invalid_argument&) {
      throw AssertionFailure("invalid --shape: " + value);
    } catch (const std::out_of_range&) {
      throw AssertionFailure("invalid --shape: " + value);
    }
    if (end == std::string::npos) break;
    start = end + 1;
  }
  if (shape.empty() || shape.size() > 3) {
    throw AssertionFailure("--shape rank must be between 1 and 3");
  }
  return shape;
}

json case_json(const CaseSpec& spec) {
  return {
      {      "api",         fft_api_name(spec.api)},
      {    "shape",                     spec.shape},
      {     "rank",                      spec.rank},
      {    "batch",                     spec.batch},
      {"direction", direction_name(spec.direction)},
      {"placement", placement_name(spec.placement)},
  };
}

bool is_complex_api(FftApi api) {
  return api == FftApi::C2C || api == FftApi::Z2Z;
}

bool is_double_api(FftApi api) {
  return api == FftApi::Z2Z || api == FftApi::D2Z || api == FftApi::Z2D;
}

bool is_real_forward_api(FftApi api) {
  return api == FftApi::R2C || api == FftApi::D2Z;
}

bool is_real_inverse_api(FftApi api) {
  return api == FftApi::C2R || api == FftApi::Z2D;
}

flagfftType flagfft_type(FftApi api) {
  switch (api) {
    case FftApi::C2C:
      return FLAGFFT_C2C;
    case FftApi::Z2Z:
      return FLAGFFT_Z2Z;
    case FftApi::R2C:
      return FLAGFFT_R2C;
    case FftApi::D2Z:
      return FLAGFFT_D2Z;
    case FftApi::C2R:
      return FLAGFFT_C2R;
    case FftApi::Z2D:
      return FLAGFFT_Z2D;
  }
  return FLAGFFT_C2C;
}

}  // namespace flagfft::cli
