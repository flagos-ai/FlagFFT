#include "case.hpp"

#include <sstream>

namespace flagfft::cli {

std::vector<int> parse_shape(const std::string &value) {
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
    } catch (const std::invalid_argument &) {
      throw AssertionFailure("invalid --shape: " + value);
    } catch (const std::out_of_range &) {
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

json case_json(const CaseSpec &spec) {
  return {
      {      "api",         fft_api_name(spec.api)},
      {    "shape",                     spec.shape},
      {     "rank",              spec.shape.size()},
      {    "batch",                     spec.batch},
      {"direction", direction_name(spec.direction)},
      {"placement", placement_name(spec.placement)},
      { "plan_api",   plan_api_name(spec.plan_api)},
      {   "stream",                    spec.stream},
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

std::string unsupported_reason(const CaseSpec &spec, Operation operation) {
  if (spec.shape.size() != 1) {
    return "rank-2 and rank-3 transforms are expressible but not implemented";
  }
  if (operation == Operation::Tune) {
    if (spec.api != FftApi::C2C || spec.placement != Placement::OutOfPlace ||
        spec.plan_api != PlanApi::Plan1d) {
      return "tuning currently supports only 1D out-of-place c2c with plan1d";
    }
    return {};
  }
  if (spec.plan_api == PlanApi::Plan2d || spec.plan_api == PlanApi::Plan3d) {
    return "plan2d and plan3d are not implemented";
  }
  if (spec.plan_api == PlanApi::PlanMany &&
      !(spec.placement == Placement::InPlace &&
        (is_real_forward_api(spec.api) || is_real_inverse_api(spec.api)))) {
    return "planmany is enabled only for verified in-place padded real layouts";
  }
  if (is_real_forward_api(spec.api) && spec.direction != FLAGFFT_FORWARD) {
    return "real-to-complex APIs support only forward direction";
  }
  if (is_real_inverse_api(spec.api) && spec.direction != FLAGFFT_INVERSE) {
    return "complex-to-real APIs support only inverse direction";
  }
  return {};
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

cufftType cufft_type(FftApi api) {
  switch (api) {
    case FftApi::C2C:
      return CUFFT_C2C;
    case FftApi::Z2Z:
      return CUFFT_Z2Z;
    case FftApi::R2C:
      return CUFFT_R2C;
    case FftApi::D2Z:
      return CUFFT_D2Z;
    case FftApi::C2R:
      return CUFFT_C2R;
    case FftApi::Z2D:
      return CUFFT_Z2D;
  }
  return CUFFT_C2C;
}

}  // namespace flagfft::cli
