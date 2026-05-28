#pragma once

#include <string>
#include <vector>

#include "cli_utils.hpp"

namespace flagfft::cli {

struct CaseSpec {
  FftApi api = FftApi::C2C;
  std::vector<int> shape {16};
  int batch = 1;
  int direction = FLAGFFT_FORWARD;
  Placement placement = Placement::OutOfPlace;
  int rank = 1;
};

std::vector<int> parse_shape(const std::string& value);
json case_json(const CaseSpec& spec);
flagfftType flagfft_type(FftApi api);
bool is_complex_api(FftApi api);
bool is_double_api(FftApi api);
bool is_real_forward_api(FftApi api);
bool is_real_inverse_api(FftApi api);

}  // namespace flagfft::cli
