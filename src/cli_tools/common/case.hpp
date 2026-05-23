#pragma once

#include <string>
#include <vector>

#include "cli_utils.hpp"

namespace flagfft::cli {

struct CaseSpec {
    FftApi api = FftApi::C2C;
    std::vector<int> shape{16};
    int batch = 1;
    int direction = FLAGFFT_FORWARD;
    Placement placement = Placement::OutOfPlace;
    PlanApi plan_api = PlanApi::Plan1d;
    bool stream = false;
};

enum class Operation { Test, Bench, Tune };

std::vector<int> parse_shape(const std::string &value);
json case_json(const CaseSpec &spec);
std::string unsupported_reason(const CaseSpec &spec, Operation operation);
flagfftType flagfft_type(FftApi api);
cufftType cufft_type(FftApi api);
bool is_complex_api(FftApi api);
bool is_double_api(FftApi api);
bool is_real_forward_api(FftApi api);
bool is_real_inverse_api(FftApi api);

}  // namespace flagfft::cli
