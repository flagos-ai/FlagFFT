#pragma once

#include <cuda_runtime_api.h>
#include <cufft.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "flagfft/flagfft.h"

namespace flagfft::cli {

using json = nlohmann::json;

inline constexpr int kExitPassed = 0;
inline constexpr int kExitFailed = 1;
inline constexpr int kExitRuntimeError = 2;
inline constexpr int kExitSkipped = 77;

enum class FftApi { C2C, Z2Z, R2C, D2Z, C2R, Z2D };
enum class Placement { InPlace, OutOfPlace };
enum class PlanApi { Auto, Tuned };

FftApi parse_fft_api(const std::string &value);
std::string fft_api_name(FftApi api);
Placement parse_placement(const std::string &value);
std::string placement_name(Placement p);
PlanApi parse_plan_api(const std::string &value);
std::string plan_api_name(PlanApi api);

class CliException : public std::runtime_error {
public:
    CliException(std::string message, int exit_code);
    int exit_code() const noexcept;

private:
    int exit_code_;
};

class AssertionFailure : public CliException {
public:
    explicit AssertionFailure(std::string message);
};

std::string flagfft_result_name(flagfftResult result);
std::string cufft_result_name(cufftResult result);
std::string direction_name(int direction);
int parse_direction(const std::string &value);

bool has_cuda_device(std::string &reason);
std::vector<std::string> split_csv(const std::string &value);
std::vector<int> parse_lengths_csv(const std::string &value);
std::string shell_quote(const std::string &value);
std::string executable_path(const char *argv0);
std::string executable_dir(const char *argv0);
std::string default_tune_db(const char *argv0);
std::string default_tune_command(const char *argv0);

void check_cuda(cudaError_t result, const std::string &context);
void check_cufft(cufftResult result, const std::string &context);
void check_flagfft(flagfftResult result, const std::string &context);
void expect_flagfft(flagfftResult actual, flagfftResult expected, const std::string &context);
void expect_true(bool condition, const std::string &context);

int exit_code_for_report(const json &report);
int emit_json_report(json report);

}  // namespace flagfft::cli
