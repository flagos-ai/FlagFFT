#pragma once

#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "flagfft.h"

namespace flagfft::cli {

using json = nlohmann::json;

inline constexpr int kExitPassed = 0;
inline constexpr int kExitFailed = 1;
inline constexpr int kExitRuntimeError = 2;
inline constexpr int kExitSkipped = 77;

enum class FftApi { C2C, Z2Z, R2C, D2Z, C2R, Z2D };
enum class Placement { InPlace, OutOfPlace };

FftApi parse_fft_api(const std::string& value);
std::string fft_api_name(FftApi api);
Placement parse_placement(const std::string& value);
std::string placement_name(Placement p);

int parse_rank(const std::string& value);

int parse_positive(const std::string& name, const char* value, bool allow_zero = false);

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
std::string direction_name(int direction);
int parse_direction(const std::string& value);

bool has_cuda_device(std::string& reason);

void check_flagfft(flagfftResult result, const std::string& context);

}  // namespace flagfft::cli
