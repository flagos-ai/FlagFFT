#include "cli_utils.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>

#include <unistd.h>

#include "adaptor/adaptor.h"

namespace flagfft::cli {

FftApi parse_fft_api(const std::string &value) {
  if (value == "c2c" || value == "C2C") return FftApi::C2C;
  if (value == "z2z" || value == "Z2Z") return FftApi::Z2Z;
  if (value == "r2c" || value == "R2C") return FftApi::R2C;
  if (value == "d2z" || value == "D2Z") return FftApi::D2Z;
  if (value == "c2r" || value == "C2R") return FftApi::C2R;
  if (value == "z2d" || value == "Z2D") return FftApi::Z2D;
  throw AssertionFailure("unknown FFT API: " + value);
}

std::string fft_api_name(FftApi api) {
  switch (api) {
    case FftApi::C2C:
      return "c2c";
    case FftApi::Z2Z:
      return "z2z";
    case FftApi::R2C:
      return "r2c";
    case FftApi::D2Z:
      return "d2z";
    case FftApi::C2R:
      return "c2r";
    case FftApi::Z2D:
      return "z2d";
  }
  return "unknown";
}

Placement parse_placement(const std::string &value) {
  if (value == "in-place" || value == "inplace" || value == "in" || value == "ip") return Placement::InPlace;
  if (value == "out-of-place" || value == "outofplace" || value == "out" || value == "oop")
    return Placement::OutOfPlace;
  throw AssertionFailure("unknown placement: " + value);
}

std::string placement_name(Placement p) {
  return p == Placement::InPlace ? "in-place" : "out-of-place";
}

int parse_rank(const std::string &value) {
  if (value == "1") return 1;
  if (value == "2") return 2;
  if (value == "3") return 3;
  throw AssertionFailure("invalid --rank: " + value + " (expected 1, 2, or 3)");
}

std::string rank_name(int rank) {
  return std::to_string(rank);
}

CliException::CliException(std::string message, int exit_code)
    : std::runtime_error(std::move(message)), exit_code_(exit_code) {
}

int CliException::exit_code() const noexcept {
  return exit_code_;
}

AssertionFailure::AssertionFailure(std::string message) : CliException(std::move(message), kExitFailed) {
}

std::string flagfft_result_name(flagfftResult result) {
  switch (result) {
    case FLAGFFT_SUCCESS:
      return "FLAGFFT_SUCCESS";
    case FLAGFFT_INVALID_PLAN:
      return "FLAGFFT_INVALID_PLAN";
    case FLAGFFT_ALLOC_FAILED:
      return "FLAGFFT_ALLOC_FAILED";
    case FLAGFFT_INVALID_TYPE:
      return "FLAGFFT_INVALID_TYPE";
    case FLAGFFT_INVALID_VALUE:
      return "FLAGFFT_INVALID_VALUE";
    case FLAGFFT_INTERNAL_ERROR:
      return "FLAGFFT_INTERNAL_ERROR";
    case FLAGFFT_EXEC_FAILED:
      return "FLAGFFT_EXEC_FAILED";
    case FLAGFFT_SETUP_FAILED:
      return "FLAGFFT_SETUP_FAILED";
    case FLAGFFT_INVALID_SIZE:
      return "FLAGFFT_INVALID_SIZE";
    case FLAGFFT_UNALIGNED_DATA:
      return "FLAGFFT_UNALIGNED_DATA";
    case FLAGFFT_INCOMPLETE_PARAMETER_LIST:
      return "FLAGFFT_INCOMPLETE_PARAMETER_LIST";
    case FLAGFFT_INVALID_DEVICE:
      return "FLAGFFT_INVALID_DEVICE";
    case FLAGFFT_PARSE_ERROR:
      return "FLAGFFT_PARSE_ERROR";
    case FLAGFFT_NO_WORKSPACE:
      return "FLAGFFT_NO_WORKSPACE";
    case FLAGFFT_NOT_SUPPORTED:
      return "FLAGFFT_NOT_SUPPORTED";
  }
  return "FLAGFFT_UNKNOWN";
}

std::string direction_name(int direction) {
  return direction == FLAGFFT_INVERSE ? "inverse" : "forward";
}

int parse_direction(const std::string &value) {
  if (value == "forward" || value == "fwd" || value == "fft") {
    return FLAGFFT_FORWARD;
  }
  if (value == "inverse" || value == "inv" || value == "ifft") {
    return FLAGFFT_INVERSE;
  }
  throw AssertionFailure("unknown direction: " + value);
}

bool has_cuda_device(std::string &reason) {
  try {
    if (adaptor::device_count() > 0) {
      return true;
    }
    reason = "no CUDA device available";
    return false;
  } catch (const std::exception &error) {
    throw CliException(error.what(), kExitRuntimeError);
  }
}

std::string shell_quote(const std::string &value) {
  std::string out = "'";
  for (char c : value) {
    if (c == '\'') {
      out += "'\\''";
    } else {
      out += c;
    }
  }
  out += "'";
  return out;
}

std::string executable_path(const char *argv0) {
  std::array<char, 4096> buffer {};
  ssize_t n = readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
  if (n > 0) {
    buffer[static_cast<std::size_t>(n)] = '\0';
    return buffer.data();
  }
  std::filesystem::path from_argv = argv0 == nullptr ? std::filesystem::path {} : argv0;
  if (from_argv.empty()) {
    return (std::filesystem::current_path() / "flagfft-cli").string();
  }
  std::error_code ec;
  return std::filesystem::absolute(from_argv, ec).string();
}

std::string executable_dir(const char *argv0) {
  return std::filesystem::path(executable_path(argv0)).parent_path().string();
}

void check_flagfft(flagfftResult result, const std::string &context) {
  if (result != FLAGFFT_SUCCESS) {
    std::ostringstream oss;
    oss << context << ": " << flagfft_result_name(result) << " (" << static_cast<int>(result) << ")";
    throw CliException(oss.str(), kExitRuntimeError);
  }
}

void expect_flagfft(flagfftResult actual, flagfftResult expected, const std::string &context) {
  if (actual != expected) {
    std::ostringstream oss;
    oss << context << ": expected " << flagfft_result_name(expected) << ", got "
        << flagfft_result_name(actual) << " (" << static_cast<int>(actual) << ")";
    throw AssertionFailure(oss.str());
  }
}

void expect_true(bool condition, const std::string &context) {
  if (!condition) {
    throw AssertionFailure(context);
  }
}

int exit_code_for_report(const json &report) {
  if (report.contains("_exit_code")) {
    return report.at("_exit_code").get<int>();
  }
  const std::string status = report.value("status", "failed");
  if (status == "passed") {
    return kExitPassed;
  }
  if (status == "skipped" || status == "unsupported") {
    return kExitSkipped;
  }
  return kExitFailed;
}

int emit_json_report(json report) {
  const int code = exit_code_for_report(report);
  report.erase("_exit_code");
  std::cout << report.dump() << "\n";
  return code;
}

}  // namespace flagfft::cli
