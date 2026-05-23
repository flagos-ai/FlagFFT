#include "cli_utils.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>

#include <unistd.h>

namespace flagfft::cli {
namespace {

std::string cuda_error_message(cudaError_t result, const std::string &context) {
    std::ostringstream oss;
    oss << context << ": CUDA " << static_cast<int>(result)
        << " (" << cudaGetErrorString(result) << ")";
    return oss.str();
}

}  // namespace

FftApi parse_fft_api(const std::string &value) {
    if (value == "C2C") return FftApi::C2C;
    if (value == "Z2Z") return FftApi::Z2Z;
    if (value == "R2C") return FftApi::R2C;
    if (value == "D2Z") return FftApi::D2Z;
    if (value == "C2R") return FftApi::C2R;
    if (value == "Z2D") return FftApi::Z2D;
    throw AssertionFailure("unknown FFT API: " + value);
}

std::string fft_api_name(FftApi api) {
    switch (api) {
        case FftApi::C2C: return "C2C";
        case FftApi::Z2Z: return "Z2Z";
        case FftApi::R2C: return "R2C";
        case FftApi::D2Z: return "D2Z";
        case FftApi::C2R: return "C2R";
        case FftApi::Z2D: return "Z2D";
    }
    return "unknown";
}

Placement parse_placement(const std::string &value) {
    if (value == "inplace" || value == "in" || value == "ip") return Placement::InPlace;
    if (value == "outofplace" || value == "out" || value == "oop") return Placement::OutOfPlace;
    throw AssertionFailure("unknown placement: " + value);
}

std::string placement_name(Placement p) {
    return p == Placement::InPlace ? "inplace" : "outofplace";
}

PlanApi parse_plan_api(const std::string &value) {
    if (value == "auto") return PlanApi::Auto;
    if (value == "tuned") return PlanApi::Tuned;
    throw AssertionFailure("unknown plan API: " + value);
}

std::string plan_api_name(PlanApi api) {
    return api == PlanApi::Auto ? "auto" : "tuned";
}

CliException::CliException(std::string message, int exit_code)
    : std::runtime_error(std::move(message)), exit_code_(exit_code) {}

int CliException::exit_code() const noexcept {
    return exit_code_;
}

AssertionFailure::AssertionFailure(std::string message)
    : CliException(std::move(message), kExitFailed) {}

std::string flagfft_result_name(flagfftResult result) {
    switch (result) {
        case FLAGFFT_SUCCESS: return "FLAGFFT_SUCCESS";
        case FLAGFFT_INVALID_PLAN: return "FLAGFFT_INVALID_PLAN";
        case FLAGFFT_ALLOC_FAILED: return "FLAGFFT_ALLOC_FAILED";
        case FLAGFFT_INVALID_TYPE: return "FLAGFFT_INVALID_TYPE";
        case FLAGFFT_INVALID_VALUE: return "FLAGFFT_INVALID_VALUE";
        case FLAGFFT_INTERNAL_ERROR: return "FLAGFFT_INTERNAL_ERROR";
        case FLAGFFT_EXEC_FAILED: return "FLAGFFT_EXEC_FAILED";
        case FLAGFFT_SETUP_FAILED: return "FLAGFFT_SETUP_FAILED";
        case FLAGFFT_INVALID_SIZE: return "FLAGFFT_INVALID_SIZE";
        case FLAGFFT_UNALIGNED_DATA: return "FLAGFFT_UNALIGNED_DATA";
        case FLAGFFT_INCOMPLETE_PARAMETER_LIST: return "FLAGFFT_INCOMPLETE_PARAMETER_LIST";
        case FLAGFFT_INVALID_DEVICE: return "FLAGFFT_INVALID_DEVICE";
        case FLAGFFT_PARSE_ERROR: return "FLAGFFT_PARSE_ERROR";
        case FLAGFFT_NO_WORKSPACE: return "FLAGFFT_NO_WORKSPACE";
        case FLAGFFT_NOT_SUPPORTED: return "FLAGFFT_NOT_SUPPORTED";
    }
    return "FLAGFFT_UNKNOWN";
}

std::string cufft_result_name(cufftResult result) {
    switch (result) {
        case CUFFT_SUCCESS: return "CUFFT_SUCCESS";
        case CUFFT_INVALID_PLAN: return "CUFFT_INVALID_PLAN";
        case CUFFT_ALLOC_FAILED: return "CUFFT_ALLOC_FAILED";
        case CUFFT_INVALID_TYPE: return "CUFFT_INVALID_TYPE";
        case CUFFT_INVALID_VALUE: return "CUFFT_INVALID_VALUE";
        case CUFFT_INTERNAL_ERROR: return "CUFFT_INTERNAL_ERROR";
        case CUFFT_EXEC_FAILED: return "CUFFT_EXEC_FAILED";
        case CUFFT_SETUP_FAILED: return "CUFFT_SETUP_FAILED";
        case CUFFT_INVALID_SIZE: return "CUFFT_INVALID_SIZE";
        case CUFFT_UNALIGNED_DATA: return "CUFFT_UNALIGNED_DATA";
        case CUFFT_INVALID_DEVICE: return "CUFFT_INVALID_DEVICE";
        case CUFFT_NO_WORKSPACE: return "CUFFT_NO_WORKSPACE";
        case CUFFT_NOT_IMPLEMENTED: return "CUFFT_NOT_IMPLEMENTED";
        case CUFFT_NOT_SUPPORTED: return "CUFFT_NOT_SUPPORTED";
    }
    return "CUFFT_UNKNOWN";
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
    int count = 0;
    cudaError_t result = cudaGetDeviceCount(&count);
    if (result != cudaSuccess) {
        reason = cuda_error_message(result, "cudaGetDeviceCount");
        return false;
    }
    if (count <= 0) {
        reason = "no CUDA device available";
        return false;
    }
    return true;
}

std::vector<std::string> split_csv(const std::string &value) {
    std::vector<std::string> out;
    std::size_t i = 0;
    while (i <= value.size()) {
        std::size_t j = value.find(',', i);
        if (j == std::string::npos) {
            j = value.size();
        }
        if (j > i) {
            out.push_back(value.substr(i, j - i));
        }
        if (j == value.size()) {
            break;
        }
        i = j + 1;
    }
    return out;
}

std::vector<int> parse_lengths_csv(const std::string &value) {
    std::vector<int> out;
    for (const std::string &part : split_csv(value)) {
        int n = std::stoi(part);
        if (n <= 0) {
            throw AssertionFailure("FFT lengths must be positive");
        }
        out.push_back(n);
    }
    if (out.empty()) {
        throw AssertionFailure("--lengths produced no values");
    }
    return out;
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
    std::array<char, 4096> buffer{};
    ssize_t n = readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
    if (n > 0) {
        buffer[static_cast<std::size_t>(n)] = '\0';
        return buffer.data();
    }
    std::filesystem::path from_argv = argv0 == nullptr ? std::filesystem::path{} : argv0;
    if (from_argv.empty()) {
        return (std::filesystem::current_path() / "flagfft-test").string();
    }
    std::error_code ec;
    return std::filesystem::absolute(from_argv, ec).string();
}

std::string executable_dir(const char *argv0) {
    return std::filesystem::path(executable_path(argv0)).parent_path().string();
}

std::string default_tune_db(const char *argv0) {
    return (std::filesystem::path(executable_dir(argv0)) / ".flagfft" /
            "tuned_plans.sqlite")
        .string();
}

std::string default_tune_command(const char *argv0) {
    std::filesystem::path sibling =
        std::filesystem::path(executable_dir(argv0)) / "flagfft-tuner";
    std::error_code ec;
    if (std::filesystem::is_regular_file(sibling, ec)) {
        return sibling.string();
    }
    return "flagfft-tuner";
}

void check_cuda(cudaError_t result, const std::string &context) {
    if (result != cudaSuccess) {
        throw CliException(cuda_error_message(result, context), kExitRuntimeError);
    }
}

void check_cufft(cufftResult result, const std::string &context) {
    if (result != CUFFT_SUCCESS) {
        std::ostringstream oss;
        oss << context << ": " << cufft_result_name(result)
            << " (" << static_cast<int>(result) << ")";
        throw CliException(oss.str(), kExitRuntimeError);
    }
}

void check_flagfft(flagfftResult result, const std::string &context) {
    if (result != FLAGFFT_SUCCESS) {
        std::ostringstream oss;
        oss << context << ": " << flagfft_result_name(result)
            << " (" << static_cast<int>(result) << ")";
        throw CliException(oss.str(), kExitRuntimeError);
    }
}

void expect_flagfft(flagfftResult actual, flagfftResult expected, const std::string &context) {
    if (actual != expected) {
        std::ostringstream oss;
        oss << context << ": expected " << flagfft_result_name(expected)
            << ", got " << flagfft_result_name(actual)
            << " (" << static_cast<int>(actual) << ")";
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
    if (status == "skipped") {
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
