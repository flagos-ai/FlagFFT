#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace flagfft {
namespace tune {

struct BenchTiming {
    double median_ms = 0.0;
    double p90_ms = 0.0;
    double first_call_ms = 0.0;
    double compile_ms = 0.0;
};

struct BenchError {
    double max_abs = 0.0;
    double rms = 0.0;
};

struct TuneConfig {
    std::vector<int64_t> lengths;
    int64_t batch = 1;
    int n_warmup = 10;
    int n_iters = 100;
    int static_limit = 32;
    int finalists = 3;
    std::string direction = "forward";
    std::string db_path;
    bool retune = false;
};

std::vector<float> generate_random_input(int64_t n, int64_t batch);

BenchTiming bench_candidate(int64_t n, int64_t batch, const std::string &direction,
                            int n_warmup, int n_iters,
                            const std::string &plan_json_str,
                            int device_index, const std::string &device_arch);

BenchError verify_against_cufft(int64_t n, int64_t batch, const std::string &direction,
                                const std::string &plan_json_str,
                                int device_index, const std::string &device_arch);

}  // namespace tune
}  // namespace flagfft
