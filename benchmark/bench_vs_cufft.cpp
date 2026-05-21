// FlagFFT vs cuFFT benchmark.
//
// Measures only the warm exec time of flagfftExecC2C / cufftExecC2C using CUDA
// events. Plan creation, allocation, host-device copies, and warmup iterations
// are explicitly excluded from the timed region.

#include <cuda_runtime_api.h>
#include <cufft.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

#include "flagfft/flagfft.h"

namespace {

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _err = (expr);                                             \
        if (_err != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(_err));                            \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

#define FLAGFFT_CHECK(expr)                                                    \
    do {                                                                       \
        flagfftResult _res = (expr);                                           \
        if (_res != FLAGFFT_SUCCESS) {                                         \
            std::fprintf(stderr, "FlagFFT error %s:%d: code=%d\n", __FILE__,   \
                         __LINE__, static_cast<int>(_res));                    \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

#define CUFFT_CHECK(expr)                                                      \
    do {                                                                       \
        cufftResult _res = (expr);                                             \
        if (_res != CUFFT_SUCCESS) {                                           \
            std::fprintf(stderr, "cuFFT error %s:%d: code=%d\n", __FILE__,     \
                         __LINE__, static_cast<int>(_res));                    \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

struct BenchConfig {
    std::vector<int> lengths{256, 1024, 4096, 8192, 16384, 65536};
    std::string type = "c2c";
    int batch = 64;
    int n_warmup = 10;
    int n_iters = 100;
    int launches_per_sample = 10;
    int direction = FLAGFFT_FORWARD;
    bool tune = false;
    bool retune = false;
    std::string tune_db;
    std::string tune_command = "flagfft_tune";
    int tune_static_limit = 32;
    int tune_finalists = 3;
    bool print_path = false;
    bool inplace = false;
};

struct CudaStream {
    cudaStream_t stream{};
    CudaStream() {
        CUDA_CHECK(cudaStreamCreate(&stream));
    }
    ~CudaStream() {
        cudaStreamDestroy(stream);
    }
    CudaStream(const CudaStream &) = delete;
    CudaStream &operator=(const CudaStream &) = delete;
};

struct CudaTimer {
    cudaEvent_t start{};
    cudaEvent_t stop{};
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }
    ~CudaTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    CudaTimer(const CudaTimer &) = delete;
    CudaTimer &operator=(const CudaTimer &) = delete;

    void record_start(cudaStream_t s = nullptr) {
        CUDA_CHECK(cudaEventRecord(start, s));
    }
    void record_stop(cudaStream_t s = nullptr) {
        CUDA_CHECK(cudaEventRecord(stop, s));
    }
    float elapsed_ms() {
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

std::vector<flagfftComplex> sample_input(int n, int batch) {
    std::mt19937 rng(1234 + n * 17 + batch);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<flagfftComplex> data(static_cast<std::size_t>(n) * batch);
    for (auto &v : data) {
        v.x = dist(rng);
        v.y = dist(rng);
    }
    return data;
}

std::vector<flagfftDoubleComplex> sample_input_z(int n, int batch) {
    std::mt19937 rng(4321 + n * 19 + batch);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<flagfftDoubleComplex> data(static_cast<std::size_t>(n) * batch);
    for (auto &v : data) {
        v.x = dist(rng);
        v.y = dist(rng);
    }
    return data;
}

std::vector<flagfftReal> sample_input_r(int n, int batch, int distance) {
    std::mt19937 rng(2468 + n * 13 + batch);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<flagfftReal> data(static_cast<std::size_t>(distance) * batch, 0.0f);
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < n; ++i) {
            data[static_cast<std::size_t>(b) * distance + i] = dist(rng);
        }
    }
    return data;
}

std::vector<flagfftDoubleReal> sample_input_d(int n, int batch, int distance) {
    std::mt19937 rng(8642 + n * 13 + batch);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<flagfftDoubleReal> data(static_cast<std::size_t>(distance) * batch, 0.0);
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < n; ++i) {
            data[static_cast<std::size_t>(b) * distance + i] = dist(rng);
        }
    }
    return data;
}

template <typename ComplexT>
std::vector<ComplexT> sample_compact_complex(int half, int batch) {
    std::mt19937 rng(9753 + half * 23 + batch);
    using Scalar = decltype(ComplexT{}.x);
    std::uniform_real_distribution<Scalar> dist(static_cast<Scalar>(-1.0), static_cast<Scalar>(1.0));
    std::vector<ComplexT> data(static_cast<std::size_t>(half) * batch);
    for (auto &v : data) {
        v.x = dist(rng);
        v.y = dist(rng);
    }
    return data;
}

float median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct BenchResult {
    float flagfft_ms;
    float cufft_ms;
};

flagfftType flagfft_type_for(const std::string &type) {
    if (type == "c2c") return FLAGFFT_C2C;
    if (type == "z2z") return FLAGFFT_Z2Z;
    if (type == "r2c") return FLAGFFT_R2C;
    if (type == "d2z") return FLAGFFT_D2Z;
    if (type == "c2r") return FLAGFFT_C2R;
    if (type == "z2d") return FLAGFFT_Z2D;
    throw std::runtime_error("unsupported benchmark type: " + type);
}

cufftType cufft_type_for(const std::string &type) {
    if (type == "c2c") return CUFFT_C2C;
    if (type == "z2z") return CUFFT_Z2Z;
    if (type == "r2c") return CUFFT_R2C;
    if (type == "d2z") return CUFFT_D2Z;
    if (type == "c2r") return CUFFT_C2R;
    if (type == "z2d") return CUFFT_Z2D;
    throw std::runtime_error("unsupported benchmark type: " + type);
}

bool is_complex_type(const std::string &type) {
    return type == "c2c" || type == "z2z";
}

bool is_double_type(const std::string &type) {
    return type == "z2z" || type == "d2z" || type == "z2d";
}

bool is_real_forward_type(const std::string &type) {
    return type == "r2c" || type == "d2z";
}

void exec_flagfft(flagfftHandle plan, const BenchConfig &cfg, void *input, void *output) {
    if (cfg.type == "c2c") {
        FLAGFFT_CHECK(flagfftExecC2C(plan,
                                     static_cast<flagfftComplex *>(input),
                                     static_cast<flagfftComplex *>(output),
                                     cfg.direction));
    } else if (cfg.type == "z2z") {
        FLAGFFT_CHECK(flagfftExecZ2Z(plan,
                                     static_cast<flagfftDoubleComplex *>(input),
                                     static_cast<flagfftDoubleComplex *>(output),
                                     cfg.direction));
    } else if (cfg.type == "r2c") {
        FLAGFFT_CHECK(flagfftExecR2C(plan,
                                     static_cast<flagfftReal *>(input),
                                     static_cast<flagfftComplex *>(output)));
    } else if (cfg.type == "d2z") {
        FLAGFFT_CHECK(flagfftExecD2Z(plan,
                                     static_cast<flagfftDoubleReal *>(input),
                                     static_cast<flagfftDoubleComplex *>(output)));
    } else if (cfg.type == "c2r") {
        FLAGFFT_CHECK(flagfftExecC2R(plan,
                                     static_cast<flagfftComplex *>(input),
                                     static_cast<flagfftReal *>(output)));
    } else if (cfg.type == "z2d") {
        FLAGFFT_CHECK(flagfftExecZ2D(plan,
                                     static_cast<flagfftDoubleComplex *>(input),
                                     static_cast<flagfftDoubleReal *>(output)));
    }
}

void exec_cufft(cufftHandle plan, const BenchConfig &cfg, void *input, void *output) {
    const int cufft_dir = (cfg.direction == FLAGFFT_FORWARD) ? CUFFT_FORWARD : CUFFT_INVERSE;
    if (cfg.type == "c2c") {
        CUFFT_CHECK(cufftExecC2C(plan,
                                 static_cast<cufftComplex *>(input),
                                 static_cast<cufftComplex *>(output),
                                 cufft_dir));
    } else if (cfg.type == "z2z") {
        CUFFT_CHECK(cufftExecZ2Z(plan,
                                 static_cast<cufftDoubleComplex *>(input),
                                 static_cast<cufftDoubleComplex *>(output),
                                 cufft_dir));
    } else if (cfg.type == "r2c") {
        CUFFT_CHECK(cufftExecR2C(plan,
                                 static_cast<cufftReal *>(input),
                                 static_cast<cufftComplex *>(output)));
    } else if (cfg.type == "d2z") {
        CUFFT_CHECK(cufftExecD2Z(plan,
                                 static_cast<cufftDoubleReal *>(input),
                                 static_cast<cufftDoubleComplex *>(output)));
    } else if (cfg.type == "c2r") {
        CUFFT_CHECK(cufftExecC2R(plan,
                                 static_cast<cufftComplex *>(input),
                                 static_cast<cufftReal *>(output)));
    } else if (cfg.type == "z2d") {
        CUFFT_CHECK(cufftExecZ2D(plan,
                                 static_cast<cufftDoubleComplex *>(input),
                                 static_cast<cufftDoubleReal *>(output)));
    }
}

BenchResult bench_one(int n,
                      const BenchConfig &cfg) {
    const int half = n / 2 + 1;
    const int padded_real_distance = 2 * half;
    const bool fp64 = is_double_type(cfg.type);
    const bool real_forward = is_real_forward_type(cfg.type);
    const bool real_inverse = cfg.type == "c2r" || cfg.type == "z2d";
    const std::size_t real_bytes = fp64 ? sizeof(flagfftDoubleReal) : sizeof(flagfftReal);
    const std::size_t complex_bytes = fp64 ? sizeof(flagfftDoubleComplex) : sizeof(flagfftComplex);
    const std::size_t inplace_bytes =
        is_complex_type(cfg.type) ? static_cast<std::size_t>(cfg.batch) * n * complex_bytes
                                  : static_cast<std::size_t>(cfg.batch) * padded_real_distance * real_bytes;
    const std::size_t in_bytes =
        cfg.inplace ? inplace_bytes
                    : static_cast<std::size_t>(cfg.batch) *
                          (real_forward ? n * real_bytes : (real_inverse ? half * complex_bytes : n * complex_bytes));
    const std::size_t out_bytes =
        cfg.inplace ? in_bytes
                    : static_cast<std::size_t>(cfg.batch) *
                          (real_forward ? half * complex_bytes : (real_inverse ? n * real_bytes : n * complex_bytes));

    void *d_in_flagfft = nullptr;
    void *d_out_flagfft = nullptr;
    void *d_in_cufft = nullptr;
    void *d_out_cufft = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in_flagfft, in_bytes));
    CUDA_CHECK(cudaMalloc(&d_in_cufft, in_bytes));
    if (cfg.inplace) {
        d_out_flagfft = d_in_flagfft;
        d_out_cufft = d_in_cufft;
    } else {
        CUDA_CHECK(cudaMalloc(&d_out_flagfft, out_bytes));
        CUDA_CHECK(cudaMalloc(&d_out_cufft, out_bytes));
    }

    if (cfg.type == "c2c") {
        auto host = sample_input(n, cfg.batch);
        CUDA_CHECK(cudaMemcpy(d_in_flagfft, host.data(), host.size() * sizeof(flagfftComplex), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_in_cufft, host.data(), host.size() * sizeof(flagfftComplex), cudaMemcpyHostToDevice));
    } else if (cfg.type == "z2z") {
        auto host = sample_input_z(n, cfg.batch);
        CUDA_CHECK(cudaMemcpy(d_in_flagfft, host.data(), host.size() * sizeof(flagfftDoubleComplex), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_in_cufft, host.data(), host.size() * sizeof(flagfftDoubleComplex), cudaMemcpyHostToDevice));
    } else if (cfg.type == "r2c") {
        auto host = sample_input_r(n, cfg.batch, cfg.inplace ? padded_real_distance : n);
        CUDA_CHECK(cudaMemcpy(d_in_flagfft, host.data(), host.size() * sizeof(flagfftReal), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_in_cufft, host.data(), host.size() * sizeof(flagfftReal), cudaMemcpyHostToDevice));
    } else if (cfg.type == "d2z") {
        auto host = sample_input_d(n, cfg.batch, cfg.inplace ? padded_real_distance : n);
        CUDA_CHECK(cudaMemcpy(d_in_flagfft, host.data(), host.size() * sizeof(flagfftDoubleReal), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_in_cufft, host.data(), host.size() * sizeof(flagfftDoubleReal), cudaMemcpyHostToDevice));
    } else if (cfg.type == "c2r") {
        auto host = sample_compact_complex<flagfftComplex>(half, cfg.batch);
        CUDA_CHECK(cudaMemcpy(d_in_flagfft, host.data(), host.size() * sizeof(flagfftComplex), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_in_cufft, host.data(), host.size() * sizeof(flagfftComplex), cudaMemcpyHostToDevice));
    } else if (cfg.type == "z2d") {
        auto host = sample_compact_complex<flagfftDoubleComplex>(half, cfg.batch);
        CUDA_CHECK(cudaMemcpy(d_in_flagfft, host.data(), host.size() * sizeof(flagfftDoubleComplex), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_in_cufft, host.data(), host.size() * sizeof(flagfftDoubleComplex), cudaMemcpyHostToDevice));
    }
    CudaStream stream;

    flagfftHandle ff_plan = nullptr;
    FLAGFFT_CHECK(flagfftPlan1d(&ff_plan, n, flagfft_type_for(cfg.type), cfg.batch));
    FLAGFFT_CHECK(flagfftSetStream(ff_plan, stream.stream));

    if (cfg.print_path) {
        const char *desc = flagfftGetPlanDescription(ff_plan);
        if (desc != nullptr) {
            std::printf("%s\n", desc);
        }
    }

    cufftHandle cf_plan = 0;
    CUFFT_CHECK(cufftPlan1d(&cf_plan, n, cufft_type_for(cfg.type), cfg.batch));
    CUFFT_CHECK(cufftSetStream(cf_plan, stream.stream));

    for (int i = 0; i < cfg.n_warmup; ++i) {
        exec_flagfft(ff_plan, cfg, d_in_flagfft, d_out_flagfft);
        exec_cufft(cf_plan, cfg, d_in_cufft, d_out_cufft);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream.stream));

    std::vector<float> ff_times(cfg.n_iters);
    std::vector<float> cf_times(cfg.n_iters);
    CudaTimer timer;

    auto time_flagfft = [&]() -> float {
        timer.record_start(stream.stream);
        for (int launch = 0; launch < cfg.launches_per_sample; ++launch) {
            exec_flagfft(ff_plan, cfg, d_in_flagfft, d_out_flagfft);
        }
        timer.record_stop(stream.stream);
        return timer.elapsed_ms() / static_cast<float>(cfg.launches_per_sample);
    };

    auto time_cufft = [&]() -> float {
        timer.record_start(stream.stream);
        for (int launch = 0; launch < cfg.launches_per_sample; ++launch) {
            exec_cufft(cf_plan, cfg, d_in_cufft, d_out_cufft);
        }
        timer.record_stop(stream.stream);
        return timer.elapsed_ms() / static_cast<float>(cfg.launches_per_sample);
    };

    for (int i = 0; i < cfg.n_iters; ++i) {
        if ((i & 1) == 0) {
            cf_times[i] = time_cufft();
            ff_times[i] = time_flagfft();
        } else {
            ff_times[i] = time_flagfft();
            cf_times[i] = time_cufft();
        }
    }

    FLAGFFT_CHECK(flagfftDestroy(ff_plan));
    CUFFT_CHECK(cufftDestroy(cf_plan));
    CUDA_CHECK(cudaFree(d_in_flagfft));
    CUDA_CHECK(cudaFree(d_in_cufft));
    if (!cfg.inplace) {
        CUDA_CHECK(cudaFree(d_out_flagfft));
        CUDA_CHECK(cudaFree(d_out_cufft));
    }

    return {median(std::move(ff_times)), median(std::move(cf_times))};
}

std::vector<int> parse_lengths(const std::string &s) {
    std::vector<int> out;
    std::size_t i = 0;
    while (i < s.size()) {
        std::size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        if (j > i) {
            out.push_back(std::atoi(s.substr(i, j - i).c_str()));
        }
        i = j + 1;
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

std::filesystem::path executable_path(const char *argv0) {
    std::array<char, 4096> buffer{};
    ssize_t n = readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
    if (n > 0) {
        buffer[static_cast<std::size_t>(n)] = '\0';
        return std::filesystem::path(buffer.data());
    }
    std::filesystem::path from_argv = argv0 == nullptr ? std::filesystem::path{} : argv0;
    if (from_argv.empty()) {
        return std::filesystem::current_path() / "bench_vs_cufft";
    }
    std::error_code ec;
    return std::filesystem::absolute(from_argv, ec);
}

std::string default_tune_db(const char *argv0) {
    std::filesystem::path cache_dir = executable_path(argv0).parent_path() / ".flagfft";
    return (cache_dir / "tuned_plans.sqlite").string();
}

std::string api_for_direction(int direction) {
    return direction == FLAGFFT_INVERSE ? "ifft" : "fft";
}

void run_tune(const BenchConfig &cfg) {
    std::ostringstream command;
    command << shell_quote(cfg.tune_command)
            << " --api " << api_for_direction(cfg.direction)
            << " --batch " << cfg.batch
            << " --warmup " << cfg.n_warmup
            << " --iters " << cfg.n_iters
            << " --db " << shell_quote(cfg.tune_db)
            << " --static-limit " << cfg.tune_static_limit
            << " --finalists " << cfg.tune_finalists
            << " --lengths";
    for (int n : cfg.lengths) {
        command << " " << n;
    }
    if (cfg.retune) {
        command << " --retune";
    }

    std::printf("Running tune command: %s\n", command.str().c_str());
    std::fflush(stdout);
    int status = std::system(command.str().c_str());
    if (status != 0) {
        throw std::runtime_error("tune command failed with status " + std::to_string(status));
    }
}

void print_usage(const char *argv0) {
    std::printf(
        "Usage: %s [--lengths N1,N2,...] [--batch B] [--warmup W] [--iters K]\n"
        "          [--launches-per-sample K]\n"
        "          [--type c2c|z2z|r2c|d2z|c2r|z2d] [--inplace]\n"
        "          [--direction forward|inverse] [--tune|--retune]\n"
        "          [--tune-db PATH] [--tune-command CMD]\n"
        "          [--tune-static-limit N] [--tune-finalists N]\n"
        "          [--print-path]\n",
        argv0);
}

bool parse_args(int argc, char **argv, BenchConfig &cfg) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need_value = [&](const char *name) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s requires a value\n", name);
                return nullptr;
            }
            return argv[++i];
        };
        if (arg == "--lengths") {
            const char *v = need_value("--lengths");
            if (!v) return false;
            auto parsed = parse_lengths(v);
            if (parsed.empty()) {
                std::fprintf(stderr, "--lengths produced no values\n");
                return false;
            }
            cfg.lengths = std::move(parsed);
        } else if (arg == "--type") {
            const char *v = need_value("--type");
            if (!v) return false;
            cfg.type = v;
            if (cfg.type != "c2c" && cfg.type != "z2z" && cfg.type != "r2c" &&
                cfg.type != "d2z" && cfg.type != "c2r" && cfg.type != "z2d") {
                std::fprintf(stderr, "unknown type: %s\n", v);
                return false;
            }
        } else if (arg == "--batch") {
            const char *v = need_value("--batch");
            if (!v) return false;
            cfg.batch = std::atoi(v);
        } else if (arg == "--warmup") {
            const char *v = need_value("--warmup");
            if (!v) return false;
            cfg.n_warmup = std::atoi(v);
        } else if (arg == "--iters") {
            const char *v = need_value("--iters");
            if (!v) return false;
            cfg.n_iters = std::atoi(v);
        } else if (arg == "--launches-per-sample") {
            const char *v = need_value("--launches-per-sample");
            if (!v) return false;
            cfg.launches_per_sample = std::atoi(v);
        } else if (arg == "--direction") {
            const char *v = need_value("--direction");
            if (!v) return false;
            std::string d = v;
            if (d == "forward" || d == "fwd") {
                cfg.direction = FLAGFFT_FORWARD;
            } else if (d == "inverse" || d == "inv") {
                cfg.direction = FLAGFFT_INVERSE;
            } else {
                std::fprintf(stderr, "unknown direction: %s\n", v);
                return false;
            }
        } else if (arg == "--tune") {
            cfg.tune = true;
        } else if (arg == "--retune") {
            cfg.tune = true;
            cfg.retune = true;
        } else if (arg == "--tune-db") {
            const char *v = need_value("--tune-db");
            if (!v) return false;
            cfg.tune_db = v;
        } else if (arg == "--tune-command") {
            const char *v = need_value("--tune-command");
            if (!v) return false;
            cfg.tune_command = v;
        } else if (arg == "--tune-static-limit") {
            const char *v = need_value("--tune-static-limit");
            if (!v) return false;
            cfg.tune_static_limit = std::atoi(v);
        } else if (arg == "--tune-finalists") {
            const char *v = need_value("--tune-finalists");
            if (!v) return false;
            cfg.tune_finalists = std::atoi(v);
        } else if (arg == "--print-path") {
            cfg.print_path = true;
        } else if (arg == "--inplace") {
            cfg.inplace = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return false;
        }
    }
    if (cfg.batch <= 0 || cfg.n_iters <= 0 || cfg.n_warmup < 0 ||
        cfg.launches_per_sample <= 0) {
        std::fprintf(stderr,
                     "invalid config: batch=%d warmup=%d iters=%d launches_per_sample=%d\n",
                     cfg.batch,
                     cfg.n_warmup,
                     cfg.n_iters,
                     cfg.launches_per_sample);
        return false;
    }
    if (cfg.tune_static_limit <= 0 || cfg.tune_finalists <= 0) {
        std::fprintf(stderr, "invalid tune config: static_limit=%d finalists=%d\n",
                     cfg.tune_static_limit, cfg.tune_finalists);
        return false;
    }
    if (!is_complex_type(cfg.type) && cfg.direction != FLAGFFT_FORWARD) {
        std::fprintf(stderr, "--direction is only meaningful for --type c2c|z2z\n");
        return false;
    }
    if (cfg.tune && cfg.type != "c2c") {
        std::fprintf(stderr, "--tune/--retune currently supports only --type c2c\n");
        return false;
    }
    return true;
}

const char *flagfft_plan_mode(const BenchConfig &cfg) {
    if (cfg.retune) {
        return "per-shape retuned SQLite winner";
    }
    if (cfg.tune) {
        return "per-shape tuned SQLite winner when absent";
    }
    std::error_code ec;
    if (std::filesystem::is_regular_file(cfg.tune_db, ec)) {
        return "existing SQLite winner if present";
    }
    return "auto planner";
}

std::string flagfft_kernel_backend() {
    return "JIT";
}

}  // namespace

int main(int argc, char **argv) {
    BenchConfig cfg;
    if (!parse_args(argc, argv, cfg)) {
        return 1;
    }
    if (cfg.tune_db.empty()) {
        cfg.tune_db = default_tune_db(argv[0]);
    }
    setenv("FLAGFFT_TUNE_DB", cfg.tune_db.c_str(), 1);

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
        std::fprintf(stderr, "no CUDA device available\n");
        return 2;
    }

    if (cfg.tune) {
        try {
            run_tune(cfg);
        } catch (const std::exception &exc) {
            std::fprintf(stderr, "%s\n", exc.what());
            return 3;
        }
    }

    std::printf("FlagFFT vs cuFFT benchmark (type=%s, inplace=%s, warmup=%d, samples=%d, launches_per_sample=%d, direction=%s, tune_db=%s)\n",
                cfg.type.c_str(),
                cfg.inplace ? "yes" : "no",
                cfg.n_warmup,
                cfg.n_iters,
                cfg.launches_per_sample,
                cfg.direction == FLAGFFT_FORWARD ? "forward" : "inverse",
                cfg.tune_db.c_str());
    std::printf("Plan mode: FlagFFT=%s; backend=%s; cuFFT=default cufftPlan1d contiguous batch\n",
                flagfft_plan_mode(cfg),
                flagfft_kernel_backend().c_str());
    std::printf("%-10s %-7s %-13s %-12s %-9s\n",
                "n", "batch", "flagfft_ms", "cufft_ms", "speedup");
    std::printf("------------------------------------------------------\n");

    for (int n : cfg.lengths) {
        BenchResult r = bench_one(n, cfg);
        float speedup = (r.flagfft_ms > 0.f) ? (r.cufft_ms / r.flagfft_ms) : 0.f;
        std::printf("%-10d %-7d %-13.5f %-12.5f %-8.2fx\n",
                    n, cfg.batch, r.flagfft_ms, r.cufft_ms, speedup);
        std::fflush(stdout);
    }
    return 0;
}
