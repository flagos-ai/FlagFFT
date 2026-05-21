#include "tune_bench.hpp"
#include "tune_sqlite.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include "flagfft/core.hpp"
#include "flagfft/tune_json.hpp"
#include "c_api_internal.hpp"

namespace {

struct TuneArgs {
    std::vector<int64_t> lengths;
    int64_t batch = 1;
    int n_warmup = 10;
    int n_iters = 100;
    int static_limit = 32;
    int finalists = 3;
    std::string direction = "forward";
    std::string db_path;
    bool retune = false;
    std::string api = "fft";
    int device_index = 0;
    std::string device_arch = "sm_80";
};

bool parse_args(int argc, char **argv, TuneArgs &args) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need = [&](const char *name) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s requires a value\n", name);
                return nullptr;
            }
            return argv[++i];
        };
        if (arg == "--lengths") {
            // Remaining positional args are lengths
            ++i;
            while (i < argc && argv[i][0] != '-') {
                args.lengths.push_back(std::atoll(argv[i]));
                ++i;
            }
            --i;
            if (args.lengths.empty()) {
                std::fprintf(stderr, "--lengths requires at least one value\n");
                return false;
            }
        } else if (arg == "--batch") {
            const char *v = need("--batch"); if (!v) return false;
            args.batch = std::atoll(v);
        } else if (arg == "--warmup") {
            const char *v = need("--warmup"); if (!v) return false;
            args.n_warmup = std::atoi(v);
        } else if (arg == "--iters") {
            const char *v = need("--iters"); if (!v) return false;
            args.n_iters = std::atoi(v);
        } else if (arg == "--db") {
            const char *v = need("--db"); if (!v) return false;
            args.db_path = v;
        } else if (arg == "--retune") {
            args.retune = true;
        } else if (arg == "--static-limit") {
            const char *v = need("--static-limit"); if (!v) return false;
            args.static_limit = std::atoi(v);
        } else if (arg == "--finalists") {
            const char *v = need("--finalists"); if (!v) return false;
            args.finalists = std::atoi(v);
        } else if (arg == "--direction") {
            const char *v = need("--direction"); if (!v) return false;
            args.direction = v;
        } else if (arg == "--api") {
            const char *v = need("--api"); if (!v) return false;
            std::string api = v;
            if (api == "ifft") {
                args.direction = "inverse";
            } else {
                args.direction = "forward";
            }
        } else if (arg == "--device") {
            const char *v = need("--device"); if (!v) return false;
            args.device_index = std::atoi(v);
        } else if (arg == "-h" || arg == "--help") {
            std::printf(
                "Usage: flagfft_tune [--lengths N...] [--batch B] [--warmup W] [--iters K]\n"
                "                    [--db PATH] [--retune] [--static-limit N] [--finalists N]\n"
                "                    [--direction forward|inverse] [--api fft|ifft]\n");
            std::exit(0);
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            return false;
        }
    }
    if (args.lengths.empty()) {
        std::fprintf(stderr, "--lengths is required\n");
        return false;
    }
    if (args.db_path.empty()) {
        std::fprintf(stderr, "--db is required\n");
        return false;
    }
    return true;
}

bool detect_device(int &device_index, std::string &device_arch) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) {
        std::fprintf(stderr, "no CUDA device available\n");
        return false;
    }

    if (device_index < 0) {
        device_index = 0;
    }

    cudaDeviceProp props{};
    if (cudaGetDeviceProperties(&props, device_index) != cudaSuccess) {
        std::fprintf(stderr, "failed to get device properties for device %d\n", device_index);
        return false;
    }

    char arch[16];
    std::snprintf(arch, sizeof(arch), "sm_%d%d", props.major, props.minor);
    device_arch = arch;
    return true;
}

void tune_length(const TuneArgs &args, int64_t n) {
    using namespace flagfft;
    using namespace flagfft::tune;

    std::string bucket = batch_bucket(args.batch);

    FFTRequest request;
    request.fft_length = n;
    request.input_shape = {args.batch, n};
    request.input_strides = {n, 1};
    request.requested_n = n;
    request.raw_dim = 1;
    request.normalized_dim = 1;
    request.norm = "backward";
    request.input_dtype = "complex64";
    request.output_dtype = "complex64";
    request.device_type = "cuda";
    request.device_index = args.device_index;
    request.device_arch = args.device_arch;
    request.input_layout = "contiguous";
    request.direction = args.direction;
    request.batch = args.batch;

    PlanBuilder builder;
    builder.build(n, request);

    std::vector<PlanCandidate> candidates = builder.build_tune_candidates(n, 0);

    // Filter to raw-supported nodes only
    std::vector<PlanCandidate> supported;
    for (auto &c : candidates) {
        if (raw_supported_node(c.node)) {
            supported.push_back(c);
        }
    }
    if (supported.empty()) {
        std::fprintf(stderr, "  length %lld: no raw-supported candidates\n",
                     static_cast<long long>(n));
        return;
    }

    if (static_cast<int64_t>(supported.size()) > args.static_limit) {
        supported.resize(static_cast<std::size_t>(args.static_limit));
    }

    // If retune, check if there's a prior winner and add it as candidate
    if (args.retune) {
        std::string prior_json;
        if (tune::lookup_tune_winner(args.db_path, n, bucket, args.batch,
                                     args.direction, prior_json)) {
            try {
                auto json = nlohmann::json::parse(prior_json);
                PlanNodePtr prior = plan_node_from_json(builder, json.at("root"));
                if (raw_supported_node(prior)) {
                    PlanCandidate prior_cand;
                    prior_cand.node = std::move(prior);
                    prior_cand.cost = 0.0;
                    prior_cand.priority = 0;
                    supported.insert(supported.begin(), prior_cand);
                }
            } catch (const std::exception &) {
            }
        }
    }

    // Benchmark each candidate
    struct ScoredCandidate {
        PlanCandidate candidate;
        tune::BenchTiming timing;
        tune::BenchError error;
        std::string plan_json_str;
    };

    std::vector<ScoredCandidate> scored;
    for (auto &cand : supported) {
        std::string plan_json_str =
            wrap_plan_json(cand.node, request, "cpp_tune_candidate").dump();

        std::printf("    benchmarking candidate cost=%.1f priority=%d ...",
                    cand.cost, cand.priority);
        std::fflush(stdout);

        try {
            auto timing = tune::bench_candidate(n, args.batch, args.direction,
                                                args.n_warmup, args.n_iters,
                                                plan_json_str,
                                                args.device_index, args.device_arch);
            std::printf(" median=%.4fms\n", timing.median_ms);

            tune::BenchError err{};
            if (args.finalists > 0) {
                // Only verify correctness for the top candidates (we'll do after ranking)
            }

            scored.push_back({cand, timing, err, std::move(plan_json_str)});
        } catch (const std::exception &exc) {
            std::printf(" FAILED: %s\n", exc.what());
            tune::TuneMeasurement m;
            m.fft_length = n;
            m.batch = args.batch;
            m.direction = args.direction;
            m.plan_key = PlanKey::from_node(cand.node).repr();
            m.plan_json = plan_json_str;
            m.status = "failed";
            m.failure_reason = exc.what();
            tune::insert_measurement(args.db_path, m);
        }
    }

    if (scored.empty()) {
        std::fprintf(stderr, "  length %lld: all candidates failed\n",
                     static_cast<long long>(n));
        return;
    }

    // Sort by median time
    std::sort(scored.begin(), scored.end(),
              [](const ScoredCandidate &a, const ScoredCandidate &b) {
                  return a.timing.median_ms < b.timing.median_ms;
              });

    // Verify correctness for top finalists
    int verified = 0;
    for (auto &s : scored) {
        if (verified >= args.finalists) break;
        try {
            s.error = tune::verify_against_cufft(
                n, args.batch, args.direction, s.plan_json_str,
                args.device_index, args.device_arch);
            std::printf("    verified rank=%d max_abs_err=%.6f rms_err=%.6f\n",
                        verified, s.error.max_abs, s.error.rms);
            ++verified;
        } catch (const std::exception &exc) {
            std::fprintf(stderr, "    verification failed: %s\n", exc.what());
        }
    }

    // Mark previous winners superseded
    tune::mark_superseded(args.db_path, n, bucket, args.direction);

    // Insert results
    for (std::size_t i = 0; i < scored.size(); ++i) {
        auto &s = scored[i];
        tune::TuneMeasurement m;
        m.fft_length = n;
        m.batch = args.batch;
        m.direction = args.direction;
        m.plan_json = s.plan_json_str;
        m.plan_key = PlanKey::from_node(s.candidate.node).repr();
        int rank = static_cast<int>(i);
        if (rank == 0 && s.error.max_abs < 1e-3) {
            m.status = "valid";
        } else {
            m.status = "ok";
            if (rank == 0 && s.error.max_abs >= 1e-3) {
                m.status = "failed";
                m.failure_reason = "max_abs_err too large: " +
                                   std::to_string(s.error.max_abs);
            }
        }
        m.compile_ms = s.timing.compile_ms;
        m.first_call_ms = s.timing.first_call_ms;
        m.median_ms = s.timing.median_ms;
        m.p90_ms = s.timing.p90_ms;
        m.max_abs_err = s.error.max_abs;
        m.rms_err = s.error.rms;
        tune::insert_measurement(args.db_path, m);
    }

    std::printf("  length %lld: winner median=%.4fms (%zu candidates)\n",
                static_cast<long long>(n),
                scored.front().timing.median_ms,
                scored.size());
}

}  // namespace

int main(int argc, char **argv) {
    using namespace flagfft;
    using namespace flagfft::tune;
    TuneArgs args;
    if (!parse_args(argc, argv, args)) {
        return 1;
    }

    if (!detect_device(args.device_index, args.device_arch)) {
        return 2;
    }

    // Create the CWD-relative .flagfft cache dir for kernel cache
    std::filesystem::create_directories(
        std::filesystem::current_path() / ".flagfft");

    tune::init_tune_db(args.db_path);

    std::printf("Tuning lengths:");
    for (auto n : args.lengths) {
        std::printf(" %lld", static_cast<long long>(n));
    }
    std::printf(" batch=%lld direction=%s\n",
                static_cast<long long>(args.batch),
                args.direction.c_str());

    for (auto n : args.lengths) {
        std::printf("  length %lld:\n", static_cast<long long>(n));
        try {
            tune_length(args, n);
        } catch (const std::exception &exc) {
            std::fprintf(stderr, "  length %lld: unexpected error: %s\n",
                         static_cast<long long>(n), exc.what());
        }
    }

    std::printf("Done.\n");
    return 0;
}
