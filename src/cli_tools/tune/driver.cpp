#include "driver.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>

#include "bench.hpp"
#include "sqlite.hpp"
#include "flagfft/core.hpp"
#include "flagfft/tune_json.hpp"
#include "exec/c_api_internal.hpp"

namespace flagfft::cli {
namespace {

json tune_one(const CaseSpec &spec, const std::string &db_path, int warmup, int iters,
              int static_limit, int finalists, bool retune, int device_index,
              const std::string &device_arch) {
    using namespace flagfft::tune;
    const int64_t n = spec.shape[0];
    const std::string direction = direction_name(spec.direction);
    const std::string bucket = batch_bucket(spec.batch);
    FFTRequest request;
    request.fft_length = n;
    request.input_shape = {spec.batch, n};
    request.input_strides = {n, 1};
    request.requested_n = n;
    request.raw_dim = 1;
    request.normalized_dim = 1;
    request.norm = "backward";
    request.input_dtype = "complex64";
    request.output_dtype = "complex64";
    request.device_type = "cuda";
    request.device_index = device_index;
    request.device_arch = device_arch;
    request.input_layout = "contiguous";
    request.direction = direction;
    request.batch = spec.batch;

    PlanBuilder builder;
    builder.build(n, request);
    std::vector<PlanCandidate> supported;
    for (auto &candidate : builder.build_tune_candidates(n, 0)) {
        if (raw_supported_node(candidate.node)) supported.push_back(candidate);
    }
    if (supported.empty()) {
        throw CliException("no supported tune candidates for length " + std::to_string(n),
                           kExitRuntimeError);
    }
    if (static_cast<int>(supported.size()) > static_limit) supported.resize(static_limit);
    if (retune) {
        std::string prior_json;
        if (lookup_tune_winner(db_path, n, bucket, spec.batch, direction, device_arch, prior_json)) {
            try {
                PlanCandidate prior;
                prior.node = plan_node_from_json(builder, json::parse(prior_json).at("root"));
                if (raw_supported_node(prior.node)) supported.insert(supported.begin(), prior);
            } catch (const std::exception &) {
            }
        }
    }
    struct Score {
        PlanCandidate candidate;
        BenchTiming timing;
        BenchError error;
        std::string plan_json;
        bool verified = false;
    };
    std::vector<Score> scores;
    for (auto &candidate : supported) {
        const std::string plan_json = wrap_plan_json(candidate.node, request, "flagfft_cli_tune").dump();
        try {
            scores.push_back({candidate, bench_candidate(n, spec.batch, direction, warmup, iters,
                                                         plan_json, device_index, device_arch),
                              {}, plan_json, false});
        } catch (const std::exception &error) {
            TuneMeasurement failed;
            failed.fft_length = n;
            failed.batch = spec.batch;
            failed.direction = direction;
            failed.device_arch = device_arch;
            failed.plan_key = PlanKey::from_node(candidate.node).repr();
            failed.plan_json = plan_json;
            failed.status = "failed";
            failed.failure_reason = error.what();
            insert_measurement(db_path, failed);
        }
    }
    if (scores.empty()) {
        throw CliException("all tune candidates failed for length " + std::to_string(n),
                           kExitRuntimeError);
    }
    std::sort(scores.begin(), scores.end(), [](const Score &a, const Score &b) {
        return a.timing.median_ms < b.timing.median_ms;
    });
    for (int i = 0; i < std::min(finalists, static_cast<int>(scores.size())); ++i) {
        scores[i].error = verify_against_cufft(n, spec.batch, direction, scores[i].plan_json,
                                               device_index, device_arch);
        scores[i].verified = true;
    }
    mark_superseded(db_path, n, bucket, direction, device_arch);
    const double tolerance = 4.0e-3 * (spec.direction == FLAGFFT_INVERSE ? n : 1);
    const bool winner_valid = scores.front().verified && scores.front().error.max_abs <= tolerance;
    for (std::size_t i = 0; i < scores.size(); ++i) {
        TuneMeasurement measurement;
        measurement.fft_length = n;
        measurement.batch = spec.batch;
        measurement.direction = direction;
        measurement.device_arch = device_arch;
        measurement.plan_json = scores[i].plan_json;
        measurement.plan_key = PlanKey::from_node(scores[i].candidate.node).repr();
        measurement.compile_ms = scores[i].timing.compile_ms;
        measurement.first_call_ms = scores[i].timing.first_call_ms;
        measurement.median_ms = scores[i].timing.median_ms;
        measurement.p90_ms = scores[i].timing.p90_ms;
        measurement.max_abs_err = scores[i].error.max_abs;
        measurement.rms_err = scores[i].error.rms;
        measurement.status = i == 0 && winner_valid ? "valid" : "ok";
        if (i == 0 && !winner_valid) {
            measurement.status = "failed";
            measurement.failure_reason = "winning candidate failed correctness verification";
        }
        insert_measurement(db_path, measurement);
    }
    if (!winner_valid) {
        throw AssertionFailure("winning tune candidate failed correctness verification");
    }
    return {
        {"case", case_json(spec)},
        {"candidates", scores.size()},
        {"winner", {
            {"median_ms", scores.front().timing.median_ms},
            {"p90_ms", scores.front().timing.p90_ms},
            {"max_abs_error", scores.front().error.max_abs},
            {"rms_error", scores.front().error.rms},
        }},
    };
}

}  // namespace

json run_tune_cases(const std::vector<CaseSpec> &cases, const std::string &db_path,
                    int warmup, int iters, int static_limit, int finalists, bool retune) {
    int device_index = 0;
    cudaDeviceProp props{};
    check_cuda(cudaGetDeviceProperties(&props, device_index), "cudaGetDeviceProperties");
    const std::string device_arch = "sm_" + std::to_string(props.major) + std::to_string(props.minor);
    std::filesystem::path db_parent = std::filesystem::path(db_path).parent_path();
    if (!db_parent.empty()) std::filesystem::create_directories(db_parent);
    flagfft::tune::init_tune_db(db_path);
    json outputs = json::array();
    for (const auto &spec : cases) {
        outputs.push_back(tune_one(spec, db_path, warmup, iters, static_limit, finalists,
                                   retune, device_index, device_arch));
    }
    return outputs;
}

}  // namespace flagfft::cli
