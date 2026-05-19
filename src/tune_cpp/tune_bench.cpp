#include "tune_bench.hpp"

#include <cuda_runtime_api.h>
#include <cufft.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <chrono>

#include "flagfft/flagfft.h"
#include "flagfft/tune_json.hpp"
#include "c_api_internal.hpp"

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            throw std::runtime_error(std::string("CUDA error: ") +             \
                                     cudaGetErrorString(_e));                  \
        }                                                                      \
    } while (0)

#define FLAGFFT_CHECK(call)                                                    \
    do {                                                                       \
        flagfftResult _r = (call);                                             \
        if (_r != FLAGFFT_SUCCESS) {                                           \
            throw std::runtime_error("FlagFFT error: " +                       \
                                     std::to_string(static_cast<int>(_r)));    \
        }                                                                      \
    } while (0)

#define CUFFT_CHECK(call)                                                      \
    do {                                                                       \
        cufftResult _r = (call);                                               \
        if (_r != CUFFT_SUCCESS) {                                             \
            throw std::runtime_error("cuFFT error: " +                         \
                                     std::to_string(static_cast<int>(_r)));    \
        }                                                                      \
    } while (0)

namespace flagfft {
namespace tune {

std::vector<float> generate_random_input(int64_t n, int64_t batch) {
    std::mt19937 rng(1234 + static_cast<unsigned>(n * 17 + batch));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::size_t samples = static_cast<std::size_t>(n * batch * 2);
    std::vector<float> data(samples);
    for (auto &v : data) {
        v = dist(rng);
    }
    return data;
}

static double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static double p90(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    std::size_t idx = static_cast<std::size_t>(0.9 * static_cast<double>(v.size()));
    if (idx >= v.size()) idx = v.size() - 1;
    return v[idx];
}

static flagfftHandle build_plan_from_json_str(const std::string &plan_json_str,
                                                int64_t n, int64_t batch,
                                                int device_index,
                                                const std::string &device_arch) {
    if (device_index < 0) {
        throw std::runtime_error("no CUDA device available");
    }

    PlanBuilder builder;
    auto json = nlohmann::json::parse(plan_json_str);
    PlanNodePtr root = plan_node_from_json(builder, json.at("root"));

    if (!raw_supported_node(root)) {
        throw std::runtime_error("plan node not supported by raw C API");
    }

    FFTRequest forward_req;
    forward_req.fft_length = n;
    forward_req.input_shape = {batch, n};
    forward_req.input_strides = {n, 1};
    forward_req.requested_n = n;
    forward_req.raw_dim = 1;
    forward_req.normalized_dim = 1;
    forward_req.norm = "backward";
    forward_req.input_dtype = "complex64";
    forward_req.output_dtype = "complex64";
    forward_req.device_type = "cuda";
    forward_req.device_index = device_index;
    forward_req.device_arch = device_arch;
    forward_req.input_layout = "contiguous";
    forward_req.direction = "forward";
    forward_req.batch = batch;

    TritonCompiler compiler;
    auto compiled = compiler.compile_raw_node(root, forward_req, batch);

    flagfftHandle handle = new flagfftPlan_t();
    auto plan = new FlagFFTPlan();
    plan->desc.rank = 1;
    plan->desc.n = {n};
    plan->desc.batch = static_cast<int>(batch);
    plan->desc.type = FLAGFFT_C2C;
    plan->desc.device_index = device_index;
    plan->desc.device_arch = device_arch;
    plan->executable.root = std::move(root);
    plan->executable.forward = std::move(compiled);
    plan->executable.forward_request = forward_req;
    plan->state.initialized = true;
    handle->impl = plan;
    return handle;
}

BenchTiming bench_candidate(int64_t n, int64_t batch, const std::string &direction,
                            int n_warmup, int n_iters,
                            const std::string &plan_json_str,
                            int device_index, const std::string &device_arch) {
    BenchTiming result{};

    auto host = generate_random_input(n, batch);
    std::size_t bytes = host.size() * sizeof(float);
    float *d_in = nullptr;
    float *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_in), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_out), bytes));
    CUDA_CHECK(cudaMemcpy(d_in, host.data(), bytes, cudaMemcpyHostToDevice));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaEvent_t ev_start = nullptr, ev_stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    auto t0 = std::chrono::steady_clock::now();
    flagfftHandle plan = build_plan_from_json_str(plan_json_str, n, batch,
                                                    device_index, device_arch);
    auto t1 = std::chrono::steady_clock::now();
    result.compile_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    FLAGFFT_CHECK(flagfftSetStream(plan, stream));

    int ff_dir = direction == "forward" ? FLAGFFT_FORWARD : FLAGFFT_INVERSE;

    // First call timing
    CUDA_CHECK(cudaEventRecord(ev_start, stream));
    FLAGFFT_CHECK(flagfftExecC2C(plan,
                                  reinterpret_cast<flagfftComplex *>(d_in),
                                  reinterpret_cast<flagfftComplex *>(d_out),
                                  ff_dir));
    CUDA_CHECK(cudaEventRecord(ev_stop, stream));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float first_ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&first_ms, ev_start, ev_stop));
    result.first_call_ms = static_cast<double>(first_ms);

    // Warmup
    for (int i = 1; i < n_warmup; ++i) {
        FLAGFFT_CHECK(flagfftExecC2C(plan,
                                      reinterpret_cast<flagfftComplex *>(d_in),
                                      reinterpret_cast<flagfftComplex *>(d_out),
                                      ff_dir));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Measurement
    std::vector<double> times(n_iters);
    for (int i = 0; i < n_iters; ++i) {
        CUDA_CHECK(cudaEventRecord(ev_start, stream));
        FLAGFFT_CHECK(flagfftExecC2C(plan,
                                      reinterpret_cast<flagfftComplex *>(d_in),
                                      reinterpret_cast<flagfftComplex *>(d_out),
                                      ff_dir));
        CUDA_CHECK(cudaEventRecord(ev_stop, stream));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        times[i] = static_cast<double>(ms);
    }

    result.median_ms = median(times);
    result.p90_ms = p90(times);

    FLAGFFT_CHECK(flagfftDestroy(plan));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return result;
}

BenchError verify_against_cufft(int64_t n, int64_t batch, const std::string &direction,
                                const std::string &plan_json_str,
                                int device_index, const std::string &device_arch) {
    auto host = generate_random_input(n, batch);
    std::size_t bytes = host.size() * sizeof(float);
    float *d_in = nullptr;
    float *d_out_ff = nullptr;
    float *d_out_cf = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_in), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_out_ff), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_out_cf), bytes));
    CUDA_CHECK(cudaMemcpy(d_in, host.data(), bytes, cudaMemcpyHostToDevice));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    flagfftHandle ff_plan = build_plan_from_json_str(plan_json_str, n, batch,
                                                       device_index, device_arch);
    FLAGFFT_CHECK(flagfftSetStream(ff_plan, stream));

    cufftHandle cf_plan = 0;
    CUFFT_CHECK(cufftPlan1d(&cf_plan, static_cast<int>(n), CUFFT_C2C, static_cast<int>(batch)));
    CUFFT_CHECK(cufftSetStream(cf_plan, stream));

    int ff_dir = direction == "forward" ? FLAGFFT_FORWARD : FLAGFFT_INVERSE;
    int cf_dir = direction == "forward" ? CUFFT_FORWARD : CUFFT_INVERSE;

    FLAGFFT_CHECK(flagfftExecC2C(ff_plan,
                                  reinterpret_cast<flagfftComplex *>(d_in),
                                  reinterpret_cast<flagfftComplex *>(d_out_ff),
                                  ff_dir));
    CUFFT_CHECK(cufftExecC2C(cf_plan,
                             reinterpret_cast<cufftComplex *>(d_in),
                             reinterpret_cast<cufftComplex *>(d_out_cf),
                             cf_dir));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<float> out_ff(host.size());
    std::vector<float> out_cf(host.size());
    CUDA_CHECK(cudaMemcpy(out_ff.data(), d_out_ff, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out_cf.data(), d_out_cf, bytes, cudaMemcpyDeviceToHost));

    BenchError err{};
    double sum_sq = 0.0;
    for (std::size_t i = 0; i < out_ff.size(); ++i) {
        double diff = static_cast<double>(out_ff[i]) - static_cast<double>(out_cf[i]);
        double abs_diff = std::abs(diff);
        if (abs_diff > err.max_abs) {
            err.max_abs = abs_diff;
        }
        sum_sq += diff * diff;
    }
    err.rms = std::sqrt(sum_sq / static_cast<double>(out_ff.size()));

    FLAGFFT_CHECK(flagfftDestroy(ff_plan));
    CUFFT_CHECK(cufftDestroy(cf_plan));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out_ff));
    CUDA_CHECK(cudaFree(d_out_cf));
    return err;
}

}  // namespace tune
}  // namespace flagfft
