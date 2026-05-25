#include "bench.hpp"

#include <cufft.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>

#include "c_api_internal.hpp"
#include "cli_tools/common/cli_utils.hpp"
#include "cli_tools/common/plan_handles.hpp"
#include "cli_tools/common/runtime_raii.hpp"
#include "flagfft.h"
#include "flagfft/tune_json.hpp"

namespace flagfft {
namespace tune {

  using flagfft::cli::check_cufft;
  using flagfft::cli::check_flagfft;

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

  static flagfft::cli::FlagfftPlanHandle build_plan_from_json_str(const std::string &plan_json_str,
                                                                  int64_t n,
                                                                  int64_t batch,
                                                                  const std::string &direction,
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
    forward_req.device_type = adaptor::backend_name();
    forward_req.device_index = device_index;
    forward_req.device_arch = device_arch;
    forward_req.input_layout = "contiguous";
    forward_req.direction = direction;
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
    if (direction == "inverse") {
      plan->executable.inverse = std::move(compiled);
      plan->executable.inverse_request = forward_req;
    } else {
      plan->executable.forward = std::move(compiled);
      plan->executable.forward_request = forward_req;
    }
    plan->state.initialized = true;
    handle->impl = plan;
    return flagfft::cli::FlagfftPlanHandle(handle);
  }

  BenchTiming bench_candidate(int64_t n,
                              int64_t batch,
                              const std::string &direction,
                              int n_warmup,
                              int n_iters,
                              const std::string &plan_json_str,
                              int device_index,
                              const std::string &device_arch) {
    BenchTiming result {};

    auto host = generate_random_input(n, batch);
    std::size_t bytes = host.size() * sizeof(float);
    flagfft::cli::DeviceMemory d_in(bytes);
    flagfft::cli::DeviceMemory d_out(bytes);
    d_in.copy_from_host(host.data(), bytes);

    flagfft::cli::Stream stream;
    flagfft::cli::Timer timer;

    auto t0 = std::chrono::steady_clock::now();
    auto plan = build_plan_from_json_str(plan_json_str, n, batch, direction, device_index, device_arch);
    auto t1 = std::chrono::steady_clock::now();
    result.compile_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    check_flagfft(flagfftSetStream(plan.get(), stream.get()), "flagfftSetStream");

    int ff_dir = direction == "forward" ? FLAGFFT_FORWARD : FLAGFFT_INVERSE;

    timer.start(stream.get());
    check_flagfft(flagfftExecC2C(plan.get(),
                                 reinterpret_cast<flagfftComplex *>(d_in.get()),
                                 reinterpret_cast<flagfftComplex *>(d_out.get()),
                                 ff_dir),
                  "flagfftExecC2C (first)");
    timer.stop(stream.get());
    result.first_call_ms = static_cast<double>(timer.elapsed_ms());

    for (int i = 1; i < n_warmup; ++i) {
      check_flagfft(flagfftExecC2C(plan.get(),
                                   reinterpret_cast<flagfftComplex *>(d_in.get()),
                                   reinterpret_cast<flagfftComplex *>(d_out.get()),
                                   ff_dir),
                    "flagfftExecC2C (warmup)");
    }
    stream.sync();

    std::vector<double> times(n_iters);
    for (int i = 0; i < n_iters; ++i) {
      timer.start(stream.get());
      check_flagfft(flagfftExecC2C(plan.get(),
                                   reinterpret_cast<flagfftComplex *>(d_in.get()),
                                   reinterpret_cast<flagfftComplex *>(d_out.get()),
                                   ff_dir),
                    "flagfftExecC2C (bench)");
      timer.stop(stream.get());
      times[i] = static_cast<double>(timer.elapsed_ms());
    }

    result.median_ms = median(times);
    result.p90_ms = p90(times);
    return result;
  }

  BenchError verify_against_cufft(int64_t n,
                                  int64_t batch,
                                  const std::string &direction,
                                  const std::string &plan_json_str,
                                  int device_index,
                                  const std::string &device_arch) {
    auto host = generate_random_input(n, batch);
    std::size_t bytes = host.size() * sizeof(float);
    flagfft::cli::DeviceMemory d_in(bytes);
    flagfft::cli::DeviceMemory d_out_ff(bytes);
    flagfft::cli::DeviceMemory d_out_cf(bytes);
    d_in.copy_from_host(host.data(), bytes);

    flagfft::cli::Stream stream;

    auto ff_plan = build_plan_from_json_str(plan_json_str, n, batch, direction, device_index, device_arch);
    check_flagfft(flagfftSetStream(ff_plan.get(), stream.get()), "flagfftSetStream");

    flagfft::cli::CufftPlanHandle cf_plan;
    check_cufft(cufftPlan1d(cf_plan.put(), static_cast<int>(n), CUFFT_C2C, static_cast<int>(batch)),
                "cufftPlan1d");
    check_cufft(cufftSetStream(cf_plan.get(), reinterpret_cast<cudaStream_t>(stream.get())),
                "cufftSetStream");

    int ff_dir = direction == "forward" ? FLAGFFT_FORWARD : FLAGFFT_INVERSE;
    int cf_dir = direction == "forward" ? CUFFT_FORWARD : CUFFT_INVERSE;

    check_flagfft(flagfftExecC2C(ff_plan.get(),
                                 reinterpret_cast<flagfftComplex *>(d_in.get()),
                                 reinterpret_cast<flagfftComplex *>(d_out_ff.get()),
                                 ff_dir),
                  "flagfftExecC2C");
    check_cufft(cufftExecC2C(cf_plan.get(),
                             reinterpret_cast<cufftComplex *>(d_in.get()),
                             reinterpret_cast<cufftComplex *>(d_out_cf.get()),
                             cf_dir),
                "cufftExecC2C");
    stream.sync();

    std::vector<float> out_ff(host.size());
    std::vector<float> out_cf(host.size());
    d_out_ff.copy_to_host(out_ff.data(), bytes);
    d_out_cf.copy_to_host(out_cf.data(), bytes);

    BenchError err {};
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
    return err;
  }

}  // namespace tune
}  // namespace flagfft
