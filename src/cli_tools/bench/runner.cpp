#include "cli_tools/bench/runner.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "adaptor/adaptor.h"
#include "adaptor/test_adaptor.h"
#include "cli_tools/common/cli_utils.hpp"
#include "cli_tools/common/plan_handles.hpp"
#include "cli_tools/common/runtime_raii.hpp"
#include "flagfft.h"

namespace flagfft::cli::bench {
namespace {

  double percentile(std::vector<float> times, double fraction) {
    std::sort(times.begin(), times.end());
    const std::size_t index =
        std::min(times.size() - 1, static_cast<std::size_t>(fraction * static_cast<double>(times.size())));
    return times[index];
  }

  struct BufferLayout {
    int n;
    int half;
    int padded;
    std::size_t scalar_bytes;
    std::size_t allocation_bytes;
  };

  BufferLayout layout_for(const CaseSpec& spec) {
    const int n = spec.shape[0];
    const int half = n / 2 + 1;
    const int padded = 2 * half;
    const std::size_t scalar = is_double_api(spec.api) ? sizeof(double) : sizeof(float);
    const bool real_forward = is_real_forward_api(spec.api);
    const bool real_inverse = is_real_inverse_api(spec.api);
    const bool complex = is_complex_api(spec.api);

    const int inplace_scalars = complex ? 2 * n : padded;

    const int in_scalars = real_forward ? n : (real_inverse ? 2 * half : 2 * n);
    const int out_scalars = real_forward ? 2 * half : (real_inverse ? n : 2 * n);

    const std::size_t eff_in = spec.placement == Placement::InPlace ? inplace_scalars : in_scalars;
    const std::size_t eff_out = spec.placement == Placement::InPlace ? inplace_scalars : out_scalars;

    const std::size_t in_bytes = static_cast<std::size_t>(spec.batch) * eff_in * scalar;
    const std::size_t out_bytes = static_cast<std::size_t>(spec.batch) * eff_out * scalar;

    return {n, half, padded, scalar, std::max(in_bytes, out_bytes)};
  }

  void seed_input(DeviceMemory& device, const BufferLayout& layout, const CaseSpec& spec) {
    const std::size_t count = layout.allocation_bytes / layout.scalar_bytes;
    const int n = layout.n;

    if (layout.scalar_bytes == sizeof(float)) {
      std::vector<float> host(count);
      for (std::size_t i = 0; i < count; ++i) {
        host[i] = std::sin(static_cast<float>(i + 1) * 0.173f);
      }
      if (is_real_inverse_api(spec.api)) {
        for (int b = 0; b < spec.batch; ++b) {
          host[static_cast<std::size_t>(b * layout.padded + 1)] = 0.0f;
          if (n % 2 == 0) host[static_cast<std::size_t>(b * layout.padded + n + 1)] = 0.0f;
        }
      }
      device.copy_from_host(host.data(), layout.allocation_bytes);
    } else {
      std::vector<double> host(count);
      for (std::size_t i = 0; i < count; ++i) {
        host[i] = std::sin(static_cast<double>(i + 1) * 0.173);
      }
      if (is_real_inverse_api(spec.api)) {
        for (int b = 0; b < spec.batch; ++b) {
          host[static_cast<std::size_t>(b * layout.padded + 1)] = 0.0;
          if (n % 2 == 0) host[static_cast<std::size_t>(b * layout.padded + n + 1)] = 0.0;
        }
      }
      device.copy_from_host(host.data(), layout.allocation_bytes);
    }
  }

  FlagfftPlanHandle make_flagfft_plan(const CaseSpec& spec, const BufferLayout& layout) {
    flagfftHandle raw = nullptr;
    flagfftResult result = FLAGFFT_NOT_SUPPORTED;
    if (spec.rank == 1) {
      result = flagfftPlan1d(&raw, layout.n, flagfft_type(spec.api), spec.batch);
    } else if (spec.rank == 2) {
      result = flagfftPlan2d(&raw, spec.shape[0], spec.shape[1], flagfft_type(spec.api));
    } else if (spec.rank == 3) {
      result = flagfftPlan3d(&raw, spec.shape[0], spec.shape[1], spec.shape[2], flagfft_type(spec.api));
    }
    check_flagfft(result, "create FlagFFT plan");
    return FlagfftPlanHandle(raw);
  }

  test_adaptor::RefPlanHandle make_ref_plan(const CaseSpec& spec, const BufferLayout& layout) {
    test_adaptor::RefPlanHandle plan;
    if (spec.rank == 1) {
      test_adaptor::ref_plan_1d(plan, layout.n, flagfft_type(spec.api), spec.batch);
    } else if (spec.rank == 2) {
      test_adaptor::ref_plan_2d(plan, spec.shape[0], spec.shape[1], flagfft_type(spec.api));
    } else if (spec.rank == 3) {
      test_adaptor::ref_plan_3d(plan, spec.shape[0], spec.shape[1], spec.shape[2], flagfft_type(spec.api));
    }
    return plan;
  }

  void exec_flagfft(flagfftHandle plan, const CaseSpec& spec, void* input, void* output) {
    switch (spec.api) {
      case FftApi::C2C:
        check_flagfft(flagfftExecC2C(plan,
                                     static_cast<flagfftComplex*>(input),
                                     static_cast<flagfftComplex*>(output),
                                     spec.direction),
                      "flagfftExecC2C");
        break;
      case FftApi::Z2Z:
        check_flagfft(flagfftExecZ2Z(plan,
                                     static_cast<flagfftDoubleComplex*>(input),
                                     static_cast<flagfftDoubleComplex*>(output),
                                     spec.direction),
                      "flagfftExecZ2Z");
        break;
      case FftApi::R2C:
        check_flagfft(
            flagfftExecR2C(plan, static_cast<flagfftReal*>(input), static_cast<flagfftComplex*>(output)),
            "flagfftExecR2C");
        break;
      case FftApi::D2Z:
        check_flagfft(flagfftExecD2Z(plan,
                                     static_cast<flagfftDoubleReal*>(input),
                                     static_cast<flagfftDoubleComplex*>(output)),
                      "flagfftExecD2Z");
        break;
      case FftApi::C2R:
        check_flagfft(
            flagfftExecC2R(plan, static_cast<flagfftComplex*>(input), static_cast<flagfftReal*>(output)),
            "flagfftExecC2R");
        break;
      case FftApi::Z2D:
        check_flagfft(flagfftExecZ2D(plan,
                                     static_cast<flagfftDoubleComplex*>(input),
                                     static_cast<flagfftDoubleReal*>(output)),
                      "flagfftExecZ2D");
        break;
    }
  }

  void exec_ref(test_adaptor::RefPlanHandle& plan, const CaseSpec& spec, void* input, void* output) {
    switch (spec.api) {
      case FftApi::C2C:
        test_adaptor::ref_exec_c2c(plan,
                                   static_cast<flagfftComplex*>(input),
                                   static_cast<flagfftComplex*>(output),
                                   spec.direction);
        break;
      case FftApi::Z2Z:
        test_adaptor::ref_exec_z2z(plan,
                                   static_cast<flagfftDoubleComplex*>(input),
                                   static_cast<flagfftDoubleComplex*>(output),
                                   spec.direction);
        break;
      case FftApi::R2C:
        test_adaptor::ref_exec_r2c(plan,
                                   static_cast<flagfftReal*>(input),
                                   static_cast<flagfftComplex*>(output));
        break;
      case FftApi::D2Z:
        test_adaptor::ref_exec_d2z(plan,
                                   static_cast<flagfftDoubleReal*>(input),
                                   static_cast<flagfftDoubleComplex*>(output));
        break;
      case FftApi::C2R:
        test_adaptor::ref_exec_c2r(plan,
                                   static_cast<flagfftComplex*>(input),
                                   static_cast<flagfftReal*>(output));
        break;
      case FftApi::Z2D:
        test_adaptor::ref_exec_z2d(plan,
                                   static_cast<flagfftDoubleComplex*>(input),
                                   static_cast<flagfftDoubleReal*>(output));
        break;
    }
  }

}  // namespace

BenchResult run_benchmark(const CaseSpec& spec, int warmup, int iters, bool include_path) {
  const BufferLayout layout = layout_for(spec);

  DeviceMemory ff_in(layout.allocation_bytes);
  DeviceMemory ref_in(layout.allocation_bytes);

  DeviceMemory ff_out;
  DeviceMemory ref_out;
  if (spec.placement == Placement::OutOfPlace) {
    ff_out.allocate(layout.allocation_bytes);
    ref_out.allocate(layout.allocation_bytes);
  }

  seed_input(ff_in, layout, spec);
  seed_input(ref_in, layout, spec);

  FlagfftPlanHandle ff_plan = make_flagfft_plan(spec, layout);
  test_adaptor::RefPlanHandle ref_plan = make_ref_plan(spec, layout);

  auto ff_output = [&]() -> void* {
    return spec.placement == Placement::InPlace ? ff_in.get() : ff_out.get();
  };
  auto ref_output = [&]() -> void* {
    return spec.placement == Placement::InPlace ? ref_in.get() : ref_out.get();
  };

  Stream stream;
  Timer timer;

  // Warmup
  for (int i = 0; i < warmup; ++i) {
    exec_flagfft(ff_plan.get(), spec, ff_in.get(), ff_output());
    exec_ref(ref_plan, spec, ref_in.get(), ref_output());
  }
  adaptor::synchronize();

  // Benchmark
  std::vector<float> ff_times;
  std::vector<float> ref_times;
  ff_times.reserve(iters);
  ref_times.reserve(iters);

  for (int i = 0; i < iters; ++i) {
    if ((i & 1) == 0) {
      timer.start(stream.get());
      exec_ref(ref_plan, spec, ref_in.get(), ref_output());
      timer.stop(stream.get());
      ref_times.push_back(timer.elapsed_ms());

      timer.start(stream.get());
      exec_flagfft(ff_plan.get(), spec, ff_in.get(), ff_output());
      timer.stop(stream.get());
      ff_times.push_back(timer.elapsed_ms());
    } else {
      timer.start(stream.get());
      exec_flagfft(ff_plan.get(), spec, ff_in.get(), ff_output());
      timer.stop(stream.get());
      ff_times.push_back(timer.elapsed_ms());

      timer.start(stream.get());
      exec_ref(ref_plan, spec, ref_in.get(), ref_output());
      timer.stop(stream.get());
      ref_times.push_back(timer.elapsed_ms());
    }
  }
  stream.sync();

  TimingStats ff_stats {percentile(ff_times, 0.5), percentile(ff_times, 0.9), ff_times};
  TimingStats ref_stats {percentile(ref_times, 0.5), percentile(ref_times, 0.9), ref_times};
  double speedup = ff_stats.median_ms > 0.0 ? ref_stats.median_ms / ff_stats.median_ms : 0.0;

  BenchResult result {ff_stats, ref_stats, speedup, ""};
  if (include_path) {
    const char* desc = flagfftGetPlanDescription(ff_plan.get());
    result.plan_description = desc ? desc : "";
  }
  return result;
}

}  // namespace flagfft::cli::bench
