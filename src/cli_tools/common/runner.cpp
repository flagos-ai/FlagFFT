#include "runner.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <numeric>
#include <vector>

#include "plan_handles.hpp"
#include "runtime_raii.hpp"

namespace flagfft::cli {
namespace {

struct BufferLayout {
    int n;
    int half;
    int padded;
    std::size_t scalar_bytes;
    std::size_t in_bytes;
    std::size_t out_bytes;
    std::size_t allocation_bytes;
};

BufferLayout layout_for(const CaseSpec &spec) {
    const int n = spec.shape[0];
    const int half = n / 2 + 1;
    const int padded = 2 * half;
    const std::size_t scalar = is_double_api(spec.api) ? sizeof(double) : sizeof(float);
    const bool real_forward = is_real_forward_api(spec.api);
    const bool real_inverse = is_real_inverse_api(spec.api);
    const int in_scalars = real_forward ? n : (real_inverse ? 2 * half : 2 * n);
    const int out_scalars = real_forward ? 2 * half : (real_inverse ? n : 2 * n);
    const int inplace_scalars = is_complex_api(spec.api) ? 2 * n : padded;
    const std::size_t in_bytes = static_cast<std::size_t>(spec.batch) *
                                 (spec.placement == Placement::InPlace ? inplace_scalars : in_scalars) * scalar;
    const std::size_t out_bytes = static_cast<std::size_t>(spec.batch) *
                                  (spec.placement == Placement::InPlace ? inplace_scalars : out_scalars) * scalar;
    return {n, half, padded, scalar, in_bytes, out_bytes, std::max(in_bytes, out_bytes)};
}

void seed_input(void *device, const BufferLayout &layout, const CaseSpec &spec) {
    const std::size_t count = layout.in_bytes / layout.scalar_bytes;
    if (layout.scalar_bytes == sizeof(float)) {
        std::vector<float> host(count);
        for (std::size_t i = 0; i < count; ++i) {
            host[i] = std::sin(static_cast<float>(i + 1) * 0.173f);
        }
        if (is_real_inverse_api(spec.api)) {
            for (int batch = 0; batch < spec.batch; ++batch) {
                const int row = layout.padded;
                host[static_cast<std::size_t>(batch * row + 1)] = 0.0f;
                if (layout.n % 2 == 0) host[static_cast<std::size_t>(batch * row + layout.n + 1)] = 0.0f;
            }
        }
        check_cuda(cudaMemcpy(device, host.data(), layout.in_bytes, cudaMemcpyHostToDevice),
                   "copy seeded float input");
    } else {
        std::vector<double> host(count);
        for (std::size_t i = 0; i < count; ++i) {
            host[i] = std::sin(static_cast<double>(i + 1) * 0.173);
        }
        if (is_real_inverse_api(spec.api)) {
            for (int batch = 0; batch < spec.batch; ++batch) {
                const int row = layout.padded;
                host[static_cast<std::size_t>(batch * row + 1)] = 0.0;
                if (layout.n % 2 == 0) host[static_cast<std::size_t>(batch * row + layout.n + 1)] = 0.0;
            }
        }
        check_cuda(cudaMemcpy(device, host.data(), layout.in_bytes, cudaMemcpyHostToDevice),
                   "copy seeded double input");
    }
}

FlagfftPlanHandle make_flagfft_plan(const CaseSpec &spec, const BufferLayout &layout) {
    flagfftHandle raw = nullptr;
    flagfftResult result = FLAGFFT_NOT_SUPPORTED;
    if (spec.plan_api == PlanApi::Plan1d) {
        result = flagfftPlan1d(&raw, layout.n, flagfft_type(spec.api), spec.batch);
    } else if (spec.plan_api == PlanApi::PlanMany) {
        int dims[1] = {layout.n};
        int padded[1] = {layout.padded};
        int compact[1] = {layout.half};
        result = is_real_forward_api(spec.api)
                     ? flagfftPlanMany(&raw, 1, dims, padded, 1, layout.padded, compact, 1,
                                       layout.half, flagfft_type(spec.api), spec.batch)
                     : flagfftPlanMany(&raw, 1, dims, compact, 1, layout.half, padded, 1,
                                       layout.padded, flagfft_type(spec.api), spec.batch);
    }
    check_flagfft(result, "create FlagFFT plan");
    return FlagfftPlanHandle(raw);
}

CufftPlanHandle make_cufft_plan(const CaseSpec &spec, const BufferLayout &layout) {
    CufftPlanHandle plan;
    if (spec.plan_api == PlanApi::PlanMany) {
        int dims[1] = {layout.n};
        int padded[1] = {layout.padded};
        int compact[1] = {layout.half};
        check_cufft(is_real_forward_api(spec.api)
                        ? cufftPlanMany(plan.put(), 1, dims, padded, 1, layout.padded, compact, 1,
                                        layout.half, cufft_type(spec.api), spec.batch)
                        : cufftPlanMany(plan.put(), 1, dims, compact, 1, layout.half, padded, 1,
                                        layout.padded, cufft_type(spec.api), spec.batch),
                    "create cuFFT planmany");
    } else {
        check_cufft(cufftPlan1d(plan.put(), layout.n, cufft_type(spec.api), spec.batch),
                    "create cuFFT plan1d");
    }
    return plan;
}

void exec_flagfft(flagfftHandle plan, const CaseSpec &spec, void *input, void *output) {
    switch (spec.api) {
        case FftApi::C2C:
            check_flagfft(flagfftExecC2C(plan, static_cast<flagfftComplex *>(input),
                                         static_cast<flagfftComplex *>(output), spec.direction),
                          "flagfftExecC2C");
            break;
        case FftApi::Z2Z:
            check_flagfft(flagfftExecZ2Z(plan, static_cast<flagfftDoubleComplex *>(input),
                                         static_cast<flagfftDoubleComplex *>(output), spec.direction),
                          "flagfftExecZ2Z");
            break;
        case FftApi::R2C:
            check_flagfft(flagfftExecR2C(plan, static_cast<flagfftReal *>(input),
                                         static_cast<flagfftComplex *>(output)), "flagfftExecR2C");
            break;
        case FftApi::D2Z:
            check_flagfft(flagfftExecD2Z(plan, static_cast<flagfftDoubleReal *>(input),
                                         static_cast<flagfftDoubleComplex *>(output)), "flagfftExecD2Z");
            break;
        case FftApi::C2R:
            check_flagfft(flagfftExecC2R(plan, static_cast<flagfftComplex *>(input),
                                         static_cast<flagfftReal *>(output)), "flagfftExecC2R");
            break;
        case FftApi::Z2D:
            check_flagfft(flagfftExecZ2D(plan, static_cast<flagfftDoubleComplex *>(input),
                                         static_cast<flagfftDoubleReal *>(output)), "flagfftExecZ2D");
            break;
    }
}

void exec_cufft(cufftHandle plan, const CaseSpec &spec, void *input, void *output) {
    const int direction = spec.direction == FLAGFFT_FORWARD ? CUFFT_FORWARD : CUFFT_INVERSE;
    switch (spec.api) {
        case FftApi::C2C:
            check_cufft(cufftExecC2C(plan, static_cast<cufftComplex *>(input),
                                     static_cast<cufftComplex *>(output), direction), "cufftExecC2C");
            break;
        case FftApi::Z2Z:
            check_cufft(cufftExecZ2Z(plan, static_cast<cufftDoubleComplex *>(input),
                                     static_cast<cufftDoubleComplex *>(output), direction), "cufftExecZ2Z");
            break;
        case FftApi::R2C:
            check_cufft(cufftExecR2C(plan, static_cast<cufftReal *>(input),
                                     static_cast<cufftComplex *>(output)), "cufftExecR2C");
            break;
        case FftApi::D2Z:
            check_cufft(cufftExecD2Z(plan, static_cast<cufftDoubleReal *>(input),
                                     static_cast<cufftDoubleComplex *>(output)), "cufftExecD2Z");
            break;
        case FftApi::C2R:
            check_cufft(cufftExecC2R(plan, static_cast<cufftComplex *>(input),
                                     static_cast<cufftReal *>(output)), "cufftExecC2R");
            break;
        case FftApi::Z2D:
            check_cufft(cufftExecZ2D(plan, static_cast<cufftDoubleComplex *>(input),
                                     static_cast<cufftDoubleReal *>(output)), "cufftExecZ2D");
            break;
    }
}

struct Execution {
    explicit Execution(const CaseSpec &s, bool explicit_stream)
        : spec(s), layout(layout_for(s)), ff_in(layout.allocation_bytes),
          cf_in(layout.allocation_bytes) {
        if (spec.placement == Placement::OutOfPlace) {
            ff_out.allocate(layout.out_bytes);
            cf_out.allocate(layout.out_bytes);
        }
        seed_input(ff_in.get(), layout, spec);
        seed_input(cf_in.get(), layout, spec);
        ff_plan = make_flagfft_plan(spec, layout);
        cf_plan = make_cufft_plan(spec, layout);
        if (explicit_stream) {
            stream = std::make_unique<Stream>();
            check_flagfft(flagfftSetStream(ff_plan.get(), stream->get()), "flagfftSetStream");
            check_cufft(cufftSetStream(cf_plan.get(), stream->get()), "cufftSetStream");
        }
    }

    void *ff_output() { return spec.placement == Placement::InPlace ? ff_in.get() : ff_out.get(); }
    void *cf_output() { return spec.placement == Placement::InPlace ? cf_in.get() : cf_out.get(); }
    void sync() {
        if (stream) stream->sync();
        else check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }

    CaseSpec spec;
    BufferLayout layout;
    DeviceMemory ff_in;
    DeviceMemory cf_in;
    DeviceMemory ff_out;
    DeviceMemory cf_out;
    FlagfftPlanHandle ff_plan;
    CufftPlanHandle cf_plan;
    std::unique_ptr<Stream> stream;
};

json compare_outputs(Execution &run) {
    const std::size_t bytes = run.spec.placement == Placement::InPlace
                                  ? run.layout.allocation_bytes : run.layout.out_bytes;
    double max_abs = 0.0;
    double sum_sq = 0.0;
    std::size_t compared = 0;
    std::size_t non_finite_values = 0;
    auto compare_value = [&](double actual, double expected) {
        const double delta = actual - expected;
        ++compared;
        if (!std::isfinite(actual) || !std::isfinite(expected) || !std::isfinite(delta)) {
            ++non_finite_values;
            return;
        }
        max_abs = std::max(max_abs, std::abs(delta));
        sum_sq += delta * delta;
    };
    if (run.layout.scalar_bytes == sizeof(float)) {
        std::vector<float> actual(bytes / sizeof(float));
        std::vector<float> expected(actual.size());
        check_cuda(cudaMemcpy(actual.data(), run.ff_output(), bytes, cudaMemcpyDeviceToHost), "copy FlagFFT output");
        check_cuda(cudaMemcpy(expected.data(), run.cf_output(), bytes, cudaMemcpyDeviceToHost), "copy cuFFT output");
        for (int b = 0; b < run.spec.batch; ++b) {
            const int count = is_real_inverse_api(run.spec.api) ? run.layout.n
                              : is_real_forward_api(run.spec.api) ? 2 * run.layout.half
                              : 2 * run.layout.n;
            const int stride = run.spec.placement == Placement::InPlace &&
                                       is_real_inverse_api(run.spec.api) ? run.layout.padded : count;
            for (int i = 0; i < count; ++i) {
                compare_value(static_cast<double>(actual[b * stride + i]),
                              static_cast<double>(expected[b * stride + i]));
            }
        }
    } else {
        std::vector<double> actual(bytes / sizeof(double));
        std::vector<double> expected(actual.size());
        check_cuda(cudaMemcpy(actual.data(), run.ff_output(), bytes, cudaMemcpyDeviceToHost), "copy FlagFFT output");
        check_cuda(cudaMemcpy(expected.data(), run.cf_output(), bytes, cudaMemcpyDeviceToHost), "copy cuFFT output");
        for (int b = 0; b < run.spec.batch; ++b) {
            const int count = is_real_inverse_api(run.spec.api) ? run.layout.n
                              : is_real_forward_api(run.spec.api) ? 2 * run.layout.half
                              : 2 * run.layout.n;
            const int stride = run.spec.placement == Placement::InPlace &&
                                       is_real_inverse_api(run.spec.api) ? run.layout.padded : count;
            for (int i = 0; i < count; ++i) {
                compare_value(actual[b * stride + i], expected[b * stride + i]);
            }
        }
    }
    const double tolerance = is_double_api(run.spec.api)
                                 ? 2.0e-9 * (run.spec.direction == FLAGFFT_INVERSE ? run.layout.n : 1)
                                 : 4.0e-3 * (run.spec.direction == FLAGFFT_INVERSE ? run.layout.n : 1);
    const bool finite = non_finite_values == 0;
    return {
        {"max_abs_error", finite ? json(max_abs) : json(nullptr)},
        {"rms_error", finite ? json(std::sqrt(sum_sq / static_cast<double>(compared))) : json(nullptr)},
        {"non_finite_values", non_finite_values},
        {"tolerance", tolerance},
        {"passed", finite && max_abs <= tolerance},
    };
}

double percentile(std::vector<float> times, double fraction) {
    std::sort(times.begin(), times.end());
    const std::size_t index = std::min(times.size() - 1,
        static_cast<std::size_t>(fraction * static_cast<double>(times.size())));
    return times[index];
}

}  // namespace

json run_correctness(const CaseSpec &spec, bool include_path) {
    Execution run(spec, spec.stream);
    exec_flagfft(run.ff_plan.get(), spec, run.ff_in.get(), run.ff_output());
    exec_cufft(run.cf_plan.get(), spec, run.cf_in.get(), run.cf_output());
    run.sync();
    json result = {{"case", case_json(spec)}, {"correctness", compare_outputs(run)}};
    if (include_path) {
        const char *description = flagfftGetPlanDescription(run.ff_plan.get());
        result["plan_description"] = description == nullptr ? "" : description;
    }
    return result;
}

json run_benchmark(const CaseSpec &spec, const TimingConfig &timing, bool include_path) {
    Execution run(spec, true);
    exec_flagfft(run.ff_plan.get(), spec, run.ff_in.get(), run.ff_output());
    exec_cufft(run.cf_plan.get(), spec, run.cf_in.get(), run.cf_output());
    run.sync();
    const json correctness = compare_outputs(run);
    if (!correctness.at("passed").get<bool>()) {
        return {{"case", case_json(spec)}, {"correctness", correctness}};
    }
    for (int i = 0; i < timing.warmup; ++i) {
        exec_flagfft(run.ff_plan.get(), spec, run.ff_in.get(), run.ff_output());
        exec_cufft(run.cf_plan.get(), spec, run.cf_in.get(), run.cf_output());
    }
    run.sync();
    Timer timer;
    std::vector<float> flagfft_times;
    std::vector<float> cufft_times;
    flagfft_times.reserve(timing.iters);
    cufft_times.reserve(timing.iters);
    auto time_flagfft = [&]() {
        timer.start(run.stream->get());
        for (int i = 0; i < timing.launches_per_sample; ++i)
            exec_flagfft(run.ff_plan.get(), spec, run.ff_in.get(), run.ff_output());
        timer.stop(run.stream->get());
        return timer.elapsed_ms() / timing.launches_per_sample;
    };
    auto time_cufft = [&]() {
        timer.start(run.stream->get());
        for (int i = 0; i < timing.launches_per_sample; ++i)
            exec_cufft(run.cf_plan.get(), spec, run.cf_in.get(), run.cf_output());
        timer.stop(run.stream->get());
        return timer.elapsed_ms() / timing.launches_per_sample;
    };
    for (int i = 0; i < timing.iters; ++i) {
        if ((i & 1) == 0) {
            cufft_times.push_back(time_cufft());
            flagfft_times.push_back(time_flagfft());
        } else {
            flagfft_times.push_back(time_flagfft());
            cufft_times.push_back(time_cufft());
        }
    }
    const double ff_median = percentile(flagfft_times, 0.5);
    const double cf_median = percentile(cufft_times, 0.5);
    json result = {
        {"case", case_json(spec)},
        {"correctness", correctness},
        {"timing", {
            {"warmup", timing.warmup}, {"iters", timing.iters},
            {"launches_per_sample", timing.launches_per_sample},
            {"flagfft_median_ms", ff_median}, {"flagfft_p90_ms", percentile(flagfft_times, 0.9)},
            {"cufft_median_ms", cf_median}, {"cufft_p90_ms", percentile(cufft_times, 0.9)},
            {"speedup", ff_median > 0.0 ? cf_median / ff_median : 0.0},
        }},
    };
    if (include_path) {
        const char *description = flagfftGetPlanDescription(run.ff_plan.get());
        result["plan_description"] = description == nullptr ? "" : description;
    }
    return result;
}

}  // namespace flagfft::cli
