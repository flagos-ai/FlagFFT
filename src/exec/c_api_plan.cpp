#include "c_api_internal.hpp"
#include "flagfft/tune_json.hpp"
#include "runtime/device.hpp"

namespace flagfft {
namespace {

PlanNodePtr raw_compatible_bluestein_plan(int64_t length,
                                          PlanBuilder &builder,
                                          const FFTRequest &request) {
    int64_t conv_length = ceil_power_of_two(2 * length - 1);
    PlanNodePtr child = builder.build(conv_length, request);
    PlanNodePtr candidate =
        std::make_shared<BluesteinPlanNode>(length, conv_length, std::move(child));
    return raw_supported_node(candidate) ? candidate : nullptr;
}

PlanNodePtr lookup_or_build_root(PlanBuilder &builder, const FFTRequest &request) {
    auto tuned = lookup_tuned_plan_json(request);
    if (tuned.has_value()) {
        try {
            PlanNodePtr root = plan_node_from_json(builder, tuned->at("root"));
            if (raw_supported_node(root)) {
                return root;
            }
        } catch (const std::exception &) {
        }
    }
    return builder.build(request.requested_n, request);
}

}  // namespace

flagfftResult build_plan(flagfftHandle *out, FlagFFTPlanDesc desc) {
    if (out == nullptr) {
        return FLAGFFT_INVALID_VALUE;
    }
    *out = nullptr;
    if (!is_supported_minimal_desc(desc)) {
        return FLAGFFT_NOT_SUPPORTED;
    }

    flagfftResult device_result = runtime::ensure_device(desc.device_index, desc.device_arch);
    if (device_result != FLAGFFT_SUCCESS) {
        return device_result;
    }

    try {
        auto plan = std::make_unique<FlagFFTPlan>();
        plan->desc = std::move(desc);
        plan->executable.forward_request = request_from_desc(plan->desc, "forward");
        plan->executable.inverse_request = request_from_desc(plan->desc, "inverse");

        PlanBuilder builder;

        plan->executable.root = lookup_or_build_root(builder, plan->executable.forward_request);
        if (!raw_supported_node(plan->executable.root)) {
            plan->executable.root = lookup_or_build_root(builder, plan->executable.inverse_request);
        }
        if (!raw_supported_node(plan->executable.root)) {
            if (PlanNodePtr fallback = raw_compatible_bluestein_plan(
                    plan->executable.forward_request.requested_n,
                    builder,
                    plan->executable.forward_request)) {
                plan->executable.root = std::move(fallback);
            }
        }
        if (!raw_supported_node(plan->executable.root)) {
            return FLAGFFT_NOT_SUPPORTED;
        }

        plan->executable.forward_problem_key =
            ProblemKey::from_request(plan->executable.forward_request);
        plan->executable.inverse_problem_key =
            ProblemKey::from_request(plan->executable.inverse_request);
        plan->executable.plan_key = PlanKey::from_node(plan->executable.root);

        TritonCompiler compiler;
        if (plan->desc.type == FLAGFFT_R2C || plan->desc.type == FLAGFFT_D2Z) {
            plan->executable.forward = compiler.compile_raw_r2c_node(
                plan->executable.root, plan->executable.forward_request, plan->desc.batch);
        } else if (plan->desc.type == FLAGFFT_C2R || plan->desc.type == FLAGFFT_Z2D) {
            plan->executable.inverse = compiler.compile_raw_c2r_node(
                plan->executable.root, plan->executable.inverse_request, plan->desc.batch);
        } else {
            plan->executable.forward = compiler.compile_raw_node(
                plan->executable.root, plan->executable.forward_request, plan->desc.batch);
            plan->executable.inverse = compiler.compile_raw_node(
                plan->executable.root, plan->executable.inverse_request, plan->desc.batch);
        }
        plan->state.initialized = true;

        std::unique_ptr<flagfftPlan_t> handle(new flagfftPlan_t());
        handle->impl = plan.release();
        *out = handle.release();
        return FLAGFFT_SUCCESS;
    } catch (const std::bad_alloc &) {
        return FLAGFFT_ALLOC_FAILED;
    } catch (const std::exception &) {
        return FLAGFFT_SETUP_FAILED;
    }
}

}  // namespace flagfft

extern "C" flagfftResult flagfftPlan1d(flagfftHandle *plan,
                                       int nx,
                                       flagfftType type,
                                       int batch) {
    int n[1] = {nx};
    const bool real_forward = type == FLAGFFT_R2C || type == FLAGFFT_D2Z;
    const bool real_inverse = type == FLAGFFT_C2R || type == FLAGFFT_Z2D;
    const int half = nx / 2 + 1;
    const int idist = real_inverse ? half : nx;
    const int odist = real_forward ? half : nx;
    return flagfftPlanMany(plan, 1, n, nullptr, 1, idist, nullptr, 1, odist, type, batch);
}

extern "C" flagfftResult flagfftPlan2d(flagfftHandle *plan,
                                       int nx,
                                       int ny,
                                       flagfftType type) {
    int n[2] = {nx, ny};
    return flagfftPlanMany(plan, 2, n, nullptr, 1, nx * ny, nullptr, 1, nx * ny, type, 1);
}

extern "C" flagfftResult flagfftPlan3d(flagfftHandle *plan,
                                       int nx,
                                       int ny,
                                       int nz,
                                       flagfftType type) {
    int n[3] = {nx, ny, nz};
    return flagfftPlanMany(plan, 3, n, nullptr, 1, nx * ny * nz, nullptr, 1, nx * ny * nz, type, 1);
}

extern "C" flagfftResult flagfftPlanMany(flagfftHandle *plan,
                                         int rank,
                                         int *n,
                                         int *inembed,
                                         int istride,
                                         int idist,
                                         int *onembed,
                                         int ostride,
                                         int odist,
                                         flagfftType type,
                                         int batch) {
    if (plan == nullptr) {
        return FLAGFFT_INVALID_VALUE;
    }
    *plan = nullptr;
    if (rank <= 0 || n == nullptr || batch <= 0) {
        return FLAGFFT_INVALID_VALUE;
    }
    for (int i = 0; i < rank; ++i) {
        if (n[i] <= 0) {
            return FLAGFFT_INVALID_SIZE;
        }
    }

    flagfft::FlagFFTPlanDesc desc;
    desc.rank = rank;
    desc.n = flagfft::copy_dims(n, rank);
    desc.istride = istride;
    desc.idist = idist;
    desc.ostride = ostride;
    desc.odist = odist;
    desc.batch = batch;
    desc.type = type;

    flagfftResult type_result = flagfft::type_metadata(
        type, desc.precision, desc.kind, desc.real_input, desc.real_output);
    if (type_result != FLAGFFT_SUCCESS) {
        return type_result;
    }

    if (inembed != nullptr) {
        desc.inembed = flagfft::copy_dims(inembed, rank);
    } else if ((type == FLAGFFT_C2R || type == FLAGFFT_Z2D) && rank == 1) {
        desc.inembed = {desc.n[0] / 2 + 1};
    } else {
        desc.inembed = desc.n;
    }
    if (onembed != nullptr) {
        desc.onembed = flagfft::copy_dims(onembed, rank);
    } else if ((type == FLAGFFT_R2C || type == FLAGFFT_D2Z) && rank == 1) {
        desc.onembed = {desc.n[0] / 2 + 1};
    } else {
        desc.onembed = desc.n;
    }

    return flagfft::build_plan(plan, std::move(desc));
}
