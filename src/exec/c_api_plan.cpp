#include "c_api_internal.hpp"

namespace flagfft {

flagfftResult build_plan(flagfftHandle *out, FlagFFTPlanDesc desc) {
    if (out == nullptr) {
        return FLAGFFT_INVALID_VALUE;
    }
    *out = nullptr;
    if (!is_supported_minimal_desc(desc)) {
        return FLAGFFT_NOT_SUPPORTED;
    }

    flagfftResult device_result = ensure_current_cuda_device(desc.device_index, desc.device_arch);
    if (device_result != FLAGFFT_SUCCESS) {
        return device_result;
    }

    try {
        auto plan = std::make_unique<FlagFFTPlan>();
        plan->desc = std::move(desc);
        plan->executable.forward_request = request_from_desc(plan->desc, "forward");
        plan->executable.inverse_request = request_from_desc(plan->desc, "inverse");

        PlanBuilder builder;
        auto tuned_root_for = [&](const FFTRequest &request) -> PlanNodePtr {
            if (auto tuned_plan = lookup_tuned_plan_dict(request)) {
                try {
                    nb::gil_scoped_acquire acquire;
                    PlanNodePtr root = plan_node_from_wrapped_dict(builder, *tuned_plan);
                    if (raw_supported_node(root)) {
                        return root;
                    }
                } catch (const nb::python_error &) {
                    nb::gil_scoped_acquire acquire;
                    PyErr_Clear();
                } catch (const std::exception &) {
                }
            }
            return nullptr;
        };

        plan->executable.root = tuned_root_for(plan->executable.forward_request);
        if (plan->executable.root == nullptr) {
            plan->executable.root = tuned_root_for(plan->executable.inverse_request);
        }
        if (plan->executable.root == nullptr) {
            plan->executable.root =
                builder.build(plan->executable.forward_request.requested_n,
                              plan->executable.forward_request);
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
        plan->executable.forward = compiler.compile_raw_node(
            plan->executable.root, plan->executable.forward_request, plan->desc.batch);
        plan->executable.inverse = compiler.compile_raw_node(
            plan->executable.root, plan->executable.inverse_request, plan->desc.batch);
        plan->state.initialized = true;

        std::unique_ptr<flagfftPlan_t> handle(new flagfftPlan_t());
        handle->impl = plan.release();
        *out = handle.release();
        return FLAGFFT_SUCCESS;
    } catch (const nb::python_error &) {
        if (Py_IsInitialized()) {
            PyErr_Clear();
        }
        return FLAGFFT_SETUP_FAILED;
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
    return flagfftPlanMany(plan, 1, n, nullptr, 1, nx, nullptr, 1, nx, type, batch);
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
    desc.inembed = inembed == nullptr ? desc.n : flagfft::copy_dims(inembed, rank);
    desc.istride = istride;
    desc.idist = idist;
    desc.onembed = onembed == nullptr ? desc.n : flagfft::copy_dims(onembed, rank);
    desc.ostride = ostride;
    desc.odist = odist;
    desc.batch = batch;
    desc.type = type;

    flagfftResult type_result = flagfft::type_metadata(
        type, desc.precision, desc.kind, desc.real_input, desc.real_output);
    if (type_result != FLAGFFT_SUCCESS) {
        return type_result;
    }

    return flagfft::build_plan(plan, std::move(desc));
}
