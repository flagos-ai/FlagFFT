#include "flagfft/core.hpp"

namespace flagfft {

std::shared_ptr<ExecutablePlan> resolve_plan(nb::object input,
                                             nb::object n_obj,
                                             int64_t dim,
                                             nb::object norm_obj,
                                             std::string direction) {
    FFTRequest request = build_request(input, n_obj, dim, norm_obj, std::move(direction));
    validate_request(request);
    return plan_cache().get_or_create(request);
}

nb::object fft(nb::object input, nb::object n_obj, int64_t dim, nb::object norm_obj) {
    std::shared_ptr<ExecutablePlan> executable =
        resolve_plan(input, n_obj, dim, norm_obj, "forward");
    return executable->execute(input);
}

nb::object ifft(nb::object input, nb::object n_obj, int64_t dim, nb::object norm_obj) {
    std::shared_ptr<ExecutablePlan> executable =
        resolve_plan(input, n_obj, dim, norm_obj, "inverse");
    return executable->execute(input);
}

nb::object execute_with_plan(nb::object input,
                             nb::dict plan,
                             nb::object n_obj,
                             int64_t dim,
                             nb::object norm_obj,
                             std::string direction) {
    FFTRequest request = build_request(input, n_obj, dim, norm_obj, std::move(direction));
    validate_request(request);
    PlanBuilder builder;
    PlanNodePtr root = plan_node_from_wrapped_dict(builder, plan);
    auto executable = build_executable_from_root(request, builder, root, "forced");
    return executable->execute(input);
}

nb::object fft_with_plan(nb::object input,
                         nb::dict plan,
                         nb::object n_obj,
                         int64_t dim,
                         nb::object norm_obj) {
    return execute_with_plan(input, std::move(plan), n_obj, dim, norm_obj, "forward");
}

nb::object ifft_with_plan(nb::object input,
                          nb::dict plan,
                          nb::object n_obj,
                          int64_t dim,
                          nb::object norm_obj) {
    return execute_with_plan(input, std::move(plan), n_obj, dim, norm_obj, "inverse");
}

nb::dict debug_request(nb::object input,
                       nb::object n_obj,
                       int64_t dim,
                       nb::object norm_obj,
                       std::string direction) {
    return request_to_dict(build_request(input, n_obj, dim, norm_obj, std::move(direction)));
}

nb::dict debug_keys(nb::object input,
                    nb::object n_obj,
                    int64_t dim,
                    nb::object norm_obj,
                    std::string direction) {
    FFTRequest request = build_request(input, n_obj, dim, norm_obj, std::move(direction));
    validate_request(request);
    PlanBuilder builder;
    PlanNodePtr root = builder.build(request.requested_n, request);
    nb::dict out;
    out["problem"] = problem_key_to_dict(ProblemKey::from_request(request));
    out["plan"] = plan_key_to_dict(PlanKey::from_node(root));
    out["kernels"] = kernel_keys_for_plan(root, request);
    return out;
}

nb::dict debug_plan(nb::object input,
                    nb::object n_obj,
                    int64_t dim,
                    nb::object norm_obj,
                    std::string direction) {
    FFTRequest request = build_request(input, n_obj, dim, norm_obj, std::move(direction));
    validate_request(request);
    PlanBuilder builder;
    PlanNodePtr root = builder.build(request.requested_n, request);
    return builder.wrap_plan_dict(root, request);
}

nb::dict debug_resolved_plan(nb::object input,
                             nb::object n_obj,
                             int64_t dim,
                             nb::object norm_obj,
                             std::string direction) {
    std::shared_ptr<ExecutablePlan> executable =
        resolve_plan(input, n_obj, dim, norm_obj, std::move(direction));
    nb::dict out(executable->plan_dict);
    out["kernels"] = kernel_keys_for_plan(executable->root, executable->request);
    return out;
}

nb::dict debug_forced_plan(nb::object input,
                           nb::dict plan,
                           nb::object n_obj,
                           int64_t dim,
                           nb::object norm_obj,
                           std::string direction) {
    FFTRequest request = build_request(input, n_obj, dim, norm_obj, std::move(direction));
    validate_request(request);
    PlanBuilder builder;
    PlanNodePtr root = plan_node_from_wrapped_dict(builder, plan);
    nb::dict out = builder.wrap_forced_plan_dict(root, request, "forced");
    out["kernels"] = kernel_keys_for_plan(root, request);
    return out;
}

nb::list enumerate_plan_candidates(nb::object input,
                                   nb::object n_obj,
                                   int64_t dim,
                                   nb::object norm_obj,
                                   std::string direction) {
    FFTRequest request = build_request(input, n_obj, dim, norm_obj, std::move(direction));
    validate_request(request);
    PlanBuilder builder;
    return builder.enumerate_candidate_plans(request.requested_n, request);
}

nb::dict tune_fingerprints() {
    nb::dict out;
    out["planner"] = "planner-schema-2-device-smem-leaf";
    out["codegen"] = "codegen-schema-3-directional-ifft";
    out["runtime"] = "runtime-schema-3-directional-ifft";
    out["benchmark"] = "benchmark-schema-2-api-dispatch";
    return out;
}

}  // namespace flagfft
