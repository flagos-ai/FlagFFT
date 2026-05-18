#include "flagfft/core.hpp"

namespace flagfft {

std::shared_ptr<ExecutablePlan> PlanCache::get_or_create(const FFTRequest &request) {
    ProblemKey problem_key = ProblemKey::from_request(request);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = problem_cache_.find(problem_key);
        if (it != problem_cache_.end()) {
            ++problem_hits_;
            return it->second;
        }
        ++problem_misses_;
    }

    PlanBuilder builder;
    std::shared_ptr<ExecutablePlan> tuned_executable;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++tuned_db_lookups_;
    }
    if (auto tuned_plan = lookup_tuned_plan_dict(request)) {
        try {
            PlanNodePtr tuned_root = plan_node_from_wrapped_dict(builder, *tuned_plan);
            tuned_executable =
                build_executable_from_root(request, builder, tuned_root, "sqlite_tuned");
        } catch (const nb::python_error &) {
            PyErr_Clear();
        } catch (const std::exception &) {
        }
    }

    PlanNodePtr root = tuned_executable ? tuned_executable->root : builder.build(request.requested_n, request);
    PlanKey plan_key = tuned_executable ? tuned_executable->plan_key : PlanKey::from_node(root);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = plan_cache_.find(plan_key);
        if (it != plan_cache_.end()) {
            ++plan_hits_;
            if (!tuned_executable) {
                root = it->second;
            }
        } else {
            ++plan_misses_;
            plan_cache_.emplace(plan_key, root);
        }
    }
    auto executable = tuned_executable ? tuned_executable
                                       : build_executable_from_root(request, builder, root, "cpp_auto");

    std::lock_guard<std::mutex> lock(mutex_);
    auto [it, inserted] = problem_cache_.emplace(problem_key, executable);
    return inserted ? executable : it->second;
}

void PlanCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    problem_cache_.clear();
    plan_cache_.clear();
    problem_hits_ = 0;
    problem_misses_ = 0;
    plan_hits_ = 0;
    plan_misses_ = 0;
    tuned_db_lookups_ = 0;
    TritonCompiler::clear_kernel_cache();
}

nb::dict PlanCache::info() const {
    std::lock_guard<std::mutex> lock(mutex_);
    nb::dict out;
    out["problem_size"] = problem_cache_.size();
    out["problem_hits"] = problem_hits_;
    out["problem_misses"] = problem_misses_;
    out["plan_size"] = plan_cache_.size();
    out["plan_hits"] = plan_hits_;
    out["plan_misses"] = plan_misses_;
    out["tuned_db_lookups"] = tuned_db_lookups_;
    nb::dict kernel = TritonCompiler::kernel_cache_info();
    out["kernel_size"] = kernel["kernel_size"];
    out["kernel_hits"] = kernel["kernel_hits"];
    out["kernel_misses"] = kernel["kernel_misses"];
    return out;
}

nb::dict PlanCache::keys() const {
    std::lock_guard<std::mutex> lock(mutex_);
    nb::dict out;
    nb::list problems;
    nb::list plans;
    for (const auto &entry : problem_cache_) {
        problems.append(problem_key_to_dict(entry.first));
    }
    for (const auto &entry : plan_cache_) {
        plans.append(plan_key_to_dict(entry.first));
    }
    out["problems"] = std::move(problems);
    out["plans"] = std::move(plans);
    out["kernels"] = TritonCompiler::kernel_cache_keys();
    return out;
}

PlanCache &plan_cache() {
    static PlanCache *cache = new PlanCache();
    return *cache;
}

}  // namespace flagfft
