#pragma once

#include "flagfft/codegen.hpp"

namespace flagfft {

enum class ExecutionBackend { JitCuda, TorchFFT };

struct ExecutablePlan {
    ProblemKey problem_key;
    PlanKey plan_key;
    FFTRequest request;
    PlanNodePtr root;
    nb::dict plan_dict;
    ExecutionBackend backend;
    std::shared_ptr<CompiledNode> compiled_root;

    nb::object execute(nb::object input) const;
};

class PlanCache {
public:
    std::shared_ptr<ExecutablePlan> get_or_create(const FFTRequest &request);
    void clear();
    nb::dict info() const;
    nb::dict keys() const;

private:
    mutable std::mutex mutex_;
    std::unordered_map<ProblemKey, std::shared_ptr<ExecutablePlan>, ProblemKeyHash> problem_cache_;
    std::unordered_map<PlanKey, PlanNodePtr, PlanKeyHash> plan_cache_;
    int64_t problem_hits_ = 0;
    int64_t problem_misses_ = 0;
    int64_t plan_hits_ = 0;
    int64_t plan_misses_ = 0;
    int64_t tuned_db_lookups_ = 0;
};

PlanCache &plan_cache();
std::shared_ptr<ExecutablePlan> resolve_plan(nb::object input,
                                             nb::object n_obj,
                                             int64_t dim,
                                             nb::object norm_obj,
                                             std::string direction);
nb::object fft(nb::object input, nb::object n_obj, int64_t dim, nb::object norm_obj);
nb::object ifft(nb::object input, nb::object n_obj, int64_t dim, nb::object norm_obj);
nb::object fft_with_plan(nb::object input,
                         nb::dict plan,
                         nb::object n_obj,
                         int64_t dim,
                         nb::object norm_obj);
nb::object ifft_with_plan(nb::object input,
                          nb::dict plan,
                          nb::object n_obj,
                          int64_t dim,
                          nb::object norm_obj);
nb::dict debug_request(nb::object input,
                       nb::object n_obj,
                       int64_t dim,
                       nb::object norm_obj,
                       std::string direction);
nb::dict debug_keys(nb::object input,
                    nb::object n_obj,
                    int64_t dim,
                    nb::object norm_obj,
                    std::string direction);
nb::dict debug_plan(nb::object input,
                    nb::object n_obj,
                    int64_t dim,
                    nb::object norm_obj,
                    std::string direction);
nb::dict debug_resolved_plan(nb::object input,
                             nb::object n_obj,
                             int64_t dim,
                             nb::object norm_obj,
                             std::string direction);
nb::dict debug_forced_plan(nb::object input,
                           nb::dict plan,
                           nb::object n_obj,
                           int64_t dim,
                           nb::object norm_obj,
                           std::string direction);
nb::list enumerate_plan_candidates(nb::object input,
                                   nb::object n_obj,
                                   int64_t dim,
                                   nb::object norm_obj,
                                   std::string direction);
nb::dict tune_fingerprints();

int64_t four_step_col_inner_pack_for(int64_t n1, int64_t n2);
bool request_has_flat_batch_shape(const FFTRequest &request);
std::string batch_bucket(int64_t batch);
bool env_flag_enabled(const char *value);
double normalization_scale(const FFTRequest &request);
std::optional<std::filesystem::path> tuned_db_path();
PlanNodePtr plan_node_from_wrapped_dict(PlanBuilder &builder, nb::dict plan);
std::shared_ptr<ExecutablePlan> build_executable_from_root(const FFTRequest &request,
                                                           PlanBuilder &builder,
                                                           PlanNodePtr root,
                                                           std::string source);
std::optional<nb::dict> lookup_tuned_plan_dict(const FFTRequest &request);

}  // namespace flagfft
