#include "flagfft/core.hpp"

namespace flagfft {

inline constexpr int64_t kFourStepColInnerPack = 2;
inline constexpr int64_t kFourStepColInnerPackMinN1 = 128;

int64_t four_step_col_inner_pack_for(int64_t n1, int64_t n2) {
    (void)n2;
    if (n1 < kFourStepColInnerPackMinN1) {
        return 1;
    }
    return kFourStepColInnerPack;
}

bool request_has_flat_batch_shape(const FFTRequest &request) {
    return request.input_shape.size() == 2 &&
           request.input_shape[0] == request.batch &&
           request.input_shape[1] == request.requested_n;
}

std::string batch_bucket(int64_t batch) {
    if (batch <= 1) {
        return "1";
    }
    if (batch <= 8) {
        return "2-8";
    }
    if (batch <= 64) {
        return "9-64";
    }
    if (batch <= 512) {
        return "65-512";
    }
    return "513+";
}

bool env_flag_enabled(const char *value) {
    if (value == nullptr) {
        return false;
    }
    std::string normalized(value);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return !(normalized.empty() || normalized == "0" || normalized == "false" ||
             normalized == "off" || normalized == "no");
}

double normalization_scale(const FFTRequest &request) {
    if (request.norm == "ortho") {
        return 1.0 / std::sqrt(static_cast<double>(request.requested_n));
    }
    if (request.direction == "forward" && request.norm == "forward") {
        return 1.0 / static_cast<double>(request.requested_n);
    }
    if (request.direction == "inverse" && request.norm == "backward") {
        return 1.0 / static_cast<double>(request.requested_n);
    }
    return 1.0;
}

std::optional<std::filesystem::path> tuned_db_path() {
    if (env_flag_enabled(std::getenv("FLAGFFT_TUNE_DISABLE"))) {
        return std::nullopt;
    }
    const char *override_path = std::getenv("FLAGFFT_TUNE_DB");
    if (override_path != nullptr && std::string(override_path).size() > 0) {
        return std::filesystem::path(override_path);
    }
    return default_cache_dir() / "tuned_plans.sqlite";
}

PlanNodePtr plan_node_from_wrapped_dict(PlanBuilder &builder, nb::dict plan) {
    nb::dict root = plan.contains("root") ? nb::cast<nb::dict>(plan["root"]) : plan;
    return builder.node_from_dict(root);
}

std::shared_ptr<ExecutablePlan> build_executable_from_root(const FFTRequest &request,
                                                           PlanBuilder &builder,
                                                           PlanNodePtr root,
                                                           std::string source) {
    PlanKey plan_key = PlanKey::from_node(root);
    nb::dict plan_dict = builder.wrap_forced_plan_dict(root, request, std::move(source));

    ExecutionBackend backend = ExecutionBackend::TorchFFT;
    std::shared_ptr<CompiledNode> compiled_root;
    if (request.output_dtype == "complex64") {
        backend = ExecutionBackend::AotCuda;
        TritonCompiler compiler;
        compiled_root = compiler.compile_node(root, request, request.batch);
    }

    return std::make_shared<ExecutablePlan>(ExecutablePlan{
        ProblemKey::from_request(request), plan_key, request, std::move(root), std::move(plan_dict),
        backend, std::move(compiled_root)});
}

std::optional<nb::dict> lookup_tuned_plan_dict(const FFTRequest &request) {
    auto db_path = tuned_db_path();
    if (!db_path.has_value()) {
        return std::nullopt;
    }
    std::error_code ec;
    if (!std::filesystem::is_regular_file(*db_path, ec)) {
        return std::nullopt;
    }
    try {
        nb::dict fps = tune_fingerprints();
        nb::module_ sqlite3 = nb::module_::import_("sqlite3");
        nb::object conn = sqlite3.attr("connect")(db_path->string());
        nb::object row = nb::none();
        try {
            nb::object cursor = conn.attr("execute")(
                "SELECT plan_json FROM tuned_measurements "
                "WHERE schema_version=? AND status='valid' AND rank=0 "
                "AND device_arch=? AND fft_length=? AND batch_bucket=? AND dtype=? "
                "AND direction=? AND norm=? AND input_layout=? "
                "AND planner_fingerprint=? AND codegen_fingerprint=? "
                "AND runtime_fingerprint=? "
                "ORDER BY measured_at DESC LIMIT 1",
                nb::make_tuple(kPlanSchemaVersion,
                               request.device_arch,
                               request.requested_n,
                               batch_bucket(request.batch),
                               request.input_dtype,
                               request.direction,
                               request.norm,
                               request.input_layout,
                               fps["planner"],
                               fps["codegen"],
                               fps["runtime"]));
            row = cursor.attr("fetchone")();
        } catch (...) {
            conn.attr("close")();
            throw;
        }
        conn.attr("close")();
        if (row.is_none()) {
            return std::nullopt;
        }
        nb::tuple row_tuple = nb::cast<nb::tuple>(row);
        std::string plan_json = nb::cast<std::string>(row_tuple[0]);
        nb::module_ json = nb::module_::import_("json");
        return nb::cast<nb::dict>(json.attr("loads")(plan_json));
    } catch (const nb::python_error &) {
        PyErr_Clear();
        return std::nullopt;
    } catch (const std::exception &) {
        return std::nullopt;
    }
}

}  // namespace flagfft
