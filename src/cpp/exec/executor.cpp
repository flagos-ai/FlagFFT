#include "flagfft/core.hpp"

namespace flagfft {
namespace {

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
    return std::filesystem::path(".flagfft") / "tuned_plans.sqlite";
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

}  // namespace

CompiledLeafNode::CompiledLeafNode(int64_t length,
                                   std::shared_ptr<AotKernel> kernel,
                                   std::vector<nb::object> tables)
    : length(length), kernel(std::move(kernel)), tables(std::move(tables)) {}

nb::object CompiledLeafNode::execute(const nb::object &input, const ExecutionContext &context) const {
    bool root_leaf = length == context.request.requested_n;
    int64_t batch = root_leaf ? context.request.batch : tensor_numel(input) / length;
    nb::object x_contig = (root_leaf && request_has_flat_batch_shape(context.request))
                              ? input
                              : input.attr("reshape")(nb::make_tuple(batch, length));
    nb::module_ torch = nb::module_::import_("torch");
    nb::object result = torch.attr("empty_like")(x_contig);

    std::vector<AotKernelArg> args;
    args.reserve(3 + tables.size());
    args.push_back(AotKernelArg::device(tensor_data_ptr(x_contig)));
    args.push_back(AotKernelArg::device(tensor_data_ptr(result)));
    for (const nb::object &table : tables) {
        args.push_back(AotKernelArg::device(tensor_data_ptr(table)));
    }
    args.push_back(AotKernelArg::i32(static_cast<int32_t>(batch)));

    kernel->launch(context.stream, args, ceil_div(batch, kernel->batch_per_block), 1, 1);
    return result;
}

CompiledFourStepNode::CompiledFourStepNode(int64_t length,
                                           int64_t n1,
                                           int64_t n2,
                                           std::shared_ptr<CompiledNode> row,
                                           std::shared_ptr<CompiledNode> col,
                                           std::shared_ptr<AotKernel> transpose_kernel,
                                           std::shared_ptr<AotKernel> twiddle_transpose_kernel,
                                           nb::object twiddle,
                                           nb::object stage0,
                                           nb::object stage2)
    : length(length),
      n1(n1),
      n2(n2),
      row(std::move(row)),
      col(std::move(col)),
      transpose_kernel(std::move(transpose_kernel)),
      twiddle_transpose_kernel(std::move(twiddle_transpose_kernel)),
      twiddle(std::move(twiddle)),
      stage0(std::move(stage0)),
      stage2(std::move(stage2)) {}

nb::object CompiledFourStepNode::execute(const nb::object &input, const ExecutionContext &context) const {
    bool root_node = length == context.request.requested_n;
    int64_t batch = root_node ? context.request.batch : tensor_numel(input) / length;
    nb::object x_contig = (root_node && request_has_flat_batch_shape(context.request))
                              ? input
                              : input.attr("reshape")(nb::make_tuple(batch, length));

    launch_transpose(context.stream, x_contig.attr("reshape")(nb::make_tuple(batch, n1, n2)), stage0);
    nb::object stage1 = row->execute(
        stage0.attr("reshape")(nb::make_tuple(batch * n2, n1)), context);
    stage1 = stage1.attr("reshape")(nb::make_tuple(batch, n2, n1));

    launch_twiddle_transpose(context.stream, stage1, twiddle, stage2);

    nb::object stage3 = col->execute(
        stage2.attr("reshape")(nb::make_tuple(batch * n1, n2)), context);
    stage3 = stage3.attr("reshape")(nb::make_tuple(batch, n1, n2));

    nb::object out = empty_complex64_tensor(context.request, nb::make_tuple(batch, n2, n1));
    launch_transpose(context.stream, stage3, out);
    return out.attr("reshape")(nb::make_tuple(batch, length));
}

void CompiledFourStepNode::launch_transpose(CUstream stream, const nb::object &src, const nb::object &dst) const {
    int64_t batch = tensor_size(src, 0);
    int64_t rows = tensor_size(src, 1);
    int64_t cols = tensor_size(src, 2);
    std::vector<AotKernelArg> args = {
        AotKernelArg::device(tensor_data_ptr(src)),
        AotKernelArg::device(tensor_data_ptr(dst)),
        AotKernelArg::i64(tensor_stride(src, 0) * 2),
        AotKernelArg::i64(tensor_stride(src, 1) * 2),
        AotKernelArg::i64(tensor_stride(src, 2) * 2),
        AotKernelArg::i64(tensor_stride(dst, 0) * 2),
        AotKernelArg::i64(tensor_stride(dst, 1) * 2),
        AotKernelArg::i64(tensor_stride(dst, 2) * 2),
        AotKernelArg::i64(rows),
        AotKernelArg::i64(cols),
    };
    transpose_kernel->launch(stream, args, ceil_div(cols, kFourStepTileCols),
                             ceil_div(rows, kFourStepTileRows), batch);
}

void CompiledFourStepNode::launch_twiddle_transpose(CUstream stream,
                                                    const nb::object &src,
                                                    const nb::object &twiddle,
                                                    const nb::object &dst) const {
    int64_t batch = tensor_size(src, 0);
    int64_t rows = tensor_size(src, 1);
    int64_t cols = tensor_size(src, 2);
    std::vector<AotKernelArg> args = {
        AotKernelArg::device(tensor_data_ptr(src)),
        AotKernelArg::device(tensor_data_ptr(twiddle)),
        AotKernelArg::device(tensor_data_ptr(dst)),
        AotKernelArg::i64(tensor_stride(src, 0) * 2),
        AotKernelArg::i64(tensor_stride(src, 1) * 2),
        AotKernelArg::i64(tensor_stride(src, 2) * 2),
        AotKernelArg::i64(tensor_stride(twiddle, 0) * 2),
        AotKernelArg::i64(tensor_stride(twiddle, 1) * 2),
        AotKernelArg::i64(tensor_stride(dst, 0) * 2),
        AotKernelArg::i64(tensor_stride(dst, 1) * 2),
        AotKernelArg::i64(tensor_stride(dst, 2) * 2),
        AotKernelArg::i64(rows),
        AotKernelArg::i64(cols),
    };
    twiddle_transpose_kernel->launch(stream, args, ceil_div(cols, kFourStepTileCols),
                                     ceil_div(rows, kFourStepTileRows), batch);
}

CompiledFourStepFusedNode::CompiledFourStepFusedNode(int64_t length,
                                                     int64_t n1,
                                                     int64_t n2,
                                                     std::shared_ptr<AotKernel> row_kernel,
                                                     std::vector<nb::object> row_tables,
                                                     std::shared_ptr<AotKernel> col_kernel,
                                                     std::vector<nb::object> col_tables,
                                                     nb::object twiddle,
                                                     nb::object stage1)
    : length(length),
      n1(n1),
      n2(n2),
      row_kernel(std::move(row_kernel)),
      row_tables(std::move(row_tables)),
      col_kernel(std::move(col_kernel)),
      col_tables(std::move(col_tables)),
      twiddle(std::move(twiddle)),
      stage1(std::move(stage1)) {}

nb::object CompiledFourStepFusedNode::execute(const nb::object &input,
                                              const ExecutionContext &context) const {
    bool root_node = length == context.request.requested_n;
    int64_t batch = root_node ? context.request.batch : tensor_numel(input) / length;
    nb::object x_contig = (root_node && request_has_flat_batch_shape(context.request))
                              ? input
                              : input.attr("reshape")(nb::make_tuple(batch, length));

    launch_row(context.stream, x_contig, stage1, batch);
    nb::object out = empty_complex64_tensor(context.request, nb::make_tuple(batch, length));
    launch_col(context.stream, stage1, out, batch);
    return out;
}

void CompiledFourStepFusedNode::launch_row(CUstream stream,
                                           const nb::object &src,
                                           const nb::object &dst,
                                           int64_t batch) const {
    std::vector<AotKernelArg> args;
    args.reserve(3 + row_tables.size());
    args.push_back(AotKernelArg::device(tensor_data_ptr(src)));
    args.push_back(AotKernelArg::device(tensor_data_ptr(dst)));
    for (const nb::object &table : row_tables) {
        args.push_back(AotKernelArg::device(tensor_data_ptr(table)));
    }
    args.push_back(AotKernelArg::i32(static_cast<int32_t>(batch)));

    row_kernel->launch(stream, args, n2, batch, 1);
}

void CompiledFourStepFusedNode::launch_col(CUstream stream,
                                           const nb::object &src,
                                           const nb::object &dst,
                                           int64_t batch) const {
    std::vector<AotKernelArg> args;
    args.reserve(4 + col_tables.size());
    args.push_back(AotKernelArg::device(tensor_data_ptr(src)));
    args.push_back(AotKernelArg::device(tensor_data_ptr(twiddle)));
    args.push_back(AotKernelArg::device(tensor_data_ptr(dst)));
    for (const nb::object &table : col_tables) {
        args.push_back(AotKernelArg::device(tensor_data_ptr(table)));
    }
    args.push_back(AotKernelArg::i32(static_cast<int32_t>(batch)));

    col_kernel->launch(stream, args, ceil_div(n1, four_step_col_inner_pack_for(n1, n2)), batch, 1);
}

CompiledBluesteinNode::CompiledBluesteinNode(int64_t length,
                                             int64_t conv_length,
                                             std::shared_ptr<CompiledNode> fft,
                                             std::shared_ptr<AotKernel> prepare_kernel,
                                             std::shared_ptr<AotKernel> pointwise_kernel,
                                             std::shared_ptr<AotKernel> finalize_kernel,
                                             nb::object chirp,
                                             nb::object b_time,
                                             nb::object a_buf,
                                             nb::object work_buf,
                                             nb::object b_fft_buf)
    : length(length),
      conv_length(conv_length),
      fft(std::move(fft)),
      prepare_kernel(std::move(prepare_kernel)),
      pointwise_kernel(std::move(pointwise_kernel)),
      finalize_kernel(std::move(finalize_kernel)),
      chirp(std::move(chirp)),
      b_time(std::move(b_time)),
      a_buf(std::move(a_buf)),
      work_buf(std::move(work_buf)),
      b_fft_buf(std::move(b_fft_buf)) {}

void CompiledBluesteinNode::ensure_b_fft(const ExecutionContext &context) const {
    std::lock_guard<std::mutex> lock(b_fft_mutex);
    if (b_fft_ready) {
        return;
    }
    b_fft_buf = fft->execute(b_time, context);
    b_fft_ready = true;
}

nb::object CompiledBluesteinNode::execute(const nb::object &input, const ExecutionContext &context) const {
    ensure_b_fft(context);

    bool root_node = length == context.request.requested_n;
    int64_t batch = root_node ? context.request.batch : tensor_numel(input) / length;
    nb::object x_contig = (root_node && request_has_flat_batch_shape(context.request))
                              ? input
                              : input.attr("reshape")(nb::make_tuple(batch, length));

    std::vector<AotKernelArg> prepare_args = {
        AotKernelArg::device(tensor_data_ptr(x_contig)),
        AotKernelArg::device(tensor_data_ptr(chirp)),
        AotKernelArg::device(tensor_data_ptr(a_buf)),
        AotKernelArg::i64(length),
        AotKernelArg::i64(conv_length),
        AotKernelArg::i32(static_cast<int32_t>(batch)),
    };
    prepare_kernel->launch(context.stream, prepare_args, ceil_div(conv_length, 256), batch, 1);

    nb::object a_fft = fft->execute(a_buf, context);

    std::vector<AotKernelArg> pointwise_args = {
        AotKernelArg::device(tensor_data_ptr(a_fft)),
        AotKernelArg::device(tensor_data_ptr(b_fft_buf)),
        AotKernelArg::device(tensor_data_ptr(work_buf)),
        AotKernelArg::i64(conv_length),
        AotKernelArg::i32(static_cast<int32_t>(batch)),
    };
    pointwise_kernel->launch(context.stream, pointwise_args, ceil_div(conv_length, 256), batch, 1);

    nb::object conv = fft->execute(work_buf, context);
    nb::object out = empty_complex64_tensor(context.request, nb::make_tuple(batch, length));
    std::vector<AotKernelArg> finalize_args = {
        AotKernelArg::device(tensor_data_ptr(conv)),
        AotKernelArg::device(tensor_data_ptr(chirp)),
        AotKernelArg::device(tensor_data_ptr(out)),
        AotKernelArg::i64(length),
        AotKernelArg::i64(conv_length),
        AotKernelArg::i32(static_cast<int32_t>(batch)),
    };
    finalize_kernel->launch(context.stream, finalize_args, ceil_div(length, 256), batch, 1);
    return out;
}

nb::object ExecutablePlan::execute(nb::object input) const {
    if (backend == ExecutionBackend::TorchFFT) {
        nb::module_ torch = nb::module_::import_("torch");
        const char *op_name = request.direction == "inverse" ? "ifft" : "fft";
        return torch.attr("fft").attr(op_name)(input, "n"_a = request.requested_n,
                                               "dim"_a = request.normalized_dim,
                                               "norm"_a = request.norm);
    }

    nb::object exec_input = input;
    if (request.input_dtype == "float32") {
        nb::module_ torch = nb::module_::import_("torch");
        nb::object zeros = torch.attr("zeros_like")(input);
        exec_input = torch.attr("complex")(input, zeros);
        if (request.requires_contiguous_copy) {
            exec_input = exec_input.attr("contiguous")();
        }
    } else if (request.requires_contiguous_copy) {
        exec_input = exec_input.attr("contiguous")();
    }
    ExecutionContext context{request, current_cuda_stream(request)};
    nb::object result = compiled_root->execute(exec_input, context);
    if (!request_has_flat_batch_shape(request)) {
        result = result.attr("reshape")(nb::cast(request.input_shape));
    }

    double scale = normalization_scale(request);
    if (scale != 1.0) {
        return result.attr("mul")(scale);
    }
    return result;
}

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
