#include "flagfft/core.hpp"

namespace flagfft {

CompiledLeafNode::CompiledLeafNode(int64_t length,
                                   std::shared_ptr<AotKernel> kernel,
                                   std::vector<nb::object> tables)
    : length(length), kernel(std::move(kernel)), tables(std::move(tables)) {}

nb::object CompiledLeafNode::execute(const nb::object &input, const ExecutionContext &context) const {
    int64_t batch = tensor_numel(input) / length;
    nb::object x_contig = input.attr("contiguous")().attr("reshape")(nb::make_tuple(batch, length));
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

    kernel->launch(context.stream, args, batch, 1, 1);
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
    int64_t batch = tensor_numel(input) / length;
    nb::object x_contig = input.attr("contiguous")().attr("reshape")(nb::make_tuple(batch, length));

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

nb::object ExecutablePlan::execute(nb::object input) const {
    if (backend == ExecutionBackend::TorchFFT) {
        nb::module_ torch = nb::module_::import_("torch");
        return torch.attr("fft").attr("fft")(input, "n"_a = request.requested_n,
                                             "dim"_a = request.normalized_dim,
                                             "norm"_a = request.norm);
    }

    nb::object exec_input = input;
    if (request.input_dtype == "float32") {
        nb::module_ torch = nb::module_::import_("torch");
        nb::object zeros = torch.attr("zeros_like")(input);
        exec_input = torch.attr("complex")(input, zeros);
    }

    exec_input = exec_input.attr("contiguous")();
    ExecutionContext context{request, current_cuda_stream(request)};
    nb::object result = compiled_root->execute(exec_input, context);
    result = result.attr("reshape")(nb::cast(request.input_shape));

    if (request.norm == "forward") {
        return result.attr("mul")(1.0 / static_cast<double>(request.requested_n));
    }
    if (request.norm == "ortho") {
        return result.attr("mul")(1.0 / std::sqrt(static_cast<double>(request.requested_n)));
    }
    return result;
}

std::shared_ptr<ExecutablePlan> PlanCache::get_or_create(const PlanKey &key, const FFTRequest &request) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            ++hits_;
            return it->second;
        }
        ++misses_;
    }

    PlanBuilder builder;
    PlanNodePtr root = builder.build(request.requested_n);
    nb::dict plan_dict = builder.wrap_plan_dict(root, request);

    ExecutionBackend backend = ExecutionBackend::TorchFFT;
    std::shared_ptr<CompiledNode> compiled_root;
    if (request.output_dtype == "complex64") {
        backend = ExecutionBackend::AotCuda;
        TritonCompiler compiler;
        compiled_root = compiler.compile_node(root, request, request.batch);
    }

    auto executable = std::make_shared<ExecutablePlan>(ExecutablePlan{
        key, request, root, plan_dict, backend, std::move(compiled_root)});

    std::lock_guard<std::mutex> lock(mutex_);
    auto [it, inserted] = cache_.emplace(key, executable);
    return inserted ? executable : it->second;
}

void PlanCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
    hits_ = 0;
    misses_ = 0;
}

nb::dict PlanCache::info() const {
    std::lock_guard<std::mutex> lock(mutex_);
    nb::dict out;
    out["size"] = cache_.size();
    out["hits"] = hits_;
    out["misses"] = misses_;
    return out;
}

nb::list PlanCache::keys() const {
    std::lock_guard<std::mutex> lock(mutex_);
    nb::list out;
    for (const auto &entry : cache_) {
        out.append(key_to_dict(entry.first));
    }
    return out;
}

PlanCache &plan_cache() {
    static PlanCache *cache = new PlanCache();
    return *cache;
}

std::shared_ptr<ExecutablePlan> resolve_plan(nb::object input,
                                             nb::object n_obj,
                                             int64_t dim,
                                             nb::object norm_obj) {
    FFTRequest request = build_request(input, n_obj, dim, norm_obj);
    PlanKey key = PlanKey::from_request(request);
    validate_request(request);
    return plan_cache().get_or_create(key, request);
}

nb::object fft(nb::object input, nb::object n_obj, int64_t dim, nb::object norm_obj) {
    std::shared_ptr<ExecutablePlan> executable = resolve_plan(input, n_obj, dim, norm_obj);
    return executable->execute(input);
}

nb::dict debug_request(nb::object input, nb::object n_obj, int64_t dim, nb::object norm_obj) {
    return request_to_dict(build_request(input, n_obj, dim, norm_obj));
}

nb::dict debug_plan_key(nb::object input, nb::object n_obj, int64_t dim, nb::object norm_obj) {
    FFTRequest request = build_request(input, n_obj, dim, norm_obj);
    return key_to_dict(PlanKey::from_request(request));
}

nb::dict debug_plan(nb::object input, nb::object n_obj, int64_t dim, nb::object norm_obj) {
    FFTRequest request = build_request(input, n_obj, dim, norm_obj);
    validate_request(request);
    PlanBuilder builder;
    PlanNodePtr root = builder.build(request.requested_n);
    return builder.wrap_plan_dict(root, request);
}

}  // namespace flagfft
