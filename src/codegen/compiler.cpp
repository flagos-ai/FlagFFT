#include "flagfft/core.hpp"

namespace flagfft {

std::shared_ptr<CompiledNode> TritonCompiler::compile_node(const PlanNodePtr &node,
                                                           const FFTRequest &request,
                                                           int64_t batch) {
    if (auto leaf = std::dynamic_pointer_cast<LeafPlanNode>(node)) {
        return compile_leaf(*leaf, request);
    }
    if (auto four_step = std::dynamic_pointer_cast<FourStepPlanNode>(node)) {
        auto row_leaf = std::dynamic_pointer_cast<LeafPlanNode>(four_step->row_plan);
        auto col_leaf = std::dynamic_pointer_cast<LeafPlanNode>(four_step->col_plan);
        if (row_leaf != nullptr && col_leaf != nullptr) {
            nb::object twiddle =
                build_four_step_twiddle_tensor(request, four_step->n1, four_step->n2);
            nb::object stage1 =
                empty_complex64_tensor(request, nb::make_tuple(batch, four_step->n2, four_step->n1));
            return std::make_shared<CompiledFourStepFusedNode>(
                four_step->length,
                four_step->n1,
                four_step->n2,
                compile_four_step_row_kernel(*row_leaf, request, four_step->n1, four_step->n2),
                build_leaf_tables(*row_leaf, request),
                compile_four_step_col_kernel(*col_leaf, request, four_step->n1, four_step->n2),
                build_leaf_tables(*col_leaf, request),
                std::move(twiddle),
                std::move(stage1));
        }
        std::shared_ptr<CompiledNode> row =
            compile_node(four_step->row_plan, request, batch * four_step->n2);
        std::shared_ptr<CompiledNode> col =
            compile_node(four_step->col_plan, request, batch * four_step->n1);
        nb::object twiddle = build_four_step_twiddle_tensor(request, four_step->n1, four_step->n2);
        nb::object stage0 =
            empty_complex64_tensor(request, nb::make_tuple(batch, four_step->n2, four_step->n1));
        nb::object stage2 =
            empty_complex64_tensor(request, nb::make_tuple(batch, four_step->n1, four_step->n2));
        return std::make_shared<CompiledFourStepNode>(
            four_step->length, four_step->n1, four_step->n2, std::move(row), std::move(col),
            compile_transpose_kernel(request), compile_twiddle_transpose_kernel(request),
            std::move(twiddle), std::move(stage0), std::move(stage2));
    }
    if (auto bluestein = std::dynamic_pointer_cast<BluesteinPlanNode>(node)) {
        FFTRequest child_request = forward_child_request(request);
        std::shared_ptr<CompiledNode> fft = compile_node(bluestein->fft_plan, child_request, batch);
        nb::object chirp = build_bluestein_chirp_tensor(
            request, bluestein->length, request.direction == "inverse");
        nb::object b_time = build_bluestein_b_tensor(request, bluestein->length, bluestein->conv_length);
        nb::object a_buf =
            empty_complex64_tensor(request, nb::make_tuple(batch, bluestein->conv_length));
        nb::object work_buf =
            empty_complex64_tensor(request, nb::make_tuple(batch, bluestein->conv_length));
        nb::object b_fft_buf =
            empty_complex64_tensor(request, nb::make_tuple(1, bluestein->conv_length));
        return std::make_shared<CompiledBluesteinNode>(
            bluestein->length,
            bluestein->conv_length,
            std::move(fft),
            compile_bluestein_prepare_kernel(request, bluestein->length, bluestein->conv_length),
            compile_bluestein_pointwise_kernel(request, bluestein->length, bluestein->conv_length),
            compile_bluestein_finalize_kernel(request, bluestein->length, bluestein->conv_length),
            std::move(chirp),
            std::move(b_time),
            std::move(a_buf),
            std::move(work_buf),
            std::move(b_fft_buf));
    }
    std::string op_name = request.direction == "inverse" ? "ifft" : "fft";
    raise_python(PyExc_NotImplementedError,
                 "flagfft." + op_name +
                     " C++ AOT backend does not support plan node kind: " +
                     plan_node_kind_name(node->kind));
}

std::shared_ptr<CompiledRawNode> TritonCompiler::compile_raw_node(const PlanNodePtr &node,
                                                                  const FFTRequest &request,
                                                                  int64_t batch) {
    if (auto leaf = std::dynamic_pointer_cast<LeafPlanNode>(node)) {
        return compile_raw_leaf(*leaf, request);
    }
    if (auto four_step = std::dynamic_pointer_cast<FourStepPlanNode>(node)) {
        auto row_leaf = std::dynamic_pointer_cast<LeafPlanNode>(four_step->row_plan);
        auto col_leaf = std::dynamic_pointer_cast<LeafPlanNode>(four_step->col_plan);
        if (row_leaf == nullptr || col_leaf == nullptr) {
            throw std::runtime_error("raw C API currently supports only fused leaf/leaf four-step plans");
        }
        DeviceAllocation twiddle = build_raw_four_step_twiddle(request, four_step->n1, four_step->n2);
        DeviceAllocation stage1 = allocate_device_bytes(
            static_cast<std::size_t>(batch * four_step->length * 2 * sizeof(float)));
        return std::make_shared<CompiledRawFourStepFusedNode>(
            four_step->length,
            four_step->n1,
            four_step->n2,
            compile_four_step_row_kernel(*row_leaf, request, four_step->n1, four_step->n2),
            build_raw_leaf_tables(*row_leaf, request),
            compile_four_step_col_kernel(*col_leaf, request, four_step->n1, four_step->n2),
            build_raw_leaf_tables(*col_leaf, request),
            std::move(twiddle),
            std::move(stage1));
    }
    if (auto bluestein = std::dynamic_pointer_cast<BluesteinPlanNode>(node)) {
        FFTRequest child_request = forward_child_request(request);
        std::shared_ptr<CompiledRawNode> fft =
            compile_raw_node(bluestein->fft_plan, child_request, batch);
        DeviceAllocation chirp = build_raw_bluestein_chirp(
            request, bluestein->length, request.direction == "inverse");
        DeviceAllocation b_time =
            build_raw_bluestein_b(request, bluestein->length, bluestein->conv_length);
        DeviceAllocation a_buf = allocate_device_bytes(
            static_cast<std::size_t>(batch * bluestein->conv_length * 2 * sizeof(float)));
        DeviceAllocation work_buf = allocate_device_bytes(
            static_cast<std::size_t>(batch * bluestein->conv_length * 2 * sizeof(float)));
        DeviceAllocation b_fft_buf = allocate_device_bytes(
            static_cast<std::size_t>(bluestein->conv_length * 2 * sizeof(float)));
        return std::make_shared<CompiledRawBluesteinNode>(
            bluestein->length,
            bluestein->conv_length,
            std::move(fft),
            compile_bluestein_prepare_kernel(request, bluestein->length, bluestein->conv_length),
            compile_bluestein_pointwise_kernel(request, bluestein->length, bluestein->conv_length),
            compile_bluestein_finalize_kernel(request, bluestein->length, bluestein->conv_length),
            std::move(chirp),
            std::move(b_time),
            std::move(a_buf),
            std::move(work_buf),
            std::move(b_fft_buf));
    }
    throw std::runtime_error("raw C API does not support plan node kind: " +
                             plan_node_kind_name(node->kind));
}

std::shared_ptr<CompiledNode> TritonCompiler::compile_leaf(const LeafPlanNode &leaf, const FFTRequest &request) {
    std::string target = triton_target_for_request(request);
    KernelKey key = KernelKey::leaf(target,
                                    request.direction,
                                    leaf.length,
                                    leaf.factors,
                                    leaf.lanes,
                                    leaf.num_warps,
                                    leaf.generic_radices,
                                    leaf.smem_size);
    std::ostringstream command;
    command << shell_quote(python_executable())
            << " " << triton_aot_entrypoint()
            << " --kernel leaf"
            << " --length " << leaf.length
            << " --factors " << shell_quote(join_ints(leaf.factors))
            << " --lanes " << leaf.lanes
            << " --num-warps " << leaf.num_warps
            << " --generic-radices " << shell_quote(join_ints(leaf.generic_radices))
            << " --smem-size " << leaf.smem_size
            << " --direction " << shell_quote(request.direction)
            << " --target " << shell_quote(target)
            << " --out-dir " << shell_quote(out_dir().string());

    std::shared_ptr<AotKernel> kernel = compile_kernel(key, command.str());

    return std::make_shared<CompiledLeafNode>(leaf.length, std::move(kernel),
                                              build_leaf_tables(leaf, request));
}

std::shared_ptr<CompiledRawNode> TritonCompiler::compile_raw_leaf(const LeafPlanNode &leaf,
                                                                  const FFTRequest &request) {
    std::string target = triton_target_for_request(request);
    KernelKey key = KernelKey::leaf(target,
                                    request.direction,
                                    leaf.length,
                                    leaf.factors,
                                    leaf.lanes,
                                    leaf.num_warps,
                                    leaf.generic_radices,
                                    leaf.smem_size);
    std::ostringstream command;
    command << shell_quote(python_executable())
            << " " << triton_aot_entrypoint()
            << " --kernel leaf"
            << " --length " << leaf.length
            << " --factors " << shell_quote(join_ints(leaf.factors))
            << " --lanes " << leaf.lanes
            << " --num-warps " << leaf.num_warps
            << " --generic-radices " << shell_quote(join_ints(leaf.generic_radices))
            << " --smem-size " << leaf.smem_size
            << " --direction " << shell_quote(request.direction)
            << " --target " << shell_quote(target)
            << " --out-dir " << shell_quote(out_dir().string());

    std::shared_ptr<AotKernel> kernel = compile_kernel(key, command.str());
    return std::make_shared<CompiledRawLeafNode>(
        leaf.length, std::move(kernel), build_raw_leaf_tables(leaf, request));
}

std::shared_ptr<AotKernel> TritonCompiler::compile_four_step_row_kernel(const LeafPlanNode &leaf,
                                                                        const FFTRequest &request,
                                                                        int64_t n1,
                                                                        int64_t n2) {
    std::string target = triton_target_for_request(request);
    KernelKey key = KernelKey::four_step_row(target,
                                             request.direction,
                                             n1,
                                             n2,
                                             leaf.length,
                                             leaf.factors,
                                             leaf.lanes,
                                             leaf.num_warps,
                                             leaf.generic_radices,
                                             leaf.smem_size);
    std::ostringstream command;
    command << shell_quote(python_executable())
            << " " << triton_aot_entrypoint()
            << " --kernel four_step_row"
            << " --length " << leaf.length
            << " --factors " << shell_quote(join_ints(leaf.factors))
            << " --lanes " << leaf.lanes
            << " --num-warps " << leaf.num_warps
            << " --generic-radices " << shell_quote(join_ints(leaf.generic_radices))
            << " --smem-size " << leaf.smem_size
            << " --four-step-n1 " << n1
            << " --four-step-n2 " << n2
            << " --direction " << shell_quote(request.direction)
            << " --target " << shell_quote(target)
            << " --out-dir " << shell_quote(out_dir().string());
    return compile_kernel(key, command.str());
}

std::shared_ptr<AotKernel> TritonCompiler::compile_four_step_col_kernel(const LeafPlanNode &leaf,
                                                                        const FFTRequest &request,
                                                                        int64_t n1,
                                                                        int64_t n2) {
    std::string target = triton_target_for_request(request);
    KernelKey key = KernelKey::four_step_col(target,
                                             request.direction,
                                             n1,
                                             n2,
                                             leaf.length,
                                             leaf.factors,
                                             leaf.lanes,
                                             leaf.num_warps,
                                             leaf.generic_radices,
                                             leaf.smem_size);
    std::ostringstream command;
    command << shell_quote(python_executable())
            << " " << triton_aot_entrypoint()
            << " --kernel four_step_col"
            << " --length " << leaf.length
            << " --factors " << shell_quote(join_ints(leaf.factors))
            << " --lanes " << leaf.lanes
            << " --num-warps " << leaf.num_warps
            << " --generic-radices " << shell_quote(join_ints(leaf.generic_radices))
            << " --smem-size " << leaf.smem_size
            << " --four-step-n1 " << n1
            << " --four-step-n2 " << n2
            << " --direction " << shell_quote(request.direction)
            << " --target " << shell_quote(target)
            << " --out-dir " << shell_quote(out_dir().string());
    return compile_kernel(key, command.str());
}

std::shared_ptr<AotKernel> TritonCompiler::compile_transpose_kernel(const FFTRequest &request) {
    if (transpose_kernel != nullptr) {
        return transpose_kernel;
    }
    std::string target = triton_target_for_request(request);
    KernelKey key = KernelKey::transpose(target);
    std::ostringstream command;
    command << shell_quote(python_executable())
            << " " << triton_aot_entrypoint()
            << " --kernel transpose"
            << " --target " << shell_quote(target)
            << " --out-dir " << shell_quote(out_dir().string());
    transpose_kernel = compile_kernel(key, command.str());
    return transpose_kernel;
}

std::shared_ptr<AotKernel> TritonCompiler::compile_twiddle_transpose_kernel(const FFTRequest &request) {
    if (twiddle_transpose_kernel != nullptr) {
        return twiddle_transpose_kernel;
    }
    std::string target = triton_target_for_request(request);
    KernelKey key = KernelKey::twiddle_transpose(target);
    std::ostringstream command;
    command << shell_quote(python_executable())
            << " " << triton_aot_entrypoint()
            << " --kernel twiddle_transpose"
            << " --target " << shell_quote(target)
            << " --out-dir " << shell_quote(out_dir().string());
    twiddle_transpose_kernel = compile_kernel(key, command.str());
    return twiddle_transpose_kernel;
}

std::shared_ptr<AotKernel> TritonCompiler::compile_bluestein_prepare_kernel(const FFTRequest &request,
                                                                            int64_t n,
                                                                            int64_t m) {
    std::string target = triton_target_for_request(request);
    KernelKey key = KernelKey::bluestein_prepare(target, n, m);
    std::ostringstream command;
    command << shell_quote(python_executable())
            << " " << triton_aot_entrypoint()
            << " --kernel bluestein_prepare"
            << " --bluestein-n " << n
            << " --bluestein-m " << m
            << " --target " << shell_quote(target)
            << " --out-dir " << shell_quote(out_dir().string());
    return compile_kernel(key, command.str());
}

std::shared_ptr<AotKernel> TritonCompiler::compile_bluestein_pointwise_kernel(const FFTRequest &request,
                                                                              int64_t n,
                                                                              int64_t m) {
    std::string target = triton_target_for_request(request);
    KernelKey key = KernelKey::bluestein_pointwise(target, n, m);
    std::ostringstream command;
    command << shell_quote(python_executable())
            << " " << triton_aot_entrypoint()
            << " --kernel bluestein_pointwise"
            << " --bluestein-n " << n
            << " --bluestein-m " << m
            << " --target " << shell_quote(target)
            << " --out-dir " << shell_quote(out_dir().string());
    return compile_kernel(key, command.str());
}

std::shared_ptr<AotKernel> TritonCompiler::compile_bluestein_finalize_kernel(const FFTRequest &request,
                                                                             int64_t n,
                                                                             int64_t m) {
    std::string target = triton_target_for_request(request);
    KernelKey key = KernelKey::bluestein_finalize(target, n, m);
    std::ostringstream command;
    command << shell_quote(python_executable())
            << " " << triton_aot_entrypoint()
            << " --kernel bluestein_finalize"
            << " --bluestein-n " << n
            << " --bluestein-m " << m
            << " --target " << shell_quote(target)
            << " --out-dir " << shell_quote(out_dir().string());
    return compile_kernel(key, command.str());
}

}  // namespace flagfft
