#include "flagfft/core.hpp"

namespace flagfft {

std::shared_ptr<CompiledRawNode> TritonCompiler::compile_raw_node(const PlanNodePtr &node,
                                                                  const FFTRequest &request,
                                                                  int64_t batch) {
    if (auto leaf = std::dynamic_pointer_cast<LeafPlanNode>(node)) {
        return compile_raw_leaf(*leaf, request);
    }
    if (auto four_step = std::dynamic_pointer_cast<FourStepPlanNode>(node)) {
        auto row_leaf = std::dynamic_pointer_cast<LeafPlanNode>(four_step->row_plan);
        auto col_leaf = std::dynamic_pointer_cast<LeafPlanNode>(four_step->col_plan);
        const int64_t element_bytes = complex_element_bytes(request.input_dtype);
        if (row_leaf != nullptr && col_leaf != nullptr) {
            DeviceAllocation twiddle = build_raw_four_step_twiddle(request, four_step->n1, four_step->n2);
            DeviceAllocation stage1 = adaptor::Memory(
                static_cast<std::size_t>(batch * four_step->length * element_bytes));
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
        return compile_raw_four_step_generic(*four_step, request, batch);
    }
    if (auto bluestein = std::dynamic_pointer_cast<BluesteinPlanNode>(node)) {
        FFTRequest child_request = forward_child_request(request);
        std::shared_ptr<CompiledRawNode> fft =
            compile_raw_node(bluestein->fft_plan, child_request, batch);
        DeviceAllocation chirp = build_raw_bluestein_chirp(
            request, bluestein->length, request.direction == "inverse");
        DeviceAllocation b_time =
            build_raw_bluestein_b(request, bluestein->length, bluestein->conv_length);
        const int64_t element_bytes = complex_element_bytes(request.input_dtype);
        DeviceAllocation a_buf = adaptor::Memory(
            static_cast<std::size_t>(batch * bluestein->conv_length * element_bytes));
        DeviceAllocation work_buf = adaptor::Memory(
            static_cast<std::size_t>(batch * bluestein->conv_length * element_bytes));
        DeviceAllocation b_fft_buf = adaptor::Memory(
            static_cast<std::size_t>(bluestein->conv_length * element_bytes));
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

std::shared_ptr<CompiledRawNode> TritonCompiler::compile_raw_r2c_node(const PlanNodePtr &node,
                                                                      const FFTRequest &request,
                                                                      int64_t batch) {
    const int64_t element_bytes = complex_element_bytes(request.input_dtype);
    const int64_t n = request.requested_n;
    DeviceAllocation complex_input =
        adaptor::Memory(static_cast<std::size_t>(batch * n * element_bytes));
    DeviceAllocation full_output =
        adaptor::Memory(static_cast<std::size_t>(batch * n * element_bytes));
    return std::make_shared<CompiledRawR2CNode>(
        n,
        compile_real_to_complex_kernel(request, n),
        compile_raw_node(node, request, batch),
        compile_r2c_half_pack_kernel(request, n),
        std::move(complex_input),
        std::move(full_output));
}

std::shared_ptr<CompiledRawNode> TritonCompiler::compile_raw_c2r_node(const PlanNodePtr &node,
                                                                      const FFTRequest &request,
                                                                      int64_t batch) {
    const int64_t element_bytes = complex_element_bytes(request.input_dtype);
    const int64_t n = request.requested_n;
    DeviceAllocation full_input =
        adaptor::Memory(static_cast<std::size_t>(batch * n * element_bytes));
    DeviceAllocation full_output =
        adaptor::Memory(static_cast<std::size_t>(batch * n * element_bytes));
    return std::make_shared<CompiledRawC2RNode>(
        n,
        compile_compact_to_hermitian_full_kernel(request, n),
        compile_raw_node(node, request, batch),
        compile_complex_to_real_kernel(request, n),
        std::move(full_input),
        std::move(full_output));
}

std::shared_ptr<CompiledRawNode> TritonCompiler::compile_raw_leaf(const LeafPlanNode &leaf,
                                                                  const FFTRequest &request) {
    std::string target = triton_target_for_request(request);
    KernelKey key = KernelKey::leaf(target,
                                    request.direction,
                                    request.input_dtype,
                                    leaf.length,
                                    leaf.factors,
                                    leaf.lanes,
                                    leaf.num_warps,
                                    leaf.generic_radices,
                                    leaf.smem_size);
    std::shared_ptr<JitKernel> kernel = compile_kernel(key);
    return std::make_shared<CompiledRawLeafNode>(
        leaf.length, std::move(kernel), build_raw_leaf_tables(leaf, request));
}

std::shared_ptr<JitKernel> TritonCompiler::compile_four_step_row_kernel(const LeafPlanNode &leaf,
                                                                        const FFTRequest &request,
                                                                        int64_t n1,
                                                                        int64_t n2) {
    std::string target = triton_target_for_request(request);
    KernelKey key = KernelKey::four_step_row(target,
                                             request.direction,
                                             request.input_dtype,
                                             n1,
                                             n2,
                                             leaf.length,
                                             leaf.factors,
                                             leaf.lanes,
                                             leaf.num_warps,
                                             leaf.generic_radices,
                                             leaf.smem_size);
    return compile_kernel(key);
}

std::shared_ptr<JitKernel> TritonCompiler::compile_four_step_col_kernel(const LeafPlanNode &leaf,
                                                                        const FFTRequest &request,
                                                                        int64_t n1,
                                                                        int64_t n2) {
    std::string target = triton_target_for_request(request);
    KernelKey key = KernelKey::four_step_col(target,
                                             request.direction,
                                             request.input_dtype,
                                             n1,
                                             n2,
                                             leaf.length,
                                             leaf.factors,
                                             leaf.lanes,
                                             leaf.num_warps,
                                             leaf.generic_radices,
                                             leaf.smem_size);
    return compile_kernel(key);
}

std::shared_ptr<JitKernel> TritonCompiler::compile_bluestein_prepare_kernel(const FFTRequest &request,
                                                                            int64_t n,
                                                                            int64_t m) {
    std::string target = triton_target_for_request(request);
    KernelKey key = KernelKey::bluestein_prepare(target, request.input_dtype, n, m);
    return compile_kernel(key);
}

std::shared_ptr<JitKernel> TritonCompiler::compile_bluestein_pointwise_kernel(const FFTRequest &request,
                                                                              int64_t n,
                                                                              int64_t m) {
    std::string target = triton_target_for_request(request);
    KernelKey key = KernelKey::bluestein_pointwise(target, request.input_dtype, n, m);
    return compile_kernel(key);
}

std::shared_ptr<JitKernel> TritonCompiler::compile_bluestein_finalize_kernel(const FFTRequest &request,
                                                                             int64_t n,
                                                                             int64_t m) {
    std::string target = triton_target_for_request(request);
    KernelKey key = KernelKey::bluestein_finalize(target, request.input_dtype, n, m);
    return compile_kernel(key);
}

std::shared_ptr<JitKernel> TritonCompiler::compile_reshape_pack_kernel(const FFTRequest &request,
                                                                           int64_t n1,
                                                                           int64_t n2) {
    std::string target = triton_target_for_request(request);
    KernelKey key = KernelKey::reshape_pack(target, request.input_dtype, n1, n2);
    return compile_kernel(key);
}

std::shared_ptr<JitKernel> TritonCompiler::compile_twiddle_reshape_pack_kernel(const FFTRequest &request,
                                                                                   int64_t n1,
                                                                                   int64_t n2) {
    std::string target = triton_target_for_request(request);
    KernelKey key = KernelKey::twiddle_reshape_pack(target, request.input_dtype, n1, n2);
    return compile_kernel(key);
}

std::shared_ptr<JitKernel> TritonCompiler::compile_real_to_complex_kernel(const FFTRequest &request,
                                                                              int64_t n) {
    std::string target = triton_target_for_request(request);
    KernelKey key = KernelKey::real_to_complex(target, request.input_dtype, n);
    return compile_kernel(key);
}

std::shared_ptr<JitKernel> TritonCompiler::compile_r2c_half_pack_kernel(const FFTRequest &request,
                                                                           int64_t n) {
    std::string target = triton_target_for_request(request);
    KernelKey key = KernelKey::r2c_half_pack(target, request.input_dtype, n);
    return compile_kernel(key);
}

std::shared_ptr<JitKernel> TritonCompiler::compile_compact_to_hermitian_full_kernel(
    const FFTRequest &request,
    int64_t n) {
    std::string target = triton_target_for_request(request);
    KernelKey key = KernelKey::compact_to_hermitian_full(target, request.input_dtype, n);
    return compile_kernel(key);
}

std::shared_ptr<JitKernel> TritonCompiler::compile_complex_to_real_kernel(
    const FFTRequest &request,
    int64_t n) {
    std::string target = triton_target_for_request(request);
    KernelKey key = KernelKey::complex_to_real(target, request.input_dtype, n);
    return compile_kernel(key);
}

std::shared_ptr<CompiledRawNode> TritonCompiler::compile_raw_four_step_generic(const FourStepPlanNode &node,
                                                                               const FFTRequest &request,
                                                                               int64_t batch) {
    const int64_t element_bytes = complex_element_bytes(request.input_dtype);
    const int64_t n = node.length;
    const int64_t n1 = node.n1;
    const int64_t n2 = node.n2;

    std::shared_ptr<CompiledRawNode> row_child =
        compile_raw_node(node.row_plan, request, batch * n2);
    std::shared_ptr<CompiledRawNode> col_child =
        compile_raw_node(node.col_plan, request, batch * n1);

    DeviceAllocation twiddle = build_raw_four_step_twiddle(request, n1, n2);
    DeviceAllocation stage1 = adaptor::Memory(
        static_cast<std::size_t>(batch * n * element_bytes));
    DeviceAllocation stage2 = adaptor::Memory(
        static_cast<std::size_t>(batch * n * element_bytes));

    auto reshape_in_kernel = compile_reshape_pack_kernel(request, n1, n2);
    auto twiddle_reshape_kernel = compile_twiddle_reshape_pack_kernel(request, n2, n1);
    auto final_pack_kernel = compile_reshape_pack_kernel(request, n1, n2);

    return std::make_shared<CompiledRawFourStepGenericNode>(
        n,
        n1,
        n2,
        std::move(row_child),
        std::move(col_child),
        std::move(reshape_in_kernel),
        std::move(twiddle_reshape_kernel),
        std::move(final_pack_kernel),
        std::move(twiddle),
        std::move(stage1),
        std::move(stage2));
}

}  // namespace flagfft
