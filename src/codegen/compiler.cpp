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
    std::shared_ptr<RuntimeKernel> kernel = compile_kernel(key);
    return std::make_shared<CompiledRawLeafNode>(
        leaf.length, std::move(kernel), build_raw_leaf_tables(leaf, request));
}

std::shared_ptr<RuntimeKernel> TritonCompiler::compile_four_step_row_kernel(const LeafPlanNode &leaf,
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
    return compile_kernel(key);
}

std::shared_ptr<RuntimeKernel> TritonCompiler::compile_four_step_col_kernel(const LeafPlanNode &leaf,
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
    return compile_kernel(key);
}

std::shared_ptr<RuntimeKernel> TritonCompiler::compile_bluestein_prepare_kernel(const FFTRequest &request,
                                                                            int64_t n,
                                                                            int64_t m) {
    std::string target = triton_target_for_request(request);
    KernelKey key = KernelKey::bluestein_prepare(target, n, m);
    return compile_kernel(key);
}

std::shared_ptr<RuntimeKernel> TritonCompiler::compile_bluestein_pointwise_kernel(const FFTRequest &request,
                                                                              int64_t n,
                                                                              int64_t m) {
    std::string target = triton_target_for_request(request);
    KernelKey key = KernelKey::bluestein_pointwise(target, n, m);
    return compile_kernel(key);
}

std::shared_ptr<RuntimeKernel> TritonCompiler::compile_bluestein_finalize_kernel(const FFTRequest &request,
                                                                             int64_t n,
                                                                             int64_t m) {
    std::string target = triton_target_for_request(request);
    KernelKey key = KernelKey::bluestein_finalize(target, n, m);
    return compile_kernel(key);
}

}  // namespace flagfft
