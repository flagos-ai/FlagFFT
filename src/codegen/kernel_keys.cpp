#include "flagfft/core.hpp"

namespace flagfft {

FFTRequest forward_child_request(const FFTRequest &request) {
    FFTRequest child = request;
    child.direction = "forward";
    child.norm = "backward";
    return child;
}

void append_kernel_keys(const PlanNodePtr &node,
                        const FFTRequest &request,
                        const std::string &target,
                        nb::list &out) {
    if (auto leaf = std::dynamic_pointer_cast<LeafPlanNode>(node)) {
        out.append(kernel_key_to_dict(KernelKey::leaf(target,
                                                      request.direction,
                                                      leaf->length,
                                                      leaf->factors,
                                                      leaf->lanes,
                                                      leaf->num_warps,
                                                      leaf->generic_radices,
                                                      leaf->smem_size)));
        return;
    }
    if (auto four_step = std::dynamic_pointer_cast<FourStepPlanNode>(node)) {
        auto row_leaf = std::dynamic_pointer_cast<LeafPlanNode>(four_step->row_plan);
        auto col_leaf = std::dynamic_pointer_cast<LeafPlanNode>(four_step->col_plan);
        if (row_leaf != nullptr && col_leaf != nullptr) {
            out.append(kernel_key_to_dict(KernelKey::four_step_row(target,
                                                                   request.direction,
                                                                   four_step->n1,
                                                                   four_step->n2,
                                                                   row_leaf->length,
                                                                   row_leaf->factors,
                                                                   row_leaf->lanes,
                                                                   row_leaf->num_warps,
                                                                   row_leaf->generic_radices,
                                                                   row_leaf->smem_size)));
            out.append(kernel_key_to_dict(KernelKey::four_step_col(target,
                                                                   request.direction,
                                                                   four_step->n1,
                                                                   four_step->n2,
                                                                   col_leaf->length,
                                                                   col_leaf->factors,
                                                                   col_leaf->lanes,
                                                                   col_leaf->num_warps,
                                                                   col_leaf->generic_radices,
                                                                   col_leaf->smem_size)));
            return;
        }
        append_kernel_keys(four_step->row_plan, request, target, out);
        append_kernel_keys(four_step->col_plan, request, target, out);
        out.append(kernel_key_to_dict(KernelKey::transpose(target)));
        out.append(kernel_key_to_dict(KernelKey::twiddle_transpose(target)));
        return;
    }
    if (auto bluestein = std::dynamic_pointer_cast<BluesteinPlanNode>(node)) {
        out.append(kernel_key_to_dict(KernelKey::bluestein_prepare(
            target, bluestein->length, bluestein->conv_length)));
        out.append(kernel_key_to_dict(KernelKey::bluestein_pointwise(
            target, bluestein->length, bluestein->conv_length)));
        out.append(kernel_key_to_dict(KernelKey::bluestein_finalize(
            target, bluestein->length, bluestein->conv_length)));
        FFTRequest child_request = forward_child_request(request);
        append_kernel_keys(bluestein->fft_plan, child_request, target, out);
    }
}

std::string triton_target_for_request(const FFTRequest &request) {
    std::string arch = request.device_arch;
    if (arch.rfind("sm_", 0) == 0) {
        arch.erase(0, 3);
    }
    return "cuda:" + arch + ":32";
}

nb::list kernel_keys_for_plan(const PlanNodePtr &node, const FFTRequest &request) {
    nb::list out;
    if (request.output_dtype != "complex64") {
        return out;
    }
    append_kernel_keys(node, request, triton_target_for_request(request), out);
    return out;
}

}  // namespace flagfft
