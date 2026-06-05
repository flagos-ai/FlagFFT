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
      DeviceAllocation stage1 =
          adaptor::Memory(static_cast<std::size_t>(batch * four_step->length * element_bytes));
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
    std::shared_ptr<CompiledRawNode> fft = compile_raw_node(bluestein->fft_plan, child_request, batch);
    DeviceAllocation chirp =
        build_raw_bluestein_chirp(request, bluestein->length, request.direction == "inverse");
    DeviceAllocation b_time = build_raw_bluestein_b(request, bluestein->length, bluestein->conv_length);
    const int64_t element_bytes = complex_element_bytes(request.input_dtype);
    DeviceAllocation a_buf =
        adaptor::Memory(static_cast<std::size_t>(batch * bluestein->conv_length * element_bytes));
    DeviceAllocation work_buf =
        adaptor::Memory(static_cast<std::size_t>(batch * bluestein->conv_length * element_bytes));
    DeviceAllocation b_fft_buf =
        adaptor::Memory(static_cast<std::size_t>(bluestein->conv_length * element_bytes));
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
  if (auto rader = std::dynamic_pointer_cast<RaderPlanNode>(node)) {
    FFTRequest child_request = forward_child_request(request);
    std::shared_ptr<CompiledRawNode> fft = compile_raw_node(rader->conv_plan, child_request, batch);
    DeviceAllocation idx = build_raw_rader_idx_table(rader->idx);
    DeviceAllocation b_time = build_raw_rader_conv_kernel(request, rader->prime, rader->idx);
    const int64_t conv_length = rader->prime - 1;
    const int64_t element_bytes = complex_element_bytes(request.input_dtype);
    DeviceAllocation a_buf = adaptor::Memory(static_cast<std::size_t>(batch * conv_length * element_bytes));
    DeviceAllocation work_buf =
        adaptor::Memory(static_cast<std::size_t>(batch * conv_length * element_bytes));
    DeviceAllocation b_fft_buf = adaptor::Memory(static_cast<std::size_t>(conv_length * element_bytes));
    return std::make_shared<CompiledRawRaderNode>(
        rader->prime,
        conv_length,
        std::move(fft),
        compile_rader_prepare_kernel(request, rader->prime, conv_length),
        compile_rader_pointwise_kernel(request, rader->prime, conv_length),
        compile_rader_finalize_kernel(request, rader->prime, conv_length),
        std::move(idx),
        std::move(b_time),
        std::move(a_buf),
        std::move(work_buf),
        std::move(b_fft_buf));
  }
  if (auto two_dim = std::dynamic_pointer_cast<TwoDimPlanNode>(node)) {
    return compile_raw_2d_node(two_dim, request, batch);
  }
  throw std::runtime_error("raw C API does not support plan node kind: " + plan_node_kind_name(node->kind));
}

std::shared_ptr<CompiledRawNode> TritonCompiler::compile_raw_r2c_node(const PlanNodePtr &node,
                                                                      const FFTRequest &request,
                                                                      int64_t batch) {
  const int64_t element_bytes = complex_element_bytes(request.input_dtype);
  const int64_t n = request.requested_n;
  DeviceAllocation complex_input = adaptor::Memory(static_cast<std::size_t>(batch * n * element_bytes));
  DeviceAllocation full_output = adaptor::Memory(static_cast<std::size_t>(batch * n * element_bytes));
  return std::make_shared<CompiledRawR2CNode>(n,
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
  DeviceAllocation full_input = adaptor::Memory(static_cast<std::size_t>(batch * n * element_bytes));
  DeviceAllocation full_output = adaptor::Memory(static_cast<std::size_t>(batch * n * element_bytes));
  return std::make_shared<CompiledRawC2RNode>(n,
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
  return std::make_shared<CompiledRawLeafNode>(leaf.length,
                                               std::move(kernel),
                                               build_raw_leaf_tables(leaf, request));
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

std::shared_ptr<JitKernel> TritonCompiler::compile_rader_prepare_kernel(const FFTRequest &request,
                                                                        int64_t n,
                                                                        int64_t m) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::rader_prepare(target, request.input_dtype, n, m);
  return compile_kernel(key);
}

std::shared_ptr<JitKernel> TritonCompiler::compile_rader_pointwise_kernel(const FFTRequest &request,
                                                                          int64_t n,
                                                                          int64_t m) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::rader_pointwise(target, request.input_dtype, n, m);
  return compile_kernel(key);
}

std::shared_ptr<JitKernel> TritonCompiler::compile_rader_finalize_kernel(const FFTRequest &request,
                                                                         int64_t n,
                                                                         int64_t m) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::rader_finalize(target, request.input_dtype, n, m);
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

std::shared_ptr<JitKernel> TritonCompiler::compile_compact_to_hermitian_full_kernel(const FFTRequest &request,
                                                                                    int64_t n) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::compact_to_hermitian_full(target, request.input_dtype, n);
  return compile_kernel(key);
}

std::shared_ptr<JitKernel> TritonCompiler::compile_complex_to_real_kernel(const FFTRequest &request,
                                                                          int64_t n) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::complex_to_real(target, request.input_dtype, n);
  return compile_kernel(key);
}

std::shared_ptr<JitKernel> TritonCompiler::compile_tiled_transpose_kernel(const FFTRequest &request,
                                                                          int64_t n0,
                                                                          int64_t n1) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::tiled_transpose(target, request.input_dtype, n0, n1);
  return compile_kernel(key);
}

std::shared_ptr<CompiledRaw2DNode> TritonCompiler::compile_raw_2d_node(
    const std::shared_ptr<TwoDimPlanNode> &node, const FFTRequest &request, int64_t batch) {
  const int64_t element_bytes = complex_element_bytes(request.input_dtype);
  const int64_t n0 = node->n0;
  const int64_t n1 = node->n1;

  // Build row FFT request (axis-1, length=n1, batch=batch*n0)
  FFTRequest row_request = request;
  row_request.fft_length = n1;
  row_request.input_shape = {batch * n0, n1};
  row_request.input_strides = {n1, 1};
  row_request.requested_n = n1;
  row_request.batch = batch * n0;

  // Build col FFT request (axis-0, length=n0, batch=batch*n1)
  FFTRequest col_request = request;
  col_request.fft_length = n0;
  col_request.input_shape = {batch * n1, n0};
  col_request.input_strides = {n0, 1};
  col_request.requested_n = n0;
  col_request.batch = batch * n1;

  // Compile row and col FFT nodes
  std::shared_ptr<CompiledRawNode> row_fft = compile_raw_node(node->row_plan, row_request, batch * n0);
  std::shared_ptr<CompiledRawNode> col_fft = compile_raw_node(node->col_plan, col_request, batch * n1);

  // Compile transpose kernels
  auto transpose_fwd = compile_tiled_transpose_kernel(request, n0, n1);
  auto transpose_inv = compile_tiled_transpose_kernel(request, n1, n0);

  // Allocate temporary buffers
  DeviceAllocation temp1 = adaptor::Memory(static_cast<std::size_t>(batch * n0 * n1 * element_bytes));
  DeviceAllocation temp2 = adaptor::Memory(static_cast<std::size_t>(batch * n0 * n1 * element_bytes));

  return std::make_shared<CompiledRaw2DNode>(n0,
                                             n1,
                                             std::move(row_fft),
                                             std::move(col_fft),
                                             std::move(transpose_fwd),
                                             std::move(transpose_inv),
                                             std::move(temp1),
                                             std::move(temp2));
}

std::shared_ptr<CompiledRawNode> TritonCompiler::compile_raw_four_step_generic(const FourStepPlanNode &node,
                                                                               const FFTRequest &request,
                                                                               int64_t batch) {
  const int64_t element_bytes = complex_element_bytes(request.input_dtype);
  const int64_t n = node.length;
  const int64_t n1 = node.n1;
  const int64_t n2 = node.n2;

  std::shared_ptr<CompiledRawNode> row_child = compile_raw_node(node.row_plan, request, batch * n2);
  std::shared_ptr<CompiledRawNode> col_child = compile_raw_node(node.col_plan, request, batch * n1);

  DeviceAllocation twiddle = build_raw_four_step_twiddle(request, n1, n2);
  DeviceAllocation stage1 = adaptor::Memory(static_cast<std::size_t>(batch * n * element_bytes));
  DeviceAllocation stage2 = adaptor::Memory(static_cast<std::size_t>(batch * n * element_bytes));

  auto reshape_in_kernel = compile_reshape_pack_kernel(request, n1, n2);
  auto twiddle_reshape_kernel = compile_twiddle_reshape_pack_kernel(request, n2, n1);
  auto final_pack_kernel = compile_reshape_pack_kernel(request, n1, n2);

  return std::make_shared<CompiledRawFourStepGenericNode>(n,
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
