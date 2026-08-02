// Copyright 2026 FlagOS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "flagfft/core.hpp"

namespace flagfft {

std::shared_ptr<CompiledRawNode> TritonCompiler::compile_raw_node(const PlanNodePtr &node,
                                                                  const FFTRequest &request,
                                                                  int64_t batch) {
  if (auto leaf = std::dynamic_pointer_cast<LeafPlanNode>(node)) {
    return compile_raw_leaf(*leaf, request);
  }
  if (auto direct = std::dynamic_pointer_cast<DirectDFTPlanNode>(node)) {
    return compile_raw_direct_dft(*direct, request, batch);
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
    // The generic Bluestein pipeline uses per-batch convolution buffers
    // (a_buf/work_buf plus the child FFT workspace).  For large primes and
    // large batch these can exceed device memory, so the batch is compiled at
    // chunk granularity and executed as a sequence of chunks.  The chunk is
    // sized by a byte budget instead of a fixed count: small convolutions
    // (e.g. 997 -> conv 2048) run in one launch, while 2^20 convolutions keep
    // the previous 32-transform chunks.
    const int64_t bluestein_element_bytes = complex_element_bytes(request.input_dtype);
    const int64_t conv_bytes = bluestein->conv_length * bluestein_element_bytes;
    constexpr int64_t kBluesteinChunkByteBudget = 256 * 1024 * 1024;
    const int64_t chunk_batch = std::min<int64_t>(
        batch,
        std::max<int64_t>(1, kBluesteinChunkByteBudget / std::max<int64_t>(1, conv_bytes)));
    auto leaf = std::dynamic_pointer_cast<LeafPlanNode>(bluestein->fft_plan);
    auto four_step = std::dynamic_pointer_cast<FourStepPlanNode>(bluestein->fft_plan);
    auto row_leaf = four_step ? std::dynamic_pointer_cast<LeafPlanNode>(four_step->row_plan) : nullptr;
    auto col_leaf = four_step ? std::dynamic_pointer_cast<LeafPlanNode>(four_step->col_plan) : nullptr;
    std::shared_ptr<CompiledRawNode> fft =
        compile_raw_node(bluestein->fft_plan, child_request, chunk_batch);
    DeviceAllocation chirp =
        build_raw_bluestein_chirp(request, bluestein->length, request.direction == "inverse");
    DeviceAllocation b_time = build_raw_bluestein_b(request, bluestein->length, bluestein->conv_length);
    const bool use_full_leaf = request.input_dtype == "complex64" && leaf != nullptr;
    const bool use_four_step =
        request.input_dtype == "complex64" && four_step != nullptr && row_leaf != nullptr &&
        col_leaf != nullptr && row_leaf->length < 512 && col_leaf->length < 512;
    const int64_t element_bytes = complex_element_bytes(request.input_dtype);
    DeviceAllocation b_fft_buf =
        adaptor::Memory(static_cast<std::size_t>(bluestein->conv_length * element_bytes));
    if (use_full_leaf) {
      std::vector<int64_t> fused_factors = leaf->factors;
      if (fused_factors.size() >= 3) {
        std::reverse(fused_factors.begin() + 1, fused_factors.end());
      }
      LeafPlanNode fused_leaf(leaf->length,
                              std::move(fused_factors),
                              leaf->remainder,
                              leaf->lanes,
                              leaf->num_warps,
                              leaf->generic_radices,
                              leaf->smem_size);
      return std::make_shared<CompiledRawBluesteinFullLeafNode>(
          bluestein->length,
          bluestein->conv_length,
          std::move(fft),
          compile_leaf_bluestein_kernel(fused_leaf, child_request, bluestein->length),
          build_raw_leaf_tables(fused_leaf, child_request),
          std::move(chirp),
          std::move(b_time),
          std::move(b_fft_buf));
    }
    if (use_four_step) {
      auto make_boundary_leaf = [](const LeafPlanNode &source) {
        std::vector<int64_t> factors = source.factors;
        if (factors.size() == 2 && factors.front() > factors.back()) {
          std::reverse(factors.begin(), factors.end());
        }
        return LeafPlanNode(source.length,
                            std::move(factors),
                            source.remainder,
                            source.lanes,
                            source.num_warps,
                            source.generic_radices,
                            source.smem_size);
      };
      LeafPlanNode boundary_row = make_boundary_leaf(*row_leaf);
      LeafPlanNode boundary_col = make_boundary_leaf(*col_leaf);
      auto compile_boundary_kernel = [&](const LeafPlanNode &boundary_leaf,
                                         KernelKind kind,
                                         bool is_row) {
        KernelKey key =
            is_row ? KernelKey::four_step_row(triton_target_for_request(child_request),
                                              child_request.direction,
                                              child_request.input_dtype,
                                              four_step->n1,
                                              four_step->n2,
                                              boundary_leaf.length,
                                              boundary_leaf.factors,
                                              boundary_leaf.lanes,
                                              boundary_leaf.num_warps,
                                              boundary_leaf.generic_radices,
                                              boundary_leaf.smem_size)
                   : KernelKey::four_step_col(triton_target_for_request(child_request),
                                              child_request.direction,
                                              child_request.input_dtype,
                                              four_step->n1,
                                              four_step->n2,
                                              boundary_leaf.length,
                                              boundary_leaf.factors,
                                              boundary_leaf.lanes,
                                              boundary_leaf.num_warps,
                                              boundary_leaf.generic_radices,
                                              boundary_leaf.smem_size);
        key.kind = kind;
        key.bluestein_n = bluestein->length;
        key.bluestein_m = bluestein->conv_length;
        return compile_kernel(key);
      };

      DeviceAllocation twiddle =
          build_raw_four_step_twiddle(child_request, four_step->n1, four_step->n2);
      DeviceAllocation stage1 =
          adaptor::Memory(static_cast<std::size_t>(batch * bluestein->conv_length * element_bytes));
      DeviceAllocation work_buf =
          adaptor::Memory(static_cast<std::size_t>(batch * bluestein->conv_length * element_bytes));
      return std::make_shared<CompiledRawBluesteinFourStepNode>(
          bluestein->length,
          bluestein->conv_length,
          four_step->n1,
          four_step->n2,
          std::move(fft),
          compile_boundary_kernel(boundary_row, KernelKind::BluesteinFourStepPrepareRow, true),
          compile_four_step_col_kernel(boundary_col, child_request, four_step->n1, four_step->n2),
          compile_boundary_kernel(boundary_row, KernelKind::BluesteinFourStepPointwiseRow, true),
          compile_boundary_kernel(boundary_col, KernelKind::BluesteinFourStepFinishCol, false),
          build_raw_leaf_tables(boundary_row, child_request),
          build_raw_leaf_tables(boundary_col, child_request),
          std::move(twiddle),
          std::move(chirp),
          std::move(b_time),
          std::move(stage1),
          std::move(work_buf),
          std::move(b_fft_buf));
    }
    DeviceAllocation work_buf =
        adaptor::Memory(static_cast<std::size_t>(chunk_batch * bluestein->conv_length * element_bytes));
    DeviceAllocation a_buf =
        adaptor::Memory(static_cast<std::size_t>(chunk_batch * bluestein->conv_length * element_bytes));
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
        std::move(b_fft_buf),
        chunk_batch);
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
    DeviceAllocation input_copy =
        adaptor::Memory(static_cast<std::size_t>(batch * rader->prime * element_bytes));
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
        std::move(b_fft_buf),
        std::move(input_copy));
  }
  if (auto two_dim = std::dynamic_pointer_cast<TwoDimPlanNode>(node)) {
    return compile_raw_2d_node(two_dim, request, batch);
  }
  if (auto three_dim = std::dynamic_pointer_cast<ThreeDimPlanNode>(node)) {
    return compile_raw_3d_node(three_dim, request, batch);
  }
  throw std::runtime_error("raw C API does not support plan node kind: " + plan_node_kind_name(node->kind));
}

std::shared_ptr<CompiledRawNode> TritonCompiler::compile_raw_r2c_node(const PlanNodePtr &node,
                                                                      const FFTRequest &request,
                                                                      int64_t batch) {
  const int64_t element_bytes = complex_element_bytes(request.input_dtype);
  const int64_t n = request.requested_n;
  if (auto four_step = std::dynamic_pointer_cast<FourStepPlanNode>(node)) {
    auto row_leaf = std::dynamic_pointer_cast<LeafPlanNode>(four_step->row_plan);
    auto col_leaf = std::dynamic_pointer_cast<LeafPlanNode>(four_step->col_plan);
    if (row_leaf != nullptr && col_leaf != nullptr) {
      DeviceAllocation twiddle = build_raw_four_step_twiddle(request, four_step->n1, four_step->n2);
      DeviceAllocation stage1 =
          adaptor::Memory(static_cast<std::size_t>(batch * four_step->length * element_bytes));
      return std::make_shared<CompiledRawR2CFourStepRealInHalfOutNode>(
          n,
          four_step->n1,
          four_step->n2,
          compile_four_step_real_row_kernel(*row_leaf, request, four_step->n1, four_step->n2),
          build_raw_leaf_tables(*row_leaf, request),
          compile_four_step_r2c_col_kernel(*col_leaf, request, four_step->n1, four_step->n2),
          build_raw_leaf_tables(*col_leaf, request),
          std::move(twiddle),
          std::move(stage1));
    }
  }
  if (auto leaf = std::dynamic_pointer_cast<LeafPlanNode>(node)) {
    return std::make_shared<CompiledRawR2CLeafNode>(n,
                                                    compile_leaf_r2c_kernel(*leaf, request),
                                                    build_raw_leaf_tables(*leaf, request));
  }
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
  if (auto four_step = std::dynamic_pointer_cast<FourStepPlanNode>(node)) {
    auto row_leaf = std::dynamic_pointer_cast<LeafPlanNode>(four_step->row_plan);
    auto col_leaf = std::dynamic_pointer_cast<LeafPlanNode>(four_step->col_plan);
    if (row_leaf != nullptr && col_leaf != nullptr) {
      DeviceAllocation twiddle = build_raw_four_step_twiddle(request, four_step->n1, four_step->n2);
      DeviceAllocation stage1 =
          adaptor::Memory(static_cast<std::size_t>(batch * four_step->length * element_bytes));
      return std::make_shared<CompiledRawC2RFourStepCompactInRealOutNode>(
          n,
          four_step->n1,
          four_step->n2,
          compile_four_step_hermitian_row_kernel(*row_leaf, request, four_step->n1, four_step->n2),
          build_raw_leaf_tables(*row_leaf, request),
          compile_four_step_c2r_col_kernel(*col_leaf, request, four_step->n1, four_step->n2),
          build_raw_leaf_tables(*col_leaf, request),
          std::move(twiddle),
          std::move(stage1));
    }
  }
  if (auto leaf = std::dynamic_pointer_cast<LeafPlanNode>(node)) {
    return std::make_shared<CompiledRawC2RLeafNode>(n,
                                                    compile_leaf_c2r_kernel(*leaf, request),
                                                    build_raw_leaf_tables(*leaf, request));
  }
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

std::shared_ptr<CompiledRawNode> TritonCompiler::compile_raw_direct_dft(const DirectDFTPlanNode &node,
                                                                        const FFTRequest &request,
                                                                        int64_t batch) {
  const int64_t element_bytes = complex_element_bytes(request.input_dtype);
  DeviceAllocation input_copy =
      adaptor::Memory(static_cast<std::size_t>(batch * node.length * element_bytes));
  return std::make_shared<CompiledRawDirectDftNode>(node.length,
                                                    compile_direct_dft_kernel(request, node.length),
                                                    build_raw_direct_dft_tables(node.length, request),
                                                    std::move(input_copy));
}

std::shared_ptr<JitKernel> TritonCompiler::compile_leaf_r2c_kernel(const LeafPlanNode &leaf,
                                                                   const FFTRequest &request) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::leaf_r2c(target,
                                      request.direction,
                                      request.input_dtype,
                                      leaf.length,
                                      leaf.factors,
                                      leaf.lanes,
                                      leaf.num_warps,
                                      leaf.generic_radices,
                                      leaf.smem_size);
  return compile_kernel(key);
}

std::shared_ptr<JitKernel> TritonCompiler::compile_leaf_c2r_kernel(const LeafPlanNode &leaf,
                                                                   const FFTRequest &request) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::leaf_c2r(target,
                                      request.direction,
                                      request.input_dtype,
                                      leaf.length,
                                      leaf.factors,
                                      leaf.lanes,
                                      leaf.num_warps,
                                      leaf.generic_radices,
                                      leaf.smem_size);
  return compile_kernel(key);
}

std::shared_ptr<JitKernel> TritonCompiler::compile_direct_dft_kernel(const FFTRequest &request, int64_t n) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::direct_dft(target, request.direction, request.input_dtype, n);
  return compile_kernel(key);
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

std::shared_ptr<JitKernel> TritonCompiler::compile_four_step_real_row_kernel(const LeafPlanNode &leaf,
                                                                             const FFTRequest &request,
                                                                             int64_t n1,
                                                                             int64_t n2) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::four_step_real_row(target,
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

std::shared_ptr<JitKernel> TritonCompiler::compile_four_step_hermitian_row_kernel(const LeafPlanNode &leaf,
                                                                                  const FFTRequest &request,
                                                                                  int64_t n1,
                                                                                  int64_t n2) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::four_step_hermitian_row(target,
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

std::shared_ptr<JitKernel> TritonCompiler::compile_four_step_r2c_col_kernel(const LeafPlanNode &leaf,
                                                                            const FFTRequest &request,
                                                                            int64_t n1,
                                                                            int64_t n2) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::four_step_r2c_col(target,
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

std::shared_ptr<JitKernel> TritonCompiler::compile_four_step_c2r_col_kernel(const LeafPlanNode &leaf,
                                                                            const FFTRequest &request,
                                                                            int64_t n1,
                                                                            int64_t n2) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::four_step_c2r_col(target,
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

std::shared_ptr<JitKernel> TritonCompiler::compile_leaf_bluestein_kernel(const LeafPlanNode &leaf,
                                                                         const FFTRequest &request,
                                                                         int64_t n) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::leaf_bluestein(target,
                                            request.direction,
                                            request.input_dtype,
                                            n,
                                            leaf.length,
                                            leaf.factors,
                                            leaf.lanes,
                                            leaf.num_warps,
                                            leaf.generic_radices,
                                            leaf.smem_size);
  return compile_kernel(key);
}

std::shared_ptr<JitKernel> TritonCompiler::compile_leaf_bluestein_prepare_kernel(
    const LeafPlanNode &leaf,
    const FFTRequest &request,
    int64_t n) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::leaf_bluestein_prepare(target,
                                                    request.direction,
                                                    request.input_dtype,
                                                    n,
                                                    leaf.length,
                                                    leaf.factors,
                                                    leaf.lanes,
                                                    leaf.num_warps,
                                                    leaf.generic_radices,
                                                    leaf.smem_size);
  return compile_kernel(key);
}

std::shared_ptr<JitKernel> TritonCompiler::compile_leaf_bluestein_finish_kernel(
    const LeafPlanNode &leaf,
    const FFTRequest &request,
    int64_t n) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::leaf_bluestein_finish(target,
                                                   request.direction,
                                                   request.input_dtype,
                                                   n,
                                                   leaf.length,
                                                   leaf.factors,
                                                   leaf.lanes,
                                                   leaf.num_warps,
                                                   leaf.generic_radices,
                                                   leaf.smem_size);
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

std::shared_ptr<JitKernel> TritonCompiler::compile_transpose3d_kernel(
    const FFTRequest &request, int64_t n0, int64_t n1, int64_t n2, const std::string &order) {
  std::string target = triton_target_for_request(request);
  KernelKey key = KernelKey::transpose3d(target, request.input_dtype, n0, n1, n2, order);
  return compile_kernel(key);
}

std::shared_ptr<CompiledRaw3DNode> TritonCompiler::compile_raw_3d_node(
    const std::shared_ptr<ThreeDimPlanNode> &node, const FFTRequest &request, int64_t batch) {
  const int64_t element_bytes = complex_element_bytes(request.input_dtype);
  const int64_t n0 = node->n0;
  const int64_t n1 = node->n1;
  const int64_t n2 = node->n2;

  // Build per-axis C2C requests.  Each axis is processed as a batch of
  // contiguous rows after the corresponding axis permutation.
  FFTRequest n2_request = request;
  n2_request.fft_length = n2;
  n2_request.input_shape = {batch * n0 * n1, n2};
  n2_request.input_strides = {n2, 1};
  n2_request.requested_n = n2;
  n2_request.batch = batch * n0 * n1;

  FFTRequest n1_request = request;
  n1_request.fft_length = n1;
  n1_request.input_shape = {batch * n0 * n2, n1};
  n1_request.input_strides = {n1, 1};
  n1_request.requested_n = n1;
  n1_request.batch = batch * n0 * n2;

  FFTRequest n0_request = request;
  n0_request.fft_length = n0;
  n0_request.input_shape = {batch * n1 * n2, n0};
  n0_request.input_strides = {n0, 1};
  n0_request.requested_n = n0;
  n0_request.batch = batch * n1 * n2;

  std::shared_ptr<CompiledRawNode> n2_fft = compile_raw_node(node->n2_plan, n2_request, batch * n0 * n1);
  std::shared_ptr<CompiledRawNode> n1_fft = compile_raw_node(node->n1_plan, n1_request, batch * n0 * n2);
  std::shared_ptr<CompiledRawNode> n0_fft = compile_raw_node(node->n0_plan, n0_request, batch * n1 * n2);

  // Forward: (n0,n1,n2) -021-> (n0,n2,n1) -210-> (n1,n2,n0) -201-> (n0,n1,n2).
  // Inverse: (n0,n1,n2) -120-> (n1,n2,n0) -210-> (n0,n2,n1) -021-> (n0,n1,n2).
  auto perm_021_fwd = compile_transpose3d_kernel(request, n0, n1, n2, "021");
  auto perm_210_fwd = compile_transpose3d_kernel(request, n0, n2, n1, "210");
  auto perm_201_fwd = compile_transpose3d_kernel(request, n1, n2, n0, "201");
  auto perm_120_inv = compile_transpose3d_kernel(request, n0, n1, n2, "120");
  auto perm_210_inv = compile_transpose3d_kernel(request, n1, n2, n0, "210");
  auto perm_021_inv = compile_transpose3d_kernel(request, n0, n2, n1, "021");

  DeviceAllocation temp1 = adaptor::Memory(static_cast<std::size_t>(batch * n0 * n1 * n2 * element_bytes));
  DeviceAllocation temp2 = adaptor::Memory(static_cast<std::size_t>(batch * n0 * n1 * n2 * element_bytes));

  return std::make_shared<CompiledRaw3DNode>(n0,
                                             n1,
                                             n2,
                                             std::move(n2_fft),
                                             std::move(n1_fft),
                                             std::move(n0_fft),
                                             std::move(perm_021_fwd),
                                             std::move(perm_210_fwd),
                                             std::move(perm_201_fwd),
                                             std::move(perm_120_inv),
                                             std::move(perm_210_inv),
                                             std::move(perm_021_inv),
                                             std::move(temp1),
                                             std::move(temp2));
}

std::shared_ptr<CompiledRawNode> TritonCompiler::compile_raw_3d_r2c_node(
    const std::shared_ptr<ThreeDimPlanNode> &node, const FFTRequest &request, int64_t batch) {
  const int64_t element_bytes = complex_element_bytes(request.input_dtype);
  const int64_t n0 = node->n0;
  const int64_t n1 = node->n1;
  const int64_t n2 = node->n2;
  const int64_t half = n2 / 2 + 1;

  // The innermost axis n2 runs expand + C2C FFT + half-pack; the remaining
  // two axes are plain C2C after the axis permutations.
  FFTRequest n2_request = request;
  n2_request.fft_length = n2;
  n2_request.input_shape = {batch * n0 * n1, n2};
  n2_request.input_strides = {n2, 1};
  n2_request.requested_n = n2;
  n2_request.batch = batch * n0 * n1;

  FFTRequest n1_request = request;
  n1_request.fft_length = n1;
  n1_request.input_shape = {batch * n0 * half, n1};
  n1_request.input_strides = {n1, 1};
  n1_request.requested_n = n1;
  n1_request.batch = batch * n0 * half;

  FFTRequest n0_request = request;
  n0_request.fft_length = n0;
  n0_request.input_shape = {batch * n1 * half, n0};
  n0_request.input_strides = {n0, 1};
  n0_request.requested_n = n0;
  n0_request.batch = batch * n1 * half;

  auto expand_kernel = compile_real_to_complex_kernel(request, n2);
  std::shared_ptr<CompiledRawNode> n2_fft = compile_raw_node(node->n2_plan, n2_request, batch * n0 * n1);
  auto pack_kernel = compile_r2c_half_pack_kernel(request, n2);
  std::shared_ptr<CompiledRawNode> n1_fft = compile_raw_node(node->n1_plan, n1_request, batch * n0 * half);
  std::shared_ptr<CompiledRawNode> n0_fft = compile_raw_node(node->n0_plan, n0_request, batch * n1 * half);

  // (n0,n1,half) -021-> (n0,half,n1) -210-> (n1,half,n0) -201-> (n0,n1,half).
  auto perm_021 = compile_transpose3d_kernel(request, n0, n1, half, "021");
  auto perm_210 = compile_transpose3d_kernel(request, n0, half, n1, "210");
  auto perm_201 = compile_transpose3d_kernel(request, n1, half, n0, "201");

  DeviceAllocation row_fft_buf =
      adaptor::Memory(static_cast<std::size_t>(batch * n0 * n1 * n2 * element_bytes));
  DeviceAllocation temp1 = adaptor::Memory(static_cast<std::size_t>(batch * n0 * n1 * half * element_bytes));
  DeviceAllocation temp2 = adaptor::Memory(static_cast<std::size_t>(batch * n0 * n1 * half * element_bytes));

  return std::make_shared<CompiledRaw3DR2CNode>(n0,
                                                n1,
                                                n2,
                                                std::move(expand_kernel),
                                                std::move(n2_fft),
                                                std::move(pack_kernel),
                                                std::move(n1_fft),
                                                std::move(n0_fft),
                                                std::move(perm_021),
                                                std::move(perm_210),
                                                std::move(perm_201),
                                                std::move(row_fft_buf),
                                                std::move(temp1),
                                                std::move(temp2));
}

std::shared_ptr<CompiledRawNode> TritonCompiler::compile_raw_3d_c2r_node(
    const std::shared_ptr<ThreeDimPlanNode> &node, const FFTRequest &request, int64_t batch) {
  const int64_t element_bytes = complex_element_bytes(request.input_dtype);
  const int64_t n0 = node->n0;
  const int64_t n1 = node->n1;
  const int64_t n2 = node->n2;
  const int64_t half = n2 / 2 + 1;

  // C2R is the reverse of R2C: permute the half-packed cube, IFFT along n0
  // and n1, expand half -> full Hermitian, IFFT along n2, pack complex -> real.
  FFTRequest n0_request = request;
  n0_request.fft_length = n0;
  n0_request.input_shape = {batch * n1 * half, n0};
  n0_request.input_strides = {n0, 1};
  n0_request.requested_n = n0;
  n0_request.batch = batch * n1 * half;

  FFTRequest n1_request = request;
  n1_request.fft_length = n1;
  n1_request.input_shape = {batch * n0 * half, n1};
  n1_request.input_strides = {n1, 1};
  n1_request.requested_n = n1;
  n1_request.batch = batch * n0 * half;

  FFTRequest n2_request = request;
  n2_request.fft_length = n2;
  n2_request.input_shape = {batch * n0 * n1, n2};
  n2_request.input_strides = {n2, 1};
  n2_request.requested_n = n2;
  n2_request.batch = batch * n0 * n1;

  std::shared_ptr<CompiledRawNode> n0_fft = compile_raw_node(node->n0_plan, n0_request, batch * n1 * half);
  std::shared_ptr<CompiledRawNode> n1_fft = compile_raw_node(node->n1_plan, n1_request, batch * n0 * half);
  auto expand_kernel = compile_compact_to_hermitian_full_kernel(request, n2);
  std::shared_ptr<CompiledRawNode> n2_fft = compile_raw_node(node->n2_plan, n2_request, batch * n0 * n1);
  auto pack_kernel = compile_complex_to_real_kernel(request, n2);

  // (n0,n1,half) -120-> (n1,half,n0) -210-> (n0,half,n1) -021-> (n0,n1,half).
  auto perm_120 = compile_transpose3d_kernel(request, n0, n1, half, "120");
  auto perm_210 = compile_transpose3d_kernel(request, n1, half, n0, "210");
  auto perm_021 = compile_transpose3d_kernel(request, n0, half, n1, "021");

  DeviceAllocation temp1 = adaptor::Memory(static_cast<std::size_t>(batch * n0 * n1 * half * element_bytes));
  DeviceAllocation temp2 = adaptor::Memory(static_cast<std::size_t>(batch * n0 * n1 * half * element_bytes));
  DeviceAllocation full_buf = adaptor::Memory(static_cast<std::size_t>(batch * n0 * n1 * n2 * element_bytes));

  return std::make_shared<CompiledRaw3DC2RNode>(n0,
                                                n1,
                                                n2,
                                                std::move(perm_120),
                                                std::move(perm_210),
                                                std::move(perm_021),
                                                std::move(n0_fft),
                                                std::move(n1_fft),
                                                std::move(expand_kernel),
                                                std::move(n2_fft),
                                                std::move(pack_kernel),
                                                std::move(temp1),
                                                std::move(temp2),
                                                std::move(full_buf));
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

std::shared_ptr<CompiledRawNode> TritonCompiler::compile_raw_2d_r2c_node(
    const std::shared_ptr<TwoDimPlanNode> &node, const FFTRequest &request, int64_t batch) {
  const int64_t element_bytes = complex_element_bytes(request.input_dtype);
  const int64_t n0 = node->n0;
  const int64_t n1 = node->n1;
  const int64_t half_n1 = n1 / 2 + 1;

  // Build row C2C FFT request (axis-1, length=n1, batch=batch*n0)
  FFTRequest row_request = request;
  row_request.fft_length = n1;
  row_request.input_shape = {batch * n0, n1};
  row_request.input_strides = {n1, 1};
  row_request.requested_n = n1;
  row_request.batch = batch * n0;

  // Build col C2C FFT request (axis-0, length=n0, batch=batch*half_n1)
  FFTRequest col_request = request;
  col_request.fft_length = n0;
  col_request.input_shape = {batch * half_n1, n0};
  col_request.input_strides = {n0, 1};
  col_request.requested_n = n0;
  col_request.batch = batch * half_n1;

  // Compile kernels
  auto expand_kernel = compile_real_to_complex_kernel(request, n1);
  std::shared_ptr<CompiledRawNode> row_fft = compile_raw_node(node->row_plan, row_request, batch * n0);
  auto pack_kernel = compile_r2c_half_pack_kernel(request, n1);
  std::shared_ptr<CompiledRawNode> col_fft = compile_raw_node(node->col_plan, col_request, batch * half_n1);

  // R2C transposes: (n0, half_n1) <-> (half_n1, n0)
  auto transpose_fwd = compile_tiled_transpose_kernel(request, n0, half_n1);
  auto transpose_inv = compile_tiled_transpose_kernel(request, half_n1, n0);

  // Allocate buffers
  // row_fft_buf: full complex output from row R2C FFT (batch*n0*n1 complex)
  DeviceAllocation row_fft_buf = adaptor::Memory(static_cast<std::size_t>(batch * n0 * n1 * element_bytes));
  // temp1: transposed data (batch * half_n1 * n0 complex)
  DeviceAllocation temp1 = adaptor::Memory(static_cast<std::size_t>(batch * half_n1 * n0 * element_bytes));
  // temp2: col FFT output (batch * half_n1 * n0 complex)
  DeviceAllocation temp2 = adaptor::Memory(static_cast<std::size_t>(batch * half_n1 * n0 * element_bytes));

  return std::make_shared<CompiledRaw2DR2CNode>(n0,
                                                n1,
                                                std::move(expand_kernel),
                                                std::move(row_fft),
                                                std::move(pack_kernel),
                                                std::move(col_fft),
                                                std::move(transpose_fwd),
                                                std::move(transpose_inv),
                                                std::move(row_fft_buf),
                                                std::move(temp1),
                                                std::move(temp2));
}

std::shared_ptr<CompiledRawNode> TritonCompiler::compile_raw_2d_c2r_node(
    const std::shared_ptr<TwoDimPlanNode> &node, const FFTRequest &request, int64_t batch) {
  const int64_t element_bytes = complex_element_bytes(request.input_dtype);
  const int64_t n0 = node->n0;
  const int64_t n1 = node->n1;
  const int64_t half_n1 = n1 / 2 + 1;

  // C2R is the reverse of R2C:
  // 1. Transpose (n0, half_n1) -> (half_n1, n0)
  // 2. Col IFFT along n0 (batch * half_n1)
  // 3. Transpose back (half_n1, n0) -> (n0, half_n1)
  // 4. Expand half-packed -> full Hermitian (n0, half_n1) -> (n0, n1)
  // 5. Row IFFT along n1 (batch * n0)
  // 6. Pack complex -> real

  // Build col C2C IFFT request (length=n0, batch=batch*half_n1)
  FFTRequest col_request = request;
  col_request.fft_length = n0;
  col_request.input_shape = {batch * half_n1, n0};
  col_request.input_strides = {n0, 1};
  col_request.requested_n = n0;
  col_request.batch = batch * half_n1;

  // Build row C2C IFFT request (length=n1, batch=batch*n0)
  FFTRequest row_request = request;
  row_request.fft_length = n1;
  row_request.input_shape = {batch * n0, n1};
  row_request.input_strides = {n1, 1};
  row_request.requested_n = n1;
  row_request.batch = batch * n0;

  // Compile kernels
  auto expand_kernel = compile_compact_to_hermitian_full_kernel(request, n1);
  std::shared_ptr<CompiledRawNode> col_fft = compile_raw_node(node->col_plan, col_request, batch * half_n1);
  std::shared_ptr<CompiledRawNode> row_fft = compile_raw_node(node->row_plan, row_request, batch * n0);
  auto pack_kernel = compile_complex_to_real_kernel(request, n1);

  // C2R transposes: (n0, half_n1) <-> (half_n1, n0)
  auto transpose_fwd = compile_tiled_transpose_kernel(request, n0, half_n1);
  auto transpose_inv = compile_tiled_transpose_kernel(request, half_n1, n0);

  // Allocate buffers
  // temp1: transposed data (batch * half_n1 * n0 complex)
  DeviceAllocation temp1 = adaptor::Memory(static_cast<std::size_t>(batch * half_n1 * n0 * element_bytes));
  // temp2: col IFFT output (batch * half_n1 * n0 complex)
  DeviceAllocation temp2 = adaptor::Memory(static_cast<std::size_t>(batch * half_n1 * n0 * element_bytes));
  // temp3: expanded full Hermitian (batch * n0 * n1 complex) + row IFFT output
  DeviceAllocation temp3 = adaptor::Memory(static_cast<std::size_t>(batch * n0 * n1 * element_bytes));

  return std::make_shared<CompiledRaw2DC2RNode>(n0,
                                                n1,
                                                std::move(expand_kernel),
                                                std::move(col_fft),
                                                std::move(row_fft),
                                                std::move(transpose_fwd),
                                                std::move(transpose_inv),
                                                std::move(pack_kernel),
                                                std::move(temp1),
                                                std::move(temp2),
                                                std::move(temp3));
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
