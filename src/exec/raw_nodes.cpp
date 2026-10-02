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

#if defined(FLAGFFT_BACKEND_NPU)
#include "adaptor/backend/npu/ascendc_fft64.hpp"
#include "adaptor/backend/npu/ascendc_fft_small.hpp"
#include "adaptor/backend/npu/ascendc_fft128.hpp"
#include "adaptor/backend/npu/ascendc_fft2048.hpp"
#include "adaptor/backend/npu/ascendc_fft256.hpp"
#include "adaptor/backend/npu/ascendc_transpose3d.hpp"
#endif

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <sstream>

namespace flagfft {

CompiledRawGraphNode::CompiledRawGraphNode(std::shared_ptr<CompiledRawNode> inner)
    : inner(std::move(inner)) {}

std::string CompiledRawGraphNode::describe() const {
  return "CompiledRawGraph(inner=" + inner->describe() + ")";
}

flagfftResult CompiledRawGraphNode::execute(adaptor::DevicePtr input,
                                            adaptor::DevicePtr output,
                                            const RawExecutionContext &context) const {
  try {
    if (graph && graph_input == input && graph_output == output && graph_batch == context.batch) {
      graph->launch(context.stream);
      return FLAGFFT_SUCCESS;
    }

    flagfftResult result = inner->execute(input, output, context);
    if (result != FLAGFFT_SUCCESS) return result;
    if (!graph && !graph_failed) {
      try {
        auto captured = std::make_unique<adaptor::CudaGraph>();
        captured->begin_capture(context.stream);
        result = inner->execute(input, output, context);
        captured->end_capture(context.stream);
        if (result != FLAGFFT_SUCCESS) {
          graph_failed = true;
          return result;
        }
        captured->launch(context.stream);
        graph = std::move(captured);
        graph_input = input;
        graph_output = output;
        graph_batch = context.batch;
      } catch (const std::exception &) {
        graph_failed = true;
      }
    }
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] graph execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

namespace {

  std::vector<JitKernelArg> raw_kernel_args(std::initializer_list<adaptor::DevicePtr> ptrs,
                                            const std::vector<DeviceAllocation> &tables,
                                            int64_t batch) {
    std::vector<JitKernelArg> args;
    args.reserve(ptrs.size() + tables.size() + 1);
    for (adaptor::DevicePtr ptr : ptrs) {
      args.push_back(JitKernelArg::device(ptr));
    }
    for (const DeviceAllocation &table : tables) {
      args.push_back(JitKernelArg::device(table.get()));
    }
    args.push_back(JitKernelArg::i32(static_cast<int32_t>(batch)));
    return args;
  }

  bool ix_ct_batch_graph_enabled(const RawExecutionContext &context, int64_t length) {
    const FFTRequest &request = context.request;
    const char *setting = std::getenv("FLAGFFT_IX_CT_BATCH_GRAPH");
    const bool enabled = setting == nullptr || std::string(setting) == "1";
    const bool measured_shape = length == 1024 ||
        (length == 2048 && (request.real_transform_kind == "r2c" ||
                            request.real_transform_kind == "c2r"));
    return enabled && measured_shape && request.device_type == "ix" &&
           request.device_arch == "71" && request.origin_rank <= 1 && request.raw_dim == 1 &&
           request.requested_n == length &&
           context.batch == 64 && request.input_dtype == "complex64" &&
           request.output_dtype == "complex64";
  }

  bool replay_leaf_graph(Raw1DGraphState &state,
                         const RawExecutionContext &context,
                         adaptor::DevicePtr input,
                         adaptor::DevicePtr output,
                         int64_t input_distance,
                         int64_t output_distance) {
    if (!state.graph || state.input != input || state.output != output ||
        state.batch != context.batch || state.input_distance != input_distance ||
        state.output_distance != output_distance) return false;
    state.graph->launch(context.stream);
    return true;
  }

  template <typename Launch>
  void capture_leaf_graph(Raw1DGraphState &state,
                          const RawExecutionContext &context,
                          adaptor::DevicePtr input,
                          adaptor::DevicePtr output,
                          int64_t input_distance,
                          int64_t output_distance,
                          Launch launch) {
    if (state.graph || state.failed) return;
    try {
      auto graph = std::make_unique<adaptor::CudaGraph>();
      graph->begin_capture(context.stream);
      launch();
      graph->end_capture(context.stream);
      state.graph = std::move(graph);
      state.input = input;
      state.output = output;
      state.batch = context.batch;
      state.input_distance = input_distance;
      state.output_distance = output_distance;
    } catch (const std::exception &) {
      state.failed = true;
    }
  }

  std::vector<JitKernelArg> raw_distance_col_kernel_args(std::initializer_list<adaptor::DevicePtr> ptrs,
                                                         const std::vector<DeviceAllocation> &tables,
                                                         int64_t output_distance,
                                                         int64_t batch) {
    std::vector<JitKernelArg> args;
    args.reserve(ptrs.size() + tables.size() + 2);
    for (adaptor::DevicePtr ptr : ptrs) {
      args.push_back(JitKernelArg::device(ptr));
    }
    for (const DeviceAllocation &table : tables) {
      args.push_back(JitKernelArg::device(table.get()));
    }
    args.push_back(JitKernelArg::i64(output_distance));
    args.push_back(JitKernelArg::i32(static_cast<int32_t>(batch)));
    return args;
  }

  // Must match the BLOCK baked into kernels.py:_build_tiled_transpose3d_kernel_source.
  constexpr int64_t kPerm3dBlock = 1024;
  constexpr int64_t kMaxGridY = 65535;

  int64_t block_limit_per_launch() {
    static const int64_t limit = std::max<int64_t>(1, adaptor::max_launch_blocks());
    return limit;
  }

  int64_t grid_rows(const std::shared_ptr<JitKernel> &kernel, int64_t rows) {
    return ceil_div(rows, kernel->rows_per_block);
  }

  // Row-oriented kernels describe their work as `rows` rows of `rows_per_block`
  // rows per block, with grid_x covering the columns or tiles of one row. Split
  // the rows so every launch stays inside both the per-dimension grid cap and
  // the backend's total-block limit.
  template <typename Launch>
  void launch_grid_y_chunks(int64_t grid_x, int64_t rows, int64_t rows_per_block, Launch &&launch) {
    const int64_t max_grid_y = std::max<int64_t>(
        1,
        std::min<int64_t>(kMaxGridY, block_limit_per_launch() / std::max<int64_t>(1, grid_x)));
    const int64_t chunk_span = max_grid_y * std::max<int64_t>(1, rows_per_block);
    for (int64_t offset = 0; offset < rows; offset += chunk_span) {
      const int64_t chunk = std::min(chunk_span, rows - offset);
      launch(offset, chunk);
    }
  }

  void launch_perm3d(const std::shared_ptr<JitKernel> &kernel,
                     adaptor::StreamHandle stream,
                     adaptor::DevicePtr input,
                     adaptor::DevicePtr output,
                     int64_t elements_per_batch,
                     int64_t batch,
                     int64_t element_bytes) {
    const int64_t grid_x =
        kernel->grid_x_override > 0 ? kernel->grid_x_override : ceil_div(elements_per_batch, kPerm3dBlock);
    const int64_t batch_chunk = std::max<int64_t>(1, block_limit_per_launch() / std::max<int64_t>(1, grid_x));
    for (int64_t batch_offset = 0; batch_offset < batch; batch_offset += batch_chunk) {
      const int64_t count = std::min(batch_chunk, batch - batch_offset);
      const int64_t byte_offset = batch_offset * elements_per_batch * element_bytes;
      std::vector<JitKernelArg> args = {
          JitKernelArg::device(input + byte_offset),
          JitKernelArg::device(output + byte_offset),
          JitKernelArg::i32(static_cast<int32_t>(count)),
      };
      kernel->launch(stream, args, grid_x, 1, count);
    }
  }

  flagfftResult launch_perm3d_with_optional_npu_native(
      const std::shared_ptr<JitKernel> &kernel,
      const std::vector<DeviceAllocation> &npu_indices,
      adaptor::StreamHandle stream,
      adaptor::DevicePtr input,
      adaptor::DevicePtr output,
      int64_t n0,
      int64_t n1,
      int64_t n2,
      int32_t axis0,
      int32_t axis1,
      int32_t axis2,
      int64_t batch,
      int64_t element_bytes) {
#if defined(FLAGFFT_BACKEND_NPU)
    if (!npu_indices.empty()) {
      if (npu_indices.size() != 6 || element_bytes != 2 * sizeof(float) ||
          n0 > std::numeric_limits<int32_t>::max() ||
          n1 > std::numeric_limits<int32_t>::max() ||
          n2 > std::numeric_limits<int32_t>::max() ||
          batch > std::numeric_limits<int32_t>::max()) {
        return FLAGFFT_INVALID_SIZE;
      }
      std::size_t index = 5;
      if (axis0 == 0 && axis1 == 2 && axis2 == 1) index = 0;       // 021
      else if (axis0 == 2 && axis1 == 1 && axis2 == 0) index = 1;  // 210
      else if (axis0 == 2 && axis1 == 0 && axis2 == 1) index = 2;  // 201
      else if (axis0 == 1 && axis1 == 2 && axis2 == 0) index = 3;  // 120
      else if (axis0 == 1 && axis1 == 0 && axis2 == 2) index = 4;  // 102
      const uint64_t per_batch_tiles =
          static_cast<uint64_t>((n0 + 15) / 16) * ((n1 + 15) / 16) * ((n2 + 15) / 16);
      const int64_t batch_chunk = std::max<int64_t>(
          1, static_cast<int64_t>(block_limit_per_launch() / std::max<uint64_t>(1, per_batch_tiles)));
      const int64_t elements_per_batch = n0 * n1 * n2;
      for (int64_t batch_offset = 0; batch_offset < batch; batch_offset += batch_chunk) {
        const int64_t chunk = std::min(batch_chunk, batch - batch_offset);
        const adaptor::DevicePtr byte_offset = static_cast<adaptor::DevicePtr>(
            batch_offset * elements_per_batch * element_bytes);
        const flagfftResult result = adaptor::npu::launch_ascendc_transpose3d(
            input + byte_offset,
            output + byte_offset,
            npu_indices[index].get(),
            static_cast<int32_t>(n0),
            static_cast<int32_t>(n1),
            static_cast<int32_t>(n2),
            static_cast<int32_t>(chunk),
            axis0,
            axis1,
            axis2,
            stream);
        if (result != FLAGFFT_SUCCESS) return result;
      }
      return FLAGFFT_SUCCESS;
    }
#endif
    launch_perm3d(kernel, stream, input, output, n0 * n1 * n2, batch, element_bytes);
    return FLAGFFT_SUCCESS;
  }

}  // namespace

CompiledRawLeafNode::CompiledRawLeafNode(int64_t length,
                                         std::shared_ptr<JitKernel> kernel,
                                         std::vector<DeviceAllocation> tables)
    : length(length), kernel(std::move(kernel)), tables(std::move(tables)) {
}

std::string CompiledRawLeafNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRawLeaf(n=" << length << ", kernel=" << (kernel ? kernel->execution_description() : "null")
      << ", num_warps=" << (kernel ? kernel->num_warps : 0)
      << ", module=" << (kernel ? kernel->module_path : "null") << ", tables=" << tables.size() << ")";
  return oss.str();
}

flagfftResult CompiledRawLeafNode::execute(adaptor::DevicePtr input,
                                           adaptor::DevicePtr output,
                                           const RawExecutionContext &context) const {
  try {
    const bool graph_enabled = ix_ct_batch_graph_enabled(context, length);
    if (graph_enabled && replay_leaf_graph(graph_state, context, input, output,
                                           context.input_distance, context.output_distance)) {
      return FLAGFFT_SUCCESS;
    }
    const bool chunk_npu_3d_complex_leaf = context.request.device_type == "npu" &&
        context.request.origin_rank == 3 && context.request.real_transform_kind.empty() &&
        context.request.input_dtype == "complex64" && context.request.output_dtype == "complex64";
    const int64_t batch_limit = std::max<int64_t>(
        1, kernel->batch_per_block * block_limit_per_launch());
    auto launch = [&]() {
      if (!chunk_npu_3d_complex_leaf || context.batch <= batch_limit) {
        std::vector<JitKernelArg> args = raw_kernel_args({input, output}, tables, context.batch);
        kernel->launch(context.stream,
                       args,
                       ceil_div(context.batch, kernel->batch_per_block),
                       1,
                       1);
        return;
      }
      const int64_t element_bytes = complex_element_bytes(context.request.input_dtype);
      for (int64_t batch_offset = 0; batch_offset < context.batch; batch_offset += batch_limit) {
        const int64_t chunk_batch = std::min(batch_limit, context.batch - batch_offset);
        const adaptor::DevicePtr input_chunk = input + batch_offset * length * element_bytes;
        const adaptor::DevicePtr output_chunk = output + batch_offset * length * element_bytes;
        std::vector<JitKernelArg> args = raw_kernel_args(
            {input_chunk, output_chunk}, tables, chunk_batch);
        kernel->launch(context.stream,
                       args,
                       ceil_div(chunk_batch, kernel->batch_per_block),
                       1,
                       1);
      }
    };
    launch();
    if (graph_enabled) capture_leaf_graph(graph_state, context, input, output,
                                          context.input_distance, context.output_distance, launch);
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] Leaf execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRawFourStepFusedNode::CompiledRawFourStepFusedNode(int64_t length,
                                                           int64_t n1,
                                                           int64_t n2,
                                                           std::shared_ptr<JitKernel> row_kernel,
                                                           std::vector<DeviceAllocation> row_tables,
                                                           std::shared_ptr<JitKernel> col_kernel,
                                                           std::vector<DeviceAllocation> col_tables,
                                                           DeviceAllocation twiddle,
                                                           DeviceAllocation stage1)
    : length(length),
      n1(n1),
      n2(n2),
      row_kernel(std::move(row_kernel)),
      row_tables(std::move(row_tables)),
      col_kernel(std::move(col_kernel)),
      col_tables(std::move(col_tables)),
      twiddle(std::move(twiddle)),
      stage1(std::move(stage1)) {
}

std::string CompiledRawFourStepFusedNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRawFourStepFused(n=" << length << ", n1=" << n1 << ", n2=" << n2
      << ", row_kernel=" << (row_kernel ? row_kernel->execution_description() : "null")
      << ", col_kernel=" << (col_kernel ? col_kernel->execution_description() : "null") << ")";
  return oss.str();
}

flagfftResult CompiledRawFourStepFusedNode::execute(adaptor::DevicePtr input,
                                                    adaptor::DevicePtr output,
                                                    const RawExecutionContext &context) const {
  const char *stage = "row";
  try {
    const bool fused_twiddle = row_kernel->tle_fused_twiddle;
    int64_t batch_chunk = context.batch;
    if (context.request.device_type == "npu") {
      const int64_t grid_x =
          std::max(ceil_div(n2, row_kernel->inner_pack), ceil_div(n1, col_kernel->inner_pack));
      const int64_t max_blocks = adaptor::max_launch_blocks();
      if (grid_x <= 0 || max_blocks <= 0) {
        throw std::runtime_error("invalid NPU FourStep launch grid");
      }
      batch_chunk = std::max<int64_t>(1, max_blocks / grid_x);
    }
    const int64_t element_bytes = complex_element_bytes(context.request.input_dtype);
    for (int64_t batch_offset = 0; batch_offset < context.batch; batch_offset += batch_chunk) {
      const int64_t chunk_batch = std::min(batch_chunk, context.batch - batch_offset);
      const auto byte_offset = static_cast<adaptor::DevicePtr>(batch_offset * length * element_bytes);
      const adaptor::DevicePtr input_chunk = input + byte_offset;
      const adaptor::DevicePtr stage1_chunk = stage1.get() + byte_offset;
      const adaptor::DevicePtr output_chunk = output + byte_offset;

      std::vector<JitKernelArg> row_args =
          fused_twiddle ? raw_kernel_args({input_chunk, twiddle.get(), stage1_chunk}, row_tables, chunk_batch)
                        : raw_kernel_args({input_chunk, stage1_chunk}, row_tables, chunk_batch);
      row_kernel->launch(context.stream, row_args, ceil_div(n2, row_kernel->inner_pack), chunk_batch, 1);

      stage = "column";
      std::vector<JitKernelArg> col_args =
          fused_twiddle
              ? raw_kernel_args({stage1_chunk, output_chunk}, col_tables, chunk_batch)
              : raw_kernel_args({stage1_chunk, twiddle.get(), output_chunk}, col_tables, chunk_batch);
      col_kernel->launch(context.stream, col_args, ceil_div(n1, col_kernel->inner_pack), chunk_batch, 1);
      stage = "row";
    }
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] FourStepFused execute failed in %s stage: %s\n", stage, e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRawFourStepStridedNode::CompiledRawFourStepStridedNode(int64_t length,
                                                               int64_t n1,
                                                               int64_t n2,
                                                               int64_t outer_stride,
                                                               std::shared_ptr<JitKernel> row_kernel,
                                                               std::vector<DeviceAllocation> row_tables,
                                                               std::shared_ptr<JitKernel> col_kernel,
                                                               std::vector<DeviceAllocation> col_tables,
                                                               DeviceAllocation twiddle,
                                                               DeviceAllocation stage1)
    : length(length),
      n1(n1),
      n2(n2),
      outer_stride(outer_stride),
      row_kernel(std::move(row_kernel)),
      row_tables(std::move(row_tables)),
      col_kernel(std::move(col_kernel)),
      col_tables(std::move(col_tables)),
      twiddle(std::move(twiddle)),
      stage1(std::move(stage1)) {
}

std::string CompiledRawFourStepStridedNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRawFourStepStrided(n=" << length << ", n1=" << n1 << ", n2=" << n2
      << ", outer_stride=" << outer_stride
      << ", row_kernel=" << (row_kernel ? row_kernel->execution_description() : "null")
      << ", col_kernel=" << (col_kernel ? col_kernel->execution_description() : "null") << ")";
  return oss.str();
}

flagfftResult CompiledRawFourStepStridedNode::execute(adaptor::DevicePtr input,
                                                      adaptor::DevicePtr output,
                                                      const RawExecutionContext &context) const {
  try {
    auto append_strided = [&](std::vector<JitKernelArg> &args, const std::vector<DeviceAllocation> &tables) {
      for (const DeviceAllocation &table : tables) {
        args.push_back(JitKernelArg::device(table.get()));
      }
      args.push_back(JitKernelArg::i64(outer_stride));
      args.push_back(JitKernelArg::i32(static_cast<int32_t>(context.batch)));
    };

    std::vector<JitKernelArg> row_args = {
        JitKernelArg::device(input),
        JitKernelArg::device(stage1.get()),
    };
    append_strided(row_args, row_tables);
    row_kernel->launch(context.stream, row_args, ceil_div(n2, row_kernel->inner_pack), context.batch, 1);

    std::vector<JitKernelArg> col_args = {
        JitKernelArg::device(stage1.get()),
        JitKernelArg::device(twiddle.get()),
        JitKernelArg::device(output),
    };
    append_strided(col_args, col_tables);
    col_kernel->launch(context.stream, col_args, ceil_div(n1, col_kernel->inner_pack), context.batch, 1);
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] FourStepStrided execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRawStridedLeafNode::CompiledRawStridedLeafNode(int64_t length,
                                                       int64_t outer_stride,
                                                       std::shared_ptr<JitKernel> kernel,
                                                       std::vector<DeviceAllocation> tables)
    : length(length), outer_stride(outer_stride), kernel(std::move(kernel)), tables(std::move(tables)) {
}

std::string CompiledRawStridedLeafNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRawStridedLeaf(n=" << length << ", outer_stride=" << outer_stride
      << ", kernel=" << (kernel ? kernel->execution_description() : "null")
      << ", num_warps=" << (kernel ? kernel->num_warps : 0)
      << ", module=" << (kernel ? kernel->module_path : "null") << ", tables=" << tables.size() << ")";
  return oss.str();
}

flagfftResult CompiledRawStridedLeafNode::execute(adaptor::DevicePtr input,
                                                  adaptor::DevicePtr output,
                                                  const RawExecutionContext &context) const {
  try {
    std::vector<JitKernelArg> args;
    args.reserve(tables.size() + 3);
    args.push_back(JitKernelArg::device(input));
    args.push_back(JitKernelArg::device(output));
    for (const DeviceAllocation &table : tables) {
      args.push_back(JitKernelArg::device(table.get()));
    }
    args.push_back(JitKernelArg::i64(outer_stride));
    args.push_back(JitKernelArg::i32(static_cast<int32_t>(context.batch)));
    kernel->launch(context.stream, args, ceil_div(context.batch, kernel->batch_per_block), 1, 1);
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] StridedLeaf execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

#if defined(FLAGFFT_BACKEND_NPU)
CompiledRawNpuAivFFT64Node::CompiledRawNpuAivFFT64Node(
    int64_t stride,
    int64_t group_size,
    std::shared_ptr<DeviceAllocation> indices,
    std::shared_ptr<DeviceAllocation> twiddles,
    NpuAivFFT64Mode mode)
    : stride(stride),
      group_size(group_size),
      mode(mode),
      indices(std::move(indices)),
      twiddles(std::move(twiddles)) {
}

CompiledRawNpuAivFFTSmallNode::CompiledRawNpuAivFFTSmallNode(
    int64_t length,
    int64_t stride,
    int64_t group_size,
    std::shared_ptr<DeviceAllocation> indices,
    std::shared_ptr<DeviceAllocation> twiddles)
    : length(length),
      stride(stride),
      group_size(group_size),
      indices(std::move(indices)),
      twiddles(std::move(twiddles)) {}

std::string CompiledRawNpuAivFFTSmallNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRawNpuAivFFTSmall(n=" << length << ", stride=" << stride
      << ", group_size=" << group_size << ")";
  return oss.str();
}

flagfftResult CompiledRawNpuAivFFTSmallNode::execute(
    adaptor::DevicePtr input,
    adaptor::DevicePtr output,
    const RawExecutionContext &context) const {
  const int64_t expected_group = length == 16 ? 8 : length == 32 ? 4 : 0;
  if (context.batch <= 0 || context.batch > std::numeric_limits<int32_t>::max() ||
      (length != 16 && length != 32) || group_size != expected_group || stride <= 0 ||
      (stride != 1 && (stride < group_size || stride % group_size != 0)) ||
      context.batch % group_size != 0 || indices == nullptr || twiddles == nullptr) {
    return FLAGFFT_INVALID_SIZE;
  }

  constexpr int64_t kMaxBlocksPerLaunch = 65535;
  const int64_t blocks = context.batch / group_size;
  const int64_t element_bytes = 2 * sizeof(float);
  for (int64_t block_offset = 0; block_offset < blocks;
       block_offset += kMaxBlocksPerLaunch) {
    const int64_t chunk_blocks = std::min(kMaxBlocksPerLaunch, blocks - block_offset);
    const int64_t transform_offset = block_offset * group_size;
    const int64_t pointer_offset = stride == 1
        ? transform_offset * length * element_bytes
        : (transform_offset / stride) * length * stride * element_bytes +
              (transform_offset % stride) * element_bytes;
    const flagfftResult result = adaptor::npu::launch_ascendc_fft_small(
        static_cast<int32_t>(length), input + pointer_offset, output + pointer_offset,
        indices->get(), twiddles->get(), static_cast<int32_t>(chunk_blocks),
        static_cast<int32_t>(stride), context.stream);
    if (result != FLAGFFT_SUCCESS) return result;
  }
  return FLAGFFT_SUCCESS;
}

std::string CompiledRawNpuAivFFT64Node::describe() const {
  std::ostringstream oss;
  oss << "CompiledRawNpuAivFFT64(stride=" << stride << ", group_size=" << group_size << ")";
  if (mode == NpuAivFFT64Mode::RealForward) oss << "[r2c-row]";
  if (mode == NpuAivFFT64Mode::RealInverse) oss << "[c2r-row]";
  return oss.str();
}

flagfftResult CompiledRawNpuAivFFT64Node::execute(adaptor::DevicePtr input,
                                                  adaptor::DevicePtr output,
                                                  const RawExecutionContext &context) const {
  if (context.batch <= 0 || context.batch > std::numeric_limits<int32_t>::max() ||
      stride <= 0 ||
      (group_size != 1 && group_size != 4 && group_size != 8) ||
      (group_size != 1 && stride != 1 && stride != 64) ||
      (mode != NpuAivFFT64Mode::Complex && stride != 1) ||
      indices == nullptr || twiddles == nullptr) {
    return FLAGFFT_INVALID_SIZE;
  }
  int32_t transform_count = static_cast<int32_t>(context.batch);
  if (group_size > 1) {
    if (transform_count % group_size != 0) return FLAGFFT_INVALID_SIZE;
    transform_count /= static_cast<int32_t>(group_size);
  }
  return adaptor::npu::launch_ascendc_fft64(input,
                                             output,
                                             indices->get(),
                                             twiddles->get(),
                                             transform_count,
                                             static_cast<int32_t>(stride),
                                             static_cast<int32_t>(group_size),
                                             static_cast<int32_t>(mode),
                                             context.stream);
}

CompiledRawNpuAivFFTNode::CompiledRawNpuAivFFTNode(
    int64_t length,
    std::shared_ptr<DeviceAllocation> indices,
    std::shared_ptr<DeviceAllocation> twiddles,
    int64_t group_size)
    : length(length),
      group_size(group_size),
      indices(std::move(indices)),
      twiddles(std::move(twiddles)) {}

std::string CompiledRawNpuAivFFTNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRawNpuAivFFT(n=" << length << ", group_size=" << group_size << ")";
  return oss.str();
}

flagfftResult CompiledRawNpuAivFFTNode::execute(adaptor::DevicePtr input,
                                                adaptor::DevicePtr output,
                                                const RawExecutionContext &context) const {
  if (context.batch <= 0 || indices == nullptr || twiddles == nullptr ||
      (length == 128 && group_size != 4 && group_size != 8) ||
      (length == 2048 && group_size != 1) || (length != 128 && length != 2048) ||
      context.batch % group_size != 0) {
    return FLAGFFT_INVALID_SIZE;
  }

  constexpr int64_t kMaxBlocksPerLaunch = 65535;
  const int64_t max_transforms_per_launch = kMaxBlocksPerLaunch * group_size;
  const int64_t element_bytes = 2 * sizeof(float);
  for (int64_t offset = 0; offset < context.batch; offset += max_transforms_per_launch) {
    const int64_t chunk = std::min(max_transforms_per_launch, context.batch - offset);
    if (chunk % group_size != 0) return FLAGFFT_INVALID_SIZE;
    const adaptor::DevicePtr input_chunk = input + offset * length * element_bytes;
    const adaptor::DevicePtr output_chunk = output + offset * length * element_bytes;
    flagfftResult result;
    if (length == 128) {
      result = adaptor::npu::launch_ascendc_fft128(
          input_chunk, output_chunk, indices->get(), twiddles->get(),
          static_cast<int32_t>(chunk / group_size), 1, static_cast<int32_t>(group_size), 0,
          context.stream);
    } else {
      result = adaptor::npu::launch_ascendc_fft2048(
          input_chunk, output_chunk, indices->get(), twiddles->get(),
          static_cast<int32_t>(chunk), context.stream);
    }
    if (result != FLAGFFT_SUCCESS) return result;
  }
  return FLAGFFT_SUCCESS;
}

CompiledRawNpuAivFFT256Node::CompiledRawNpuAivFFT256Node(
    std::shared_ptr<DeviceAllocation> indices,
    std::shared_ptr<DeviceAllocation> twiddles,
    int64_t group_size,
    bool pair_mode,
    bool transposed_store,
    bool radix4_mode,
    int64_t transposed_output_row_stride)
    : indices(std::move(indices)),
      twiddles(std::move(twiddles)),
      group_size(group_size),
      pair_mode(pair_mode),
      transposed_store(transposed_store),
      radix4_mode(radix4_mode),
      transposed_output_row_stride(transposed_output_row_stride) {}

std::string CompiledRawNpuAivFFT256Node::describe() const {
  std::ostringstream oss;
  oss << "CompiledRawNpuAivFFT256(group_size=" << group_size
      << ", pair_mode=" << pair_mode
      << ", transposed_store=" << transposed_store
      << ", radix4_mode=" << radix4_mode
      << ", transposed_output_row_stride=" << transposed_output_row_stride << ")";
  return oss.str();
}

flagfftResult CompiledRawNpuAivFFT256Node::execute(adaptor::DevicePtr input,
                                                   adaptor::DevicePtr output,
                                                   const RawExecutionContext &context) const {
  if (context.batch <= 0 || context.batch > std::numeric_limits<int32_t>::max() ||
      (group_size != 1 && group_size != 4 && group_size != 8) ||
      (pair_mode && group_size != 8) ||
      (transposed_store && !pair_mode) ||
      (radix4_mode && (!pair_mode || !transposed_store || group_size != 8)) ||
      context.batch % group_size != 0 ||
      indices == nullptr || twiddles == nullptr) {
    return FLAGFFT_INVALID_SIZE;
  }
  const int64_t output_row_stride =
      transposed_output_row_stride > 0 ? transposed_output_row_stride : context.batch;
  if (transposed_store && (output_row_stride < group_size ||
                           output_row_stride > std::numeric_limits<int32_t>::max() ||
                           output_row_stride % group_size != 0)) {
    return FLAGFFT_INVALID_SIZE;
  }
  const int64_t batch_chunk = block_limit_per_launch() * group_size;
  if (batch_chunk <= 0) return FLAGFFT_INVALID_SIZE;
  const int64_t element_bytes = complex_element_bytes(context.request.input_dtype);
  for (int64_t batch_offset = 0; batch_offset < context.batch; batch_offset += batch_chunk) {
    const int64_t chunk_batch = std::min(batch_chunk, context.batch - batch_offset);
    const adaptor::DevicePtr byte_offset =
        static_cast<adaptor::DevicePtr>(batch_offset * 256 * element_bytes);
    const flagfftResult result = adaptor::npu::launch_ascendc_fft256(
        input + byte_offset,
        transposed_store ? output : output + byte_offset,
        indices->get(),
        twiddles->get(),
        static_cast<int32_t>(chunk_batch),
        static_cast<int32_t>(group_size),
        pair_mode,
        transposed_store,
        radix4_mode,
        transposed_store ? static_cast<int32_t>(output_row_stride) : 0,
        transposed_store ? static_cast<int32_t>(batch_offset) : 0,
        context.stream);
    if (result != FLAGFFT_SUCCESS) return result;
  }
  return FLAGFFT_SUCCESS;
}
#endif

CompiledRawStridedDirectDftNode::CompiledRawStridedDirectDftNode(int64_t length,
                                                                 int64_t outer_stride,
                                                                 std::shared_ptr<JitKernel> kernel,
                                                                 std::vector<DeviceAllocation> tables)
    : length(length), outer_stride(outer_stride), kernel(std::move(kernel)), tables(std::move(tables)) {
}

std::string CompiledRawStridedDirectDftNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRawStridedDirectDft(n=" << length << ", outer_stride=" << outer_stride
      << ", kernel=" << (kernel ? kernel->execution_description() : "null")
      << ", num_warps=" << (kernel ? kernel->num_warps : 0)
      << ", module=" << (kernel ? kernel->module_path : "null") << ", tables=" << tables.size() << ")";
  return oss.str();
}

flagfftResult CompiledRawStridedDirectDftNode::execute(adaptor::DevicePtr input,
                                                       adaptor::DevicePtr output,
                                                       const RawExecutionContext &context) const {
  try {
    std::vector<JitKernelArg> args;
    args.reserve(tables.size() + 3);
    args.push_back(JitKernelArg::device(input));
    args.push_back(JitKernelArg::device(output));
    for (const DeviceAllocation &table : tables) {
      args.push_back(JitKernelArg::device(table.get()));
    }
    args.push_back(JitKernelArg::i64(outer_stride));
    args.push_back(JitKernelArg::i32(static_cast<int32_t>(context.batch)));
    kernel->launch(context.stream, args, ceil_div(context.batch, kernel->batch_per_block), 1, 1);
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] StridedDirectDft execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRawStockhamNode::CompiledRawStockhamNode(int64_t length,
                                                 std::vector<int64_t> factors,
                                                 std::vector<std::shared_ptr<JitKernel>> kernels,
                                                 DeviceAllocation twiddle,
                                                 DeviceAllocation first,
                                                 DeviceAllocation second)
    : length(length),
      factors(std::move(factors)),
      kernels(std::move(kernels)),
      twiddle(std::move(twiddle)),
      first(std::move(first)),
      second(std::move(second)) {
}

std::string CompiledRawStockhamNode::describe() const {
  return "CompiledRawStockham(n=" + std::to_string(length) + ")";
}

flagfftResult CompiledRawStockhamNode::execute(adaptor::DevicePtr input,
                                               adaptor::DevicePtr output,
                                               const RawExecutionContext &context) const {
  try {
    adaptor::DevicePtr current = input;
    int64_t span = 1;
    const int64_t element_bytes = complex_element_bytes(context.request.input_dtype);
    // One stage launch covers ceil(batch * length / factor / block) blocks, which
    // exceeds the NPU launch limit for large batched transforms. Batches are
    // independent, so a stage is split into batch chunks with pointer offsets.
    const int64_t block_limit = block_limit_per_launch();
    for (std::size_t stage = 0; stage < factors.size(); ++stage) {
      const int64_t factor = factors[stage];
      const int64_t block = kernels[stage]->butterflies_per_block;
      const int64_t programs_per_batch = (length / factor + block - 1) / block;
      const int64_t batch_chunk = std::max<int64_t>(
          1,
          std::min<int64_t>(context.batch, block_limit / std::max<int64_t>(1, programs_per_batch)));
      const bool last = stage + 1 == factors.size();
      adaptor::DevicePtr dst =
          last && current != output ? output : (stage % 2 == 0 ? first.get() : second.get());
      for (int64_t batch_offset = 0; batch_offset < context.batch; batch_offset += batch_chunk) {
        const int64_t count = std::min(batch_chunk, context.batch - batch_offset);
        const int64_t byte_offset = batch_offset * length * element_bytes;
        std::vector<JitKernelArg> args = {JitKernelArg::device(current + byte_offset),
                                          JitKernelArg::device(dst + byte_offset),
                                          JitKernelArg::device(twiddle.get()),
                                          JitKernelArg::i64(span),
                                          JitKernelArg::i32(static_cast<int32_t>(count))};
        const int64_t butterflies = count * (length / factor);
        kernels[stage]->launch(context.stream, args, (butterflies + block - 1) / block, 1, 1);
      }
      current = dst;
      span *= factor;
    }
    if (current != output) {
      adaptor::copy_device_to_device(output,
                                     current,
                                     static_cast<std::size_t>(context.batch * length * element_bytes),
                                     context.stream);
    }
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] Stockham execute failed: %s\n", e.what());
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRawDirectDftNode::CompiledRawDirectDftNode(int64_t length,
                                                   std::shared_ptr<JitKernel> kernel,
                                                   std::vector<DeviceAllocation> tables,
                                                   DeviceAllocation input_copy)
    : length(length),
      kernel(std::move(kernel)),
      tables(std::move(tables)),
      input_copy(std::move(input_copy)) {
}

std::string CompiledRawDirectDftNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRawDirectDft(n=" << length
      << ", kernel=" << (kernel ? kernel->execution_description() : "null") << ")";
  return oss.str();
}

flagfftResult CompiledRawDirectDftNode::execute(adaptor::DevicePtr input,
                                                adaptor::DevicePtr output,
                                                const RawExecutionContext &context) const {
  try {
    adaptor::DevicePtr effective_input = input;
    if (input == output) {
      adaptor::copy_device_to_device(input_copy.get(), input, input_copy.size(), context.stream);
      effective_input = input_copy.get();
    }

    std::vector<JitKernelArg> args = raw_kernel_args({effective_input, output}, tables, context.batch);
    kernel->launch(context.stream,
                   args,
                   ceil_div(context.batch, kernel->batch_per_block),
                   std::max<int64_t>(1, kernel->grid_y_override),
                   1);
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] DirectDFT execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRawBluesteinNode::CompiledRawBluesteinNode(int64_t length,
                                                   int64_t conv_length,
                                                   std::shared_ptr<CompiledRawNode> fft,
                                                   std::shared_ptr<JitKernel> prepare_kernel,
                                                   std::shared_ptr<JitKernel> pointwise_kernel,
                                                   std::shared_ptr<JitKernel> finalize_kernel,
                                                   DeviceAllocation chirp,
                                                   DeviceAllocation b_time,
                                                   DeviceAllocation a_buf,
                                                   DeviceAllocation work_buf,
                                                   DeviceAllocation b_fft_buf,
                                                   int64_t batch_chunk)
    : length(length),
      conv_length(conv_length),
      batch_chunk(batch_chunk),
      fft(std::move(fft)),
      prepare_kernel(std::move(prepare_kernel)),
      pointwise_kernel(std::move(pointwise_kernel)),
      finalize_kernel(std::move(finalize_kernel)),
      chirp(std::move(chirp)),
      b_time(std::move(b_time)),
      a_buf(std::move(a_buf)),
      work_buf(std::move(work_buf)),
      b_fft_buf(std::move(b_fft_buf)) {
}

std::string CompiledRawBluesteinNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRawBluestein(n=" << length << ", conv_length=" << conv_length
      << ", prepare_kernel=" << (prepare_kernel ? prepare_kernel->execution_description() : "null")
      << ", pointwise_kernel=" << (pointwise_kernel ? pointwise_kernel->execution_description() : "null")
      << ", finalize_kernel=" << (finalize_kernel ? finalize_kernel->execution_description() : "null")
      << ", fft=" << (fft ? fft->describe() : "null") << ")";
  return oss.str();
}

void CompiledRawBluesteinNode::ensure_b_fft(const RawExecutionContext &context) const {
  std::lock_guard<std::mutex> lock(b_fft_mutex);
  if (b_fft_ready) {
    return;
  }
  RawExecutionContext child_context {context.request, context.stream, 1};
  flagfftResult result = fft->execute(b_time.get(), b_fft_buf.get(), child_context);
  if (result != FLAGFFT_SUCCESS) {
    throw std::runtime_error("failed to precompute Bluestein convolution FFT");
  }
  b_fft_ready = true;
}

flagfftResult CompiledRawBluesteinNode::execute(adaptor::DevicePtr input,
                                                adaptor::DevicePtr output,
                                                const RawExecutionContext &context) const {
  try {
    ensure_b_fft(context);
    const int64_t element_bytes = complex_element_bytes(context.request.input_dtype);
    const int64_t stride_bytes = length * element_bytes;
    for (int64_t batch_offset = 0; batch_offset < context.batch; batch_offset += batch_chunk) {
      const int64_t chunk = std::min(batch_chunk, context.batch - batch_offset);
      const int64_t input_offset = batch_offset * stride_bytes;
      const int64_t output_offset = batch_offset * stride_bytes;

      std::vector<JitKernelArg> prepare_args = {
          JitKernelArg::device(input + input_offset),
          JitKernelArg::device(chirp.get()),
          JitKernelArg::device(a_buf.get()),
          JitKernelArg::i64(length),
          JitKernelArg::i64(conv_length),
          JitKernelArg::i32(static_cast<int32_t>(chunk)),
      };
      prepare_kernel->launch(context.stream, prepare_args, ceil_div(conv_length, 256), chunk, 1);

      RawExecutionContext child_context {context.request, context.stream, chunk};
      flagfftResult result = fft->execute(a_buf.get(), work_buf.get(), child_context);
      if (result != FLAGFFT_SUCCESS) {
        return result;
      }

      std::vector<JitKernelArg> pointwise_args = {
          JitKernelArg::device(work_buf.get()),
          JitKernelArg::device(b_fft_buf.get()),
          JitKernelArg::device(a_buf.get()),
          JitKernelArg::i64(conv_length),
          JitKernelArg::i32(static_cast<int32_t>(chunk)),
      };
      pointwise_kernel->launch(context.stream, pointwise_args, ceil_div(conv_length, 256), chunk, 1);

      result = fft->execute(a_buf.get(), work_buf.get(), child_context);
      if (result != FLAGFFT_SUCCESS) {
        return result;
      }

      std::vector<JitKernelArg> finalize_args = {
          JitKernelArg::device(work_buf.get()),
          JitKernelArg::device(chirp.get()),
          JitKernelArg::device(output + output_offset),
          JitKernelArg::i64(length),
          JitKernelArg::i64(conv_length),
          JitKernelArg::i32(static_cast<int32_t>(chunk)),
      };
      finalize_kernel->launch(context.stream, finalize_args, ceil_div(length, 256), chunk, 1);
    }
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] Bluestein execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRawBluesteinLeafNode::CompiledRawBluesteinLeafNode(int64_t length,
                                                           int64_t conv_length,
                                                           std::shared_ptr<CompiledRawNode> fft,
                                                           std::shared_ptr<JitKernel> prepare_kernel,
                                                           std::shared_ptr<JitKernel> finish_kernel,
                                                           std::vector<DeviceAllocation> tables,
                                                           DeviceAllocation chirp,
                                                           DeviceAllocation b_time,
                                                           DeviceAllocation work_buf,
                                                           DeviceAllocation b_fft_buf)
    : length(length),
      conv_length(conv_length),
      fft(std::move(fft)),
      prepare_kernel(std::move(prepare_kernel)),
      finish_kernel(std::move(finish_kernel)),
      tables(std::move(tables)),
      chirp(std::move(chirp)),
      b_time(std::move(b_time)),
      work_buf(std::move(work_buf)),
      b_fft_buf(std::move(b_fft_buf)) {
}

std::string CompiledRawBluesteinLeafNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRawBluesteinLeaf(n=" << length << ", conv_length=" << conv_length
      << ", prepare_kernel=" << (prepare_kernel ? prepare_kernel->execution_description() : "null")
      << ", finish_kernel=" << (finish_kernel ? finish_kernel->execution_description() : "null")
      << ", fft=" << (fft ? fft->describe() : "null") << ")";
  return oss.str();
}

void CompiledRawBluesteinLeafNode::ensure_b_fft(const RawExecutionContext &context) const {
  std::lock_guard<std::mutex> lock(b_fft_mutex);
  if (b_fft_ready) {
    return;
  }
  RawExecutionContext child_context {context.request, context.stream, 1};
  flagfftResult result = fft->execute(b_time.get(), b_fft_buf.get(), child_context);
  if (result != FLAGFFT_SUCCESS) {
    throw std::runtime_error("failed to precompute fused Bluestein convolution FFT");
  }
  b_fft_ready = true;
}

flagfftResult CompiledRawBluesteinLeafNode::execute(adaptor::DevicePtr input,
                                                    adaptor::DevicePtr output,
                                                    const RawExecutionContext &context) const {
  try {
    ensure_b_fft(context);

    std::vector<JitKernelArg> prepare_args =
        raw_kernel_args({input, chirp.get(), work_buf.get()}, tables, context.batch);
    prepare_kernel->launch(context.stream,
                           prepare_args,
                           ceil_div(context.batch, prepare_kernel->batch_per_block),
                           1,
                           1);

    std::vector<JitKernelArg> finish_args =
        raw_kernel_args({work_buf.get(), b_fft_buf.get(), chirp.get(), output}, tables, context.batch);
    finish_kernel->launch(context.stream,
                          finish_args,
                          ceil_div(context.batch, finish_kernel->batch_per_block),
                          1,
                          1);
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] BluesteinLeaf execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRawBluesteinFullLeafNode::CompiledRawBluesteinFullLeafNode(int64_t length,
                                                                   int64_t conv_length,
                                                                   std::shared_ptr<CompiledRawNode> fft,
                                                                   std::shared_ptr<JitKernel> kernel,
                                                                   std::vector<DeviceAllocation> tables,
                                                                   DeviceAllocation chirp,
                                                                   DeviceAllocation b_time,
                                                                   DeviceAllocation b_fft_buf,
                                                                   std::string real_kind,
                                                                   std::function<std::shared_ptr<CompiledRawNode>()> make_layout_fallback)
    : length(length),
      conv_length(conv_length),
      fft(std::move(fft)),
      kernel(std::move(kernel)),
      tables(std::move(tables)),
      chirp(std::move(chirp)),
      b_time(std::move(b_time)),
      b_fft_buf(std::move(b_fft_buf)),
      real_kind(std::move(real_kind)),
      make_layout_fallback(std::move(make_layout_fallback)) {
}

std::string CompiledRawBluesteinFullLeafNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRawBluesteinFullLeaf(n=" << length << ", conv_length=" << conv_length
      << ", kernel=" << (kernel ? kernel->execution_description() : "null")
      << ", fft=" << (fft ? fft->describe() : "null") << ")";
  return oss.str();
}

void CompiledRawBluesteinFullLeafNode::ensure_b_fft(const RawExecutionContext &context) const {
  std::lock_guard<std::mutex> lock(b_fft_mutex);
  if (b_fft_ready) {
    return;
  }
  RawExecutionContext child_context {context.request, context.stream, 1};
  flagfftResult result = fft->execute(b_time.get(), b_fft_buf.get(), child_context);
  if (result != FLAGFFT_SUCCESS) {
    throw std::runtime_error("failed to precompute fully fused Bluestein convolution FFT");
  }
  b_fft_ready = true;
}

flagfftResult CompiledRawBluesteinFullLeafNode::execute(adaptor::DevicePtr input,
                                                        adaptor::DevicePtr output,
                                                        const RawExecutionContext &context) const {
  try {
    if (!real_kind.empty()) {
      const int64_t half = length / 2 + 1;
      const int64_t dense_input = real_kind == "r2c" ? length : half;
      const int64_t dense_output = real_kind == "r2c" ? half : length;
      if (input == output || (context.input_distance > 0 && context.input_distance != dense_input) ||
          (context.output_distance > 0 && context.output_distance != dense_output)) {
        std::shared_ptr<CompiledRawNode> fallback;
        {
          std::lock_guard<std::mutex> lock(layout_mutex);
          if (!layout_fallback && make_layout_fallback) layout_fallback = make_layout_fallback();
          fallback = layout_fallback;
        }
        return fallback ? fallback->execute(input, output, context) : FLAGFFT_INVALID_VALUE;
      }
    }
    ensure_b_fft(context);
    std::vector<JitKernelArg> args =
        raw_kernel_args({input, b_fft_buf.get(), chirp.get(), output}, tables, context.batch);
    kernel->launch(context.stream, args, ceil_div(context.batch, kernel->batch_per_block), 1, 1);
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] BluesteinFullLeaf execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRawBluesteinFourStepNode::CompiledRawBluesteinFourStepNode(
    int64_t length,
    int64_t conv_length,
    int64_t n1,
    int64_t n2,
    std::shared_ptr<CompiledRawNode> fft,
    std::shared_ptr<JitKernel> prepare_row_kernel,
    std::shared_ptr<JitKernel> first_col_kernel,
    std::shared_ptr<JitKernel> pointwise_row_kernel,
    std::shared_ptr<JitKernel> finish_col_kernel,
    std::vector<DeviceAllocation> row_tables,
    std::vector<DeviceAllocation> col_tables,
    DeviceAllocation twiddle,
    DeviceAllocation chirp,
    DeviceAllocation b_time,
    DeviceAllocation stage1,
    DeviceAllocation work_buf,
    DeviceAllocation b_fft_buf,
    std::string real_kind,
    std::function<std::shared_ptr<CompiledRawNode>()> make_layout_fallback)
    : length(length),
      conv_length(conv_length),
      n1(n1),
      n2(n2),
      fft(std::move(fft)),
      prepare_row_kernel(std::move(prepare_row_kernel)),
      first_col_kernel(std::move(first_col_kernel)),
      pointwise_row_kernel(std::move(pointwise_row_kernel)),
      finish_col_kernel(std::move(finish_col_kernel)),
      row_tables(std::move(row_tables)),
      col_tables(std::move(col_tables)),
      twiddle(std::move(twiddle)),
      chirp(std::move(chirp)),
      b_time(std::move(b_time)),
      stage1(std::move(stage1)),
      work_buf(std::move(work_buf)),
      b_fft_buf(std::move(b_fft_buf)),
      real_kind(std::move(real_kind)),
      make_layout_fallback(std::move(make_layout_fallback)) {
}

std::string CompiledRawBluesteinFourStepNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRawBluesteinFourStep(n=" << length << ", conv_length=" << conv_length << ", n1=" << n1
      << ", n2=" << n2
      << ", prepare_row=" << (prepare_row_kernel ? prepare_row_kernel->execution_description() : "null")
      << ", first_col=" << (first_col_kernel ? first_col_kernel->execution_description() : "null")
      << ", pointwise_row=" << (pointwise_row_kernel ? pointwise_row_kernel->execution_description() : "null")
      << ", finish_col=" << (finish_col_kernel ? finish_col_kernel->execution_description() : "null") << ")";
  return oss.str();
}

void CompiledRawBluesteinFourStepNode::ensure_b_fft(const RawExecutionContext &context) const {
  std::lock_guard<std::mutex> lock(b_fft_mutex);
  if (b_fft_ready) {
    return;
  }
  RawExecutionContext child_context {context.request, context.stream, 1};
  flagfftResult result = fft->execute(b_time.get(), b_fft_buf.get(), child_context);
  if (result != FLAGFFT_SUCCESS) {
    throw std::runtime_error("failed to precompute four-step Bluestein convolution FFT");
  }
  b_fft_ready = true;
}

flagfftResult CompiledRawBluesteinFourStepNode::execute(adaptor::DevicePtr input,
                                                        adaptor::DevicePtr output,
                                                        const RawExecutionContext &context) const {
  const char *stage = "precompute";
  try {
    if (!real_kind.empty()) {
      const int64_t half = length / 2 + 1;
      const int64_t dense_input = real_kind == "r2c" ? length : half;
      const int64_t dense_output = real_kind == "r2c" ? half : length;
      if (input == output || (context.input_distance > 0 && context.input_distance != dense_input) ||
          (context.output_distance > 0 && context.output_distance != dense_output)) {
        std::shared_ptr<CompiledRawNode> fallback;
        {
          std::lock_guard<std::mutex> lock(layout_mutex);
          if (!layout_fallback && make_layout_fallback) layout_fallback = make_layout_fallback();
          fallback = layout_fallback;
        }
        return fallback ? fallback->execute(input, output, context) : FLAGFFT_INVALID_VALUE;
      }
    }
    ensure_b_fft(context);
    const bool fused_twiddle = prepare_row_kernel->tle_fused_twiddle;

    stage = "prepare-row";
    std::vector<JitKernelArg> prepare_args =
        fused_twiddle
            ? raw_kernel_args({input, chirp.get(), twiddle.get(), stage1.get()}, row_tables, context.batch)
            : raw_kernel_args({input, chirp.get(), stage1.get()}, row_tables, context.batch);
    prepare_row_kernel->launch(context.stream,
                               prepare_args,
                               ceil_div(n2, prepare_row_kernel->inner_pack),
                               context.batch,
                               1);

    stage = "first-col";
    std::vector<JitKernelArg> first_col_args =
        fused_twiddle
            ? raw_kernel_args({stage1.get(), work_buf.get()}, col_tables, context.batch)
            : raw_kernel_args({stage1.get(), twiddle.get(), work_buf.get()}, col_tables, context.batch);
    first_col_kernel->launch(context.stream,
                             first_col_args,
                             ceil_div(n1, first_col_kernel->inner_pack),
                             context.batch,
                             1);

    stage = "pointwise-row";
    std::vector<JitKernelArg> pointwise_args =
        fused_twiddle
            ? raw_kernel_args({work_buf.get(), b_fft_buf.get(), twiddle.get(), stage1.get()},
                              row_tables,
                              context.batch)
            : raw_kernel_args({work_buf.get(), b_fft_buf.get(), stage1.get()}, row_tables, context.batch);
    pointwise_row_kernel->launch(context.stream,
                                 pointwise_args,
                                 ceil_div(n2, pointwise_row_kernel->inner_pack),
                                 context.batch,
                                 1);

    stage = "finish-col";
    std::vector<JitKernelArg> finish_args =
        fused_twiddle
            ? raw_kernel_args({stage1.get(), chirp.get(), output}, col_tables, context.batch)
            : raw_kernel_args({stage1.get(), chirp.get(), twiddle.get(), output}, col_tables, context.batch);
    finish_col_kernel->launch(context.stream,
                              finish_args,
                              ceil_div(n1, finish_col_kernel->inner_pack),
                              context.batch,
                              1);
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] BluesteinFourStep %s failed: %s\n", stage, e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRawRaderNode::CompiledRawRaderNode(int64_t length,
                                           int64_t conv_length,
                                           std::shared_ptr<CompiledRawNode> fft,
                                           std::shared_ptr<JitKernel> prepare_kernel,
                                           std::shared_ptr<JitKernel> pointwise_kernel,
                                           std::shared_ptr<JitKernel> finalize_kernel,
                                           DeviceAllocation idx,
                                           DeviceAllocation b_time,
                                           DeviceAllocation a_buf,
                                           DeviceAllocation work_buf,
                                           DeviceAllocation b_fft_buf,
                                           DeviceAllocation input_copy,
                                           std::shared_ptr<JitKernel> fused_leaf_kernel,
                                           std::vector<DeviceAllocation> fused_leaf_tables,
                                           std::shared_ptr<JitKernel> boundary_prepare_kernel,
                                           std::shared_ptr<JitKernel> boundary_finish_kernel,
                                           std::vector<DeviceAllocation> boundary_tables)
    : length(length),
      conv_length(conv_length),
      fft(std::move(fft)),
      prepare_kernel(std::move(prepare_kernel)),
      pointwise_kernel(std::move(pointwise_kernel)),
      finalize_kernel(std::move(finalize_kernel)),
      fused_leaf_kernel(std::move(fused_leaf_kernel)),
      fused_leaf_tables(std::move(fused_leaf_tables)),
      boundary_prepare_kernel(std::move(boundary_prepare_kernel)),
      boundary_finish_kernel(std::move(boundary_finish_kernel)),
      boundary_tables(std::move(boundary_tables)),
      idx(std::move(idx)),
      b_time(std::move(b_time)),
      a_buf(std::move(a_buf)),
      work_buf(std::move(work_buf)),
      b_fft_buf(std::move(b_fft_buf)),
      input_copy(std::move(input_copy)) {
}

std::string CompiledRawRaderNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRawRader(n=" << length << ", conv_length=" << conv_length
      << ", prepare_kernel=" << (prepare_kernel ? prepare_kernel->execution_description() : "null")
      << ", pointwise_kernel=" << (pointwise_kernel ? pointwise_kernel->execution_description() : "null")
      << ", finalize_kernel=" << (finalize_kernel ? finalize_kernel->execution_description() : "null")
      << ", fused_leaf_kernel=" << (fused_leaf_kernel ? fused_leaf_kernel->execution_description() : "null")
      << ", boundary_prepare_kernel=" << (boundary_prepare_kernel ? boundary_prepare_kernel->execution_description() : "null")
      << ", boundary_finish_kernel=" << (boundary_finish_kernel ? boundary_finish_kernel->execution_description() : "null")
      << ", fft=" << (fft ? fft->describe() : "null") << ")";
  return oss.str();
}

void CompiledRawRaderNode::ensure_b_fft(const RawExecutionContext &context) const {
  std::lock_guard<std::mutex> lock(b_fft_mutex);
  if (b_fft_ready) {
    return;
  }
  RawExecutionContext child_context {context.request, context.stream, 1};
  flagfftResult result = fft->execute(b_time.get(), b_fft_buf.get(), child_context);
  if (result != FLAGFFT_SUCCESS) {
    throw std::runtime_error("failed to precompute Rader convolution FFT");
  }
  b_fft_ready = true;
}

flagfftResult CompiledRawRaderNode::execute(adaptor::DevicePtr input,
                                            adaptor::DevicePtr output,
                                            const RawExecutionContext &context) const {
  try {
    ensure_b_fft(context);
    adaptor::DevicePtr effective_input = input;
    if (input == output) {
      adaptor::copy_device_to_device(input_copy.get(), input, input_copy.size(), context.stream);
      effective_input = input_copy.get();
    }
    if (fused_leaf_kernel) {
      std::vector<JitKernelArg> args = raw_kernel_args(
          {effective_input, b_fft_buf.get(), idx.get(), output}, fused_leaf_tables, context.batch);
      fused_leaf_kernel->launch(context.stream, args,
                                ceil_div(context.batch, fused_leaf_kernel->batch_per_block), 1, 1);
      return FLAGFFT_SUCCESS;
    }
    if (boundary_prepare_kernel && boundary_finish_kernel) {
      std::vector<JitKernelArg> prepare_args = raw_kernel_args(
          {effective_input, idx.get(), a_buf.get()}, boundary_tables, context.batch);
      boundary_prepare_kernel->launch(
          context.stream, prepare_args,
          ceil_div(context.batch, boundary_prepare_kernel->batch_per_block), 1, 1);
      std::vector<JitKernelArg> finish_args = raw_kernel_args(
          {a_buf.get(), b_fft_buf.get(), idx.get(), effective_input, output},
          boundary_tables, context.batch);
      boundary_finish_kernel->launch(
          context.stream, finish_args,
          ceil_div(context.batch, boundary_finish_kernel->batch_per_block), 1, 1);
      return FLAGFFT_SUCCESS;
    }

    std::vector<JitKernelArg> prepare_args = {
        JitKernelArg::device(effective_input),
        JitKernelArg::device(idx.get()),
        JitKernelArg::device(a_buf.get()),
        JitKernelArg::i64(length),
        JitKernelArg::i64(conv_length),
        JitKernelArg::i32(static_cast<int32_t>(context.batch)),
    };
    prepare_kernel->launch(context.stream, prepare_args, ceil_div(conv_length, 256), context.batch, 1);

    RawExecutionContext child_context {context.request, context.stream, context.batch};
    flagfftResult result = fft->execute(a_buf.get(), work_buf.get(), child_context);
    if (result != FLAGFFT_SUCCESS) {
      return result;
    }

    std::vector<JitKernelArg> pointwise_args = {
        JitKernelArg::device(work_buf.get()),
        JitKernelArg::device(b_fft_buf.get()),
        JitKernelArg::device(a_buf.get()),
        JitKernelArg::device(effective_input),
        JitKernelArg::device(output),
        JitKernelArg::i64(length),
        JitKernelArg::i64(conv_length),
        JitKernelArg::i32(static_cast<int32_t>(context.batch)),
    };
    pointwise_kernel->launch(context.stream, pointwise_args, ceil_div(conv_length, 256), context.batch, 1);

    result = fft->execute(a_buf.get(), work_buf.get(), child_context);
    if (result != FLAGFFT_SUCCESS) {
      return result;
    }

    std::vector<JitKernelArg> finalize_args = {
        JitKernelArg::device(effective_input),
        JitKernelArg::device(work_buf.get()),
        JitKernelArg::device(idx.get()),
        JitKernelArg::device(output),
        JitKernelArg::i64(length),
        JitKernelArg::i64(conv_length),
        JitKernelArg::i32(static_cast<int32_t>(context.batch)),
    };
    finalize_kernel->launch(context.stream, finalize_args, ceil_div(conv_length, 256), context.batch, 1);
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] Rader execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRawFourStepGenericNode::CompiledRawFourStepGenericNode(
    int64_t length,
    int64_t n1,
    int64_t n2,
    std::shared_ptr<CompiledRawNode> row_child,
    std::shared_ptr<CompiledRawNode> col_child,
    std::shared_ptr<JitKernel> reshape_in_kernel,
    std::shared_ptr<JitKernel> twiddle_reshape_kernel,
    std::shared_ptr<JitKernel> final_pack_kernel,
    DeviceAllocation twiddle,
    DeviceAllocation stage1,
    DeviceAllocation stage2)
    : length(length),
      n1(n1),
      n2(n2),
      row_child(std::move(row_child)),
      col_child(std::move(col_child)),
      reshape_in_kernel(std::move(reshape_in_kernel)),
      twiddle_reshape_kernel(std::move(twiddle_reshape_kernel)),
      final_pack_kernel(std::move(final_pack_kernel)),
      twiddle(std::move(twiddle)),
      stage1(std::move(stage1)),
      stage2(std::move(stage2)) {
}

std::string CompiledRawFourStepGenericNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRawFourStepGeneric(n=" << length << ", n1=" << n1 << ", n2=" << n2
      << ", row_child=" << (row_child ? row_child->describe() : "null")
      << ", col_child=" << (col_child ? col_child->describe() : "null") << ")";
  return oss.str();
}

flagfftResult CompiledRawFourStepGenericNode::execute(adaptor::DevicePtr input,
                                                      adaptor::DevicePtr output,
                                                      const RawExecutionContext &context) const {
  try {
    const int64_t total = n1 * n2;
    const int64_t reshape_block = 256;
    const int64_t grid_x = ceil_div(total, reshape_block);
    int64_t batch_chunk = context.batch;
    if (context.request.device_type == "npu") {
      // The input, twiddle and final layout kernels use grid_y=batch. Chunk
      // large 3D axis batches so grid_x*grid_y stays under the ACL block cap.
      batch_chunk = std::max<int64_t>(
          1, std::min(context.batch, block_limit_per_launch() / std::max<int64_t>(1, grid_x)));
    }
    const int64_t element_bytes = complex_element_bytes(context.request.input_dtype);
    for (int64_t batch_offset = 0; batch_offset < context.batch; batch_offset += batch_chunk) {
      const int64_t chunk_batch = std::min(batch_chunk, context.batch - batch_offset);
      const int64_t byte_offset = batch_offset * total * element_bytes;
      const adaptor::DevicePtr input_chunk = input + byte_offset;
      const adaptor::DevicePtr output_chunk = output + byte_offset;
      const adaptor::DevicePtr stage1_chunk = stage1.get() + byte_offset;
      const adaptor::DevicePtr stage2_chunk = stage2.get() + byte_offset;

      std::vector<JitKernelArg> reshape_in_args = {
          JitKernelArg::device(input_chunk),
          JitKernelArg::device(stage1_chunk),
          JitKernelArg::i32(static_cast<int32_t>(chunk_batch)),
      };
      reshape_in_kernel->launch(context.stream, reshape_in_args, grid_x, chunk_batch, 1);

      RawExecutionContext row_context {context.request, context.stream, chunk_batch * n2};
      flagfftResult result = row_child->execute(stage1_chunk, stage2_chunk, row_context);
      if (result != FLAGFFT_SUCCESS) {
        return result;
      }

      std::vector<JitKernelArg> twiddle_args = {
          JitKernelArg::device(stage2_chunk),
          JitKernelArg::device(twiddle.get()),
          JitKernelArg::device(stage1_chunk),
          JitKernelArg::i32(static_cast<int32_t>(chunk_batch)),
      };
      twiddle_reshape_kernel->launch(context.stream, twiddle_args, grid_x, chunk_batch, 1);

      RawExecutionContext col_context {context.request, context.stream, chunk_batch * n1};
      result = col_child->execute(stage1_chunk, stage2_chunk, col_context);
      if (result != FLAGFFT_SUCCESS) {
        return result;
      }

      std::vector<JitKernelArg> final_args = {
          JitKernelArg::device(stage2_chunk),
          JitKernelArg::device(output_chunk),
          JitKernelArg::i32(static_cast<int32_t>(chunk_batch)),
      };
      final_pack_kernel->launch(context.stream, final_args, grid_x, chunk_batch, 1);
    }
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] FourStepGeneric execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRawR2CNode::CompiledRawR2CNode(int64_t length,
                                       std::shared_ptr<JitKernel> expand_kernel,
                                       std::shared_ptr<CompiledRawNode> fft,
                                       std::shared_ptr<JitKernel> pack_kernel,
                                       DeviceAllocation complex_input,
                                       DeviceAllocation full_output)
    : length(length),
      expand_kernel(std::move(expand_kernel)),
      fft(std::move(fft)),
      pack_kernel(std::move(pack_kernel)),
      complex_input(std::move(complex_input)),
      full_output(std::move(full_output)) {
}

std::string CompiledRawR2CNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRawR2C(n=" << length
      << ", expand_kernel=" << (expand_kernel ? expand_kernel->execution_description() : "null")
      << ", fft=" << (fft ? fft->describe() : "null")
      << ", pack_kernel=" << (pack_kernel ? pack_kernel->execution_description() : "null") << ")";
  return oss.str();
}

flagfftResult CompiledRawR2CNode::execute(adaptor::DevicePtr input,
                                          adaptor::DevicePtr output,
                                          const RawExecutionContext &context) const {
  try {
    constexpr int64_t block = 256;
    const int64_t half = length / 2 + 1;
    const bool in_place = input == output;
    const int64_t padded_real_distance = 2 * half;
    const int64_t input_distance = in_place ? std::max(context.input_distance, padded_real_distance)
                                            : (context.input_distance > 0 ? context.input_distance : length);
    const int64_t output_distance = context.output_distance > 0 ? context.output_distance : half;
    const int64_t complex_bytes = complex_element_bytes(context.request.input_dtype);
    const int64_t real_bytes = complex_bytes / 2;
    launch_grid_y_chunks(
        ceil_div(length, block),
        context.batch,
        expand_kernel->rows_per_block,
        [&](int64_t row_offset, int64_t chunk_rows) {
          std::vector<JitKernelArg> expand_args = {
              JitKernelArg::device(input + row_offset * input_distance * real_bytes),
              JitKernelArg::device(complex_input.get() + row_offset * length * complex_bytes),
              JitKernelArg::i64(input_distance),
              JitKernelArg::i32(static_cast<int32_t>(chunk_rows)),
          };
          expand_kernel->launch(context.stream,
                                expand_args,
                                ceil_div(length, block),
                                grid_rows(expand_kernel, chunk_rows),
                                1);
        });

    flagfftResult result = fft->execute(complex_input.get(), full_output.get(), context);
    if (result != FLAGFFT_SUCCESS) {
      return result;
    }

    launch_grid_y_chunks(ceil_div(length / 2 + 1, block),
                         context.batch,
                         pack_kernel->rows_per_block,
                         [&](int64_t row_offset, int64_t chunk_rows) {
                           std::vector<JitKernelArg> pack_args = {
                               JitKernelArg::device(full_output.get() + row_offset * length * complex_bytes),
                               JitKernelArg::device(output + row_offset * output_distance * real_bytes),
                               JitKernelArg::i64(output_distance),
                               JitKernelArg::i32(static_cast<int32_t>(chunk_rows)),
                           };
                           pack_kernel->launch(context.stream,
                                               pack_args,
                                               ceil_div(length / 2 + 1, block),
                                               grid_rows(pack_kernel, chunk_rows),
                                               1);
                         });
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] R2C execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRawPackedR2CNode::CompiledRawPackedR2CNode(
    int64_t length,
    std::shared_ptr<CompiledRawNode> fft,
    std::shared_ptr<JitKernel> postprocess_kernel,
    DeviceAllocation twiddle,
    DeviceAllocation packed_output,
    std::function<std::shared_ptr<CompiledRawNode>()> make_layout_fallback)
    : length(length),
      fft(std::move(fft)),
      postprocess_kernel(std::move(postprocess_kernel)),
      twiddle(std::move(twiddle)),
      packed_output(std::move(packed_output)),
      make_layout_fallback(std::move(make_layout_fallback)) {
}

std::string CompiledRawPackedR2CNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRawPackedR2C(n=" << length << ", packed_n=" << length / 2
      << ", fft=" << (fft ? fft->describe() : "null") << ", postprocess_kernel="
      << (postprocess_kernel ? postprocess_kernel->execution_description() : "null") << ")";
  return oss.str();
}

flagfftResult CompiledRawPackedR2CNode::execute(adaptor::DevicePtr input,
                                                adaptor::DevicePtr output,
                                                const RawExecutionContext &context) const {
  try {
    constexpr int64_t block = 256;
    const int64_t packed = length / 2;
    const int64_t half = packed + 1;
    const int64_t output_distance = context.output_distance > 0 ? context.output_distance : half;
    // Reinterpreting adjacent real pairs as complex values requires a dense
    // input batch. In-place real rows carry Nyquist padding; preserve their
    // original distance-aware implementation.
    if (context.batch > 1 &&
        (input == output || (context.input_distance > 0 && context.input_distance != length))) {
      std::lock_guard<std::mutex> lock(layout_mutex);
      if (!layout_fallback && make_layout_fallback) layout_fallback = make_layout_fallback();
      if (!layout_fallback) return FLAGFFT_INVALID_VALUE;
      return layout_fallback->execute(input, output, context);
    }

    RawExecutionContext child_context {context.request, context.stream, context.batch};
    flagfftResult result = fft->execute(input, packed_output.get(), child_context);
    if (result != FLAGFFT_SUCCESS) {
      return result;
    }

    std::vector<JitKernelArg> args = {
        JitKernelArg::device(packed_output.get()),
        JitKernelArg::device(twiddle.get()),
        JitKernelArg::device(output),
        JitKernelArg::i64(output_distance),
        JitKernelArg::i32(static_cast<int32_t>(context.batch)),
    };
    postprocess_kernel->launch(context.stream,
                               args,
                               ceil_div(half, block),
                               grid_rows(postprocess_kernel, context.batch),
                               1);
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] PackedR2C execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRawR2CLeafNode::CompiledRawR2CLeafNode(int64_t length,
                                               std::shared_ptr<JitKernel> kernel,
                                               std::vector<DeviceAllocation> tables,
                                               DeviceAllocation twiddle)
    : length(length), kernel(std::move(kernel)), tables(std::move(tables)), twiddle(std::move(twiddle)) {
}

std::string CompiledRawR2CLeafNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRawR2CLeaf(n=" << length
      << ", kernel=" << (kernel ? kernel->execution_description() : "null")
      << ", num_warps=" << (kernel ? kernel->num_warps : 0)
      << ", module=" << (kernel ? kernel->module_path : "null") << ", tables=" << tables.size() << ")";
  return oss.str();
}

flagfftResult CompiledRawR2CLeafNode::execute(adaptor::DevicePtr input,
                                              adaptor::DevicePtr output,
                                              const RawExecutionContext &context) const {
  try {
    const int64_t half = length / 2 + 1;
    const bool in_place = input == output;
    const int64_t padded_real_distance = 2 * half;
    const int64_t input_distance = in_place ? std::max(context.input_distance, padded_real_distance)
                                            : (context.input_distance > 0 ? context.input_distance : length);
    const int64_t output_distance = context.output_distance > 0 ? context.output_distance : half;
    const bool graph_enabled = ix_ct_batch_graph_enabled(context, length);
    if (graph_enabled && replay_leaf_graph(graph_state, context, input, output,
                                           input_distance, output_distance)) {
      return FLAGFFT_SUCCESS;
    }

    std::vector<JitKernelArg> args;
    args.reserve(3 + tables.size() + 3);
    args.push_back(JitKernelArg::device(input));
    args.push_back(JitKernelArg::device(output));
    if (twiddle.get()) args.push_back(JitKernelArg::device(twiddle.get()));
    for (const DeviceAllocation &table : tables) {
      args.push_back(JitKernelArg::device(table.get()));
    }
    args.push_back(JitKernelArg::i64(input_distance));
    args.push_back(JitKernelArg::i64(output_distance));
    args.push_back(JitKernelArg::i32(static_cast<int32_t>(context.batch)));
    auto launch = [&]() {
      kernel->launch(context.stream, args, ceil_div(context.batch, kernel->batch_per_block), 1, 1);
    };
    launch();
    if (graph_enabled) capture_leaf_graph(graph_state, context, input, output,
                                          input_distance, output_distance, launch);
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] R2CLeaf execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRawR2CFourStepHalfOutNode::CompiledRawR2CFourStepHalfOutNode(int64_t length,
                                                                     int64_t n1,
                                                                     int64_t n2,
                                                                     std::shared_ptr<JitKernel> expand_kernel,
                                                                     std::shared_ptr<JitKernel> row_kernel,
                                                                     std::vector<DeviceAllocation> row_tables,
                                                                     std::shared_ptr<JitKernel> col_kernel,
                                                                     std::vector<DeviceAllocation> col_tables,
                                                                     DeviceAllocation twiddle,
                                                                     DeviceAllocation complex_input,
                                                                     DeviceAllocation stage1)
    : length(length),
      n1(n1),
      n2(n2),
      expand_kernel(std::move(expand_kernel)),
      row_kernel(std::move(row_kernel)),
      row_tables(std::move(row_tables)),
      col_kernel(std::move(col_kernel)),
      col_tables(std::move(col_tables)),
      twiddle(std::move(twiddle)),
      complex_input(std::move(complex_input)),
      stage1(std::move(stage1)) {
}

std::string CompiledRawR2CFourStepHalfOutNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRawR2CFourStepHalfOut(n=" << length << ", n1=" << n1 << ", n2=" << n2
      << ", expand_kernel=" << (expand_kernel ? expand_kernel->execution_description() : "null")
      << ", row_kernel=" << (row_kernel ? row_kernel->execution_description() : "null")
      << ", col_kernel=" << (col_kernel ? col_kernel->execution_description() : "null") << ")";
  return oss.str();
}

flagfftResult CompiledRawR2CFourStepHalfOutNode::execute(adaptor::DevicePtr input,
                                                         adaptor::DevicePtr output,
                                                         const RawExecutionContext &context) const {
  try {
    constexpr int64_t block = 256;
    const int64_t half = length / 2 + 1;
    const bool in_place = input == output;
    const int64_t padded_real_distance = 2 * half;
    const int64_t input_distance = in_place ? std::max(context.input_distance, padded_real_distance)
                                            : (context.input_distance > 0 ? context.input_distance : length);
    const int64_t output_distance = context.output_distance > 0 ? context.output_distance : half;

    std::vector<JitKernelArg> expand_args = {
        JitKernelArg::device(input),
        JitKernelArg::device(complex_input.get()),
        JitKernelArg::i64(input_distance),
        JitKernelArg::i32(static_cast<int32_t>(context.batch)),
    };
    expand_kernel->launch(context.stream,
                          expand_args,
                          ceil_div(length, block),
                          grid_rows(expand_kernel, context.batch),
                          1);

    const bool fused_twiddle = row_kernel->tle_fused_twiddle;
    std::vector<JitKernelArg> row_args =
        fused_twiddle
            ? raw_kernel_args({complex_input.get(), twiddle.get(), stage1.get()}, row_tables, context.batch)
            : raw_kernel_args({complex_input.get(), stage1.get()}, row_tables, context.batch);
    row_kernel->launch(context.stream, row_args, ceil_div(n2, row_kernel->inner_pack), context.batch, 1);

    std::vector<JitKernelArg> col_args =
        fused_twiddle
            ? raw_distance_col_kernel_args({stage1.get(), output}, col_tables, output_distance, context.batch)
            : raw_distance_col_kernel_args({stage1.get(), twiddle.get(), output},
                                           col_tables,
                                           output_distance,
                                           context.batch);
    col_kernel->launch(context.stream, col_args, ceil_div(n1, col_kernel->inner_pack), context.batch, 1);
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] R2CFourStepHalfOut execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRawR2CFourStepRealInHalfOutNode::CompiledRawR2CFourStepRealInHalfOutNode(
    int64_t length,
    int64_t n1,
    int64_t n2,
    std::shared_ptr<JitKernel> row_kernel,
    std::vector<DeviceAllocation> row_tables,
    std::shared_ptr<JitKernel> col_kernel,
    std::vector<DeviceAllocation> col_tables,
    DeviceAllocation twiddle,
    DeviceAllocation stage1)
    : length(length),
      n1(n1),
      n2(n2),
      row_kernel(std::move(row_kernel)),
      row_tables(std::move(row_tables)),
      col_kernel(std::move(col_kernel)),
      col_tables(std::move(col_tables)),
      twiddle(std::move(twiddle)),
      stage1(std::move(stage1)) {
}

std::string CompiledRawR2CFourStepRealInHalfOutNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRawR2CFourStepRealInHalfOut(n=" << length << ", n1=" << n1 << ", n2=" << n2
      << ", row_kernel=" << (row_kernel ? row_kernel->execution_description() : "null")
      << ", col_kernel=" << (col_kernel ? col_kernel->execution_description() : "null") << ")";
  return oss.str();
}

flagfftResult CompiledRawR2CFourStepRealInHalfOutNode::execute(adaptor::DevicePtr input,
                                                               adaptor::DevicePtr output,
                                                               const RawExecutionContext &context) const {
  try {
    const int64_t half = length / 2 + 1;
    const bool in_place = input == output;
    const int64_t padded_real_distance = 2 * half;
    const int64_t input_distance = in_place ? std::max(context.input_distance, padded_real_distance)
                                            : (context.input_distance > 0 ? context.input_distance : length);
    const int64_t output_distance = context.output_distance > 0 ? context.output_distance : half;

    const bool fused_twiddle = row_kernel->tle_fused_twiddle;
    std::vector<JitKernelArg> row_args =
        fused_twiddle
            ? raw_distance_col_kernel_args({input, twiddle.get(), stage1.get()},
                                           row_tables,
                                           input_distance,
                                           context.batch)
            : raw_distance_col_kernel_args({input, stage1.get()}, row_tables, input_distance, context.batch);
    row_kernel->launch(context.stream, row_args, ceil_div(n2, row_kernel->inner_pack), context.batch, 1);

    std::vector<JitKernelArg> col_args =
        fused_twiddle
            ? raw_distance_col_kernel_args({stage1.get(), output}, col_tables, output_distance, context.batch)
            : raw_distance_col_kernel_args({stage1.get(), twiddle.get(), output},
                                           col_tables,
                                           output_distance,
                                           context.batch);
    col_kernel->launch(context.stream, col_args, ceil_div(n1, col_kernel->inner_pack), context.batch, 1);
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] R2CFourStepRealInHalfOut execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRawC2RNode::CompiledRawC2RNode(int64_t length,
                                       std::shared_ptr<JitKernel> expand_kernel,
                                       std::shared_ptr<CompiledRawNode> fft,
                                       std::shared_ptr<JitKernel> pack_kernel,
                                       DeviceAllocation full_input,
                                       DeviceAllocation full_output)
    : length(length),
      expand_kernel(std::move(expand_kernel)),
      fft(std::move(fft)),
      pack_kernel(std::move(pack_kernel)),
      full_input(std::move(full_input)),
      full_output(std::move(full_output)) {
}

std::string CompiledRawC2RNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRawC2R(n=" << length
      << ", expand_kernel=" << (expand_kernel ? expand_kernel->execution_description() : "null")
      << ", fft=" << (fft ? fft->describe() : "null")
      << ", pack_kernel=" << (pack_kernel ? pack_kernel->execution_description() : "null") << ")";
  return oss.str();
}

flagfftResult CompiledRawC2RNode::execute(adaptor::DevicePtr input,
                                          adaptor::DevicePtr output,
                                          const RawExecutionContext &context) const {
  try {
    constexpr int64_t block = 256;
    const int64_t half = length / 2 + 1;
    const bool in_place = input == output;
    const int64_t padded_real_distance = 2 * half;
    const int64_t input_distance = context.input_distance > 0 ? context.input_distance : half;
    const int64_t output_distance = in_place
                                        ? std::max(context.output_distance, padded_real_distance)
                                        : (context.output_distance > 0 ? context.output_distance : length);
    const int64_t complex_bytes = complex_element_bytes(context.request.input_dtype);
    const int64_t real_bytes = complex_bytes / 2;
    launch_grid_y_chunks(ceil_div(length, block),
                         context.batch,
                         expand_kernel->rows_per_block,
                         [&](int64_t row_offset, int64_t chunk_rows) {
                           std::vector<JitKernelArg> expand_args = {
                               JitKernelArg::device(input + row_offset * input_distance * complex_bytes),
                               JitKernelArg::device(full_input.get() + row_offset * length * complex_bytes),
                               JitKernelArg::i64(input_distance),
                               JitKernelArg::i32(static_cast<int32_t>(chunk_rows)),
                           };
                           expand_kernel->launch(context.stream,
                                                 expand_args,
                                                 ceil_div(length, block),
                                                 grid_rows(expand_kernel, chunk_rows),
                                                 1);
                         });

    flagfftResult result = fft->execute(full_input.get(), full_output.get(), context);
    if (result != FLAGFFT_SUCCESS) {
      return result;
    }

    launch_grid_y_chunks(ceil_div(length, block),
                         context.batch,
                         pack_kernel->rows_per_block,
                         [&](int64_t row_offset, int64_t chunk_rows) {
                           std::vector<JitKernelArg> pack_args = {
                               JitKernelArg::device(full_output.get() + row_offset * length * complex_bytes),
                               JitKernelArg::device(output + row_offset * output_distance * real_bytes),
                               JitKernelArg::i64(output_distance),
                               JitKernelArg::i32(static_cast<int32_t>(chunk_rows)),
                           };
                           pack_kernel->launch(context.stream,
                                               pack_args,
                                               ceil_div(length, block),
                                               grid_rows(pack_kernel, chunk_rows),
                                               1);
                         });
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] C2R execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRawPackedC2RNode::CompiledRawPackedC2RNode(
    int64_t length,
    std::shared_ptr<JitKernel> preprocess_kernel,
    std::shared_ptr<CompiledRawNode> fft,
    DeviceAllocation twiddle,
    DeviceAllocation packed_input,
    std::function<std::shared_ptr<CompiledRawNode>()> make_layout_fallback)
    : length(length),
      preprocess_kernel(std::move(preprocess_kernel)),
      fft(std::move(fft)),
      twiddle(std::move(twiddle)),
      packed_input(std::move(packed_input)),
      make_layout_fallback(std::move(make_layout_fallback)) {
}

std::string CompiledRawPackedC2RNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRawPackedC2R(n=" << length << ", packed_n=" << length / 2
      << ", preprocess_kernel=" << (preprocess_kernel ? preprocess_kernel->execution_description() : "null")
      << ", fft=" << (fft ? fft->describe() : "null") << ")";
  return oss.str();
}

flagfftResult CompiledRawPackedC2RNode::execute(adaptor::DevicePtr input,
                                                adaptor::DevicePtr output,
                                                const RawExecutionContext &context) const {
  try {
    constexpr int64_t block = 256;
    const int64_t packed = length / 2;
    const int64_t half = packed + 1;
    const int64_t input_distance = context.input_distance > 0 ? context.input_distance : half;
    if (context.batch > 1 &&
        (input == output || (context.output_distance > 0 && context.output_distance != length))) {
      std::lock_guard<std::mutex> lock(layout_mutex);
      if (!layout_fallback && make_layout_fallback) layout_fallback = make_layout_fallback();
      if (!layout_fallback) return FLAGFFT_INVALID_VALUE;
      return layout_fallback->execute(input, output, context);
    }

    std::vector<JitKernelArg> args = {
        JitKernelArg::device(input),
        JitKernelArg::device(twiddle.get()),
        JitKernelArg::device(packed_input.get()),
        JitKernelArg::i64(input_distance),
        JitKernelArg::i32(static_cast<int32_t>(context.batch)),
    };
    preprocess_kernel->launch(context.stream,
                              args,
                              ceil_div(packed, block),
                              grid_rows(preprocess_kernel, context.batch),
                              1);

    RawExecutionContext child_context {context.request, context.stream, context.batch};
    return fft->execute(packed_input.get(), output, child_context);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] PackedC2R execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRaw2DNode::CompiledRaw2DNode(int64_t n0,
                                     int64_t n1,
                                     std::shared_ptr<CompiledRawNode> row_fft,
                                     std::shared_ptr<CompiledRawNode> col_fft,
                                     std::shared_ptr<JitKernel> transpose_fwd,
                                     std::shared_ptr<JitKernel> transpose_inv,
                                     DeviceAllocation temp1,
                                     DeviceAllocation temp2,
                                     bool enable_graph)
    : n0(n0),
      n1(n1),
      row_fft(std::move(row_fft)),
      col_fft(std::move(col_fft)),
      transpose_fwd(std::move(transpose_fwd)),
      transpose_inv(std::move(transpose_inv)),
      temp1(std::move(temp1)),
      temp2(std::move(temp2)),
      graph_enabled_(enable_graph) {
}

CompiledRaw2DFusedNode::CompiledRaw2DFusedNode(int64_t n0,
                                               int64_t n1,
                                               std::shared_ptr<JitKernel> kernel,
                                               DeviceAllocation tw_r,
                                               DeviceAllocation tw_i,
                                               DeviceAllocation temp)
    : n0(n0), n1(n1), kernel(std::move(kernel)),
      tw_r(std::move(tw_r)), tw_i(std::move(tw_i)), temp(std::move(temp)) {
}

std::string CompiledRaw2DFusedNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRaw2DFused(n0=" << n0 << ", n1=" << n1
      << ", kernel=" << kernel->execution_description() << ")";
  return oss.str();
}

flagfftResult CompiledRaw2DFusedNode::execute(adaptor::DevicePtr input,
                                              adaptor::DevicePtr output,
                                              const RawExecutionContext &context) const {
  try {
    std::vector<JitKernelArg> args = {
        JitKernelArg::device(input),
        JitKernelArg::device(temp.get()),
        JitKernelArg::device(tw_r.get()),
        JitKernelArg::device(tw_i.get()),
    };
    kernel->launch(context.stream, args, context.batch * n0, 1, 1);
    args[0] = JitKernelArg::device(temp.get());
    args[1] = JitKernelArg::device(output);
    kernel->launch(context.stream, args, context.batch * n1, 1, 1);
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] fused 2D execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

std::string CompiledRaw2DNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRaw2D(n0=" << n0 << ", n1=" << n1
      << ", row_fft=" << (row_fft ? row_fft->describe() : "null")
      << ", col_fft=" << (col_fft ? col_fft->describe() : "null")
      << ", transpose_fwd=" << (transpose_fwd ? transpose_fwd->execution_description() : "null")
      << ", transpose_inv=" << (transpose_inv ? transpose_inv->execution_description() : "null") << ")";
  return oss.str();
}

flagfftResult CompiledRaw2DNode::execute(adaptor::DevicePtr input,
                                         adaptor::DevicePtr output,
                                         const RawExecutionContext &context) const {
  try {
    const int64_t batch = context.batch;
    const int64_t total = n0 * n1;
    // Grid tile size for 2D transpose.  Must match the tile_size baked into
    // the compiled transpose kernel — see kernels.py:_build_tiled_transpose_kernel_source
    // (default tile_size=32) and jit_source.py:_emit_tiled_transpose_jit_kernel (ditto).
    constexpr int64_t tile_size = 32;

    if (graph_ != nullptr && graph_in_ == input && graph_out_ == output) {
      graph_->launch(context.stream);
      return FLAGFFT_SUCCESS;
    }

    auto run_sequence = [&]() -> flagfftResult {
      // Step 1: Row FFT (input -> temp1)
      RawExecutionContext row_context {context.request, context.stream, batch * n0};
      flagfftResult result = row_fft->execute(input, temp1.get(), row_context);
      if (result != FLAGFFT_SUCCESS) {
        return result;
      }

      // Step 2: Transpose (temp1 -> temp2)
      std::vector<JitKernelArg> transpose_fwd_args = {
          JitKernelArg::device(temp1.get()),
          JitKernelArg::device(temp2.get()),
          JitKernelArg::i32(static_cast<int32_t>(batch)),
      };
      transpose_fwd->launch(context.stream,
                            transpose_fwd_args,
                            ceil_div(n1, tile_size),
                            ceil_div(n0, tile_size),
                            batch);

      // Step 3: Col FFT (temp2 -> temp1)
      RawExecutionContext col_context {context.request, context.stream, batch * n1};
      result = col_fft->execute(temp2.get(), temp1.get(), col_context);
      if (result != FLAGFFT_SUCCESS) {
        return result;
      }

      // Step 4: Transpose back (temp1 -> output)
      std::vector<JitKernelArg> transpose_inv_args = {
          JitKernelArg::device(temp1.get()),
          JitKernelArg::device(output),
          JitKernelArg::i32(static_cast<int32_t>(batch)),
      };
      transpose_inv->launch(context.stream,
                            transpose_inv_args,
                            ceil_div(n0, tile_size),
                            ceil_div(n1, tile_size),
                            batch);
      return FLAGFFT_SUCCESS;
    };

    flagfftResult result = run_sequence();
    if (result != FLAGFFT_SUCCESS) {
      return result;
    }

    // Build a replay graph for stable input/output pointers.  Kernels are
    // already compiled by the direct run above, so capture only records
    // launches.  Any capture/instantiation failure falls back to direct
    // launches for the lifetime of the plan.
    if (graph_enabled_ && graph_ == nullptr && !graph_failed_) {
      try {
        auto graph = std::make_unique<adaptor::CudaGraph>();
        graph->begin_capture(context.stream);
        run_sequence();
        graph->end_capture(context.stream);
        graph->launch(context.stream);
        graph_ = std::move(graph);
        graph_in_ = input;
        graph_out_ = output;
      } catch (const std::exception &) {
        graph_failed_ = true;
      }
    }

    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] 2D execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRaw2DRCNode::CompiledRaw2DRCNode(int64_t n0,
                                         int64_t n1,
                                         std::shared_ptr<CompiledRawNode> row_fft,
                                         std::shared_ptr<CompiledRawNode> col_fft,
                                         DeviceAllocation temp1,
                                         bool enable_graph)
    : n0(n0),
      n1(n1),
      row_fft(std::move(row_fft)),
      col_fft(std::move(col_fft)),
      temp1(std::move(temp1)),
      graph_enabled_(enable_graph) {
}

std::string CompiledRaw2DRCNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRaw2DRC(n0=" << n0 << ", n1=" << n1
      << ", row_fft=" << (row_fft ? row_fft->describe() : "null")
      << ", col_fft=" << (col_fft ? col_fft->describe() : "null") << ")";
  return oss.str();
}

flagfftResult CompiledRaw2DRCNode::execute(adaptor::DevicePtr input,
                                           adaptor::DevicePtr output,
                                           const RawExecutionContext &context) const {
  try {
    const int64_t batch = context.batch;

    if (graph_ != nullptr && graph_in_ == input && graph_out_ == output) {
      graph_->launch(context.stream);
      return FLAGFFT_SUCCESS;
    }

    auto run_sequence = [&]() -> flagfftResult {
      // Step 1: Transform the contiguous input axis into temp1. Some compiled
      // paths store the result transposed so the second pass can also read
      // contiguous data.
      RawExecutionContext row_context {context.request, context.stream, batch * n0};
      flagfftResult result = row_fft->execute(input, temp1.get(), row_context);
      if (result != FLAGFFT_SUCCESS) {
        return result;
      }

      // Step 2: Transform the second axis and let its compiled output layout
      // produce the natural (batch, n0, n1) result. This is strided for the
      // portable RC path and contiguous for the transposed Cube path.
      RawExecutionContext col_context {context.request, context.stream, batch * n1};
      result = col_fft->execute(temp1.get(), output, col_context);
      return result;
    };

    flagfftResult result = run_sequence();
    if (result != FLAGFFT_SUCCESS) {
      return result;
    }

    if (graph_enabled_ && graph_ == nullptr && !graph_failed_) {
      try {
        auto graph = std::make_unique<adaptor::CudaGraph>();
        graph->begin_capture(context.stream);
        run_sequence();
        graph->end_capture(context.stream);
        graph->launch(context.stream);
        graph_ = std::move(graph);
        graph_in_ = input;
        graph_out_ = output;
      } catch (const std::exception &) {
        graph_failed_ = true;
      }
    }

    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] 2D RC execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRaw1DAs2DNode::CompiledRaw1DAs2DNode(std::shared_ptr<CompiledRawNode> fft, int64_t batch)
    : fft(std::move(fft)), batch(batch) {
}

std::string CompiledRaw1DAs2DNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRaw1DAs2D(batch=" << batch << ", fft=" << (fft ? fft->describe() : "null") << ")";
  return oss.str();
}

flagfftResult CompiledRaw1DAs2DNode::execute(adaptor::DevicePtr input,
                                             adaptor::DevicePtr output,
                                             const RawExecutionContext &context) const {
  RawExecutionContext child_context {context.request, context.stream, batch};
  return fft->execute(input, output, child_context);
}

CompiledRawC2RLeafNode::CompiledRawC2RLeafNode(int64_t length,
                                               std::shared_ptr<JitKernel> kernel,
                                               std::vector<DeviceAllocation> tables)
    : length(length), kernel(std::move(kernel)), tables(std::move(tables)) {
}

std::string CompiledRawC2RLeafNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRawC2RLeaf(n=" << length
      << ", kernel=" << (kernel ? kernel->execution_description() : "null")
      << ", num_warps=" << (kernel ? kernel->num_warps : 0)
      << ", module=" << (kernel ? kernel->module_path : "null") << ", tables=" << tables.size() << ")";
  return oss.str();
}

flagfftResult CompiledRawC2RLeafNode::execute(adaptor::DevicePtr input,
                                              adaptor::DevicePtr output,
                                              const RawExecutionContext &context) const {
  try {
    const int64_t half = length / 2 + 1;
    const bool in_place = input == output;
    const int64_t padded_real_distance = 2 * half;
    const int64_t input_distance = context.input_distance > 0 ? context.input_distance : half;
    const int64_t output_distance = in_place
                                        ? std::max(context.output_distance, padded_real_distance)
                                        : (context.output_distance > 0 ? context.output_distance : length);
    const bool graph_enabled = ix_ct_batch_graph_enabled(context, length);
    if (graph_enabled && replay_leaf_graph(graph_state, context, input, output,
                                           input_distance, output_distance)) {
      return FLAGFFT_SUCCESS;
    }

    std::vector<JitKernelArg> args;
    args.reserve(2 + tables.size() + 3);
    args.push_back(JitKernelArg::device(input));
    args.push_back(JitKernelArg::device(output));
    for (const DeviceAllocation &table : tables) {
      args.push_back(JitKernelArg::device(table.get()));
    }
    args.push_back(JitKernelArg::i64(input_distance));
    args.push_back(JitKernelArg::i64(output_distance));
    args.push_back(JitKernelArg::i32(static_cast<int32_t>(context.batch)));
    auto launch = [&]() {
      kernel->launch(context.stream, args, ceil_div(context.batch, kernel->batch_per_block), 1, 1);
    };
    launch();
    if (graph_enabled) capture_leaf_graph(graph_state, context, input, output,
                                          input_distance, output_distance, launch);
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] C2RLeaf execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRaw2DR2CRowNode::CompiledRaw2DR2CRowNode(int64_t n0,
                                                 int64_t n1,
                                                 std::shared_ptr<CompiledRawNode> row_r2c,
                                                 std::shared_ptr<CompiledRawNode> col_fft,
                                                 std::shared_ptr<JitKernel> transpose_fwd,
                                                 std::shared_ptr<JitKernel> transpose_inv,
                                                 DeviceAllocation temp1,
                                                 DeviceAllocation temp2)
    : n0(n0),
      n1(n1),
      row_r2c(std::move(row_r2c)),
      col_fft(std::move(col_fft)),
      transpose_fwd(std::move(transpose_fwd)),
      transpose_inv(std::move(transpose_inv)),
      temp1(std::move(temp1)),
      temp2(std::move(temp2)) {
}

std::string CompiledRaw2DR2CRowNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRaw2DR2CRow(n0=" << n0 << ", n1=" << n1
      << ", row_r2c=" << (row_r2c ? row_r2c->describe() : "null")
      << ", col_fft=" << (col_fft ? col_fft->describe() : "null")
      << ", transpose_fwd=" << (transpose_fwd ? transpose_fwd->execution_description() : "null")
      << ", transpose_inv=" << (transpose_inv ? transpose_inv->execution_description() : "null") << ")";
  return oss.str();
}

flagfftResult CompiledRaw2DR2CRowNode::execute(adaptor::DevicePtr input,
                                               adaptor::DevicePtr output,
                                               const RawExecutionContext &context) const {
  try {
    constexpr int64_t tile_size = 32;
    const int64_t batch = context.batch;
    const int64_t half_n1 = n1 / 2 + 1;

    // The row child writes to scratch rather than directly to the user output.
    // This preserves exact in-place safety: all real input is consumed before
    // the final transpose writes the compact output back to `output`.
    RawExecutionContext row_context {context.request, context.stream, batch * n0, n1, half_n1};
    flagfftResult result = row_r2c->execute(input, temp1.get(), row_context);
    if (result != FLAGFFT_SUCCESS) {
      return result;
    }

    std::vector<JitKernelArg> transpose_fwd_args = {
        JitKernelArg::device(temp1.get()),
        JitKernelArg::device(temp2.get()),
        JitKernelArg::i32(static_cast<int32_t>(batch)),
    };
    transpose_fwd->launch(context.stream,
                          transpose_fwd_args,
                          ceil_div(half_n1, tile_size),
                          ceil_div(n0, tile_size),
                          batch);

    RawExecutionContext col_context {context.request, context.stream, batch * half_n1, n0, n0};
    result = col_fft->execute(temp2.get(), temp1.get(), col_context);
    if (result != FLAGFFT_SUCCESS) {
      return result;
    }

    std::vector<JitKernelArg> transpose_inv_args = {
        JitKernelArg::device(temp1.get()),
        JitKernelArg::device(output),
        JitKernelArg::i32(static_cast<int32_t>(batch)),
    };
    transpose_inv->launch(context.stream,
                          transpose_inv_args,
                          ceil_div(n0, tile_size),
                          ceil_div(half_n1, tile_size),
                          batch);
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] 2D R2C row-boundary execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRaw2DC2RRowNode::CompiledRaw2DC2RRowNode(int64_t n0,
                                                 int64_t n1,
                                                 std::shared_ptr<CompiledRawNode> col_fft,
                                                 std::shared_ptr<CompiledRawNode> row_c2r,
                                                 std::shared_ptr<JitKernel> transpose_fwd,
                                                 std::shared_ptr<JitKernel> transpose_inv,
                                                 DeviceAllocation temp1,
                                                 DeviceAllocation temp2)
    : n0(n0),
      n1(n1),
      col_fft(std::move(col_fft)),
      row_c2r(std::move(row_c2r)),
      transpose_fwd(std::move(transpose_fwd)),
      transpose_inv(std::move(transpose_inv)),
      temp1(std::move(temp1)),
      temp2(std::move(temp2)) {
}

std::string CompiledRaw2DC2RRowNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRaw2DC2RRow(n0=" << n0 << ", n1=" << n1
      << ", col_fft=" << (col_fft ? col_fft->describe() : "null")
      << ", row_c2r=" << (row_c2r ? row_c2r->describe() : "null")
      << ", transpose_fwd=" << (transpose_fwd ? transpose_fwd->execution_description() : "null")
      << ", transpose_inv=" << (transpose_inv ? transpose_inv->execution_description() : "null") << ")";
  return oss.str();
}

flagfftResult CompiledRaw2DC2RRowNode::execute(adaptor::DevicePtr input,
                                               adaptor::DevicePtr output,
                                               const RawExecutionContext &context) const {
  try {
    constexpr int64_t tile_size = 32;
    const int64_t batch = context.batch;
    const int64_t half_n1 = n1 / 2 + 1;

    std::vector<JitKernelArg> transpose_fwd_args = {
        JitKernelArg::device(input),
        JitKernelArg::device(temp1.get()),
        JitKernelArg::i32(static_cast<int32_t>(batch)),
    };
    transpose_fwd->launch(context.stream,
                          transpose_fwd_args,
                          ceil_div(half_n1, tile_size),
                          ceil_div(n0, tile_size),
                          batch);

    RawExecutionContext col_context {context.request, context.stream, batch * half_n1, n0, n0};
    flagfftResult result = col_fft->execute(temp1.get(), temp2.get(), col_context);
    if (result != FLAGFFT_SUCCESS) {
      return result;
    }

    std::vector<JitKernelArg> transpose_inv_args = {
        JitKernelArg::device(temp2.get()),
        JitKernelArg::device(temp1.get()),
        JitKernelArg::i32(static_cast<int32_t>(batch)),
    };
    transpose_inv->launch(context.stream,
                          transpose_inv_args,
                          ceil_div(n0, tile_size),
                          ceil_div(half_n1, tile_size),
                          batch);

    // The row child is the final stage and writes real output only after the
    // compact input has been consumed, so exact in-place C2R is safe here.
    RawExecutionContext row_context {context.request, context.stream, batch * n0, half_n1, n1};
    return row_c2r->execute(temp1.get(), output, row_context);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] 2D C2R row-boundary execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRaw2DR2CNode::CompiledRaw2DR2CNode(int64_t n0,
                                           int64_t n1,
                                           std::shared_ptr<JitKernel> expand_kernel,
                                           std::shared_ptr<CompiledRawNode> row_fft,
                                           std::shared_ptr<JitKernel> pack_kernel,
                                           std::shared_ptr<CompiledRawNode> col_fft,
                                           std::shared_ptr<JitKernel> transpose_fwd,
                                           std::shared_ptr<JitKernel> transpose_inv,
                                           DeviceAllocation row_fft_buf,
                                           DeviceAllocation temp1,
                                           DeviceAllocation temp2)
    : n0(n0),
      n1(n1),
      expand_kernel(std::move(expand_kernel)),
      row_fft(std::move(row_fft)),
      pack_kernel(std::move(pack_kernel)),
      col_fft(std::move(col_fft)),
      transpose_fwd(std::move(transpose_fwd)),
      transpose_inv(std::move(transpose_inv)),
      row_fft_buf(std::move(row_fft_buf)),
      temp1(std::move(temp1)),
      temp2(std::move(temp2)) {
}

std::string CompiledRaw2DR2CNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRaw2DR2C(n0=" << n0 << ", n1=" << n1
      << ", expand_kernel=" << (expand_kernel ? expand_kernel->execution_description() : "null")
      << ", row_fft=" << (row_fft ? row_fft->describe() : "null")
      << ", pack_kernel=" << (pack_kernel ? pack_kernel->execution_description() : "null")
      << ", col_fft=" << (col_fft ? col_fft->describe() : "null")
      << ", transpose_fwd=" << (transpose_fwd ? transpose_fwd->execution_description() : "null")
      << ", transpose_inv=" << (transpose_inv ? transpose_inv->execution_description() : "null") << ")";
  return oss.str();
}

flagfftResult CompiledRaw2DR2CNode::execute(adaptor::DevicePtr input,
                                            adaptor::DevicePtr output,
                                            const RawExecutionContext &context) const {
  try {
    const int64_t batch = context.batch;
    constexpr int64_t block = 256;
    constexpr int64_t tile_size = 32;
    const int64_t half_n1 = n1 / 2 + 1;

    // Step 1: Expand real input to complex
    // Input: (batch*n0, n1) real -> row_fft_buf: (batch*n0, n1) complex
    // Each row is processed independently, so total rows = batch * n0
    // input_distance is per-row distance in the input buffer
    const int64_t input_distance = n1;  // Each row in input has n1 real elements
    const int64_t total_rows = batch * n0;
    std::vector<JitKernelArg> expand_args = {
        JitKernelArg::device(input),
        JitKernelArg::device(row_fft_buf.get()),
        JitKernelArg::i64(input_distance),
        JitKernelArg::i32(static_cast<int32_t>(total_rows)),
    };
    expand_kernel->launch(context.stream,
                          expand_args,
                          ceil_div(n1, block),
                          grid_rows(expand_kernel, total_rows),
                          1);

    // Step 2: Row C2C FFT
    // row_fft_buf: (batch*n0, n1) complex -> row_fft_buf: (batch*n0, n1) complex (in-place)
    RawExecutionContext row_context {context.request, context.stream, batch * n0};
    flagfftResult result = row_fft->execute(row_fft_buf.get(), row_fft_buf.get(), row_context);
    if (result != FLAGFFT_SUCCESS) {
      return result;
    }

    // Step 3: Pack half spectrum
    // row_fft_buf: (batch*n0, n1) complex -> output: (batch*n0, n1/2+1) complex
    // Each row is processed independently, so total rows = batch * n0
    // output_distance is per-row distance in the output buffer
    const int64_t output_distance = half_n1;  // Each row in output has half_n1 complex elements
    std::vector<JitKernelArg> pack_args = {
        JitKernelArg::device(row_fft_buf.get()),
        JitKernelArg::device(output),
        JitKernelArg::i64(output_distance),
        JitKernelArg::i32(static_cast<int32_t>(total_rows)),
    };
    pack_kernel->launch(context.stream,
                        pack_args,
                        ceil_div(half_n1, block),
                        grid_rows(pack_kernel, total_rows),
                        1);

    // Step 4: Transpose (n0, n1/2+1) -> (n1/2+1, n0)
    // transpose_fwd kernel is compiled for (n0, half_n1) -> (half_n1, n0)
    std::vector<JitKernelArg> transpose_fwd_args = {
        JitKernelArg::device(output),
        JitKernelArg::device(temp1.get()),
        JitKernelArg::i32(static_cast<int32_t>(batch)),
    };
    transpose_fwd->launch(context.stream,
                          transpose_fwd_args,
                          ceil_div(half_n1, tile_size),
                          ceil_div(n0, tile_size),
                          batch);

    // Step 5: Col C2C FFT
    // temp1: (n1/2+1, n0) complex -> temp2: (n1/2+1, n0) complex
    RawExecutionContext col_context {context.request, context.stream, batch * half_n1};
    result = col_fft->execute(temp1.get(), temp2.get(), col_context);
    if (result != FLAGFFT_SUCCESS) {
      return result;
    }

    // Step 6: Transpose back (n1/2+1, n0) -> (n0, n1/2+1)
    // transpose_inv kernel is compiled for (half_n1, n0) -> (n0, half_n1)
    std::vector<JitKernelArg> transpose_inv_args = {
        JitKernelArg::device(temp2.get()),
        JitKernelArg::device(output),
        JitKernelArg::i32(static_cast<int32_t>(batch)),
    };
    transpose_inv->launch(context.stream,
                          transpose_inv_args,
                          ceil_div(n0, tile_size),
                          ceil_div(half_n1, tile_size),
                          batch);

    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] 2D R2C execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRaw2DR2CRCNode::CompiledRaw2DR2CRCNode(int64_t n0,
                                               int64_t n1,
                                               std::shared_ptr<JitKernel> expand_kernel,
                                               std::shared_ptr<CompiledRawNode> row_fft,
                                               std::shared_ptr<JitKernel> pack_kernel,
                                               std::shared_ptr<CompiledRawNode> col_fft,
                                               DeviceAllocation row_fft_buf)
    : n0(n0),
      n1(n1),
      expand_kernel(std::move(expand_kernel)),
      row_fft(std::move(row_fft)),
      pack_kernel(std::move(pack_kernel)),
      col_fft(std::move(col_fft)),
      row_fft_buf(std::move(row_fft_buf)) {
}

std::string CompiledRaw2DR2CRCNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRaw2DR2CRC(n0=" << n0 << ", n1=" << n1
      << ", expand_kernel=" << (expand_kernel ? expand_kernel->execution_description() : "null")
      << ", row_fft=" << (row_fft ? row_fft->describe() : "null")
      << ", pack_kernel=" << (pack_kernel ? pack_kernel->execution_description() : "null")
      << ", col_fft=" << (col_fft ? col_fft->describe() : "null") << ")";
  return oss.str();
}

flagfftResult CompiledRaw2DR2CRCNode::execute(adaptor::DevicePtr input,
                                              adaptor::DevicePtr output,
                                              const RawExecutionContext &context) const {
  try {
    const int64_t batch = context.batch;
    constexpr int64_t block = 256;
    const int64_t half_n1 = n1 / 2 + 1;
    const int64_t total_rows = batch * n0;

    if (expand_kernel == nullptr && pack_kernel == nullptr) {
      // The row child writes the compact half spectrum into scratch. This
      // keeps exact in-place R2C safe while still avoiding full-complex
      // expansion and the separate half-spectrum pack kernel.
      RawExecutionContext row_context {context.request, context.stream, total_rows, n1, half_n1};
      flagfftResult result = row_fft->execute(input, row_fft_buf.get(), row_context);
      if (result != FLAGFFT_SUCCESS) {
        return result;
      }

      RawExecutionContext col_context {context.request, context.stream, batch * half_n1};
      return col_fft->execute(row_fft_buf.get(), output, col_context);
    }

    // Step 1: Expand real input to complex.
    std::vector<JitKernelArg> expand_args = {
        JitKernelArg::device(input),
        JitKernelArg::device(row_fft_buf.get()),
        JitKernelArg::i64(n1),
        JitKernelArg::i32(static_cast<int32_t>(total_rows)),
    };
    expand_kernel->launch(context.stream, expand_args, ceil_div(n1, block), total_rows, 1);

    // Step 2: Row C2C FFT (in-place on the expanded buffer).
    RawExecutionContext row_context {context.request, context.stream, total_rows};
    flagfftResult result = row_fft->execute(row_fft_buf.get(), row_fft_buf.get(), row_context);
    if (result != FLAGFFT_SUCCESS) {
      return result;
    }

    // Step 3: Pack the half spectrum into the user output buffer.
    std::vector<JitKernelArg> pack_args = {
        JitKernelArg::device(row_fft_buf.get()),
        JitKernelArg::device(output),
        JitKernelArg::i64(half_n1),
        JitKernelArg::i32(static_cast<int32_t>(total_rows)),
    };
    pack_kernel->launch(context.stream, pack_args, ceil_div(half_n1, block), total_rows, 1);

    // Step 4: Column C2C FFT directly on the half-packed rows (no transposes).
    RawExecutionContext col_context {context.request, context.stream, batch * half_n1};
    return col_fft->execute(output, output, col_context);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] 2D R2C RC execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRawC2RFourStepRealOutNode::CompiledRawC2RFourStepRealOutNode(int64_t length,
                                                                     int64_t n1,
                                                                     int64_t n2,
                                                                     std::shared_ptr<JitKernel> expand_kernel,
                                                                     std::shared_ptr<JitKernel> row_kernel,
                                                                     std::vector<DeviceAllocation> row_tables,
                                                                     std::shared_ptr<JitKernel> col_kernel,
                                                                     std::vector<DeviceAllocation> col_tables,
                                                                     DeviceAllocation twiddle,
                                                                     DeviceAllocation full_input,
                                                                     DeviceAllocation stage1)
    : length(length),
      n1(n1),
      n2(n2),
      expand_kernel(std::move(expand_kernel)),
      row_kernel(std::move(row_kernel)),
      row_tables(std::move(row_tables)),
      col_kernel(std::move(col_kernel)),
      col_tables(std::move(col_tables)),
      twiddle(std::move(twiddle)),
      full_input(std::move(full_input)),
      stage1(std::move(stage1)) {
}

std::string CompiledRawC2RFourStepRealOutNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRawC2RFourStepRealOut(n=" << length << ", n1=" << n1 << ", n2=" << n2
      << ", expand_kernel=" << (expand_kernel ? expand_kernel->execution_description() : "null")
      << ", row_kernel=" << (row_kernel ? row_kernel->execution_description() : "null")
      << ", col_kernel=" << (col_kernel ? col_kernel->execution_description() : "null") << ")";
  return oss.str();
}

flagfftResult CompiledRawC2RFourStepRealOutNode::execute(adaptor::DevicePtr input,
                                                         adaptor::DevicePtr output,
                                                         const RawExecutionContext &context) const {
  try {
    constexpr int64_t block = 256;
    const int64_t half = length / 2 + 1;
    const bool in_place = input == output;
    const int64_t padded_real_distance = 2 * half;
    const int64_t input_distance = context.input_distance > 0 ? context.input_distance : half;
    const int64_t output_distance = in_place
                                        ? std::max(context.output_distance, padded_real_distance)
                                        : (context.output_distance > 0 ? context.output_distance : length);

    std::vector<JitKernelArg> expand_args = {
        JitKernelArg::device(input),
        JitKernelArg::device(full_input.get()),
        JitKernelArg::i64(input_distance),
        JitKernelArg::i32(static_cast<int32_t>(context.batch)),
    };
    expand_kernel->launch(context.stream,
                          expand_args,
                          ceil_div(length, block),
                          grid_rows(expand_kernel, context.batch),
                          1);

    const bool fused_twiddle = row_kernel->tle_fused_twiddle;
    std::vector<JitKernelArg> row_args =
        fused_twiddle
            ? raw_kernel_args({full_input.get(), twiddle.get(), stage1.get()}, row_tables, context.batch)
            : raw_kernel_args({full_input.get(), stage1.get()}, row_tables, context.batch);
    row_kernel->launch(context.stream, row_args, ceil_div(n2, row_kernel->inner_pack), context.batch, 1);

    std::vector<JitKernelArg> col_args =
        fused_twiddle
            ? raw_distance_col_kernel_args({stage1.get(), output}, col_tables, output_distance, context.batch)
            : raw_distance_col_kernel_args({stage1.get(), twiddle.get(), output},
                                           col_tables,
                                           output_distance,
                                           context.batch);
    col_kernel->launch(context.stream, col_args, ceil_div(n1, col_kernel->inner_pack), context.batch, 1);
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] C2RFourStepRealOut execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRaw2DC2RNode::CompiledRaw2DC2RNode(int64_t n0,
                                           int64_t n1,
                                           std::shared_ptr<JitKernel> expand_kernel,
                                           std::shared_ptr<CompiledRawNode> col_fft,
                                           std::shared_ptr<CompiledRawNode> row_fft,
                                           std::shared_ptr<JitKernel> transpose_fwd,
                                           std::shared_ptr<JitKernel> transpose_inv,
                                           std::shared_ptr<JitKernel> pack_kernel,
                                           DeviceAllocation temp1,
                                           DeviceAllocation temp2,
                                           DeviceAllocation temp3)
    : n0(n0),
      n1(n1),
      expand_kernel(std::move(expand_kernel)),
      col_fft(std::move(col_fft)),
      row_fft(std::move(row_fft)),
      transpose_fwd(std::move(transpose_fwd)),
      transpose_inv(std::move(transpose_inv)),
      pack_kernel(std::move(pack_kernel)),
      temp1(std::move(temp1)),
      temp2(std::move(temp2)),
      temp3(std::move(temp3)) {
}

std::string CompiledRaw2DC2RNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRaw2DC2R(n0=" << n0 << ", n1=" << n1
      << ", expand_kernel=" << (expand_kernel ? expand_kernel->execution_description() : "null")
      << ", col_fft=" << (col_fft ? col_fft->describe() : "null")
      << ", row_fft=" << (row_fft ? row_fft->describe() : "null")
      << ", transpose_fwd=" << (transpose_fwd ? transpose_fwd->execution_description() : "null")
      << ", transpose_inv=" << (transpose_inv ? transpose_inv->execution_description() : "null")
      << ", pack_kernel=" << (pack_kernel ? pack_kernel->execution_description() : "null") << ")";
  return oss.str();
}

flagfftResult CompiledRaw2DC2RNode::execute(adaptor::DevicePtr input,
                                            adaptor::DevicePtr output,
                                            const RawExecutionContext &context) const {
  try {
    const int64_t batch = context.batch;
    constexpr int64_t block = 256;
    constexpr int64_t tile_size = 32;
    const int64_t half_n1 = n1 / 2 + 1;
    const int64_t total_rows = batch * n0;

    // C2R is the reverse of R2C:
    // 1. Transpose (n0, half_n1) -> (half_n1, n0)
    // 2. Col IFFT along n0 (batch = batch * half_n1)
    // 3. Transpose back (half_n1, n0) -> (n0, half_n1)
    // 4. Expand half-packed -> full Hermitian (n0, half_n1) -> (n0, n1)
    // 5. Row IFFT along n1 (batch = batch * n0)
    // 6. Pack complex -> real

    // Step 1: Transpose (n0, half_n1) -> (half_n1, n0)
    std::vector<JitKernelArg> transpose_fwd_args = {
        JitKernelArg::device(input),
        JitKernelArg::device(temp1.get()),
        JitKernelArg::i32(static_cast<int32_t>(batch)),
    };
    transpose_fwd->launch(context.stream,
                          transpose_fwd_args,
                          ceil_div(half_n1, tile_size),
                          ceil_div(n0, tile_size),
                          batch);

    // Step 2: Col C2C IFFT along n0 (batch = batch * half_n1)
    RawExecutionContext col_context {context.request, context.stream, batch * half_n1};
    flagfftResult result = col_fft->execute(temp1.get(), temp2.get(), col_context);
    if (result != FLAGFFT_SUCCESS) {
      return result;
    }

    // Step 3: Transpose back (half_n1, n0) -> (n0, half_n1)
    std::vector<JitKernelArg> transpose_inv_args = {
        JitKernelArg::device(temp2.get()),
        JitKernelArg::device(temp1.get()),
        JitKernelArg::i32(static_cast<int32_t>(batch)),
    };
    transpose_inv->launch(context.stream,
                          transpose_inv_args,
                          ceil_div(n0, tile_size),
                          ceil_div(half_n1, tile_size),
                          batch);

    // Step 4: Expand half-packed -> full Hermitian
    // temp1: (batch*n0, half_n1) complex -> temp3: (batch*n0, n1) complex
    std::vector<JitKernelArg> expand_args = {
        JitKernelArg::device(temp1.get()),
        JitKernelArg::device(temp3.get()),
        JitKernelArg::i64(half_n1),
        JitKernelArg::i32(static_cast<int32_t>(total_rows)),
    };
    expand_kernel->launch(context.stream,
                          expand_args,
                          ceil_div(n1, block),
                          grid_rows(expand_kernel, total_rows),
                          1);

    // Step 5: Row C2C IFFT along n1 (batch = batch * n0, in-place)
    RawExecutionContext row_context {context.request, context.stream, total_rows};
    result = row_fft->execute(temp3.get(), temp3.get(), row_context);
    if (result != FLAGFFT_SUCCESS) {
      return result;
    }

    // Step 6: Pack complex -> real
    // temp3: (batch*n0, n1) complex -> output: (batch*n0, n1) real
    std::vector<JitKernelArg> pack_args = {
        JitKernelArg::device(temp3.get()),
        JitKernelArg::device(output),
        JitKernelArg::i64(n1),
        JitKernelArg::i32(static_cast<int32_t>(total_rows)),
    };
    pack_kernel->launch(context.stream,
                        pack_args,
                        ceil_div(n1, block),
                        grid_rows(pack_kernel, total_rows),
                        1);

    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] 2D C2R execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRaw2DC2RRCNode::CompiledRaw2DC2RRCNode(int64_t n0,
                                               int64_t n1,
                                               std::shared_ptr<CompiledRawNode> col_fft,
                                               std::shared_ptr<JitKernel> expand_kernel,
                                               std::shared_ptr<CompiledRawNode> row_fft,
                                               std::shared_ptr<JitKernel> pack_kernel,
                                               DeviceAllocation temp_half,
                                               DeviceAllocation temp_full)
    : n0(n0),
      n1(n1),
      col_fft(std::move(col_fft)),
      expand_kernel(std::move(expand_kernel)),
      row_fft(std::move(row_fft)),
      pack_kernel(std::move(pack_kernel)),
      temp_half(std::move(temp_half)),
      temp_full(std::move(temp_full)) {
}

std::string CompiledRaw2DC2RRCNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRaw2DC2RRC(n0=" << n0 << ", n1=" << n1
      << ", col_fft=" << (col_fft ? col_fft->describe() : "null")
      << ", expand_kernel=" << (expand_kernel ? expand_kernel->execution_description() : "null")
      << ", row_fft=" << (row_fft ? row_fft->describe() : "null")
      << ", pack_kernel=" << (pack_kernel ? pack_kernel->execution_description() : "null") << ")";
  return oss.str();
}

flagfftResult CompiledRaw2DC2RRCNode::execute(adaptor::DevicePtr input,
                                              adaptor::DevicePtr output,
                                              const RawExecutionContext &context) const {
  try {
    const int64_t batch = context.batch;
    constexpr int64_t block = 256;
    const int64_t half_n1 = n1 / 2 + 1;
    const int64_t total_rows = batch * n0;

    // Step 1: Column C2C IFFT directly on the half-packed input (no transposes).
    RawExecutionContext col_context {context.request, context.stream, batch * half_n1};
    flagfftResult result = col_fft->execute(input, temp_half.get(), col_context);
    if (result != FLAGFFT_SUCCESS) {
      return result;
    }

    if (expand_kernel == nullptr && pack_kernel == nullptr) {
      // The compact column result can be consumed directly by each row's
      // existing 1D C2R boundary. Keep its input in scratch until the final
      // real output is written so exact in-place C2R remains safe.
      RawExecutionContext row_context {context.request, context.stream, total_rows, half_n1, n1};
      return row_fft->execute(temp_half.get(), output, row_context);
    }

    // Step 2: Expand half-packed -> full Hermitian.
    std::vector<JitKernelArg> expand_args = {
        JitKernelArg::device(temp_half.get()),
        JitKernelArg::device(temp_full.get()),
        JitKernelArg::i64(half_n1),
        JitKernelArg::i32(static_cast<int32_t>(total_rows)),
    };
    expand_kernel->launch(context.stream, expand_args, ceil_div(n1, block), total_rows, 1);

    // Step 3: Row C2C IFFT (in-place on the full Hermitian buffer).
    RawExecutionContext row_context {context.request, context.stream, total_rows};
    result = row_fft->execute(temp_full.get(), temp_full.get(), row_context);
    if (result != FLAGFFT_SUCCESS) {
      return result;
    }

    // Step 4: Pack complex -> real.
    std::vector<JitKernelArg> pack_args = {
        JitKernelArg::device(temp_full.get()),
        JitKernelArg::device(output),
        JitKernelArg::i64(n1),
        JitKernelArg::i32(static_cast<int32_t>(total_rows)),
    };
    pack_kernel->launch(context.stream, pack_args, ceil_div(n1, block), total_rows, 1);
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] 2D C2R RC execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRawC2RFourStepCompactInRealOutNode::CompiledRawC2RFourStepCompactInRealOutNode(
    int64_t length,
    int64_t n1,
    int64_t n2,
    std::shared_ptr<JitKernel> row_kernel,
    std::vector<DeviceAllocation> row_tables,
    std::shared_ptr<JitKernel> col_kernel,
    std::vector<DeviceAllocation> col_tables,
    DeviceAllocation twiddle,
    DeviceAllocation stage1)
    : length(length),
      n1(n1),
      n2(n2),
      row_kernel(std::move(row_kernel)),
      row_tables(std::move(row_tables)),
      col_kernel(std::move(col_kernel)),
      col_tables(std::move(col_tables)),
      twiddle(std::move(twiddle)),
      stage1(std::move(stage1)) {
}

std::string CompiledRawC2RFourStepCompactInRealOutNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRawC2RFourStepCompactInRealOut(n=" << length << ", n1=" << n1 << ", n2=" << n2
      << ", row_kernel=" << (row_kernel ? row_kernel->execution_description() : "null")
      << ", col_kernel=" << (col_kernel ? col_kernel->execution_description() : "null") << ")";
  return oss.str();
}

flagfftResult CompiledRawC2RFourStepCompactInRealOutNode::execute(adaptor::DevicePtr input,
                                                                  adaptor::DevicePtr output,
                                                                  const RawExecutionContext &context) const {
  try {
    const int64_t half = length / 2 + 1;
    const bool in_place = input == output;
    const int64_t padded_real_distance = 2 * half;
    const int64_t input_distance = context.input_distance > 0 ? context.input_distance : half;
    const int64_t output_distance = in_place
                                        ? std::max(context.output_distance, padded_real_distance)
                                        : (context.output_distance > 0 ? context.output_distance : length);

    const bool fused_twiddle = row_kernel->tle_fused_twiddle;
    std::vector<JitKernelArg> row_args =
        fused_twiddle
            ? raw_distance_col_kernel_args({input, twiddle.get(), stage1.get()},
                                           row_tables,
                                           input_distance,
                                           context.batch)
            : raw_distance_col_kernel_args({input, stage1.get()}, row_tables, input_distance, context.batch);
    row_kernel->launch(context.stream, row_args, ceil_div(n2, row_kernel->inner_pack), context.batch, 1);

    std::vector<JitKernelArg> col_args =
        fused_twiddle
            ? raw_distance_col_kernel_args({stage1.get(), output}, col_tables, output_distance, context.batch)
            : raw_distance_col_kernel_args({stage1.get(), twiddle.get(), output},
                                           col_tables,
                                           output_distance,
                                           context.batch);
    col_kernel->launch(context.stream, col_args, ceil_div(n1, col_kernel->inner_pack), context.batch, 1);
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] C2RFourStepCompactInRealOut execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}
CompiledRaw3DNode::CompiledRaw3DNode(int64_t n0,
                                     int64_t n1,
                                     int64_t n2,
                                     std::shared_ptr<CompiledRawNode> n2_fft,
                                     std::shared_ptr<CompiledRawNode> n1_fft,
                                     std::shared_ptr<CompiledRawNode> n0_fft,
                                     std::shared_ptr<JitKernel> perm_021_fwd,
                                     std::shared_ptr<JitKernel> perm_210_fwd,
                                     std::shared_ptr<JitKernel> perm_201_fwd,
                                     std::shared_ptr<JitKernel> perm_120_inv,
                                     std::shared_ptr<JitKernel> perm_210_inv,
                                     std::shared_ptr<JitKernel> perm_021_inv,
                                     DeviceAllocation temp1,
                                     DeviceAllocation temp2,
                                     std::vector<DeviceAllocation> npu_transpose_indices,
                                     bool npu_pair_fused_store)
    : n0(n0),
      n1(n1),
      n2(n2),
      n2_fft(std::move(n2_fft)),
      n1_fft(std::move(n1_fft)),
      n0_fft(std::move(n0_fft)),
      perm_021_fwd(std::move(perm_021_fwd)),
      perm_210_fwd(std::move(perm_210_fwd)),
      perm_201_fwd(std::move(perm_201_fwd)),
      perm_120_inv(std::move(perm_120_inv)),
      perm_210_inv(std::move(perm_210_inv)),
      perm_021_inv(std::move(perm_021_inv)),
      temp1(std::move(temp1)),
      temp2(std::move(temp2)),
      npu_transpose_indices(std::move(npu_transpose_indices)),
      npu_pair_fused_store(npu_pair_fused_store) {
}

std::string CompiledRaw3DNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRaw3D(n0=" << n0 << ", n1=" << n1 << ", n2=" << n2
      << ", n2_fft=" << (n2_fft ? n2_fft->describe() : "null")
      << ", n1_fft=" << (n1_fft ? n1_fft->describe() : "null")
      << ", n0_fft=" << (n0_fft ? n0_fft->describe() : "null")
      << ", perm_021_fwd=" << (perm_021_fwd ? perm_021_fwd->execution_description() : "null")
      << ", perm_210_fwd=" << (perm_210_fwd ? perm_210_fwd->execution_description() : "null")
      << ", perm_201_fwd=" << (perm_201_fwd ? perm_201_fwd->execution_description() : "null")
      << ", npu_native_transpose=" << (!npu_transpose_indices.empty())
      << ", npu_pair_fused_store=" << npu_pair_fused_store << ")";
  return oss.str();
}

flagfftResult CompiledRaw3DNode::execute(adaptor::DevicePtr input,
                                         adaptor::DevicePtr output,
                                         const RawExecutionContext &context) const {
  try {
    const int64_t batch = context.batch;
    const int64_t total = n0 * n1 * n2;
    const bool inverse = context.request.direction == "inverse";

    RawExecutionContext n2_context {context.request, context.stream, batch * n0 * n1};
    RawExecutionContext n1_context {context.request, context.stream, batch * n0 * n2};
    RawExecutionContext n0_context {context.request, context.stream, batch * n1 * n2};

    if (npu_pair_fused_store) {
      flagfftResult result = n2_fft->execute(input, temp1.get(), n2_context);
      if (result != FLAGFFT_SUCCESS) return result;
      result = n1_fft->execute(temp1.get(), temp2.get(), n1_context);
      if (result != FLAGFFT_SUCCESS) return result;
      return n0_fft->execute(temp2.get(), output, n0_context);
    }

    auto permute = [&](const std::shared_ptr<JitKernel> &kernel,
                       adaptor::DevicePtr source,
                       adaptor::DevicePtr destination,
                       int64_t d0,
                       int64_t d1,
                       int64_t d2,
                       int32_t axis0,
                       int32_t axis1,
                       int32_t axis2) {
      return launch_perm3d_with_optional_npu_native(kernel,
                                                     npu_transpose_indices,
                                                     context.stream,
                                                     source,
                                                     destination,
                                                     d0,
                                                     d1,
                                                     d2,
                                                     axis0,
                                                     axis1,
                                                     axis2,
                                                     batch,
                                                     complex_element_bytes(context.request.input_dtype));
    };

    flagfftResult result;
    if (inverse) {
      // (n0,n1,n2) -> perm(1,2,0) -> (n1,n2,n0), IFFT along n0
      result = permute(perm_120_inv, input, temp1.get(), n0, n1, n2, 1, 2, 0);
      if (result != FLAGFFT_SUCCESS) return result;
      result = n0_fft->execute(temp1.get(), temp2.get(), n0_context);
      if (result != FLAGFFT_SUCCESS) {
        return result;
      }
      // (n1,n2,n0) -> perm(2,1,0) -> (n0,n2,n1), IFFT along n1
      result = permute(perm_210_inv, temp2.get(), temp1.get(), n1, n2, n0, 2, 1, 0);
      if (result != FLAGFFT_SUCCESS) return result;
      result = n1_fft->execute(temp1.get(), temp2.get(), n1_context);
      if (result != FLAGFFT_SUCCESS) {
        return result;
      }
      // (n0,n2,n1) -> perm(0,2,1) -> (n0,n1,n2), IFFT along n2
      result = permute(perm_021_inv, temp2.get(), temp1.get(), n0, n2, n1, 0, 2, 1);
      if (result != FLAGFFT_SUCCESS) return result;
      result = n2_fft->execute(temp1.get(), output, n2_context);
      if (result != FLAGFFT_SUCCESS) {
        return result;
      }
    } else {
      // FFT along n2 (contiguous in (n0,n1,n2))
      result = n2_fft->execute(input, temp1.get(), n2_context);
      if (result != FLAGFFT_SUCCESS) {
        return result;
      }
      // (n0,n1,n2) -> perm(0,2,1) -> (n0,n2,n1), FFT along n1
      result = permute(perm_021_fwd, temp1.get(), temp2.get(), n0, n1, n2, 0, 2, 1);
      if (result != FLAGFFT_SUCCESS) return result;
      result = n1_fft->execute(temp2.get(), temp1.get(), n1_context);
      if (result != FLAGFFT_SUCCESS) {
        return result;
      }
      // (n0,n2,n1) -> perm(2,1,0) -> (n1,n2,n0), FFT along n0
      result = permute(perm_210_fwd, temp1.get(), temp2.get(), n0, n2, n1, 2, 1, 0);
      if (result != FLAGFFT_SUCCESS) return result;
      result = n0_fft->execute(temp2.get(), temp1.get(), n0_context);
      if (result != FLAGFFT_SUCCESS) {
        return result;
      }
      // (n1,n2,n0) -> perm(2,0,1) -> (n0,n1,n2)
      result = permute(perm_201_fwd, temp1.get(), output, n1, n2, n0, 2, 0, 1);
      if (result != FLAGFFT_SUCCESS) return result;
    }
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] 3D execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRaw3DFusedPlaneNode::CompiledRaw3DFusedPlaneNode(
    int64_t n,
    int64_t outer_rows_per_cube,
    std::shared_ptr<JitKernel> plane_fft,
    std::shared_ptr<CompiledRawNode> outer_fft,
    DeviceAllocation temp,
    DeviceAllocation tw_r,
    DeviceAllocation tw_i)
    : n(n),
      outer_rows_per_cube(outer_rows_per_cube),
      plane_fft(std::move(plane_fft)),
      outer_fft(std::move(outer_fft)),
      temp(std::move(temp)),
      tw_r(std::move(tw_r)),
      tw_i(std::move(tw_i)) {
}

std::string CompiledRaw3DFusedPlaneNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRaw3DFusedPlane(n=" << n << ", outer_rows=" << outer_rows_per_cube
      << ", plane_fft=" << plane_fft->execution_description()
      << ", outer_fft=" << outer_fft->describe() << ")";
  return oss.str();
}

flagfftResult CompiledRaw3DFusedPlaneNode::execute(adaptor::DevicePtr input,
                                                   adaptor::DevicePtr output,
                                                   const RawExecutionContext &context) const {
  try {
    const int64_t batch = context.batch;
    std::vector<JitKernelArg> args = {
        JitKernelArg::device(input),
        JitKernelArg::device(temp.get()),
        JitKernelArg::device(tw_r.get()),
        JitKernelArg::device(tw_i.get()),
    };
    plane_fft->launch(context.stream, args, batch * n, 1, 1);
    RawExecutionContext outer_context {context.request, context.stream,
                                       batch * outer_rows_per_cube};
    return outer_fft->execute(temp.get(), output, outer_context);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] 3D fused plane execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRaw3DFusedCubeNode::CompiledRaw3DFusedCubeNode(
    std::shared_ptr<JitKernel> kernel, DeviceAllocation tw_r, DeviceAllocation tw_i)
    : kernel(std::move(kernel)), tw_r(std::move(tw_r)), tw_i(std::move(tw_i)) {
}

std::string CompiledRaw3DFusedCubeNode::describe() const {
  return "CompiledRaw3DFusedCube(kernel=" + kernel->execution_description() + ")";
}

flagfftResult CompiledRaw3DFusedCubeNode::execute(adaptor::DevicePtr input,
                                                  adaptor::DevicePtr output,
                                                  const RawExecutionContext &context) const {
  try {
    std::vector<JitKernelArg> args = {
        JitKernelArg::device(input),
        JitKernelArg::device(output),
        JitKernelArg::device(tw_r.get()),
        JitKernelArg::device(tw_i.get()),
    };
    kernel->launch(context.stream, args, context.batch * 16, 1, 1);
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] 3D fused cube execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRaw3DColumnNode::CompiledRaw3DColumnNode(
    int64_t outer_stride, int64_t columns, std::shared_ptr<JitKernel> kernel,
    DeviceAllocation tw_r, DeviceAllocation tw_i)
    : outer_stride(outer_stride), columns(columns), kernel(std::move(kernel)),
      tw_r(std::move(tw_r)), tw_i(std::move(tw_i)) {
}

std::string CompiledRaw3DColumnNode::describe() const {
  return "CompiledRaw3DColumn(outer_stride=" + std::to_string(outer_stride) +
         ", columns=" + std::to_string(columns) +
         ", kernel=" + kernel->execution_description() + ")";
}

flagfftResult CompiledRaw3DColumnNode::execute(adaptor::DevicePtr input,
                                               adaptor::DevicePtr output,
                                               const RawExecutionContext &context) const {
  try {
    if (context.batch % outer_stride != 0) {
      throw std::runtime_error("3D column batch must be a multiple of the outer stride");
    }
    const int64_t cube_batch = context.batch / outer_stride;
    std::vector<JitKernelArg> args = {
        JitKernelArg::device(input),
        JitKernelArg::device(output),
        JitKernelArg::device(tw_r.get()),
        JitKernelArg::device(tw_i.get()),
        JitKernelArg::i64(outer_stride),
    };
    kernel->launch(context.stream, args, ceil_div(outer_stride, columns), cube_batch, 1);
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] 3D column execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRaw3DStridedNode::CompiledRaw3DStridedNode(int64_t n0,
                                                   int64_t n1,
                                                   int64_t n2,
                                                   std::shared_ptr<CompiledRawNode> n2_fft,
                                                   std::shared_ptr<CompiledRawNode> n1_fft,
                                                   std::shared_ptr<CompiledRawNode> n0_fft,
                                                   DeviceAllocation temp1,
                                                   DeviceAllocation temp2)
    : n0(n0),
      n1(n1),
      n2(n2),
      n2_fft(std::move(n2_fft)),
      n1_fft(std::move(n1_fft)),
      n0_fft(std::move(n0_fft)),
      temp1(std::move(temp1)),
      temp2(std::move(temp2)) {
}

std::string CompiledRaw3DStridedNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRaw3DStrided(n0=" << n0 << ", n1=" << n1 << ", n2=" << n2
      << ", n2_fft=" << (n2_fft ? n2_fft->describe() : "null")
      << ", n1_fft=" << (n1_fft ? n1_fft->describe() : "null")
      << ", n0_fft=" << (n0_fft ? n0_fft->describe() : "null") << ")";
  return oss.str();
}

flagfftResult CompiledRaw3DStridedNode::execute(adaptor::DevicePtr input,
                                                adaptor::DevicePtr output,
                                                const RawExecutionContext &context) const {
  try {
    const int64_t batch = context.batch;

    // The cube stays in its natural (n0,n1,n2) layout throughout; each
    // non-contiguous axis is transformed in place with its own stride.
    RawExecutionContext n2_context {context.request, context.stream, batch * n0 * n1};
    RawExecutionContext n1_context {context.request, context.stream, batch * n0 * n2};
    RawExecutionContext n0_context {context.request, context.stream, batch * n1 * n2};

    flagfftResult result = n2_fft->execute(input, temp1.get(), n2_context);
    if (result != FLAGFFT_SUCCESS) {
      return result;
    }
    result = n1_fft->execute(temp1.get(), temp2.get(), n1_context);
    if (result != FLAGFFT_SUCCESS) {
      return result;
    }
    return n0_fft->execute(temp2.get(), output, n0_context);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] 3D strided execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRaw3DHybridNode::CompiledRaw3DHybridNode(int64_t n0,
                                                 int64_t n1,
                                                 int64_t n2,
                                                 std::shared_ptr<CompiledRawNode> n2_fft,
                                                 std::shared_ptr<CompiledRawNode> n1_fft,
                                                 std::shared_ptr<CompiledRawNode> n0_fft,
                                                 std::shared_ptr<JitKernel> perm_210,
                                                 std::shared_ptr<JitKernel> perm_201,
                                                 DeviceAllocation temp1,
                                                 DeviceAllocation temp2)
    : n0(n0),
      n1(n1),
      n2(n2),
      n2_fft(std::move(n2_fft)),
      n1_fft(std::move(n1_fft)),
      n0_fft(std::move(n0_fft)),
      perm_210(std::move(perm_210)),
      perm_201(std::move(perm_201)),
      temp1(std::move(temp1)),
      temp2(std::move(temp2)) {
}

std::string CompiledRaw3DHybridNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRaw3DHybrid(n0=" << n0 << ", n1=" << n1 << ", n2=" << n2
      << ", final_transpose=" << (perm_201 != nullptr)
      << ", n2_fft=" << n2_fft->describe() << ", n1_fft=" << n1_fft->describe()
      << ", n0_fft=" << n0_fft->describe() << ")";
  return oss.str();
}

flagfftResult CompiledRaw3DHybridNode::execute(adaptor::DevicePtr input,
                                               adaptor::DevicePtr output,
                                               const RawExecutionContext &context) const {
  try {
    const int64_t batch = context.batch;
    const int64_t total = n0 * n1 * n2;
    RawExecutionContext n2_context {context.request, context.stream, batch * n0 * n1};
    RawExecutionContext n1_context {context.request, context.stream, batch * n0 * n2};
    RawExecutionContext n0_context {context.request, context.stream, batch * n1 * n2};

    flagfftResult result = n2_fft->execute(input, temp1.get(), n2_context);
    if (result != FLAGFFT_SUCCESS) return result;
    result = n1_fft->execute(temp1.get(), temp2.get(), n1_context);
    if (result != FLAGFFT_SUCCESS) return result;
    launch_perm3d(perm_210, context.stream, temp2.get(), temp1.get(), total, batch,
                  complex_element_bytes(context.request.input_dtype));
    result = n0_fft->execute(temp1.get(), perm_201 ? temp2.get() : output, n0_context);
    if (result != FLAGFFT_SUCCESS) return result;
    if (perm_201) {
      launch_perm3d(perm_201, context.stream, temp2.get(), output, total, batch,
                    complex_element_bytes(context.request.input_dtype));
    }
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] 3D hybrid execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRaw3DRealLeafNode::CompiledRaw3DRealLeafNode(int64_t n0,
                                                     int64_t n1,
                                                     int64_t n2,
                                                     bool inverse,
                                                     bool fused_store,
                                                     std::shared_ptr<CompiledRawNode> n2_real_fft,
                                                     std::shared_ptr<CompiledRawNode> n1_fft,
                                                     std::shared_ptr<CompiledRawNode> n0_fft,
                                                     std::shared_ptr<JitKernel> perm_021,
                                                     DeviceAllocation temp1,
                                                     DeviceAllocation temp2)
    : n0(n0),
      n1(n1),
      n2(n2),
      inverse(inverse),
      fused_store(fused_store),
      n2_real_fft(std::move(n2_real_fft)),
      n1_fft(std::move(n1_fft)),
      n0_fft(std::move(n0_fft)),
      perm_021(std::move(perm_021)),
      temp1(std::move(temp1)),
      temp2(std::move(temp2)) {
}

std::string CompiledRaw3DRealLeafNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRaw3DRealLeaf(n0=" << n0 << ", n1=" << n1 << ", n2=" << n2
      << ", inverse=" << inverse << ", fused_store=" << fused_store
      << ", n2_real_fft=" << n2_real_fft->describe()
      << ", n1_fft=" << n1_fft->describe() << ", n0_fft=" << n0_fft->describe() << ")";
  return oss.str();
}

flagfftResult CompiledRaw3DRealLeafNode::execute(adaptor::DevicePtr input,
                                                 adaptor::DevicePtr output,
                                                 const RawExecutionContext &context) const {
  try {
    const int64_t batch = context.batch;
    const int64_t half = n2 / 2 + 1;
    const int64_t packed = n0 * n1 * half;
    RawExecutionContext n2_context {context.request, context.stream, batch * n0 * n1};
    RawExecutionContext n1_context {context.request, context.stream, batch * n0 * half};
    RawExecutionContext n0_context {context.request, context.stream, batch * n1 * half};
    flagfftResult result;

    if (!inverse) {
      // The real leaf produces compact rows in natural (n0,n1,half) order.
      result = n2_real_fft->execute(input, temp1.get(), n2_context);
      if (result != FLAGFFT_SUCCESS) return result;
      if (fused_store) {
        // n1 wants contiguous rows in (n0,half,n1) order.  Its store and
        // the n0 store both apply the following layout change.
        launch_perm3d(perm_021, context.stream, temp1.get(), temp2.get(), packed, batch,
                      complex_element_bytes(context.request.input_dtype));
        result = n1_fft->execute(temp2.get(), temp1.get(), n1_context);
      } else {
        result = n1_fft->execute(temp1.get(), temp2.get(), n1_context);
      }
      if (result != FLAGFFT_SUCCESS) return result;
      return n0_fft->execute(fused_store ? temp1.get() : temp2.get(), output, n0_context);
    }

    // Axes commute, so the compact n1/n0 transforms can precede the real
    // inverse n2 boundary.  This also keeps the compact cube throughout.
    if (fused_store) {
      launch_perm3d(perm_021, context.stream, input, temp1.get(), packed, batch,
                    complex_element_bytes(context.request.input_dtype));
      result = n1_fft->execute(temp1.get(), temp2.get(), n1_context);
    } else {
      result = n1_fft->execute(input, temp1.get(), n1_context);
    }
    if (result != FLAGFFT_SUCCESS) return result;
    result = n0_fft->execute(fused_store ? temp2.get() : temp1.get(),
                             fused_store ? temp1.get() : temp2.get(), n0_context);
    if (result != FLAGFFT_SUCCESS) return result;
    return n2_real_fft->execute(fused_store ? temp1.get() : temp2.get(), output, n2_context);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] 3D real leaf execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRaw3DRealRTRTNode::CompiledRaw3DRealRTRTNode(
    int64_t n0,
    int64_t n1,
    int64_t n2,
    bool inverse,
    std::shared_ptr<CompiledRawNode> n2_real_fft,
    std::shared_ptr<CompiledRawNode> n1_fft,
    std::shared_ptr<CompiledRawNode> n0_fft,
    std::shared_ptr<JitKernel> perm_021,
    std::shared_ptr<JitKernel> perm_210,
    std::shared_ptr<JitKernel> perm_201,
    DeviceAllocation temp1,
    DeviceAllocation temp2,
    std::vector<DeviceAllocation> npu_transpose_indices)
    : n0(n0),
      n1(n1),
      n2(n2),
      inverse(inverse),
      n2_real_fft(std::move(n2_real_fft)),
      n1_fft(std::move(n1_fft)),
      n0_fft(std::move(n0_fft)),
      perm_021(std::move(perm_021)),
      perm_210(std::move(perm_210)),
      perm_201(std::move(perm_201)),
      temp1(std::move(temp1)),
      temp2(std::move(temp2)),
      npu_transpose_indices(std::move(npu_transpose_indices)) {
}

std::string CompiledRaw3DRealRTRTNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRaw3DRealRTRT(n0=" << n0 << ", n1=" << n1 << ", n2=" << n2
      << ", inverse=" << inverse
      << ", fused_first=" << (perm_021 == nullptr && npu_transpose_indices.empty())
      << ", fused_middle=" << (perm_210 == nullptr && npu_transpose_indices.empty())
      << ", fused_n0=" << (perm_201 == nullptr && npu_transpose_indices.empty())
      << ", npu_native_transpose=" << (!npu_transpose_indices.empty())
      << ", n2_real_fft=" << n2_real_fft->describe()
      << ", n1_fft=" << n1_fft->describe() << ", n0_fft=" << n0_fft->describe() << ")";
  return oss.str();
}

flagfftResult CompiledRaw3DRealRTRTNode::execute(adaptor::DevicePtr input,
                                                 adaptor::DevicePtr output,
                                                 const RawExecutionContext &context) const {
  try {
    const int64_t batch = context.batch;
    const int64_t half = n2 / 2 + 1;
    const int64_t packed = n0 * n1 * half;
    RawExecutionContext n2_context {context.request, context.stream, batch * n0 * n1};
    FFTRequest n1_request = context.request;
    FFTRequest n0_request = context.request;
    if (context.request.device_type == "npu") {
      n1_request.real_transform_kind.clear();
      n1_request.real_transform = false;
      n0_request.real_transform_kind.clear();
      n0_request.real_transform = false;
    }
    RawExecutionContext n1_context {n1_request, context.stream, batch * n0 * half};
    RawExecutionContext n0_context {n0_request, context.stream, batch * n1 * half};
    const bool has_perm_021 = perm_021 != nullptr || !npu_transpose_indices.empty();
    const bool has_perm_210 = perm_210 != nullptr || !npu_transpose_indices.empty();
    const bool has_perm_201 = perm_201 != nullptr || !npu_transpose_indices.empty();
    auto permute = [&](const std::shared_ptr<JitKernel> &kernel,
                       adaptor::DevicePtr source,
                       adaptor::DevicePtr destination,
                       int64_t d0,
                       int64_t d1,
                       int64_t d2,
                       int32_t axis0,
                       int32_t axis1,
                       int32_t axis2) {
      return launch_perm3d_with_optional_npu_native(
          kernel, npu_transpose_indices, context.stream, source, destination,
          d0, d1, d2, axis0, axis1, axis2, batch,
          complex_element_bytes(context.request.input_dtype));
    };

    if (!inverse) {
      if (!has_perm_021) n2_context.output_distance = n1;
      adaptor::DevicePtr n2_output = has_perm_021 ? temp1.get() : temp2.get();
      flagfftResult result = n2_real_fft->execute(input, n2_output, n2_context);
      if (result != FLAGFFT_SUCCESS) return result;
      if (has_perm_021) {
        result = permute(perm_021, temp1.get(), temp2.get(), n0, n1, half, 0, 2, 1);
        if (result != FLAGFFT_SUCCESS) return result;
      }
      result = n1_fft->execute(temp2.get(), temp1.get(), n1_context);
      if (result != FLAGFFT_SUCCESS) return result;
      if (has_perm_210) {
        result = permute(perm_210, temp1.get(), temp2.get(), n0, half, n1, 2, 1, 0);
        if (result != FLAGFFT_SUCCESS) return result;
      }
      adaptor::DevicePtr n0_output = has_perm_201
          ? (has_perm_210 ? temp1.get() : temp2.get()) : output;
      result = n0_fft->execute(has_perm_210 ? temp2.get() : temp1.get(),
                               n0_output, n0_context);
      if (result != FLAGFFT_SUCCESS) return result;
      if (has_perm_201) {
        result = permute(perm_201, n0_output, output, n1, half, n0, 2, 0, 1);
        if (result != FLAGFFT_SUCCESS) return result;
      }
      return FLAGFFT_SUCCESS;
    }

    // The outer transforms commute, so both use the same compact layouts
    // before the final real inverse along n2.
    flagfftResult result = permute(perm_021, input, temp1.get(), n0, n1, half, 0, 2, 1);
    if (result != FLAGFFT_SUCCESS) return result;
    result = n1_fft->execute(temp1.get(), temp2.get(), n1_context);
    if (result != FLAGFFT_SUCCESS) return result;
    result = permute(perm_210, temp2.get(), temp1.get(), n0, half, n1, 2, 1, 0);
    if (result != FLAGFFT_SUCCESS) return result;
    result = n0_fft->execute(temp1.get(), temp2.get(), n0_context);
    if (result != FLAGFFT_SUCCESS) return result;
    if (has_perm_201) {
      result = permute(perm_201, temp2.get(), temp1.get(), n1, half, n0, 2, 0, 1);
      if (result != FLAGFFT_SUCCESS) return result;
    }
    return n2_real_fft->execute(has_perm_201 ? temp1.get() : temp2.get(), output, n2_context);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] 3D real RTRT execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRaw3DR2CNode::CompiledRaw3DR2CNode(int64_t n0,
                                           int64_t n1,
                                           int64_t n2,
                                           std::shared_ptr<JitKernel> expand_kernel,
                                           std::shared_ptr<CompiledRawNode> n2_fft,
                                           std::shared_ptr<JitKernel> pack_kernel,
                                           std::shared_ptr<CompiledRawNode> n1_fft,
                                           std::shared_ptr<CompiledRawNode> n0_fft,
                                           std::shared_ptr<JitKernel> perm_021,
                                           std::shared_ptr<JitKernel> perm_210,
                                           std::shared_ptr<JitKernel> perm_201,
                                           DeviceAllocation row_fft_buf,
                                           DeviceAllocation temp1,
                                           DeviceAllocation temp2)
    : n0(n0),
      n1(n1),
      n2(n2),
      expand_kernel(std::move(expand_kernel)),
      n2_fft(std::move(n2_fft)),
      pack_kernel(std::move(pack_kernel)),
      n1_fft(std::move(n1_fft)),
      n0_fft(std::move(n0_fft)),
      perm_021(std::move(perm_021)),
      perm_210(std::move(perm_210)),
      perm_201(std::move(perm_201)),
      row_fft_buf(std::move(row_fft_buf)),
      temp1(std::move(temp1)),
      temp2(std::move(temp2)) {
}

std::string CompiledRaw3DR2CNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRaw3DR2C(n0=" << n0 << ", n1=" << n1 << ", n2=" << n2
      << ", expand_kernel=" << (expand_kernel ? expand_kernel->execution_description() : "null")
      << ", n2_fft=" << (n2_fft ? n2_fft->describe() : "null")
      << ", pack_kernel=" << (pack_kernel ? pack_kernel->execution_description() : "null")
      << ", n1_fft=" << (n1_fft ? n1_fft->describe() : "null")
      << ", n0_fft=" << (n0_fft ? n0_fft->describe() : "null") << ")";
  return oss.str();
}

flagfftResult CompiledRaw3DR2CNode::execute(adaptor::DevicePtr input,
                                            adaptor::DevicePtr output,
                                            const RawExecutionContext &context) const {
  try {
    const int64_t batch = context.batch;
    constexpr int64_t block = 256;
    const int64_t half = n2 / 2 + 1;
    const int64_t total_rows = batch * n0 * n1;
    const int64_t complex_bytes = complex_element_bytes(context.request.input_dtype);
    const int64_t real_bytes = complex_bytes / 2;

    // Step 1: Expand real -> complex rows of length n2.
    launch_grid_y_chunks(ceil_div(n2, block),
                         total_rows,
                         expand_kernel->rows_per_block,
                         [&](int64_t row_offset, int64_t chunk_rows) {
                           std::vector<JitKernelArg> expand_args = {
                               JitKernelArg::device(input + row_offset * n2 * real_bytes),
                               JitKernelArg::device(row_fft_buf.get() + row_offset * n2 * complex_bytes),
                               JitKernelArg::i64(n2),
                               JitKernelArg::i32(static_cast<int32_t>(chunk_rows)),
                           };
                           expand_kernel->launch(context.stream,
                                                 expand_args,
                                                 ceil_div(n2, block),
                                                 grid_rows(expand_kernel, chunk_rows),
                                                 1);
                         });

    // Step 2: FFT along n2 in-place.
    RawExecutionContext n2_context {context.request, context.stream, total_rows};
    flagfftResult result = n2_fft->execute(row_fft_buf.get(), row_fft_buf.get(), n2_context);
    if (result != FLAGFFT_SUCCESS) {
      return result;
    }

    // Step 3: Half-pack rows into the output (n0, n1, half) layout.
    launch_grid_y_chunks(ceil_div(half, block),
                         total_rows,
                         pack_kernel->rows_per_block,
                         [&](int64_t row_offset, int64_t chunk_rows) {
                           std::vector<JitKernelArg> pack_args = {
                               JitKernelArg::device(row_fft_buf.get() + row_offset * n2 * complex_bytes),
                               JitKernelArg::device(output + row_offset * half * complex_bytes),
                               JitKernelArg::i64(half),
                               JitKernelArg::i32(static_cast<int32_t>(chunk_rows)),
                           };
                           pack_kernel->launch(context.stream,
                                               pack_args,
                                               ceil_div(half, block),
                                               grid_rows(pack_kernel, chunk_rows),
                                               1);
                         });

    // Step 4: (n0,n1,half) -> (n0,half,n1), FFT along n1.
    const int64_t packed = n0 * n1 * half;
    launch_perm3d(perm_021, context.stream, output, temp1.get(), packed, batch,
                  complex_element_bytes(context.request.input_dtype));
    RawExecutionContext n1_context {context.request, context.stream, batch * n0 * half};
    result = n1_fft->execute(temp1.get(), temp2.get(), n1_context);
    if (result != FLAGFFT_SUCCESS) {
      return result;
    }

    // Step 5: (n0,half,n1) -> (n1,half,n0), FFT along n0.
    launch_perm3d(perm_210, context.stream, temp2.get(), temp1.get(), packed, batch,
                  complex_element_bytes(context.request.input_dtype));
    RawExecutionContext n0_context {context.request, context.stream, batch * n1 * half};
    result = n0_fft->execute(temp1.get(), temp2.get(), n0_context);
    if (result != FLAGFFT_SUCCESS) {
      return result;
    }

    // Step 6: (n1,half,n0) -> (n0,n1,half) natural output layout.
    launch_perm3d(perm_201, context.stream, temp2.get(), output, packed, batch,
                  complex_element_bytes(context.request.input_dtype));
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] 3D R2C execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRaw3DC2RNode::CompiledRaw3DC2RNode(int64_t n0,
                                           int64_t n1,
                                           int64_t n2,
                                           std::shared_ptr<JitKernel> perm_120,
                                           std::shared_ptr<JitKernel> perm_210,
                                           std::shared_ptr<JitKernel> perm_021,
                                           std::shared_ptr<CompiledRawNode> n0_fft,
                                           std::shared_ptr<CompiledRawNode> n1_fft,
                                           std::shared_ptr<JitKernel> expand_kernel,
                                           std::shared_ptr<CompiledRawNode> n2_fft,
                                           std::shared_ptr<JitKernel> pack_kernel,
                                           DeviceAllocation temp1,
                                           DeviceAllocation temp2,
                                           DeviceAllocation full_buf)
    : n0(n0),
      n1(n1),
      n2(n2),
      perm_120(std::move(perm_120)),
      perm_210(std::move(perm_210)),
      perm_021(std::move(perm_021)),
      n0_fft(std::move(n0_fft)),
      n1_fft(std::move(n1_fft)),
      expand_kernel(std::move(expand_kernel)),
      n2_fft(std::move(n2_fft)),
      pack_kernel(std::move(pack_kernel)),
      temp1(std::move(temp1)),
      temp2(std::move(temp2)),
      full_buf(std::move(full_buf)) {
}

std::string CompiledRaw3DC2RNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRaw3DC2R(n0=" << n0 << ", n1=" << n1 << ", n2=" << n2
      << ", n0_fft=" << (n0_fft ? n0_fft->describe() : "null")
      << ", n1_fft=" << (n1_fft ? n1_fft->describe() : "null")
      << ", expand_kernel=" << (expand_kernel ? expand_kernel->execution_description() : "null")
      << ", n2_fft=" << (n2_fft ? n2_fft->describe() : "null")
      << ", pack_kernel=" << (pack_kernel ? pack_kernel->execution_description() : "null") << ")";
  return oss.str();
}

flagfftResult CompiledRaw3DC2RNode::execute(adaptor::DevicePtr input,
                                            adaptor::DevicePtr output,
                                            const RawExecutionContext &context) const {
  try {
    const int64_t batch = context.batch;
    constexpr int64_t block = 256;
    const int64_t complex_bytes = complex_element_bytes(context.request.input_dtype);
    const int64_t real_bytes = complex_bytes / 2;
    const int64_t half = n2 / 2 + 1;
    const int64_t total_rows = batch * n0 * n1;
    const int64_t packed = n0 * n1 * half;

    // Step 1: (n0,n1,half) -> (n1,half,n0), IFFT along n0.
    launch_perm3d(perm_120, context.stream, input, temp1.get(), packed, batch,
                  complex_element_bytes(context.request.input_dtype));
    RawExecutionContext n0_context {context.request, context.stream, batch * n1 * half};
    flagfftResult result = n0_fft->execute(temp1.get(), temp2.get(), n0_context);
    if (result != FLAGFFT_SUCCESS) {
      return result;
    }

    // Step 2: (n1,half,n0) -> (n0,half,n1), IFFT along n1.
    launch_perm3d(perm_210, context.stream, temp2.get(), temp1.get(), packed, batch,
                  complex_element_bytes(context.request.input_dtype));
    RawExecutionContext n1_context {context.request, context.stream, batch * n0 * half};
    result = n1_fft->execute(temp1.get(), temp2.get(), n1_context);
    if (result != FLAGFFT_SUCCESS) {
      return result;
    }

    // Step 3: (n0,half,n1) -> (n0,n1,half), expand half -> full Hermitian.
    launch_perm3d(perm_021, context.stream, temp2.get(), temp1.get(), packed, batch,
                  complex_element_bytes(context.request.input_dtype));
    launch_grid_y_chunks(ceil_div(n2, block),
                         total_rows,
                         expand_kernel->rows_per_block,
                         [&](int64_t row_offset, int64_t chunk_rows) {
                           std::vector<JitKernelArg> expand_args = {
                               JitKernelArg::device(temp1.get() + row_offset * half * complex_bytes),
                               JitKernelArg::device(full_buf.get() + row_offset * n2 * complex_bytes),
                               JitKernelArg::i64(half),
                               JitKernelArg::i32(static_cast<int32_t>(chunk_rows)),
                           };
                           expand_kernel->launch(context.stream,
                                                 expand_args,
                                                 ceil_div(n2, block),
                                                 grid_rows(expand_kernel, chunk_rows),
                                                 1);
                         });

    // Step 4: IFFT along n2 (in-place on full complex rows), pack complex -> real.
    RawExecutionContext n2_context {context.request, context.stream, total_rows};
    result = n2_fft->execute(full_buf.get(), full_buf.get(), n2_context);
    if (result != FLAGFFT_SUCCESS) {
      return result;
    }
    launch_grid_y_chunks(ceil_div(n2, block),
                         total_rows,
                         pack_kernel->rows_per_block,
                         [&](int64_t row_offset, int64_t chunk_rows) {
                           std::vector<JitKernelArg> pack_args = {
                               JitKernelArg::device(full_buf.get() + row_offset * n2 * complex_bytes),
                               JitKernelArg::device(output + row_offset * n2 * real_bytes),
                               JitKernelArg::i64(n2),
                               JitKernelArg::i32(static_cast<int32_t>(chunk_rows)),
                           };
                           pack_kernel->launch(context.stream,
                                               pack_args,
                                               ceil_div(n2, block),
                                               grid_rows(pack_kernel, chunk_rows),
                                               1);
                         });
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] 3D C2R execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

}  // namespace flagfft
