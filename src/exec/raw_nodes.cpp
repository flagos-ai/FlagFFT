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

#include <algorithm>
#include <cstdio>
#include <sstream>

namespace flagfft {
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

  template <typename Launch>
  void launch_grid_y_chunks(int64_t rows, Launch &&launch) {
    for (int64_t offset = 0; offset < rows; offset += kMaxGridY) {
      const int64_t chunk = std::min(kMaxGridY, rows - offset);
      launch(offset, chunk);
    }
  }

  void launch_perm3d(const std::shared_ptr<JitKernel> &kernel,
                     adaptor::StreamHandle stream,
                     adaptor::DevicePtr input,
                     adaptor::DevicePtr output,
                     int64_t elements_per_batch,
                     int64_t batch) {
    std::vector<JitKernelArg> args = {
        JitKernelArg::device(input),
        JitKernelArg::device(output),
        JitKernelArg::i32(static_cast<int32_t>(batch)),
    };
    kernel->launch(stream, args, ceil_div(elements_per_batch, kPerm3dBlock), 1, batch);
  }

}  // namespace

CompiledRawLeafNode::CompiledRawLeafNode(int64_t length,
                                         std::shared_ptr<JitKernel> kernel,
                                         std::vector<DeviceAllocation> tables)
    : length(length), kernel(std::move(kernel)), tables(std::move(tables)) {
}

std::string CompiledRawLeafNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRawLeaf(n=" << length << ", kernel=" << (kernel ? kernel->kernel_name : "null")
      << ", num_warps=" << (kernel ? kernel->num_warps : 0)
      << ", module=" << (kernel ? kernel->module_path : "null") << ", tables=" << tables.size() << ")";
  return oss.str();
}

flagfftResult CompiledRawLeafNode::execute(adaptor::DevicePtr input,
                                           adaptor::DevicePtr output,
                                           const RawExecutionContext &context) const {
  try {
    std::vector<JitKernelArg> args = raw_kernel_args({input, output}, tables, context.batch);

    kernel->launch(context.stream, args, ceil_div(context.batch, kernel->batch_per_block), 1, 1);
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
      << ", row_kernel=" << (row_kernel ? row_kernel->kernel_name : "null")
      << ", col_kernel=" << (col_kernel ? col_kernel->kernel_name : "null") << ")";
  return oss.str();
}

flagfftResult CompiledRawFourStepFusedNode::execute(adaptor::DevicePtr input,
                                                    adaptor::DevicePtr output,
                                                    const RawExecutionContext &context) const {
  try {
    const bool fused_twiddle = row_kernel->tle_fused_twiddle;
    std::vector<JitKernelArg> row_args =
        fused_twiddle ? raw_kernel_args({input, twiddle.get(), stage1.get()}, row_tables, context.batch)
                      : raw_kernel_args({input, stage1.get()}, row_tables, context.batch);
    row_kernel->launch(context.stream, row_args, ceil_div(n2, row_kernel->inner_pack), context.batch, 1);

    std::vector<JitKernelArg> col_args =
        fused_twiddle ? raw_kernel_args({stage1.get(), output}, col_tables, context.batch)
                      : raw_kernel_args({stage1.get(), twiddle.get(), output}, col_tables, context.batch);
    col_kernel->launch(context.stream, col_args, ceil_div(n1, col_kernel->inner_pack), context.batch, 1);
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] FourStepFused execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRawDirectDftNode::CompiledRawDirectDftNode(int64_t length,
                                                   std::shared_ptr<JitKernel> kernel,
                                                   std::vector<DeviceAllocation> tables)
    : length(length), kernel(std::move(kernel)), tables(std::move(tables)) {
}

std::string CompiledRawDirectDftNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRawDirectDft(n=" << length << ", kernel=" << (kernel ? kernel->kernel_name : "null") << ")";
  return oss.str();
}

flagfftResult CompiledRawDirectDftNode::execute(adaptor::DevicePtr input,
                                                adaptor::DevicePtr output,
                                                const RawExecutionContext &context) const {
  try {
    std::vector<JitKernelArg> args = raw_kernel_args({input, output}, tables, context.batch);
    kernel->launch(context.stream, args, context.batch, 1, 1);
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
                                                   DeviceAllocation b_fft_buf)
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
      b_fft_buf(std::move(b_fft_buf)) {
}

std::string CompiledRawBluesteinNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRawBluestein(n=" << length << ", conv_length=" << conv_length
      << ", prepare_kernel=" << (prepare_kernel ? prepare_kernel->kernel_name : "null")
      << ", pointwise_kernel=" << (pointwise_kernel ? pointwise_kernel->kernel_name : "null")
      << ", finalize_kernel=" << (finalize_kernel ? finalize_kernel->kernel_name : "null")
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

    std::vector<JitKernelArg> prepare_args = {
        JitKernelArg::device(input),
        JitKernelArg::device(chirp.get()),
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
        JitKernelArg::i64(conv_length),
        JitKernelArg::i32(static_cast<int32_t>(context.batch)),
    };
    pointwise_kernel->launch(context.stream, pointwise_args, ceil_div(conv_length, 256), context.batch, 1);

    result = fft->execute(a_buf.get(), work_buf.get(), child_context);
    if (result != FLAGFFT_SUCCESS) {
      return result;
    }

    std::vector<JitKernelArg> finalize_args = {
        JitKernelArg::device(work_buf.get()),
        JitKernelArg::device(chirp.get()),
        JitKernelArg::device(output),
        JitKernelArg::i64(length),
        JitKernelArg::i64(conv_length),
        JitKernelArg::i32(static_cast<int32_t>(context.batch)),
    };
    finalize_kernel->launch(context.stream, finalize_args, ceil_div(length, 256), context.batch, 1);
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] Bluestein execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRawBluesteinLeafNode::CompiledRawBluesteinLeafNode(
    int64_t length,
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
      << ", prepare_kernel=" << (prepare_kernel ? prepare_kernel->kernel_name : "null")
      << ", finish_kernel=" << (finish_kernel ? finish_kernel->kernel_name : "null")
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
    prepare_kernel->launch(
        context.stream, prepare_args, ceil_div(context.batch, prepare_kernel->batch_per_block), 1, 1);

    std::vector<JitKernelArg> finish_args =
        raw_kernel_args({work_buf.get(), b_fft_buf.get(), chirp.get(), output}, tables, context.batch);
    finish_kernel->launch(
        context.stream, finish_args, ceil_div(context.batch, finish_kernel->batch_per_block), 1, 1);
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] BluesteinLeaf execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRawBluesteinFullLeafNode::CompiledRawBluesteinFullLeafNode(
    int64_t length,
    int64_t conv_length,
    std::shared_ptr<CompiledRawNode> fft,
    std::shared_ptr<JitKernel> kernel,
    std::vector<DeviceAllocation> tables,
    DeviceAllocation chirp,
    DeviceAllocation b_time,
    DeviceAllocation b_fft_buf)
    : length(length),
      conv_length(conv_length),
      fft(std::move(fft)),
      kernel(std::move(kernel)),
      tables(std::move(tables)),
      chirp(std::move(chirp)),
      b_time(std::move(b_time)),
      b_fft_buf(std::move(b_fft_buf)) {
}

std::string CompiledRawBluesteinFullLeafNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRawBluesteinFullLeaf(n=" << length << ", conv_length=" << conv_length
      << ", kernel=" << (kernel ? kernel->kernel_name : "null")
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
    DeviceAllocation b_fft_buf)
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
      b_fft_buf(std::move(b_fft_buf)) {
}

std::string CompiledRawBluesteinFourStepNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRawBluesteinFourStep(n=" << length << ", conv_length=" << conv_length
      << ", n1=" << n1 << ", n2=" << n2
      << ", prepare_row=" << (prepare_row_kernel ? prepare_row_kernel->kernel_name : "null")
      << ", first_col=" << (first_col_kernel ? first_col_kernel->kernel_name : "null")
      << ", pointwise_row=" << (pointwise_row_kernel ? pointwise_row_kernel->kernel_name : "null")
      << ", finish_col=" << (finish_col_kernel ? finish_col_kernel->kernel_name : "null") << ")";
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
    ensure_b_fft(context);
    const bool fused_twiddle = prepare_row_kernel->tle_fused_twiddle;

    stage = "prepare-row";
    std::vector<JitKernelArg> prepare_args =
        fused_twiddle
            ? raw_kernel_args({input, chirp.get(), twiddle.get(), stage1.get()}, row_tables, context.batch)
            : raw_kernel_args({input, chirp.get(), stage1.get()}, row_tables, context.batch);
    prepare_row_kernel->launch(
        context.stream, prepare_args, ceil_div(n2, prepare_row_kernel->inner_pack), context.batch, 1);

    stage = "first-col";
    std::vector<JitKernelArg> first_col_args =
        fused_twiddle ? raw_kernel_args({stage1.get(), work_buf.get()}, col_tables, context.batch)
                      : raw_kernel_args({stage1.get(), twiddle.get(), work_buf.get()}, col_tables, context.batch);
    first_col_kernel->launch(
        context.stream, first_col_args, ceil_div(n1, first_col_kernel->inner_pack), context.batch, 1);

    stage = "pointwise-row";
    std::vector<JitKernelArg> pointwise_args =
        fused_twiddle
            ? raw_kernel_args({work_buf.get(), b_fft_buf.get(), twiddle.get(), stage1.get()},
                              row_tables,
                              context.batch)
            : raw_kernel_args({work_buf.get(), b_fft_buf.get(), stage1.get()}, row_tables, context.batch);
    pointwise_row_kernel->launch(
        context.stream, pointwise_args, ceil_div(n2, pointwise_row_kernel->inner_pack), context.batch, 1);

    stage = "finish-col";
    std::vector<JitKernelArg> finish_args =
        fused_twiddle
            ? raw_kernel_args({stage1.get(), chirp.get(), output}, col_tables, context.batch)
            : raw_kernel_args({stage1.get(), chirp.get(), twiddle.get(), output}, col_tables, context.batch);
    finish_col_kernel->launch(
        context.stream, finish_args, ceil_div(n1, finish_col_kernel->inner_pack), context.batch, 1);
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
                                           DeviceAllocation input_copy)
    : length(length),
      conv_length(conv_length),
      fft(std::move(fft)),
      prepare_kernel(std::move(prepare_kernel)),
      pointwise_kernel(std::move(pointwise_kernel)),
      finalize_kernel(std::move(finalize_kernel)),
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
      << ", prepare_kernel=" << (prepare_kernel ? prepare_kernel->kernel_name : "null")
      << ", pointwise_kernel=" << (pointwise_kernel ? pointwise_kernel->kernel_name : "null")
      << ", finalize_kernel=" << (finalize_kernel ? finalize_kernel->kernel_name : "null")
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

    std::vector<JitKernelArg> reshape_in_args = {
        JitKernelArg::device(input),
        JitKernelArg::device(stage1.get()),
        JitKernelArg::i32(static_cast<int32_t>(context.batch)),
    };
    reshape_in_kernel->launch(context.stream,
                              reshape_in_args,
                              ceil_div(total, reshape_block),
                              context.batch,
                              1);

    RawExecutionContext row_context {context.request, context.stream, context.batch * n2};
    flagfftResult result = row_child->execute(stage1.get(), stage2.get(), row_context);
    if (result != FLAGFFT_SUCCESS) {
      return result;
    }

    std::vector<JitKernelArg> twiddle_args = {
        JitKernelArg::device(stage2.get()),
        JitKernelArg::device(twiddle.get()),
        JitKernelArg::device(stage1.get()),
        JitKernelArg::i32(static_cast<int32_t>(context.batch)),
    };
    twiddle_reshape_kernel->launch(context.stream,
                                   twiddle_args,
                                   ceil_div(total, reshape_block),
                                   context.batch,
                                   1);

    RawExecutionContext col_context {context.request, context.stream, context.batch * n1};
    result = col_child->execute(stage1.get(), stage2.get(), col_context);
    if (result != FLAGFFT_SUCCESS) {
      return result;
    }

    std::vector<JitKernelArg> final_args = {
        JitKernelArg::device(stage2.get()),
        JitKernelArg::device(output),
        JitKernelArg::i32(static_cast<int32_t>(context.batch)),
    };
    final_pack_kernel->launch(context.stream, final_args, ceil_div(total, reshape_block), context.batch, 1);
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
      << ", expand_kernel=" << (expand_kernel ? expand_kernel->kernel_name : "null")
      << ", fft=" << (fft ? fft->describe() : "null")
      << ", pack_kernel=" << (pack_kernel ? pack_kernel->kernel_name : "null") << ")";
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
    std::vector<JitKernelArg> expand_args = {
        JitKernelArg::device(input),
        JitKernelArg::device(complex_input.get()),
        JitKernelArg::i64(input_distance),
        JitKernelArg::i32(static_cast<int32_t>(context.batch)),
    };
    expand_kernel->launch(context.stream, expand_args, ceil_div(length, block), context.batch, 1);

    flagfftResult result = fft->execute(complex_input.get(), full_output.get(), context);
    if (result != FLAGFFT_SUCCESS) {
      return result;
    }

    std::vector<JitKernelArg> pack_args = {
        JitKernelArg::device(full_output.get()),
        JitKernelArg::device(output),
        JitKernelArg::i64(output_distance),
        JitKernelArg::i32(static_cast<int32_t>(context.batch)),
    };
    pack_kernel->launch(context.stream, pack_args, ceil_div(length / 2 + 1, block), context.batch, 1);
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] R2C execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRawR2CLeafNode::CompiledRawR2CLeafNode(int64_t length,
                                               std::shared_ptr<JitKernel> kernel,
                                               std::vector<DeviceAllocation> tables)
    : length(length), kernel(std::move(kernel)), tables(std::move(tables)) {
}

std::string CompiledRawR2CLeafNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRawR2CLeaf(n=" << length << ", kernel=" << (kernel ? kernel->kernel_name : "null")
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
    kernel->launch(context.stream, args, ceil_div(context.batch, kernel->batch_per_block), 1, 1);
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
      << ", expand_kernel=" << (expand_kernel ? expand_kernel->kernel_name : "null")
      << ", row_kernel=" << (row_kernel ? row_kernel->kernel_name : "null")
      << ", col_kernel=" << (col_kernel ? col_kernel->kernel_name : "null") << ")";
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
    expand_kernel->launch(context.stream, expand_args, ceil_div(length, block), context.batch, 1);

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
      << ", row_kernel=" << (row_kernel ? row_kernel->kernel_name : "null")
      << ", col_kernel=" << (col_kernel ? col_kernel->kernel_name : "null") << ")";
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
      << ", expand_kernel=" << (expand_kernel ? expand_kernel->kernel_name : "null")
      << ", fft=" << (fft ? fft->describe() : "null")
      << ", pack_kernel=" << (pack_kernel ? pack_kernel->kernel_name : "null") << ")";
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
    std::vector<JitKernelArg> expand_args = {
        JitKernelArg::device(input),
        JitKernelArg::device(full_input.get()),
        JitKernelArg::i64(input_distance),
        JitKernelArg::i32(static_cast<int32_t>(context.batch)),
    };
    expand_kernel->launch(context.stream, expand_args, ceil_div(length, block), context.batch, 1);

    flagfftResult result = fft->execute(full_input.get(), full_output.get(), context);
    if (result != FLAGFFT_SUCCESS) {
      return result;
    }

    std::vector<JitKernelArg> pack_args = {
        JitKernelArg::device(full_output.get()),
        JitKernelArg::device(output),
        JitKernelArg::i64(output_distance),
        JitKernelArg::i32(static_cast<int32_t>(context.batch)),
    };
    pack_kernel->launch(context.stream, pack_args, ceil_div(length, block), context.batch, 1);
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] C2R execute failed: %s\n", e.what());
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
                                     DeviceAllocation temp2)
    : n0(n0),
      n1(n1),
      row_fft(std::move(row_fft)),
      col_fft(std::move(col_fft)),
      transpose_fwd(std::move(transpose_fwd)),
      transpose_inv(std::move(transpose_inv)),
      temp1(std::move(temp1)),
      temp2(std::move(temp2)) {
}

std::string CompiledRaw2DNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRaw2D(n0=" << n0 << ", n1=" << n1
      << ", row_fft=" << (row_fft ? row_fft->describe() : "null")
      << ", col_fft=" << (col_fft ? col_fft->describe() : "null")
      << ", transpose_fwd=" << (transpose_fwd ? transpose_fwd->kernel_name : "null")
      << ", transpose_inv=" << (transpose_inv ? transpose_inv->kernel_name : "null") << ")";
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

    // Step 1: Row FFT (input -> temp1)
    // Shape: (batch, n0, n1) -> row FFT along last dimension
    // batch for row FFT = batch * n0
    RawExecutionContext row_context {context.request, context.stream, batch * n0};
    flagfftResult result = row_fft->execute(input, temp1.get(), row_context);
    if (result != FLAGFFT_SUCCESS) {
      return result;
    }

    // Step 2: Transpose (temp1 -> temp2)
    // Shape: (batch, n0, n1) -> (batch, n1, n0)
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
    // After transpose, shape is (batch, n1, n0)
    // Col FFT along last dimension (n0), batch = batch * n1
    RawExecutionContext col_context {context.request, context.stream, batch * n1};
    result = col_fft->execute(temp2.get(), temp1.get(), col_context);
    if (result != FLAGFFT_SUCCESS) {
      return result;
    }

    // Step 4: Transpose back (temp1 -> output)
    // Shape: (batch, n1, n0) -> (batch, n0, n1)
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
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] 2D execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

CompiledRawC2RLeafNode::CompiledRawC2RLeafNode(int64_t length,
                                               std::shared_ptr<JitKernel> kernel,
                                               std::vector<DeviceAllocation> tables)
    : length(length), kernel(std::move(kernel)), tables(std::move(tables)) {
}

std::string CompiledRawC2RLeafNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRawC2RLeaf(n=" << length << ", kernel=" << (kernel ? kernel->kernel_name : "null")
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
    kernel->launch(context.stream, args, ceil_div(context.batch, kernel->batch_per_block), 1, 1);
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] C2RLeaf execute failed: %s\n", e.what());
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
      << ", expand_kernel=" << (expand_kernel ? expand_kernel->kernel_name : "null")
      << ", row_fft=" << (row_fft ? row_fft->describe() : "null")
      << ", pack_kernel=" << (pack_kernel ? pack_kernel->kernel_name : "null")
      << ", col_fft=" << (col_fft ? col_fft->describe() : "null")
      << ", transpose_fwd=" << (transpose_fwd ? transpose_fwd->kernel_name : "null")
      << ", transpose_inv=" << (transpose_inv ? transpose_inv->kernel_name : "null") << ")";
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
    expand_kernel->launch(context.stream, expand_args, ceil_div(n1, block), total_rows, 1);

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
    pack_kernel->launch(context.stream, pack_args, ceil_div(half_n1, block), total_rows, 1);

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
      << ", expand_kernel=" << (expand_kernel ? expand_kernel->kernel_name : "null")
      << ", row_kernel=" << (row_kernel ? row_kernel->kernel_name : "null")
      << ", col_kernel=" << (col_kernel ? col_kernel->kernel_name : "null") << ")";
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
    expand_kernel->launch(context.stream, expand_args, ceil_div(length, block), context.batch, 1);

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
      << ", expand_kernel=" << (expand_kernel ? expand_kernel->kernel_name : "null")
      << ", col_fft=" << (col_fft ? col_fft->describe() : "null")
      << ", row_fft=" << (row_fft ? row_fft->describe() : "null")
      << ", transpose_fwd=" << (transpose_fwd ? transpose_fwd->kernel_name : "null")
      << ", transpose_inv=" << (transpose_inv ? transpose_inv->kernel_name : "null")
      << ", pack_kernel=" << (pack_kernel ? pack_kernel->kernel_name : "null") << ")";
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
    expand_kernel->launch(context.stream, expand_args, ceil_div(n1, block), total_rows, 1);

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
    pack_kernel->launch(context.stream, pack_args, ceil_div(n1, block), total_rows, 1);

    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] 2D C2R execute failed: %s\n", e.what());
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
      << ", row_kernel=" << (row_kernel ? row_kernel->kernel_name : "null")
      << ", col_kernel=" << (col_kernel ? col_kernel->kernel_name : "null") << ")";
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
                                     DeviceAllocation temp2)
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
      temp2(std::move(temp2)) {
}

std::string CompiledRaw3DNode::describe() const {
  std::ostringstream oss;
  oss << "CompiledRaw3D(n0=" << n0 << ", n1=" << n1 << ", n2=" << n2
      << ", n2_fft=" << (n2_fft ? n2_fft->describe() : "null")
      << ", n1_fft=" << (n1_fft ? n1_fft->describe() : "null")
      << ", n0_fft=" << (n0_fft ? n0_fft->describe() : "null")
      << ", perm_021_fwd=" << (perm_021_fwd ? perm_021_fwd->kernel_name : "null")
      << ", perm_210_fwd=" << (perm_210_fwd ? perm_210_fwd->kernel_name : "null")
      << ", perm_201_fwd=" << (perm_201_fwd ? perm_201_fwd->kernel_name : "null") << ")";
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

    flagfftResult result;
    if (inverse) {
      // (n0,n1,n2) -> perm(1,2,0) -> (n1,n2,n0), IFFT along n0
      launch_perm3d(perm_120_inv, context.stream, input, temp1.get(), total, batch);
      result = n0_fft->execute(temp1.get(), temp2.get(), n0_context);
      if (result != FLAGFFT_SUCCESS) {
        return result;
      }
      // (n1,n2,n0) -> perm(2,1,0) -> (n0,n2,n1), IFFT along n1
      launch_perm3d(perm_210_inv, context.stream, temp2.get(), temp1.get(), total, batch);
      result = n1_fft->execute(temp1.get(), temp2.get(), n1_context);
      if (result != FLAGFFT_SUCCESS) {
        return result;
      }
      // (n0,n2,n1) -> perm(0,2,1) -> (n0,n1,n2), IFFT along n2
      launch_perm3d(perm_021_inv, context.stream, temp2.get(), temp1.get(), total, batch);
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
      launch_perm3d(perm_021_fwd, context.stream, temp1.get(), temp2.get(), total, batch);
      result = n1_fft->execute(temp2.get(), temp1.get(), n1_context);
      if (result != FLAGFFT_SUCCESS) {
        return result;
      }
      // (n0,n2,n1) -> perm(2,1,0) -> (n1,n2,n0), FFT along n0
      launch_perm3d(perm_210_fwd, context.stream, temp1.get(), temp2.get(), total, batch);
      result = n0_fft->execute(temp2.get(), temp1.get(), n0_context);
      if (result != FLAGFFT_SUCCESS) {
        return result;
      }
      // (n1,n2,n0) -> perm(2,0,1) -> (n0,n1,n2)
      launch_perm3d(perm_201_fwd, context.stream, temp1.get(), output, total, batch);
    }
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] 3D execute failed: %s\n", e.what());
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
      << ", expand_kernel=" << (expand_kernel ? expand_kernel->kernel_name : "null")
      << ", n2_fft=" << (n2_fft ? n2_fft->describe() : "null")
      << ", pack_kernel=" << (pack_kernel ? pack_kernel->kernel_name : "null")
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
    launch_grid_y_chunks(total_rows, [&](int64_t row_offset, int64_t chunk_rows) {
      std::vector<JitKernelArg> expand_args = {
          JitKernelArg::device(input + row_offset * n2 * real_bytes),
          JitKernelArg::device(row_fft_buf.get() + row_offset * n2 * complex_bytes),
          JitKernelArg::i64(n2),
          JitKernelArg::i32(static_cast<int32_t>(chunk_rows)),
      };
      expand_kernel->launch(context.stream, expand_args, ceil_div(n2, block), chunk_rows, 1);
    });

    // Step 2: FFT along n2 in-place.
    RawExecutionContext n2_context {context.request, context.stream, total_rows};
    flagfftResult result = n2_fft->execute(row_fft_buf.get(), row_fft_buf.get(), n2_context);
    if (result != FLAGFFT_SUCCESS) {
      return result;
    }

    // Step 3: Half-pack rows into the output (n0, n1, half) layout.
    launch_grid_y_chunks(total_rows, [&](int64_t row_offset, int64_t chunk_rows) {
      std::vector<JitKernelArg> pack_args = {
          JitKernelArg::device(row_fft_buf.get() + row_offset * n2 * complex_bytes),
          JitKernelArg::device(output + row_offset * half * complex_bytes),
          JitKernelArg::i64(half),
          JitKernelArg::i32(static_cast<int32_t>(chunk_rows)),
      };
      pack_kernel->launch(context.stream, pack_args, ceil_div(half, block), chunk_rows, 1);
    });

    // Step 4: (n0,n1,half) -> (n0,half,n1), FFT along n1.
    const int64_t packed = n0 * n1 * half;
    launch_perm3d(perm_021, context.stream, output, temp1.get(), packed, batch);
    RawExecutionContext n1_context {context.request, context.stream, batch * n0 * half};
    result = n1_fft->execute(temp1.get(), temp2.get(), n1_context);
    if (result != FLAGFFT_SUCCESS) {
      return result;
    }

    // Step 5: (n0,half,n1) -> (n1,half,n0), FFT along n0.
    launch_perm3d(perm_210, context.stream, temp2.get(), temp1.get(), packed, batch);
    RawExecutionContext n0_context {context.request, context.stream, batch * n1 * half};
    result = n0_fft->execute(temp1.get(), temp2.get(), n0_context);
    if (result != FLAGFFT_SUCCESS) {
      return result;
    }

    // Step 6: (n1,half,n0) -> (n0,n1,half) natural output layout.
    launch_perm3d(perm_201, context.stream, temp2.get(), output, packed, batch);
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
      << ", expand_kernel=" << (expand_kernel ? expand_kernel->kernel_name : "null")
      << ", n2_fft=" << (n2_fft ? n2_fft->describe() : "null")
      << ", pack_kernel=" << (pack_kernel ? pack_kernel->kernel_name : "null") << ")";
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
    launch_perm3d(perm_120, context.stream, input, temp1.get(), packed, batch);
    RawExecutionContext n0_context {context.request, context.stream, batch * n1 * half};
    flagfftResult result = n0_fft->execute(temp1.get(), temp2.get(), n0_context);
    if (result != FLAGFFT_SUCCESS) {
      return result;
    }

    // Step 2: (n1,half,n0) -> (n0,half,n1), IFFT along n1.
    launch_perm3d(perm_210, context.stream, temp2.get(), temp1.get(), packed, batch);
    RawExecutionContext n1_context {context.request, context.stream, batch * n0 * half};
    result = n1_fft->execute(temp1.get(), temp2.get(), n1_context);
    if (result != FLAGFFT_SUCCESS) {
      return result;
    }

    // Step 3: (n0,half,n1) -> (n0,n1,half), expand half -> full Hermitian.
    launch_perm3d(perm_021, context.stream, temp2.get(), temp1.get(), packed, batch);
    launch_grid_y_chunks(total_rows, [&](int64_t row_offset, int64_t chunk_rows) {
      std::vector<JitKernelArg> expand_args = {
          JitKernelArg::device(temp1.get() + row_offset * half * complex_bytes),
          JitKernelArg::device(full_buf.get() + row_offset * n2 * complex_bytes),
          JitKernelArg::i64(half),
          JitKernelArg::i32(static_cast<int32_t>(chunk_rows)),
      };
      expand_kernel->launch(context.stream, expand_args, ceil_div(n2, block), chunk_rows, 1);
    });

    // Step 4: IFFT along n2 (in-place on full complex rows), pack complex -> real.
    RawExecutionContext n2_context {context.request, context.stream, total_rows};
    result = n2_fft->execute(full_buf.get(), full_buf.get(), n2_context);
    if (result != FLAGFFT_SUCCESS) {
      return result;
    }
    launch_grid_y_chunks(total_rows, [&](int64_t row_offset, int64_t chunk_rows) {
      std::vector<JitKernelArg> pack_args = {
          JitKernelArg::device(full_buf.get() + row_offset * n2 * complex_bytes),
          JitKernelArg::device(output + row_offset * n2 * real_bytes),
          JitKernelArg::i64(n2),
          JitKernelArg::i32(static_cast<int32_t>(chunk_rows)),
      };
      pack_kernel->launch(context.stream, pack_args, ceil_div(n2, block), chunk_rows, 1);
    });
    return FLAGFFT_SUCCESS;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[flagfft] 3D C2R execute failed: %s\n", e.what());
    std::fflush(stderr);
    return FLAGFFT_EXEC_FAILED;
  }
}

}  // namespace flagfft
